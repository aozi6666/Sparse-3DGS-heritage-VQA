#!/usr/bin/env python
"""Serve interactive, causal video question answering with Mage-VL.

The browser sends its current playback timestamp with every question.  The
server constructs a bounded causal clip ending at that timestamp, so Mage-VL
sees the current frame plus a short history, never future frames.  Responses
are returned as Server-Sent Events for token-by-token rendering.
"""
from __future__ import annotations

import argparse
import asyncio
import base64
import io
import json
import logging
import math
import mimetypes
import os
import shutil
import subprocess
import threading
import uuid
from dataclasses import dataclass
from pathlib import Path
from queue import Queue
from typing import Iterator, Protocol

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse, HTMLResponse, StreamingResponse
from pydantic import BaseModel, Field


APP_DIR = Path(__file__).resolve().parent
WEB_PAGE = APP_DIR / "web" / "index.html"
BUILD_ID = "deepseek-sse-framing-2026-09-24"
SUPPORTED_VIDEO_SUFFIXES = {".mp4", ".mov", ".m4v", ".webm", ".avi", ".mkv"}
LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class VideoAsset:
    path: Path
    duration: float


class QuestionRequest(BaseModel):
    video_id: str = Field(min_length=1)
    timestamp: float = Field(ge=0)
    question: str = Field(min_length=1, max_length=2_000)


def causal_window(timestamp: float, duration: float, context_seconds: float) -> tuple[float, float]:
    """Return a clip ending at the playhead without leaking future frames."""
    end = min(max(timestamp, 0.0), duration)
    return max(0.0, end - context_seconds), end


def video_duration(path: Path) -> float:
    import cv2

    capture = cv2.VideoCapture(str(path))
    try:
        fps = float(capture.get(cv2.CAP_PROP_FPS))
        frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        if fps <= 0 or frame_count <= 0:
            raise ValueError("video has no valid FPS")
        return frame_count / fps
    finally:
        capture.release()


def sample_causal_frames(path: Path, start: float, end: float, count: int) -> list[str]:
    """Sample chronological JPEG data URLs from a causal time range."""
    import cv2
    import numpy as np
    from PIL import Image

    capture = cv2.VideoCapture(str(path))
    try:
        fps = float(capture.get(cv2.CAP_PROP_FPS))
        frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        if fps <= 0 or frame_count <= 0:
            raise ValueError("video has no valid FPS")
        first = min(max(math.ceil(start * fps), 0), frame_count - 1)
        last = min(max(math.floor(end * fps), first), frame_count - 1)
        indices = np.linspace(first, last, min(count, last - first + 1), dtype=int)
        images = []
        for index in indices:
            capture.set(cv2.CAP_PROP_POS_FRAMES, int(index))
            ok, frame = capture.read()
            if not ok:
                continue
            image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            buffer = io.BytesIO()
            image.save(buffer, format="JPEG", quality=85)
            images.append("data:image/jpeg;base64," + base64.b64encode(buffer.getvalue()).decode("ascii"))
        if not images:
            raise ValueError("could not decode frames from the selected time window")
        return images
    finally:
        capture.release()


def cut_clip(source: Path, start: float, end: float, destination: Path) -> Path:
    """Make a self-contained causal clip. Re-encoding keeps cut boundaries exact."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    command = [
        "ffmpeg", "-y", "-loglevel", "error", "-ss", f"{start:.3f}",
        "-t", f"{max(end - start, 0.05):.3f}", "-i", str(source),
        "-map", "0:v:0", "-an", "-c:v", "libx264", "-preset", "veryfast",
        "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(destination),
    ]
    subprocess.run(command, check=True, capture_output=True, timeout=120)
    return destination


class MageVideoQA:
    def __init__(self, checkpoint: str, backend: str, device: str, max_pixels: int,
                 num_frames: int, target_fps: float, max_new_tokens: int, clip_dir: Path):
        self.checkpoint = checkpoint
        self.backend = backend
        self.device = device
        self.max_pixels = max_pixels
        self.num_frames = num_frames
        self.target_fps = target_fps
        self.max_new_tokens = max_new_tokens
        self.clip_dir = clip_dir
        self.processor = None
        self.model = None
        self.model_lock = threading.Lock()

    def load(self) -> None:
        if self.model is not None:
            return
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoProcessor
        except ImportError as error:
            raise RuntimeError("Mage mode requires mage_vl/requirements.txt.") from error
        self.torch = torch
        self.processor = AutoProcessor.from_pretrained(self.checkpoint, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            self.checkpoint,
            trust_remote_code=True,
            torch_dtype="auto",
            device_map="auto" if self.device == "auto" else None,
        ).eval()
        if self.device != "auto":
            self.model.to(self.device)

    def _inputs(self, clip_path: Path, question: str) -> dict:
        assert self.processor is not None
        messages = [{"role": "user", "content": [
            {"type": "video"},
            {"type": "text", "text": question},
        ]}]
        prompt = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
        )
        kwargs = {
            "text": [prompt], "videos": [str(clip_path)],
            "video_backend": self.backend, "return_tensors": "pt", "padding": True,
        }
        if self.backend == "codec":
            kwargs.update(codec_config={"engine": "hevc", "patch": 16, "max_pixels": self.max_pixels},
                          max_pixels=self.max_pixels)
        else:
            kwargs.update(num_frames=self.num_frames, target_fps=self.target_fps)
        inputs = self.processor(**kwargs)
        model_device = self.model.device
        return {
            key: (value.to(device=model_device, dtype=self.model.dtype) if key == "pixel_values" else value.to(model_device))
            for key, value in inputs.items()
        }

    def stream_answer(self, video: VideoAsset, timestamp: float, context_seconds: float,
                      question: str) -> Iterator[str]:
        self.load()
        start, end = causal_window(timestamp, video.duration, context_seconds)
        clip = self.clip_dir / f"{uuid.uuid4().hex}.mp4"
        try:
            cut_clip(video.path, start, end, clip)
            inputs = self._inputs(clip, question)
            from transformers import TextIteratorStreamer

            streamer = TextIteratorStreamer(self.processor.tokenizer, skip_prompt=True,
                                            skip_special_tokens=True, timeout=120.0)
            error_queue: Queue[Exception] = Queue(maxsize=1)

            def generate() -> None:
                try:
                    with self.model_lock, self.torch.inference_mode():
                        self.model.generate(**inputs, max_new_tokens=self.max_new_tokens,
                                            do_sample=False, streamer=streamer)
                except Exception as error:
                    error_queue.put(error)

            thread = threading.Thread(target=generate, daemon=True)
            thread.start()
            for token in streamer:
                yield token
            thread.join()
            if not error_queue.empty():
                raise error_queue.get()
        finally:
            clip.unlink(missing_ok=True)


class VideoQAEngine(Protocol):
    def stream_answer(self, video: VideoAsset, timestamp: float, context_seconds: float,
                      question: str) -> Iterator[str]: ...


class DeepSeekVideoQA:
    """VideoQA through DeepSeek's OpenAI-compatible, native-vision API."""

    def __init__(self, api_key: str, base_url: str, model: str, max_images: int,
                 max_new_tokens: int, thinking: str, image_detail: str, response_mode: str):
        from openai import OpenAI

        self.client = OpenAI(api_key=api_key, base_url=base_url)
        self.model = model
        self.max_images = max_images
        self.max_new_tokens = max_new_tokens
        self.thinking = thinking
        self.image_detail = image_detail
        self.response_mode = response_mode

    def stream_answer(self, video: VideoAsset, timestamp: float, context_seconds: float,
                      question: str) -> Iterator[str]:
        start, end = causal_window(timestamp, video.duration, context_seconds)
        frames = sample_causal_frames(video.path, start, end, self.max_images)
        content = [{"type": "text", "text": (
            f"这些画面按时间先后排列，来自视频 {start:.1f} 到 {end:.1f} 秒；最后一张是当前播放位置。"
            "请用中文直接回答用户问题。请只依据这些画面回答，不要假定未显示的未来内容；若证据不足，请明确说明。"
            f"\n\n用户问题：{question}"
        )}]
        content.extend({"type": "image_url", "image_url": {"url": frame, "detail": self.image_detail}}
                       for frame in frames)
        request = {
            "model": self.model,
            "messages": [{"role": "user", "content": content}],
            "max_tokens": self.max_new_tokens,
            "extra_body": {"thinking": {"type": self.thinking}},
        }
        if self.response_mode == "final":
            completion = self.client.chat.completions.create(stream=False, **request)
            answer = completion.choices[0].message.content if completion.choices else None
            if answer:
                LOGGER.info("DeepSeek returned a final answer (%d characters)", len(answer))
                yield answer
                return
            LOGGER.error("DeepSeek final response did not contain content: %r", completion)
            raise RuntimeError("DeepSeek returned an empty final answer. Check videoqa.log for details.")

        stream = self.client.chat.completions.create(stream=True, **request)
        emitted = False
        for chunk in stream:
            if not chunk.choices:
                continue
            token = getattr(chunk.choices[0].delta, "content", None)
            if token:
                emitted = True
                yield token
        if not emitted:
            LOGGER.warning("DeepSeek stream returned no visible content; retrying without streaming")
            completion = self.client.chat.completions.create(stream=False, **request)
            answer = completion.choices[0].message.content if completion.choices else None
            if answer:
                yield answer
                return
            raise RuntimeError(
                "DeepSeek returned an empty final answer after retrying. Check DEEPSEEK_API_KEY, "
                "account balance, and videoqa.log for the API response."
            )


def sse(event: str, payload: dict) -> str:
    return f"event: {event}\ndata: {json.dumps(payload, ensure_ascii=False)}\n\n"


def create_app(engine: VideoQAEngine, upload_dir: Path, context_seconds: float) -> FastAPI:
    app = FastAPI(title="Mage-VL Streaming Video QA")
    assets: dict[str, VideoAsset] = {}
    upload_dir.mkdir(parents=True, exist_ok=True)

    @app.get("/", response_class=HTMLResponse)
    def index() -> str:
        return WEB_PAGE.read_text(encoding="utf-8")

    @app.get("/api/health")
    def health() -> dict:
        return {"status": "ok", "build": BUILD_ID, "active_videos": len(assets)}

    @app.post("/api/videos")
    async def upload_video(video: UploadFile = File(...)) -> dict:
        suffix = Path(video.filename or "video.mp4").suffix.lower() or ".mp4"
        is_video = (video.content_type or "").startswith("video/")
        if not is_video and suffix not in SUPPORTED_VIDEO_SUFFIXES:
            raise HTTPException(415, "Unsupported format. Please upload MP4, MOV, WebM, AVI, or MKV.")
        asset_id = uuid.uuid4().hex
        destination = upload_dir / f"{asset_id}{suffix}"
        with destination.open("wb") as output:
            shutil.copyfileobj(video.file, output)
        try:
            duration = await asyncio.to_thread(video_duration, destination)
        except Exception as error:
            destination.unlink(missing_ok=True)
            raise HTTPException(422, f"Unable to decode video: {error}") from error
        assets[asset_id] = VideoAsset(destination, duration)
        return {"video_id": asset_id, "duration": duration, "url": f"/api/videos/{asset_id}"}

    @app.get("/api/videos/{video_id}")
    def get_video(video_id: str) -> FileResponse:
        asset = assets.get(video_id)
        if asset is None:
            raise HTTPException(404, "Video session not found. Upload the video again.")
        media_type = mimetypes.guess_type(asset.path.name)[0] or "application/octet-stream"
        return FileResponse(asset.path, media_type=media_type)

    @app.post("/api/questions")
    def question(request: QuestionRequest) -> StreamingResponse:
        asset = assets.get(request.video_id)
        if asset is None:
            raise HTTPException(404, "Video session not found. Upload the video again.")

        def response() -> Iterator[str]:
            start, end = causal_window(request.timestamp, asset.duration, context_seconds)
            yield sse("context", {"start": start, "end": end})
            try:
                for token in engine.stream_answer(asset, request.timestamp, context_seconds, request.question):
                    yield sse("token", {"text": token})
                yield sse("done", {})
            except Exception as error:
                LOGGER.exception("VideoQA request failed for video %s at %.2fs", request.video_id, request.timestamp)
                yield sse("error", {"message": str(error) or type(error).__name__})

        return StreamingResponse(response(), media_type="text/event-stream",
                                 headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})

    return app


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", default="microsoft/Mage-VL")
    parser.add_argument("--engine", choices=("mage", "deepseek"), default="mage")
    parser.add_argument("--backend", choices=("codec", "frames"), default="codec")
    parser.add_argument("--context-seconds", type=float, default=12.0)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--max-pixels", type=int, default=150000)
    parser.add_argument("--num-frames", type=int, default=16)
    parser.add_argument("--target-fps", type=float, default=2.0)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--api-key-env", default="DEEPSEEK_API_KEY",
                        help="Environment variable containing the DeepSeek API key.")
    parser.add_argument("--api-base-url", default="https://api.deepseek.com")
    parser.add_argument("--api-model", default="deepseek-flash")
    parser.add_argument("--api-max-images", type=int, default=8)
    parser.add_argument("--api-thinking", choices=("disabled", "enabled"), default="disabled",
                        help="Disable thinking for faster interactive answers (default).")
    parser.add_argument("--api-image-detail", choices=("low", "original", "auto"), default="low")
    parser.add_argument("--api-response-mode", choices=("final", "stream"), default="final",
                        help="Use the reliable final-response API by default; stream is experimental.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument("--upload-dir", type=Path, default=APP_DIR / ".video_qa_uploads")
    args = parser.parse_args()
    if args.context_seconds <= 0:
        parser.error("--context-seconds must be positive")
    if args.api_max_images <= 0:
        parser.error("--api-max-images must be positive")
    return args


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    cache_dir = args.upload_dir / "clips"
    if args.engine == "mage" and args.backend == "codec":
        os.environ.setdefault("ONLINE_CODEC_CACHE_DIR", str(args.upload_dir / "codec_cache"))
    if args.engine == "deepseek":
        api_key = os.environ.get(args.api_key_env)
        if not api_key:
            raise SystemExit(f"Set {args.api_key_env} before starting the DeepSeek engine.")
        engine: VideoQAEngine = DeepSeekVideoQA(
            api_key, args.api_base_url, args.api_model, args.api_max_images, args.max_new_tokens,
            args.api_thinking, args.api_image_detail, args.api_response_mode,
        )
    else:
        engine = MageVideoQA(args.checkpoint, args.backend, args.device, args.max_pixels,
                             args.num_frames, args.target_fps, args.max_new_tokens, cache_dir)
    app = create_app(engine, args.upload_dir, args.context_seconds)
    import uvicorn
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
