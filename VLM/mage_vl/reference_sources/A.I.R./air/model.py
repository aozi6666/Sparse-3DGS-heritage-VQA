import time
import os
import json
from datetime import datetime
import random

DEBUG = False
import torch
import torch.nn as nn
from typing import Tuple, Dict, List, Union, Optional
from PIL import Image
import numpy as np
from torch.cuda.amp import autocast
import matplotlib.pyplot as plt

from decord import VideoReader, cpu
from .Process_Utils.process_messages import create_analysis_messages, videomme_preprocess, \
    videomme_w_sub_preprocess, extract_score, lvb_preprocess, mlvu_preprocess
from .Process_Utils.sample_logic import generate_dynamic_samples, sample_logic
from sklearn.mixture import GaussianMixture
from .base_models.clip_model import (
    initialize_clip_model,
    encode_text_clip,
    encode_images_clip,
    compute_clip_similarity
)


class H_VPModel(nn.Module):
    def __init__(
            self,
            vl_model_name: str = "Qwen/Qwen2.5-VL-7B-Instruct",
            device_map: str = "auto",
            clip_model_name: str = "EVA02-L-14",
            attn_implementation=None,
            sub_batch: int = 12,
            min_pixels: int = 128 * 28 * 28,
            max_pixels: int = 768 * 28 * 28,
    ):
        super().__init__()
        self.sub_batch = sub_batch
        self.max_pixels = max_pixels
        self.min_pixels = min_pixels
        self.max_budget = None
        self.max_iteration = 6
        if 'Qwen2.5' in vl_model_name:
            from .base_models.Qwen2_5VL import initialize_qwen_model
            self.model_type = "Qwen2.5"
            self.vl_model_set = initialize_qwen_model(
                model_name=vl_model_name,
                device_map=device_map
            )
        elif 'InternVL' in vl_model_name:
            from .base_models.InternVL3 import initialize_intern_model
            self.model_type = "InternVL"
            need_quantization = "26B" in vl_model_name or "38B" in vl_model_name
            self.vl_model_set = initialize_intern_model(
                model_name=vl_model_name,
                device_map=device_map,
                load_in_8bit=need_quantization
            )
            self.max_budget = 64
        elif 'llava-onevision' in vl_model_name.lower():
            from .base_models.Llava_OV import initialize_llava_ov_model
            self.model_type = "LLaVA-OV"
            self.vl_model_set = initialize_llava_ov_model(
                model_name=vl_model_name,
                device_map=device_map
            )
            self.max_budget = 16
        elif 'VILA' in vl_model_name:
            from .base_models.VILA1_5 import initialize_vila_model
            self.model_type = "VILA-1.5"
            self.attn_implementation = attn_implementation
            self.vl_model_set = initialize_vila_model(
                model_name=vl_model_name,
                device_map=device_map
            )
            self.max_budget = 8
        elif 'Qwen2' in vl_model_name:
            from .base_models.Qwen2_VL import initialize_qwen2vl_model
            self.model_type = "Qwen2-VL"
            self.vl_model_set = initialize_qwen2vl_model(
                model_name=vl_model_name,
                device_map=device_map,
            )

        self.clip_model_name = clip_model_name
        self._initialize_clip_model()
        self.save_similarities = False
        self.text_features = None
        self.N_candidate = 24

    def _initialize_clip_model(self, model_name: Optional[str] = None):
        if model_name:
            self.clip_model_name = model_name

        device_str = "cuda"
        model_info = initialize_clip_model(self.clip_model_name, device_str)
        self.clip_model, self.clip_preprocess_or_processor, self.clip_tokenizer, self.clip_model_type = model_info

    def eval(self):

        self.vl_model_set[0].eval()

        if hasattr(self, 'clip_model'):
            self.clip_model.eval()
        return self

    @property
    def device(self):
        # return self.vl_model_set[0].device
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def _batch_inference(self, model_set, input_messages: list, **kwargs):
        if self.model_type == "Qwen2.5":
            from .base_models.Qwen2_5VL import qwen_batch_inference, qwen_video_inference
            if (len(input_messages) == 1 and
                    any(content.get('type') == 'video'
                        for msg in input_messages[0]
                        for content in msg.get('content', [])
                        if msg.get('role') == 'user')):
                return qwen_video_inference(
                    self.vl_model_set[0],
                    self.vl_model_set[1],
                    input_messages[0],
                    fps=kwargs.get('video_fps', 1.0),
                    device=self.device,
                    **kwargs
                )
            else:
                return qwen_batch_inference(
                    self.vl_model_set[0],
                    self.vl_model_set[1],
                    input_messages,
                    device=self.device,
                    **kwargs
                )
        elif self.model_type == "InternVL":
            from .base_models.InternVL3 import intern_batch_inference
            return intern_batch_inference(
                self.vl_model_set[0],
                self.vl_model_set[1],
                input_messages,
                **kwargs
            )
        elif self.model_type == "LLaVA-OV":
            from .base_models.Llava_OV import llava_ov_batch_inference
            responses = llava_ov_batch_inference(
                self.vl_model_set[0],
                self.vl_model_set[1],
                input_messages,
                **kwargs
            )
            return responses
        elif self.model_type == "VILA-1.5":
            from .base_models.VILA1_5 import vila_batch_inference
            responses = vila_batch_inference(
                self.vl_model_set[0],
                self.vl_model_set[1],
                input_messages,
                **kwargs
            )
            if responses and isinstance(responses[0], list):
                responses = [r[0] if isinstance(r, list) else r for r in responses]
            return responses
        elif self.model_type == "Qwen2-VL":
            from .base_models.Qwen2_VL import qwen2vl_batch_inference
            return qwen2vl_batch_inference(
                self.vl_model_set[0],
                self.vl_model_set[1],
                input_messages,
                device=self.device,
                **kwargs
            )
        else:
            raise ValueError(f"Unsupported model type: {self.model_type}")

    @torch.inference_mode()
    def forward(
            self,
            messages: list[dict] | list[list[dict]],
            videos=None,
            hisa_weights=None,
            **generation_kwargs
    ):
        self.text_features = None
        self.clip_model.to(self.device)
        if isinstance(messages[0], dict):
            messages = [messages]
        if videos is None or not isinstance(videos, list):
            videos = []
            for m in messages:
                try:
                    video = m[1]['content'][0]['video']
                    videos.append(video)
                except:
                    print(f"Pre-check: video analysis failed, message: {m}")
        assert videos is not None, "Pre-check: video is none !!!"
        bs = len(messages)

        final_messages_batch = []

        for video_idx, (v, message) in enumerate(zip(videos, messages)):
            # query_text = message[1]['content'][1]['text']

            full_text = message[1]['content'][1]['text']
            query_text = videomme_preprocess(full_text)
            # query_text = videomme_w_sub_preprocess(full_text)
            # query_text = mlvu_preprocess(full_text)
            # query_text = lvb_preprocess(full_text)
            # print(query_text)
            VR = VideoReader(v, ctx=cpu(0))
            total_frames = len(VR)
            actual_fps = VR.get_avg_fps()
            video_fps = int(actual_fps)
            video_duration = total_frames / actual_fps

            if video_duration <= 100:
                clip_fps = 3.0
                merge_gap_seconds = 2
                min_segment_frames = 1
                gmm_coefficient = 0.5
                min_event_length = 15
                min_frame_distance = 20
            else:
                clip_fps = 2.0
                merge_gap_seconds = 30
                min_segment_frames = 2
                gmm_coefficient = 0.8
                min_event_length = 30
                min_frame_distance = 30

            # min_frame_distance = int(np.clip(total_frames * 0.001, 20, 30))
            adaptive_budget = self.calculate_adaptive_budget(video_duration, self.max_budget)

            similarities, raw_sim, frame_indices, norm_params = self.compute_video_query_similarity(
                video_path=v,
                query=query_text,
                fps=clip_fps,
                return_frame_indices=True,
            )

            suggested_threshold = 0.2

            segments, threshold = self.find_relevant_segments(
                similarities,
                frame_indices,
                threshold='adaptive',
                min_segment_frames=min_segment_frames,
                video_fps=video_fps,
                merge_gap_seconds=merge_gap_seconds,
                gmm_coefficient=gmm_coefficient,
            )

            if segments:
                event_splits = [[seg['start_frame'], seg['end_frame']] for seg in segments]
            else:
                num_segments = 4
                segment_length = len(VR) // num_segments
                event_splits = [[i * segment_length, min((i + 1) * segment_length - 1, len(VR) - 1)]
                                for i in range(num_segments)]

            sampled_frames, sampled_frames_indices = self.adaptive_initial_sampling(
                VR, event_splits,
                use_sampling=True,
                max_frames_per_event=5,
                min_event_length=min_event_length,
                total_budget=adaptive_budget,
                clip_similarities=similarities,
                clip_frame_indices=frame_indices
            )

            final_frames, final_frames_indices = self.iteration_algorithm(
                sampled_frames,
                sampled_frames_indices,
                VR,
                query=query_text,
                generation_kwargs=generation_kwargs,
                budget=adaptive_budget,
                min_frame_distance=min_frame_distance,
                video_fps=video_fps,
                i2t_norm_params=norm_params,
                hisa_weights=hisa_weights,
                clip_i2t_threshold=suggested_threshold,
                clip_i2i_threshold=0.8,
                clip_similarities=similarities,
                raw_clip_similarities=raw_sim,
                clip_frame_indices=frame_indices
            )

            prompt_template = """Select the best answer to the following multiple-choice question based on the video. Respond with only the letter (A, B, C, D or E) of the correct option.
            {}"""
            final_message = [
                {
                    "role": "assistant",
                    "content": [{"type": "text", "text": "You are a helpful assistant"}]
                },
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "video",
                            "video": final_frames,
                            "max_pixels": self.max_pixels,
                            "min_pixels": self.min_pixels,
                            # "total_pixels": 20480 * 28 * 28
                        },
                        {
                            "type": "text",
                            "text": message[1]['content'][1]['text']
                            # "text": prompt_template.format(message[1]['content'][1]['text'])
                        }
                    ],
                }
            ]
            final_messages_batch.append(final_message)

        inference_start = time.time()
        if self.model_type in ["Qwen2.5", "Qwen2-VL"]:
            response = []
            for single_message in final_messages_batch:
                single_response = self._batch_inference(
                    self.vl_model_set,
                    [single_message],
                    **generation_kwargs
                )
                response.append(single_response[0])
        else:
            response = self._batch_inference(
                self.vl_model_set,
                final_messages_batch,
                **generation_kwargs
            )

        return response

    def _process_video_for_clip(
            self,
            video: str,
            text_features: torch.Tensor,
            fps: float,
            batch_size: int,
            query: Union[str, List[str]]
    ):
        if hasattr(self.clip_model, 'parameters'):
            model_device = next(self.clip_model.parameters()).device

            if model_device.type != 'cuda' and torch.cuda.is_available():
                self.clip_model = self.clip_model.cuda()

        stats = {
            'batch_times': [],
            'batch_gpu_memory': [],
            'total_time': 0,
            'peak_gpu_memory': 0,
            'num_batches': 0
        }
        initial_memory = 0

        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
            initial_memory = torch.cuda.memory_allocated() / 1024 ** 3  # GB

        VR = VideoReader(video, ctx=cpu(0))
        total_frames = len(VR)
        video_fps = VR.get_avg_fps()

        sample_interval = int(video_fps / fps)
        expected_frame_indices = list(range(0, total_frames, sample_interval))
        if len(expected_frame_indices) == 0 or expected_frame_indices[-1] != total_frames - 1:
            expected_frame_indices.append(total_frames - 1)
        video_path = video
        video_name = video_path.split('/')[-1].split('.')[0]
        cache_dir = os.environ.get('AIR_CLIP_CACHE_DIR', os.path.join(os.getcwd(), 'clip_cache'))
        os.makedirs(cache_dir, exist_ok=True)
        cache_path = os.path.join(cache_dir, f"{video_name}_frames{total_frames}_fps{fps}.npz")

        cache_valid = False
        if os.path.exists(cache_path):
            data = np.load(cache_path, allow_pickle=True)
            cached_frame_indices = data['frame_indices'].tolist()

            if cached_frame_indices == expected_frame_indices:
                cache_valid = True
                frame_indices = cached_frame_indices
                all_image_features = torch.from_numpy(data['image_features']).to(self.device)
                with autocast():
                    similarities_raw = compute_clip_similarity(
                        text_features, all_image_features,
                        model=self.clip_model,
                        model_type=self.clip_model_type,
                        normalization_method="none"
                    )
            else:
                os.remove(cache_path)

        if not cache_valid:
            frame_indices = expected_frame_indices
            all_image_features = []
            all_similarities = []

            num_batches = (len(frame_indices) + batch_size - 1) // batch_size
            stats['num_batches'] = num_batches

            for i in range(0, len(frame_indices), batch_size):

                batch_indices = frame_indices[i:i + batch_size]
                frames = VR.get_batch(batch_indices).asnumpy()
                pil_images = [Image.fromarray(frame) for frame in frames]

                with autocast():
                    if self.clip_model_type in ["openai", "open_clip", "eva_clip", "long_clip"]:
                        image_features = encode_images_clip(
                            self.clip_model, pil_images,
                            preprocess=self.clip_preprocess_or_processor,
                            processor=None,
                            model_type=self.clip_model_type,
                            device=self.device
                        )
                    else:
                        image_features = encode_images_clip(
                            self.clip_model, pil_images,
                            preprocess=None,
                            processor=self.clip_preprocess_or_processor,
                            model_type=self.clip_model_type,
                            device=self.device
                        )

                    all_image_features.append(image_features.cpu())

                    batch_similarities = compute_clip_similarity(
                        text_features, image_features,
                        model=self.clip_model,
                        model_type=self.clip_model_type,
                        normalization_method="none"
                    )

                all_similarities.append(batch_similarities)

            similarities_raw = torch.cat(all_similarities, dim=1)

            all_image_features = torch.cat(all_image_features, dim=0)
            np.savez(cache_path,
                     image_features=all_image_features.cpu().numpy(),
                     frame_indices=np.array(list(frame_indices), dtype=np.int32))

        similarities, norm_params = self._normalize_full_video(similarities_raw)
        similarities = similarities.cpu().numpy()

        if similarities.size > 0:
            sim_percentiles = np.percentile(similarities, [10, 25, 50, 75, 90])

            suggested_threshold = sim_percentiles[1]

            norm_params['suggested_i2t_threshold'] = float(suggested_threshold)
            norm_params['percentiles'] = {
                'p10': float(sim_percentiles[0]),
                'p25': float(sim_percentiles[1]),
                'p50': float(sim_percentiles[2]),
                'p75': float(sim_percentiles[3]),
                'p90': float(sim_percentiles[4])
            }

        if isinstance(query, str) or len(query) == 1:
            similarities = similarities.squeeze(0)

        return similarities, similarities_raw, frame_indices, stats, norm_params


    @torch.inference_mode()
    def compute_video_query_similarity(
            self,
            video_path: Union[str, List[str], List[List[dict]], VideoReader],
            query: Union[str, List[str], None] = None,
            clip_model_name: Optional[str] = None,
            fps: float = 1.0,
            batch_size: int = 80,
            return_frame_indices: bool = True,
            return_stats: bool = False
    ):
        if clip_model_name and clip_model_name != self.clip_model_name:
            self._initialize_clip_model(clip_model_name)
        if self.text_features is None:
            with autocast():
                if self.clip_model_type in ["openai", "open_clip", "eva_clip", "long_clip"]:
                    self.text_features = encode_text_clip(
                        self.clip_model, query,
                        tokenizer=self.clip_tokenizer,
                        processor=None,
                        model_type=self.clip_model_type,
                        device=self.device
                    )
                else:
                    self.text_features = encode_text_clip(
                        self.clip_model, query,
                        tokenizer=None,
                        processor=self.clip_preprocess_or_processor,
                        model_type=self.clip_model_type,
                        device=self.device
                    )
        self.text_features = self.text_features.to(self.device)

        similarities, raw_sim, frame_indices, stats, norm_params = self._process_video_for_clip(
            video_path, self.text_features, fps, batch_size, query
        )

        if return_stats:
            return similarities, raw_sim, frame_indices, stats, norm_params
        elif return_frame_indices:
            return similarities, raw_sim, frame_indices, norm_params
        else:
            return similarities, raw_sim

    def find_relevant_segments(self, sims, frame_indices, threshold='adaptive',
                               min_segment_frames=3,
                               video_fps=24.0,
                               merge_gap_seconds=1.0, gmm_coefficient=0.8):
        segments = []
        similarities = np.power(sims, 1)
        if threshold == 'adaptive':
            gmm = GaussianMixture(n_components=2, random_state=42)
            gmm.fit(similarities.reshape(-1, 1))
            means = gmm.means_.flatten()
            stds = np.sqrt(gmm.covariances_.flatten())
            high_mean_idx = np.argmax(means)
            threshold = means[high_mean_idx] - gmm_coefficient * stds[high_mean_idx]
            threshold = max(threshold, (means[0] + means[1]) / 2)

        above_threshold = similarities >= threshold

        if not np.any(above_threshold):
            return segments

        current_start = None
        current_frames = []

        for i, (is_above, sim, frame_idx) in enumerate(zip(above_threshold, similarities, frame_indices)):
            if is_above:
                if current_start is None:
                    current_start = i
                    current_frames = [i]
                else:
                    current_frames.append(i)
            else:
                if current_start is not None and len(current_frames) >= min_segment_frames:
                    segment_sims = similarities[current_frames]
                    segments.append({
                        'start_idx': current_start,
                        'end_idx': current_frames[-1],
                        'start_frame': frame_indices[current_start],
                        'end_frame': frame_indices[current_frames[-1]],
                        'start_time': frame_indices[current_start] / video_fps,
                        'end_time': frame_indices[current_frames[-1]] / video_fps,
                        'duration': (frame_indices[current_frames[-1]] - frame_indices[current_start]) / video_fps,
                        'num_frames': len(current_frames),
                        'avg_similarity': segment_sims.mean(),
                        'max_similarity': segment_sims.max(),
                        'min_similarity': segment_sims.min(),
                        'frame_indices': current_frames
                    })
                current_start = None
                current_frames = []

        if current_start is not None and len(current_frames) >= min_segment_frames:
            segment_sims = similarities[current_frames]
            segments.append({
                'start_idx': current_start,
                'end_idx': current_frames[-1],
                'start_frame': frame_indices[current_start],
                'end_frame': frame_indices[current_frames[-1]],
                'start_time': frame_indices[current_start] / video_fps,
                'end_time': frame_indices[current_frames[-1]] / video_fps,
                'duration': (frame_indices[current_frames[-1]] - frame_indices[current_start]) / video_fps,
                'num_frames': len(current_frames),
                'avg_similarity': segment_sims.mean(),
                'max_similarity': segment_sims.max(),
                'min_similarity': segment_sims.min(),
                'frame_indices': current_frames
            })

        if merge_gap_seconds > 0 and len(segments) > 1:
            merged_segments = []
            current_segment = segments[0]

            for next_segment in segments[1:]:
                gap = next_segment['start_time'] - current_segment['end_time']

                if gap <= merge_gap_seconds:
                    all_frames = current_segment['frame_indices'] + next_segment['frame_indices']
                    all_sims = similarities[all_frames]

                    current_segment = {
                        'start_idx': current_segment['start_idx'],
                        'end_idx': next_segment['end_idx'],
                        'start_frame': current_segment['start_frame'],
                        'end_frame': next_segment['end_frame'],
                        'start_time': current_segment['start_time'],
                        'end_time': next_segment['end_time'],
                        'duration': next_segment['end_time'] - current_segment['start_time'],
                        'num_frames': len(all_frames),
                        'avg_similarity': all_sims.mean(),
                        'max_similarity': all_sims.max(),
                        'min_similarity': all_sims.min(),
                        'frame_indices': all_frames
                    }
                else:
                    merged_segments.append(current_segment)
                    current_segment = next_segment

            merged_segments.append(current_segment)
            segments = merged_segments

        segments.sort(key=lambda x: x['start_time'])

        return segments, threshold

    def adaptive_initial_sampling(self, video, event_splits,
                                  use_sampling=True, max_frames_per_event=4,
                                  min_event_length=10,
                                  initial_sampling_ratio=0.5, total_budget=None,
                                  clip_similarities=None, clip_frame_indices=None):
        vr = video
        total_frames = len(vr)

        if not use_sampling:
            all_indices = []
            for event in event_splits:
                start, end = int(event[0]), int(event[1])
                all_indices.extend(range(start, min(end + 1, total_frames)))
            sampled_frames = vr.get_batch(all_indices).asnumpy()
            return sampled_frames, all_indices

        initial_target = max(self.N_candidate * 2, total_budget * 2)

        event_data = []

        for event_idx, event in enumerate(event_splits):
            start_frame = int(event[0])
            end_frame = int(event[1])
            length = end_frame - start_frame + 1

            if length > 0 and end_frame < total_frames:
                avg_similarity = 0.0
                event_sims = []
                clip_count = 0

                if clip_similarities is not None and clip_frame_indices is not None:
                    for i, frame_idx in enumerate(clip_frame_indices):
                        if start_frame <= frame_idx <= end_frame:
                            event_sims.append(clip_similarities[i])

                    clip_count = len(event_sims)
                    if event_sims:
                        avg_similarity = np.mean(event_sims)

                event_data.append({
                    'start': start_frame,
                    'end': end_frame,
                    'length': length,
                    'avg_similarity': avg_similarity,
                    'clip_count': clip_count,
                    'allocated_frames': 1
                })

        if clip_similarities is not None:
            original_count = len(event_data)
            event_data = [e for e in event_data if e['clip_count'] > 0]


        if not event_data:
            return np.array([]), []

        if len(event_data) > initial_target:
            event_data.sort(key=lambda x: x['avg_similarity'], reverse=True)
            selected_events = []
            selected_frames = 0

            for event in event_data:
                frames_needed = 1

                if selected_frames + frames_needed <= initial_target:
                    selected_events.append(event)
                    selected_frames += frames_needed
                else:
                    break

            remaining_budget = initial_target - selected_frames
            if remaining_budget > 0:
                for event in selected_events:
                    if remaining_budget <= 0:
                        break

                    if event['length'] > min_event_length:
                        extra_frames = min(
                            max_frames_per_event - 1,
                            event['length'] // 10,
                            remaining_budget
                        )
                        event['allocated_frames'] = 1 + extra_frames
                        remaining_budget -= extra_frames
                    else:
                        event['allocated_frames'] = 1
            else:
                for event in selected_events:
                    event['allocated_frames'] = 1

            valid_events = selected_events
        else:
            valid_events = event_data

            valid_events.sort(key=lambda x: x['avg_similarity'], reverse=True)

            for event in valid_events:
                event['allocated_frames'] = 1

            allocated_count = len(valid_events)

            event_index = 0
            while allocated_count < initial_target:
                event = valid_events[event_index]
                if event['allocated_frames'] < min(event['length'], max_frames_per_event):
                    event['allocated_frames'] += 1
                    allocated_count += 1

                event_index = (event_index + 1) % len(valid_events)

                all_maxed = all(
                    e['allocated_frames'] >= min(e['length'], max_frames_per_event)
                    for e in valid_events
                )
                if all_maxed:
                    break

        sampled_indices = []

        for event_idx, event in enumerate(valid_events):
            start_frame = event['start']
            end_frame = event['end']
            num_frames = event['allocated_frames']
            event_length = event['length']

            if event_length <= min_event_length:
                mid_frame = (start_frame + end_frame) // 2
                if clip_frame_indices is not None:
                    in_range_indices = [idx for idx in clip_frame_indices
                                        if start_frame <= idx <= end_frame]
                    if in_range_indices:
                        closest_idx = min(in_range_indices,
                                          key=lambda x: abs(x - mid_frame))
                        sampled_indices.append(closest_idx)
                else:
                    sampled_indices.append(mid_frame)
                continue

            if clip_similarities is not None and clip_frame_indices is not None:
                event_points = []
                for i, frame_idx in enumerate(clip_frame_indices):
                    if start_frame <= frame_idx <= end_frame:
                        event_points.append((frame_idx, clip_similarities[i]))

                if event_points:
                    event_points.sort(key=lambda x: x[1], reverse=True)

                    selected_count = min(num_frames, len(event_points))
                    selected_indices = [point[0] for point in event_points[:selected_count]]

                    sampled_indices.extend(sorted(selected_indices))
            else:
                if num_frames == 1:
                    mid_frame = (start_frame + end_frame) // 2
                    sampled_indices.append(mid_frame)
                else:
                    step = (event_length - 1) / (num_frames - 1) if num_frames > 1 else 0
                    positions = [start_frame + int(i * step) for i in range(num_frames)]
                    sampled_indices.extend(positions)

        sampled_indices = sorted(list(set(sampled_indices)))

        if not sampled_indices:
            sampled_indices = list(np.linspace(0, total_frames - 1, min(8, total_frames), dtype=int))

        sampled_frames = vr.get_batch(sampled_indices).asnumpy()

        return sampled_frames, sampled_indices

    def iteration_algorithm(self,
                            video_frames, video_frames_indices,
                            VR, **kwargs):
        budget = kwargs.get("budget")
        min_frame_distance = kwargs.get("min_frame_distance", 6)
        query = kwargs.get("query", None)
        video_fps = kwargs.get("video_fps", 24.0)
        i2t_norm_params = kwargs.get("i2t_norm_params", None)

        clip_i2t_threshold = kwargs.get("clip_i2t_threshold", 0.25)

        hisa_weights = kwargs.get('hisa_weights', None)
        clip_sims = kwargs['clip_similarities']
        raw_clip_sims = kwargs.get('raw_clip_similarities', None)
        clip_indices = kwargs['clip_frame_indices']

        T = len(VR)

        ordered_indices = []
        ordered_frames = []

        all_frame_indices = []
        if video_frames_indices and isinstance(video_frames_indices[0], list):
            for event_indices in video_frames_indices:
                all_frame_indices.extend(event_indices)
        else:
            all_frame_indices = list(video_frames_indices)

        ordered_indices = list(all_frame_indices)
        ordered_frames = list(video_frames)

        dense_query_sim = np.full(T, np.nan)
        raw_dense_query_sim = np.full(T, np.nan)
        mapped_count = 0

        for i, idx in enumerate(clip_indices):
            if 0 <= idx < T:
                dense_query_sim[idx] = clip_sims[i]

                if raw_clip_sims is not None:
                    if raw_clip_sims.dim() == 2:
                        raw_dense_query_sim[idx] = float(raw_clip_sims[0, i].item())
                    elif raw_clip_sims.dim() == 1:
                        raw_dense_query_sim[idx] = float(raw_clip_sims[i].item())
                    elif isinstance(raw_clip_sims, np.ndarray):
                        raw_dense_query_sim[idx] = float(raw_clip_sims[i])
                    else:
                        raw_dense_query_sim[idx] = float(raw_clip_sims[i])
                mapped_count += 1

        current_sampled = list(ordered_indices) # F_initial

        iteration = 1
        early_stop = False

        dense_analyzed_query_sim = np.full(T, np.nan)
        total_budget = kwargs.get("budget")
        F_final = []  # F^*_final
        F_probe = current_sampled  # F
        # make iteration fixed number, adding budget overall limit.
        while iteration <= self.max_iteration and not early_stop:

            new_sampled_indices, dense_analyzed_query_sim, updated_sampled, early_stop, iter_vlm_responses = self.iteration_step(
                probed_indices=F_probe,
                query_sim=dense_query_sim,
                analyze_sim=dense_analyzed_query_sim,
                budget=total_budget,
                weights=hisa_weights,
                N_candidate=max(budget//self.max_iteration, self.N_candidate),
                VR=VR,  # decord VideoReader instance
                query=kwargs.get('query', ''),
                generation_kwargs=kwargs.get('generation_kwargs', {}),
                # save_dir=iter_dir,
                low_score_threshold=2,
                min_frame_distance=min_frame_distance,
                iteration=iteration,
            )

            new_frames = VR.get_batch(new_sampled_indices).asnumpy()

            for frame_idx, frame_array in zip(new_sampled_indices, new_frames):
                assert self.text_features is not None
                pil_image = Image.fromarray(frame_array)
                with autocast():
                    if self.clip_model_type in ["openai", "open_clip", "eva_clip", "long_clip"]:
                        image_features = encode_images_clip(
                            self.clip_model, [pil_image],
                            preprocess=self.clip_preprocess_or_processor,
                            processor=None,
                            model_type=self.clip_model_type,
                            device=self.device
                        )
                    else:
                        image_features = encode_images_clip(
                            self.clip_model, [pil_image],
                            preprocess=None,
                            processor=self.clip_preprocess_or_processor,
                            model_type=self.clip_model_type,
                            device=self.device
                        )

                clip_i2t_sim = compute_clip_similarity(
                    self.text_features, image_features,
                    model=self.clip_model,
                    model_type=self.clip_model_type,
                    normalization_method="none"
                ).item()

                # update the original sim using raw_clip_sims
                if clip_i2t_sim > i2t_norm_params['max']:
                    i2t_norm_params['max'] = clip_i2t_sim
                    min_val = i2t_norm_params['min']
                    max_val = i2t_norm_params['max']
                    range_val = max_val - min_val
                    if range_val > 1e-8:
                        valid_mask = ~np.isnan(raw_dense_query_sim)
                        dense_query_sim[valid_mask] = (raw_dense_query_sim[valid_mask] - min_val) / range_val
                        dense_query_sim[valid_mask] = np.clip(dense_query_sim[valid_mask], 0, 1)

                elif clip_i2t_sim < i2t_norm_params['min']:
                    i2t_norm_params['min'] = clip_i2t_sim
                    min_val = i2t_norm_params['min']
                    max_val = i2t_norm_params['max']
                    range_val = max_val - min_val
                    if range_val > 1e-8:
                        valid_mask = ~np.isnan(raw_dense_query_sim)
                        dense_query_sim[valid_mask] = (raw_dense_query_sim[valid_mask] - min_val) / range_val
                        dense_query_sim[valid_mask] = np.clip(dense_query_sim[valid_mask], 0, 1)
                if i2t_norm_params:
                    clip_i2t_sim_norm = (clip_i2t_sim - i2t_norm_params['min']) / \
                                        (i2t_norm_params['max'] - i2t_norm_params['min'] + 1e-8)
                    clip_i2t_sim_norm = np.clip(clip_i2t_sim_norm, 0, 1)
                else:
                    clip_i2t_sim_norm = clip_i2t_sim
                dense_query_sim[frame_idx] = clip_i2t_sim_norm
                raw_dense_query_sim[frame_idx] = clip_i2t_sim

            filtered_indices = [
                idx for idx in new_sampled_indices if dense_query_sim[idx] >= clip_i2t_threshold
            ]
            filtered_updated_indices = [
                idx for idx in updated_sampled if dense_query_sim[idx] >= clip_i2t_threshold
            ]
            # filtered_updated_indices = updated_sampled
            F_probe = sorted(list(set(F_probe + filtered_updated_indices + filtered_indices)))  # update F
            F_final = sorted(list(set(F_final + filtered_updated_indices)))

            if not filtered_indices:

                iteration += 1
                continue

            iteration += 1

        if len(F_final) == 0:
            F_final = ordered_indices[:min(budget, len(ordered_indices))]

        if len(F_final) > budget:
            final_scores = []
            for idx in F_final:
                clip_score = dense_query_sim[idx] if not np.isnan(dense_query_sim[idx]) else 0.5
                vlm_score = dense_analyzed_query_sim[idx] if not np.isnan(dense_analyzed_query_sim[idx]) else 0

                if vlm_score > 0:
                    combined = clip_score / 4 + vlm_score / 5
                else:
                    combined = clip_score
                final_scores.append({'idx': idx, 'score': combined})

            final_scores.sort(key=lambda x: x['score'], reverse=True)

            final_selection = []
            selected_indices = []

            for item in final_scores:
                current_idx = item['idx']
                is_valid_distance = True

                for selected_idx in selected_indices:
                    if abs(current_idx - selected_idx) < min_frame_distance:
                        is_valid_distance = False
                        break

                if is_valid_distance:
                    final_selection.append(item)
                    selected_indices.append(current_idx)

                if len(final_selection) >= budget:
                    break

            F_final = sorted([item['idx'] for item in final_selection])



        final_frames = VR.get_batch(F_final).asnumpy()  # order, indices - > frames

        processed_frames = []
        processed_frame_indices = []

        for idx, frame in zip(F_final, final_frames):
            processed_frames.append(Image.fromarray(frame))
            processed_frame_indices.append(int(idx))

        return processed_frames, processed_frame_indices

    def calculate_adaptive_budget(self, video_duration_seconds, max_budget):
        if self.model_type == 'VILA-1.5':
            return 8
        elif self.model_type == 'LLaVA-OV':
            return 8
        elif self.model_type == 'Qwen2-VL':
            return 32
        else:
            return 32

    def _normalize(self, data: np.ndarray) -> np.ndarray:
        """
        Min-max normalizes a 1D numpy array to the  range.
        Handles the edge case where all values are the same.
        """
        min_val = np.min(data)
        max_val = np.max(data)
        if max_val == min_val:
            # If all values are identical, return an array of 0.5s
            return np.full_like(data, 0.5, dtype=float)
        return (data - min_val) / (max_val - min_val)

    def iteration_step(self,
                       probed_indices: list[int],
                       query_sim: np.ndarray,
                       analyze_sim: np.ndarray,
                       budget: int,
                       weights: dict = None,
                       N_candidate: int = 24,
                       VR=None,
                       query: str = "",
                       generation_kwargs: dict = None,
                       save_dir=None,
                       low_score_threshold: int = 2,
                       min_frame_distance: int = 4,
                       iteration: int = 1,
                       ):
        if weights is None:
            weights = {'length': 0.2, 'complexity': 0.5}

        total_frames = len(query_sim)

        combined_sim = query_sim.copy()
        has_vlm_info = not np.all(np.isnan(analyze_sim))

        already_analyzed = []

        if has_vlm_info:
            for i in range(total_frames):
                if not np.isnan(analyze_sim[i]):
                    combined_sim[i] = np.nansum(query_sim[i]) / 2 + np.nansum(analyze_sim[i] / 4.0)
                    already_analyzed.append(i)
        else:
            combined_sim = query_sim

        intervals = []
        if not probed_indices:
            intervals.append([0, total_frames - 1])
        else:
            probed_indices = sorted(probed_indices)
            if probed_indices[0] > 0:
                intervals.append([0, probed_indices[0]])
            for i in range(len(probed_indices) - 1):
                intervals.append([probed_indices[i], probed_indices[i + 1]])
            if probed_indices[-1] < total_frames - 1:
                intervals.append([probed_indices[-1], total_frames - 1])

        valid_intervals = [(start, end) for start, end in intervals if end > start + 1]

        if not valid_intervals:
            return [], analyze_sim, [], False

        all_scores = []
        all_complexity = []
        for start, end in valid_intervals:
            # 1. Calculate Length (L)
            length = float(end - start)
            # Handle edge cases for very short intervals

            explored_count = sum(1 for idx in range(start, end)
                                 if idx in probed_indices)
            exploration_ratio = explored_count / length if length > 0 else 1.0

            exploration_bonus = 1.0 + (3.0 / (1 + np.exp(10 * (exploration_ratio - 0.7))))
            # exploration_bonus = np.exp(2.0 * (1.0 - exploration_ratio))

            if length > 2:
                interval_sim = combined_sim[start:end]
                # 2. Calculate Relevance (R)
                # This is the `np.nanmean` from your code.
                relevance = np.nanmean(interval_sim) + (
                            np.nanmean(combined_sim[start]) + np.nanmean(combined_sim[end])) / 6
                # 3. Calculate Complexity (C) using Total Variation
                # This is ∫|sim'(t)|dt, calculated as the sum of absolute differences.
                # We normalize by (length - 1) to get the *average change per frame step*,
                # making the metric less dependent on the interval's length itself.
                non_nan_sim = interval_sim[~np.isnan(interval_sim)]
                # 3.b. Check if there are enough valid points to calculate change
                if len(non_nan_sim) < 2:
                    complexity = 0.0
                else:
                    total_variation = np.sum(np.abs(np.diff(non_nan_sim)))
                    complexity = total_variation / (length - 1.0)
                # 4. Assemble the RCL Score
                # Bonus from Complexity
                complexity_bonus = 1.0 + weights['complexity'] * complexity
                # Bonus from Length (log-scaled for diminishing returns)
                length_bonus = 1.0 + weights['length'] * np.log(length)
                # Final multiplicative score
                score = relevance * complexity_bonus * length_bonus * exploration_bonus
            else:
                score = 0.01
            all_scores.append(score)
        scores = np.array(all_scores)
        sorted_interval_indices = np.argsort(scores)[::-1]

        all_possible_candidates = []

        for interval_idx in sorted_interval_indices:
            start, end = valid_intervals[int(interval_idx)]
            all_possible_candidates.append(start)
            all_possible_candidates.append(end)
        all_possible_candidates = sorted(set(all_possible_candidates))
        if not all_possible_candidates:
            return [], analyze_sim, [], False

        initial_candidates = all_possible_candidates[:N_candidate]

        frames_needed_analyze = [f for f in initial_candidates if f not in already_analyzed]
        vlm_responses_dict = {}
        if len(frames_needed_analyze) > 0:

            candidate_frames = VR.get_batch(frames_needed_analyze).asnumpy()
            analyzed_scores, vlm_responses = self.analyze_frames_with_query(
                candidate_frames,
                query=query,
                sub_batch_size=min(len(frames_needed_analyze), self.sub_batch),
                generation_kwargs=generation_kwargs or {},
                save_dir=save_dir
            )

            for i, idx in enumerate(frames_needed_analyze):
                analyze_sim[idx] = analyzed_scores[i]
                combined_sim[idx] = combined_sim[idx] + analyzed_scores[i] / 4.0
                vlm_responses_dict[idx] = vlm_responses[i]

        else:
            if DEBUG:
                print(f"\n{len(initial_candidates)} all analyzed")

        final_candidates = []
        neutral_candidates = []
        final_scores = []
        low_score_indices = []
        sampled_indices = []
        for idx in initial_candidates:
            if not np.isnan(analyze_sim[idx]) and analyze_sim[idx] > low_score_threshold + 1:   # 3 is neutral, unsure.
                final_candidates.append(idx)
                final_scores.append(analyze_sim[idx])
            elif not np.isnan(analyze_sim[idx]) and analyze_sim[idx] == low_score_threshold + 1:
                neutral_candidates.append(idx)
            else:
                low_score_indices.append(idx)

        if len(final_candidates) > (budget * 2):
            return [], analyze_sim, final_candidates, True  # early stop

        sampled_indices = list(final_candidates)

        high_score_ratio = len(final_candidates) / (budget * 2)
        neutral_ratio = len(neutral_candidates) / (budget * 2)

        if neutral_candidates:
            neutral_sorted = sorted(neutral_candidates,
                                    key=lambda idx: combined_sim[idx],
                                    reverse=True)

            if high_score_ratio < 0.5:
                sampled_indices.extend(neutral_sorted)
            else:
                top_neutral_count = max(len(neutral_sorted) // 2, budget // 4)
                sampled_indices.extend(neutral_sorted[:top_neutral_count])

        current_coverage = len(sampled_indices) / (budget * 2)

        if low_score_indices:
            low_sorted = sorted(low_score_indices,
                                key=lambda idx: combined_sim[idx],
                                reverse=True)

            if current_coverage < 0.5 and iteration <= 3:
                explore_count = min(len(low_sorted)//2, budget // 2)
            elif high_score_ratio > 0.8:
                explore_count = min(2, len(low_sorted)//2)
            else:
                explore_count = min(
                    len(low_sorted) // 3,
                    max(budget // 6, 4)
                )

            sampled_indices.extend(low_sorted[:explore_count])

        if not sampled_indices:
            all_available = sorted(
                list(range(len(combined_sim))),
                key=lambda idx: combined_sim[idx] if not np.isnan(combined_sim[idx]) else -1,
                reverse=True
            )
            min_batch = max(budget // self.max_iteration, self.N_candidate // 2, 8)
            sampled_indices = all_available[:min_batch]

        sampled_indices = list(dict.fromkeys(sampled_indices))

        temp_analyze_sim = np.full_like(analyze_sim, np.nan, dtype=float)
        log_budget_factor = np.log2(max(8, budget)) / np.log2(8)  # 1.0 for budget=8, 4.0 for budget=128

        base_samples = int(8 * log_budget_factor)  # 8-32

        if len(sampled_indices) > 0:
            peak_samples = len(sampled_indices) * (2 + log_budget_factor * 0.5)
            sample_budget = int((base_samples + peak_samples) / 2)
        else:
            sample_budget = base_samples

        sample_budget = int(sample_budget * (1.3 - iteration * 0.1))

        sample_budget = max(8, min(sample_budget, budget))

        for idx in sampled_indices:
            score = analyze_sim[idx]
            if np.isnan(query_sim[idx]):
                temp_analyze_sim[idx] = score
                # print(f"caution! video {len(query_sim)} {query} query_sim[{idx}] is NaN, this should not happen in HISA logic")
            else:
                temp_analyze_sim[idx] = score * 3 / 5 + np.nansum(query_sim[idx])

        new_sampled_indices = sample_logic(temp_analyze_sim,
                                           sample_budget,
                                           min_frame_length=min_frame_distance,
                                           HIGH_SIM_THRESHOLD=2)

        return new_sampled_indices, analyze_sim, sampled_indices, False, vlm_responses_dict

    def analyze_frames_with_query(self, sampled_frames, query, sub_batch_size=12,
                                  generation_kwargs=None, debug=False, save_dir="./vlm_analysis"):

        if generation_kwargs is None:
            generation_kwargs = {}

        analysis_messages = create_analysis_messages(
            sampled_frames, query, self.max_pixels, self.min_pixels, self.model_type
        )

        scores = []
        num_sub_batches = (len(analysis_messages) + sub_batch_size - 1) // sub_batch_size

        all_responses = []
        for j in range(num_sub_batches):
            start_idx = j * sub_batch_size
            end_idx = min((j + 1) * sub_batch_size, len(analysis_messages))
            batch_messages = analysis_messages[start_idx:end_idx]

            batch_response = self._batch_inference(
                self.vl_model_set,
                batch_messages,
                **generation_kwargs
            )
            all_responses.extend(batch_response)

            batch_scores = []
            for response in batch_response:
                score = extract_score(response.lower())
                batch_scores.append(score)

            scores.extend(batch_scores)

        return scores, all_responses

    def _normalize_full_video(self, similarity):

        if similarity.dtype not in [torch.float32, torch.float64]:
            similarity = similarity.float()

        min_s = similarity.min()
        max_s = similarity.max()

        p10 = torch.quantile(similarity.flatten(), 0.1)
        p90 = torch.quantile(similarity.flatten(), 0.9)

        norm_params = {
            'min': min_s.item(),
            'max': max_s.item(),
            'p10': p10.item(),
            'p90': p90.item(),
            'mean': similarity.mean().item(),
            'std': similarity.std().item()
        }

        if (max_s - min_s) < 1e-8:
            return torch.full_like(similarity, 0.5), norm_params

        normalized = (similarity - min_s) / (max_s - min_s)

        return normalized, norm_params

    def _extract_from_messages(self, messages):
        if isinstance(messages, list) and len(messages) > 0 and isinstance(messages[0], list):
            extracted_videos = []
            extracted_queries = []
            for message in messages:
                try:
                    video = message[1]['content'][0]['video']
                    text = message[1]['content'][1]['text']
                    if '？' in text:
                        text = text.split('？')[0] + '？'
                    elif '?' in text:
                        text = text.split('?')[0] + '?'
                    extracted_videos.append(video)
                    extracted_queries.append(text)
                except:
                    raise ValueError(f"Invalid message format: {message}")

            video_path = extracted_videos[0] if len(extracted_videos) == 1 else extracted_videos
            query = extracted_queries[0] if len(extracted_queries) == 1 else extracted_queries
            return video_path, query
        else:
            return None, None