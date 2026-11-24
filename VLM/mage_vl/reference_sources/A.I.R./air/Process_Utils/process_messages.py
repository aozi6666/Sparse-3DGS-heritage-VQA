"""Prompt/message construction for the A.I.R. inference pipeline.

Exposes six public helpers used by ``air/model.py``:

* ``create_analysis_messages``  – per-frame relevance scoring prompts
* ``videomme_preprocess``       – strip boilerplate from Video-MME prompts
* ``videomme_w_sub_preprocess`` – Video-MME with subtitles variant
* ``lvb_preprocess``            – LongVideoBench
* ``mlvu_preprocess``           – MLVU
* ``extract_score``             – parse an integer score from a VLM reply
"""
import re
from PIL import Image


# ---------------------------------------------------------------------------
# Benchmark-specific prompt preprocessors
# ---------------------------------------------------------------------------

def videomme_preprocess(full_prompt: str) -> str:
    """Extract just the question and options from a Video-MME prompt.

    Video-MME prompts contain an instruction on the first line and a
    ``The best answer is:`` suffix on the last line. This strips both.
    """
    lines = full_prompt.strip().split('\n')
    query_text = '\n'.join(lines[1:-1])
    return query_text


def videomme_w_sub_preprocess(full_prompt: str) -> str:
    """Extract the question+options from the subtitle-augmented Video-MME prompt.

    Handles both the subtitled and non-subtitled templates used by lmms-eval.
    """
    # Case 1: subtitles are included.
    if "This video's subtitles are listed below" in full_prompt:
        if "based on the video and the subtitles" in full_prompt:
            parts = full_prompt.split(
                "Respond with only the letter (A, B, C, or D) of the correct option."
            )
            if len(parts) > 1:
                question_and_options = parts[1].split("The best answer is:")[0].strip()
                return question_and_options

    # Case 2: no subtitles.
    elif "Select the best answer to the following multiple-choice question based on the video." in full_prompt:
        parts = full_prompt.split(
            "Respond with only the letter (A, B, C, or D) of the correct option."
        )
        if len(parts) > 1:
            question_and_options = parts[1].split("The best answer is:")[0].strip()
            return question_and_options

    # Fallback: return the original prompt unchanged.
    return full_prompt


def lvb_preprocess(full_prompt: str) -> str:
    """Strip the trailing answer cue from a LongVideoBench prompt."""
    lines = full_prompt.strip().split('\n')
    query_text = '\n'.join(lines[:-1])
    return query_text


def mlvu_preprocess(full_prompt: str) -> str:
    """Strip the trailing ``Only give the best option.`` / ``Best option: (`` lines from an MLVU prompt."""
    lines = full_prompt.strip().split('\n')
    query_text = '\n'.join(lines[:-2])
    return query_text


# ---------------------------------------------------------------------------
# Relevance-scoring message builder
# ---------------------------------------------------------------------------

ANALYSIS_PROMPT_TEMPLATE = """You are an expert visual reasoner. Step by step analyze how well this image matches the following query (with options):
Query: {query}

Rate the relevance from 1 to 5, using these exact definitions:
1 – Not relevant at all: no relation between image content and the query.
2 – Slightly relevant: only minor contextual hints, but not central to answering.
3 – Moderately relevant: contains preparatory or follow-up context (e.g. setup actions) related to the query.
4 – Highly relevant: shows clear evidence to support query but still has ambiguity or missing detail.
5 – Perfectly matches: fully sufficient and unambiguous evidence to answer the query.

Avoid overthink. Trust your immediate judgment.
Examples:
– Image of an empty room (no people or objects) → Reasoning: There is nothing related to the query. Score: 1
– Image showing exactly the queried action (clear, direct match) → Reasoning: It perfectly depicts the required event. Score: 5
Respond only in this format:

Score: <1-5>
Reasoning: <A brief one-sentence justification>
"""


def create_analysis_messages(event_frames, query, max_pixels, min_pixels, model_type="Qwen2.5"):
    """Build per-frame relevance-scoring chat messages.

    Args:
        event_frames: Iterable of HxWxC ``np.ndarray`` frames.
        query: The benchmark question (with options) to score against.
        max_pixels: Processor upper pixel budget.
        min_pixels: Processor lower pixel budget.
        model_type: One of ``"Qwen2.5"``, ``"Qwen2"``, ``"InternVL"``,
            ``"LLaVA-OV"``, or ``"VILA-1.5"``. Controls the chat schema
            used (Qwen flat vs. assistant+user two-turn).

    Returns:
        A list of chat message lists, one per frame, ready for
        ``*_batch_inference`` helpers.
    """
    analysis_prompt = ANALYSIS_PROMPT_TEMPLATE.format(query=query)

    batch_messages = []
    for frame_array in event_frames:
        frame = Image.fromarray(frame_array.squeeze())

        if model_type in ("InternVL", "LLaVA-OV", "VILA-1.5"):
            message = [
                {
                    "role": "assistant",
                    "content": [{"type": "text", "text": "You are a helpful assistant"}],
                },
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "image": frame,
                            "max_pixels": max_pixels,
                            "min_pixels": min_pixels,
                        },
                        {"type": "text", "text": analysis_prompt},
                    ],
                },
            ]
        else:  # Qwen2.5 / Qwen2 single-turn schema
            message = [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "image": frame,
                            "max_pixels": max_pixels,
                            "min_pixels": min_pixels,
                        },
                        {"type": "text", "text": analysis_prompt},
                    ],
                }
            ]

        batch_messages.append(message)
    return batch_messages


# ---------------------------------------------------------------------------
# Score parsing
# ---------------------------------------------------------------------------

_SCORE_PATTERNS = (
    r'score:\s*(\d+)',
    r'rating:\s*(\d+)',
    r'\b([1-5])\s*/\s*5\b',
    r'\b([1-5])\s*out of\s*5\b',
    r'\b([1-5])\b',
)


def extract_score(response: str) -> int:
    """Parse a 1..5 integer score from a VLM reply. Returns 3 on parse failure."""
    text = response.lower()
    for pattern in _SCORE_PATTERNS:
        match = re.search(pattern, text)
        if match:
            score = int(match.group(1))
            if 1 <= score <= 5:
                return score
    return 3
