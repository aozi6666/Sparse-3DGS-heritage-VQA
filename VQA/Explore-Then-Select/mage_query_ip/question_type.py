"""Keyword-based static vs dynamic question typing (no LLM call)."""

from __future__ import annotations

from enum import Enum
import re


class QuestionType(str, Enum):
    STATIC = "static"
    DYNAMIC = "dynamic"


# Heritage / detail oriented (prefer low-motion / I-like retention).
_STATIC_PATTERNS = [
    r"纹饰", r"纹样", r"题记", r"碑文", r"铭文", r"材质", r"形制", r"颜色",
    r"图案", r"浮雕", r"文字", r"写了什么", r"是什么纹",
    r"\bpattern\b", r"\binscription\b", r"\btexture\b", r"\bmaterial\b",
    r"\bcolor\b", r"\bcolour\b", r"\bwhat.*(on|written|carved)\b",
    r"\bdescribe\b.*(detail|appearance|look)",
    r"jersey", r"logo", r"scoreboard", r"banner", r"sign",
]

# Event / motion oriented (prefer high-motion / P-like retention).
_DYNAMIC_PATTERNS = [
    r"走过", r"走过了", r"仪式", r"动作", r"谁在", r"发生了什么", r"移动",
    r"奔跑", r"进球", r"传球", r"射门", r"庆祝",
    r"\bwho\b", r"\bwalk", r"\bmov(e|ing)\b", r"\brunn", r"\bgoal\b",
    r"\bpass(ed|ing)?\b", r"\baction\b", r"\bhappen", r"\bevent\b",
    r"\bplayer\b", r"\bkick", r"\bcelebrat",
]


def classify_question(question: str, default: QuestionType = QuestionType.DYNAMIC) -> QuestionType:
    """Return static/dynamic. Default dynamic if neither side clearly wins."""
    text = (question or "").strip().lower()
    if not text:
        return default
    static_hits = sum(1 for p in _STATIC_PATTERNS if re.search(p, text, flags=re.I))
    dynamic_hits = sum(1 for p in _DYNAMIC_PATTERNS if re.search(p, text, flags=re.I))
    if static_hits > dynamic_hits:
        return QuestionType.STATIC
    if dynamic_hits > static_hits:
        return QuestionType.DYNAMIC
    return default


# α(q), β(q) for score' = α·motion + β·residual_proxy
WEIGHT_TABLE = {
    # Static: down-weight motion so ego-motion / busy backgrounds lose priority.
    QuestionType.STATIC: {"alpha": 0.25, "beta": 0.75},
    # Dynamic: up-weight motion proxies.
    QuestionType.DYNAMIC: {"alpha": 0.85, "beta": 0.15},
}


def weights_for(qtype: QuestionType) -> dict[str, float]:
    return dict(WEIGHT_TABLE[qtype])
