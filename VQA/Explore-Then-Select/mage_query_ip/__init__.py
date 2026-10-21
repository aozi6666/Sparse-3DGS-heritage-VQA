"""Query-conditioned Mage codec reweight (MVP).

Dual-mode same-K: pixel proxy (``queryip``) and official bitcost (``queryip_bitcost``).
Route A (true MV+residual) archived — open stack only exports bitcost.
"""

from .question_type import QuestionType, classify_question
from .codec_reweight import (
    ReweightResult,
    reweight_codec_canvases,
    select_patches_samek,
    jaccard,
)
from .mvres_score import select_patches_bitcost, score_patches_bitcost

__all__ = [
    "QuestionType",
    "classify_question",
    "ReweightResult",
    "reweight_codec_canvases",
    "select_patches_samek",
    "select_patches_bitcost",
    "score_patches_bitcost",
    "jaccard",
]
