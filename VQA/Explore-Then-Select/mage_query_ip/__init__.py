"""Query-conditioned Mage codec reweight (MVP).

Route B: patch-level motion/residual proxies + fixed budget K (same-K ablations).
True cv-preinfer bitcost/MV dumps are not available (route A deferred).
"""

from .question_type import QuestionType, classify_question
from .codec_reweight import (
    ReweightResult,
    reweight_codec_canvases,
    select_patches_samek,
    jaccard,
)

__all__ = [
    "QuestionType",
    "classify_question",
    "ReweightResult",
    "reweight_codec_canvases",
    "select_patches_samek",
    "jaccard",
]
