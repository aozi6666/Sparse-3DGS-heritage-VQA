"""Query-IP：Mage codec 选区包装。

主推模式：``queryip``（像素 proxy）。
【归档/不用作贡献】``queryip_dyn``/``sta``（α/β）；``queryip_clip``；
``queryip_bitcost``；``queryip_region*``；``queryip_edge*``（CLIP 语义选区已停）。
真 MV 路线 A：不做。本包保持轻量：不在此 import torch/transformers。
"""

from .question_type import QuestionType, classify_question
from .frame_budget import allocate_budgets

__all__ = [
    "QuestionType",
    "classify_question",
    "allocate_budgets",
    "ReweightResult",
    "reweight_codec_canvases",
    "select_patches_samek",
    "select_patches_bitcost",
    "score_patches_bitcost",
    "jaccard",
    "ClipFrameScorer",
    "sample_video_frames",
]


def __getattr__(name: str):
    if name in (
        "ReweightResult",
        "reweight_codec_canvases",
        "select_patches_samek",
        "jaccard",
    ):
        from . import codec_reweight as _cr

        return getattr(_cr, name)
    if name in ("select_patches_bitcost", "score_patches_bitcost"):
        from . import mvres_score as _mv

        return getattr(_mv, name)
    if name in ("ClipFrameScorer", "sample_video_frames"):
        from . import clip_frame_score as _cf

        return getattr(_cf, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
