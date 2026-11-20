"""Default paths for Cultural LoRA (local Mage-VL weights + datasets)."""

from __future__ import annotations

from pathlib import Path

MAGE_VL_ROOT = Path(__file__).resolve().parents[1]
VQA_ROOT = MAGE_VL_ROOT.parent
REPO_ROOT = VQA_ROOT.parent

DEFAULT_MODEL_DIR = MAGE_VL_ROOT / "models"
DEFAULT_CIQI_DIR = MAGE_VL_ROOT / "data" / "CiQi-VQA"
DEFAULT_MSVD_DIR = MAGE_VL_ROOT / "data" / "MSVD-QA"
DEFAULT_OUTPUT_DIR = MAGE_VL_ROOT / "outputs" / "cultural_lora"

CIQI_SUBSET_JSONL = "train_16k.jsonl"
CIQI_MANIFEST = "subset_manifest_16k.json"
MSVD_SUBSET_DIR = "train_8k"
MSVD_MANIFEST = "subset_manifest_8k.json"
