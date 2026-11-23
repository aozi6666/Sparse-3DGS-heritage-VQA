"""Shared paths and the historical answer parser for the midpoint control."""
import re
from pathlib import Path

HERE = Path(__file__).resolve().parent
DATA = HERE.parent / "mage_vl/data/VideoMME"
def letter(raw):
    match = re.fullmatch(r"\s*([ABCD])(?:\s*[.)](?:\s+[^\n]*)?)?\s*", str(raw))
    return match[1] if match else None


