"""Download SD base + ControlNet-Tile into this directory (GaussianObject/models/)."""
from huggingface_hub import hf_hub_download

# Mainline SD1.5-compatible base (Realistic Vision V5.1, noVAE)
hf_hub_download(
    repo_id="SG161222/Realistic_Vision_V5.1_noVAE",
    filename="Realistic_Vision_V5.1.safetensors",
    local_dir=".",
)

# ControlNet-Tile (unchanged)
hf_hub_download(
    repo_id="lllyasviel/ControlNet-v1-1",
    revision="69fc48b9cbd98661f6d0288dc59b59a5ccb32a6b",
    filename="control_v11f1e_sd15_tile.pth",
    local_dir=".",
)
