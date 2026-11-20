# Historical experiment maintenance — 2026-11-20

This branch has independent historical experiment commits. Its core source files and history are preserved; it is not replaced by main. VCAR, Seg, dedicated segmentation caches and the obsolete VQA directory are not present at this branch tip. Runtime weights/logs/environments are excluded from new tracking.

The historical CLIP weight is removed only from the current Git index. Its original metadata is recorded below, its history is retained, and no server-side training asset was deleted by this index-only operation. Prepare CLIP-ViT-L/14 weights locally at the original path when reproducing this branch; do not interpret the model file as obsolete mathematics/code.

For current VGGT-Object DPT/LoRA and VLM workflows use main/vlm. This legacy branch keeps its original GaussianObject layout.

```json
{
  "original_branch": "finish_vggt",
  "original_head": "49d662827e95609992e985b4c4f7ddcc85d8835f",
  "weights": [
    {
      "path": "GaussianObject/pretrained/clip-vit-large-patch14/model.safetensors",
      "git_blob": "34cca545bfc07f566fa79345178b6db761f9d53b",
      "bytes": 134,
      "original_pointer": "version https://git-lfs.github.com/spec/v1\noid sha256:778d02eb9e707c3fbaae0b67b79ea0d1399b52e624fb634f2f19375ae7c047c3\nsize 492265168\n"
    }
  ]
}
```
