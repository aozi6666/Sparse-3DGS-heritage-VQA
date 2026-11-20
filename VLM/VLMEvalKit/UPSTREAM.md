# Upstream

- Source: https://github.com/open-compass/VLMEvalKit
- Vendored into this monorepo under `VQA/VLMEvalKit/` **including local project edits**
  (notably `vlmeval/config.py`, `vlmeval/dataset/videomme.py`, `vlmeval/vlm/__init__.py`).
- Base commit at vendor time: see `patches/vlmevalkit/` companion notes.
- Diffs vs clean upstream (for sync): `../patches/vlmevalkit/local-edits.patch`

This directory is a **plain vendored tree** (no nested `.git`). Do not treat it as a live submodule.
