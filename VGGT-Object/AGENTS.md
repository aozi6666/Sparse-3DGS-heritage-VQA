# VGGT-Object engineering boundaries

- Current default after paired review: full Depth Head fine-tuning, whole Aggregator and other heads frozen. Conv-LoRA is a controlled alternative, not assumed superior.
- User-authorized incremental A16/A8 mode: LoRA only in global blocks18–23 qkv/proj, original Aggregator weights frozen; full DPT retained. GPU smoke is user-launched only, 20 steps by supplied phase.
- Reuse ../3DGS/vggt and ../.venv-ddgs-vggt; do not edit upstream modules or other experiments.
- No automatic dataset/model download. Raw input directories belong to user; never modify or delete originals.
- Separate each experiment under experiments/<run>. Keep best + latest for depth head/adapter only; no full VGGT copies. Do not overwrite an existing run.
- Do not launch GPU training/inference as part of setup/check. Explicit train/eval/gs commands are GPU actions.
- All depth is floating OpenCV camera Z; conversions must declare native semantics. No silent pose/scale guesses.
- Object/sequence disjoint train/val/test; do not use test objects for model selection.
- Tests with synthetic data prove interface/numerical correctness, not research accuracy or convergence.
- Document gaps honestly: raw data conversion and GPU end-to-end checks remain pending until real inputs/hardware exist.
