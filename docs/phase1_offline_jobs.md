# Offline Phase 1 jobs

Some batch systems run jobs on nodes that have neither Git nor outbound
network access. Prepare the KIVI source once on a machine with Git:

```bash
bash scripts/prepare_phase1_kivi_bundle.sh
```

Upload the generated `phase1-kivi-bundle.tar.gz` to the job input directory.
The one-command offline entrypoint extracts it when needed:

```bash
PHASE1_KIVI_BUNDLE=/path/to/phase1-kivi-bundle.tar.gz \
PYTORCH_CUDA_VARIANT=cu128 \
  bash scripts/run_phase1_offline.sh
```

The bundle contains the pinned KIVI source and `third_party/KIVI.commit`; the
launcher then uses that staged source without invoking Git.
When the bundle is committed at the repository root, no environment variable
or separate upload is needed.

Use `SKIP_BOOTSTRAP=1` when the job already has the Python environment and
local model checkpoints. The launcher will fail early if the KIVI source or
commit manifest is missing.

The active Python environment must already contain `torch==2.9.1` built for
the selected CUDA runtime. Use `PYTORCH_CUDA_VARIANT=cu128` for CUDA 12.8 or
`PYTORCH_CUDA_VARIANT=cu130` for CUDA 13.0. Offline mode does not download
Python packages, Torch, models, embeddings, or datasets, and CUDA wheels are
not stored in Git. Preflight rejects a CPU wheel, a mismatched CUDA build, or a
KIVI Triton kernel that cannot compile on the assigned GPU.

## Prefill failure guarantees

- Checkpoint validation fails before model loading when local model or
  tokenizer assets are incomplete.
- A sample prefill error stops the stage and preserves the original exception.
- `prefill.done` is written only after every dataset row has an Origin cache.
- KIVI materialization fails immediately when an Origin source cache is
  missing.
- After correcting a failed prefill, use a new `RUN_ID`; do not resume a run
  whose prefill marker was created by an older launcher.

These checks prove artifact completeness. A real GPU run is still required to
validate the model-specific Transformers and CUDA execution path.
