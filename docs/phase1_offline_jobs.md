# Offline Phase 1 jobs

Some batch systems run jobs on nodes that have neither Git nor outbound
network access. Prepare the KIVI source once on a machine with Git:

```bash
bash scripts/prepare_phase1_kivi_bundle.sh
```

Upload the generated `phase1-kivi-bundle-<commit>.tar.gz` to the job input
directory. The one-command offline entrypoint extracts it when needed:

```bash
PHASE1_KIVI_BUNDLE=/path/to/phase1-kivi-bundle-<commit>.tar.gz \
  bash scripts/run_phase1_offline.sh
```

The bundle contains the pinned KIVI source and `third_party/KIVI.commit`; the
launcher then uses that staged source without invoking Git.

Use `SKIP_BOOTSTRAP=1` when the job already has the Python environment and
local model checkpoints. The launcher will fail early if the KIVI source or
commit manifest is missing.
