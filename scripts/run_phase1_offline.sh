#!/usr/bin/env bash
# Run Phase 1 from a pre-staged offline KIVI bundle.
set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

KIVI_ENTRYPOINT="$REPO_ROOT/third_party/KIVI/quant/new_pack.py"
if [[ ! -f "$KIVI_ENTRYPOINT" ]]; then
  BUNDLE_PATH="${PHASE1_KIVI_BUNDLE:-}"
  if [[ -z "$BUNDLE_PATH" ]]; then
    shopt -s nullglob
    bundles=("$REPO_ROOT"/phase1-kivi-bundle-*.tar.gz)
    if [[ "${#bundles[@]}" -eq 1 ]]; then
      BUNDLE_PATH="${bundles[0]}"
    else
      echo "Set PHASE1_KIVI_BUNDLE to the offline KIVI bundle." >&2
      exit 1
    fi
  fi
  [[ -f "$BUNDLE_PATH" ]] || {
    echo "KIVI bundle not found: $BUNDLE_PATH" >&2
    exit 1
  }
  command -v tar >/dev/null 2>&1 || {
    echo "tar is required to extract the offline KIVI bundle." >&2
    exit 1
  }
  tar -xzf "$BUNDLE_PATH" -C "$REPO_ROOT"
fi

[[ -f "$KIVI_ENTRYPOINT" ]] || {
  echo "Offline KIVI bundle is incomplete: $KIVI_ENTRYPOINT" >&2
  exit 1
}

export SKIP_BOOTSTRAP=1
exec bash "$REPO_ROOT/scripts/run_phase1_main.sh" "$@"
