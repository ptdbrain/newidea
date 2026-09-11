#!/usr/bin/env bash
# Prepare the offline KIVI input used by a compute-node job.
set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
KIVI_ROOT="$REPO_ROOT/third_party/KIVI"
COMMIT_FILE="$REPO_ROOT/third_party/KIVI.commit"

if ! command -v git >/dev/null 2>&1; then
  echo "Git is required on the preparation machine." >&2
  exit 1
fi
if ! command -v tar >/dev/null 2>&1; then
  echo "tar is required to create the offline KIVI bundle." >&2
  exit 1
fi

git -C "$REPO_ROOT" submodule update --init --recursive third_party/KIVI
[[ -f "$KIVI_ROOT/quant/new_pack.py" ]] || {
  echo "KIVI checkout is incomplete: $KIVI_ROOT/quant/new_pack.py" >&2
  exit 1
}

actual_commit="$(git -C "$KIVI_ROOT" rev-parse HEAD)"
expected_commit="$(python -c 'from src.provenance import KIVI_COMMIT; print(KIVI_COMMIT)')"
[[ "$actual_commit" == "$expected_commit" ]] || {
  echo "KIVI commit mismatch: expected $expected_commit, got $actual_commit" >&2
  exit 1
}

printf '%s\n' "$actual_commit" > "$COMMIT_FILE"
BUNDLE_PATH="${1:-$REPO_ROOT/phase1-kivi-bundle-${actual_commit}.tar.gz}"
if [[ "$BUNDLE_PATH" != /* ]]; then
  BUNDLE_PATH="$REPO_ROOT/$BUNDLE_PATH"
fi

tar \
  --exclude='third_party/KIVI/.git' \
  -czf "$BUNDLE_PATH" \
  -C "$REPO_ROOT" \
  third_party/KIVI \
  third_party/KIVI.commit

echo "KIVI bundle created: $BUNDLE_PATH"
echo "KIVI commit: $actual_commit"
