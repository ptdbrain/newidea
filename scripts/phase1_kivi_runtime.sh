#!/usr/bin/env bash

# Runtime helpers for a KIVI checkout that may have been staged without Git.
# The source tree and KIVI.commit are the portable job inputs.

phase1_kivi_source_ready() {
  local kivi_root="$1"
  [[ -f "$kivi_root/quant/new_pack.py" ]]
}

phase1_kivi_commit() {
  local kivi_root="$1"
  local commit_file="$2"
  local commit=""

  if [[ -s "$commit_file" ]]; then
    IFS= read -r commit < "$commit_file" || true
    [[ -n "$commit" ]] || {
      echo "KIVI commit manifest is empty: $commit_file" >&2
      return 1
    }
    printf '%s\n' "$commit"
    return 0
  fi

  if command -v git >/dev/null 2>&1; then
    git -C "$kivi_root" rev-parse HEAD
    return 0
  fi

  echo "KIVI commit manifest is missing and Git is unavailable: $commit_file" >&2
  return 1
}

phase1_kivi_prepare() {
  local repo_root="$1"
  local kivi_root="$2"
  local commit_file="$3"

  if ! phase1_kivi_source_ready "$kivi_root"; then
    if ! command -v git >/dev/null 2>&1; then
      echo "KIVI source is not pre-staged and Git is unavailable." >&2
      echo "Prepare the KIVI bundle on a machine with Git, then place it at:" >&2
      echo "  $kivi_root" >&2
      return 1
    fi
    echo "[bootstrap] initializing third_party/KIVI"
    git -C "$repo_root" submodule update --init --recursive third_party/KIVI
  fi

  if ! phase1_kivi_source_ready "$kivi_root"; then
    echo "KIVI checkout is incomplete: missing $kivi_root/quant/new_pack.py" >&2
    return 1
  fi

  if [[ ! -s "$commit_file" ]]; then
    if ! command -v git >/dev/null 2>&1; then
      echo "KIVI source is present but its commit manifest is missing:" >&2
      echo "  $commit_file" >&2
      return 1
    fi
    mkdir -p "$(dirname "$commit_file")"
    git -C "$kivi_root" rev-parse HEAD > "$commit_file"
  fi

  phase1_kivi_commit "$kivi_root" "$commit_file" >/dev/null
  echo "[bootstrap] using pre-staged KIVI: $kivi_root"
}

phase1_project_commit() {
  local repo_root="$1"
  if command -v git >/dev/null 2>&1; then
    git -C "$repo_root" rev-parse HEAD 2>/dev/null || printf 'unavailable\n'
  else
    printf 'unavailable\n'
  fi
}
