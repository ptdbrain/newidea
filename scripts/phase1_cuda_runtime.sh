#!/usr/bin/env bash
# CUDA/PyTorch selection helpers for the Phase 1 launcher.

PHASE1_TORCH_VERSION="2.9.1"

phase1_cuda_runtime_version() {
  local variant="$1"
  case "$variant" in
    cu128) printf '%s\n' "12.8" ;;
    cu130) printf '%s\n' "13.0" ;;
    *)
      echo "Unsupported PYTORCH_CUDA_VARIANT: $variant. Supported values: cu128, cu130" >&2
      return 2
      ;;
  esac
}

phase1_torch_index_url() {
  local variant="$1"
  phase1_cuda_runtime_version "$variant" >/dev/null || return
  printf 'https://download.pytorch.org/whl/%s\n' "$variant"
}

phase1_dependency_fingerprint() {
  local requirements_path="$1"
  local variant="$2"

  phase1_cuda_runtime_version "$variant" >/dev/null || return
  [[ -f "$requirements_path" ]] || {
    echo "Requirements file not found: $requirements_path" >&2
    return 2
  }
  command -v sha256sum >/dev/null 2>&1 || {
    echo "sha256sum is required to fingerprint Phase 1 dependencies." >&2
    return 1
  }

  {
    cat "$requirements_path"
    printf '\ntorch==%s\nvariant=%s\n' "$PHASE1_TORCH_VERSION" "$variant"
  } | sha256sum | awk '{print $1}'
}
