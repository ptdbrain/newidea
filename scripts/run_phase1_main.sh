#!/usr/bin/env bash
# Run the reproducible Phase 1 main experiment (100 prompts).
#
# This launcher bootstraps its own virtualenv, installs the pinned Python
# dependencies, initializes KIVI, downloads the model and MPNet checkpoint,
# and then runs the complete pipeline. Existing environments/checkpoints are
# reused and interrupted runs can be resumed.

set -Eeuo pipefail

usage() {
  cat <<'EOF'
Usage:
  bash scripts/run_phase1_main.sh
  bash scripts/run_phase_1
  RESUME=1 RUN_DIR=/abs/path/to/logs/phase1_main_<run-id> bash scripts/run_phase1_main.sh
  DRY_RUN=1 bash scripts/run_phase1_main.sh
  bash scripts/run_phase1_main.sh --bootstrap-only
  SKIP_BOOTSTRAP=1 bash scripts/run_phase1_main.sh

Environment overrides:
  MODEL_NAME=Llama-3.2-1B
  MODEL_ID=meta-llama/Llama-3.2-1B
  MODEL_PATH=$PWD/.models/Llama-3.2-1B
  EMBEDDING_ID=sentence-transformers/all-mpnet-base-v2
  EMBEDDING_PATH=$PWD/.models/all-mpnet-base-v2
  HF_TOKEN=<required for gated Hugging Face model downloads>
  VENV_DIR=$PWD/.venv
  DEVICE=cuda:0
  DTYPE=float16
  SEED=42
  RUN_COLLISION_PLUS=0   # set to 1 to add frozen CPA calibration/evaluation
  MAX_NEW_TOKENS=16
  RESUME=0
  RUN_DIR=<auto-created timestamped directory>
  SKIP_BOOTSTRAP=0       # use an already prepared Python environment
  PIP_INSTALL_ARGS=      # optional extra arguments for pip install

The attack stage requires an explicit authorization acknowledgement. This
launcher supplies --i-understand-risks; run it only on data and systems you
are authorized to test.

The default Llama checkpoint is gated by Hugging Face. Export HF_TOKEN before
running, or set MODEL_ID to a model you are authorized to download. The
embedding checkpoint and default dataset are public.
EOF
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  usage
  exit 0
fi

BOOTSTRAP_ONLY=0
for arg in "$@"; do
  case "$arg" in
    --bootstrap-only) BOOTSTRAP_ONLY=1 ;;
    --no-bootstrap) SKIP_BOOTSTRAP=1 ;;
    *) echo "Unknown argument: $arg" >&2; usage >&2; exit 2 ;;
  esac
done

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

MODEL_NAME="${MODEL_NAME:-Llama-3.2-1B}"
MODEL_ID="${MODEL_ID:-meta-llama/Llama-3.2-1B}"
MODEL_ROOT="${MODEL_ROOT:-$REPO_ROOT/.models}"
MODEL_PATH="${MODEL_PATH:-$MODEL_ROOT/$MODEL_NAME}"
EMBEDDING_ID="${EMBEDDING_ID:-sentence-transformers/all-mpnet-base-v2}"
EMBEDDING_PATH="${EMBEDDING_PATH:-$MODEL_ROOT/all-mpnet-base-v2}"
MODEL_REVISION="${MODEL_REVISION:-}"
EMBEDDING_REVISION="${EMBEDDING_REVISION:-}"
VENV_DIR="${VENV_DIR:-$REPO_ROOT/.venv}"
DEVICE="${DEVICE:-cuda:0}"
DTYPE="${DTYPE:-float16}"
SEED="${SEED:-42}"
RUN_COLLISION_PLUS="${RUN_COLLISION_PLUS:-0}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-16}"
RESUME="${RESUME:-0}"
DRY_RUN="${DRY_RUN:-0}"
SKIP_BOOTSTRAP="${SKIP_BOOTSTRAP:-0}"

if [[ -z "${PIP_INSTALL_ARGS:-}" ]]; then
  PIP_INSTALL_ARGS=()
else
  read -r -a PIP_INSTALL_ARGS <<< "$PIP_INSTALL_ARGS"
fi

absolute_path() {
  local path="$1"
  local parent
  parent="$(dirname "$path")"
  if [[ "$DRY_RUN" != "1" ]]; then
    mkdir -p "$parent"
  fi
  if [[ -d "$parent" ]]; then
    printf '%s/%s\n' "$(cd "$parent" && pwd)" "$(basename "$path")"
  elif [[ "$path" = /* ]]; then
    printf '%s\n' "$path"
  else
    printf '%s/%s\n' "$REPO_ROOT/$parent" "$(basename "$path")"
  fi
}

MODEL_PATH="$(absolute_path "$MODEL_PATH")"
EMBEDDING_PATH="$(absolute_path "$EMBEDDING_PATH")"

if [[ "$SKIP_BOOTSTRAP" != "1" && "$DRY_RUN" != "1" ]]; then
  command -v git >/dev/null 2>&1 || {
    echo "git is required to initialize third_party/KIVI." >&2
    exit 1
  }
  HOST_PYTHON="${PYTHON:-}"
  if [[ -z "$HOST_PYTHON" ]]; then
    HOST_PYTHON="$(command -v python3 || command -v python || true)"
  fi
  [[ -n "$HOST_PYTHON" ]] || {
    echo "Python 3 is required. Install python3 and python3-venv, then rerun." >&2
    exit 1
  }
  if [[ ! -x "$VENV_DIR/bin/python" ]]; then
    echo "[bootstrap] creating virtualenv: $VENV_DIR"
    "$HOST_PYTHON" -m venv "$VENV_DIR" || {
      echo "Could not create a virtualenv. On Debian/Ubuntu install python3-venv." >&2
      exit 1
    }
  fi
  export PATH="$VENV_DIR/bin:$PATH"
  REQUIREMENTS_HASH="$(sha256sum requirements.txt | awk '{print $1}')"
  REQUIREMENTS_STAMP="$VENV_DIR/.phase1_requirements.sha256"
  if [[ ! -f "$REQUIREMENTS_STAMP" || "$(<"$REQUIREMENTS_STAMP")" != "$REQUIREMENTS_HASH" ]]; then
    echo "[bootstrap] installing Python dependencies from requirements.txt"
    python -m pip install --disable-pip-version-check --upgrade pip setuptools wheel
    python -m pip install --disable-pip-version-check "${PIP_INSTALL_ARGS[@]}" -r requirements.txt
    printf '%s\n' "$REQUIREMENTS_HASH" > "$REQUIREMENTS_STAMP"
  else
    echo "[bootstrap] Python dependencies already match requirements.txt"
  fi
  echo "[bootstrap] initializing third_party/KIVI"
  git submodule update --init --recursive
else
  if [[ "$DRY_RUN" == "1" ]]; then
    echo "[dry-run] would create/reuse $VENV_DIR, install requirements.txt, initialize KIVI, and download checkpoints"
  fi
  if [[ -x "$VENV_DIR/bin/python" ]]; then
    export PATH="$VENV_DIR/bin:$PATH"
  fi
fi

checkpoint_has_weights() {
  local target_dir="$1"
  find "$target_dir" -type f \
    \( -name '*.safetensors' -o -name '*.bin' -o -name '*.pt' \) \
    -print -quit | grep -q .
}

download_checkpoint() {
  local repo_id="$1"
  local target_dir="$2"
  local revision="$3"
  local label="$4"
  if [[ -f "$target_dir/config.json" || -f "$target_dir/modules.json" ]] && checkpoint_has_weights "$target_dir"; then
    echo "[bootstrap] $label checkpoint already exists: $target_dir"
    return 0
  fi
  if [[ "$DRY_RUN" == "1" ]]; then
    echo "[dry-run] would download $label: $repo_id -> $target_dir"
    return 0
  fi
  mkdir -p "$target_dir"
  echo "[bootstrap] downloading $label checkpoint: $repo_id"
  if ! python - "$repo_id" "$target_dir" "$revision" <<'PY'
import sys

from huggingface_hub import snapshot_download

repo_id, target_dir, revision = sys.argv[1:]
kwargs = {"repo_id": repo_id, "local_dir": target_dir}
if revision:
    kwargs["revision"] = revision
snapshot_download(**kwargs)
PY
  then
    cat >&2 <<'EOF'
Checkpoint download failed. For the default gated Llama model, accept the
model license at https://huggingface.co/meta-llama/Llama-3.2-1B and export:

  export HF_TOKEN=hf_...

Then rerun the same command. You can also set MODEL_ID to an accessible model.
EOF
    return 1
  fi
}

download_checkpoint "$MODEL_ID" "$MODEL_PATH" "$MODEL_REVISION" "base model"
download_checkpoint "$EMBEDDING_ID" "$EMBEDDING_PATH" "$EMBEDDING_REVISION" "embedding model"

if [[ "$BOOTSTRAP_ONLY" == "1" ]]; then
  if [[ "$DRY_RUN" == "1" ]]; then
    echo "Bootstrap dry-run completed. Environment would be: $VENV_DIR"
  else
    echo "Bootstrap completed. Environment: $VENV_DIR"
  fi
  echo "Base model: $MODEL_PATH"
  echo "Embedding model: $EMBEDDING_PATH"
  exit 0
fi

if [[ -n "${RUN_DIR:-}" ]]; then
  RUN_DIR="$(absolute_path "$RUN_DIR")"
  RUN_ID="${RUN_ID:-$(basename "$RUN_DIR" | sed 's/^phase1_main_//')}"
else
  RUN_ID="${RUN_ID:-$(date -u +%Y%m%dT%H%M%SZ)}"
  RUN_DIR="$REPO_ROOT/logs/phase1_main_${RUN_ID}"
fi

DATASET_PATH="${DATASET_PATH:-$RUN_DIR/phase1_main100_${RUN_ID}.jsonl}"
MANIFEST_PATH="${MANIFEST_PATH:-$RUN_DIR/dataset.provenance.json}"
STATE_DIR="$RUN_DIR/state"
CACHE_ROOT="$REPO_ROOT/cache/$DTYPE/$(basename "${DATASET_PATH%.jsonl}")/$MODEL_NAME"
RAW_DIR="$RUN_DIR/raw_attacks"
AGG_DIR="$RUN_DIR/aggregate"

export PYTHONPATH="$REPO_ROOT:$REPO_ROOT/third_party/KIVI${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONHASHSEED="$SEED"

if [[ "$DRY_RUN" != "1" ]]; then
  mkdir -p "$RUN_DIR" "$STATE_DIR" "$RAW_DIR" "$AGG_DIR"
  exec > >(tee -a "$RUN_DIR/run.log") 2>&1
fi

echo "Phase 1 main run: $RUN_ID"
echo "Run directory: $RUN_DIR"
echo "Model: $MODEL_NAME ($MODEL_PATH)"
echo "Dataset: $DATASET_PATH"
echo "Device/dtype: $DEVICE / $DTYPE"

if [[ "$DRY_RUN" != "1" ]]; then
  exec 9>"$RUN_DIR/.run.lock"
  if ! flock -n 9; then
    echo "Another process is already using $RUN_DIR" >&2
    exit 2
  fi
fi

run_stage() {
  local stage="$1"
  shift
  if [[ "$DRY_RUN" == "1" ]]; then
    echo "[dry-run] would execute $stage"
    return 0
  fi
  local marker="$STATE_DIR/${stage}.done"
  if [[ "$RESUME" == "1" && -f "$marker" ]]; then
    echo "[resume] skip $stage"
    return 0
  fi
  echo "===== START $stage ====="
  "$@"
  date -u +%Y-%m-%dT%H:%M:%SZ > "$marker"
  echo "===== DONE $stage ====="
}

stage_preflight() {
  if [[ "$DRY_RUN" == "1" ]]; then
    return 0
  fi
  [[ -d "$MODEL_PATH" ]] || { echo "MODEL_PATH not found: $MODEL_PATH" >&2; return 1; }
  [[ -f "$MODEL_PATH/config.json" ]] || { echo "MODEL_PATH is not a Transformers checkpoint: $MODEL_PATH" >&2; return 1; }
  [[ -d "$EMBEDDING_PATH" ]] || { echo "EMBEDDING_PATH not found: $EMBEDDING_PATH" >&2; return 1; }
  [[ -f "$EMBEDDING_PATH/modules.json" || -f "$EMBEDDING_PATH/config.json" ]] || { echo "EMBEDDING_PATH is not a local Sentence Transformers checkpoint: $EMBEDDING_PATH" >&2; return 1; }
  python -c 'import torch, transformers, datasets, sentence_transformers; print("torch", torch.__version__); print("transformers", transformers.__version__); print("cuda", torch.cuda.is_available())'
  if [[ "$DEVICE" == cuda* ]]; then
    python -c 'import sys, torch; sys.exit("CUDA is required for the KIVI Triton kernels, but torch.cuda.is_available() is false") if not torch.cuda.is_available() else None'
  fi
  python -c 'from src.kivi_adapter import KIVIConfig; print("KIVI adapter import: OK", KIVIConfig(4, 4, 32, 32))'
  {
    echo "run_id=$RUN_ID"
    echo "model_name=$MODEL_NAME"
    echo "model_id=$MODEL_ID"
    echo "model_path=$MODEL_PATH"
    echo "model_revision=$MODEL_REVISION"
    echo "embedding_id=$EMBEDDING_ID"
    echo "embedding_path=$EMBEDDING_PATH"
    echo "embedding_revision=$EMBEDDING_REVISION"
    echo "device=$DEVICE"
    echo "dtype=$DTYPE"
    echo "seed=$SEED"
    echo "run_collision_plus=$RUN_COLLISION_PLUS"
    echo "dataset_path=$DATASET_PATH"
    echo "cache_root=$CACHE_ROOT"
    python --version
    python -c 'import torch; print("torch=" + torch.__version__); print("cuda=" + str(torch.version.cuda)); print("cuda_available=" + str(torch.cuda.is_available()))'
    git rev-parse HEAD
    git -C third_party/KIVI rev-parse HEAD
    git status --short
    if command -v nvidia-smi >/dev/null 2>&1; then nvidia-smi; fi
  } > "$RUN_DIR/environment.txt"
  (
    cd "$MODEL_PATH"
    find . -type f -print0 | sort -z | xargs -0 sha256sum
  ) > "$RUN_DIR/model.files"
  (
    cd "$EMBEDDING_PATH"
    find . -type f -print0 | sort -z | xargs -0 sha256sum
  ) > "$RUN_DIR/embedding.files"
}

stage_prepare_dataset() {
  python -m dataset.prepare_phase1 \
    --output "$DATASET_PATH" \
    --manifest "$MANIFEST_PATH" \
    --tokenizer "$MODEL_PATH" \
    --source-dataset "${SOURCE_DATASET:-natong19/lmsys-chat-1m-filtered}" \
    --revision "${SOURCE_REVISION:-1887528a022e25be62eb9bb15e62675f2a69353b}" \
    --license "${SOURCE_LICENSE:-CC-BY-4.0}" \
    --short 25 --medium 50 --long 25 \
    --max-length 128 --candidate-multiplier 4 \
    --seed "$SEED"
}

stage_prefill() {
  python inference/get_kvcache.py \
    --model-name "$MODEL_NAME" \
    --model-path "$MODEL_PATH" \
    --dataset "$DATASET_PATH" \
    --dtype "$DTYPE" \
    --device "$DEVICE" \
    --max-samples 100 \
    --minimal-cache
}

stage_materialize() {
  python defense/baseline/kivi_kvcache.py --model-name "$MODEL_NAME" --dataset-path "$DATASET_PATH" --cache-root "$CACHE_ROOT" --dtype "$DTYPE" --device "$DEVICE" --k-bits 4 --v-bits 4 --group-size 32 --residual-length 32 --mode standard
  python defense/baseline/kivi_kvcache.py --model-name "$MODEL_NAME" --dataset-path "$DATASET_PATH" --cache-root "$CACHE_ROOT" --dtype "$DTYPE" --device "$DEVICE" --k-bits 4 --v-bits 4 --group-size 32 --residual-length 32 --mode full_prompt_quantized
  python defense/baseline/kivi_kvcache.py --model-name "$MODEL_NAME" --dataset-path "$DATASET_PATH" --cache-root "$CACHE_ROOT" --dtype "$DTYPE" --device "$DEVICE" --k-bits 2 --v-bits 2 --group-size 32 --residual-length 32 --mode standard
  python defense/baseline/kivi_kvcache.py --model-name "$MODEL_NAME" --dataset-path "$DATASET_PATH" --cache-root "$CACHE_ROOT" --dtype "$DTYPE" --device "$DEVICE" --k-bits 2 --v-bits 2 --group-size 32 --residual-length 32 --mode full_prompt_quantized
}

stage_validate_cache() {
  python -m scripts.validate_phase1_cache \
    --dataset "$DATASET_PATH" \
    --cache-root "$CACHE_ROOT" \
    --output "$RUN_DIR/cache_validation.json"
}

attack_condition() {
  local label="$1"
  local protect_type="$2"
  local output="$RAW_DIR/${label}.jsonl"
  if [[ "$DRY_RUN" != "1" && -f "$output" ]]; then
    mv "$output" "$output.partial.$(date -u +%Y%m%dT%H%M%SZ)"
  fi
  python attack/attacks.py \
    --target-model-name "$MODEL_NAME" \
    --base-model-name "$MODEL_NAME" \
    --base-model-path "$MODEL_PATH" \
    --eval-model-path "$EMBEDDING_PATH" \
    --dataset-path "$DATASET_PATH" \
    --cache-root "$CACHE_ROOT" \
    --protect-type "$protect_type" \
    --dtype "$DTYPE" --device "$DEVICE" \
    --run-inversion --run-collision --run-injection \
    --output "$output" \
    --i-understand-risks
  [[ "$DRY_RUN" == "1" || -s "$output" ]]
}

stage_attacks() {
  attack_condition FP16 origin
  attack_condition KIVI4_STD kivi_k4_v4_g32_r32
  attack_condition KIVI4_FQ kivi_k4_v4_g32_fq
  attack_condition KIVI2_STD kivi_k2_v2_g32_r32
  attack_condition KIVI2_FQ kivi_k2_v2_g32_fq
}

stage_utility() {
  python -m scripts.evaluate_generation_agreement \
    --dataset "$DATASET_PATH" \
    --cache-root "$CACHE_ROOT" \
    --model-path "$MODEL_PATH" \
    --device "$DEVICE" \
    --dtype "$DTYPE" \
    --max-new-tokens "$MAX_NEW_TOKENS" \
    --output "$RUN_DIR/generation_agreement.jsonl"
}

stage_aggregate() {
  local raw_args=(
    --raw "FP16=$RAW_DIR/FP16.jsonl"
    --raw "KIVI4-STD=$RAW_DIR/KIVI4_STD.jsonl"
    --raw "KIVI4-FQ=$RAW_DIR/KIVI4_FQ.jsonl"
    --raw "KIVI2-STD=$RAW_DIR/KIVI2_STD.jsonl"
    --raw "KIVI2-FQ=$RAW_DIR/KIVI2_FQ.jsonl"
  )
  if [[ "$RUN_COLLISION_PLUS" == "1" ]]; then
    raw_args=(
      --raw "FP16=$RAW_DIR/FP16.jsonl,$RAW_DIR/FP16_collision_plus.jsonl"
      --raw "KIVI4-STD=$RAW_DIR/KIVI4_STD.jsonl,$RAW_DIR/KIVI4_STD_collision_plus.jsonl"
      --raw "KIVI4-FQ=$RAW_DIR/KIVI4_FQ.jsonl,$RAW_DIR/KIVI4_FQ_collision_plus.jsonl"
      --raw "KIVI2-STD=$RAW_DIR/KIVI2_STD.jsonl,$RAW_DIR/KIVI2_STD_collision_plus.jsonl"
      --raw "KIVI2-FQ=$RAW_DIR/KIVI2_FQ.jsonl,$RAW_DIR/KIVI2_FQ_collision_plus.jsonl"
    )
  fi
  python -m scripts.aggregate_phase1 \
    --dataset "$DATASET_PATH" \
    "${raw_args[@]}" \
    --model "$MODEL_NAME" \
    --dataset-name "${SOURCE_DATASET:-natong19/lmsys-chat-1m-filtered}" \
    --seed "$SEED" \
    --output-dir "$AGG_DIR"
}

stage_plots() {
  python -m scripts.plot_phase1 \
    --normalized "$AGG_DIR/normalized_records.jsonl" \
    --validation "$RUN_DIR/cache_validation.json" \
    --output-dir "$RUN_DIR/plots"
}

stage_collision_plus() {
  if [[ "$RUN_COLLISION_PLUS" == "1" ]]; then
    local calibration_dataset="$RUN_DIR/collision_plus_calibration.jsonl"
    if [[ "$DRY_RUN" != "1" ]]; then
      python -c 'import json,sys; from src.config import BITTER_LESSON_TEXT; json.dump({"prompt": BITTER_LESSON_TEXT}, open(sys.argv[1], "w", encoding="utf-8")); open(sys.argv[1], "a", encoding="utf-8").write("\\n")' "$calibration_dataset"
    fi
    python inference/get_kvcache.py \
      --model-name "$MODEL_NAME" --model-path "$MODEL_PATH" \
      --dataset "$calibration_dataset" --dtype "$DTYPE" --device "$DEVICE" \
      --max-samples 1 --minimal-cache
    local calibration_root="$REPO_ROOT/cache/$DTYPE/$(basename "${calibration_dataset%.jsonl}")/$MODEL_NAME"
    python defense/baseline/kivi_kvcache.py --model-name "$MODEL_NAME" --dataset-path "$calibration_dataset" --cache-root "$calibration_root" --dtype "$DTYPE" --device "$DEVICE" --k-bits 4 --v-bits 4 --group-size 32 --residual-length 32 --mode standard
    python defense/baseline/kivi_kvcache.py --model-name "$MODEL_NAME" --dataset-path "$calibration_dataset" --cache-root "$calibration_root" --dtype "$DTYPE" --device "$DEVICE" --k-bits 4 --v-bits 4 --group-size 32 --residual-length 32 --mode full_prompt_quantized
    python defense/baseline/kivi_kvcache.py --model-name "$MODEL_NAME" --dataset-path "$calibration_dataset" --cache-root "$calibration_root" --dtype "$DTYPE" --device "$DEVICE" --k-bits 2 --v-bits 2 --group-size 32 --residual-length 32 --mode standard
    python defense/baseline/kivi_kvcache.py --model-name "$MODEL_NAME" --dataset-path "$calibration_dataset" --cache-root "$calibration_root" --dtype "$DTYPE" --device "$DEVICE" --k-bits 2 --v-bits 2 --group-size 32 --residual-length 32 --mode full_prompt_quantized

    local calibration_hash
    calibration_hash="$(python -c 'import hashlib,sys; from src.config import BITTER_LESSON_TEXT; print(hashlib.sha1(BITTER_LESSON_TEXT.encode("utf-8")).hexdigest())')"
    local labels=(FP16 KIVI4_STD KIVI4_FQ KIVI2_STD KIVI2_FQ)
    local protects=(origin kivi_k4_v4_g32_r32 kivi_k4_v4_g32_fq kivi_k2_v2_g32_r32 kivi_k2_v2_g32_fq)
    local index
    for index in "${!labels[@]}"; do
      local label="${labels[$index]}"
      local protect="${protects[$index]}"
      local target="$calibration_root/$calibration_hash/$protect/past_key_values.pt"
      local config_path="$REPO_ROOT/attack/config/$protect/$DTYPE/$MODEL_NAME.json"
      if [[ -f "$config_path" ]]; then
        cp "$config_path" "$RUN_DIR/previous_collision_config_${label}.json"
      fi
      python attack/get_collision_threshold.py \
        --model_path "$MODEL_PATH" --target_data_path "$target" \
        --input_text "$(python -c 'from src.config import BITTER_LESSON_TEXT; print(BITTER_LESSON_TEXT)')" \
        --protect_type "$protect" --target_model_name "$MODEL_NAME" \
        --dtype "$DTYPE" --device "$DEVICE" --batch_size 512
      local output="$RAW_DIR/${label}_collision_plus.jsonl"
      if [[ "$DRY_RUN" != "1" && -f "$output" ]]; then
        mv "$output" "$output.partial.$(date -u +%Y%m%dT%H%M%SZ)"
      fi
      python attack/attacks.py \
        --target-model-name "$MODEL_NAME" --base-model-name "$MODEL_NAME" \
        --base-model-path "$MODEL_PATH" --eval-model-path "$EMBEDDING_PATH" \
        --dataset-path "$DATASET_PATH" --cache-root "$CACHE_ROOT" \
        --protect-type "$protect" --dtype "$DTYPE" --device "$DEVICE" \
        --run-collision --no-run-inversion --no-run-injection \
        --enhance --output "$output" --i-understand-risks
    done
    echo "collision_plus=complete" > "$RUN_DIR/collision_plus.status"
  else
    echo "collision_plus=not_requested; protocol=Protocol_A_only" > "$RUN_DIR/collision_plus.status"
  fi
}

run_stage preflight stage_preflight
run_stage prepare_dataset stage_prepare_dataset
run_stage prefill stage_prefill
run_stage materialize stage_materialize
run_stage validate_cache stage_validate_cache
run_stage attacks stage_attacks
run_stage utility stage_utility
run_stage collision_plus stage_collision_plus
run_stage aggregate stage_aggregate
run_stage plots stage_plots

if [[ "$DRY_RUN" != "1" ]]; then
  {
    echo "status=complete"
    echo "protocol=quantization_mismatch (Protocol A)"
    echo "dataset_manifest=$MANIFEST_PATH"
    echo "cache_validation=$RUN_DIR/cache_validation.json"
    echo "raw_attacks=$RAW_DIR"
    echo "generation_agreement=$RUN_DIR/generation_agreement.jsonl"
    echo "prompt_ppl=$RUN_DIR/prompt_ppl.jsonl"
    echo "aggregate=$AGG_DIR"
    echo "plots=$RUN_DIR/plots"
    echo "collision_plus=$RUN_DIR/collision_plus.status"
  } > "$RUN_DIR/FINAL_STATUS.txt"
fi
echo "Phase 1 main pipeline completed. Artifacts: $RUN_DIR"
