# Collision+ Exact Streaming Calibration Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the broken GPU-growing Collision+ calibration with exact full-vocabulary, current-position distance collection, bounded CPU streaming statistics, and resumable checkpoints.

**Architecture:** `attack/collision_threshold_stats.py` owns device-independent running statistics, stable cache identity, checkpoint serialization, count validation, and final config construction. `attack/get_collision_threshold.py` owns model/cache traversal and transfers only one candidate batch of scalar distances to CPU. The Phase 1 launcher exposes a validated `COLLISION_PLUS_BATCH_SIZE` and keeps the existing five-condition protocol.

**Tech Stack:** Python 3.10, PyTorch 2.9.1, Transformers DynamicCache, Bash, pytest.

## Global Constraints

- Evaluate every vocabulary token exactly once per calibration position; no sampling.
- Compare only the newly appended candidate cache position with the matching target position.
- Produce separate scalar K and V distances by reducing over KV heads and head dimension.
- Keep cross-position statistics on CPU using float64 mergeable Welford accumulators.
- Keep at most one candidate batch of distance tensors on GPU.
- Default `COLLISION_PLUS_BATCH_SIZE` to 128 and accept only positive integers.
- Checkpoint after every completed sequence position and reject mismatched checkpoint metadata.
- Preserve the Collision+ JSON schema consumed by `attack/collision.py`.
- Do not alter inversion, standard collision, injection, KIVI quantization, utility evaluation, or previous artifacts.
- Do not silently sample, skip candidates, change dtype, fall back to CPU, or mark partial calibration complete.

---

### Task 1: Exact Streaming Statistics Core

**Files:**
- Create: `attack/collision_threshold_stats.py`
- Create: `tests/test_collision_threshold_streaming.py`

**Interfaces:**
- Consumes: one CPU vector of K distances and one CPU vector of V distances per layer/position, plus the true token ID.
- Produces: `RunningStats`, `new_layer_statistics(num_layers)`, `update_position_statistics(...)`, and `finalize_statistics(...)`.

- [ ] **Step 1: Write failing tests for mergeable statistics**

Create literal, hand-computed fixtures:

```python
def test_running_stats_matches_direct_float64_statistics():
    stats = RunningStats()
    stats.update(torch.tensor([1.0, 2.0]))
    stats.update(torch.tensor([3.0, 4.0]))

    assert stats.count == 4
    assert stats.mean == pytest.approx(2.5)
    assert stats.std == pytest.approx(1.2909944487358056)
    assert stats.minimum == 1.0
    assert stats.maximum == 4.0


def test_running_stats_empty_and_singleton_have_zero_std():
    empty = RunningStats()
    singleton = RunningStats()
    singleton.update(torch.tensor([7.0]))

    assert empty.std == 0.0
    assert singleton.std == 0.0
```

- [ ] **Step 2: Run Task 1 tests and verify RED**

Run:

```bash
python -m pytest tests/test_collision_threshold_streaming.py -q
```

Expected: collection fails because `attack.collision_threshold_stats` does not exist.

- [ ] **Step 3: Implement `RunningStats` with chunked Welford merge**

Use this public state:

```python
@dataclass
class RunningStats:
    count: int = 0
    mean_value: float = 0.0
    m2: float = 0.0
    minimum: float = math.inf
    maximum: float = -math.inf

    def update(self, values: torch.Tensor) -> None: ...
    @property
    def mean(self) -> float: ...
    @property
    def std(self) -> float: ...
    def to_dict(self) -> dict[str, float | int]: ...
    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "RunningStats": ...
```

`update` must detach, flatten, move to CPU float64, reject non-finite values,
compute the chunk mean/M2/min/max, and merge the chunk with the accumulated
state. `std` uses `sqrt(m2 / (count - 1))` for `count >= 2`.

- [ ] **Step 4: Write failing tests for target/others separation and schema**

Use two positions and vocabulary size three:

```python
def test_position_updates_exclude_exact_true_token():
    state = new_layer_statistics(1)
    update_position_statistics(
        state,
        [(torch.tensor([0.0, 1.0, 2.0]), torch.tensor([10.0, 11.0, 12.0]))],
        true_token_id=0,
    )
    update_position_statistics(
        state,
        [(torch.tensor([3.0, 4.0, 5.0]), torch.tensor([13.0, 14.0, 15.0]))],
        true_token_id=2,
    )

    config = finalize_statistics(state, sequence_length=2, vocab_size=3)
    assert config[0]["target_mean"] == pytest.approx([2.5, 12.5])
    assert config[0]["target_max"] == pytest.approx([5.0, 15.0])
    assert config[0]["others_mean"] == pytest.approx([2.5, 12.5])
    assert config[0]["others_min"] == pytest.approx([1.0, 11.0])
```

Add tests proving that a wrong vector length, out-of-range true token, wrong
layer count, non-finite value, or incomplete final observation count raises an
actionable exception.

- [ ] **Step 5: Implement layer state and final schema**

Represent each layer as:

```python
{
    "target": [RunningStats(), RunningStats()],
    "others": [RunningStats(), RunningStats()],
}
```

`update_position_statistics` requires all vectors to be one-dimensional, on
CPU, and equal length across layers/components. It updates the target with the
single true-token value and updates others with the slices before and after
that index, avoiding a copied full non-target vector.

`finalize_statistics` requires target count `sequence_length` and others count
`sequence_length * (vocab_size - 1)` for both K and V in every layer. It emits
the existing list-of-layer-dicts schema with two-element `[K, V]` arrays and
rejects non-finite output.

- [ ] **Step 6: Run Task 1 tests and verify GREEN**

Run the Task 1 command and require all tests to pass.

- [ ] **Step 7: Commit Task 1**

```bash
git add attack/collision_threshold_stats.py tests/test_collision_threshold_streaming.py
git commit -m "feat(collision): add streaming distance stats"
```

### Task 2: Correct Current-Position Full-Vocabulary Collection

**Files:**
- Modify: `attack/get_collision_threshold.py`
- Modify: `tests/test_collision_threshold_streaming.py`

**Interfaces:**
- Consumes: model, target legacy/Dynamic KV cache, known-prefix cache, sequence position, vocabulary size, and positive candidate batch size.
- Produces: `_current_position_distances(...) -> list[tuple[Tensor, Tensor]]` and `collect_position_distances(...) -> list[tuple[Tensor, Tensor]]`, with returned vectors on CPU in token-ID order.

- [ ] **Step 1: Write failing tests for current-position distance**

Construct one fake layer where old prefix values are deliberately far from the
target but the final position is known:

```python
def test_current_position_distance_ignores_old_prefix():
    current = FakeCache(
        key_cache=[torch.tensor([[[[999.0], [3.0]]], [[[999.0], [5.0]]]])],
        value_cache=[torch.tensor([[[[999.0], [7.0]]], [[[999.0], [11.0]]]])],
    )
    target = ((torch.tensor([[[[3.0]]]]), torch.tensor([[[[7.0]]]])),)

    distances = _current_position_distances(current, target, seq_id=0)

    assert distances[0][0].tolist() == pytest.approx([0.0, 2.0])
    assert distances[0][1].tolist() == pytest.approx([0.0, 4.0])
```

Add shape-error tests for target sequence bounds, layer-count mismatch, head
mismatch, and candidate K/V mismatch.

- [ ] **Step 2: Run the focused test and verify RED**

```bash
python -m pytest tests/test_collision_threshold_streaming.py -q
```

Expected: import or assertion fails because current-position collection is not
implemented.

- [ ] **Step 3: Implement exact scalar distance extraction**

Normalize cache access across DynamicCache and legacy tuples. For each layer:

```python
candidate_k = current_keys[layer_idx][:, :, -1, :]
candidate_v = current_values[layer_idx][:, :, -1, :]
target_k = target_datas[layer_idx][0][:, :, seq_id, :]
target_v = target_datas[layer_idx][1][:, :, seq_id, :]
k_dist = torch.linalg.vector_norm(candidate_k - target_k, dim=(1, 2))
v_dist = torch.linalg.vector_norm(candidate_v - target_v, dim=(1, 2))
```

The result for each component must have shape `[candidate_batch]`.

- [ ] **Step 4: Write failing full-vocabulary and batch-invariance tests**

Use a lightweight CPU fake model with vocabulary size five. It returns a fake
cache whose final K value equals the candidate token ID and whose final V value
equals twice that ID. Run with batch sizes two and three:

```python
expected_k = [2.0, 1.0, 0.0, 1.0, 2.0]
expected_v = [4.0, 2.0, 0.0, 2.0, 4.0]
assert collect_position_distances(..., batch_size=2)[0][0].tolist() == expected_k
assert collect_position_distances(..., batch_size=3)[0][0].tolist() == expected_k
```

Assert each returned tensor is one-dimensional, CPU-resident, finite, and has
exactly `vocab_size` elements. Assert candidate batches cover IDs
`[0, 1, 2, 3, 4]` exactly once.

- [ ] **Step 5: Implement bounded candidate traversal**

Create candidate IDs directly per contiguous range instead of materializing a
GPU vocabulary tensor or rebuilding a Python list each position. Rebuild an
expanded DynamicCache for each batch from the immutable known-prefix cache.
Call the model under `torch.inference_mode()` with `use_cache=True` and
`output_hidden_states=False`.

Append only `distance.detach().to(device="cpu", dtype=torch.float32)` to CPU
chunk lists. Delete model outputs and expanded-cache references before the next
batch. Concatenate CPU chunks once per layer/component after all candidate IDs
have been evaluated.

Catch `torch.OutOfMemoryError` and raise a contextual error containing sequence
position, candidate start/end, and batch size; do not skip or retry candidates
silently.

- [ ] **Step 6: Run Task 2 tests and verify GREEN**

Run the focused test file and require all tests to pass.

- [ ] **Step 7: Commit Task 2**

```bash
git add attack/get_collision_threshold.py tests/test_collision_threshold_streaming.py
git commit -m "fix(collision): bound candidate distance memory"
```

### Task 3: Stable Checkpointing and Config Generation

**Files:**
- Modify: `attack/collision_threshold_stats.py`
- Modify: `attack/get_collision_threshold.py`
- Modify: `tests/test_collision_threshold_streaming.py`

**Interfaces:**
- Consumes: target cache, calibration identity, layer statistics, and next sequence position.
- Produces: `stable_cache_digest(...)`, `save_checkpoint(...)`, `load_checkpoint(...)`, `rebuild_prefix_cache(...)`, and an atomically written final Collision+ JSON config.

- [ ] **Step 1: Write failing stable-digest and checkpoint tests**

Build two equivalent caches with distinct tensor objects and require equal
digests. Change one value, shape, or dtype and require a different digest.

Roundtrip a checkpoint and assert:

```python
next_seq_id, restored = load_checkpoint(path, expected_metadata)
assert next_seq_id == 2
assert restored[0]["target"][0].count == 2
assert restored[0]["target"][0].mean == pytest.approx(2.5)
```

Add tests proving semantic metadata mismatch names the differing field, a
truncated checkpoint fails, and the temporary sibling is replaced atomically.
Changing only candidate batch size must preserve the checkpoint and update the
recorded execution setting.

- [ ] **Step 2: Run tests and verify RED**

Run the focused test file. Expected: missing checkpoint/digest APIs fail.

- [ ] **Step 3: Implement stable identity and versioned checkpoint**

`stable_cache_digest` hashes, in layer order:

```text
layer index | component label K/V | dtype | shape | contiguous CPU bytes
```

Use format string `collision-plus-streaming-v2`. Serialize only primitives and
CPU accumulator state. Save to a sibling ending `.tmp`, then call
`os.replace(temp_path, checkpoint_path)`.

`load_checkpoint` uses safe Torch loading, validates format/metadata/layer
shape, and reconstructs `RunningStats` through `from_dict`.

- [ ] **Step 4: Write failing resume/config tests**

Use a fake model and a checkpoint at `next_seq_id=2`. Assert prefix rebuild
feeds exactly `input_ids[:, :2]` and candidate traversal starts at position 2.
Assert a completed checkpoint (`next_seq_id == sequence_length`) performs no
candidate model calls and still writes the existing config schema.

- [ ] **Step 5: Replace legacy collection/analysis in the CLI**

In `main()`:

1. load target cache on CPU and compute the stable content digest;
2. build semantic metadata containing target digest, calibration-input SHA-1,
   model path/name, vocab size, layer count, dtype, and target type; record
   batch size separately as execution provenance;
3. move the target cache to the selected device;
4. load `streaming_stats_v2.pt` when present and matching;
5. rebuild the true-prefix cache up to `next_seq_id`;
6. for each remaining position, collect exact full-vocabulary distances,
   update statistics, advance the true-prefix cache, and atomically checkpoint;
7. validate exact counts and atomically write the final JSON config.

Remove the main-path dependency on legacy `seq=*.pt`, `load_target_dists`, and
`analyze_distances`. Leave legacy files untouched and print a warning that they
are ignored.

- [ ] **Step 6: Run Task 3 tests and verify GREEN**

Run the focused test file and require all tests to pass.

- [ ] **Step 7: Commit Task 3**

```bash
git add attack/collision_threshold_stats.py attack/get_collision_threshold.py tests/test_collision_threshold_streaming.py
git commit -m "feat(collision): resume exact calibration"
```

### Task 4: Phase 1 Launcher Integration

**Files:**
- Modify: `scripts/run_phase1_main.sh`
- Modify: `tests/test_phase1_cuda_runtime.py`
- Modify: `docs/phase1_main_experiment.md`
- Modify: `docs/kivi_integration.md`

**Interfaces:**
- Consumes: `COLLISION_PLUS_BATCH_SIZE`, default `128`.
- Produces: validated launcher configuration forwarded to all five threshold-calibration commands and recorded in `environment.txt`.

- [ ] **Step 1: Write failing real-launcher tests**

Run `DRY_RUN=1 --bootstrap-only` and assert output reports:

```text
Collision+ candidate batch size: 128
```

Run with `COLLISION_PLUS_BATCH_SIZE=64` and assert `64`. Run with `0`, `-1`,
and `abc`, asserting exit code 2 and:

```text
COLLISION_PLUS_BATCH_SIZE must be a positive integer
```

Use the existing Python recorder fixture to run into the Collision+ stage and
assert every `attack/get_collision_threshold.py` invocation includes the exact
configured `--batch_size` value.

- [ ] **Step 2: Run launcher tests and verify RED**

```bash
python -m pytest tests/test_phase1_cuda_runtime.py -q
```

Expected: the new output, rejection, and forwarding assertions fail.

- [ ] **Step 3: Implement launcher configuration**

Normalize and validate near other environment variables:

```bash
COLLISION_PLUS_BATCH_SIZE="${COLLISION_PLUS_BATCH_SIZE:-128}"
[[ "$COLLISION_PLUS_BATCH_SIZE" =~ ^[1-9][0-9]*$ ]] || {
  echo "COLLISION_PLUS_BATCH_SIZE must be a positive integer; got $COLLISION_PLUS_BATCH_SIZE" >&2
  exit 2
}
```

Print the value during dry-run, record it in `environment.txt`, and replace the
hard-coded `--batch_size 512` with
`--batch_size "$COLLISION_PLUS_BATCH_SIZE"`.

- [ ] **Step 4: Update operator documentation**

Document the default, the lower-memory override, per-position checkpoint path,
and exact resume command:

```bash
COLLISION_PLUS_BATCH_SIZE=128 \
RUN_COLLISION_PLUS=1 RESUME=1 RUN_DIR=/absolute/run/path \
bash scripts/run_phase1_offline.sh
```

State that old `seq=*.pt` files are ignored and not deleted.

- [ ] **Step 5: Run Task 4 tests and syntax checks**

```bash
python -m pytest tests/test_phase1_cuda_runtime.py -q
bash -n scripts/run_phase1_main.sh
git diff --check
```

Require all commands to exit zero.

- [ ] **Step 6: Commit Task 4**

```bash
git add scripts/run_phase1_main.sh tests/test_phase1_cuda_runtime.py docs/phase1_main_experiment.md docs/kivi_integration.md
git commit -m "feat(runner): configure Collision+ batch size"
```

### Task 5: Regression, Review, and Server Handoff

**Files:**
- Modify only if review finds a verified defect in the files changed by Tasks 1-4.

**Interfaces:**
- Consumes: completed implementation and repository test suite.
- Produces: review findings, fresh verification evidence, remote feature-branch SHA, and a server resume command.

- [ ] **Step 1: Run focused regression verification**

```bash
python -m pytest \
  tests/test_collision_threshold_streaming.py \
  tests/test_phase1_cuda_runtime.py \
  tests/test_phase1_kivi_runtime.py \
  tests/test_kvcache_dataset_input.py \
  tests/test_prefill_cache.py \
  tests/test_kivi_strict_source.py \
  -q --basetemp=.test-tmp/collision-plus-focused
python -m compileall -q attack scripts src inference defense tests
bash -n scripts/run_phase1_main.sh
git diff --check
```

Expected: all focused tests pass and all static commands exit zero.

- [ ] **Step 2: Run the broad suite and record environment limits**

```bash
python -m pytest -q --basetemp=.test-tmp/collision-plus-full
```

Do not convert missing Windows Triton or `dp_accounting` collection errors into
success. Do not claim real CUDA memory behavior from CPU tests.

- [ ] **Step 3: Request an independent code review**

Review the implementation range against:

```text
docs/superpowers/specs/2026-09-13-collision-plus-streaming-calibration-design.md
docs/superpowers/plans/2026-09-13-collision-plus-streaming-calibration.md
```

Fix all Critical and Important findings through a fresh failing test before
proceeding.

- [ ] **Step 4: Push the feature branch without force**

Fetch the exact remote ref, verify it is an ancestor of local HEAD, then push
the completed feature branch. Verify the remote SHA independently with
`git ls-remote`.

- [ ] **Step 5: Run the server resume validation**

After staging the pushed code on the server, resume the failed run:

```bash
qsub -V \
  -v PYTORCH_CUDA_VARIANT=cu128,COLLISION_PLUS_BATCH_SIZE=128,RUN_COLLISION_PLUS=1,RESUME=1,RUN_DIR=/shared/homes/u26466553/projects/leakage-kv-cache-detection/logs/phase1_main_20260912T122445Z \
  run_phase1_main.pbs
```

Success requires Collision+ to pass sequence position 37, GPU allocated memory
to remain bounded rather than grow with position, all five calibrated config
files to be written, and the pipeline to produce `FINAL_STATUS.txt`. This
server-side result remains pending until the submitted job completes.
