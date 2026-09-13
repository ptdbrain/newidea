# Collision+ Exact Streaming Calibration Design

## Goal

Make Collision+ threshold calibration complete on Llama-3.2-1B without CUDA
out-of-memory failures while preserving an exact full-vocabulary search and
producing the configuration schema consumed by `attack/collision.py`.

## Failure and correctness diagnosis

The current implementation compares each candidate's complete prefix cache
against one target position:

```python
current_pkv.key_cache[layer_idx] - target_k
```

At sequence position `s`, that broadcasts the target across positions
`0..s`. The resulting distance retains candidate, KV-head, and prefix axes.
Those tensors are repeatedly concatenated and retained on the GPU for as many
as 100 sequence positions. GPU memory therefore grows with sequence length
until the 95 GiB device is exhausted.

The generated shape is also incompatible with analysis. Collision consumes
one K distance and one V distance per vocabulary candidate, but the current
calibration retains head and prefix dimensions. Reducing only the batch size
would delay the OOM without correcting this mismatch.

## Exact calibration contract

For sequence position `s`, layer `l`, and candidate token `c`, calibration
must compute:

```text
dK[l,s,c] = ||Kcandidate[l,c,-1,:,:] - Ktarget[l,s,:,:]||2
dV[l,s,c] = ||Vcandidate[l,c,-1,:,:] - Vtarget[l,s,:,:]||2
```

The norm covers all KV heads and the head dimension. The candidate cache uses
its final position because the model receives exactly one candidate after the
known prefix. The target cache uses position `s`.

Every token ID from `0` through `vocab_size - 1` is evaluated exactly once at
every calibration position. Candidate batches stay in ascending token-ID
order, so concatenated vectors remain directly indexed by token ID.

For the true token ID at position `s`:

- its K and V distances update the target distributions;
- all other token distances update the non-target distributions.

For each model layer, the final configuration contains two values per field,
ordered `[K, V]`:

- `target_mean`, `target_std`, `target_max`;
- `others_mean`, `others_std`, `others_min`.

This is the existing schema expected by `get_collision_threshold()`.

## Bounded-memory data flow

### GPU

The GPU retains only:

- the model and target calibration cache;
- the known-prefix cache;
- one candidate batch and its model outputs;
- one batch of K/V distance vectors.

The default candidate batch decreases from 512 to 128. The operator may set
`COLLISION_PLUS_BATCH_SIZE` to another positive integer. Batch size changes
execution granularity only; it does not change which vocabulary candidates
are evaluated.

Candidate model calls use `torch.inference_mode()`, `use_cache=True`, and do
not request hidden states for KV calibration. Each batch compares only the
newly appended cache position. Distance vectors are detached, converted to
float32, and transferred to CPU before the next batch.

### CPU

CPU chunks are retained for only one sequence position. With vocabulary size
128,256, 16 layers, and separate K/V vectors, this is approximately 16 MiB of
float32 distance data. At the end of the position, chunks are concatenated per
layer/component, the true token is separated, and all long vectors are
released.

Cross-position distributions use float64 mergeable Welford accumulators. Each
accumulator stores only count, mean, second central moment, minimum, and
maximum. Sample standard deviation matches `torch.std()`'s default unbiased
semantics; a distribution with fewer than two values reports zero standard
deviation.

No distance tensor from an earlier sequence position remains on the GPU.
`torch.cuda.empty_cache()` is not used as a substitute for releasing live
references.

## Checkpoint and resume

Calibration writes a versioned checkpoint after every completed sequence
position:

```text
<target-cache-directory>/past_key_values_dist/streaming_stats_v2.pt
```

The checkpoint contains only CPU values:

- format version;
- next sequence position;
- serialized target/non-target running statistics for every layer and K/V;
- stable target-cache tensor-content SHA-256;
- calibration-input SHA-1;
- model path/name, vocabulary size, layer count, dtype, and target type;
- configured and effective candidate batch size.

Writes use a temporary sibling followed by `os.replace`, so interruption
cannot expose a partially written checkpoint.

On resume, semantic metadata must match the current invocation exactly. A
mismatch fails with a message naming the differing field; it is never silently
reused. Candidate batch size is execution granularity rather than semantic
identity, so it is recorded for provenance but may be reduced or increased
when resuming without invalidating completed positions. Legacy `seq=*.pt`
files from the broken implementation are ignored. They are not deleted
automatically.

The target digest is computed from each layer's K/V label, dtype, shape, and
contiguous CPU tensor bytes. It does not hash the outer `torch.save` container,
whose serialization metadata is not part of the scientific cache identity.

The GPU prefix cache is not checkpointed. When resuming at position `s`, the
script rebuilds the known prefix with one inference over tokens `0..s-1`, then
continues candidate evaluation at `s`. A checkpoint whose `next_seq_id` equals
the calibration sequence length can regenerate the final JSON config without
performing model candidate batches again.

## Output and validation

Before writing the Collision+ configuration, calibration verifies:

- every layer has exactly `sequence_length` target observations for K and V;
- every layer has exactly `sequence_length * (vocab_size - 1)` non-target
  observations for K and V;
- every emitted statistic is finite;
- target and non-target arrays each have exactly two elements in `[K, V]`
  order.

The final JSON is written atomically to the existing location:

```text
attack/config/<protect_type>/<dtype>/<target_model_name>.json
```

The schema remains compatible with `attack/collision.py`. Existing inversion,
standard collision, injection, KIVI materialization, utility evaluation, and
aggregation are unchanged.

## Launcher behavior

`scripts/run_phase1_main.sh` defines:

```bash
COLLISION_PLUS_BATCH_SIZE="${COLLISION_PLUS_BATCH_SIZE:-128}"
```

It validates that the value is a positive integer, records it in
`environment.txt`, and passes it to all five Collision+ calibration commands.
`RUN_COLLISION_PLUS=0` continues to skip calibration unchanged.

If the PBS job ends during calibration, submitting the same `RUN_DIR` with
`RESUME=1` re-enters the unfinished `collision_plus` stage. The per-condition
streaming checkpoint then resumes at its next incomplete sequence position.

## Error handling

- Invalid batch sizes fail before pipeline work begins.
- Shape mismatches fail with layer, sequence position, expected shape, and
  actual shape.
- Non-finite distances fail before updating statistics.
- CUDA OOM reports the current sequence position, candidate range, and batch
  size. The completed-position checkpoint remains valid, allowing a retry with
  a smaller `COLLISION_PLUS_BATCH_SIZE`.
- Existing output configuration is replaced only after complete validation.

The implementation does not silently sample vocabulary, skip candidates,
change dtype, fall back to CPU, or treat partial calibration as success.

## Testing

CPU tests use small deterministic tensors and a lightweight fake cache/model
boundary to verify:

1. Only the current candidate position participates in K/V distance.
2. Each candidate produces exactly one K and one V scalar per layer.
3. Ascending batches cover every vocabulary token exactly once.
4. Target-token exclusion produces exact target and non-target populations.
5. Mergeable Welford results match direct float64 mean, unbiased standard
   deviation, minimum, and maximum.
6. Streaming results are invariant to candidate batch partitioning.
7. Checkpoint roundtrip preserves statistics and resumes at the next position.
8. Metadata mismatch and incomplete counts fail closed.
9. The launcher defaults to 128 and forwards an explicit
   `COLLISION_PLUS_BATCH_SIZE`.

The focused CPU suite, Python compilation, Bash syntax, and diff checks must
pass locally. A Linux/CUDA server run must still validate real peak memory,
Triton/PyTorch interaction, checkpoint resume, and final Collision+ output.

## Expected resource change

The broken implementation retains data proportional to sequence length,
vocabulary size, layers, KV heads, and accumulated prefix length. The new GPU
distance state is proportional only to candidate batch size, layers, and KV
dimensions. CPU distance state is bounded to one sequence position, while
cross-position state is constant per layer.

Runtime remains proportional to full vocabulary size and calibration sequence
length. Batch 128 may be slower than 512, but it preserves exact coverage and
substantially lowers peak forward-pass memory.

## Non-goals

- Vocabulary sampling or approximate thresholds.
- Changing the Collision+ classifier or threshold optimization formula.
- Changing KIVI quantization or cache formats.
- Claiming a real GPU memory bound without the server-side run.
- Deleting legacy partial statistics or prior experiment artifacts.
