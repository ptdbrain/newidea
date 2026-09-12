# CUDA 12.8/13.0 Phase 1 Runtime Design

## Goal

Make the Phase 1 launcher install and validate a Python 3.10 PyTorch runtime
for CUDA 12.8 or CUDA 13.0 while preserving the existing offline, Git-free
compute-node path and the fail-fast KIVI cache guarantees.

## Current problem

The root `requirements.txt` pins `torch==2.6.0`. Official PyTorch binaries for
2.6.0 stop at CUDA 12.6, so that pin cannot install a `cu128` or `cu130`
runtime. The project also verifies only `torch.cuda.is_available()` before
starting the expensive pipeline; that does not prove that the selected Torch
build or KIVI's Triton kernel matches the job node.

## Supported runtime matrix

- Python: 3.10 on the Linux compute node.
- PyTorch: exactly `2.9.1`.
- CUDA variants: `cu128` and `cu130` only.
- Default variant: `cu128`.
- Override: `PYTORCH_CUDA_VARIANT=cu130`.
- Package indexes:
  - `https://download.pytorch.org/whl/cu128`
  - `https://download.pytorch.org/whl/cu130`

PyTorch 2.9.1 is the first patched release line in the official version table
that provides both variants and Python 3.10 Linux wheels. CUDA 13.x requires an
NVIDIA driver from the 580 series or newer. The launcher must use an actual
KIVI kernel smoke test instead of inferring PTX compatibility only from the
driver's advertised CUDA version.

## Dependency boundary

`requirements.txt` contains the application dependencies but must not select a
Torch build. The launcher installs `torch==2.9.1` first from the selected
official CUDA index, then installs `requirements.txt`. Installing Torch first
prevents transitive dependencies such as Sentence Transformers from selecting
an arbitrary CPU or PyPI build.

The dependency stamp must include:

- the contents of `requirements.txt`;
- `torch==2.9.1`;
- the selected CUDA variant.

Changing `cu128` to `cu130`, or changing the Torch pin, therefore forces the
launcher to reconcile the virtual environment instead of reusing a stale
stamp.

## Bootstrap modes

### Normal bootstrap

`scripts/run_phase1_main.sh` creates or reuses `.venv`, installs the selected
PyTorch build from the official index, installs the application requirements,
then runs the existing model and dataset bootstrap.

### Pre-staged offline job

`scripts/run_phase1_offline.sh` remains network-free and sets
`SKIP_BOOTSTRAP=1`. It uses the active Conda environment, or the existing
project `.venv`, and relies on preflight to prove that the pre-staged runtime is
valid. CUDA wheels are not committed to Git because they are multi-gigabyte,
platform-specific artifacts. A site-provided environment or separately
pre-staged wheelhouse remains the deployment boundary for an offline cluster.

## Validation contract

Before any dataset, model-prefill, or cache stage runs on a CUDA device,
preflight must:

1. import Torch and report its version, compiled CUDA runtime, CUDA
   availability, GPU name, and compute capability;
2. require exactly Torch 2.9.1;
3. require the compiled CUDA runtime to equal the selected variant (`12.8` for
   `cu128`, `13.0` for `cu130`);
4. reject unknown `PYTORCH_CUDA_VARIANT` values before creating an environment;
5. import the pinned KIVI adapter; and
6. execute a small public KIVI quantize/dequantize operation on the selected
   GPU, verifying the reconstructed shape and finite values.

Any mismatch exits non-zero with an actionable error before Phase 1 writes a
stage marker. The launcher must not silently fall back to CPU, a different CUDA
variant, or a non-KIVI implementation.

## Testing

CPU/Windows tests execute shell helpers with controlled fake Python binaries so
they can prove:

- `cu128` is the default and maps to the official `cu128` index;
- `cu130` maps to the official `cu130` index;
- unsupported variants fail before pip is invoked;
- the dependency stamp changes with the CUDA variant;
- preflight rejects mismatched Torch/CUDA metadata; and
- the launcher invokes the KIVI GPU smoke program before pipeline stages.

The existing CUDA test remains the real Linux/GPU compatibility gate. Local
Windows tests can verify control flow and error handling, but cannot establish
that Triton compiles on the server GPU.

## Non-goals

- Committing PyTorch/CUDA wheels, model checkpoints, embeddings, or datasets to
  Git.
- Supporting CUDA versions older than 12.8.
- Upgrading the pinned KIVI source revision.
- Replacing KIVI's Triton primitive with a different quantization backend.
- Claiming a real GPU pass without running the server-side kernel smoke test.

## References

- PyTorch previous versions: <https://pytorch.org/get-started/previous-versions/>
- PyTorch CUDA 12.8 wheel index: <https://download.pytorch.org/whl/cu128/torch/>
- PyTorch CUDA 13.0 wheel index: <https://download.pytorch.org/whl/cu130/torch/>
- NVIDIA CUDA compatibility: <https://docs.nvidia.com/deploy/cuda-compatibility/minor-version-compatibility.html>
