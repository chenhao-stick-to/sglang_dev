# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

SGLang is a high-performance serving framework for large language models and multimodal models. It supports 190+ model architectures across NVIDIA, AMD, Intel, Google TPU, and Ascend NPU hardware.

The repository is a monorepo with multiple packages:
- `python/` -- Main `sglang` Python package (the serving runtime)
- `sgl-kernel/` -- Separate `sglang-kernel` PyPI package (pre-compiled CUDA/C++ kernels, CMake-based)
- `sgl-model-gateway/` -- Rust-based model gateway service
- `rust/sglang-grpc/` -- Rust gRPC server extension linked into the Python wheel via PyO3

## Build and Install

```bash
# Development install (main package)
pip install -e "python[dev]" --index-strategy unsafe-best-match --prerelease allow

# With diffusion support
pip install -e "python[diffusion]"

# sgl-kernel development install
cd sgl-kernel && make install   # pip install -e . --no-build-isolation

# sgl-kernel build wheel
cd sgl-kernel && make build     # uv build --wheel
```

## Linting and Formatting

Pre-commit handles all linting. The configured tools are:
- **black** (v26.1.0): Python formatting
- **isort** (v7.0.0): Import sorting (profile=black, known_first_party=sglang)
- **ruff** (v0.15.1): Only checks F401 (unused imports) and F821 (undefined names)
- **clang-format** (v20.1.7): C++/CUDA formatting
- **codespell**: Spell checking

```bash
# Run all pre-commit checks
pre-commit run --all-files

# Run specific hook
pre-commit run black --all-files
pre-commit run isort --all-files
pre-commit run ruff --all-files

# sgl-kernel formatting
cd sgl-kernel && make format
```

## Testing

Tests use both unittest and pytest. The CI runner executes `python filename.py -f` (failfast).

```bash
# Single test file
python3 test/registered/core/test_srt_endpoint.py

# Single test method
python3 test/registered/core/test_srt_endpoint.py TestSRTEndpoint.test_simple_decode

# JIT kernel test
python3 python/sglang/jit_kernel/tests/test_add_constant.py

# Run a CI suite
python3 test/run_suite.py --hw cuda --suite base-b-test-1-gpu-small

# With auto-partitioning (split suite across N parallel jobs)
python3 test/run_suite.py --hw cuda --suite base-b-test-1-gpu-small \
    --auto-partition-id 0 --auto-partition-size 4

# sgl-kernel tests
cd sgl-kernel && make test
```

### Test Registration

Every CI test file under `test/registered/` must call a registration function at module level:

```python
# New style (preferred for CUDA) — stage + runner_config auto-generates suite name
from sglang.test.ci.ci_register import register_cuda_ci
register_cuda_ci(est_time=80, stage="base-b", runner_config="1-gpu-small")

# Legacy style (still used for AMD/CPU/NPU) — single suite string
register_cpu_ci(est_time=15, suite="base-a-test-cpu")
```

`est_time`, `stage`, and `runner_config` must be **literal values** (AST-parsed by `run_suite.py`). The two modes are mutually exclusive: use either `(stage, runner_config)` or `suite`, never both. Test files must end with `unittest.main()` or `pytest.main([__file__])`. Do not add custom argparse before these calls.

### CI Pipeline

Three sequential stages: **A** (pre-flight, ~3 min) → **B** (basic, ~30 min) → **C** (advanced, ~30 min). Kernel and multimodal-gen tests run in parallel with stage B. Most new tests should target `base-b-test-1-gpu-small`.

## Architecture

### Three-Process Design

The runtime (`python/sglang/srt/`) uses three cooperating processes communicating via ZMQ IPC:

1. **TokenizerManager** (main process): Tokenizes requests, routes to scheduler, manages async request state
2. **Scheduler** (subprocess): Schedules batches, manages KV cache via RadixCache, coordinates model workers
3. **DetokenizerManager** (subprocess): Incremental detokenization, sends results back to TokenizerManager

Entry points:
- HTTP server: `python/sglang/srt/entrypoints/http_server.py` (`launch_server()`)
- Python API: `python/sglang/srt/entrypoints/engine.py` (`Engine` class)
- CLI: `sglang serve` / `sglang generate`

### Request Flow

```
HTTP/API request
  -> entrypoints/openai/ (or ollama/, anthropic/) converts to GenerateReqInput
  -> TokenizerManager tokenizes, sends TokenizedGenerateReqInput via ZMQ
  -> [DataParallelController routes if dp_size > 1]
  -> Scheduler: prefix-matches against RadixCache, schedules batch, builds ForwardBatch
  -> TpModelWorker -> ModelRunner.forward() -> model layers -> sampling
  -> Output tokens sent via ZMQ to DetokenizerManager
  -> Detokenized text sent back to TokenizerManager -> HTTP response
```

### Data Structure Pipeline

`ScheduleBatch` (CPU scheduling) → `ForwardBatch` (GPU tensors, constructed via `ForwardBatch.init_new`)

### Forward Modes

Defined in `model_executor/forward_batch_info.py` as `ForwardMode` enum:
- `EXTEND`: Prefill new tokens
- `DECODE`: Generate one token per request
- `MIXED`: Chunked prefill (extend + decode combined)
- `TARGET_VERIFY` / `DRAFT_EXTEND`: Speculative decoding phases
- `DRAFT_EXTEND_V2`: EAGLE v2 speculative decoding
- `DLLM_EXTEND`: Distributed LLM extend

### Key Subsystems

- **Models** (`srt/models/`): 190+ model implementations with a registry in `registry.py`
- **Attention** (`srt/layers/attention/`): 20+ backends via registry pattern (FlashInfer, FlashAttention, FlashMLA, Triton, TRT-LLM, etc.)
- **MoE** (`srt/layers/moe/`): Multiple MoE implementations (Triton fused, CUTLASS, EP, etc.)
- **Speculative Decoding** (`srt/speculative/`): Plugin-based system supporting EAGLE, MTP, N-gram, DFlash, standalone draft models
- **Memory Cache** (`srt/mem_cache/`): RadixCache for KV cache prefix sharing, hierarchical caching (HiCache), eviction policies
- **Disaggregation** (`srt/disaggregation/`): Prefill-decode separation across GPUs with multiple transfer backends (NIXL, Mooncake, Mori)
- **Quantization** (`srt/layers/quantization/`): 30+ methods (AWQ, GPTQ, FP8, FP4, Marlin, etc.)

### Scheduler Design

The `Scheduler` class in `managers/scheduler.py` uses a mixin pattern for feature composition. Current mixins: `SchedulerDisaggregationDecodeMixin`, `SchedulerDisaggregationPrefillMixin`, `SchedulerMultiplexMixin`, `SchedulerPPMixin`, `SchedulerDllmMixin`, `SchedulerMlxOverlapMixin`.

### Environment Variables

SGLANG_*/SGL_* environment variables are managed through a descriptor-based system in `srt/environ.py`. **Always use `envs.FLAG.get()` to read values** — bare `envs.FLAG` raises RuntimeError. For temporary overrides, use the context manager:

```python
from sglang.srt.environ import envs

value = envs.SGLANG_INIT_NEW_TOKEN_RATIO.get()      # read
envs.SGLANG_INIT_NEW_TOKEN_RATIO.set(0.7)            # write
with envs.SGLANG_TEST_RETRACT.override(True):        # temporary override
    ...
```

Do not use raw `os.environ` for these variables.

### Parallelism Hierarchy

```
Attention: Global(TP) -> DP -> ATTN_CP -> ATTN_TP (innermost)
MoE:       Global(TP) -> MOE_DP -> EP -> MOE_TP (innermost)
```

### JIT Kernels vs AOT Kernels

- **JIT kernels** (`python/sglang/jit_kernel/`): Triton-based, compiled at runtime. 50+ kernel files.
- **AOT kernels** (`sgl-kernel/`): C++/CUDA, pre-compiled into the `sglang-kernel` wheel. CMake-based build with 19 subdirectories in `csrc/`.

## Conventions

- **Documentation**: Legacy `docs/` is frozen (pre-commit rejects changes). New docs go in `docs_new/`.
- **Test organization**: CI tests go in `test/registered/<category>/`. Manual/debug tests go in `test/manual/`.
- **Server launch is expensive**: Tests should share servers across methods via `setUpClass`. Each test file should take < 500 seconds.
- **Generated code**: gRPC files (`*_pb2.py`, `*_pb2_grpc.py`) are auto-generated from `proto/` definitions — do not edit directly.
- **Multimodal generation**: Separate subsystem in `python/sglang/multimodal_gen/` with its own CLAUDE.md, architecture (ComposedPipeline with stages), and test infrastructure.
- **Speculative decoding naming**: See `.claude/rules/speculative-naming.md` for naming conventions (e.g., `accept_tokens` not `accepted_token_ids`, `bonus_token` not `verified_id`).