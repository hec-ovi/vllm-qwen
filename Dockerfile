# vllm-qwen source-build image — Ubuntu 26.04 + TheRock ROCm + vLLM from source.
#
# Minimal port of kyuz0/amd-strix-halo-vllm-toolboxes' Fedora recipe, stripped
# down to what's needed for FP8 inference on a single gfx1151 iGPU:
#   - AITER custom build  : SKIPPED. We disable AITER at runtime via
#                           VLLM_ROCM_USE_AITER=0; building it is wasted cycles.
#   - Flash-Attention      : SKIPPED. vLLM's built-in Triton attention covers us.
#   - bitsandbytes ROCm    : SKIPPED. We're not doing bnb quants; FP8 is native
#                           in the Qwen3.6-27B-FP8 checkpoint.
#   - Custom RCCL          : SKIPPED. Single-GPU; no multi-node NCCL.
#
# What survives unchanged from kyuz0:
#   - TheRock tarball install (scripts/install_rocm_sdk.sh)
#   - vLLM source-level patches (scripts/patch_strix.py) — 10 patches, none
#     of which depend on OS.

FROM ubuntu:26.04

# 1. Build + inference-runtime system deps.
# Inference-only. No InfiniBand, no ffmpeg, no rsync, no editors, no multi-node
# networking. python3.12 is NOT installed via apt — Ubuntu 26.04 ships 3.14
# and has no python3.12 apt package; uv provisions its own 3.12 in step 3.
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
      ca-certificates curl git \
      build-essential cmake ninja-build \
      aria2 tar xz-utils \
      libatomic1 libnuma-dev libgomp1 libelf1t64 \
      libdrm-dev zlib1g-dev libssl-dev \
      libgoogle-perftools4 \
      procps \
    && rm -rf /var/lib/apt/lists/*

# 2. TheRock ROCm SDK → /opt/rocm (tarball method, OS-agnostic).
WORKDIR /tmp
ARG ROCM_MAJOR_VER=7
ARG GFX=gfx1151
COPY scripts/install_rocm_sdk.sh /tmp/install_rocm_sdk.sh
RUN chmod +x /tmp/install_rocm_sdk.sh && \
    ROCM_MAJOR_VER=${ROCM_MAJOR_VER} GFX=${GFX} /tmp/install_rocm_sdk.sh && \
    rm /tmp/install_rocm_sdk.sh

# 3. Python 3.12 venv via uv (Ubuntu 26.04 has no python3.12 apt package).
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv
ENV VIRTUAL_ENV=/opt/venv
ENV PATH=/opt/venv/bin:/opt/rocm/bin:/opt/rocm/llvm/bin:$PATH
RUN uv venv /opt/venv --python 3.12 && \
    uv pip install --upgrade pip wheel packaging "setuptools<80.0.0"

# 4. PyTorch + triton from AMD gfx1151 prerelease index.
# Same URL we validated in test/trace.py — resolves to rocm7.12.0rc1 today.
RUN uv pip install --pre torch torchvision torchaudio \
      --index-url https://rocm.prereleases.amd.com/whl/gfx1151/ && \
    rm -rf /root/.cache/uv /root/.cache/pip

# 5. Build tool deps for the vLLM native build.
RUN uv pip install --upgrade cmake ninja packaging wheel numpy \
      "setuptools-scm>=8" "setuptools<80.0.0" scikit-build-core pybind11

# 6. Clone vLLM and apply the Strix Halo patch bundle.
# VLLM_COMMIT can pin a specific sha for reproducibility; empty tracks HEAD.
ARG VLLM_COMMIT=
RUN git clone https://github.com/vllm-project/vllm.git /opt/vllm
WORKDIR /opt/vllm
RUN if [ -n "$VLLM_COMMIT" ]; then \
      echo "Pinning vLLM to ${VLLM_COMMIT}"; \
      git checkout "${VLLM_COMMIT}"; \
    else \
      echo "Tracking vLLM HEAD: $(git rev-parse --short HEAD)"; \
    fi

COPY scripts/patch_strix.py /opt/vllm/patch_strix.py
RUN python /opt/vllm/patch_strix.py

# 7. Build vLLM against TheRock ROCm.
# CC/CXX forced to ROCm's clang so the compiled extensions have an ABI
# that matches the torch wheels (kyuz0 notes this as the fix for a subtle
# segfault with the Fedora host gcc).
ENV ROCM_HOME=/opt/rocm \
    ROCM_PATH=/opt/rocm \
    HIP_PATH=/opt/rocm \
    HIP_PLATFORM=amd \
    CMAKE_PREFIX_PATH=/opt/rocm \
    VLLM_TARGET_DEVICE=rocm \
    PYTORCH_ROCM_ARCH=gfx1151 \
    HIP_ARCHITECTURES=gfx1151 \
    GPU_TARGETS=gfx1151 \
    MAX_JOBS=4 \
    CC=/opt/rocm/llvm/bin/clang \
    CXX=/opt/rocm/llvm/bin/clang++

RUN export HIP_DEVICE_LIB_PATH=$(find /opt/rocm -type d -name bitcode -print -quit) && \
    echo "Building with bitcode: $HIP_DEVICE_LIB_PATH" && \
    export CMAKE_ARGS="-DROCM_PATH=/opt/rocm -DHIP_PATH=/opt/rocm -DGPU_TARGETS=gfx1151 -DHIP_ARCHITECTURES=gfx1151" && \
    uv pip install --no-build-isolation --no-deps -v . && \
    rm -rf /root/.cache/uv /root/.cache/pip /tmp/*

# 7b. Install vLLM's runtime dependencies.
# vLLM declares deps as `dynamic` in pyproject.toml and loads them from
# requirements/common.txt + requirements/rocm.txt at build time. Since we
# built with --no-deps, install the same lists directly now. The constraint
# pins torch/triton to our ROCm prerelease wheels so uv doesn't try to
# replace them with vanilla CUDA torch from PyPI when resolving the graph.
RUN printf "torch==2.10.0+rocm7.12.0rc1\ntriton==3.6.0+rocm7.12.0rc1\n" > /tmp/constraints.txt && \
    uv pip install --no-build-isolation \
      -r /opt/vllm/requirements/common.txt \
      -r /opt/vllm/requirements/rocm.txt \
      --constraint /tmp/constraints.txt && \
    rm -rf /root/.cache/uv /root/.cache/pip /tmp/constraints.txt

# 8. Runtime env (mirrors /etc/profile.d/rocm-sdk.sh for non-interactive usage).
ENV LD_LIBRARY_PATH=/opt/rocm/lib:/opt/rocm/lib64:/opt/rocm/llvm/lib \
    HIP_CLANG_PATH=/opt/rocm/llvm/bin \
    ROCBLAS_USE_HIPBLASLT=1 \
    TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL=1 \
    HIP_FORCE_DEV_KERNARG=1 \
    RAY_EXPERIMENTAL_NOSET_ROCR_VISIBLE_DEVICES=1 \
    LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libtcmalloc_minimal.so.4 \
    HSA_OVERRIDE_GFX_VERSION=11.5.1 \
    VLLM_ROCM_USE_AITER=0

WORKDIR /opt
CMD ["/bin/bash"]
