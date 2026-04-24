#!/bin/bash
# Install TheRock ROCm SDK for gfx1151 via tarball. OS-agnostic.
# Adapted from kyuz0/amd-strix-halo-vllm-toolboxes/scripts/install_rocm_sdk.sh
set -euo pipefail

ROCM_MAJOR_VER="${ROCM_MAJOR_VER:-7}"
GFX="${GFX:-gfx1151}"

echo "=== Installing TheRock ROCm SDK ($GFX, major version $ROCM_MAJOR_VER) ==="

cd /tmp

BASE="https://therock-nightly-tarball.s3.amazonaws.com"
PREFIX="therock-dist-linux-${GFX}-${ROCM_MAJOR_VER}"

# Resolve latest tarball key from S3 bucket listing.
KEY="$(curl -s "${BASE}?list-type=2&prefix=${PREFIX}" \
  | tr '<' '\n' \
  | grep -o "therock-dist-linux-${GFX}-${ROCM_MAJOR_VER}\..*\.tar\.gz" \
  | sort -V | tail -n1)"

if [ -z "$KEY" ]; then
  echo "ERROR: no tarball matching ${PREFIX} found at ${BASE}" >&2
  exit 1
fi

echo "Downloading tarball: ${KEY}"
aria2c -x 16 -s 16 -j 16 --file-allocation=none "${BASE}/${KEY}" -o therock.tar.gz

mkdir -p /opt/rocm
tar xzf therock.tar.gz -C /opt/rocm --strip-components=1
rm therock.tar.gz

BITCODE_PATH=$(find /opt/rocm -type d -name bitcode -print -quit)

# Drop a profile.d fragment so interactive shells in the container pick up
# the ROCm env automatically. The Dockerfile ALSO sets the same variables
# via ENV so non-interactive RUN layers see them during build.
cat > /etc/profile.d/rocm-sdk.sh <<EOF
export ROCM_PATH=/opt/rocm
export HIP_PLATFORM=amd
export HIP_PATH=/opt/rocm
export HIP_CLANG_PATH=/opt/rocm/llvm/bin
export HIP_DEVICE_LIB_PATH=${BITCODE_PATH}
export PATH=/opt/rocm/bin:/opt/rocm/llvm/bin:\$PATH
export LD_LIBRARY_PATH=/opt/rocm/lib:/opt/rocm/lib64:/opt/rocm/llvm/lib:\$LD_LIBRARY_PATH
export ROCBLAS_USE_HIPBLASLT=1
export TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL=1
export VLLM_TARGET_DEVICE=rocm
export HIP_FORCE_DEV_KERNARG=1
export RAY_EXPERIMENTAL_NOSET_ROCR_VISIBLE_DEVICES=1
export LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libtcmalloc_minimal.so.4
EOF
chmod 0644 /etc/profile.d/rocm-sdk.sh

echo "Bitcode path: ${BITCODE_PATH}"
echo "=== ROCm SDK install complete ==="
