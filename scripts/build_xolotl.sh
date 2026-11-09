#!/usr/bin/env bash
# Clone and build the Xolotl engine (the `strata` target only) for RTX 30-series cards (sm_86).
#
#   XOLOTL_DIR=~/xolotl ./scripts/build_xolotl.sh
#
# Tested with Xolotl commit 8486ead, llama.cpp pin 6050b467, CUDA 12.8, CMake 3.28.
# Change CUDA_ARCH for other cards: 89 = RTX 40, 120 = RTX 50 (use CUDA 13.x for RTX 50).
set -euo pipefail

XOLOTL_DIR=${XOLOTL_DIR:-$HOME/Xolotl}
XOLOTL_REF=${XOLOTL_REF:-8486ead}
CUDA_ARCH=${CUDA_ARCH:-86}
CUDA_HOME=${CUDA_HOME:-/usr/local/cuda-12.8}
JOBS=${JOBS:-$(nproc)}

export PATH="$CUDA_HOME/bin:$PATH"
command -v ninja >/dev/null || { echo "ninja not found (pip install ninja, or apt install ninja-build)"; exit 1; }

if [ ! -d "$XOLOTL_DIR/.git" ]; then
  git clone https://github.com/Indras-Mirror/Xolotl "$XOLOTL_DIR"
fi
cd "$XOLOTL_DIR"
git fetch -q origin "$XOLOTL_REF" 2>/dev/null || true
git checkout -q "$XOLOTL_REF" 2>/dev/null || echo "note: could not check out $XOLOTL_REF, building $(git rev-parse --short HEAD)"

# llama.cpp at Xolotl's pinned commit plus tools/xolotl/ggml/xolotl-ggml.patch -> third_party/llama.cpp
bash tools/xolotl/fetch-llama.sh

cmake -S . -B build-xolotl -G Ninja -DCMAKE_BUILD_TYPE=Release \
  -DSTRATA_ENABLE_CUDA=ON -DSTRATA_GGML_CUDA=ON -DSTRATA_MMQ_KQUANTS=ON \
  -DCMAKE_CUDA_ARCHITECTURES="$CUDA_ARCH" \
  -DSTRATA_GGML_DIR="$PWD/third_party/llama.cpp"
ninja -C build-xolotl -j "$JOBS" strata

echo "built: $XOLOTL_DIR/build-xolotl/strata"
