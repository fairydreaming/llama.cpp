#!/usr/bin/env bash
# CPU-only AVX-512 build of llama-server and llama-bench for DeepSeek-V4.1-Flash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BUILD_DIR="$ROOT/build-cpu"
JOBS=48

cmake -S "$ROOT" -B "$BUILD_DIR" \
    -DCMAKE_BUILD_TYPE=Release \
    -DGGML_NATIVE=ON \
    -DGGML_AVX512=ON \
    -DGGML_BLAS=ON \
    -DGGML_BLAS_VENDOR=OpenBLAS \
    -DGGML_CUDA=OFF \
    -DGGML_VULKAN=OFF \
    -DGGML_METAL=OFF \
    -DGGML_SYCL=OFF \
    -DGGML_HIP=OFF \
    -DGGML_OPENMP=ON \
    -DGGML_LTO=OFF \
    -DGGML_CCACHE=ON \
    -DLLAMA_BUILD_TESTS=OFF \
    -DLLAMA_BUILD_EXAMPLES=OFF \
    -DLLAMA_BUILD_SERVER=ON

cmake --build "$BUILD_DIR" --target llama-server llama-bench -j"$JOBS"

echo
echo "binaries:"
echo "  $BUILD_DIR/bin/llama-server"
echo "  $BUILD_DIR/bin/llama-bench"