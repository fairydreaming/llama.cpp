#!/bin/bash

for impl in bitonic argsort radix DeviceTopK
do
	export GGML_CUDA_TOP_K_IMPL="$impl" 
	./bin/test-backend-ops perf -o TOP_K 2>&1 | tee "top_k_dev_${CUDA_VISIBLE_DEVICES}_impl_${GGML_CUDA_TOP_K_IMPL}.txt"
done
