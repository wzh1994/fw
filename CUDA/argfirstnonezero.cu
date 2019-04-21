#include "kernel.h"
#include "kernels.h"
#include "cuda_runtime.h"
#include "corecrt_math.h"

// 为了让__syncthreads()通过语法检查
#ifndef __CUDACC__
#define __CUDACC__
#endif // !__CUDACC__
#include "device_launch_parameters.h"
#include "device_functions.h"
#include <cstdio>

__global__ void argFirstNoneZero(size_t* matrix, size_t* result) {
	size_t tid = threadIdx.x;
	size_t bid = blockIdx.x;
	size_t* array = matrix + bid * blockDim.x;
	extern __shared__ size_t mem[];
	mem[tid] = array[tid] == 0 ? UINT_MAX : tid;

	for (int s = 1; s < blockDim.x; s *= 2) {
		if (tid % (2 * s) == 0 && s + tid < blockDim.x) {
			mem[tid] = mem[tid] < mem[tid + s] ? mem[tid] : mem[tid + s];
		}
		__syncthreads();
	}
	if (threadIdx.x == 0) {
		result[bid] = mem[0];
	}
	// printf("%llu %llu : %llu\n", bid, tid, mem[tid]);
}

void argFirstNoneZero(size_t* dMatrix, size_t* result,
		size_t nGroups, size_t size){
	argFirstNoneZero << <nGroups, size, size * sizeof(size_t)>> > (
		dMatrix, result);
	CUDACHECK(cudaGetLastError());
}