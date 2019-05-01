#include "kernel.h"
#include "kernels.h"
#include "cuda_runtime.h"

// lint only
#ifndef __CUDACC__
#define __CUDACC__
#endif // !__CUDACC__
#include "device_launch_parameters.h"
#include "device_functions.h"

namespace cudaKernel {

template<typename T, class Func>
__global__ void reduceOld(T *matrix, T* result, Func f) {
	__shared__ T mem[1024];
	size_t tid = threadIdx.x;
	size_t bid = blockIdx.x;
	T* array = matrix + bid * blockDim.x;
	mem[tid] = array[tid];
	for (uint32_t s = 1; s < blockDim.x; s = s << 1) {
		if (tid % (2 * s) == 0) {
			mem[tid] = f(mem[tid], mem[tid + s]);
		}
		__syncthreads();
	}
	if (threadIdx.x == 0) {
		result[bid] = mem[0];
	}
}

template<typename T, class Func>
__global__ void reduce(T *matrix, T* result, Func f) {
	__shared__ T mem2[1024];
	size_t tid = threadIdx.x;
	size_t bid = blockIdx.x;
	T* array = matrix + bid * blockDim.x;
	mem2[tid] = array[tid];
	for (uint32_t s = blockDim.x / 2; s > 0; s =  s >> 1) {
		if (tid < s) {
			mem2[tid] += mem2[tid + s];
		}
		__syncthreads();
	}
	if (threadIdx.x == 0) {
		result[bid] = mem2[0];
	}
}

template<typename T>
__device__ T sum(T lhs, T rhs) {
	return lhs + rhs;
}

template<typename T>
__device__ T min(T lhs, T rhs) {
	return lhs < rhs ? lhs : rhs;
}

__device__ binary_func_t sum_float_d = sum;
__device__ binary_func_t min_float_d = min;
void reduce(float *dMatrix, float* dResult,
	size_t nGroups, size_t size, ReduceOption op) {
	binary_func_t f;
	switch (op) {
	case ReduceOption::min:
		CUDACHECK(cudaMemcpyFromSymbol(
			&f, min_float_d, sizeof(binary_func_t)));
		break;
	case ReduceOption::sum:
	default:
		CUDACHECK(cudaMemcpyFromSymbol(
			&f, sum_float_d, sizeof(binary_func_t)));
	}
	reduce << < nGroups, size >> > (dMatrix, dResult, f);
	CUDACHECK(cudaGetLastError());
}

void reduce2(float *dMatrix, float* dResult,
	size_t nGroups, size_t size, ReduceOption op) {
	binary_func_t f;
	switch (op) {
	case ReduceOption::min:
		CUDACHECK(cudaMemcpyFromSymbol(
			&f, min_float_d, sizeof(binary_func_t)));
		break;
	case ReduceOption::sum:
	default:
		CUDACHECK(cudaMemcpyFromSymbol(
			&f, sum_float_d, sizeof(binary_func_t)));
	}
	reduceOld << < nGroups, size >> > (dMatrix, dResult, f);
	CUDACHECK(cudaGetLastError());
}

__device__ binary_func_size_t_t sum_size_t_d = sum;
__device__ binary_func_size_t_t min_size_t_d = min;
void reduce(size_t *dMatrix, size_t* dResult,
	size_t nGroups, size_t size, ReduceOption op) {
	binary_func_size_t_t f;
	switch (op) {
	case ReduceOption::min:
		CUDACHECK(cudaMemcpyFromSymbol(
			&f, min_size_t_d, sizeof(binary_func_size_t_t)));
		break;
	case ReduceOption::sum:
	default:
		CUDACHECK(cudaMemcpyFromSymbol(
			&f, sum_size_t_d, sizeof(binary_func_size_t_t)));
	}
	reduce << < nGroups, size, size * sizeof(size_t) >> > (dMatrix, dResult, f);
	CUDACHECK(cudaGetLastError());
}

void reduceMin(float *dMatrix, float* dResult, size_t nGroups, size_t size) {
	reduce(dMatrix, dResult, nGroups, size, ReduceOption::min);
}

void reduceMin(size_t *dMatrix, size_t* dResult, size_t nGroups, size_t size) {
	reduce(dMatrix, dResult, nGroups, size, ReduceOption::min);
}

}