#include "kernel.h"
#include "kernels.h"
#include "cuda_runtime.h"

// lint only
#ifndef __CUDACC__
#define __CUDACC__
#endif // !__CUDACC__
#include "device_launch_parameters.h"
#include "device_functions.h"

#include <cstdio>
#include "test.h"

namespace cudaKernel {

template<typename T, class F>
__global__ void scan(T* out, const T* in, F func, size_t groupSize) {
	size_t offset = blockIdx.x * groupSize + blockIdx.y * blockDim.x;
	const T* pIn = in + offset;
	T* pOut = out + offset;
	__shared__ T result[1024];
	size_t idx = threadIdx.x;
	result[idx] = pIn[idx];
	__syncthreads();
	size_t step = 1;
	T temp = 0;
	if (idx >= step) {
		temp = result[idx - step];
	}
	temp = func(result[idx], temp);

	__syncthreads();
	result[idx] = temp;
	__syncthreads();
	for (step *= 2; step < blockDim.x; step *= 2) {
		T temp = func(result[idx],
			(idx >= step ? result[idx - step] : 0));
		__syncthreads();
		result[idx] = temp;
		__syncthreads();
	}
	if (blockIdx.y * blockDim.x + idx < groupSize) {
		pOut[idx] = result[idx];
	}
}

template<typename T, class F>
__global__ void scanGroup(
	T* out, const T* in, F func, size_t groupSize, size_t stride) {
	uint32_t offset = blockIdx.x * groupSize;
	const T* pIn = in + offset;
	T* pOut = out + blockIdx.x * blockDim.x;
	__shared__ T result[1024];
	uint32_t idx = threadIdx.x;
	result[idx] = pIn[(idx + 1) * stride - 1];
	__syncthreads();
	size_t step = 1;
	T temp = func(result[idx],
		(idx >= step ? result[idx - step] : 0));
	__syncthreads();
	result[idx] = temp;
	__syncthreads();
	for (step *= 2; step < blockDim.x; step *= 2) {
		T temp = func(result[idx],
			(idx >= step ? result[idx - step] : 0));
		__syncthreads();
		result[idx] = temp;
		__syncthreads();
	}
	pOut[idx] = result[idx];
}

template<typename T, class F>
__global__ void groupResultToOut(T* out, const T* in, F f, size_t groupSize) {
	uint32_t offset = blockIdx.x * groupSize + blockIdx.y * blockDim.x;
	T* dOut = out + offset;
	if (blockIdx.y > 0 && blockIdx.y * blockDim.x + threadIdx.x < groupSize) {
		T temp = in[blockIdx.x * gridDim.y + blockIdx.y - 1];
		dOut[threadIdx.x] = f(temp, dOut[threadIdx.x]);
	}
}
template<typename T, class F>
void callScanKernel(T* dOut, const T* dIn, size_t size, size_t nGroups, F f) {
	size_t nBlockDim = kMmaxBlockDim;
	if (size <= nBlockDim) {
		scan << <nGroups, size >> > (dOut, dIn, f, size);
		CUDACHECK(cudaGetLastError());
	}
	else if (size <= nBlockDim * nBlockDim) {
		size_t nSubGroups = ceilAlign(size, nBlockDim);
		scan << <dim3(nGroups, nSubGroups), nBlockDim >> > (dOut, dIn, f, size);
		CUDACHECK(cudaGetLastError());
		T* groupScanResults;
		cudaMallocAlign(&groupScanResults, nGroups * nSubGroups * sizeof(T));
		scanGroup << <nGroups, nSubGroups >> > (
			groupScanResults, dOut, f, size, nBlockDim);
		CUDACHECK(cudaGetLastError());
		groupResultToOut << <dim3(nGroups, nSubGroups), nBlockDim >> > (
			dOut, groupScanResults, f, size);
		CUDACHECK(cudaGetLastError());
		CUDACHECK(cudaFree(groupScanResults));
	}
	else {
		throw std::runtime_error("Max size of each group is "
			+ std::to_string(nBlockDim * nBlockDim) + "in scan!");
	}
	CUDACHECK(cudaGetLastError());
	cudaDeviceSynchronize();
}

// ================================
//
//    求开始到当前节点的累加和
//
// ================================

template<typename T>
__device__ T add(T lhs, T rhs) {
	return lhs + rhs;
}

__device__ binary_func_t add_float_d = add;
void cuSum(float* dOut, const float* dIn, size_t size, size_t numGroup) {
	binary_func_t hAdd;
	// 必须要将device的函数指针传到host上面，才可以回调
	cudaMemcpyFromSymbol(&hAdd, add_float_d, sizeof(binary_func_t));
	callScanKernel(dOut, dIn, size, numGroup, hAdd);
}

__device__ binary_func_size_t_t add_size_t_d = add;
void cuSum(size_t* dOut, const size_t* dIn, size_t size, size_t numGroup) {
	binary_func_size_t_t hAdd;
	cudaMemcpyFromSymbol(&hAdd, add_size_t_d, sizeof(binary_func_size_t_t));
	callScanKernel(dOut, dIn, size, numGroup, hAdd);
}

// ================================
//
//    求开始到当前节点的最大值
//
// ================================

template<typename T>
__device__ T max(T lhs, T rhs) {
	return lhs > rhs ? lhs : rhs;
}

__device__ binary_func_t max_float_d = max;
void cuMax(float* dOut, const float* dIn, size_t size, size_t numGroup) {
	binary_func_t hMax;
	cudaMemcpyFromSymbol(&hMax, max_float_d, sizeof(binary_func_t));
	callScanKernel(dOut, dIn, size, numGroup, hMax);
}

}