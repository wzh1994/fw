#include "kernel.h"
#include "kernels.h"
#include "cuda_runtime.h"
#include <stdexcept>

// lint only
#ifndef __CUDACC__
#define __CUDACC__
#endif // !__CUDACC__
#include "device_launch_parameters.h"
#include "device_functions.h"
#include "utils.h"
namespace cudaKernel {

template <class T>
__global__ void fillSmall(T* array, const T* data, size_t step) {
	size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
	//deviceDebugPrint("%u, %u, %llu\n", blockIdx.x, threadIdx.x, idx);
	for (size_t i = 0; i < step; ++i) {
		array[idx * step + i] = data[i];
	}
}

template <class T>
__global__ void fillLarge(T* array, const T* data) {
	array[blockIdx.x * blockDim.x + threadIdx.x] = data[threadIdx.x];
}

template<typename T>
void fillImpl(T* dArray, const T* data, size_t size, size_t step) {
	T* dData;
	cudaMallocAndCopy(dData, data, step);
	if (step < 16) {
		size_t nBlockDims = ceilAlign(size, 64);
		fillSmall << <nBlockDims, 64 >> > (dArray, dData, step);
	}
	else if (step <= kMmaxBlockDim) {
		fillLarge << <size, step >> > (dArray, dData);
	}
	else {
		throw std::runtime_error("Max step supported is 1024");
	}
	CUDACHECK(cudaGetLastError());
	cudaFreeAll(dData);
}

void fill(size_t* dArray, const size_t* data, size_t size, size_t step) {
	fillImpl(dArray, data, size, step);
}

void fill(float* dArray, const float* data, size_t size, size_t step) {
	fillImpl(dArray, data, size, step);
}

template <class T>
__global__ void fill(T* array, T data) {
	size_t idx = blockIdx.x*blockDim.x + threadIdx.x;
	array[idx] = data;
}

template<typename T>
void fillImpl(T* dArray, T data, size_t size) {
	size_t nBlockDims = ceilAlign(size, 256);
	fill << <nBlockDims, 256 >> > (dArray, data);
	CUDACHECK(cudaGetLastError());
}

void fill(size_t* dArray, size_t data, size_t size) {
	fillImpl(dArray, data, size);
}

void fill(float* dArray, float data, size_t size) {
	fillImpl(dArray, data, size);
}
}