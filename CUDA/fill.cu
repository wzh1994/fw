#include "kernel.h"
#include "kernels.h"
#include "cuda_runtime.h"

// lint only
#ifndef __CUDACC__
#define __CUDACC__
#endif // !__CUDACC__
#include "device_launch_parameters.h"
#include "device_functions.h"
#include "utils.h"

template <class T>
__global__ void fillSmall(T* array, const T* data, size_t step) {
	for (size_t i = 0; i < step; ++i) {
		array[threadIdx.x * step + i] = data[i];
	}
}

template <class T>
__global__ void fillLarge(T* array, const T* data) {
	array[blockIdx.x * blockDim.x + threadIdx.x] = data[threadIdx.x];
}

void fill(size_t* dArray, const size_t* data, size_t size, size_t step) {
	size_t* dData;
	cudaMallocAndCopy(dData, data, step);
	if (step < 16) {
		fillSmall << <1, size >> > (dArray, dData, step);
	} else {
		fillLarge << <size, step >> > (dArray, dData);
	}
	CUDACHECK(cudaGetLastError());
	CUDACHECK(cudaFree(dData));
}

void fill(float* dArray, const float* data, size_t size, size_t step) {
	float* dData;
	cudaMallocAndCopy(dData, data, step);
	if (step < 16) {
		fillSmall << <1, size >> > (dArray, dData, step);
	} else {
		fillLarge << <size, step >> > (dArray, dData);
	}
	CUDACHECK(cudaGetLastError());
	CUDACHECK(cudaFree(dData));
}

template <class T>
__global__ void fill(T* array, T data) {
	array[threadIdx.x] = data;
}


void fill(size_t* dArray, size_t data, size_t size) {
	fill << <1, size >> > (dArray, data);
	CUDACHECK(cudaGetLastError());
}

void fill(float* dArray, float data, size_t size) {
	fill << <1, size >> > (dArray, data);
	CUDACHECK(cudaGetLastError());
}