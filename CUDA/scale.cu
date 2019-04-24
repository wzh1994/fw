#include "kernel.h"
#include "kernels.h"
#include "cuda_runtime.h"
#include "corecrt_math.h"
#define DEBUG_PRINT
#include "utils.h"

// 为了让__syncthreads()通过语法检查
#ifndef __CUDACC__
#define __CUDACC__
#endif // !__CUDACC__
#include "device_launch_parameters.h"
#include "device_functions.h"

__global__ void scale(float* array, float rate) {
	size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
	array[idx] *= rate;
}

void scale(float* dArray, float rate, size_t size) {
	size_t nBlockDims = ceilAlign(size, 256);
	scale<<<nBlockDims, 256 >>>(dArray, rate);
	CUDACHECK(cudaGetLastError());
}