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

__global__ void rescale(float* dIn, float alpha) {
	size_t bIdx = blockIdx.x;
	size_t tIdx = threadIdx.x;
	size_t idx = bIdx * blockDim.x + tIdx;
	dIn[idx] = alpha * dIn[idx];
}

__global__ void fillForceMatrix(float* dIn) {
	size_t bIdx = blockIdx.x + 1;
	size_t tIdx = threadIdx.x;
	size_t idx = bIdx * blockDim.x + tIdx;
	if (bIdx > tIdx) {
		dIn[idx] = 0;
	} else {
		dIn[idx] = dIn[tIdx];
	}
}

void calcShiftingByOutsideForce(float* dIn, size_t size, size_t count, float time) {
	interpolation(dIn, 1, size, count);
	size_t numPerRow = size + count * (size - 1);
	rescale<<<1, numPerRow >>>(
		dIn, time / static_cast<float>(count + 1));
	CUDACHECK(cudaGetLastError());
	fillForceMatrix<<<numPerRow, numPerRow >>>(dIn);
	CUDACHECK(cudaGetLastError());
	cuSum(dIn, dIn, numPerRow, numPerRow);
	cuSum(dIn, dIn, numPerRow, numPerRow);
	rescale << <numPerRow, numPerRow >> > (
		dIn, time / static_cast<float>((count + 1)));
	CUDACHECK(cudaGetLastError());
}