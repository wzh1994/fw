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

__global__ void dispatch(float* vbo, float* points, float* colors)
{
	unsigned int i = threadIdx.x;
	vbo[(i / 3) * 6 + i % 3] = points[i];
	vbo[(i / 3) * 6 + i % 3 + 3] = colors[i];

}

size_t getTrianglesAndIndices(
	float* vbo, uint32_t* dIndices, float* dPoints,
	float* dColors, float* dSizes, size_t size) {

	dispatch << <1, size * 3 >> > (vbo, dPoints, dColors);
	uint32_t indice[] = {
		0, 1, 2,
		0, 2, 3,
		0, 3, 1,
		1, 2, 3
	};
	cudaMemcpy(dIndices, indice, 12 * sizeof(uint32_t), cudaMemcpyHostToDevice);
	return 12;
}

__global__ void scan(float* dOut, float* dIn, binary_func_t func)
{
	uint32_t size = blockDim.x;
	uint32_t offset = blockIdx.x * size;
	uint32_t idx = threadIdx.x;
	size_t step = 1;
	dOut[idx + offset] = func(dIn[idx + offset],
		(idx >= step ? dIn[idx + offset - step] : 0));
	__syncthreads();
	for (step*=2; step<size; step*=2){
		dOut[idx + offset] = func(dOut[idx + offset],
				(idx >= step ?  dOut[idx + offset - step] : 0));
		__syncthreads();
	}
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
void cuSum(float* dOut, float* dIn, size_t size, size_t numGroup) {
	binary_func_t add;
	// 必须要将device的函数指针传到host上面，才可以回调
	cudaMemcpyFromSymbol(&add, add_float_d, sizeof(binary_func_t));
	scan << <numGroup, size >> > (dOut, dIn, add);
	cudaDeviceSynchronize();
}

__global__ void cusum(size_t* dOut, const size_t* dIn)
{
	uint32_t size = blockDim.x;
	uint32_t offset = blockIdx.x * size;
	uint32_t idx = threadIdx.x;
	size_t step = 1;
	size_t sum = dIn[idx + offset] +
		(idx >= step ? dIn[idx + offset - step] : 0);
	__syncthreads();
	dOut[idx + offset] = sum;
	__syncthreads();
	for (step *= 2; step < size; step *= 2) {
		size_t sum = dOut[idx + offset] +
			(idx >= step ? dOut[idx + offset - step] : 0);
		__syncthreads();
		dOut[idx + offset] = sum;
		__syncthreads();
	}
	printf("cusum: %llu, %llu\n", idx, dOut[idx]);
}

void cuSum(size_t* dOut, const size_t* dIn, size_t size, size_t numGroup) {
	cusum << <numGroup, size >> > (dOut, dIn);
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
void cuMax(float* dOut, float* dIn, size_t size, size_t numGroup) {
	binary_func_t max;
	cudaMemcpyFromSymbol(&max, max_float_d, sizeof(binary_func_t));
	scan << <numGroup, size >> > (dOut, dIn, max);
	cudaDeviceSynchronize();
}