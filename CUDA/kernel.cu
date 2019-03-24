#include "kernel.h"
#include "kernels.h"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "device_functions.h"

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

__global__ void scan(float* dOut, float* dIn, size_t size, binary_func_t func)
{
	unsigned int idx = threadIdx.x;
	size_t step = 1;
	dOut[idx] = func(dIn[idx], (idx >= step ? dIn[idx - step] : 0));
	__syncthreads();
	for (step*=2; step<size; step*=2){
		dOut[idx] = func(dOut[idx], (idx >= step ?  dOut[idx - step] : 0));
		__syncthreads();
	}
}

void cuSum(float* dOut, float* dIn, size_t size) {
	scan << <1, size >> > (dOut, dIn, size, [](float lhs, float rhs) {
		return lhs + rhs;
	});
}