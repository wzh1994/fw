#include "kernel.h"
#include "kernels.h"
#include "cuda_runtime.h"
#include "corecrt_math.h"
#include "utils.h"
#include "test.h"
#include <corecrt_math_defines.h>
#include <chrono>
#include <mutex>

// 为了让__syncthreads()通过语法检查
#ifndef __CUDACC__
#define __CUDACC__
#endif // !__CUDACC__
#include "device_launch_parameters.h"
#include "device_functions.h"
#include "curand_kernel.h" 
#include <cstdio>

namespace cudaKernel {

__global__ void kernel_set_random(
		curandState *curandStates, long seed) {
	int idx = threadIdx.x + blockIdx.x * blockDim.x;
	curand_init(seed, idx, 0, curandStates + idx);
}

namespace normalFw{

__global__ void getNumbersPerGroup(size_t *sizes, float *angles, size_t n) {
	angles[threadIdx.x] =
		static_cast<float>(threadIdx.x) * M_PI * 2 / static_cast<float>(n);
	sizes[threadIdx.x] =
		static_cast<size_t>(fmaxf(1.0f, n * sinf(angles[threadIdx.x])));
}

__global__ void getDirections(float *directions, const float* angles,
		const size_t* sizes, const size_t* offsets, float xStretch,
		float xRate, float yStretch, float yRate, float zStretch,
		float zRate, curandState *devStates) {
	size_t bid = blockIdx.x;
	size_t tid = threadIdx.x;
	size_t idx = offsets[bid] + tid;
	size_t n = sizes[bid];
	float* basePtr = directions + idx * 3;
	if (tid < n) {
		float angle1 = angles[bid];
		float angle2 =
			static_cast<float>(threadIdx.x) * M_PI * 2 / static_cast<float>(n);
		xRate = (2 * abs(curand_uniform(devStates + idx)) - 1) * xRate + xStretch;
		yRate = (2 * abs(curand_uniform(devStates + idx)) - 1) * yRate + yStretch;
		zRate = (2 * abs(curand_uniform(devStates + idx)) - 1) * zRate + zStretch;
		
		basePtr[0] = cosf(angle1) * xRate;
		basePtr[1] = sinf(angle1) * cosf(angle2) * yRate;
		basePtr[2] = sinf(angle1) * sinf(angle2) * zRate;
	}
}

}

void initRand(curandState *dev_states, size_t size) {
	static clock_t seed;
	static std::once_flag inited;
	std::call_once(inited, [] { seed = clock(); });
	size_t blockDim = 256;
	size_t gridDim = ceilAlign(size, blockDim);
	kernel_set_random<<<gridDim, blockDim>>>(dev_states, seed);
}

size_t normalFireworkDirections(float* dDirections,
		size_t nIntersectingSurfaceParticle,
		float xStretch, float xRate,
		float yStretch, float yRate,
		float zStretch, float zRate) {
	size_t nGroups = nIntersectingSurfaceParticle / 2 + 1;
	size_t *dSizes, *dOffsets;
	float* dAngles;
	cudaMallocAlign(&dSizes, nGroups * sizeof(size_t));
	cudaMallocAlign(&dOffsets, (nGroups + 1) * sizeof(size_t));
	cudaMallocAlign(&dAngles, (nGroups) * sizeof(float));
	normalFw::getNumbersPerGroup<<<1, nGroups>>>(
		dSizes, dAngles, nIntersectingSurfaceParticle);
	CUDACHECK(cudaGetLastError());
	CUDACHECK(cudaMemset(dOffsets, 0, (nGroups + 1) * sizeof(size_t)));
	cuSum(dOffsets + 1, dSizes, nGroups);
	CUDACHECK(cudaGetLastError());
	size_t numDirections;
	CUDACHECK(cudaMemcpy(&numDirections, dOffsets + nGroups,
		sizeof(size_t), cudaMemcpyDeviceToHost));
	curandState *devStates;
	CUDACHECK(cudaMallocAlign(&devStates, sizeof(curandState) * numDirections));
	initRand(devStates, numDirections);

	normalFw::getDirections << <nGroups, nIntersectingSurfaceParticle >> > (
		dDirections, dAngles, dSizes, dOffsets, xStretch, xRate,
		yStretch, yRate, zStretch, zRate, devStates);
	cudaFreeAll(dSizes, dOffsets, dAngles, devStates);
	return numDirections;
}

}  // end namespace cudaKernel