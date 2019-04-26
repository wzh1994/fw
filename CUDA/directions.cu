#include "kernel.h"
#include "kernels.h"
#include "cuda_runtime.h"
#include "corecrt_math.h"
#include "utils.h"
#include "test.h"
#include <corecrt_math_defines.h>

// 为了让__syncthreads()通过语法检查
#ifndef __CUDACC__
#define __CUDACC__
#endif // !__CUDACC__
#include "device_launch_parameters.h"
#include "device_functions.h"
#include <cstdio>

namespace cudaKernel {

namespace normalFw{

__global__ void getNumbersPerGroup(size_t *sizes, float *angles, size_t n) {
	angles[threadIdx.x] =
		static_cast<float>(threadIdx.x) * M_PI * 2 / static_cast<float>(n);
	sizes[threadIdx.x] =
		static_cast<size_t>(fmaxf(1.0f, n * sinf(angles[threadIdx.x])));
}

__global__ void getDirections(float *directions, const float* angles,
		const size_t* sizes, const size_t* offsets) {
	size_t bid = blockIdx.x;
	size_t tid = threadIdx.x;
	size_t n = sizes[bid];
	float* basePtr = directions + offsets[bid] * 3;
	if (tid < n) {
		float angle1 = angles[bid];
		float angle2 =
			static_cast<float>(threadIdx.x) * M_PI * 2 / static_cast<float>(n);
		basePtr[3 * tid] = cosf(angle1);
		basePtr[3 * tid + 1] = sinf(angle1) * cosf(angle2);
		basePtr[3 * tid + 2] = sinf(angle1) * sinf(angle2);
	}
}

}

size_t normalFireworkDirections(
		float* dDirections, size_t nIntersectingSurfaceParticle) {
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
	normalFw::getDirections<<<nGroups, nIntersectingSurfaceParticle >>>(
		dDirections, dAngles, dSizes, dOffsets);
	CUDACHECK(cudaGetLastError());
	size_t numDirections;
	CUDACHECK(cudaMemcpy(&numDirections, dOffsets + nGroups,
		sizeof(size_t), cudaMemcpyDeviceToHost));
	cudaFreeAll(dSizes, dOffsets, dAngles);
	return numDirections;
}

}  // end namespace cudaKernel