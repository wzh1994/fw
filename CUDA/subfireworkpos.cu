#include "kernel.h"
#include "kernels.h"
#include "cuda_runtime.h"
#include "corecrt_math.h"
#include "utils.h"

// 为了让__syncthreads()通过语法检查
#ifndef __CUDACC__
#define __CUDACC__
#endif // !__CUDACC__
#include "device_launch_parameters.h"
#include "device_functions.h"
#include <cstdio>

namespace cudaKernel{

__global__ void getSubFireworkPositions(
		float* startPoses, float* directions, const float* subDirs,
		size_t nDirs, size_t stride, const float* relativePos,
		size_t kShift, const float* shiftX_, const float* shiftY_) {
	size_t bid = blockIdx.x;
	size_t tid = threadIdx.x;
	size_t idx = bid * blockDim.x + tid;
	const float* dir = directions + bid * stride * 3;
	float* targetDir = directions + (nDirs + bid * blockDim.x + tid) * 3;
	startPoses[3 * idx] = dir[0] * *relativePos + shiftX_[kShift];
	startPoses[3 * idx + 1] = dir[1] * *relativePos + shiftY_[kShift];
	startPoses[3 * idx + 2] = dir[2] * *relativePos;
	targetDir[0] = subDirs[tid * 3];
	targetDir[1] = subDirs[tid * 3 + 1];
	targetDir[2] = subDirs[tid * 3 + 2];
}

void getSubFireworkPositions(float* dStartPoses, float* dDirections,
		const float* dSubDirs, size_t nDirs, size_t nSubDirs,
		size_t nSubGroups, const float* dCentrifugalPos_, size_t startFrame,
		size_t kShift, const float* dShiftX_, const float* dShiftY_) {
	size_t stride = nDirs / nSubGroups;
	const float* relativePos = dCentrifugalPos_ + startFrame;
	getSubFireworkPositions << <nSubGroups, nSubDirs >> > (
		dStartPoses, dDirections, dSubDirs, nDirs,
		stride, relativePos, kShift, dShiftX_, dShiftY_);
	CUDACHECK(cudaGetLastError());
	CUDACHECK(cudaDeviceSynchronize());
}

}