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
namespace cudaKernel {

__global__ void calcFinalPosition(
		float* points, size_t nInterpolation, size_t frame, 
		const size_t* groupOffsets, const size_t* groupStarts,
		const size_t* startFrames, const float* xShiftMatrix,
		const float* yShiftMatrix, size_t shiftsize) {
	size_t bid = blockIdx.x;
	size_t tid = threadIdx.x;
	float* basePtr = points + groupOffsets[bid] * 3;
	size_t numPointsThisGroup = groupOffsets[bid + 1] - groupOffsets[bid];
	if (tid < numPointsThisGroup) {
		size_t start = startFrames[bid] * (nInterpolation + 1);
		size_t end = groupStarts[bid] * (nInterpolation + 1) + tid;
		basePtr[3 * tid] += xShiftMatrix[start * shiftsize + end];
		basePtr[3 * tid + 1] += yShiftMatrix[start * shiftsize + end];
		/*if (bid == 0) {
			printf("FinalPos: (%llu, %llu, %llu) : %llu, %llu, %f, %f\n",
				bid, tid, numPointsThisGroup, start, end,
				dXShiftMatrix[start * shiftsize + end],
				dXShiftMatrix[start * shiftsize + end]);
		}*/
	}
}

void calcFinalPosition(float* dPoints, size_t nGroups, size_t maxSize,
	size_t nInterpolation, size_t frame, const size_t* dGroupOffsets,
	const size_t* dGroupStarts, const size_t* dStartFrames,
	const float* dXShiftMatrix, const float* dYShiftMatrix, size_t shiftsize) {
	calcFinalPosition << <nGroups, maxSize >> > (
		dPoints, nInterpolation, frame, dGroupOffsets, dGroupStarts,
		dStartFrames, dXShiftMatrix, dYShiftMatrix, shiftsize);
	CUDACHECK(cudaGetLastError());
}
}