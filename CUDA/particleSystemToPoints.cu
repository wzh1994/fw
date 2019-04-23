#include "kernel.h"
#include "kernels.h"
#include "cuda_runtime.h"
#include "corecrt_math.h"
#include "utils.h"
#include "test.h"

// 为了让__syncthreads()通过语法检查
#ifndef __CUDACC__
#define __CUDACC__
#endif // !__CUDACC__
#include "device_launch_parameters.h"
#include "device_functions.h"
#include <cstdio>

__global__ void getColorAndSizeMatrix(
		const float* startColors, const float* startSizes, float colorDecay,
		float sizeDecay, float* dColorMatrix, float* dSizeMatrix) {
	size_t bidx = blockIdx.x;
	size_t tidx = threadIdx.x;
	size_t idx = bidx * blockDim.x + tidx;
	
	/*在没有找到更合适的下降函数之前，暂定为指数级别的下降*/
	float colorRate = powf(colorDecay, tidx);
	float sizeRate = powf(sizeDecay, tidx);

	deviceDebugPrint(
		"getColorAndSizeMatrix: %llu, %llu, %llu: %f, %f - %f, %f\n",
		bidx, tidx, idx, colorRate, sizeRate,
		startColors[3 * bidx], startSizes[bidx]);

	dColorMatrix[3 * idx] = startColors[3 * bidx] * colorRate;
	dColorMatrix[3 * idx + 1] = startColors[3 * bidx + 1] * colorRate;
	dColorMatrix[3 * idx + 2] = startColors[3 * bidx + 2] * colorRate;
	dSizeMatrix[idx] = startSizes[bidx] * sizeRate;
}

void getColorAndSizeMatrix(
		const float* startColors, const float* startSizes,
		/*在没有找到更合适的下降函数之前，暂定为指数级别的下降*/
		size_t nFrames, float colorDecay, float sizeDecay,
		float* dColorMatrix, float* dSizeMatrix) {
	float *dStartColors, *dStartSizes;
	cudaMallocAndCopy(dStartColors, startColors, 3 * nFrames);
	cudaMallocAndCopy(dStartSizes, startSizes, nFrames);
	debugShow(dStartColors, 3 * nFrames);
	debugShow(dStartSizes, nFrames);
	getColorAndSizeMatrix<<<nFrames, nFrames>>>(dStartColors, dStartSizes,
		colorDecay, sizeDecay, dColorMatrix, dSizeMatrix);
	CUDACHECK(cudaGetLastError());
	cudaFreeAll(dStartColors, dStartSizes);
}

__global__ void particleSystemToPoints(
		float* points, float* colors, float* sizes, size_t* groupStarts,
		const size_t* startFrames, const float* directions, 
		const float* speeds, const float* poses, size_t currFrame,
		const float* colorMatrix, const float* sizeMatrix, float time) {
	size_t bid = blockIdx.x;
	size_t tid = threadIdx.x;
	size_t idx = bid * blockDim.x + tid;
	if (tid == 0) {
		groupStarts[bid] = startFrames[bid];
	}
	ll startFrame = static_cast<ll>(startFrames[bid]) + static_cast<ll>(tid);
	ll existFrame = static_cast<ll>(currFrame) - startFrame;
	size_t mIdx = tid * blockDim.x + existFrame;
	if (existFrame >= 0) {
		points[3 * idx] = poses[bid * 3] + directions[bid * 3] *
			static_cast<float>(existFrame) * speeds[bid];
		points[3 * idx + 1] = poses[bid * 3 + 1] + directions[bid * 3 + 1] *
			static_cast<float>(existFrame) * speeds[bid];
		points[3 * idx + 2] = poses[bid * 3 + 2] + directions[bid * 3 + 2] *
			static_cast<float>(existFrame) * speeds[bid];
		colors[3 * idx] = colorMatrix[3 * mIdx];
		colors[3 * idx + 1] = colorMatrix[3 * mIdx + 1];
		colors[3 * idx + 2] = colorMatrix[3 * mIdx + 2];
		sizes[idx] = sizeMatrix[mIdx];
	} else {
		points[3 * idx] = 0;
		points[3 * idx + 1] = 0;
		points[3 * idx + 2] = 0;
		colors[3 * idx] = 0;
		colors[3 * idx + 1] = 0;
		colors[3 * idx + 2] = 0;
		sizes[idx] = 0;
	}
}

void particleSystemToPoints(
		float* dPoints, float* dColors, float* dSizes, size_t* dGroupStarts,
		const size_t* dStartFrames, size_t nGroups, const float* dDirections,
		const float* dSpeeds, const float* dStartPoses, size_t currFrame,
		size_t nFrames, const float* dColorMatrix,
		const float* dSizeMatrix, float time) {
	particleSystemToPoints<<<nGroups, nFrames>>>(
		dPoints, dColors, dSizes, dGroupStarts, dStartFrames, dDirections,
		dSpeeds, dStartPoses, currFrame, dColorMatrix, dSizeMatrix, time);
	CUDACHECK(cudaGetLastError());
}