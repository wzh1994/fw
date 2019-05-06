#include "kernel.h"
#include "kernels.h"
#include "cuda_runtime.h"
#include "corecrt_math.h"
#include <stdexcept>
#include <memory>
#include <mutex>
#include "utils.h"
#include "test.h"

// 为了让__syncthreads()通过语法检查
#ifndef __CUDACC__
#define __CUDACC__
#endif // !__CUDACC__
#include "device_launch_parameters.h"
#include "device_functions.h"
#include <cstdio>

namespace cudaKernel {

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
	if (nFrames > kMmaxBlockDim) {
		throw std::runtime_error("Max nFrames allowed is "
			+ std::to_string(kMmaxBlockDim) + "!");
	}
	getColorAndSizeMatrix << <nFrames, nFrames >> > (dStartColors, dStartSizes,
		colorDecay, sizeDecay, dColorMatrix, dSizeMatrix);
	CUDACHECK(cudaGetLastError());
	cudaFreeAll(dStartColors, dStartSizes);
}

void getColorAndSizeMatrixDevInput(
	const float* dStartColors, const float* dStartSizes,
	/*在没有找到更合适的下降函数之前，暂定为指数级别的下降*/
	size_t nFrames, float colorDecay, float sizeDecay,
	float* dColorMatrix, float* dSizeMatrix) {
	if (nFrames > kMmaxBlockDim) {
		throw std::runtime_error("Max nFrames allowed is "
			+ std::to_string(kMmaxBlockDim) + "!");
	}
	getColorAndSizeMatrix << <nFrames, nFrames >> > (dStartColors, dStartSizes,
		colorDecay, sizeDecay, dColorMatrix, dSizeMatrix);
	CUDACHECK(cudaGetLastError());
}

__global__ void particleSystemToPoints(
		float* points, float* colors, float* sizes, size_t* groupStarts,
		const size_t* startFrames, const size_t *lifeTime,
		const float* directions, const float* centrifugalPos,
		const float* poses, size_t currFrame, const size_t* colorAndSizeStarts,
		const float* colorMatrix, const float* sizeMatrix, float time) {
	size_t bid = blockIdx.x;
	size_t tid = threadIdx.x;
	size_t idx = bid * blockDim.x + tid;
	if (tid == 0) {
		groupStarts[bid] = startFrames[bid];
	}
	ll startFrame = static_cast<ll>(startFrames[bid]) + static_cast<ll>(tid);
	ll existFrame = static_cast<ll>(currFrame) - startFrame;
	size_t mIdx = (tid + colorAndSizeStarts[bid]) * blockDim.x + existFrame;
	if (existFrame >= 0 && startFrame <= lifeTime[bid]) {
		points[3 * idx] = poses[bid * 3] + directions[bid * 3] *
			centrifugalPos[tid];
		points[3 * idx + 1] = poses[bid * 3 + 1] + directions[bid * 3 + 1] *
			centrifugalPos[tid];
		points[3 * idx + 2] = poses[bid * 3 + 2] + directions[bid * 3 + 2] *
			centrifugalPos[tid];
		colors[3 * idx] = colorMatrix[3 * mIdx];
		colors[3 * idx + 1] = colorMatrix[3 * mIdx + 1];
		colors[3 * idx + 2] = colorMatrix[3 * mIdx + 2];
		sizes[idx] = sizeMatrix[mIdx];
	}
	else {
		points[3 * idx] = 0;
		points[3 * idx + 1] = 0;
		points[3 * idx + 2] = 0;
		colors[3 * idx] = 0;
		colors[3 * idx + 1] = 0;
		colors[3 * idx + 2] = 0;
		sizes[idx] = 0;
	}
}

void particleSystemToPoints(float* dPoints, float* dColors, float* dSizes,
	size_t* dGroupStarts, const size_t* dStartFrames,
	const size_t* dLifeTime, size_t nGroups, const float* dDirections,
	const float* dCentrifugalPos, const float* dStartPoses, size_t currFrame,
	size_t nFrames, const size_t* dColorAndSizeStarts,
	const float* dColorMatrix, const float* dSizeMatrix, float time) {
	if (nFrames > kMmaxBlockDim) {
		throw std::runtime_error("Max nFrames allowed is "
			+ std::to_string(kMmaxBlockDim) + "!");
	}
	particleSystemToPoints << <nGroups, nFrames >> > (dPoints, dColors,
		dSizes, dGroupStarts, dStartFrames, dLifeTime, 
		dDirections, dCentrifugalPos, dStartPoses,
		currFrame, dColorAndSizeStarts, dColorMatrix, dSizeMatrix, time);
	CUDACHECK(cudaGetLastError());
}

namespace {
struct CUDAPointerDeleter{
	template <typename T>
	void operator()(T* p) {
		// CUDACHECK(cudaFreeAll(p));
	}
};
}

void particleSystemToPoints(
		float* dPoints, float* dColors, float* dSizes,
		size_t* dGroupStarts, const size_t* dStartFrames,
		const size_t* dLifeTime, size_t nGroups,
		const float* dDirections, const float* dCentrifugalPos,
		const float* dStartPoses, size_t currFrame, size_t nFrames, 
		const float* dColorMatrix, const float* dSizeMatrix, float time) {
	static std::unique_ptr<size_t, CUDAPointerDeleter> zero;
	static std::once_flag inited;
	std::call_once(inited, [] {
		size_t* pZeros;
		cudaMallocAlign(&pZeros, 2000 * sizeof(size_t));
		CUDACHECK(cudaMemset(pZeros, 0, 2000*sizeof(size_t)));
		zero.reset(pZeros);
	});
	particleSystemToPoints(dPoints, dColors, dSizes, dGroupStarts, dStartFrames,
		dLifeTime, nGroups, dDirections, dCentrifugalPos, dStartPoses, currFrame,
		nFrames, zero.get(), dColorMatrix, dSizeMatrix, time);
}

}