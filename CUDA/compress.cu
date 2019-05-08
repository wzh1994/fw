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
namespace cudaKernel {

__global__ void judge(float* dColors, float* dSizes, size_t* indices) {
	size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (fmaxf(fmaxf(dColors[3 * idx], dColors[3 * idx + 1]),
		dColors[3 * idx + 2]) < 0.1f || dSizes[idx] < 0.002f) {
		indices[idx] = 0;
	}
	else {
		indices[idx] = 1;
	}
	deviceDebugPrint("judging: %llu, %llu\n",
		idx, indices[idx]);
}

__global__ void getOffsets(
	const size_t *indices, size_t size, size_t* dGroupOffsets) {
	size_t idx = threadIdx.x;
	if (idx == 0) {
		dGroupOffsets[0] = 0;
	}
	dGroupOffsets[idx + 1] = indices[(idx + 1) * size - 1];
	deviceDebugPrint("getOffsets: %llu, %llu\n",
		idx, dGroupOffsets[idx + 1]);
}

__global__ void getGroupFlag(size_t *judgement, size_t* groupFlag) {
	size_t tidx = threadIdx.x;
	size_t bidx = blockIdx.x;
	size_t idx = bidx * blockDim.x + tidx;
	__shared__ size_t sum[1024];
	sum[tidx] = judgement[idx];
	for (int s = 1; s < blockDim.x; s *= 2) {
		if (tidx % (2 * s) == 0) {
			sum[tidx] += sum[tidx + s];
		}
		__syncthreads();
	}
	size_t flag = (sum[0] > 0);
	if (tidx == 0) {
		groupFlag[bidx] = flag;
	}
	judgement[idx] = judgement[idx] & flag;
	deviceDebugPrint("getGroupFlag: %llu, %llu, %llu\n",
		idx, groupFlag[bidx], judgement[idx]);
}

__global__ void getGroupFlag(size_t *judgement, size_t* groupFlag,
		float rate, curandState *devStates) {
	size_t tidx = threadIdx.x;
	size_t bidx = blockIdx.x;
	size_t idx = bidx * blockDim.x + tidx;
	__shared__ size_t sum[1024];
	sum[tidx] = judgement[idx];
	for (int s = 1; s < blockDim.x; s *= 2) {
		if (tidx % (2 * s) == 0) {
			sum[tidx] += sum[tidx + s];
		}
		__syncthreads();
	}
	float r = curand_uniform(devStates + bidx);
	size_t flag = static_cast<size_t>((sum[0] > 0) && (r < rate));
	if (tidx == 0) {
		groupFlag[bidx] = flag;
		deviceDebugPrint("%llu: %llu %llu %f\n", bidx, sum[0], flag, r);
	}
	judgement[idx] = judgement[idx] & flag;
	deviceDebugPrint("getGroupFlag: %llu, %llu, %llu\n",
		idx, groupFlag[bidx], judgement[idx]);
}

__global__ void compressData(float* points, float* colors, float* sizes,
	const float* pointsIn, const float* colorsIn, const float* sizesIn,
	size_t* judgement, size_t* indices) {
	size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (judgement[idx]) {
		deviceDebugPrint("compressData: %llu, %u, %u\n",
			idx, threadIdx.x, threadIdx.y);
		size_t targetIdx = indices[idx] - 1;
		points[3 * targetIdx] = pointsIn[3 * idx];
		points[3 * targetIdx + 1] = pointsIn[3 * idx + 1];
		points[3 * targetIdx + 2] = pointsIn[3 * idx + 2];
		colors[3 * targetIdx] = colorsIn[3 * idx];
		colors[3 * targetIdx + 1] = colorsIn[3 * idx + 1];
		colors[3 * targetIdx + 2] = colorsIn[3 * idx + 2];
		sizes[targetIdx] = sizesIn[idx];
	}
}
__global__ void compressIndex(size_t* dGroupOffsets, size_t* dGroupStarts,
	const size_t* groupFlag, const size_t* groupPos, size_t* dNumGroup) {
	size_t idx = threadIdx.x;
	size_t offset = dGroupOffsets[idx + 1];
	size_t start = dGroupStarts[idx];
	__syncthreads();
	deviceDebugPrint("compressIndex-comp: %llu, %llu, %llu, %llu\n",
		idx, groupFlag[idx], groupPos[idx], offset);
	if (groupFlag[idx]) {
		dGroupOffsets[groupPos[idx]] = offset;
		dGroupStarts[groupPos[idx] - 1] = start;
	}

	// 求有效的组数
	__shared__ size_t sum[1000];
	sum[idx] = groupFlag[idx];
	deviceDebugPrint("compressIndex: %llu, %llu\n", idx, sum[idx]);
	for (int s = 1; s < blockDim.x; s *= 2) {
		if (idx % (2 * s) == 0 && idx + s < blockDim.x) {
			sum[idx] += sum[idx + s];
		}
		__syncthreads();
	}
	if (idx == 0) {
		dNumGroup[0] = sum[0];
		deviceDebugPrint("compressIndex done: %llu\n", dNumGroup[0]);
	}
}

size_t compress(float* dPoints, float* dColors, float* dSizes,
		size_t nGroups, size_t size, size_t* dGroupOffsets,
		size_t* dGroupStarts, float rate, curandState* devStates) {
	size_t *indices, *judgement, *groupFlag, *groupPos, *dNumGroup;
	cudaMallocAlign(&judgement, nGroups * size * sizeof(size_t));
	cudaMallocAlign(&indices, nGroups * size * sizeof(size_t));
	cudaMallocAlign(&groupFlag, nGroups * sizeof(size_t));
	cudaMallocAlign(&groupPos, nGroups * sizeof(size_t));
	cudaMallocAlign(&dNumGroup, sizeof(size_t));
	float *dPointsTemp, *dColorsTemp, *dSizesTemp;
	cudaMallocAndCopy(dPointsTemp, dPoints, 3 * nGroups * size);
	cudaMallocAndCopy(dColorsTemp, dColors, 3 * nGroups * size);
	cudaMallocAndCopy(dSizesTemp, dSizes, nGroups * size);

	judge << <nGroups, size >> > (dColors, dSizes, judgement);
	CUDACHECK(cudaGetLastError());

	if (rate < 1.0f) {
		FW_ASSERT(devStates != nullptr);
		getGroupFlag << <nGroups, size >> > (
			judgement, groupFlag, rate, devStates);
	} else {
		getGroupFlag << <nGroups, size >> > (judgement, groupFlag);
	}
	CUDACHECK(cudaGetLastError());
	argFirstNoneZero(judgement, dGroupStarts, nGroups, size);

	cuSum(indices, judgement, size * nGroups);
	cuSum(groupPos, groupFlag, nGroups);
	getOffsets << <1, nGroups >> > (indices, size, dGroupOffsets);
	CUDACHECK(cudaGetLastError());
	compressData << <nGroups, size >> > (dPoints, dColors, dSizes,
		dPointsTemp, dColorsTemp, dSizesTemp, judgement, indices);
	CUDACHECK(cudaGetLastError());
	compressIndex << <1, nGroups >> > (dGroupOffsets, dGroupStarts,
		groupFlag, groupPos, dNumGroup);
	CUDACHECK(cudaGetLastError());
	// printSplitLine();
	size_t numGroup;
	CUDACHECK(cudaMemcpy(
		&numGroup, dNumGroup, sizeof(size_t), cudaMemcpyDeviceToHost));

	cudaFreeAll(dPointsTemp, dColorsTemp, dSizesTemp, judgement,
		indices, groupFlag, groupPos, dNumGroup);
	return numGroup;
}

}