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

__global__ void judge(float* dColors, float* dSizes, size_t* indices) {
	size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (fminf(fmaxf(fmaxf(dColors[3 * idx], dColors[3 * idx + 1]),
			dColors[3 * idx + 2]), dSizes[idx]) < 0.1f) {
		indices[idx] = 0;
	} else {
		indices[idx] = 1;
	}
	// printf("judging: %llu, %llu\n", idx, indices[idx]);
}

__global__ void getOffsets(size_t *indices, size_t size, size_t* dGroupOffsets) {
	size_t idx = threadIdx.x;
	if (idx == 0) {
		dGroupOffsets[0] = 0;
	}
	dGroupOffsets[idx + 1] = indices[(idx + 1) * size - 1];
	// printf("getOffsets: %llu, %llu\n", idx, dGroupOffsets[idx + 1]);
}

__global__ void getGroupFlag(size_t *judgement, size_t* groupFlag) {
	size_t tidx = threadIdx.x;
	size_t bidx = blockIdx.x;
	size_t idx = bidx * blockDim.x + tidx;
	__shared__ size_t sum[100];
	sum[tidx] = judgement[idx];
	for (int s = 1; s < blockDim.x; s *= 2) {
		if (tidx % (2 * s) == 0) {
			sum[tidx] += sum[tidx + s];
		}
		__syncthreads();
	}
	size_t flag = sum[0] > 1 ? 1 : 0;
	if (tidx == 0) {
		groupFlag[bidx] = flag;
	}
	judgement[idx] = judgement[idx] & flag;
	// printf("getGroupFlag: %llu, %llu, %llu\n", idx, groupFlag[bidx], judgement[idx]);
}
__global__ void compressData(float* dPoints, float* dColors, float* dSizes,
		size_t* judgement, size_t* indices) {
	size_t idx = threadIdx.x * blockDim.y + threadIdx.y;
	float x = dPoints[3 * idx];
	float y = dPoints[3 * idx + 1];
	float z = dPoints[3 * idx + 2];
	float r = dColors[3 * idx];
	float g = dColors[3 * idx + 1];
	float b = dColors[3 * idx + 2];
	float s = dSizes[idx];
	__syncthreads();
	if (judgement[idx]) {
		// printf("compressData: %llu, %u, %u\n", idx, threadIdx.x, threadIdx.y);
		size_t targetIdx = indices[idx] - 1;
		dPoints[3 * targetIdx] = x;
		dPoints[3 * targetIdx + 1] = y;
		dPoints[3 * targetIdx + 2] = z;
		dColors[3 * targetIdx] = r;
		dColors[3 * targetIdx + 1] = g;
		dColors[3 * targetIdx + 2] = b;
		dSizes[targetIdx] = s;
	}
}
__global__ void compressIndex(size_t* dGroupOffsets, size_t* dGroupStarts,
		size_t* groupFlag, size_t* groupPos, size_t* dNumGroup) {
	size_t idx = threadIdx.x;
	size_t offset = dGroupOffsets[idx + 1];
	size_t start = dGroupStarts[idx];
	__syncthreads();
	// printf("compressIndex-comp: %llu, %llu, %llu, %llu\n", idx, groupFlag[idx], groupPos[idx], offset);
	if (groupFlag[idx]) {
		dGroupOffsets[groupPos[idx]] = offset;
		dGroupStarts[groupPos[idx] - 1] = start;
	}

	// 求有效的组数
	__shared__ size_t sum[1000];
	sum[idx] = groupFlag[idx];
	// printf("compressIndex: %llu, %llu\n", idx, sum[idx]);
	for (int s = 1; s < blockDim.x; s *= 2) {
		if (idx % (2 * s) == 0) {
			sum[idx] += sum[idx + s];
		}
		__syncthreads();
	}
	if (idx == 0) {
		dNumGroup[0] = sum[0];
		// printf("compressIndex done: %llu\n", dNumGroup[0]);
	}
}

size_t compress(float* dPoints, float* dColors, float* dSizes, size_t nGroups,
		size_t size, size_t* dGroupOffsets, size_t* dGroupStarts) {
	dim3 dimBlock(nGroups, size);
	size_t *indices, *judgement, *groupFlag, *groupPos, *dNumGroup;
	cudaMalloc(&judgement, nGroups * size * sizeof(size_t));
	cudaMalloc(&indices, nGroups * size * sizeof(size_t));
	cudaMalloc(&groupFlag, nGroups * sizeof(size_t));
	cudaMalloc(&groupPos, nGroups * sizeof(size_t));
	cudaMalloc(&dNumGroup, sizeof(size_t));

	judge<<<nGroups, size >>>(dColors, dSizes, judgement);
	getGroupFlag<<<nGroups, size>>>(judgement, groupFlag);
	argFirstNoneZero(judgement, dGroupStarts, nGroups, size);
	cuSum(indices, judgement, nGroups * size);
	// printf("\n");
	cuSum(groupPos, groupFlag, nGroups);
	// printf("\n");
	getOffsets<<<1, nGroups>>>(indices, size, dGroupOffsets);
	compressData<<<1, dimBlock >>>(dPoints, dColors, dSizes, judgement, indices);
	compressIndex<<<1, nGroups>>>(dGroupOffsets, dGroupStarts,
		groupFlag, groupPos, dNumGroup);
	size_t numGroup;
	cudaMemcpy(&numGroup, dNumGroup, sizeof(size_t), cudaMemcpyDeviceToHost);
	// printf("%llu\n", numGroup);

	cudaFree(judgement);
	cudaFree(indices);
	cudaFree(groupFlag);
	cudaFree(groupPos);
	cudaFree(dNumGroup);
	return numGroup;
}