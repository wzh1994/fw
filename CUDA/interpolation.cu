#include "kernel.h"
#include "kernels.h"
#include "cuda_runtime.h"
#include "corecrt_math.h"
#include <stdexcept>
#include <string>
#include "utils.h"


// 为了让__syncthreads()通过语法检查
#ifndef __CUDACC__
#define __CUDACC__
#endif // !__CUDACC__
#include "device_launch_parameters.h"
#include "device_functions.h"

#include <cstdio>

__device__ float interpolationValue(
		float l, float r, size_t lOffset, size_t totalNum) {
	return l + lOffset * (r - l) / static_cast<float>(totalNum);
}

__global__ void interpolationMatrix(float* array, size_t size, size_t count) {
	float temp;
	size_t idx_x = threadIdx.x;
	size_t idx_y = threadIdx.y;
	float* basePtr = array + idx_x * size;
	size_t idx = idx_y / (count + 1);
	size_t lOffset = idx_y % (count + 1);

	if (lOffset == 0) {
		temp = basePtr[idx];
	} else {
		temp = interpolationValue(
			basePtr[idx], basePtr[idx + 1], lOffset, count + 1);
	}
	// printf("%llu, %llu, %llu %llu, %f\n", idx_x, idx_y, idx_x * size, idx, temp);
	__syncthreads();

	array[idx_x * blockDim.y + idx_y] = temp;
}

__global__ void interpolationMatrixOut(
		float* arrayOut, const float* arrayIn, size_t size, size_t count) {
	size_t bid = blockIdx.x;
	size_t tid = threadIdx.x;
	const float* basePtr = arrayIn + bid * size;
	size_t idx = tid / (count + 1);
	size_t lOffset = tid % (count + 1);

	if (lOffset == 0) {
		arrayOut[bid * blockDim.x + tid] = basePtr[idx];
	} else {
		arrayOut[bid * blockDim.x + tid] = interpolationValue(
			basePtr[idx], basePtr[idx + 1], lOffset, count + 1);
	}
}

void interpolation(float* dArray, size_t nGroups, size_t size, size_t count) {
	if (nGroups * (size + count * (size - 1)) < kMmaxBlockDim) {
		dim3 dimBlock(nGroups, size + count * (size - 1));
		interpolationMatrix << <1, dimBlock >> > (dArray, size, count);
		CUDACHECK(cudaGetLastError());
	} else {
		if (size + count * (size - 1) > kMmaxBlockDim) {
			throw std::runtime_error("Max result length allowed is "
				+ std::to_string(kMmaxBlockDim) + "!");
		}
		float* tempArray;
		cudaMallocAndCopy(tempArray, dArray, nGroups * size);
		interpolationMatrixOut << <nGroups, size + count * (size - 1) >> > (
			dArray, tempArray, size, count);
		CUDACHECK(cudaGetLastError());
		CUDACHECK(cudaFree(tempArray));
	}
}

__global__ void interpolationPoints(float* dPoints, float* dColors, float* dSizes,
		size_t* dGroupOffsets, size_t count) {
	float x, y, z, s, r, g, b;
	size_t idx_x = threadIdx.x;
	size_t idx_y = threadIdx.y;
	size_t groupNum = dGroupOffsets[idx_x + 1] - dGroupOffsets[idx_x];
	if (idx_y >= groupNum * (count + 1) - count)
		return;
	size_t offset = dGroupOffsets[idx_x];
	size_t idx = idx_y / (count + 1);
	size_t lOffset = idx_y % (count + 1);

	if (lOffset == 0) {
		x = dPoints[3 * (offset + idx)];
		y = dPoints[3 * (offset + idx) + 1];
		z = dPoints[3 * (offset + idx) + 2];
		r = dColors[3 * (offset + idx)];
		g = dColors[3 * (offset + idx) + 1];
		b = dColors[3 * (offset + idx) + 2];
		s = dSizes[offset + idx];
	} else {
		x = interpolationValue(dPoints[3 * (offset + idx)],
				dPoints[3 * (offset + idx + 1)], lOffset, count + 1);
		y = interpolationValue(dPoints[3 * (offset + idx) + 1],
				dPoints[3 * (offset + idx + 1) + 1], lOffset, count + 1);
		z = interpolationValue(dPoints[3 * (offset + idx) + 2],
				dPoints[3 * (offset + idx + 1) + 2], lOffset, count + 1);
		r = interpolationValue(dColors[3 * (offset + idx)],
				dColors[3 * (offset + idx + 1)], lOffset, count + 1);
		g = interpolationValue(dColors[3 * (offset + idx) + 1],
				dColors[3 * (offset + idx + 1) + 1], lOffset, count + 1);
		b = interpolationValue(dColors[3 * (offset + idx) + 2],
				dColors[3 * (offset + idx + 1) + 2], lOffset, count + 1);
		s = interpolationValue(dSizes[offset + idx],
				dSizes[offset + idx + 1], lOffset, count + 1);
	}
	//printf("%llu, %llu, %llu, %llu, %llu, %f\n", idx_x, idx_y, offset, lOffset, idx, s);
	__syncthreads();
	size_t resultOffset = dGroupOffsets[idx_x] * (count + 1) - idx_x * count;
	dPoints[(resultOffset + idx_y) * 3] = x;
	dPoints[(resultOffset + idx_y) * 3 + 1] = y;
	dPoints[(resultOffset + idx_y) * 3 + 2] = z;
	dColors[(resultOffset + idx_y) * 3] = r;
	dColors[(resultOffset + idx_y) * 3 + 1] = g;
	dColors[(resultOffset + idx_y) * 3 + 2] = b;
	dSizes[resultOffset + idx_y] = s;
	if (idx_y == 0) {
		dGroupOffsets[idx_x + 1] =
			dGroupOffsets[idx_x + 1] * (count + 1) - (idx_x + 1) * count;
	}
}

void interpolation(float* dPoints, float* dColors, float* dSizes,
		size_t* dGroupOffsets, size_t nGroups, size_t maxSize, size_t count) {
	dim3 dimBlock(nGroups, maxSize + count * (maxSize - 1));
	interpolationPoints << <1, dimBlock >> > (
		dPoints, dColors, dSizes, dGroupOffsets, count);
	CUDACHECK(cudaGetLastError());
}