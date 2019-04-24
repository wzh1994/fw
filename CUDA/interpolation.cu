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

__global__ void interpolationPoints(float* points, float* colors, float* sizes,
		const float* pointsIn, const float* colorsIn, const float* sizesIn,
		size_t* dGroupOffsets, size_t count) {
	float x, y, z, s, r, g, b;
	size_t bidx = blockIdx.x;
	size_t tidx = threadIdx.x;
	size_t groupNum = dGroupOffsets[bidx + 1] - dGroupOffsets[bidx];
	if (tidx >= groupNum * (count + 1) - count)
		return;
	size_t offset = dGroupOffsets[bidx];
	size_t idx = tidx / (count + 1);
	size_t lOffset = tidx % (count + 1);

	if (lOffset == 0) {
		x = pointsIn[3 * (offset + idx)];
		y = pointsIn[3 * (offset + idx) + 1];
		z = pointsIn[3 * (offset + idx) + 2];
		r = colorsIn[3 * (offset + idx)];
		g = colorsIn[3 * (offset + idx) + 1];
		b = colorsIn[3 * (offset + idx) + 2];
		s = sizesIn[offset + idx];
	} else {
		x = interpolationValue(pointsIn[3 * (offset + idx)],
			pointsIn[3 * (offset + idx + 1)], lOffset, count + 1);
		y = interpolationValue(pointsIn[3 * (offset + idx) + 1],
			pointsIn[3 * (offset + idx + 1) + 1], lOffset, count + 1);
		z = interpolationValue(pointsIn[3 * (offset + idx) + 2],
			pointsIn[3 * (offset + idx + 1) + 2], lOffset, count + 1);
		r = interpolationValue(colorsIn[3 * (offset + idx)],
			colorsIn[3 * (offset + idx + 1)], lOffset, count + 1);
		g = interpolationValue(colorsIn[3 * (offset + idx) + 1],
			colorsIn[3 * (offset + idx + 1) + 1], lOffset, count + 1);
		b = interpolationValue(colorsIn[3 * (offset + idx) + 2],
			colorsIn[3 * (offset + idx + 1) + 2], lOffset, count + 1);
		s = interpolationValue(sizesIn[offset + idx],
			sizesIn[offset + idx + 1], lOffset, count + 1);
	}
	//printf("%llu, %llu, %llu, %llu, %llu, %f\n", idx_x, idx_y, offset, lOffset, idx, s);
	__syncthreads();
	size_t resultOffset = dGroupOffsets[bidx] * (count + 1) - bidx * count;
	points[(resultOffset + tidx) * 3] = x;
	points[(resultOffset + tidx) * 3 + 1] = y;
	points[(resultOffset + tidx) * 3 + 2] = z;
	colors[(resultOffset + tidx) * 3] = r;
	colors[(resultOffset + tidx) * 3 + 1] = g;
	colors[(resultOffset + tidx) * 3 + 2] = b;
	sizes[resultOffset + tidx] = s;
	if (tidx == 0) {
		dGroupOffsets[bidx + 1] =
			dGroupOffsets[bidx + 1] * (count + 1) - (bidx + 1) * count;
	}
}

void interpolation(float* dPoints, float* dColors, float* dSizes,
		size_t* dGroupOffsets, size_t nGroups, size_t maxSize, size_t count) {
	float *dPointsTemp, *dColorsTemp, *dSizesTemp;
	cudaMallocAndCopy(dPointsTemp, dPoints, 3 * nGroups * maxSize);
	cudaMallocAndCopy(dColorsTemp, dColors, 3 * nGroups * maxSize);
	cudaMallocAndCopy(dSizesTemp, dSizes, nGroups * maxSize);
	interpolationPoints << <nGroups, maxSize + count * (maxSize - 1) >> > (
		dPoints, dColors, dSizes, dPointsTemp, dColorsTemp,
		dSizesTemp, dGroupOffsets, count);
	CUDACHECK(cudaGetLastError());
}