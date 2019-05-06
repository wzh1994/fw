#include "kernel.h"
#include "kernels.h"
#include "cuda_runtime.h"
#include "corecrt_math.h"
#include <stdexcept>
#include <string>
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

__device__ float interpolationValue(
	float l, float r, size_t lOffset, size_t totalNum) {
	return l + static_cast<float>(lOffset) * (r - l) / static_cast<float>(totalNum);
}

__global__ void interpolationMatrix(
		float* array, size_t size, size_t nInterpolation) {
	float temp;
	size_t idx_x = threadIdx.x;
	size_t idx_y = threadIdx.y;
	float* basePtr = array + idx_x * size;
	size_t idx = idx_y / (nInterpolation + 1);
	size_t lOffset = idx_y % (nInterpolation + 1);

	if (lOffset == 0) {
		temp = basePtr[idx];
	} else {
		temp = interpolationValue(
			basePtr[idx], basePtr[idx + 1], lOffset, nInterpolation + 1);
	}

	deviceDebugPrint("%llu, %llu, %llu %llu, %f\n",
		idx_x, idx_y, idx_x * size, idx, temp);
	__syncthreads();

	array[idx_x * blockDim.y + idx_y] = temp;
}

__global__ void interpolationMatrixOut(float* arrayOut,
		const float* arrayIn, size_t size, size_t nInterpolation) {
	size_t bid = blockIdx.x;
	size_t tid = threadIdx.x;
	const float* basePtr = arrayIn + bid * size;
	size_t idx = tid / (nInterpolation + 1);
	size_t lOffset = tid % (nInterpolation + 1);

	if (lOffset == 0) {
		arrayOut[bid * blockDim.x + tid] = basePtr[idx];
	}
	else {
		arrayOut[bid * blockDim.x + tid] = interpolationValue(
			basePtr[idx], basePtr[idx + 1], lOffset, nInterpolation + 1);
	}
}

void interpolation(
		float* dArray, size_t nGroups, size_t size, size_t nInterpolation) {
	if (nGroups * (size + nInterpolation * (size - 1)) < kMmaxBlockDim) {
		dim3 dimBlock(nGroups, size + nInterpolation * (size - 1));
		interpolationMatrix << <1, dimBlock >> > (
			dArray, size, nInterpolation);
		CUDACHECK(cudaGetLastError());
	}
	else {
		if (size + nInterpolation * (size - 1) > kMmaxBlockDim) {
			throw std::runtime_error("Max result length allowed is "
				+ std::to_string(kMmaxBlockDim) + "!");
		}
		float* tempArray;
		cudaMallocAndCopy(tempArray, dArray, nGroups * size);
		size_t nBlockDim = size + nInterpolation * (size - 1);
		interpolationMatrixOut <<<nGroups, nBlockDim >>> (
			dArray, tempArray, size, nInterpolation);
		CUDACHECK(cudaGetLastError());
		cudaFreeAll(tempArray);
	}
}
__global__ void interpolationOffsets(size_t* dGroupOffsets,
		const size_t* dGroupOffsetsBrfore, size_t nInterpolation) {
	size_t tid = threadIdx.x;
	dGroupOffsets[tid + 1] = dGroupOffsets[tid + 1] * (nInterpolation + 1) -
		(tid + 1) * nInterpolation;
}
__global__ void interpolationPoints(float* points, float* colors, float* sizes,
		const float* pointsIn, const float* colorsIn, const float* sizesIn,
		const size_t* dGroupOffsetsBrfore, const size_t* dGroupOffsets,
		size_t nInterpolation) {
	size_t bidx = blockIdx.x;
	size_t tidx = threadIdx.x;

	const float* pPointsIn = pointsIn + 3 * dGroupOffsetsBrfore[bidx];
	const float* pColorsIn = colorsIn + 3 * dGroupOffsetsBrfore[bidx];
	const float* pSizesIn = sizesIn + dGroupOffsetsBrfore[bidx];

	__syncthreads();
	if (tidx < dGroupOffsets[bidx + 1] - dGroupOffsets[bidx]) {
		float* pPointsOut = points + 3 * dGroupOffsets[bidx];
		float* pColorsOut = colors + 3 * dGroupOffsets[bidx];
		float* pSizezOut = sizes + dGroupOffsets[bidx];

		size_t idx = tidx / (nInterpolation + 1);
		size_t lOffset = tidx % (nInterpolation + 1);
		if (lOffset == 0) {
			pPointsOut[tidx * 3] = pPointsIn[3 * idx];
			pPointsOut[tidx * 3 + 1] = pPointsIn[3 * idx + 1];
			pPointsOut[tidx * 3 + 2] = pPointsIn[3 * idx + 2];
			pColorsOut[tidx * 3] = pColorsIn[3 * idx];
			pColorsOut[tidx * 3 + 1] = pColorsIn[3 * idx + 1];
			pColorsOut[tidx * 3 + 2] = pColorsIn[3 * idx + 2];
			pSizezOut[tidx] = pSizesIn[idx];
		}
		else {
			pPointsOut[tidx * 3] = interpolationValue(
				pPointsIn[3 * idx],
				pPointsIn[3 * idx + 3], lOffset, nInterpolation + 1);
			pPointsOut[tidx * 3 + 1] = interpolationValue(
				pPointsIn[3 * idx + 1],
				pPointsIn[3 * idx + 4], lOffset, nInterpolation + 1);
			pPointsOut[tidx * 3 + 2] = interpolationValue(
				pPointsIn[3 * idx + 2],
				pPointsIn[3 * idx + 5], lOffset, nInterpolation + 1);
			pColorsOut[tidx * 3] = interpolationValue(
				pColorsIn[3 * idx],
				pColorsIn[3 * idx + 3], lOffset, nInterpolation + 1);
			pColorsOut[tidx * 3 + 1] = interpolationValue(
				pColorsIn[3 * idx + 1],
				pColorsIn[3 * idx + 4], lOffset, nInterpolation + 1);
			pColorsOut[tidx * 3 + 2] = interpolationValue(
				pColorsIn[3 * idx + 2],
				pColorsIn[3 * idx + 5], lOffset, nInterpolation + 1);
			pSizezOut[tidx] = interpolationValue(
				pSizesIn[idx],
				pSizesIn[idx + 1], lOffset, nInterpolation + 1);
		}
	}
}

void interpolation(
		float* dPoints, float* dColors, float* dSizes, size_t* dGroupOffsets,
		size_t nGroups, size_t maxSize, size_t nInterpolation) {
	float *dPointsTemp, *dColorsTemp, *dSizesTemp;
	size_t *dGroupOffsetsTemp;
	cudaMallocAndCopy(dPointsTemp, dPoints, 3 * nGroups * maxSize);
	cudaMallocAndCopy(dColorsTemp, dColors, 3 * nGroups * maxSize);
	cudaMallocAndCopy(dSizesTemp, dSizes, nGroups * maxSize);
	cudaMallocAndCopy(dGroupOffsetsTemp, dGroupOffsets, nGroups + 1);
	CUDACHECK(cudaDeviceSynchronize());

	/*printSplitLine("points");
	show(dPointsTemp, dGroupOffsets, nGroups, 3);
	printSplitLine("colors");
	show(dColorsTemp, dGroupOffsets, nGroups, 3);
	printSplitLine("sizes");
	show(dSizesTemp, dGroupOffsets, nGroups, 1);
	printSplitLine("end");

	printf("%llu %llu %llu\n",
		nGroups, maxSize, maxSize + nInterpolation * (maxSize - 1));*/
	interpolationOffsets << <1, nGroups >> > (
		dGroupOffsets, dGroupOffsetsTemp, nInterpolation);
	interpolationPoints <<<nGroups, maxSize * (nInterpolation + 1)>>> (
		dPoints, dColors, dSizes, dPointsTemp, dColorsTemp,
		dSizesTemp, dGroupOffsetsTemp, dGroupOffsets, nInterpolation);
	CUDACHECK(cudaGetLastError());
	cudaFreeAll(dPointsTemp, dColorsTemp, dSizesTemp, dGroupOffsetsTemp);
	CUDACHECK(cudaDeviceSynchronize());
}

}