#include "hostmethods.hpp"
#include "utils.h"
#include <iostream>
using namespace std;

namespace hostMethod {
namespace{

float interpolationValue(
		float l, float r, size_t lOffset, size_t totalNum) {
	return l + static_cast<float>(lOffset) * (r - l) /
		static_cast<float>(totalNum);
}

}

void interpolation(
		float* a, size_t nGroups, size_t size, size_t nInterpolation) {
	float* temp;
	cudaMallocAlign(&temp, nGroups * size * sizeof(float));
	memcpy(temp, a, nGroups * size * sizeof(float));
	size_t nPerGroup = size * (nInterpolation + 1) - nInterpolation;
	for (size_t i = 0; i < nGroups; ++i) {
		for (size_t j = 0; j < size - 1; ++j) {
			a[i * nPerGroup + j * (nInterpolation + 1)] = temp[i * size + j];
			for (size_t k = 0; k < nInterpolation; ++k) {
				a[i * nPerGroup + j * (nInterpolation + 1) + k] = interpolationValue(
					temp[i * size + j], temp[i * size + j + 1], k + 1, nInterpolation + 1);
			}
		}
		a[(i + 1) * nPerGroup - 1] = temp[(i + 1) * size - 1];
	}
}

namespace {

void interpolationOffsets(size_t* dGroupOffsets,
	const size_t* dGroupOffsetsBrfore, size_t nInterpolation, size_t nGroups) {
	for (size_t i = 0; i < nGroups; ++i) {
		dGroupOffsets[i + 1] = dGroupOffsets[i + 1] * (nInterpolation + 1) -
			(i + 1) * nInterpolation;
	}
}

void interpolationPoints(float* points, float* colors, float* sizes,
		const float* pointsIn, const float* colorsIn, const float* sizesIn,
		const size_t* dGroupOffsetsBrfore, const size_t* dGroupOffsets,
		size_t nInterpolation, size_t nGroups, size_t size) {
	for (size_t bidx = 0; bidx < nGroups; ++bidx) {
		for (size_t tidx = 0; tidx < size; ++tidx) {
			const float* pPointsIn = pointsIn + 3 * dGroupOffsetsBrfore[bidx];
			const float* pColorsIn = colorsIn + 3 * dGroupOffsetsBrfore[bidx];
			const float* pSizesIn = sizesIn + dGroupOffsetsBrfore[bidx];

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
	}
}

}

// 此方法用于对点的位置，颜色，尺寸进行插值，每组的数量允许不一致
void interpolation(
	float* dPoints, // 输入&输出 粒子的位置
	float* dColors, // 输入&输出 粒子的颜色
	float* dSizes, // 输入&输出 粒子的尺寸
	size_t* dGroupOffsets, // 输入&输出 插值后每组粒子位置的偏移
	size_t nGroups, // 粒子组数
	size_t maxSize, // 插值之前，每组粒子最多的个数，一般为帧数
	size_t nInterpolation // 每两个粒子之间插入的粒子数量
) {
	float *dPointsTemp, *dColorsTemp, *dSizesTemp;
	size_t *dGroupOffsetsTemp;
	cudaMallocAndCopy(dPointsTemp, dPoints, 3 * nGroups * maxSize);
	cudaMallocAndCopy(dColorsTemp, dColors, 3 * nGroups * maxSize);
	cudaMallocAndCopy(dSizesTemp, dSizes, nGroups * maxSize);
	cudaMallocAndCopy(dGroupOffsetsTemp, dGroupOffsets, nGroups + 1);
	interpolationOffsets(
		dGroupOffsets, dGroupOffsetsTemp, nInterpolation, nGroups);
	interpolationPoints(dPoints, dColors, dSizes, dPointsTemp, dColorsTemp,
		dSizesTemp, dGroupOffsetsTemp, dGroupOffsets, nInterpolation,
		nGroups, maxSize * (nInterpolation + 1));
	cudaFreeAll(dPointsTemp, dColorsTemp, dSizesTemp, dGroupOffsetsTemp);
}
}