#include "hostmethods.hpp"
#include "utils.h"
#include <iostream>


namespace hostMethod {
namespace {
template <typename T>
T max(T a, T b) {
	return a > b ? a : b;
}

void judge(float* dColors, float* dSizes,
		size_t* indices, size_t nGroups, size_t size) {
	for (size_t i = 0; i < nGroups; ++i) {
		for (size_t j = 0; j < size; ++j) {
			size_t idx = i * size + j;
			if (max(max(dColors[3 * idx], dColors[3 * idx + 1]),
				dColors[3 * idx + 2]) < 0.1f || dSizes[idx] < 0.002f) {
				indices[idx] = 0;
			}
			else {
				indices[idx] = 1;
			}
		}
	}
}

void getGroupFlag(size_t *judgement,
		size_t* groupFlag, size_t nGroups, size_t size) {
	for (size_t i = 0; i < nGroups; ++i) {
		size_t sum = 0;
		for (size_t j = 0; j < size; ++j) {
			sum += judgement[i * size + j];
		}
		size_t flag = sum > 0 ? 1 : 0;
		groupFlag[i] = flag;
		for (size_t j = 0; j < size; ++j) {
			judgement[i * size + j] &= flag;
		}
	}
}

void getGroupFlag(size_t *judgement, size_t* groupFlag,
		size_t nGroups, size_t size, float rate) {
	for (size_t i = 0; i < nGroups; ++i) {
		size_t sum = 0;
		for (size_t j = 0; j < size; ++j) {
			sum += judgement[i * size + j];
		}
		size_t flag = sum > 0 && ((float)rand() / (float)RAND_MAX) < rate;
		groupFlag[i] = flag;
		for (size_t j = 0; j < size; ++j) {
			judgement[i * size + j] &= flag;
		}
	}
}

void getOffsets(const size_t *indices, size_t size,
		size_t* dGroupOffsets, size_t nGroups) {
	dGroupOffsets[0] = 0;
	for (size_t i = 0; i < nGroups; ++i) {
		dGroupOffsets[i + 1] = indices[(i + 1) * size - 1];
	}
}

void compressData(float* points, float* colors, float* sizes,
	const float* pointsIn, const float* colorsIn, const float* sizesIn,
	size_t* judgement, size_t* indices, size_t nGroups, size_t size) {
	for (size_t i=0;i<nGroups;++i){
		for (size_t j = 0; j < size; ++j) {
			size_t idx = i * size + j;
			if (judgement[idx]) {
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
	}
}

size_t compressIndex(size_t* dGroupOffsets, size_t* dGroupStarts,
	const size_t* groupFlag, const size_t* groupPos, size_t nGroups) {
	size_t sum = 0;
	for (size_t idx = 0; idx < nGroups; ++idx) {
		size_t offset = dGroupOffsets[idx + 1];
		size_t start = dGroupStarts[idx];
		if (groupFlag[idx]) {
			++sum;
			dGroupOffsets[groupPos[idx]] = offset;
			dGroupStarts[groupPos[idx] - 1] = start;
		}
	}
	return sum;
}

}
size_t compress(
	float* dPoints, // 输入&输出 粒子的位置
	float* dColors, // 输入&输出 粒子的颜色
	float* dSizes, // 输入&输出 粒子的尺寸
	size_t nGroups, // 粒子组数
	size_t size, // 每组粒子的个数，此方法输入的每组粒子数量相同
	size_t* dGroupOffsets, // 输出 压缩后每组粒子位置相对于起始位置的偏移
	size_t* dGroupStarts, // 输出 压缩后的每组粒子的起始帧
	float rate,
	curandState* devStates
) {
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

	judge(dColors, dSizes, judgement, nGroups, size);
	if (rate < 1.0f) {
		getGroupFlag(judgement, groupFlag, nGroups, size, rate);
	} else {
		getGroupFlag(judgement, groupFlag, nGroups, size);
	}

	argFirstNoneZero(judgement, dGroupStarts, nGroups, size);
	cuSum(indices, judgement, size * nGroups);
	cuSum(groupPos, groupFlag, nGroups);
	getOffsets(indices, size, dGroupOffsets, nGroups);
	compressData(dPoints, dColors, dSizes, dPointsTemp, dColorsTemp,
		dSizesTemp, judgement, indices, nGroups, size);
	return compressIndex(dGroupOffsets, dGroupStarts,
		groupFlag, groupPos, nGroups);
}
}