#include "hostmethods.hpp"
#include "utils.h"

namespace hostMethod {

	float interpolationValue(
			float l, float r, size_t lOffset, size_t totalNum) {
		return l + static_cast<float>(lOffset) * (r - l) /
			static_cast<float>(totalNum);
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
		
	}
}