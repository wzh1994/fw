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

	// �˷������ڶԵ��λ�ã���ɫ���ߴ���в�ֵ��ÿ�����������һ��
	void interpolation(
		float* dPoints, // ����&��� ���ӵ�λ��
		float* dColors, // ����&��� ���ӵ���ɫ
		float* dSizes, // ����&��� ���ӵĳߴ�
		size_t* dGroupOffsets, // ����&��� ��ֵ��ÿ������λ�õ�ƫ��
		size_t nGroups, // ��������
		size_t maxSize, // ��ֵ֮ǰ��ÿ���������ĸ�����һ��Ϊ֡��
		size_t nInterpolation // ÿ��������֮��������������
	) {
		
	}
}