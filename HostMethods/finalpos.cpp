#include "hostmethods.hpp"
#include <cmath>

namespace hostMethod {
	void getColorAndSizeMatrix(
		const float* startColors, // ���� ��ʼ��ɫ
		const float* startSizes, // ���� ��ʼ�ߴ�
		size_t nFrames, // �ܼ�֡��
		float colorDecay, // ��ɫ˥����
		float sizeDecay, // �ߴ�˥����
		float* colorMatrix, // �������ɫ��֡���仯����
		float* sizeMatrix // ������ߴ���֡���仯����
	) {
		for (size_t i = 0; i < nFrames; ++i) {
			for (size_t j = 0; j < nFrames; ++j) {
				size_t idx = i * nFrames + j;
				colorMatrix[idx * 3] = pow(startColors[3 * i], j);
				colorMatrix[idx * 3 + 1] = pow(startColors[3 * i + 1], j);
				colorMatrix[idx * 3 + 2] = pow(startColors[3 * i + 2], j);
				sizeMatrix[idx] = pow(sizeMatrix[i], j);
			}
		}
	}
}