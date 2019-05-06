#include "hostmethods.hpp"

namespace hostMethod {
	void particleSystemToPoints(
		float* points, // ��� ��ʼλ�� 
		float* colors, // ��� ��ʼ��ɫ
		float* sizes, // ��� ��ʼ�ߴ�
		size_t* groupStarts, // ��� �˷�����dStartFrames��������
		const size_t* startFrames, // ÿһ�����ӵ���ʼ֡
		const size_t* lifeTime, // ÿһ�����ӵ�����
		size_t nGroups, // �ܼƵ���������
		const float* directions, // ÿһ�����ӵķ���
		const float* centrifugalPos, // ÿһ�����ӵĳ�ʼ�ٶ�
		const float* startPoses, // ÿһ֡��������ʱ�������λ��
		size_t currFrame, // ��ǰ֡��
		size_t nFrames, // ��֡��
		const float* colorMatrix, // ��ɫ��֡���仯�ľ���
		const float* sizeMatrix, // �ߴ���֡���仯�ľ���
		float time  // ��֡�����ʱ��
	) {
		static size_t zeros[2000]{0};
		for (size_t i = 0; i < nGroups; ++i) {
			groupStarts[i] = startFrames[i];
			for (size_t j = 0; j < nFrames; ++j) {
				size_t idx = i * nFrames + j;
				ll startFrame = static_cast<ll>(startFrames[i]) + static_cast<ll>(j);
				ll existFrame = static_cast<ll>(currFrame) - startFrame;
				size_t mIdx = (j + zeros[i]) * nFrames + existFrame;
				if (existFrame >= 0 && startFrame <= lifeTime[i]) {
					points[3 * idx] = startPoses[i * 3] + directions[i * 3] *
						centrifugalPos[j];
					points[3 * idx + 1] = startPoses[i * 3 + 1] + directions[i * 3 + 1] *
						centrifugalPos[j];
					points[3 * idx + 2] = startPoses[i * 3 + 2] + directions[i * 3 + 2] *
						centrifugalPos[j];
					colors[3 * idx] = colorMatrix[3 * mIdx];
					colors[3 * idx + 1] = colorMatrix[3 * mIdx + 1];
					colors[3 * idx + 2] = colorMatrix[3 * mIdx + 2];
					sizes[idx] = sizeMatrix[mIdx];
				}
				else {
					points[3 * idx] = 0;
					points[3 * idx + 1] = 0;
					points[3 * idx + 2] = 0;
					colors[3 * idx] = 0;
					colors[3 * idx + 1] = 0;
					colors[3 * idx + 2] = 0;
					sizes[idx] = 0;
				}
			}
		}
	}

}