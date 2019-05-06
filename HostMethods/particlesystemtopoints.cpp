#include "hostmethods.hpp"

namespace hostMethod {
	void particleSystemToPoints(
		float* points, // 输出 起始位置 
		float* colors, // 输出 起始颜色
		float* sizes, // 输出 起始尺寸
		size_t* groupStarts, // 输出 此方法将dStartFrames拷贝过来
		const size_t* startFrames, // 每一组粒子的起始帧
		const size_t* lifeTime, // 每一组粒子的寿命
		size_t nGroups, // 总计的粒子组数
		const float* directions, // 每一组粒子的方向
		const float* centrifugalPos, // 每一组粒子的初始速度
		const float* startPoses, // 每一帧粒子生成时候的离心位置
		size_t currFrame, // 当前帧数
		size_t nFrames, // 总帧数
		const float* colorMatrix, // 颜色随帧数变化的矩阵
		const float* sizeMatrix, // 尺寸随帧数变化的矩阵
		float time  // 两帧间隔的时间
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