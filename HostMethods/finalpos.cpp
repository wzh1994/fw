#include "hostmethods.hpp"
#include <cmath>

namespace hostMethod {
	void getColorAndSizeMatrix(
		const float* startColors, // 输入 起始颜色
		const float* startSizes, // 输入 起始尺寸
		size_t nFrames, // 总计帧数
		float colorDecay, // 颜色衰减率
		float sizeDecay, // 尺寸衰减率
		float* colorMatrix, // 输出，颜色随帧数变化矩阵
		float* sizeMatrix // 输出，尺寸随帧数变化矩阵
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