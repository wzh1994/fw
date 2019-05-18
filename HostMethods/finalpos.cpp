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
				float colorRate = pow(colorDecay, j);
				float sizeRate = pow(sizeDecay, j);
				size_t idx = i * nFrames + j;
				colorMatrix[idx * 3] = startColors[3 * i] * colorRate;
				colorMatrix[idx * 3 + 1] = startColors[3 * i + 1] * colorRate;
				colorMatrix[idx * 3 + 2] = startColors[3 * i + 2] * colorRate;
				sizeMatrix[idx] = startSizes[i] * sizeRate;
			}
		}
	}

	void calcFinalPositionImpl(float* points, size_t nInterpolation,
			size_t frame, const size_t* groupOffsets, 
			const size_t* groupStarts, const size_t* startFrames,
			const float* xShiftMatrix, const float* yShiftMatrix,
			size_t shiftsize, size_t nGroups, size_t maxSize) {
		for (size_t bid = 0; bid < nGroups; ++bid) {
			float* basePtr = points + groupOffsets[bid] * 3;
			size_t numPointsThisGroup =
				groupOffsets[bid + 1] - groupOffsets[bid];
			for (size_t tid = 0; tid < maxSize; ++tid) {
				if (tid < numPointsThisGroup) {
					size_t start = startFrames[bid] * (nInterpolation + 1);
					size_t end = (startFrames[bid] + groupStarts[bid]) * (
						nInterpolation + 1) + tid;
					basePtr[3 * tid] += xShiftMatrix[start * shiftsize + end];
					basePtr[3 * tid + 1] +=
						yShiftMatrix[start * shiftsize + end];
				}
			}
		}
	}

	void calcFinalPosition(float* dPoints, size_t nGroups, size_t maxSize,
			size_t nInterpolation, size_t frame, const size_t* dGroupOffsets,
			const size_t* dGroupStarts, const size_t* dStartFrames,
			const float* dXShiftMatrix, const float* dYShiftMatrix,
			size_t shiftsize) {
		calcFinalPositionImpl(dPoints, nInterpolation, frame,
			dGroupOffsets, dGroupStarts, dStartFrames, dXShiftMatrix,
			dYShiftMatrix, shiftsize, nGroups, maxSize);
	}
}