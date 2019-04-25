#ifndef FW_KERNEL_UTILS_KERNELS_HPP
#define FW_KERNEL_UTILS_KERNELS_HPP

#include "kernel.h"
#include "cuda_runtime.h"

/*
 * 本项目生成.lib文件供fw调用cuda相关的函数。
 * 本文件声明了所有kernel函数的接口
 * 使用本项目的kernel函数，需要引入本头文件
 * 仅本文件中的函数作为整个库的对外接口
 */

/*
 * 分配cuda上面的显存，以ALIGN对其
 */
namespace cudaKernel {

template <class T>
cudaError_t cudaMallocAlign(T** ptr, size_t size) {
	size = ceilAlign(size) * kernelAlign;
	return cudaMalloc(ptr, size);
};

/*
	* 基础通用数学方法
	*/
	// 填充操作
void fill(float* dArray, float data, size_t size);
void fill(size_t* dArray, size_t data, size_t size);
void fill(float* dArray, const float* data, size_t size, size_t step);
void fill(size_t* dArray, const size_t* data, size_t size, size_t step);

// 累加和操作 每组输入的最大数量不超过kMmaxBlockDim * kMmaxBlockDim
void cuSum(float* dOut, const float* dIn, size_t size, size_t numGroup = 1);
void cuSum(size_t* dOut, const size_t* dIn, size_t size, size_t numGroup = 1);

// 正则化
void normalize(float* vectors, size_t size);

// 缩放操作
void scale(float* dArray, float rate, size_t size);

// 归约操作
enum class ReduceOption {
	sum,
	min
};
void reduce(size_t *dMatrix, size_t* dResult,
	size_t nGroups, size_t size, ReduceOption op = ReduceOption::sum);
void reduce(float *dMatrix, float* dResult,
	size_t nGroups, size_t size, ReduceOption op = ReduceOption::sum);
void reduceMin(float *dMatrix, float* dResult, size_t nGroups, size_t size);
void reduceMin(size_t *dMatrix, size_t* dResult, size_t nGroups, size_t size);

// 累计 最大值操作
void cuMax(float* dOut, float* dIn, size_t size, size_t numGroup = 1);

// 求矩阵中每一行第一个非0元素的位置
void argFirstNoneZero(size_t* dMatrix, size_t* result,
	size_t nGroups, size_t size);

/*
	* 烟花相关方法
	*/

	// 给定某一方向上每一时刻的力，求出任一时刻生成的粒子在的在该力作用下的位移
	// size表示时刻的数量，count表示插值的数量, time表示时间间隔。
void calcShiftingByOutsideForce(
	float* dIn, size_t size, size_t count, float time = 0.08333333333f);

// 获取颜色和尺寸变化的矩阵 帧数不能超过kMmaxBlockDim
void getColorAndSizeMatrix(
	const float* startColors, const float* startSizes,
	size_t nFrames, float colorDecay, float sizeDecay,
	float* dColorMatrix, float* dSizeMatrix);

// 由N个粒子系统生成N组点，求出每个点对应的位置，颜色，尺寸
// 要求： nFrames不能大于kMmaxBlockDim
void particleSystemToPoints(
	float* dPoints, float* dColors, float* dSizes, size_t* dGroupStarts,
	const size_t* dStartFrames, size_t nGroups, const float* dDirections,
	const float* dSpeeds, const float* dStartPoses, size_t currFrame,
	size_t nFrames, const float* dColorMatrix,
	const float* dSizeMatrix, float time = 0.08333333333f);

// 对烟花的粒子进行空间压缩，除去其中的不可见粒子,
// nGroups和size不能超过kMmaxBlockDim
size_t compress(float* dPoints, float* dColors, float* dSizes,
	size_t nGroups, size_t size, size_t* dGroupOffsets, size_t* dGroupStarts);

// 插值算法
// 对N组相同长度的数组做插值，每组插值的结果长度不能超过kMmaxBlockDim
// 例如 有49帧的粒子， 每组插值15个， 则共有48 * 15 + 49个粒子
void interpolation(float* dArray, size_t nGroups, size_t size, size_t count);
void interpolation(
	float* dPoints, float* dColors, float* dSizes, size_t* dGroupOffsets,
	size_t nGroups, size_t maxSize, size_t nInterpolation);

// 计算最终每个点的位置
void calcFinalPosition(
	float* dPoints, size_t nGroups, size_t maxSize, size_t count,
	size_t frame, const size_t* dGroupOffsets, const size_t* dGroupStarts,
	const float* dXShiftMatrix, const float* dYShiftMatrix, size_t shiftsize);

// 把点连成线，生成这一条线上面的三角形面片
// 输入要求： 每条线上面的点为奇数；
// 应用场景： 生成的每两个点之间插入奇数（15）个点，以保证总数为奇数
size_t pointToLine(
	const float* dPointsIn, const float* dSizesIn, const float* dColorsIn,
	size_t maxSizePerGroup, size_t* const groupOffsets, size_t nGroups,
	float* buffer, uint32_t* dIndicesOut);
}
#endif