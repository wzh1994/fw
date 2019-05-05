#ifndef FW_KERNEL_KERNELS_HPP
#define FW_KERNEL_KERNELS_HPP

#include "kernel.h"
#include "cuda_runtime.h"

/*
 * 本项目生成.lib文件供fw调用cuda相关的函数。
 * 本文件声明了所有kernel函数的接口
 * 使用本项目的kernel函数，需要引入本头文件
 * 仅本文件中的函数作为整个库的对外接口
 */

namespace cudaKernel {

static const float kFrameTime = 0.08333333333f;

/*
 * 分配cuda上面的显存，以ALIGN对其
 */
template <class T>
cudaError_t cudaMallocAlign(T** ptr, size_t size) {
	size = ceilAlign(size) * kernelAlign;
	return cudaMalloc(ptr, size);
};

/* ==================================
 * 基础通用数学方法
 * ==================================
 */

/* 
 * 填充操作
 */
void fill(float* dArray, float data, size_t size);
void fill(size_t* dArray, size_t data, size_t size);
void fill(float* dArray, const float* data, size_t size, size_t step);
void fill(size_t* dArray, const size_t* data, size_t size, size_t step);

/*
 * 扫描操作
 */
// 累加和操作 每组输入的最大数量不超过kMmaxBlockDim * kMmaxBlockDim
void cuSum(float* out, const float* in, size_t size, size_t numGroup = 1);
void cuSum(size_t* out, const size_t* in, size_t size, size_t numGroup = 1);

// 累计 最大值操作
void cuMax(float* out, float* in, size_t size, size_t numGroup = 1);

// 累计 乘积操作
void cuMul(float* out, float* in, size_t size, size_t numGroup = 1);

/*
 * 正则化
 */
void normalize(float* vectors, size_t size);

/*
 * 缩放
 */
void scale(float* dArray, float rate, size_t size);

/*
 * 归约
 */
enum class ReduceOption {
	sum,
	min
};
// 归约，直接调用默认求和
void reduce(size_t *dMatrix, size_t* dResult,
	size_t nGroups, size_t size, ReduceOption op = ReduceOption::sum);
void reduce(float *dMatrix, float* dResult,
	size_t nGroups, size_t size, ReduceOption op = ReduceOption::sum);
void reduce2(float *dMatrix, float* dResult,
	size_t nGroups, size_t size, ReduceOption op = ReduceOption::sum);
// 求最小值
void reduceMin(float *dMatrix, float* dResult, size_t nGroups, size_t size);
void reduceMin(size_t *dMatrix, size_t* dResult, size_t nGroups, size_t size);

/*
 * 求矩阵中每一行第一个非0元素的位置
 */
void argFirstNoneZero(size_t* dMatrix, size_t* result,
	size_t nGroups, size_t size);

/* ==================================
 * 通用烟花生成相关方法
 * ==================================
 */

 /*
  * 给定某一方向上每一时刻的力，求出任一时刻生成的粒子在的在该力作用下的位移
  */
// 要求： size * nInterpolation 不能超过kMmaxBlockDim
void calcShiftingByOutsideForce(
	float* dIn,
	size_t size, // 外力的数量，每一帧对应一个外力，因此等同于帧数
	size_t nInterpolation, // 插值的数量
	float time = kFrameTime // 两帧间隔的时间
);

/*
 * 获取颜色和尺寸变化的矩阵
 */
// 要求：nFrames不能超过kMmaxBlockDim
void getColorAndSizeMatrix(
	const float* startColors, // 输入 起始颜色
	const float* startSizes, // 输入 起始尺寸
	size_t nFrames, // 总计帧数
	float colorDecay, // 颜色衰减率
	float sizeDecay, // 尺寸衰减率
	float* dColorMatrix, // 输出，颜色随帧数变化矩阵
	float* dSizeMatrix // 输出，尺寸随帧数变化矩阵
);

void getColorAndSizeMatrixDevInput(
	const float* startColors, // 输入 起始颜色
	const float* startSizes, // 输入 起始尺寸
	size_t nFrames, // 总计帧数
	float colorDecay, // 颜色衰减率
	float sizeDecay, // 尺寸衰减率
	float* dColorMatrix, // 输出，颜色随帧数变化矩阵
	float* dSizeMatrix // 输出，尺寸随帧数变化矩阵
);

/*
 * 由N个粒子系统生成N组点，求出每个点对应的位置，颜色，尺寸
 */ 
// 要求：nFrames不能大于kMmaxBlockDim

// 此方法用在有着多级爆炸的烟花上面
void particleSystemToPoints(float* dPoints, float* dColors, float* dSizes,
	size_t* dGroupStarts, const size_t* dStartFrames, const size_t* dLifeTime,
	size_t nGroups, const float* dDirections, const float* dCentrifugalPos,
	const float* dStartPoses, size_t currFrame, size_t nFrames,
	const size_t* dColorAndSizeStarts, const float* dColorMatrix,
	const float* dSizeMatrix, float time = kFrameTime);

// 此方法用在没有多级爆炸的烟花上面，此时dColorAndSizeStarts默认为0
void particleSystemToPoints(
	float* dPoints, // 输出 起始位置 
	float* dColors, // 输出 起始颜色
	float* dSizes, // 输出 起始尺寸
	size_t* dGroupStarts, // 输出 此方法将dStartFrames拷贝过来
	const size_t* dStartFrames, // 每一组粒子的起始帧
	const size_t* dLifeTime, // 每一组粒子的寿命
	size_t nGroups, // 总计的粒子组数
	const float* dDirections, // 每一组粒子的方向
	const float* dCentrifugalPos, // 每一组粒子的初始速度
	const float* dStartPoses, // 每一帧粒子生成时候的离心位置
	size_t currFrame, // 当前帧数
	size_t nFrames, // 总帧数
	const float* dColorMatrix, // 颜色随帧数变化的矩阵
	const float* dSizeMatrix, // 尺寸随帧数变化的矩阵
	float time = kFrameTime // 两帧间隔的时间
);

/*
 * 对烟花的粒子进行空间压缩，除去其中的不可见粒子
 */
// 要求：nGroups和size不能超过kMmaxBlockDim
// 返回值：压缩后实际生效的粒子组数
size_t compress(
	float* dPoints, // 输入&输出 粒子的位置
	float* dColors, // 输入&输出 粒子的颜色
	float* dSizes, // 输入&输出 粒子的尺寸
	size_t nGroups, // 粒子组数
	size_t size, // 每组粒子的个数，此方法输入的每组粒子数量相同
	size_t* dGroupOffsets, // 输出 压缩后每组粒子位置相对于起始位置的偏移
	size_t* dGroupStarts // 输出 压缩后的每组粒子的起始帧
);

/*
 * 插值算法 对N组相同长度的数组做插值
 */
// 要求：每组插值的结果长度不能超过kMmaxBlockDim
// 例如 有49帧的粒子， 每组插值15个， 则共有48 * 15 + 49个粒子

// 此方法用于插值N组，每组的数量都一致的情况
void interpolation(
	float* dArray, size_t nGroups, size_t size, size_t nInterpolation);

// 此方法用于对点的位置，颜色，尺寸进行插值，每组的数量允许不一致
void interpolation(
	float* dPoints, // 输入&输出 粒子的位置
	float* dColors, // 输入&输出 粒子的颜色
	float* dSizes, // 输入&输出 粒子的尺寸
	size_t* dGroupOffsets, // 输入&输出 插值后每组粒子位置的偏移
	size_t nGroups, // 粒子组数
	size_t maxSize, // 插值之前，每组粒子最多的个数，一般为帧数
	size_t nInterpolation // 每两个粒子之间插入的粒子数量
);

/*
 * 计算最终每个点的位置
 */
// 要求： 每组粒子的最大数量不能超过kMmaxBlockDim
void calcFinalPosition(
	float* dPoints, // 输入&输出 粒子的位置
	size_t nGroups, // 粒子组数
	size_t maxSize, // 每组粒子最多的个数，一般为帧数和nInterpolation+1的乘积
	size_t nInterpolation, // 每两个粒子之间插入的粒子数量
	size_t frame, // 当前帧
	const size_t* dGroupOffsets, // 每组粒子位置相对于起始位置的偏移
	const size_t* dGroupStarts, // 每组粒子压缩后的起始帧
	const size_t* dStartFrames, // 每组粒子压缩之前的起始帧
	const float* dXShiftMatrix, // 颜色随帧数变化的矩阵
	const float* dYShiftMatrix, // 尺寸随帧数变化的矩阵
	size_t shiftsize // shiftMatrix每一行的尺寸
);

/*
 * 把点连成线，生成这一条线上面的三角形面片
 */
// 输入要求： 每组粒子的最大数量不能超过kMmaxBlockDim， 每组粒子的点为奇数；
// 应用场景： 生成的每两个点之间插入奇数（15）个点，以保证总数为奇数
// 返回值： indices的尺寸，反应要绘制的点的数量
size_t pointToLine(
	const float* dPointsIn, // 最终的粒子位置
	const float* dSizesIn, // 最终的粒子尺寸
	const float* dColorsIn, // 最终的粒子颜色
	size_t maxSizePerGroup, // 每一组粒子的最大尺寸
	size_t* const dGroupOffsets, // 每组粒子位置相对于起始位置的偏移
	size_t nGroups, // 粒子组数
	float* dBuffer, // 顶点数据缓存 vbo
	uint32_t* dIndicesOut, // 顶点序列缓存 ebo
	float outterAlpha = 0.5, // 外圈的不透明度
	float innerSize = 0.25, // 内圈的尺寸
	float innerColor = 0.8
	/*,float innerColorRate = 1*/
);

/* ==================================
 * NormalFirework相关方法
 * ==================================
 */

// 获取normalFirework类型烟花的初始方向，返回其方向的数量，即粒子的组数
size_t normalFireworkDirections(
	float* dDirections, // 输出，获取的每组粒子的方向
	size_t nIntersectingSurfaceParticle, // 横截面粒子数量
	float xRate = 0.1, float yRate = 0.1, float zRate = 0.1,
	float xStretch = 1, float yStretch = 1, float zStretch = 1
);

/* ==================================
 * StrafeFirework相关方法
 * ==================================
 */

 // 获取strafeFirework类型烟花的初始方向，返回其方向的数量，即粒子的组数
size_t strafeFireworkDirections(
	float* dDirections, size_t nGroups, size_t size);

/* ==================================
 * MultiExplosionFirework相关方法
 * ==================================
 */
// 获取二次爆炸的粒子的初位置
void getSubFireworkPositions(
	float* dStartPoses, float* dDirections, size_t nDirs,
	size_t nSubGroups, const float* dCentrifugalPos_, size_t startFrame,
	size_t kShift, const float* dShiftX_, const float* dShiftY_);
}// end namespace cudaKernel
#endif