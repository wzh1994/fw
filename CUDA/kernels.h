#pragma once
#include "kernel.h"

/*
 * 本项目生成.lib文件供fw调用cuda相关的函数。
 * 本文件声明了所有kernel函数的接口
 * 使用本项目的kernel函数，需要引入本头文件
 */


/*
 * 基础通用数学方法
 */

// 累加和操作
void cuSum(float* dOut, float* dIn, size_t size, size_t numGroup = 1);
void cuSum(size_t* dOut, const size_t* dIn, size_t size, size_t numGroup = 1);

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

// 求矩阵中每一行最大元素的位置
void argFirstNoneZero(size_t* dMatrix, size_t* result,
	size_t nGroups, size_t size);


/*
 * 烟花相关方法
 */

// 给定某一方向上每一时刻的力，求出任一时刻生成的粒子在的在该力作用下的位移
// size表示时刻的数量，count表示插值的数量, time表示时间间隔。
void calcshiftingByOutsideForce(
	float* dIn, size_t size, size_t count, float time= 0.0416666666f);

// 对烟花的粒子进行空间压缩，除去其中的不可见粒子
size_t compress(float* dPoints, float* dColors, float* dSizes,
	size_t nGroups, size_t size, size_t* dGroupOffsets, size_t* dGroupStarts);

// 插值算法
void interpolation(float* dArray, size_t nGroups, size_t size, size_t count);
void interpolation(float* dPoints, float* dColors, float* dSizes,
	size_t* dGroupOffsets, size_t nGroups, size_t maxSize, size_t count);

// 把点连成线，生成这一条线上面的三角形面片
// 输入要求： 每条线上面的点为奇数；
// 应用场景： 生成的每两个点之间插入奇数（15）个点，以保证总数为奇数
size_t pointToLine(
	const float* dPointsIn, const float* dSizesIn, const float* dColorsIn,
	size_t maxSizePerGroup, size_t* const groupOffsets, size_t nGroups,
	float* buffer, uint32_t* dIndicesOut);

// 通过一个粒子系统中粒子的尺寸和大小，生成这一个粒子系统的面片 
size_t getTrianglesAndIndices(
	float* vbo, uint32_t* dIndices, float* dPoints,
	float* dColors, float* dSizes, size_t size);

