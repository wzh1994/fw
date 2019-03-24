#pragma once
#include "kernel.h"
/*
 * 本项目生成.lib文件供fw调用cuda相关的函数。
 * 本文件声明了所有kernel函数的接口
 * 使用本项目的kernel函数，需要引入本头文件
 */

/*
 * 累加和操作
 */
void cuSum(float* dOut, float* din, size_t size);

/*
 * 通过一个粒子系统中粒子的尺寸和大小，生成这一个粒子系统的面片
 */
size_t getTrianglesAndIndices(
	float* vbo, uint32_t* dIndices, float* dPoints,
	float* dColors, float* dSizes, size_t size);