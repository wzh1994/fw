#pragma once
#include "kernel.h"

/*
 * ����Ŀ����.lib�ļ���fw����cuda��صĺ�����
 * ���ļ�����������kernel�����Ľӿ�
 * ʹ�ñ���Ŀ��kernel��������Ҫ���뱾ͷ�ļ�
 */


/*
 * ����ͨ����ѧ����
 */

// �ۼӺͲ���
void cuSum(float* dOut, float* dIn, size_t size, size_t numGroup = 1);
void cuSum(size_t* dOut, const size_t* dIn, size_t size, size_t numGroup = 1);

// ��Լ����
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

// �ۼ� ���ֵ����
void cuMax(float* dOut, float* dIn, size_t size, size_t numGroup = 1);

// �������ÿһ�����Ԫ�ص�λ��
void argFirstNoneZero(size_t* dMatrix, size_t* result,
	size_t nGroups, size_t size);


/*
 * �̻���ط���
 */

// ����ĳһ������ÿһʱ�̵����������һʱ�����ɵ������ڵ��ڸ��������µ�λ��
// size��ʾʱ�̵�������count��ʾ��ֵ������, time��ʾʱ������
void calcshiftingByOutsideForce(
	float* dIn, size_t size, size_t count, float time= 0.0416666666f);

// ���̻������ӽ��пռ�ѹ������ȥ���еĲ��ɼ�����
size_t compress(float* dPoints, float* dColors, float* dSizes,
	size_t nGroups, size_t size, size_t* dGroupOffsets, size_t* dGroupStarts);

// ��ֵ�㷨
void interpolation(float* dArray, size_t nGroups, size_t size, size_t count);
void interpolation(float* dPoints, float* dColors, float* dSizes,
	size_t* dGroupOffsets, size_t nGroups, size_t maxSize, size_t count);

// �ѵ������ߣ�������һ�����������������Ƭ
// ����Ҫ�� ÿ��������ĵ�Ϊ������
// Ӧ�ó����� ���ɵ�ÿ������֮�����������15�����㣬�Ա�֤����Ϊ����
size_t pointToLine(
	const float* dPointsIn, const float* dSizesIn, const float* dColorsIn,
	size_t maxSizePerGroup, size_t* const groupOffsets, size_t nGroups,
	float* buffer, uint32_t* dIndicesOut);

// ͨ��һ������ϵͳ�����ӵĳߴ�ʹ�С��������һ������ϵͳ����Ƭ 
size_t getTrianglesAndIndices(
	float* vbo, uint32_t* dIndices, float* dPoints,
	float* dColors, float* dSizes, size_t size);

