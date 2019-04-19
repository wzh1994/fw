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

// �ۼ� ���ֵ����
void cuMax(float* dOut, float* dIn, size_t size, size_t numGroup = 1);


/*
 * �̻���ط���
 */

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

