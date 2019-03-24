#pragma once
#include "kernel.h"
/*
 * ����Ŀ����.lib�ļ���fw����cuda��صĺ�����
 * ���ļ�����������kernel�����Ľӿ�
 * ʹ�ñ���Ŀ��kernel��������Ҫ���뱾ͷ�ļ�
 */

/*
 * �ۼӺͲ���
 */
void cuSum(float* dOut, float* din, size_t size);

/*
 * ͨ��һ������ϵͳ�����ӵĳߴ�ʹ�С��������һ������ϵͳ����Ƭ
 */
size_t getTrianglesAndIndices(
	float* vbo, uint32_t* dIndices, float* dPoints,
	float* dColors, float* dSizes, size_t size);