#include "kernel.h"
#include "kernels.h"
#include "cuda_runtime.h"
#include "corecrt_math.h"

// 为了让__syncthreads()通过语法检查
#ifndef __CUDACC__
#define __CUDACC__
#endif // !__CUDACC__
#include "device_launch_parameters.h"
#include "device_functions.h"
#include <cstdio>
#include <windows.h>

// 此处用宏定义是为了让这个常量可以同时在cuda和host上面生效
#define kCirclePoints 24
#define kCircleIndices kCirclePoints * 6
#define kHalfBallPoints kCirclePoints
#define kHalfBallIndices 0

__constant__ float circle1[kCirclePoints * 3]{
	0.000, 1.000, 0.000,
	0.000, 0.966, 0.259,
	0.000, 0.866, 0.500,
	0.000, 0.707, 0.707,
	0.000, 0.500, 0.866,
	0.000, 0.259, 0.966,
	0.000, 0.000, 1.000,
	0.000, -0.259, 0.966,
	0.000, -0.500, 0.866,
	0.000, -0.707, 0.707,
	0.000, -0.866, 0.500,
	0.000, -0.966, 0.259,
	0.000, -1.000, 0.000,
	0.000, -0.966, -0.259,
	0.000, -0.866, -0.500,
	0.000, -0.707, -0.707,
	0.000, -0.500, -0.866,
	0.000, -0.259, -0.966,
	0.000, -0.000, -1.000,
	0.000, 0.259, -0.966,
	0.000, 0.500, -0.866,
	0.000, 0.707, -0.707,
	0.000, 0.866, -0.500,
	0.000, 0.966, -0.259
};

__constant__ float circle2[kCirclePoints * 3]{
	0.000, 0.991, 0.131,
	0.000, 0.924, 0.383,
	0.000, 0.793, 0.609,
	0.000, 0.609, 0.793,
	0.000, 0.383, 0.924,
	0.000, 0.131, 0.991,
	0.000, -0.131, 0.991,
	0.000, -0.383, 0.924,
	0.000, -0.609, 0.793,
	0.000, -0.793, 0.609,
	0.000, -0.924, 0.383,
	0.000, -0.991, 0.131,
	0.000, -0.991, -0.131,
	0.000, -0.924, -0.383,
	0.000, -0.793, -0.609,
	0.000, -0.609, -0.793,
	0.000, -0.383, -0.924,
	0.000, -0.131, -0.991,
	0.000, 0.131, -0.991,
	0.000, 0.383, -0.924,
	0.000, 0.609, -0.793,
	0.000, 0.793, -0.609,
	0.000, 0.924, -0.383,
	0.000, 0.991, -0.131
};

__constant__ uint32_t indices1[kCircleIndices]{
	0, 24, 1, 1, 24, 25,
	1, 25, 2, 2, 25, 26,
	2, 26, 3, 3, 26, 27,
	3, 27, 4, 4, 27, 28,
	4, 28, 5, 5, 28, 29,
	5, 29, 6, 6, 29, 30,
	6, 30, 7, 7, 30, 31,
	7, 31, 8, 8, 31, 32,
	8, 32, 9, 9, 32, 33,
	9, 33, 10, 10, 33, 34,
	10, 34, 11, 11, 34, 35,
	11, 35, 12, 12, 35, 36,
	12, 36, 13, 13, 36, 37,
	13, 37, 14, 14, 37, 38,
	14, 38, 15, 15, 38, 39,
	15, 39, 16, 16, 39, 40,
	16, 40, 17, 17, 40, 41,
	17, 41, 18, 18, 41, 42,
	18, 42, 19, 19, 42, 43,
	19, 43, 20, 20, 43, 44,
	20, 44, 21, 21, 44, 45,
	21, 45, 22, 22, 45, 46,
	22, 46, 23, 23, 46, 47,
	23, 47, 24, 24, 47, 48
};

__constant__ uint32_t indices2[kCircleIndices]{
	0, 24, 25, 1, 0, 25,
	1, 25, 26, 2, 1, 26,
	2, 26, 27, 3, 2, 27,
	3, 27, 28, 4, 3, 28,
	4, 28, 29, 5, 4, 29,
	5, 29, 30, 6, 5, 30,
	6, 30, 31, 7, 6, 31,
	7, 31, 32, 8, 7, 32,
	8, 32, 33, 9, 8, 33,
	9, 33, 34, 10, 9, 34,
	10, 34, 35, 11, 10, 35,
	11, 35, 36, 12, 11, 36,
	12, 36, 37, 13, 12, 37,
	13, 37, 38, 14, 13, 38,
	14, 38, 39, 15, 14, 39,
	15, 39, 40, 16, 15, 40,
	16, 40, 41, 17, 16, 41,
	17, 41, 42, 18, 17, 42,
	18, 42, 43, 19, 18, 43,
	19, 43, 44, 20, 19, 44,
	20, 44, 45, 21, 20, 45,
	21, 45, 46, 22, 21, 46,
	22, 46, 47, 23, 22, 47,
	23, 47, 24, 0, 23, 24
};

__constant__ float halfBall[kHalfBallPoints * 3]{
	0.000, 1.000, 0.000,
	0.000, 0.966, 0.259,
	0.000, 0.866, 0.500,
	0.000, 0.707, 0.707,
	0.000, 0.500, 0.866,
	0.000, 0.259, 0.966,
	0.000, 0.000, 1.000,
	0.000, -0.259, 0.966,
	0.000, -0.500, 0.866,
	0.000, -0.707, 0.707,
	0.000, -0.866, 0.500,
	0.000, -0.966, 0.259,
	0.000, -1.000, 0.000,
	0.000, -0.966, -0.259,
	0.000, -0.866, -0.500,
	0.000, -0.707, -0.707,
	0.000, -0.500, -0.866,
	0.000, -0.259, -0.966,
	0.000, -0.000, -1.000,
	0.000, 0.259, -0.966,
	0.000, 0.500, -0.866,
	0.000, 0.707, -0.707,
	0.000, 0.866, -0.500,
	0.000, 0.966, -0.259
};

__device__ void normalize(float& a, float& b, float& c) {
	float temp = 1 / sqrtf(a * a + b * b + c * c);
	a = a * temp;
	b = b * temp;
	c = c * temp;
}

__device__ void rotate(float u, float v, float w, float cos_theta,
					   float sin_theta, float& a, float& b, float& c)
{
	if (fabsf(cos_theta - 1.0f) < 1e-5) { 
		printf("No need to rotate!");
		return; 
	}
	normalize(u, v, w);
	float m[3][3];
	float temp_a = a, temp_b = b, temp_c = c;
	m[0][0] = cos_theta + (u * u) * (1 - cos_theta);
	m[0][1] = u * v * (1 - cos_theta) + w * sin_theta;
	m[0][2] = u * w * (1 - cos_theta) - v * sin_theta;

	m[1][0] = u * v * (1 - cos_theta) - w * sin_theta;
	m[1][1] = cos_theta + v * v * (1 - cos_theta);
	m[1][2] = w * v * (1 - cos_theta) + u * sin_theta;

	m[2][0] = u * w * (1 - cos_theta) + v * sin_theta;
	m[2][1] = v * w * (1 - cos_theta) - u * sin_theta;
	m[2][2] = cos_theta + w * w * (1 - cos_theta);
	
	a = m[0][0] * temp_a + m[1][0] * temp_b + m[2][0] * temp_c;
	b = m[0][1] * temp_a + m[1][1] * temp_b + m[2][1] * temp_c;
	c = m[0][2] * temp_a + m[1][2] * temp_b + m[2][2] * temp_c;
}

__device__ void rotate(float u, float v, float w, float theta,
	float& a, float& b, float& c) {
	rotate(u, v, w, cosf(theta), sinf(theta), a, b, c);
}

__device__ void resizeAndTrans(float& x, float& y, float& z,
	float scale, float dx, float dy, float dz) {
	x = x * scale + dx;
	y = y * scale + dy;
	z = z * scale + dz;
}

__device__ void crossAndAngle(float& r1, float& r2, float& r3, float& cos_theta,
		float u1, float u2, float u3, float v1, float v2, float v3) {
	r1 = u2 * v3 - u3 * v2;
	r2 = u3 * v1 - u1 * v3;
	r3 = u1 * v2 - u2 * v1;
	cos_theta = (u1 * v1 + u2 * v2 + u3 * v3) / (
		sqrtf(u1*u1 + u2 * u2 + u3 * u3) * sqrtf(v1*v1 + v2 * v2 + v3 * v3));
}

__device__ void crossAndAngle(float& r1, float& r2, float& r3, float& cos_theta,
		float v1, float v2, float v3) {
	crossAndAngle(r1, r2, r3, cos_theta, 1, 0, 0, v1, v2, v3);
}

void calcHalfBall(
	const float* dPointsIn, const float* dSizesIn, const float* dColorsIn,
	const size_t* groupOffsets, size_t maxSize, size_t nGroups,
	const size_t* bufferOffsets, const size_t* indicesOffsets,
	float* buffer, uint32_t* dIndicesOut) {
	
	// leftBall

	// rightBall

}

__device__ void calcCircularTruncatedConeItem(
		float* pBufferBase, size_t bufferOffset, uint32_t* pIndexBase,
		const float* circle, const uint32_t* indices,
		float x, float y, float z, float size, float r, float g, float b,
		float normX, float normY, float normZ) {
	float axisX, axisY, axisZ, cos_theta;
	crossAndAngle(axisX, axisY, axisZ, cos_theta, normX, normY, normZ);
	float sin_theta = sqrtf(1 - cos_theta * cos_theta);
	printf("|||--- %f, %f, %f, %f, %f ---|||\n", axisX, axisY, axisZ, cos_theta, sin_theta);
	for (size_t i = 0; i < kCirclePoints; ++i) {
		pBufferBase[6 * i] = circle[3 * i];
		pBufferBase[6 * i + 1] = circle[3 * i + 1];
		pBufferBase[6 * i + 2] = circle[3 * i + 2];
		pBufferBase[6 * i + 3] = r;
		pBufferBase[6 * i + 4] = g;
		pBufferBase[6 * i + 5] = b;
		rotate(axisX, axisY, axisZ, cos_theta, sin_theta,
			pBufferBase[6 * i], pBufferBase[6 * i + 1], pBufferBase[6 * i + 2]);
		resizeAndTrans(
			pBufferBase[6 * i], pBufferBase[6 * i + 1], pBufferBase[6 * i + 2],
			size, x, y, z);
		uint32_t baseIndex = bufferOffset - kCirclePoints;
		pIndexBase[6 * i] = baseIndex + indices[6 * i];
		pIndexBase[6 * i + 1] = baseIndex + indices[6 * i + 1];
		pIndexBase[6 * i + 2] = baseIndex + indices[6 * i + 2];
		pIndexBase[6 * i + 3] = baseIndex + indices[6 * i + 3];
		pIndexBase[6 * i + 4] = baseIndex + indices[6 * i + 4];
		pIndexBase[6 * i + 5] = baseIndex + indices[6 * i + 5];
	}
}

using trans_func_t = size_t(*)(size_t);

__global__ void calcCircularTruncatedConeGroup(
		const float* dPointsIn, const float* dSizesIn, const float* dColorsIn,
		const size_t* groupOffsets, const size_t* bufferOffsets,
		const size_t* indicesOffsets, float* buffer, uint32_t* dIndicesOut,
		const float* circle, const uint32_t* indices, trans_func_t trans) {
	printf("in : %u, %u\n", blockIdx.x, threadIdx.x);
	size_t idx = trans(threadIdx.x);
	size_t bidx = blockIdx.x;
	size_t totalNum = groupOffsets[bidx + 1] - groupOffsets[bidx];
	printf("here~~ : %llu, %llu\n", idx, totalNum);
	if ((idx + 1 ) < totalNum) {
		printf("idx less than totalNum : %llu, %llu\n", idx, totalNum);
		size_t bufferOffset = bufferOffsets[bidx] / 6 +
			(kHalfBallPoints + kCirclePoints * (idx - 1));
		float* pBufferBase = buffer + bufferOffset * 6;
		uint32_t* pIndicesBase = dIndicesOut + indicesOffsets[bidx] +
			kHalfBallIndices + kCircleIndices * (idx - 1);
		float size = dSizesIn[groupOffsets[bidx] + idx];
		const float* color = dColorsIn + (groupOffsets[bidx] + idx) * 3;
		const float* pos = dPointsIn + (groupOffsets[bidx] + idx) * 3;
		float normX = *(pos + 3) - *(pos - 3);
		float normY = *(pos + 4) - *(pos - 2);
		float normZ = *(pos + 5) - *(pos - 1);
		calcCircularTruncatedConeItem(
			pBufferBase, bufferOffset, pIndicesBase, circle, indices,
			*pos, *(pos + 1), *(pos + 2), size,
			*color, *(color + 1), *(color + 2), normX, normY, normZ);
	}
}

__global__ void calcFinalIndices(
		const size_t* groupOffsets, const size_t* indicesOffsets,
		const size_t* bufferOffsets, uint32_t* dIndicesOut) {
	size_t idx = threadIdx.x;
	size_t offset = groupOffsets[idx + 1] - groupOffsets[idx] - 2;
	uint32_t* pIndicesBase = dIndicesOut + indicesOffsets[idx] +
		kHalfBallIndices + kCircleIndices * offset;
	uint32_t baseIndex = bufferOffsets[idx] / 6 +
		kHalfBallPoints + kCirclePoints * (offset - 1);
	printf("fill final : %llu, offset: %llu, off: %llu\n", idx, offset, baseIndex);
	for (int i = 0; i < kCircleIndices; ++i) {
		pIndicesBase[i] = baseIndex + indices2[i];
	}
}

__device__ size_t oddTrans(size_t x) {
	return x * 2 + 1;
}
__device__ trans_func_t d_odd = oddTrans;

__device__ size_t evenTrans(size_t x) {
	return (x + 1) * 2;
}
__device__ trans_func_t d_even = evenTrans;

void calcCircularTruncatedCone(
		const float* dPointsIn, const float* dSizesIn, const float* dColorsIn,
		const size_t* groupOffsets, size_t maxSize, size_t nGroups, 
		const size_t* bufferOffsets, const size_t* indicesOffsets,
		float* buffer, uint32_t* dIndicesOut) {
	float *circle = nullptr;
	uint32_t *indices = nullptr;
	trans_func_t odd;
	cudaMemcpyFromSymbol(&odd, d_odd, sizeof(trans_func_t));

	// 使用__constant__地址的方法来自于
	// https://devtalk.nvidia.com/default/topic/487853/can-i-use-__constant__-memory-with-pointer-to-it-as-kernel-arg/
	cudaGetSymbolAddress((void**)&circle, circle2);
	cudaGetSymbolAddress((void**)&indices, indices1);
	calcCircularTruncatedConeGroup << <nGroups, maxSize >> > (
		dPointsIn, dSizesIn, dColorsIn, groupOffsets,
		bufferOffsets, indicesOffsets, buffer, dIndicesOut,
		circle, indices, odd);
	
	trans_func_t even;
	cudaMemcpyFromSymbol(&even, d_even, sizeof(trans_func_t));
	cudaGetSymbolAddress((void**)&circle, circle1);
	cudaGetSymbolAddress((void**)&indices, indices2);
	calcCircularTruncatedConeGroup << <nGroups, maxSize >> > (
		dPointsIn, dSizesIn, dColorsIn, groupOffsets,
		bufferOffsets, indicesOffsets, buffer, dIndicesOut,
		circle, indices, even);

	// 填充最后一个indices
	calcFinalIndices<<<1, nGroups>>>(
		groupOffsets, indicesOffsets, bufferOffsets, dIndicesOut);
}

__global__ void calcOffsets(const size_t* groupOffsets,
	size_t* bufferOffsets, size_t* indicesOffsets) {
	size_t idx = threadIdx.x;
	bufferOffsets[idx] = groupOffsets[idx] * kCirclePoints * 6 + 
		12 * idx * (kHalfBallPoints - kCirclePoints);
	indicesOffsets[idx] = (groupOffsets[idx] - idx) * kCircleIndices +
		2 * idx * kHalfBallIndices;
	printf("%llu - %llu : %llu, %llu\n", idx, groupOffsets[idx], bufferOffsets[idx], indicesOffsets[idx]);
}

size_t pointToLine(
		const float* dPointsIn,
		const float* dSizesIn,
		const float* dColorsIn,
		size_t maxSizePerGroup,
		size_t* const groupOffsets,
		size_t nGroups,
		float* buffer,
		uint32_t* dIndicesOut) {
	size_t *bufferOffsets, *indicesOffsets;
	cudaMalloc(&bufferOffsets, (nGroups + 1) * sizeof(size_t));
	cudaMalloc(&indicesOffsets, (nGroups + 1) * sizeof(size_t));

	calcOffsets << <1, nGroups + 1>> > (groupOffsets,
		bufferOffsets, indicesOffsets);
	cudaDeviceSynchronize();
	//calcHalfBall << <nGroups, maxSizePerGroup >> > (
	//	dPointsIn, dSizesIn, dColorsIn, groupOffsets,
	//	bufferOffsets, indicesOffsets, buffer, dIndicesOut);
	calcCircularTruncatedCone(
		dPointsIn, dSizesIn, dColorsIn, groupOffsets, maxSizePerGroup,
		nGroups, bufferOffsets, indicesOffsets, buffer, dIndicesOut);

	size_t totalIndices;
	cudaMemcpy(&totalIndices, indicesOffsets + nGroups,
		sizeof(size_t), cudaMemcpyDeviceToHost);
	cudaFree(bufferOffsets);
	cudaFree(indicesOffsets);
	return totalIndices;
}

__global__ void testStatic(float* in) {
	rotate(in[0], in[1], in[2], in[3], in[4], in[5], in[6]);
}



void test() {
	float *a, *h;
	float in[7]{0.0f, 1.0f, 1.0f, 3.14f, 0, 1.0f, 0};
	cudaMalloc(&a, 7 * sizeof(float));
	cudaMemcpy(a, in, 7 * sizeof(float), cudaMemcpyHostToDevice);
	testStatic<<<1, 1>>>(a);
	h = new float[7];
	cudaMemcpy(h, a, 7 * sizeof(float), cudaMemcpyDeviceToHost);
	for (int i = 0; i < 7; ++i) {
		printf("%f ", h[i]);
	}
	cudaFree(a);
	delete[] h;
	system("pause");
}

