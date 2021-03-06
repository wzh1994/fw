#include "kernels.h"
#include <cstdio>
#include <cuda_runtime.h>
#include <windows.h>
#include "test.h"

/*
 * size_t pointToLine(
 *     const float* dPointsIn, const float* dSizesIn, const float* dColorsIn,
 *	   size_t maxSizePerGroup, size_t* const groupOffsets, size_t nGroups,
 *	   float* buffer, uint32_t* dIndicesOut);
 */

namespace cudaKernel {

void testOnePoint() {
	constexpr size_t kPoints = 1ull;
	float pointsIn[kPoints * 3]{
		0, 0, 0,
	};
	float sizesIn[kPoints]{
		1
	};
	float colorIn[kPoints * 3]{
		0.1, 0.2, 0.3
	};
	size_t maxSizePerGroup = 5;
	size_t groupOffsers[2]{ 0, kPoints };
	size_t nGroups = 1;
	float *dPointsIn, *dSizesIn, *dColorsIn, *buffer;
	size_t *dGroupOffsets;
	uint32_t *dIndicesOut;
	cudaMallocAndCopy(dPointsIn, pointsIn, kPoints * 3 );
	cudaMallocAndCopy(dSizesIn, sizesIn, kPoints );
	cudaMallocAndCopy(dColorsIn, colorIn, kPoints * 3);
	cudaMallocAndCopy(dGroupOffsets, groupOffsers, 2);
	cudaMallocAlign(&buffer, 10000 * sizeof(float));
	cudaMallocAlign(&dIndicesOut, 10000 * sizeof(uint32_t));
	size_t r = pointToLine(dPointsIn, dSizesIn, dColorsIn,
		maxSizePerGroup, dGroupOffsets, nGroups, buffer, dIndicesOut);
	printSplitLine();
	show(buffer, 1000, 7);
	printSplitLine();
	show(dIndicesOut, 1000, 3);
}

void testOneGroupWithHorizenLine() {
	constexpr size_t kPoints = 4ull;
	float pointsIn[kPoints * 3]{
		0, 0, 0,
		0, 1, 0,
		0, 2, 0,
		0, 3, 0
	};
	float sizesIn[kPoints]{
		1, 2, 3, 4
	};
	float colorIn[kPoints * 3]{
		0.1, 0.2, 0.3,
		0.4, 0.5, 0.6,
		0.7, 0.8, 0.9,
		1.0, 1.1, 1.2
	};
	size_t maxSizePerGroup = 5;
	size_t groupOffsers[2]{0, kPoints};
	size_t nGroups = 1;

	float *dPointsIn, *dSizesIn, *dColorsIn, *buffer;
	size_t *dGroupOffsets;
	uint32_t *dIndicesOut;
	cudaMallocAlign(&dPointsIn, kPoints * 3 * sizeof(float));
	cudaMallocAlign(&dSizesIn, kPoints * sizeof(float));
	cudaMallocAlign(&dColorsIn, kPoints * 3 * sizeof(float));
	cudaMallocAlign(&buffer, 10000 * sizeof(float));
	cudaMallocAlign(&dGroupOffsets, 2 * sizeof(size_t));
	cudaMallocAlign(&dIndicesOut, 10000 * sizeof(uint32_t));

	cudaMemcpy(dPointsIn, pointsIn, kPoints * 3 * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(dSizesIn, sizesIn, kPoints * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(dColorsIn, colorIn, kPoints * 3 * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(dGroupOffsets, groupOffsers, 2 * sizeof(size_t), cudaMemcpyHostToDevice);

	size_t r = pointToLine(dPointsIn, dSizesIn, dColorsIn,
		maxSizePerGroup, dGroupOffsets, nGroups, buffer, dIndicesOut);

	printf("r: %u\n", r);
	uint32_t* hIndices = new uint32_t[r];
	cudaMemcpy(hIndices, dIndicesOut, r * sizeof(uint32_t), cudaMemcpyDeviceToHost);
	for (size_t i = 0; i < r; ++i) {
		if (i > 0 && i % 6 == 0) printf("\n");
		printf("%u ", hIndices[i]);
	}
	printf("\n\n\n");
	delete[] hIndices;

	float* hBuffer = new float[288];
	cudaMemcpy(hBuffer, buffer, 288 * sizeof(float), cudaMemcpyDeviceToHost);
	for (size_t i = 0; i < 288; ++i) {
		if (i > 0 && i % 7 == 0) printf("\n");
		printf("%f ", hBuffer[i]);
	}
	delete[] hBuffer;

	cudaFreeAll(dPointsIn, dSizesIn, dColorsIn, buffer, dGroupOffsets, dIndicesOut);
	system("pause");
}

void testThreeneGroupWithHorizenLine() {
	constexpr size_t kPoints = 16ull;
	float pointsIn[kPoints * 3]{
		0, 0, 0,
		0, 1, 0,
		0, 2, 0, // 1
		1, 1, 0,
		1.1, 1.1, 0,
		1.2, 1.2, 0,
		1.3, 1.3, 0,
		1.4, 1.4, 0, // 2
		0, 0, 1,
		0, 0, 1.1,
		0, 0, 1.2,
		0, 0.1, 1.3,
		0, 0.2, 1.4,
		0, 0.3, 1.5,
		0, 0.4, 1.6,
		0, 0.5, 1.7
	};

	float sizesIn[kPoints]{
		1, 1.1, 1.2,
		1.1, 1.2, 1.3, 1.4, 1.5,
		0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8
	};
	float colorIn[kPoints * 3]{
		0.1, 0.2, 0.3,
		0.4, 0.5, 0.6,
		0.7, 0.8, 0.9, // 3
		0.1, 0.2, 0.3,
		0.4, 0.4, 0.4,
		0.7, 0.8, 0.9,
		1.0, 1.1, 1.2,
		0.1, 0.2, 0.3, // 5
		0.4, 0.5, 0.6,
		0.7, 0.8, 0.9,
		0.1, 0.2, 0.3,
		0.4, 0.5, 0.6,
		0.7, 0.8, 0.9,
		0.1, 0.2, 0.3,
		0.4, 0.5, 0.6,
		0.7, 0.8, 0.9 // 8
	};
	size_t maxSizePerGroup = 10;
	size_t groupOffsers[4]{ 0, 3, 8, 16 };
	size_t nGroups = 3;

	float *dPointsIn, *dSizesIn, *dColorsIn, *buffer;
	size_t *dGroupOffsets;
	uint32_t *dIndicesOut;
	cudaMallocAlign(&dPointsIn, kPoints * 3 * sizeof(float));
	cudaMallocAlign(&dSizesIn, kPoints * sizeof(float));
	cudaMallocAlign(&dColorsIn, kPoints * 3 * sizeof(float));
	cudaMallocAlign(&buffer, 10000 * sizeof(float));
	cudaMallocAlign(&dGroupOffsets, 4 * sizeof(size_t));
	cudaMallocAlign(&dIndicesOut, 10000 * sizeof(uint32_t));

	cudaMemcpy(dPointsIn, pointsIn, kPoints * 3 * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(dSizesIn, sizesIn, kPoints * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(dColorsIn, colorIn, kPoints * 3 * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(dGroupOffsets, groupOffsers, 4 * sizeof(size_t), cudaMemcpyHostToDevice);

	size_t r = pointToLine(dPointsIn, dSizesIn, dColorsIn,
		maxSizePerGroup, dGroupOffsets, nGroups, buffer, dIndicesOut);

	printf("r: %u\n", r);
	uint32_t* hIndices = new uint32_t[r];
	cudaMemcpy(hIndices, dIndicesOut, r * sizeof(uint32_t), cudaMemcpyDeviceToHost);
	for (size_t i = 0; i < r; ++i) {
		if (i > 0 && i % 6 == 0) printf("\n");
		printf("%u ", hIndices[i]);
	}
	printf("\n\n\n");
	delete[] hIndices;

	float* hBuffer = new float[2880];
	cudaMemcpy(hBuffer, buffer, 2880 * sizeof(float), cudaMemcpyDeviceToHost);
	for (size_t i = 0; i < 2880; ++i) {
		if (i > 0 && i % 6 == 0) printf("\n");
		printf("%f ", hBuffer[i]);
	}
	delete[] hBuffer;

	cudaFreeAll(dPointsIn, dSizesIn, dColorsIn, buffer, dGroupOffsets, dIndicesOut);
	system("pause");
}

}

using namespace cudaKernel;

void main() {
	testOnePoint();
	//testOneGroupWithHorizenLine();
	//testThreeneGroupWithHorizenLine();
}