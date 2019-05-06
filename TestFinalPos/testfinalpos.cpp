#include "kernels.h"
#include <cstdio>
#include <cuda_runtime.h>
#include <windows.h>
#include "test.h"

// void calcFinalPosition(
//  	float* dPoints, size_t nGroups, size_t maxSize, size_t count,
//  	size_t frame, size_t* dGroupOffsets, size_t* dGroupStarts,
//  	float* dXShiftMatrix, float* dYShiftMatrix, size_t shiftsize);

namespace cudaKernel {

void testFinalPos() {
	constexpr size_t nGroups = 1;
	constexpr size_t size = 4;
	constexpr size_t count = 15;
	constexpr size_t nFrames = 12;
	constexpr size_t shiftSize = nFrames * (count + 1) - count;

	float points[size * 3]{ 1, 1.1, 1.2, 2, 2.1, 2.2, 3, 3.1, 3.2, 4, 4.1, 4.2 };
	float colors[size * 3]{ 0.1, 0.2, 0.3, 0.11, 0.21, 0.31, 0.12, 0.22, 0.32, 0.13, 0.23, 0.33 };
	float sizes[size]{ 0.0008, 2, 3, 4 };
	size_t groupOffset[nGroups + 1] = { 0, size };
	size_t groupStart[nGroups] = { 0 };
	float forceX[nFrames]{ 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12 };
	float forceY[nFrames]{ -1, -2, -3, -4, -5, -6, -7, -8, -9, -10, -11, -12 };
	float *dPoints, *dColors, *dSizes, *dForceX, *dForceY;
	size_t *dGroupOffsets, *dGroupStarts;
	cudaMallocAndCopy(dPoints, points, 1200, size * 3);
	cudaMallocAndCopy(dColors, colors, 1200, size * 3);
	cudaMallocAndCopy(dSizes, sizes, 400, size);
	cudaMallocAndCopy(dGroupOffsets, groupOffset, nGroups + 1);
	cudaMallocAndCopy(dGroupStarts, groupStart, nGroups);
	cudaMallocAndCopy(dForceX, forceX, 50000, nFrames);
	cudaMallocAndCopy(dForceY, forceY, 50000, nFrames);

	calcShiftingByOutsideForce(dForceX, nFrames, count);
	calcShiftingByOutsideForce(dForceY, nFrames, count);

	size_t realNGroups = compress(
		dPoints, dColors, dSizes, nGroups, size, dGroupOffsets, dGroupStarts);
	interpolation(
		dPoints, dColors, dSizes, dGroupOffsets, realNGroups, size, count);
	
	constexpr size_t currFrame = 10;
	calcFinalPosition(dPoints, realNGroups, size, count, currFrame,
		dGroupOffsets, dGroupStarts, dGroupStarts, dForceX, dForceY, shiftSize);

	showAndFree(dPoints, 3 * (size * (count + 1) - count), 3 * (count + 1));
	cudaFreeAll(dColors, dSizes, dGroupOffsets, dGroupStarts);
}

void testFinalPosMultiGroups() {
	constexpr size_t nGroups = 5;
	constexpr size_t size = 4;
	constexpr size_t count = 15;
	constexpr size_t nFrames = 12;
	constexpr size_t shiftSize = nFrames * (count + 1) - count;

	float points[nGroups * size * 3]{
		1, 1.1, 1.2, 2, 2.1, 2.2, 3, 3.1, 3.2, 4, 4.1, 4.2,
		1, 1.1, 1.2, 2, 2.1, 2.2, 3, 3.1, 3.2, 4, 4.1, 4.2,
		1, 1.1, 1.2, 2, 2.1, 2.2, 3, 3.1, 3.2, 4, 4.1, 4.2,
		1, 1.1, 1.2, 2, 2.1, 2.2, 3, 3.1, 3.2, 4, 4.1, 4.2,
		1, 1.1, 1.2, 2, 2.1, 2.2, 3, 3.1, 3.2, 4, 4.1, 4.2 
	};
	float colors[nGroups * size * 3]{
		0.1, 0.2, 0.3, 0.11, 0.21, 0.31, 0.12, 0.22, 0.32, 0.13, 0.23, 0.33,
		0.1, 0.2, 0.3, 0.11, 0.21, 0.31, 0.12, 0.22, 0.32, 0.13, 0.23, 0.33,
		0.1, 0.2, 0.3, 0.11, 0.21, 0.31, 0.12, 0.22, 0.32, 0.13, 0.23, 0.33,
		0.1, 0.2, 0.3, 0.11, 0.21, 0.31, 0.12, 0.22, 0.32, 0.13, 0.23, 0.33,
		0.1, 0.2, 0.3, 0.11, 0.21, 0.31, 0.12, 0.22, 0.32, 0.13, 0.23, 0.33
	};
	float sizes[nGroups * size]{
		0.003, 2, 3, 4,
		0.003, 2, 3, 4,
		0.003, 2, 3, 4,
		0.003, 2, 3, 4,
		0.3, 2, 3, 4
	};
	// 1和2一致， 3和4一致， 4和5的偏移一致
	size_t groupStart[nGroups] = {
		0, 0, 1, 1, 2
	};
	float forceX[nFrames]{ 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12 };
	float forceY[nFrames]{ -1, -2, -3, -4, -5, -6, -7, -8, -9, -10, -11, -12 };
	float *dPoints, *dColors, *dSizes, *dForceX, *dForceY;
	size_t *dGroupOffsets, *dGroupStarts;
	cudaMallocAndCopy(dPoints, points, 6000, nGroups * size * 3);
	cudaMallocAndCopy(dColors, colors, 6000, nGroups * size * 3);
	cudaMallocAndCopy(dSizes, sizes, 2000, nGroups * size);
	myCudaMalloc(dGroupOffsets, nGroups + 1);
	cudaMallocAndCopy(dGroupStarts, groupStart, nGroups);
	cudaMallocAndCopy(dForceX, forceX, 50000, nFrames);
	cudaMallocAndCopy(dForceY, forceY, 50000, nFrames);

	calcShiftingByOutsideForce(dForceX, nFrames, count);
	calcShiftingByOutsideForce(dForceY, nFrames, count);

	size_t realNGroups = compress(
		dPoints, dColors, dSizes, nGroups, size, dGroupOffsets, dGroupStarts);
	interpolation(dPoints, dColors, dSizes, dGroupOffsets, realNGroups, size, count);
	show(dGroupOffsets, nGroups + 1);
	show(dPoints, dGroupOffsets, realNGroups, 3);
	printSplitLine();
	constexpr size_t currFrame = 10;
	calcFinalPosition(dPoints, realNGroups, size * count, count, currFrame,
		dGroupOffsets, dGroupStarts, dGroupStarts, dForceX, dForceY, shiftSize);

	showAndFree(dPoints, dGroupOffsets, realNGroups, 3);
	cudaFreeAll(dColors, dSizes, dGroupOffsets, dGroupStarts, dForceX, dForceY);
}

}

using namespace cudaKernel;

int main() {
	testFinalPos();
	printSplitLine("Multi Groups");
	testFinalPosMultiGroups();
}