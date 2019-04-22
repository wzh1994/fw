#include "kernels.h"
#include <cstdio>
#include <cuda_runtime.h>
#include <windows.h>
#include "test.h"

// void getColorAndSizeMatrix(
//	   float* startColors, float* startSizes, size_t nFrames,
//	   float colorDecay, float sizeDecay,
//	   float*& dColorMatrix, float*& dSizeMatrix)

// void particleSystemToPoints(float* dPoints, float* dColors, float* dSizes,
//     size_t* dGroupStarts, const size_t* startFrames, size_t nGroups,
//	   const float* dDirections, const float* dSpeeds, const float* dStartPoses,
//	   size_t currFrame, size_t nFrames, float* dColorMatrix,
//	   float* dSizeMatrix, float time = 0.0416666666f);

int testParticleSystemToPoints() {
	constexpr size_t nParticles = 3;
	constexpr size_t nFrames = 5;
	constexpr float colorDecay = 0.99, sizeDecay = 0.98;
	float startColors[nFrames * 3]{
		0.1, 0.11, 0.12,
		0.2, 0.21, 0.22,
		0.3, 0.31, 0.32,
		0.4, 0.41, 0.42,
		0.5, 0.51, 0.52
	};
	float startSizes[nFrames]{
		1, 2, 3, 4, 5
	};
	float *dColorMatrix=nullptr, *dSizeMatrix=nullptr;
	myCudaMalloc(dColorMatrix, 3 * nFrames * nFrames);
	myCudaMalloc(dSizeMatrix, nFrames * nFrames);
	getColorAndSizeMatrix(startColors, startSizes, nFrames,
		colorDecay, sizeDecay, dColorMatrix, dSizeMatrix);
	show(dColorMatrix, 3 * nFrames * nFrames, 3 * nFrames);
	show(dSizeMatrix, nFrames * nFrames, nFrames);
	printSplitLine("particleSystemToPoints");

	constexpr size_t currFrame = 2;
	constexpr size_t nGroups = nParticles;

	float *dPoints, *dColors, *dSizes;
	myCudaMalloc(dPoints, 3 * nGroups * nFrames);
	myCudaMalloc(dColors, 3 * nGroups * nFrames);
	myCudaMalloc(dSizes, nGroups * nFrames);

	size_t startFrames[nGroups]{0, 0, 1};
	size_t *dStartFrames, *dGroupStarts;
	cudaMallocAndCopy(dStartFrames, startFrames, nGroups);
	myCudaMalloc(dGroupStarts, nGroups);

	float directions[3 * nGroups]{
		1, 1, 1,
		1, -1, -1,
		-1, 0, 1
	};

	float speeds[3 * nGroups]{
		0.1, 0.2, 0.3,
		0.4, 0.5, 0.6,
		0.7, 0.8, 0.9
	};

	float startPoses[3 * nGroups]{
		0, 0, 0,
		0, 0, 1,
		0, 1, 0
	};

	float *dDirections, *dSpeeds, *dStartPoses;
	cudaMallocAndCopy(dDirections, directions, 3 * nGroups);
	cudaMallocAndCopy(dSpeeds, speeds, 3 * nGroups);
	cudaMallocAndCopy(dStartPoses, startPoses, 3 * nGroups);
	particleSystemToPoints(dPoints, dColors, dSizes, dGroupStarts,
		dStartFrames, nGroups, dDirections, dSpeeds, dStartPoses,
		currFrame, nFrames, dColorMatrix, dSizeMatrix);
	showAndFree(dPoints, 3 * nGroups * nFrames, 3 * nFrames);
	printSplitLine();
	showAndFree(dColors, 3 * nGroups * nFrames, 3 * nFrames);
	printSplitLine();
	showAndFree(dSizes, nGroups * nFrames, nFrames);
	cudaFreeAll(dStartFrames, dGroupStarts, dDirections, dSpeeds,
		dStartPoses, dColorMatrix, dSizeMatrix);
}

void main() {
	testParticleSystemToPoints();
}