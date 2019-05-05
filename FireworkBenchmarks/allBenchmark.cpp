#include <kernels.h>
#include <hostmethods.hpp>
#include <timer.h>
#include <../CUDA/utils.h>
#include <../hostmethods/utils.h>

using namespace std;


namespace cudaKernel {

void dirBenchmark(){
	constexpr size_t nIntersectingSurfaceParticle = 30;
	float* dDirections;
	cudaMallocAlign(&dDirections, nIntersectingSurfaceParticle * nIntersectingSurfaceParticle * sizeof(float));
	FOREACH(20, normalFireworkDirections(dDirections, nIntersectingSurfaceParticle));
	Timer t;
	t.start();
	FOREACH(1000, normalFireworkDirections(dDirections, nIntersectingSurfaceParticle));
	t.pstop("cuda normalFireworkDirections");
}
void casBenchmark() {
	float startColors[150], startSizes[150];
	float *dColorMatrix, *dSizeMatrix;
	for (size_t i = 0; i < 150; ++i) {
		startColors[i] = i;
		startSizes[i] = i;
	}
	float *dStartColors, *dStartSizes;
	cudaMallocAndCopy(dStartColors, startColors, 150);
	cudaMallocAndCopy(dStartSizes, startSizes, 50);
	cudaMallocAlign(&dColorMatrix, 2500 * sizeof(float));
	cudaMallocAlign(&dSizeMatrix, 2500 * sizeof(float));
	FOREACH(20, getColorAndSizeMatrixDevInput(
		dStartColors, dSizeMatrix, 49, 0.99, 0.99, dColorMatrix, dSizeMatrix));
	Timer t;
	t.start();
	FOREACH(1000, getColorAndSizeMatrixDevInput(
		dStartColors, dSizeMatrix, 49, 0.99, 0.99, dColorMatrix, dSizeMatrix));
	t.pstop("cuda getColorAndSizeMatrix");
}
void shiftBenchmark() {
	float in[50], *dIn;
	for (size_t i = 0; i < 50; ++i) {
		in[i] = 0;
	}
	cudaMallocAndCopy(dIn, in, 1000000, 50);
	FOREACH(1000, calcShiftingByOutsideForce(dIn, 49, 15, 1));
	printf("here\n");
	Timer t;
	t.start();
	FOREACH(1000, calcShiftingByOutsideForce(dIn, 49, 15, 1));
	t.pstop("host calcShiftingByOutsideForce");
}
void particleToPointsBenchmark() {

	float points[150], colors[150], sizes[150];
	float startColors[150], startSizes[150];
	float *dColorMatrix, *dSizeMatrix;
	size_t num[50], *dGroupStarts, *dStartFrames, *dLifeTime;
	for (size_t i = 0; i < 150; ++i) {
		points[i] = i;
		colors[i] = i;
		sizes[i] = i;
		startColors[i] = i;
		startSizes[i] = i;
		num[i] = i;
	}
	float *dStartColors, *dStartSizes, *dPoints, *dColors, *dSizes;
	cudaMallocAndCopy(dStartColors, startColors, 150);
	cudaMallocAndCopy(dStartSizes, startSizes, 50);
	cudaMallocAndCopy(dPoints, points, 150);
	cudaMallocAndCopy(dColors, colors, 150);
	cudaMallocAndCopy(dSizes, sizes, 50);
	cudaMallocAndCopy(dGroupStarts, num, 50);
	cudaMallocAndCopy(dStartFrames, num, 50);
	cudaMallocAndCopy(dLifeTime, num, 50);
	cudaMallocAlign(&dColorMatrix, 2500 * sizeof(float));
	cudaMallocAlign(&dSizeMatrix, 2500 * sizeof(float));
	getColorAndSizeMatrixDevInput(
		dStartColors, dSizeMatrix, 49, 0.99, 0.99, dColorMatrix, dSizeMatrix);

	constexpr size_t nGroups = 300;
	float fill[3 * nGroups];
	for (size_t i = 0; i < 3 * nGroups; ++i) {
		fill[i] = i;
	}
	float* dDirections, *dCentrifugalPos, *dStartPoses;
	cudaMallocAndCopy(dDirections, fill, 3 * nGroups);
	cudaMallocAndCopy(dCentrifugalPos, fill, nGroups);
	cudaMallocAndCopy(dStartPoses, fill, nGroups);

	FOREACH(20, particleSystemToPoints(dPoints, dColors, dSizes, dGroupStarts, dStartFrames,
		dLifeTime, nGroups, dDirections, dCentrifugalPos, dStartPoses,
		10, 49, dColorMatrix, dSizeMatrix));
	Timer t;
	t.start();
	FOREACH(1000, particleSystemToPoints(dPoints, dColors, dSizes, dGroupStarts, dStartFrames,
		dLifeTime, nGroups, dDirections, dCentrifugalPos, dStartPoses,
		10, 49, dColorMatrix, dSizeMatrix));
	t.pstop("cuda particleToPoints");

}
void compressBenchmark() {
	//size_t compress(float* dPoints, float* dColors, float* dSizes, size_t nGroups,
	//	size_t size, size_t* dGroupOffsets, size_t* dGroupStarts)
	constexpr size_t nGroups = 300;
	constexpr size_t size = 49;
	float fill[3 * nGroups * size];
	for (size_t i = 0; i < 3 * nGroups * size; ++i) {
		fill[i] = i;
	}
	float* dPoints, *dColors, *dSizes;
	size_t *dGroupOffsets, *dGroupStarts;
	cudaMallocAlign(&dGroupOffsets, 50 * sizeof(size_t));
	cudaMallocAlign(&dGroupStarts, 50 * sizeof(size_t));
	cudaMallocAndCopy(dPoints, fill, 3 * nGroups * size);
	cudaMallocAndCopy(dColors, fill, 3 * nGroups * size);
	cudaMallocAndCopy(dSizes, fill, nGroups * size);
	FOREACH(20, compress(dPoints, dColors, dSizes, nGroups, size, dGroupOffsets, dGroupStarts));
	Timer t;
	t.start();
	FOREACH(1000, compress(dPoints, dColors, dSizes, nGroups, size, dGroupOffsets, dGroupStarts));
	t.pstop("cuda compress");
}
void finalPosBenchmark() {

	constexpr size_t nGroups = 300;
	float fill[3 * nGroups];
	for (size_t i = 0; i < 3 * nGroups; ++i) {
		fill[i] = i;
	}
	float in[50], *shiftX, *shiftY;
	for (size_t i = 0; i < 50; ++i) {
		in[i] = 0;
	}
	cudaMallocAndCopy(shiftX, in, 1000000, 50);
	cudaMallocAndCopy(shiftY, in, 1000000, 50);
	calcShiftingByOutsideForce(shiftX, 49, 15, 1);
	calcShiftingByOutsideForce(shiftY, 49, 15, 1);

	float startColors[150], startSizes[150];
	float *dColorMatrix, *dSizeMatrix;
	size_t num[50], *dGroupStarts, *dStartFrames, *dLifeTime;
	for (size_t i = 0; i < 150; ++i) {
		startColors[i] = i;
		startSizes[i] = i;
		num[i] = i;
	}
	float *dStartColors, *dStartSizes, *dPoints, *dColors, *dSizes;
	cudaMallocAlign(&dPoints, 16 * 150 * nGroups * sizeof(size_t));
	cudaMallocAlign(&dColors, 16 * 150 * nGroups * sizeof(size_t));
	cudaMallocAlign(&dSizes, 16 * 150 * nGroups * sizeof(size_t));
	cudaMallocAndCopy(dStartColors, startColors, 150);
	cudaMallocAndCopy(dStartSizes, startSizes, 50);

	cudaMallocAndCopy(dGroupStarts, num, 50);
	cudaMallocAndCopy(dStartFrames, num, 50);
	cudaMallocAndCopy(dLifeTime, num, 50);
	cudaMallocAlign(&dColorMatrix, 2500 * sizeof(float));
	cudaMallocAlign(&dSizeMatrix, 2500 * sizeof(float));
	getColorAndSizeMatrixDevInput(
		dStartColors, dSizeMatrix, 49, 0.99, 0.99, dColorMatrix, dSizeMatrix);


	float* dDirections, *dCentrifugalPos, *dStartPoses;
	cudaMallocAndCopy(dDirections, fill, 3 * nGroups);
	cudaMallocAndCopy(dCentrifugalPos, fill, nGroups);
	cudaMallocAndCopy(dStartPoses, fill, nGroups);

	particleSystemToPoints(dPoints, dColors, dSizes, dGroupStarts, dStartFrames,
		dLifeTime, nGroups, dDirections, dCentrifugalPos, dStartPoses,
		10, 49, dColorMatrix, dSizeMatrix);

	size_t *dGroupOffsets;
	cudaMallocAlign(&dGroupOffsets, 50 * sizeof(dGroupOffsets));
	compress(dPoints, dColors, dSizes, nGroups, 49, dGroupOffsets, dGroupStarts);
	size_t shiftSize =
		49 * (15 + 1) - 15;
	Timer t;
	t.start();
	interpolation(dPoints, dColors, dSizes, dGroupStarts, nGroups, 49, 15);

	calcFinalPosition(dPoints, nGroups, shiftSize, 15, 1, dGroupOffsets,
		dGroupStarts, dStartFrames, shiftX, shiftY, shiftSize);

	t.pstop("cuda finalPosBenchmark");
}
}

namespace hostMethod {

void dirBenchmark() {
	constexpr size_t nIntersectingSurfaceParticle = 30;
	float* dDirections;
	cudaMallocAlign(&dDirections, nIntersectingSurfaceParticle * nIntersectingSurfaceParticle * sizeof(float));
	FOREACH(20, normalFireworkDirections(dDirections, nIntersectingSurfaceParticle));
	Timer t;
	t.start();
	FOREACH(1000, normalFireworkDirections(dDirections, nIntersectingSurfaceParticle));
	t.pstop("host normalFireworkDirections");
}
void casBenchmark() {
	float startColors[150], startSizes[150];
	float *dColorMatrix, *dSizeMatrix;
	for (size_t i = 0; i < 150; ++i) {
		startColors[i] = i;
		startSizes[i] = i;
	}
	cudaMallocAlign(&dColorMatrix, 2500 * sizeof(float));
	cudaMallocAlign(&dSizeMatrix, 2500 * sizeof(float));
	FOREACH(20, getColorAndSizeMatrix(startColors, startSizes, 49, 0.99, 0.99, dColorMatrix, dSizeMatrix));
	Timer t;
	t.start();
	FOREACH(1000, getColorAndSizeMatrix(startColors, startSizes, 49, 0.99, 0.99, dColorMatrix, dSizeMatrix));
	t.pstop("host getColorAndSizeMatrix");
}
void shiftBenchmark() {
	float in[50], *dIn;
	for (size_t i = 0; i < 50; ++i) {
		in[i] = 0;
	}
	cudaMallocAndCopy(dIn, in, 1000000, 49);
	FOREACH(20, calcShiftingByOutsideForce(dIn, 49, 15));
	Timer t;
	t.start();
	FOREACH(1000, calcShiftingByOutsideForce(dIn, 49, 15));
	t.pstop("cuda calcShiftingByOutsideForce");
}


}

void main() {
	//cudaKernel::dirBenchmark();
	//hostMethod::dirBenchmark();
	//cudaKernel::casBenchmark();
	//hostMethod::casBenchmark();
	//cudaKernel::shiftBenchmark();
	//hostMethod::shiftBenchmark();
	//cudaKernel::particleToPointsBenchmark();
	//cudaKernel::compressBenchmark();
	cudaKernel::finalPosBenchmark();
}
