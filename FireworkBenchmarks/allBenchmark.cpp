#include <kernels.h>
#include <hostmethods.hpp>
#include <timer.h>
#include <../CUDA/utils.h>
#include <../CUDA/test.h>
#include <../hostmethods/utils.h>

using namespace std;


namespace cudaKernel {

void mallocBenchmark() {
	float* a;
	FOREACH(20, {
		cudaMallocAlign(&a, 10000);
		cudaFreeAll(a);
		});
	Timer t;
	t.start();
	FOREACH(1000, {
		cudaMallocAlign(&a, 10000);
		cudaFreeAll(a);
		});
	t.pstop("cuda malloc");
}

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
	Timer t;
	t.start();
	FOREACH(1000, calcShiftingByOutsideForce(dIn, 49, 15, 1));
	t.pstop("cuda calcShiftingByOutsideForce");
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
	cudaMallocAlign(&dGroupOffsets, (nGroups + 1) * sizeof(size_t));
	cudaMallocAlign(&dGroupStarts, nGroups * sizeof(size_t));
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
	float *shiftX, *shiftY;
	cudaMallocAndCopy(shiftX, fill, 1000000, 50);
	cudaMallocAndCopy(shiftY, fill, 1000000, 50);
	calcShiftingByOutsideForce(shiftX, 49, 15, 1);
	calcShiftingByOutsideForce(shiftY, 49, 15, 1);

	float *dColorMatrix, *dSizeMatrix;
	size_t num1[500], num2[500], *dGroupStarts, *dStartFrames, *dLifeTime;
	for (size_t i = 0; i < 500; ++i) {
		num1[i] = 0;
	}
	for (size_t i = 0; i < 500; ++i) {
		num2[i] = 50;
	}
	float *dStartColors, *dStartSizes, *dPoints, *dColors, *dSizes;
	cudaMallocAlign(&dPoints, 16 * 150 * nGroups * sizeof(float));
	cudaMallocAlign(&dColors, 16 * 150 * nGroups * sizeof(float));
	cudaMallocAlign(&dSizes, 16 * 50 * nGroups * sizeof(float));
	cudaMallocAndCopy(dStartColors, fill, 150);
	cudaMallocAndCopy(dStartSizes, fill, 50);
	cudaMallocAndCopy(dGroupStarts, num1, nGroups + 1);
	cudaMallocAndCopy(dStartFrames, num1, nGroups + 1);
	cudaMallocAndCopy(dLifeTime, num2, nGroups);
	cudaMallocAlign(&dColorMatrix, 25000 * sizeof(float));
	cudaMallocAlign(&dSizeMatrix, 25000 * sizeof(float));
	getColorAndSizeMatrixDevInput(
		dStartColors, dStartSizes, 49, 0.99, 0.99, dColorMatrix, dSizeMatrix);

	float* dDirections, *dCentrifugalPos, *dStartPoses;
	cudaMallocAndCopy(dDirections, fill, 3 * nGroups);
	cudaMallocAndCopy(dCentrifugalPos, fill, nGroups);
	cudaMallocAndCopy(dStartPoses, fill, nGroups);
	particleSystemToPoints(dPoints, dColors, dSizes, dGroupStarts, dStartFrames,
		dLifeTime, nGroups, dDirections, dCentrifugalPos, dStartPoses,
		10, 49, dColorMatrix, dSizeMatrix);
	size_t *dGroupOffsets;
	cudaMallocAlign(&dGroupOffsets, (nGroups + 1 )* sizeof(size_t));
	size_t realNGroups = compress(dPoints, dColors, dSizes, nGroups, 49, dGroupOffsets, dGroupStarts);
	size_t shiftSize = 49 * (15 + 1) - 15;
	{
		float* tempPoints;
		float* tempColors;
		float* tempSizes;
		size_t* dGroupOffsetsTemp;
		cudaMallocAlign(&tempPoints, 16 * 150 * nGroups * sizeof(float));
		cudaMemcpy(tempPoints, dPoints, 150 * nGroups * sizeof(float), cudaMemcpyDeviceToDevice);
		cudaMallocAlign(&tempColors, 16 * 150 * nGroups * sizeof(float));
		cudaMemcpy(tempColors, dColors, 150 * nGroups * sizeof(float), cudaMemcpyDeviceToDevice);
		cudaMallocAlign(&tempSizes, 16 * 50 * nGroups * sizeof(float));
		cudaMemcpy(tempSizes, dSizes, 50 * nGroups * sizeof(float), cudaMemcpyDeviceToDevice);
		cudaMallocAlign(&dGroupOffsetsTemp, (nGroups + 1) * sizeof(size_t));
		FOREACH(20, {
			cudaMemcpy(dGroupOffsetsTemp, dGroupOffsets, (nGroups + 1) * sizeof(size_t), cudaMemcpyDeviceToDevice);
			interpolation(tempPoints, tempColors, tempSizes, dGroupOffsetsTemp, nGroups, 49, 15); 
		});
		Timer t;
		t.start();
		FOREACH(1000, {
			cudaMemcpy(dGroupOffsetsTemp, dGroupOffsets, (nGroups + 1) * sizeof(size_t), cudaMemcpyDeviceToDevice);
			interpolation(tempPoints, tempColors, tempSizes, dGroupOffsetsTemp, nGroups, 49, 15);
		});
		t.pstop("cuda interpolation");
		cudaFreeAll(tempPoints, tempColors, tempSizes, dGroupOffsetsTemp);
	}
	interpolation(dPoints, dColors, dSizes, dGroupOffsets, nGroups, 49, 15);
	{
		FOREACH(20, calcFinalPosition(dPoints, nGroups, shiftSize, 15, 1, dGroupOffsets,
			dGroupStarts, dStartFrames, shiftX, shiftY, shiftSize));
		Timer t;
		t.start();
		FOREACH(1000, calcFinalPosition(dPoints, nGroups, shiftSize, 15, 1, dGroupOffsets,
			dGroupStarts, dStartFrames, shiftX, shiftY, shiftSize));
		t.pstop("cuda finalPosBenchmark");
	}
	//float* pVboData;
	//uint32_t* pEboData;
	//cudaMallocAlign(&pVboData, 200000000 * sizeof(float));
	//cudaMallocAlign(&pEboData, 200000000 * sizeof(uint32_t));
	//FOREACH(20, pointToLine(dPoints, dSizes, dColors, 49 * 16, dGroupOffsets, realNGroups,
	//		pVboData, pEboData, 0.5, 0.5, 0.5));
	//Timer t;
	//t.start();
	//FOREACH(1000, pointToLine(dPoints, dSizes, dColors, 49 * 16, dGroupOffsets, realNGroups,
	//	pVboData, pEboData, 0.5, 0.5, 0.5));
	//t.pstop("cuda pointToLine");


}
void pointToLineBenchmark() {
	constexpr size_t nGroups = 300, realNGroups = 300;
	constexpr size_t nPoints = 700;
	float *points = new float[nGroups * nPoints * 3];
	float *sizes = new float[nGroups * nPoints];
	float *colors = new float[nGroups * nPoints * 3];
	for (size_t i = 0; i < nGroups * nPoints; ++i) {
		points[3 * i] = i;
		points[3 * i + 1] = i;
		points[3 * i + 2] = i;
		sizes[i] = 1;
		colors[3 * i] = 1;
		colors[3 * i + 1] = 1;
		colors[3 * i + 2] = 1;
	}

	float *dPoints, *dSizes, *dColors;
	size_t *dGroupOffsets;
	cudaMallocAndCopy(dPoints, points, nGroups * nPoints * 3);
	cudaMallocAndCopy(dSizes, sizes, nGroups * nPoints);
	cudaMallocAndCopy(dColors, colors, nGroups * nPoints * 3);

	size_t offsets[nGroups + 1];
	for (size_t i = 0; i < nGroups + 1; ++i) {
		offsets[i] = 700 * i;
	}
	cudaMallocAndCopy(dGroupOffsets, offsets, nGroups + 1);
	show(dGroupOffsets, nGroups + 1);

	float* pVboData;
	uint32_t* pEboData;
	cudaMalloc(&pVboData, 100000000 * sizeof(float));
	cudaMalloc(&pEboData, 100000000 * sizeof(uint32_t));
	CUDACHECK(cudaDeviceSynchronize());
	FOREACH(20, pointToLine(dPoints, dSizes, dColors, 49 * 16, dGroupOffsets, realNGroups,
		pVboData, pEboData, 0.5, 0.5, 0.5));
	Timer t;
	t.start();
	FOREACH(1000, pointToLine(dPoints, dSizes, dColors, 49 * 16, dGroupOffsets, realNGroups,
		pVboData, pEboData, 0.5, 0.5, 0.5));
	t.pstop("host pointToLine");
	delete[] points, colors, sizes;
	cudaFree(pVboData);
	cudaFree(pEboData);
}
}

namespace hostMethod {

void mallocBenchmark() {
	float* a;
	FOREACH(20, {
		cudaMallocAlign(&a, 100);
		cudaFreeAll(a);
		});
	Timer t;
	t.start();
	FOREACH(1000, {
		cudaMallocAlign(&a, 100);
		cudaFreeAll(a);
		});
	t.pstop("host malloc");
}
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
void finalPosBenchmark() {

	constexpr size_t nGroups = 300;
	float fill[3 * nGroups];
	for (size_t i = 0; i < 3 * nGroups; ++i) {
		fill[i] = i;
	}
	float *shiftX, *shiftY;
	cudaMallocAndCopy(shiftX, fill, 1000000, 50);
	cudaMallocAndCopy(shiftY, fill, 1000000, 50);
	calcShiftingByOutsideForce(shiftX, 49, 15, 1);
	calcShiftingByOutsideForce(shiftY, 49, 15, 1);

	float *dColorMatrix, *dSizeMatrix;
	size_t num1[500], num2[500], *dGroupStarts, *dStartFrames, *dLifeTime;
	for (size_t i = 0; i < 500; ++i) {
		num1[i] = 0;
	}
	for (size_t i = 0; i < 500; ++i) {
		num2[i] = 50;
	}
	float *dStartColors, *dStartSizes, *dPoints, *dColors, *dSizes;
	cudaMallocAlign(&dPoints, 16 * 150 * nGroups * sizeof(float));
	cudaMallocAlign(&dColors, 16 * 150 * nGroups * sizeof(float));
	cudaMallocAlign(&dSizes, 16 * 50 * nGroups * sizeof(float));
	cudaMallocAndCopy(dStartColors, fill, 150);
	cudaMallocAndCopy(dStartSizes, fill, 50);
	cudaMallocAndCopy(dGroupStarts, num1, nGroups + 1);
	cudaMallocAndCopy(dStartFrames, num1, nGroups + 1);
	cudaMallocAndCopy(dLifeTime, num2, nGroups);
	cudaMallocAlign(&dColorMatrix, 25000 * sizeof(float));
	cudaMallocAlign(&dSizeMatrix, 25000 * sizeof(float));
	getColorAndSizeMatrix(
		dStartColors, dStartSizes, 49, 0.99, 0.99, dColorMatrix, dSizeMatrix);

	float* dDirections, *dCentrifugalPos, *dStartPoses;
	cudaMallocAndCopy(dDirections, fill, 3 * nGroups);
	cudaMallocAndCopy(dCentrifugalPos, fill, nGroups);
	cudaMallocAndCopy(dStartPoses, fill, nGroups);
	particleSystemToPoints(dPoints, dColors, dSizes, dGroupStarts, dStartFrames,
		dLifeTime, nGroups, dDirections, dCentrifugalPos, dStartPoses,
		10, 49, dColorMatrix, dSizeMatrix);
	/*{
		float *tempPoints, *tempColor, *tempSize;
		size_t *tempGroupStarts;
		cudaMallocAlign(&tempPoints, 16 * 150 * nGroups * sizeof(float));
		cudaMallocAlign(&tempColor, 16 * 150 * nGroups * sizeof(float));
		cudaMallocAlign(&tempSize, 16 * 50 * nGroups * sizeof(float));
		cudaMallocAndCopy(tempGroupStarts, num1, nGroups + 1);
		FOREACH(20, particleSystemToPoints(tempPoints, tempColor, tempSize, tempGroupStarts, dStartFrames,
			dLifeTime, nGroups, dDirections, dCentrifugalPos, dStartPoses,
			10, 49, dColorMatrix, dSizeMatrix));
		Timer t;
		t.start();
		FOREACH(1000, particleSystemToPoints(tempPoints, tempColor, tempSize, tempGroupStarts, dStartFrames,
			dLifeTime, nGroups, dDirections, dCentrifugalPos, dStartPoses,
			10, 49, dColorMatrix, dSizeMatrix));
		t.pstop("host particleSystemToPoints");
	}*/
	
	/*{
		float *tempPoints, *tempColor, *tempSize;
		size_t *tempGroupStarts;
		size_t *tempGroupOffsets;
		cudaMallocAlign(&tempGroupStarts, (nGroups + 1) * sizeof(size_t));
		cudaMallocAlign(&tempGroupOffsets, (nGroups + 1) * sizeof(size_t));
		cudaMallocAlign(&tempPoints, 16 * 150 * nGroups * sizeof(float));
		cudaMallocAlign(&tempColor, 16 * 150 * nGroups * sizeof(float));
		cudaMallocAlign(&tempSize, 16 * 50 * nGroups * sizeof(float));
		memcpy(tempPoints, dPoints, 16 * 150 * nGroups * sizeof(float));
		memcpy(tempColor, dColors, 16 * 150 * nGroups * sizeof(float));
		memcpy(tempSize, dSizes, 16 * 50 * nGroups * sizeof(float));
		FOREACH(20, compress(tempPoints, tempColor, tempSize, nGroups, 49, tempGroupOffsets, tempGroupStarts));
		Timer t;
		t.start();
		FOREACH(1000, compress(tempPoints, tempColor, tempSize, nGroups, 49, tempGroupOffsets, tempGroupStarts));
		t.pstop("host compress");
	}*/
	size_t *dGroupOffsets;
	cudaMallocAlign(&dGroupOffsets, (nGroups + 1) * sizeof(size_t));
	size_t realNGroups = compress(dPoints, dColors, dSizes, nGroups, 49, dGroupOffsets, dGroupStarts);
	cout << "compress done" << endl;
	size_t shiftSize = 49 * (15 + 1) - 15;
	{
		float* tempPoints;
		float* tempColors;
		float* tempSizes;
		size_t* dGroupOffsetsTemp;
		cudaMallocAlign(&tempPoints, 16 * 150 * nGroups * sizeof(float));
		memcpy(tempPoints, dPoints, 150 * nGroups * sizeof(float));
		cudaMallocAlign(&tempColors, 16 * 150 * nGroups * sizeof(float));
		memcpy(tempColors, dColors, 150 * nGroups * sizeof(float));
		cudaMallocAlign(&tempSizes, 16 * 50 * nGroups * sizeof(float));
		memcpy(tempSizes, dSizes, 50 * nGroups * sizeof(float));
		cudaMallocAlign(&dGroupOffsetsTemp, (nGroups + 1) * sizeof(size_t));
		FOREACH(20, {
			memcpy(dGroupOffsetsTemp, dGroupOffsets, (nGroups + 1) * sizeof(size_t));
			interpolation(tempPoints, tempColors, tempSizes, dGroupOffsetsTemp, nGroups, 49, 15);
			});
		Timer t;
		t.start();
		FOREACH(1000, {
			memcpy(dGroupOffsetsTemp, dGroupOffsets, (nGroups + 1) * sizeof(size_t));
			interpolation(tempPoints, tempColors, tempSizes, dGroupOffsetsTemp, nGroups, 49, 15);
			});
		t.pstop("host interpolation");
	}

	interpolation(dPoints, dColors, dSizes, dGroupOffsets, nGroups, 49, 15);
	{
		cout << "here" << endl;
		FOREACH(20, calcFinalPosition(dPoints, nGroups, shiftSize, 15, 1, dGroupOffsets,
			dGroupStarts, dStartFrames, shiftX, shiftY, shiftSize));
		cout << "here" << endl;
		Timer t;
		t.start();
		FOREACH(1000, calcFinalPosition(dPoints, nGroups, shiftSize, 15, 1, dGroupOffsets,
			dGroupStarts, dStartFrames, shiftX, shiftY, shiftSize));
		t.pstop("host finalPosBenchmark");
		cout << "here" << endl;
	}

	float* pVboData;
	uint32_t* pEboData;
	cudaMallocAlign(&pVboData, 200000000 * sizeof(float));
	cudaMallocAlign(&pEboData, 200000000 * sizeof(uint32_t));
	FOREACH(20, pointToLine(dPoints, dSizes, dColors, 49 * 16, dGroupOffsets, realNGroups,
		pVboData, pEboData, 0.5, 0.5, 0.5));
	Timer t;
	t.start();
	FOREACH(1000, pointToLine(dPoints, dSizes, dColors, 49 * 16, dGroupOffsets, realNGroups,
		pVboData, pEboData, 0.5, 0.5, 0.5));
	t.pstop("host pointToLine");
}
void pointToLineBenchmark() {
	constexpr size_t nGroups = 300, realNGroups = 300;
	constexpr size_t nPoints = 700;
	float *points = new float[nGroups * nPoints * 3];
	float *sizes = new float[nGroups * nPoints];
	float *colors = new float[nGroups * nPoints * 3];
	for (size_t i = 0; i < nGroups * nPoints; ++i) {
		points[3 * i] = i;
		points[3 * i + 1] = i;
		points[3 * i + 2] = i;
		sizes[i] = 1;
		colors[3 * i] = 1;
		colors[3 * i + 1] = 1;
		colors[3 * i + 2] = 1;
	}

	float *dPoints, *dSizes, *dColors;
	size_t *dGroupOffsets;
	cudaMallocAndCopy(dPoints, points, nGroups * nPoints * 3);
	cudaMallocAndCopy(dSizes, sizes, nGroups * nPoints);
	cudaMallocAndCopy(dColors, colors, nGroups * nPoints * 3);

	size_t offsets[nGroups + 1];
	for (size_t i = 0; i < nGroups + 1; ++i) {
		offsets[i] = 700 * i;
	}
	cudaMallocAndCopy(dGroupOffsets, offsets, nGroups + 1);

	float* pVboData;
	uint32_t* pEboData;
	cudaMallocAlign(&pVboData, 200000000 * sizeof(float));
	cudaMallocAlign(&pEboData, 200000000 * sizeof(uint32_t));
	FOREACH(20, pointToLine(dPoints, dSizes, dColors, 49 * 16, dGroupOffsets, realNGroups,
		pVboData, pEboData, 0.5, 0.5, 0.5));
	Timer t;
	t.start();
	FOREACH(100, pointToLine(dPoints, dSizes, dColors, 49 * 16, dGroupOffsets, realNGroups,
		pVboData, pEboData, 0.5, 0.5, 0.5));
	t.pstop("host pointToLine");
	delete[] points, colors, sizes;
	cudaFreeAll(pVboData, pEboData);
}

}

void main() {
	//cudaKernel::mallocBenchmark();
	//hostMethod::mallocBenchmark();
	//cudaKernel::dirBenchmark();
	//hostMethod::dirBenchmark();
	//cudaKernel::casBenchmark();
	//hostMethod::casBenchmark();
	//cudaKernel::shiftBenchmark();
	//hostMethod::shiftBenchmark();
	//cudaKernel::particleToPointsBenchmark();
	//cudaKernel::compressBenchmark();
	//cudaKernel::finalPosBenchmark();
	//hostMethod::finalPosBenchmark();
	cudaKernel::pointToLineBenchmark();
	//hostMethod::pointToLineBenchmark();
}
