#include "kernels.h"
#include <cstdio>
#include <cuda_runtime.h>
#include <windows.h>
#include "test.h"

// size_t compress(float* dPoints, float* dColors, float* dSizes,
//     size_t nGroups, size_t size, size_t* dGroupOffsets);
namespace cudaKernel {

void testOneGroup(){
	float *dPoints, *dColors, *dSizes;
	size_t nGroups = 1;
	size_t size = 5;
	size_t *dGroupOffsets, *dGroupStarts;
	float points[15]{
		1, 1.1, 1.2,
		2, 2.2, 2.2,
		3, 3.2, 3.2,
		4, 4.2, 4.2,
		5, 5.2, 5.2
	};
	float colors[15]{
		0.1, 0.1, 0.1,
		0.2, 0.1, 0.3,
		0.5, 0.2, 0.1,
		0.3, 0.1, 0.1,
		0.09, 0.07, 0.01
	};
	float sizes[5]{0, 0.15, 0.4, 0.07, 1};
	cudaMallocAlign(&dPoints, 3 * size * sizeof(float));
	cudaMallocAlign(&dColors, 3 * size * sizeof(float));
	cudaMallocAlign(&dSizes, size * sizeof(float));
	cudaMallocAlign(&dGroupOffsets, (nGroups + 1) * sizeof(size_t));
	cudaMallocAlign(&dGroupStarts, nGroups * sizeof(size_t));
	cudaMemcpy(dPoints, points, 3 * size * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(dColors, colors, 3 * size * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(dSizes, sizes, size * sizeof(float), cudaMemcpyHostToDevice);

	size_t res = compress(dPoints, dColors, dSizes, nGroups, size, dGroupOffsets, dGroupStarts);
	cudaMemcpy(points, dPoints, 3 * size * sizeof(float), cudaMemcpyDeviceToHost);
	cudaMemcpy(colors, dColors, 3 * size * sizeof(float), cudaMemcpyDeviceToHost);
	cudaMemcpy(sizes, dSizes, size * sizeof(float), cudaMemcpyDeviceToHost);
	size_t* groupOffsets = new size_t[nGroups + 1];
	cudaMemcpy(groupOffsets, dGroupOffsets, (res + 1) * sizeof(size_t), cudaMemcpyDeviceToHost);
	printf("real groups: %llu\n", res);
	printf("group offsts:\n");
	for (size_t i = 0; i <= res; ++i) {
		printf("%llu ", groupOffsets[i]);
	}
	printf("\n");
	printf("poses:\n");
	for (size_t i = 0; i < groupOffsets[res] * 3; ++i) {
		printf("%f ", points[i]);
	}
	printf("\n");
	printf("colors:\n");
	for (size_t i = 0; i < groupOffsets[res] * 3; ++i) {
		printf("%f ", colors[i]);
	}
	printf("\n");
	printf("sizes:\n");
	for (size_t i = 0; i < groupOffsets[res]; ++i) {
		printf("%f ", sizes[i]);
	}
	printf("\n");
	delete[] groupOffsets;
	cudaFreeAll(dPoints, dColors, dSizes, dGroupOffsets, dGroupStarts, res);
}

void testFiveGroupsWithTwoEmpty() {
	float *dPoints, *dColors, *dSizes;
	size_t nGroups = 5;
	size_t size = 5;
	size_t *dGroupOffsets, *dGroupStarts;
	float points[75]{
		0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
		1, 1.1, 1.2,
		2, 2.1, 2.2,
		3, 3.1, 3.2,
		4, 4.1, 4.2,
		5, 5.1, 5.2,
		0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
		6, 6.1, 6.3,
		7, 7.1, 7.3,
		8, 8.1, 8.3,
		9, 9.1, 9.3,
		10, 10.1, 10.3,
		0,0,0,0,0,0,0,0,0,0,0,0,0,0,0
	};
	float colors[75]{
		0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
		0.1, 0.1, 0.1,
		0.2, 0.1, 0.3,
		0.5, 0.2, 0.1,
		0.3, 0.1, 0.1,
		0.09, 0.07, 0.01,
		0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
		0.1, 0.1, 0.1,
		0.2, 0.1, 0.3,
		0.5, 0.2, 0.1,
		0.3, 0.1, 0.1,
		0.09, 0.07, 0.01,
		0,0,0,0,0,0,0,0,0,0,0,0,0,0,0
	};
	float sizes[25]{ 
		0, 0, 0, 0, 0,
		0, 0.15, 0.4, 0.07, 1,
		0, 0, 0, 0, 0, 
		0, 0.004, 0.4, 0.17, 1,
		0, 0, 0, 0, 0
	};
	cudaMallocAlign(&dPoints, 3 * nGroups * size * sizeof(float));
	cudaMallocAlign(&dColors, 3 * nGroups *  size * sizeof(float));
	cudaMallocAlign(&dSizes, nGroups * size * sizeof(float));
	cudaMallocAlign(&dGroupOffsets, (nGroups + 1) * sizeof(size_t));
	cudaMallocAlign(&dGroupStarts, nGroups * sizeof(size_t));
	cudaMemcpy(dPoints, points, 3 * nGroups * size * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(dColors, colors, 3 * nGroups * size * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(dSizes, sizes, nGroups * size * sizeof(float), cudaMemcpyHostToDevice);

	size_t res = compress(dPoints, dColors, dSizes, nGroups, size, dGroupOffsets, dGroupStarts);
	cudaMemcpy(points, dPoints, 3 * nGroups * size * sizeof(float), cudaMemcpyDeviceToHost);
	cudaMemcpy(colors, dColors, 3 * nGroups * size * sizeof(float), cudaMemcpyDeviceToHost);
	cudaMemcpy(sizes, dSizes, nGroups * size * sizeof(float), cudaMemcpyDeviceToHost);
	size_t* groupOffsets = new size_t[nGroups + 1];
	cudaMemcpy(groupOffsets, dGroupOffsets, (res + 1) * sizeof(size_t), cudaMemcpyDeviceToHost);
	printf("real groups: %llu\n", res);
	printf("group offsts:\n");
	for (size_t i = 0; i <= res; ++i) {
		printf("%llu ", groupOffsets[i]);
	}
	printf("\n");
	printf("poses:\n");
	for (size_t i = 0; i < groupOffsets[res] * 3; ++i) {
		printf("%f ", points[i]);
	}
	printf("\n");
	printf("colors:\n");
	for (size_t i = 0; i < groupOffsets[res] * 3; ++i) {
		printf("%f ", colors[i]);
	}
	printf("\n");
	printf("sizes:\n");
	for (size_t i = 0; i < groupOffsets[res]; ++i) {
		printf("%f ", sizes[i]);
	}
	printf("\n");
	delete[] groupOffsets;
	cudaFreeAll(dPoints, dColors, dSizes, dGroupOffsets);
	showAndFree(dGroupStarts, res);
}

}

using namespace cudaKernel;

int main() {
	testOneGroup();
	printf("-----------\n------------\n-----------\n");
	testFiveGroupsWithTwoEmpty();
	system("pause");
}