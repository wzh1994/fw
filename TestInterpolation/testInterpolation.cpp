#include "kernels.h"
#include <cstdio>
#include <cuda_runtime.h>
#include <windows.h>

// void interpolation(float* dArray, size_t nGroups, size_t size, size_t count)

void testInterpolationMatrix() {
	float array[100]{1, 2, 3, 4, 5};
	float *dArray;
	cudaMalloc(&dArray, 100 * sizeof(float));
	cudaMemcpy(dArray, array, 5 * sizeof(float), cudaMemcpyHostToDevice);
	interpolation(dArray, 1, 5, 15);
	cudaMemcpy(array, dArray, 65 * sizeof(float), cudaMemcpyDeviceToHost);
	for (int i = 0; i < 65; ++i) {
		printf("%.4f ", array[i]);
	}
	cudaFree(dArray);
}

void testInterpolationMatrixMultiGroups() {
	float array[200]{ 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15};
	float *dArray;
	cudaMalloc(&dArray, 200 * sizeof(float));
	cudaMemcpy(dArray, array, 15 * sizeof(float), cudaMemcpyHostToDevice);
	interpolation(dArray, 3, 5, 15);
	cudaMemcpy(array, dArray, 195 * sizeof(float), cudaMemcpyDeviceToHost);
	for (int i = 0; i < 195;) {
		printf("%.4f ", array[i]);
		if ((++i) % 65 == 0)
			printf("\n");
	}
	cudaFree(dArray);
}

void testInterpolationPointMultiGroups() {
	float point[600]{ 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15 , 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15 };
	float color[600]{ 1.1, 2.1, 3.2, 4.1, 5.1, 6.1, 7.1, 8.1, 9.1, 10.1, 11.1, 12.1, 13.1, 14.1, 15.1,  1.1, 2.1, 3.2, 4.1, 5.1, 6.1, 7.1, 8.1, 9.1, 10.1, 11.1, 12.1, 13.1, 14.1, 15.1,  1.1, 2.1, 3.2, 4.1, 5.1, 6.1, 7.1, 8.1, 9.1, 10.1, 11.1, 12.1, 13.1, 14.1, 15.1 };
	float size[200]{ 0.1, 0.2, 0.3, 0.4, 0.5, 0.1, 0.2, 0.3, 0.4, 0.5 , 0.1, 0.2, 0.3, 0.4, 0.5 };
	size_t offsets[5]{0, 2, 7, 11, 15};
	float *dPoints, *dColors, *dSizes;
	size_t *dOffsets;
	cudaMalloc(&dPoints, 600 * sizeof(float));
	cudaMalloc(&dColors, 600 * sizeof(float));
	cudaMalloc(&dSizes, 600 * sizeof(float));
	cudaMalloc(&dOffsets, 5 * sizeof(size_t));
	cudaMemcpy(dPoints, point, 45 * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(dColors, color, 45 * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(dSizes, size, 15 * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(dOffsets, offsets, 5 * sizeof(size_t), cudaMemcpyHostToDevice);
	interpolation(dPoints, dColors, dSizes, dOffsets, 4, 10, 15);
	cudaMemcpy(offsets, dOffsets, 5 * sizeof(size_t), cudaMemcpyDeviceToHost);
	for (size_t i = 0; i < 5; ++i) {
		printf("%llu ", offsets[i]);
	}
	printf("\n");
	cudaMemcpy(point, dPoints, offsets[4] * 3 * sizeof(float), cudaMemcpyDeviceToHost);
	cudaMemcpy(color, dColors, offsets[4] * 3 * sizeof(float), cudaMemcpyDeviceToHost);
	cudaMemcpy(size, dSizes, offsets[4] * sizeof(float), cudaMemcpyDeviceToHost);

	for (int i = 0, j = 1; i < offsets[4] * 3;) {
		printf("%.4f ", point[i]);
		if ((++i) == offsets[j] * 3) {
			++j;
			printf("\n");
		}
	}
	printf("\n");
	for (int i = 0, j = 1; i < offsets[4] * 3;) {
		printf("%.4f ", color[i]);
		if ((++i) == offsets[j] * 3) {
			++j;
			printf("\n");
		}
	}
	printf("\n");
	for (int i = 0, j = 1; i < offsets[4];) {
		printf("%.4f ", size[i]);
		if ((++i) == offsets[j]) {
			++j;
			printf("\n");
		}
	}
	printf("\n");
	cudaFree(dPoints);
	cudaFree(dColors);
	cudaFree(dSizes);
	cudaFree(dOffsets);
}

int main() {
	testInterpolationMatrix();
	printf("\n"); printf("\n");
	testInterpolationMatrixMultiGroups();
	printf("\n"); printf("\n");
	testInterpolationPointMultiGroups();
	system("pause");
}