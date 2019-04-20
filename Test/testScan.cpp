#include <iostream>
#include <cuda_runtime.h>
#include "kernels.h"
using std::cout;
using std::endl;


bool testCase(int n) {
	bool result = true;
	float* arr = new float[3 * n];
	float* cpuResult = new float[3 * n];
	float* gpuResult = new float[3 * n];
	for (int i = 0; i < 3 * n; ++i) {
		arr[i] = i;
		cpuResult[i] = i % n == 0 ? arr[i] : arr[i] + cpuResult[i - 1];
	}
	float *dIn, *dOut; 
	cudaMalloc(&dIn, 3 * n * sizeof(float));
	cudaMalloc(&dOut, 3 * n * sizeof(float));
	cudaMemcpy(dIn, arr, 3 * n * sizeof(float), cudaMemcpyHostToDevice);
	cuSum(dOut, dIn, n, 3);
	cudaMemcpy(gpuResult, dOut, 3 * n * sizeof(float), cudaMemcpyDeviceToHost);
	for (int i = 0; i < 3 * n; ++i) {
		if (abs(gpuResult[i] - cpuResult[i]) > 1e-5)
			cout << "(" << n << ", "<< i << "): " <<
				gpuResult[i] << "!=" << cpuResult[i] << endl;
		result = false;
		break;
	}
	cudaFree(dIn);
	cudaFree(dOut);
	delete arr;
	delete cpuResult;
	delete gpuResult;
	return true;
}

bool testInplaceCase(int n) {
	bool result = true;
	size_t* arr = new size_t[3 * n];
	size_t* cpuResult = new size_t[3 * n];
	size_t* gpuResult = new size_t[3 * n];
	for (int i = 0; i < 3 * n; ++i) {
		arr[i] = i;
		cpuResult[i] = i % n == 0 ? arr[i] : arr[i] + cpuResult[i - 1];
	}
	size_t *dIn;
	cudaMalloc(&dIn, 3 * n * sizeof(size_t));
	cudaMemcpy(dIn, arr, 3 * n * sizeof(size_t), cudaMemcpyHostToDevice);
	cuSum(dIn, dIn, n, 3);
	cudaMemcpy(gpuResult, dIn, 3 * n * sizeof(size_t), cudaMemcpyDeviceToHost);
	for (int i = 0; i < 3 * n; ++i) {
		if (gpuResult[i] != cpuResult[i])
			cout << "(" << n << ", " << i << "): " <<
			gpuResult[i] << "!=" << cpuResult[i] << endl;
		result = false;
		break;
	}
	cudaFree(dIn);
	delete arr;
	delete cpuResult;
	delete gpuResult;
	return true;
}

int main() {
	for (int i = 10; i < 22; ++i) {
		testCase(i);
	}
	for (int i = 10; i < 22; ++i) {
		testInplaceCase(i);
	}
	system("pause");
}