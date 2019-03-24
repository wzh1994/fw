#include <iostream>
#include <cuda_runtime.h>
#include "kernels.h"
using std::cout;
using std::endl;


bool testCase(int n) {
	bool result = true;
	float* arr = new float[n];
	float* cpuResult = new float[n];
	float* gpuResult = new float[n];
	for (int i = 0; i < n; ++i) {
		arr[i] = i;
		cpuResult[i] = i == 0 ? arr[i] : arr[i] + cpuResult[i - 1];
	}
	float *dIn, *dOut; 
	cudaMalloc(&dIn, n * sizeof(float));
	cudaMalloc(&dOut, n * sizeof(float));
	cudaMemcpy(dIn, arr, n * sizeof(float), cudaMemcpyHostToDevice);
	cuSum(dOut, dIn, n);
	cudaMemcpy(gpuResult, dOut, n * sizeof(float), cudaMemcpyDeviceToHost);
	for (int i = 0; i < n; ++i) {
		if (abs(gpuResult[i] - cpuResult[i]) > 1e-5)
			cout << n << ": " << gpuResult[i] << "!=" << cpuResult[i] << endl;
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

int main() {
	for (int i = 10; i < 22; ++i) {
		testCase(i);
	}
	system("pause");
}