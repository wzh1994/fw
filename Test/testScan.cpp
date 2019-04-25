#include <iostream>
#include <cuda_runtime.h>
#include "kernels.h"
#include "test.h"
using std::cout;
using std::endl;

bool scanClose(float a, float b) {
	return abs(a - b) < 1e-5;
}
bool scanClose(size_t a, size_t b) {
	return a == b;
}

template<class T>
bool testCase(int n) {
	bool result = true;
	T* arr = new  T[3 * n];
	T* cpuResult = new  T[3 * n];
	T* gpuResult = new  T[3 * n];
	for (int i = 0; i < 3 * n; ++i) {
		arr[i] = i;
		cpuResult[i] = i % n == 0 ? arr[i] : arr[i] + cpuResult[i - 1];
	}
	T *dIn, *dOut;
	cudaMallocAlign(&dIn, 3 * n * sizeof(T));
	cudaMallocAlign(&dOut, 3 * n * sizeof(T));
	cudaMemcpy(dIn, arr, 3 * n * sizeof(T), cudaMemcpyHostToDevice);
	cuSum(dOut, dIn, n, 3);
	cudaMemcpy(gpuResult, dOut, 3 * n * sizeof(T), cudaMemcpyDeviceToHost);
	for (int i = 0; i < 3 * n; ++i) {
		if (!scanClose(gpuResult[i], cpuResult[i]))
			cout << "(" << n << ", "<< i << "): " <<
				gpuResult[i] << "!=" << cpuResult[i] << endl;
		result = false;
		break;
	}
	printSplitLine();
	show(dOut, n);
	cudaFree(dIn);
	cudaFree(dOut);
	delete arr;
	delete cpuResult;
	delete gpuResult;
	return true;
}

int main() {
	for (int i = 10; i < 22; ++i) {
		testCase<float>(i);
	}
	for (int i = 10; i < 22; ++i) {
		testCase<size_t>(i);
	}
	system("pause");
}