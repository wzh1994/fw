#include "kernels.h"
#include <cstdio>
#include <cuda_runtime.h>
#include <windows.h>
#include "test.h"

namespace cudaKernel {

void testReduce(ReduceOption op) {
	size_t matrix[25]{
		3, 2, 3, 4, 5,
		6, 7, 8, 9, 10,
		2, 1, 3, 1, 5,
		9, 7, 5, 3, 5
	};
	size_t *dMatrix, *dResult;
	cudaMallocAndCopy(dMatrix, matrix, 25);
	cudaMalloc(&dResult, 5 * sizeof(size_t));
	reduce(dMatrix, dResult, 5, 5, op);
	showAndFree(dResult, 5);
}

void testReduceFloat(ReduceOption op) {
	float matrix[25]{
		3, 2, 3, 4, 5,
		6, 7, 8, 9, 10,
		2, 1, 3, 1, 5,
		9, 7, 5, 3, 5
	};
	float *dMatrix, *dResult;
	cudaMallocAndCopy(dMatrix, matrix, 25);
	cudaMalloc(&dResult, 5 * sizeof(float));
	reduce(dMatrix, dResult, 5, 5, op);
	showAndFree(dResult, 5);
}

void testReduceMinFloat() {
	float matrix[25]{
		3, 2, 3, 4, 5,
		6, 7, 8, 9, 10,
		2, 1, 3, 1, 5,
		9, 7, 5, 3, 5
	};
	float *dMatrix, *dResult;
	cudaMallocAndCopy(dMatrix, matrix, 25);
	cudaMalloc(&dResult, 5 * sizeof(float));
	reduceMin(dMatrix, dResult, 5, 5);
	showAndFree(dResult, 5);
}

void testReduceMin() {
	size_t matrix[25]{
		3, 2, 3, 4, 5,
		6, 7, 8, 9, 10,
		2, 1, 3, 1, 5,
		9, 7, 5, 3, 5
	};
	size_t *dMatrix, *dResult;
	cudaMallocAndCopy(dMatrix, matrix, 25);
	cudaMalloc(&dResult, 5 * sizeof(size_t));
	reduceMin(dMatrix, dResult, 5, 5);
	showAndFree(dResult, 5);
}

}

using namespace cudaKernel;

void main() {
	testReduce(ReduceOption::sum);
	testReduce(ReduceOption::min);
	testReduceFloat(ReduceOption::sum);
	testReduceFloat(ReduceOption::min);
	testReduceMin();
	testReduceMinFloat();
}