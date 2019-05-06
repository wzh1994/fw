#include "kernels.h"
#include <cstdio>
#include <cuda_runtime.h>
#include <windows.h>
#include "test.h"

namespace cudaKernel {

void testLowerBound() {
	size_t judgement[25]{
		0, 0, 1, 0, 1,
		0, 0, 0, 0, 0,
		1, 1, 0, 0, 1,
		0, 0, 0, 0, 0,
		0, 0, 0, 1, 1
	};
	size_t *dJudgement, *dResult, nGroups = 5;
	cudaMallocAndCopy(dJudgement, judgement, 25);
	cudaMallocAlign(&dResult, nGroups * sizeof(size_t));
	argFirstNoneZero(dJudgement, dResult, nGroups, 5);
	showAndFree(dResult, 5);
}

}

using namespace cudaKernel;

void main() {
	testLowerBound();
}