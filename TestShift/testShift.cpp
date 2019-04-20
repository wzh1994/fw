#include "kernels.h"
#include <cstdio>
#include <cuda_runtime.h>
#include <windows.h>
#include "test.h"

// void calcshiftingByOutsideForce(float* dIn, size_t size, size_t count, float time)

void testShift() {
	float* in = new float[500000];
	in[0] = 1;
	in[1] = 2;
	in[2] = 3;
	in[3] = 4;
	in[4] = 5;
	float* dIn;
	cudaMallocAndCopy(dIn, in, 500000, 5);
	calcshiftingByOutsideForce(dIn, 5, 150, 1);
	showAndFree(dIn, 2000, 605);
	//showAndFree(dIn, 42025, 205);
	delete[] in;
}

int main() {
	testShift();
	system("pause");
}