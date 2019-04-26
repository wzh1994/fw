#include "kernels.h"
#include <cstdio>
#include <cuda_runtime.h>
#include <windows.h>
#include "test.h"

namespace cudaKernel{

template <class T>
void testFill() {
	T* arr1;
	myCudaMalloc(arr1, 10);
	fill(arr1, 1, 10);
	showAndFree(arr1, 10);
	printSplitLine();
	T* arr2;
	T data2[3]{ 1, 2, 3 };
	myCudaMalloc(arr2, 30);
	fill(arr2, data2, 10, 3);
	showAndFree(arr2, 30, 3);
}

void testScale() {
	float* dArray;
	myCudaMalloc(dArray, 3000);
	fill(dArray, 1, 3000);
	scale(dArray, 0.1, 3000);
	showAndFree(dArray, 3000);
}

}

using namespace cudaKernel;

int main() {
	testFill<float>();
	printSplitLine("size_t");
	testFill<size_t>();
	printSplitLine("scale");
	testScale();
}