#include "kernels.h"
#include <cstdio>
#include <cuda_runtime.h>
#include <windows.h>
#include "test.h"

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

int main() {
	testFill<float>();
	printSplitLine();
	testFill<size_t>();
}