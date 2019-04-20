#include <iostream>
#include <cuda_runtime.h>
#include "kernel.h"
using std::cout;
using std::endl;

template<class T>
void cudaMallocAndCopy(T* &target, T* source,
		size_t size_target, size_t size_copy) {
	CUDACHECK(cudaMalloc(&target, size_target * sizeof(T)));
	CUDACHECK(cudaMemcpy(target, source, size_copy * sizeof(T), cudaMemcpyHostToDevice));
}

template<class T>
void cudaMallocAndCopy(T*& target, T* source, size_t size) {
	cudaMallocAndCopy(target, source, size, size);
}

template<class T>
void showAndFree(T* source, size_t size, size_t step) {
	T* h = new T[size];
	CUDACHECK(cudaMemcpy(h, source, size * sizeof(T), cudaMemcpyDeviceToHost));
	for (size_t i = 0; i < size;) {
		cout << h[i] << " ";
		if ((++i) % step == 0) {
			cout << endl;
		}
	}
	CUDACHECK(cudaFree(source));
	delete[] h;
}

template<class T>
void showAndFree(T* source, size_t size) {
	showAndFree(source, size, size);
}