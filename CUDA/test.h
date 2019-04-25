#ifndef FW_KERNEL_UTILS_TEST_HPP
#define FW_KERNEL_UTILS_TEST_HPP
#include <iostream>
#include <cuda_runtime.h>
#include <string>
#include "utils.h"
#include "kernel.h"
using std::cout;
using std::endl;

namespace cudaKernel {

inline void printSplitLine() {
	cout << "-------------------------" << endl
		<< "--------- split ---------" << endl
		<< "-------------------------" << endl;
}

inline void printSplitLine(std::string s) {
	cout <<
		"----------" << std::string(s.size(), '-') << "----------" << endl <<
		"--------- " << s << " ---------" << endl <<
		"----------" << std::string(s.size(), '-') << "----------" << endl;
}

template<class T>
void show(const T* source, size_t size, size_t step) {
	T* h = new T[size];
	CUDACHECK(cudaMemcpy(h, source, size * sizeof(T), cudaMemcpyDeviceToHost));
	for (size_t i = 0; i < size;) {
		cout << h[i] << " ";
		if ((++i) % step == 0) {
			cout << endl;
		}
	}
	cout << endl;
	delete[] h;
}

template<class T>
void debugShow(T* source, size_t size) {
#ifdef DEBUG_PRINT
	show(source, size, size);
#endif
}

template<class T>
void debugShow(T* source, size_t size, size_t step) {
#ifdef DEBUG_PRINT
	show(source, size, step);
#endif
}

template<class T>
void showAndFree(T* source, size_t size, size_t step) {
	show(source, size, step);
	CUDACHECK(cudaFree(source));
}

template<class T>
void showAndFree(T* source, size_t size) {
	showAndFree(source, size, size);
}

template<class T>
void show(T* source, size_t size) {
	show(source, size, size);
}

template<class T>
void show(T* source, size_t* dSteps, size_t nGroups, size_t times) {
	size_t* hSteps = new size_t[nGroups];
	CUDACHECK(cudaMemcpy(
		hSteps, dSteps + 1, nGroups * sizeof(size_t), cudaMemcpyDeviceToHost));
	for (size_t i = 0; i < nGroups; ++i) {
		cout << hSteps[i] << " ";
	}
	cout << endl;
	size_t size = hSteps[nGroups - 1] * times;
	T* h = new T[size];
	CUDACHECK(cudaMemcpy(h, source, size * sizeof(T), cudaMemcpyDeviceToHost));
	for (size_t i = 0, j = 0; i < size;) {
		cout << h[i] << " ";
		if ((++i) == hSteps[j] * times) {
			++j;
			cout << endl;
		}
	}
	cout << endl;
	delete[] hSteps;
	delete[] h;
}

template<class T>
void debugShow(T* source, size_t* dSteps, size_t nGroups, size_t times) {
#ifdef DEBUG_PRINT
	show(source, dSteps, nGroups, times);
#endif
}

template<class T>
void showAndFree(T* source, size_t* dSteps, size_t nGroups, size_t times) {
	show(source, dSteps, nGroups, times);
	CUDACHECK(cudaFree(source));
}

}
#endif