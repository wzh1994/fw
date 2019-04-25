#include "kernel.h"
#include "kernels.h"
#include "cuda_runtime.h"
#include "corecrt_math.h"

// 为了让__syncthreads()通过语法检查
#ifndef __CUDACC__
#define __CUDACC__
#endif // !__CUDACC__
#include "device_launch_parameters.h"
#include "device_functions.h"
#include <cstdio>

template<typename T>
__device__ bool itemClose(T a, T b) {
	return  a == b;
}

template<>
__device__ bool itemClose(float a, float b) {
	return abs(a - b) < 1e-5f;
}
template<>
__device__ bool itemClose(double a, double b) {
	return abs(a - b) < 1e-5;
}