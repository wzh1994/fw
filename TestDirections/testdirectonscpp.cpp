#include "kernels.h"
#include <cstdio>
#include <cuda_runtime.h>
#include <windows.h>
#include "test.h"

namespace cudaKernel{

void testDirections(){
	float* dDirections;
	constexpr size_t n = 5;
	cudaMallocAlign(&dDirections, 3 * n * n * sizeof(float));
	size_t nGroups = normalFireworkDirections(dDirections, n);
	show(dDirections, nGroups * 3, 3);
}

} 
using namespace cudaKernel;

void main() {
	testDirections();
}