#include "kernels.h"
#include <cstdio>
#include <cuda_runtime.h>
#include <windows.h>
#include "test.h"
#include "timer.h"
#include <string>

using std::to_string;

namespace cudaKernel {
void reduceBenchmark(size_t x, size_t y) {
	// worm up
	float *a, *b;
	cudaMallocAlign(&a, x * y * sizeof(float));
	cudaMallocAlign(&b, x * y * sizeof(float));
	for (size_t i = 0; i < 100; ++i) {
		reduce(a, b, x, y);
		reduce2(a, b, x, y);
	}
	Timer t;
	t.start();
	for (size_t i = 0; i < 10000; ++i) {
		reduce(a, b, x, y);
	}
	t.pstop("reduce " + to_string(x) + "," + to_string(y));
	t.start();
	for (size_t i = 0; i < 10000; ++i) {
		reduce2(a, b, x, y);
	}
	t.pstop("reduce2 " + to_string(x) + "," + to_string(y));
}
}

using namespace cudaKernel;

int main() {
	size_t xs[4]{ 16, 64, 256, 1024 };
	for (size_t i = 0; i < 4; ++i) {
		for (size_t j = 0; j < 4; ++j) {
			reduceBenchmark(xs[i], xs[j]);
		}
	}
}