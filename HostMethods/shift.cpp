#include "hostmethods.hpp"
#include "utils.h"

namespace hostMethod {
	namespace {
		void fillForceMatrix(float* in, size_t n) {
			for (size_t i = 1; i < n; ++i) {
				for (size_t j = 0; j < n; ++j) {
					if (i > j)
						in[i * n + j] = 0;
					else
						in[i * n + j] = in[j];
				}
			}
		}
	}
	void calcShiftingByOutsideForce(
			float* in, size_t size, size_t nInterpolation, float time) {
		interpolation(in, 1, size, nInterpolation);
		size_t numPerRow = size + nInterpolation * (size - 1);
		scale(in, time / static_cast<float>(nInterpolation + 1), numPerRow);
		fillForceMatrix(in, numPerRow);
		float* tempWorkSpace;
		cudaMallocAndCopy(tempWorkSpace, in, numPerRow * numPerRow);
		cuSum(tempWorkSpace, in, numPerRow, numPerRow);
		cuSum(in, tempWorkSpace, numPerRow, numPerRow);
		CUDACHECK(cudaFree(tempWorkSpace));
		scale(in, time / static_cast<float>((nInterpolation + 1)),
			numPerRow * numPerRow);
		CUDACHECK(cudaGetLastError());
	}
}