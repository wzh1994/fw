#include "hostmethods.hpp"

namespace hostMethod {

	template <typename T>
	void fillImpl(T* a, const T* data, size_t size, size_t step = 1) {
		for (size_t i = 0; i < size; ++i) {
			for (size_t j = 0; j < step; ++j) {
				a[i * step + j] = data[j];
			}
		}
	}

	void fill(size_t* dArray, const size_t* data, size_t size, size_t step) {
		fillImpl(dArray, data, size, step);
	}

	void fill(float* dArray, const float* data, size_t size, size_t step) {
		fillImpl(dArray, data, size, step);
	}

	void fill(size_t* a, size_t data, size_t size) {
		fillImpl(a, &data, size);
	}

	void fill(float* a, float data, size_t size) {
		fillImpl(a, &data, size);
	}
}