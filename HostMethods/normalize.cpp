#include "hostmethods.hpp"
#include <cmath>

namespace hostMethod {
	void normalize(float& a, float& b, float& c) {
		float temp = 1 / sqrt(a * a + b * b + c * c);
		a = a * temp;
		b = b * temp;
		c = c * temp;
	}

	void normalize(float* vectors, size_t size) {
		for (size_t i = 0; i < size; ++i) {
			normalize(vectors[3 * i], vectors[3 * i + 1], vectors[3 * i + 2]);
		}
	}
}