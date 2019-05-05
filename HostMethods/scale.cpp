#include "hostmethods.hpp"

namespace hostMethod {

void scale(float* a, float rate, size_t size) {
	for (size_t i = 0; i < size; ++i) {
		a[i] *= rate;
	}
}

}