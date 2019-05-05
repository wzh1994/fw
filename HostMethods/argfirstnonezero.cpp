#include "hostmethods.hpp"

namespace hostMethod {
void argFirstNoneZero(size_t* matrix, size_t* result,
		size_t nGroups, size_t size) {
	for (int i = 0; i < nGroups; ++i) {
		for (int j = 0; j < size; ++j) {
			if (matrix[i * size + j] > 0) {
				result[i] = j;
				break;
			}
		}
	}
}
}