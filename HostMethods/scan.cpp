#include "hostmethods.hpp"

namespace hostMethod {

template<typename T, class Func>
void scan(T* out, const T* in, size_t size, size_t nGroups, Func f) {
	for (size_t i = 0; i < nGroups; ++i) {
		size_t j = 0;
		out[i * size + j] = in[i * size + j];
		for (++j; j < size; ++j) {
			out[i * size + j] = f(in[i * size + j - 1], in[i * size + j]);
		}
	}
}

namespace {
template <typename T>
T add(T lhs, T rhs) {
	return lhs + rhs;
}

template <typename T>
T max(T lhs, T rhs) {
	return lhs > rhs ? lhs : rhs;
}

template <typename T>
T mul(T lhs, T rhs) {
	return lhs * rhs;
}
}

void cuSum(float* out, const float* in, size_t size, size_t numGroup) {
	scan(out, in, size, numGroup, (binary_func_t)add);
}
void cuSum(size_t* out, const size_t* in, size_t size, size_t numGroup) {
	scan(out, in, size, numGroup, (binary_func_size_t_t)add);
}

void cuMax(float* out, float* in, size_t size, size_t numGroup) {
	scan(out, in, size, numGroup, (binary_func_t)max);
}

void cuMul(float* out, float* in, size_t size, size_t numGroup) {
	scan(out, in, size, numGroup, (binary_func_t)mul);
}
}