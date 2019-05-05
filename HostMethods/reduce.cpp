#include "hostmethods.hpp"

namespace hostMethod {

template <typename T, class Func>
void reduceImpl(T *matrix, T* result,
		size_t nGroups, size_t size, Func f) {
	for (size_t i = 0; i < nGroups; ++i) {
		size_t j = 0;
		result[i] = matrix[i * size + j];
		for (++j; j < size; ++j) {
			result[i] = f(result[i], matrix[i * size + j]);
		}
	}
}

namespace{
	template <typename T>
	T add(T lhs, T rhs) {
		return lhs + rhs;
	}

	template <typename T>
	T min(T lhs, T rhs) {
		return lhs < rhs ? lhs : rhs;
	}
}

void reduce(size_t *matrix, size_t* result,
		size_t nGroups, size_t size, ReduceOption op) {
	switch (op) {
	case ReduceOption::min:
		reduceImpl(matrix, result, nGroups, size, (binary_func_size_t_t)min);
		break;
	case ReduceOption::sum:
	default:
		reduceImpl(matrix, result, nGroups, size, (binary_func_size_t_t)add);
	}
}

void reduce(float *matrix, float* result,
		size_t nGroups, size_t size, ReduceOption op) {
	switch (op) {
	case ReduceOption::min:
		reduceImpl(matrix, result, nGroups, size, (binary_func_t)min);
		break;
	case ReduceOption::sum:
	default:
		reduceImpl(matrix, result, nGroups, size, (binary_func_t)add);
	}
}

// 求最小值
void reduceMin(float *matrix, float* result, size_t nGroups, size_t size) {
	reduce(matrix, result, nGroups, size, ReduceOption::min);
}
void reduceMin(size_t *matrix, size_t* result, size_t nGroups, size_t size) {
	reduce(matrix, result, nGroups, size, ReduceOption::min);
}

}