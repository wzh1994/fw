#ifndef FW_HOSTMETHODS_HOSTMETHOD_HPP
#define FW_HOSTMETHODS_HOSTMETHOD_HPP
namespace hostMethod {
	using uint32_t = unsigned int;
	using unary_func_t = float(*)(float);
	using binary_func_t = float(*)(float, float);
	using binary_func_size_t_t = size_t(*)(size_t, size_t);

	using ll = long long;
	using ull = unsigned long long;

	constexpr size_t kMmaxBlockDim = 1024;
	// 所有分配的显存都要以kMmaxBlockDim * sizeof(size_t)对齐
	constexpr size_t kernelAlign = kMmaxBlockDim * sizeof(size_t);

	inline size_t ceilAlign(size_t size, size_t align = kMmaxBlockDim) {
		return (size + align - 1ull) / align;
	}

}

#define CUDACHECK(x) x
#define cudaGetLastError() ;

#endif  // FW_HOSTMETHODS_HOSTMETHOD_HPP