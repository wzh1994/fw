#ifndef FW_HOSTMETHODS_UTILS_UTILS_HPP
#define FW_HOSTMETHODS_UTILS_UTILS_HPP
#include <memory>
namespace hostMethod {
template<class T>
void cudaMallocAndCopy(T* &target, const T* source,
	size_t size_target, size_t size_copy) {
	target = new T[size_target];
	memcpy(target, source, size_copy * sizeof(T));
}

template<class T>
void cudaMallocAndCopy(T*& target, const T* source, size_t size) {
	cudaMallocAndCopy(target, source, size, size);
}

template<class T>
inline void cudaFree(T* p) {
	delete p;
}

// µ›πÈ÷’÷π
inline void cudaFreeAll() {}

template<class T, class ...Args>
void cudaFreeAll(T* p, Args... args) {
	cudaFree(p);
	cudaFreeAll(std::forward<Args>(args)...);
}
}

#endif  // FW_HOSTMETHODS_UTILS_UTILS_HPP
