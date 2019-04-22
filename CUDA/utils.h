#ifndef FW_KERNEL_UTILS_UTILS_HPP
#define FW_KERNEL_UTILS_UTILS_HPP
#include <cuda_runtime.h>

template<class T>
void cudaMallocAndCopy(T* &target, const T* source,
		size_t size_target, size_t size_copy) {
	CUDACHECK(cudaMalloc(&target, size_target * sizeof(T)));
	CUDACHECK(cudaMemcpy(target, source, size_copy * sizeof(T), cudaMemcpyHostToDevice));
}

template<class T>
void cudaMallocAndCopy(T*& target, const T* source, size_t size) {
	cudaMallocAndCopy(target, source, size, size);
}

template<class T>
void cudaMemcpyAndMallocIfNull(T*& target, T* source,
		size_t size_target, size_t size_copy) {
	if (!target) {
		CUDACHECK(cudaMalloc(&target, size_target * sizeof(T)));
	}
	CUDACHECK(cudaMemcpy(
		target, source, size_copy * sizeof(T), cudaMemcpyHostToDevice));
}

template<class T>
void cudaMemcpyAndMallocIfNull(T*& target, T* source, size_t size) {
	cudaMemcpyAndMallocIfNull(target, source, size, size);
}

// ����my��ֹ��С���û�
template<class T>
void myCudaMalloc(T*& target, size_t size) {
	CUDACHECK(cudaMalloc(&target, size * sizeof(T)));
}

template<class T>
void cudaMallocIfNull(T*& target, size_t size) {
	if (!target) {
		CUDACHECK(cudaMalloc(&target, size * sizeof(T)));
	}
}

// �ݹ���ֹ
inline void cudaFreeAll() {}

template<class T, class ...Args>
void cudaFreeAll(T* p, Args... args) {
	CUDACHECK(cudaFree(p));
	cudaFreeAll(std::forward<Args>(args)...);
}

template<class ...Args>
__device__ void deviceDebugPrint(Args... args) {
#ifdef DEBUG_PRINT
	printf(std::forward<Args>(args)...);
#endif
}

template<class ...Args>
void debugPrint(Args... args) {
#ifdef DEBUG_PRINT
	printf(std::forward<Args>(args)...);
#endif
}

#endif