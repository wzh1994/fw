#ifndef FW_KERNEL_UTILS_KERNEL_HPP
#define FW_KERNEL_UTILS_KERNEL_HPP

#include <utility>
#include <string>
/*
 * ͷ�ļ�����
 * ����һЩ�����������ռ�ͺ궨��
 */

namespace cudaKernel{
using uint32_t = unsigned int;
using unary_func_t = float(*)(float);
using binary_func_t = float(*)(float, float);
using binary_func_size_t_t = size_t(*)(size_t, size_t);

using ll = long long;
using ull = unsigned long long;

constexpr size_t kMmaxBlockDim = 1024;
// ���з�����Դ涼Ҫ��kMmaxBlockDim * sizeof(size_t)����
constexpr size_t kernelAlign = kMmaxBlockDim * sizeof(size_t);

inline size_t ceilAlign(size_t size, size_t align = kMmaxBlockDim) {
	return (size + align - 1ull) / align;
}

}

#define CUDACHECK(cmd) do {                         \
  cudaError_t e = cmd;                              \
  if( e != cudaSuccess ) {                          \
    FW_NO_THROW(ExecutionFailed);                   \
  }                                                 \
} while(0)

#endif
