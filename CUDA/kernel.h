#ifndef FW_KERNEL_UTILS_KERNEL_HPP
#define FW_KERNEL_UTILS_KERNEL_HPP

#include <utility>
#include <string>
/*
 * 头文件引用
 * 定义一些公共的命名空间和宏定义
 */

namespace cudaKernel{
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

#define CUDACHECK(cmd) do {                         \
  cudaError_t e = cmd;                              \
  if( e != cudaSuccess ) {                          \
    FW_NO_THROW(ExecutionFailed);                   \
  }                                                 \
} while(0)

#endif
