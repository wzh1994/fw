#ifndef FW_KERNEL_UTILS_KERNEL_HPP
#define FW_KERNEL_UTILS_KERNEL_HPP

#include <utility>
/*
 * ͷ�ļ�����
 * ����һЩ�����������ռ�ͺ궨��
 */

using uint32_t = unsigned int;
using unary_func_t = float(*)(float);
using binary_func_t = float(*)(float, float);
using binary_func_size_t_t = size_t(*)(size_t, size_t);

using ll = long long;
using ull = unsigned long long;

#define CUDACHECK(cmd) do {                         \
  cudaError_t e = cmd;                              \
  if( e != cudaSuccess ) {                          \
    printf("Failed: Cuda error %s:%d '%s'\n",       \
        __FILE__,__LINE__,cudaGetErrorString(e));   \
    exit(1);                                        \
  }                                                 \
} while(0)

#endif
