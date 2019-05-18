#pragma once
#include <memory>

#define cudaMemset(_a, _b, _c) memset(_a, _b, _c)
#define cudaMemcpy(_dst, _src, _len, _type) memcpy(_dst, _src, _len)

#define CUDACHECK(x) x
#define cudaGetLastError() ;