#include "kernel.h"
#include "kernels.h"
#include "cuda_runtime.h"
#include "corecrt_math.h"
#include "utils.h"
#include "test.h"

// 为了让__syncthreads()通过语法检查
#ifndef __CUDACC__
#define __CUDACC__
#endif // !__CUDACC__
#include "device_launch_parameters.h"
#include "device_functions.h"
#include <cstdio>
#define DRAW_BALL

// 此处用宏定义是为了让这个常量可以同时在cuda和host上面生效
#define kCirclePoints 24
#define kCircleIndices kCirclePoints * 6
#ifndef DRAW_BALL
#define kHalfBallPoints kCirclePoints
#define kHalfBallIndices 0
#else
#define kHalfBallPoints 91
#define kHalfBallIndices 468
#endif

namespace cudaKernel {

__constant__ float circle1[kCirclePoints * 3]{
	0.000, 1.000, 0.000,
	0.000, 0.966, 0.259,
	0.000, 0.866, 0.500,
	0.000, 0.707, 0.707,
	0.000, 0.500, 0.866,
	0.000, 0.259, 0.966,
	0.000, 0.000, 1.000,
	0.000, -0.259, 0.966,
	0.000, -0.500, 0.866,
	0.000, -0.707, 0.707,
	0.000, -0.866, 0.500,
	0.000, -0.966, 0.259,
	0.000, -1.000, 0.000,
	0.000, -0.966, -0.259,
	0.000, -0.866, -0.500,
	0.000, -0.707, -0.707,
	0.000, -0.500, -0.866,
	0.000, -0.259, -0.966,
	0.000, -0.000, -1.000,
	0.000, 0.259, -0.966,
	0.000, 0.500, -0.866,
	0.000, 0.707, -0.707,
	0.000, 0.866, -0.500,
	0.000, 0.966, -0.259
};

__constant__ float circle2[kCirclePoints * 3]{
	0.000, 0.991, 0.131,
	0.000, 0.924, 0.383,
	0.000, 0.793, 0.609,
	0.000, 0.609, 0.793,
	0.000, 0.383, 0.924,
	0.000, 0.131, 0.991,
	0.000, -0.131, 0.991,
	0.000, -0.383, 0.924,
	0.000, -0.609, 0.793,
	0.000, -0.793, 0.609,
	0.000, -0.924, 0.383,
	0.000, -0.991, 0.131,
	0.000, -0.991, -0.131,
	0.000, -0.924, -0.383,
	0.000, -0.793, -0.609,
	0.000, -0.609, -0.793,
	0.000, -0.383, -0.924,
	0.000, -0.131, -0.991,
	0.000, 0.131, -0.991,
	0.000, 0.383, -0.924,
	0.000, 0.609, -0.793,
	0.000, 0.793, -0.609,
	0.000, 0.924, -0.383,
	0.000, 0.991, -0.131
};

__constant__ uint32_t indices1[kCircleIndices]{
	0, 24, 1, 1, 24, 25,
	1, 25, 2, 2, 25, 26,
	2, 26, 3, 3, 26, 27,
	3, 27, 4, 4, 27, 28,
	4, 28, 5, 5, 28, 29,
	5, 29, 6, 6, 29, 30,
	6, 30, 7, 7, 30, 31,
	7, 31, 8, 8, 31, 32,
	8, 32, 9, 9, 32, 33,
	9, 33, 10, 10, 33, 34,
	10, 34, 11, 11, 34, 35,
	11, 35, 12, 12, 35, 36,
	12, 36, 13, 13, 36, 37,
	13, 37, 14, 14, 37, 38,
	14, 38, 15, 15, 38, 39,
	15, 39, 16, 16, 39, 40,
	16, 40, 17, 17, 40, 41,
	17, 41, 18, 18, 41, 42,
	18, 42, 19, 19, 42, 43,
	19, 43, 20, 20, 43, 44,
	20, 44, 21, 21, 44, 45,
	21, 45, 22, 22, 45, 46,
	22, 46, 23, 23, 46, 47,
	23, 47, 0, 0, 47, 24
};

__constant__ uint32_t indices2[kCircleIndices]{
	0, 24, 25, 1, 0, 25,
	1, 25, 26, 2, 1, 26,
	2, 26, 27, 3, 2, 27,
	3, 27, 28, 4, 3, 28,
	4, 28, 29, 5, 4, 29,
	5, 29, 30, 6, 5, 30,
	6, 30, 31, 7, 6, 31,
	7, 31, 32, 8, 7, 32,
	8, 32, 33, 9, 8, 33,
	9, 33, 34, 10, 9, 34,
	10, 34, 35, 11, 10, 35,
	11, 35, 36, 12, 11, 36,
	12, 36, 37, 13, 12, 37,
	13, 37, 38, 14, 13, 38,
	14, 38, 39, 15, 14, 39,
	15, 39, 40, 16, 15, 40,
	16, 40, 41, 17, 16, 41,
	17, 41, 42, 18, 17, 42,
	18, 42, 43, 19, 18, 43,
	19, 43, 44, 20, 19, 44,
	20, 44, 45, 21, 20, 45,
	21, 45, 46, 22, 21, 46,
	22, 46, 47, 23, 22, 47,
	23, 47, 24, 0, 23, 24
};

__constant__ float halfBallLeft[kHalfBallPoints * 3]{
#ifdef DRAW_BALL
	- 1.000, 0.000, 0.000,
	-0.966, 0.259, 0.000,
	-0.966, 0.129, 0.224,
	-0.966, -0.129, 0.224,
	-0.966, -0.259, 0.000,
	-0.966, -0.129, -0.224,
	-0.966, 0.129, -0.224,
	-0.866, 0.500, 0.000,
	-0.866, 0.433, 0.250,
	-0.866, 0.250, 0.433,
	-0.866, 0.000, 0.500,
	-0.866, -0.250, 0.433,
	-0.866, -0.433, 0.250,
	-0.866, -0.500, 0.000,
	-0.866, -0.433, -0.250,
	-0.866, -0.250, -0.433,
	-0.866, -0.000, -0.500,
	-0.866, 0.250, -0.433,
	-0.866, 0.433, -0.250,
	-0.643, 0.766, 0.000,
	-0.643, 0.740, 0.198,
	-0.643, 0.663, 0.383,
	-0.643, 0.542, 0.542,
	-0.643, 0.383, 0.663,
	-0.643, 0.198, 0.740,
	-0.643, 0.000, 0.766,
	-0.643, -0.198, 0.740,
	-0.643, -0.383, 0.663,
	-0.643, -0.542, 0.542,
	-0.643, -0.663, 0.383,
	-0.643, -0.740, 0.198,
	-0.643, -0.766, 0.000,
	-0.643, -0.740, -0.198,
	-0.643, -0.663, -0.383,
	-0.643, -0.542, -0.542,
	-0.643, -0.383, -0.663,
	-0.643, -0.198, -0.740,
	-0.643, -0.000, -0.766,
	-0.643, 0.198, -0.740,
	-0.643, 0.383, -0.663,
	-0.643, 0.542, -0.542,
	-0.643, 0.663, -0.383,
	-0.643, 0.740, -0.198,
	-0.342, 0.932, 0.123,
	-0.342, 0.868, 0.360,
	-0.342, 0.746, 0.572,
	-0.342, 0.572, 0.746,
	-0.342, 0.360, 0.868,
	-0.342, 0.123, 0.932,
	-0.342, -0.123, 0.932,
	-0.342, -0.360, 0.868,
	-0.342, -0.572, 0.746,
	-0.342, -0.746, 0.572,
	-0.342, -0.868, 0.360,
	-0.342, -0.932, 0.123,
	-0.342, -0.932, -0.123,
	-0.342, -0.868, -0.360,
	-0.342, -0.746, -0.572,
	-0.342, -0.572, -0.746,
	-0.342, -0.360, -0.868,
	-0.342, -0.123, -0.932,
	-0.342, 0.123, -0.932,
	-0.342, 0.360, -0.868,
	-0.342, 0.572, -0.746,
	-0.342, 0.746, -0.572,
	-0.342, 0.868, -0.360,
	-0.342, 0.932, -0.123,
#endif
	0.000, 1.000, 0.000,
	0.000, 0.966, 0.259,
	0.000, 0.866, 0.500,
	0.000, 0.707, 0.707,
	0.000, 0.500, 0.866,
	0.000, 0.259, 0.966,
	0.000, 0.000, 1.000,
	0.000, -0.259, 0.966,
	0.000, -0.500, 0.866,
	0.000, -0.707, 0.707,
	0.000, -0.866, 0.500,
	0.000, -0.966, 0.259,
	0.000, -1.000, 0.000,
	0.000, -0.966, -0.259,
	0.000, -0.866, -0.500,
	0.000, -0.707, -0.707,
	0.000, -0.500, -0.866,
	0.000, -0.259, -0.966,
	0.000, -0.000, -1.000,
	0.000, 0.259, -0.966,
	0.000, 0.500, -0.866,
	0.000, 0.707, -0.707,
	0.000, 0.866, -0.500,
	0.000, 0.966, -0.259
};

__constant__ float halfBallRight[kHalfBallPoints * 3]{
	0.000, 1.000, 0.000,
	0.000, 0.966, 0.259,
	0.000, 0.866, 0.500,
	0.000, 0.707, 0.707,
	0.000, 0.500, 0.866,
	0.000, 0.259, 0.966,
	0.000, 0.000, 1.000,
	0.000, -0.259, 0.966,
	0.000, -0.500, 0.866,
	0.000, -0.707, 0.707,
	0.000, -0.866, 0.500,
	0.000, -0.966, 0.259,
	0.000, -1.000, 0.000,
	0.000, -0.966, -0.259,
	0.000, -0.866, -0.500,
	0.000, -0.707, -0.707,
	0.000, -0.500, -0.866,
	0.000, -0.259, -0.966,
	0.000, -0.000, -1.000,
	0.000, 0.259, -0.966,
	0.000, 0.500, -0.866,
	0.000, 0.707, -0.707,
	0.000, 0.866, -0.500,
	0.000, 0.966, -0.259,
#ifdef DRAW_BALL
	0.342, 0.932, 0.123,
	0.342, 0.868, 0.360,
	0.342, 0.746, 0.572,
	0.342, 0.572, 0.746,
	0.342, 0.360, 0.868,
	0.342, 0.123, 0.932,
	0.342, -0.123, 0.932,
	0.342, -0.360, 0.868,
	0.342, -0.572, 0.746,
	0.342, -0.746, 0.572,
	0.342, -0.868, 0.360,
	0.342, -0.932, 0.123,
	0.342, -0.932, -0.123,
	0.342, -0.868, -0.360,
	0.342, -0.746, -0.572,
	0.342, -0.572, -0.746,
	0.342, -0.360, -0.868,
	0.342, -0.123, -0.932,
	0.342, 0.123, -0.932,
	0.342, 0.360, -0.868,
	0.342, 0.572, -0.746,
	0.342, 0.746, -0.572,
	0.342, 0.868, -0.360,
	0.342, 0.932, -0.123,
	0.643, 0.766, 0.000,
	0.643, 0.740, 0.198,
	0.643, 0.663, 0.383,
	0.643, 0.542, 0.542,
	0.643, 0.383, 0.663,
	0.643, 0.198, 0.740,
	0.643, 0.000, 0.766,
	0.643, -0.198, 0.740,
	0.643, -0.383, 0.663,
	0.643, -0.542, 0.542,
	0.643, -0.663, 0.383,
	0.643, -0.740, 0.198,
	0.643, -0.766, 0.000,
	0.643, -0.740, -0.198,
	0.643, -0.663, -0.383,
	0.643, -0.542, -0.542,
	0.643, -0.383, -0.663,
	0.643, -0.198, -0.740,
	0.643, -0.000, -0.766,
	0.643, 0.198, -0.740,
	0.643, 0.383, -0.663,
	0.643, 0.542, -0.542,
	0.643, 0.663, -0.383,
	0.643, 0.740, -0.198,
	0.866, 0.500, 0.000,
	0.866, 0.433, 0.250,
	0.866, 0.250, 0.433,
	0.866, 0.000, 0.500,
	0.866, -0.250, 0.433,
	0.866, -0.433, 0.250,
	0.866, -0.500, 0.000,
	0.866, -0.433, -0.250,
	0.866, -0.250, -0.433,
	0.866, -0.000, -0.500,
	0.866, 0.250, -0.433,
	0.866, 0.433, -0.250,
	0.966, 0.259, 0.000,
	0.966, 0.129, 0.224,
	0.966, -0.129, 0.224,
	0.966, -0.259, 0.000,
	0.966, -0.129, -0.224,
	0.966, 0.129, -0.224,
	1.000, 0.000, 0.000
#endif
};

__constant__ uint32_t halfBallIndicesLeft[kHalfBallIndices + 1]{
	0, 1, 2,
0, 2, 3,
0, 3, 4,
0, 4, 5,
0, 5, 6,
0, 6, 1,
1, 7, 8,
1, 8, 9,
1, 9, 2,
2, 9, 10,
2, 10, 11,
2, 11, 3,
3, 11, 12,
3, 12, 13,
3, 13, 4,
4, 13, 14,
4, 14, 15,
4, 15, 5,
5, 15, 16,
5, 16, 17,
5, 17, 6,
6, 17, 18,
6, 18, 7,
6, 7, 1,
7, 19, 20,
7, 20, 21,
7, 21, 8,
8, 21, 22,
8, 22, 23,
8, 23, 9,
9, 23, 24,
9, 24, 25,
9, 25, 10,
10, 25, 26,
10, 26, 27,
10, 27, 11,
11, 27, 28,
11, 28, 29,
11, 29, 12,
12, 29, 30,
12, 30, 31,
12, 31, 13,
13, 31, 32,
13, 32, 33,
13, 33, 14,
14, 33, 34,
14, 34, 35,
14, 35, 15,
15, 35, 36,
15, 36, 37,
15, 37, 16,
16, 37, 38,
16, 38, 39,
16, 39, 17,
17, 39, 40,
17, 40, 41,
17, 41, 18,
18, 41, 42,
18, 42, 19,
18, 19, 7,
19, 43, 20,
20, 43, 44,
20, 44, 21,
21, 44, 45,
21, 45, 22,
22, 45, 46,
22, 46, 23,
23, 46, 47,
23, 47, 24,
24, 47, 48,
24, 48, 25,
25, 48, 49,
25, 49, 26,
26, 49, 50,
26, 50, 27,
27, 50, 51,
27, 51, 28,
28, 51, 52,
28, 52, 29,
29, 52, 53,
29, 53, 30,
30, 53, 54,
30, 54, 31,
31, 54, 55,
31, 55, 32,
32, 55, 56,
32, 56, 33,
33, 56, 57,
33, 57, 34,
34, 57, 58,
34, 58, 35,
35, 58, 59,
35, 59, 36,
36, 59, 60,
36, 60, 37,
37, 60, 61,
37, 61, 38,
38, 61, 62,
38, 62, 39,
39, 62, 63,
39, 63, 40,
40, 63, 64,
40, 64, 41,
41, 64, 65,
41, 65, 42,
42, 65, 66,
42, 66, 19,
19, 66, 43,
43, 67, 68,
44, 43, 68,
44, 68, 69,
45, 44, 69,
45, 69, 70,
46, 45, 70,
46, 70, 71,
47, 46, 71,
47, 71, 72,
48, 47, 72,
48, 72, 73,
49, 48, 73,
49, 73, 74,
50, 49, 74,
50, 74, 75,
51, 50, 75,
51, 75, 76,
52, 51, 76,
52, 76, 77,
53, 52, 77,
53, 77, 78,
54, 53, 78,
54, 78, 79,
55, 54, 79,
55, 79, 80,
56, 55, 80,
56, 80, 81,
57, 56, 81,
57, 81, 82,
58, 57, 82,
58, 82, 83,
59, 58, 83,
59, 83, 84,
60, 59, 84,
60, 84, 85,
61, 60, 85,
61, 85, 86,
62, 61, 86,
62, 86, 87,
63, 62, 87,
63, 87, 88,
64, 63, 88,
64, 88, 89,
65, 64, 89,
65, 89, 90,
66, 65, 90,
66, 90, 67,
43, 66, 67
};

__constant__ uint32_t halfBallIndicesRight[kHalfBallIndices + 1]{
	0, 24, 1,
	1, 24, 25,
	1, 25, 2,
	2, 25, 26,
	2, 26, 3,
	3, 26, 27,
	3, 27, 4,
	4, 27, 28,
	4, 28, 5,
	5, 28, 29,
	5, 29, 6,
	6, 29, 30,
	6, 30, 7,
	7, 30, 31,
	7, 31, 8,
	8, 31, 32,
	8, 32, 9,
	9, 32, 33,
	9, 33, 10,
	10, 33, 34,
	10, 34, 11,
	11, 34, 35,
	11, 35, 12,
	12, 35, 36,
	12, 36, 13,
	13, 36, 37,
	13, 37, 14,
	14, 37, 38,
	14, 38, 15,
	15, 38, 39,
	15, 39, 16,
	16, 39, 40,
	16, 40, 17,
	17, 40, 41,
	17, 41, 18,
	18, 41, 42,
	18, 42, 19,
	19, 42, 43,
	19, 43, 20,
	20, 43, 44,
	20, 44, 21,
	21, 44, 45,
	21, 45, 22,
	22, 45, 46,
	22, 46, 23,
	23, 46, 47,
	23, 47, 0,
	0, 47, 24,
	24, 48, 49,
	25, 24, 49,
	25, 49, 50,
	26, 25, 50,
	26, 50, 51,
	27, 26, 51,
	27, 51, 52,
	28, 27, 52,
	28, 52, 53,
	29, 28, 53,
	29, 53, 54,
	30, 29, 54,
	30, 54, 55,
	31, 30, 55,
	31, 55, 56,
	32, 31, 56,
	32, 56, 57,
	33, 32, 57,
	33, 57, 58,
	34, 33, 58,
	34, 58, 59,
	35, 34, 59,
	35, 59, 60,
	36, 35, 60,
	36, 60, 61,
	37, 36, 61,
	37, 61, 62,
	38, 37, 62,
	38, 62, 63,
	39, 38, 63,
	39, 63, 64,
	40, 39, 64,
	40, 64, 65,
	41, 40, 65,
	41, 65, 66,
	42, 41, 66,
	42, 66, 67,
	43, 42, 67,
	43, 67, 68,
	44, 43, 68,
	44, 68, 69,
	45, 44, 69,
	45, 69, 70,
	46, 45, 70,
	46, 70, 71,
	47, 46, 71,
	47, 71, 48,
	24, 47, 48,
	72, 48, 49,
	72, 49, 50,
	72, 50, 73,
	73, 50, 51,
	73, 51, 52,
	73, 52, 74,
	74, 52, 53,
	74, 53, 54,
	74, 54, 75,
	75, 54, 55,
	75, 55, 56,
	75, 56, 76,
	76, 56, 57,
	76, 57, 58,
	76, 58, 77,
	77, 58, 59,
	77, 59, 60,
	77, 60, 78,
	78, 60, 61,
	78, 61, 62,
	78, 62, 79,
	79, 62, 63,
	79, 63, 64,
	79, 64, 80,
	80, 64, 65,
	80, 65, 66,
	80, 66, 81,
	81, 66, 67,
	81, 67, 68,
	81, 68, 82,
	82, 68, 69,
	82, 69, 70,
	82, 70, 83,
	83, 70, 71,
	83, 71, 48,
	83, 48, 72,
	84, 72, 73,
	84, 73, 74,
	84, 74, 85,
	85, 74, 75,
	85, 75, 76,
	85, 76, 86,
	86, 76, 77,
	86, 77, 78,
	86, 78, 87,
	87, 78, 79,
	87, 79, 80,
	87, 80, 88,
	88, 80, 81,
	88, 81, 82,
	88, 82, 89,
	89, 82, 83,
	89, 83, 72,
	89, 72, 84,
	90, 84, 85,
	90, 85, 86,
	90, 86, 87,
	90, 87, 88,
	90, 88, 89,
	90, 89, 84
};

__device__ void normalize(float& a, float& b, float& c) {
	float temp = 1 / sqrtf(a * a + b * b + c * c);
	deviceDebugPrint("normalize: %f, %f, %f\n", a, b, c);
	a = a * temp;
	b = b * temp;
	c = c * temp;
	deviceDebugPrint("%f, %f, %f\n", a, b, c);
}

__global__ void normalize(float* vectors) {
	size_t idx = threadIdx.x;
	normalize(vectors[3 * idx], vectors[3 * idx + 1], vectors[3 * idx + 2]);
}

void normalize(float* vectors, size_t size) {
	normalize << <1, size >> > (vectors);
	CUDACHECK(cudaGetLastError());
}

__device__ void rotate(float u, float v, float w, float cos_theta,
	float sin_theta, float& a, float& b, float& c)
{
	if (fabsf(cos_theta - 1.0f) < 1e-6) {
		deviceDebugPrint("No need to rotate!");
		return;
	}
	normalize(u, v, w);
	float m[3][3];
	float temp_a = a, temp_b = b, temp_c = c;
	m[0][0] = cos_theta + (u * u) * (1 - cos_theta);
	m[0][1] = u * v * (1 - cos_theta) + w * sin_theta;
	m[0][2] = u * w * (1 - cos_theta) - v * sin_theta;

	m[1][0] = u * v * (1 - cos_theta) - w * sin_theta;
	m[1][1] = cos_theta + v * v * (1 - cos_theta);
	m[1][2] = w * v * (1 - cos_theta) + u * sin_theta;

	m[2][0] = u * w * (1 - cos_theta) + v * sin_theta;
	m[2][1] = v * w * (1 - cos_theta) - u * sin_theta;
	m[2][2] = cos_theta + w * w * (1 - cos_theta);

	a = m[0][0] * temp_a + m[1][0] * temp_b + m[2][0] * temp_c;
	b = m[0][1] * temp_a + m[1][1] * temp_b + m[2][1] * temp_c;
	c = m[0][2] * temp_a + m[1][2] * temp_b + m[2][2] * temp_c;
}

__device__ void rotate(float u, float v, float w, float theta,
	float& a, float& b, float& c) {
	rotate(u, v, w, cosf(theta), sinf(theta), a, b, c);
}

__device__ void resizeAndTrans(float& x, float& y, float& z,
	float scale, float dx, float dy, float dz) {
	x = x * scale + dx;
	y = y * scale + dy;
	z = z * scale + dz;
}

__device__ void crossAndAngle(float& r1, float& r2, float& r3, float& cos_theta,
	float u1, float u2, float u3, float v1, float v2, float v3) {
	r1 = u2 * v3 - u3 * v2;
	r2 = u3 * v1 - u1 * v3;
	r3 = u1 * v2 - u2 * v1;
	cos_theta = (u1 * v1 + u2 * v2 + u3 * v3) / (
		sqrtf(u1*u1 + u2 * u2 + u3 * u3) * sqrtf(v1*v1 + v2 * v2 + v3 * v3));
}

__device__ void crossAndAngle(float& r1, float& r2, float& r3, float& cos_theta,
	float v1, float v2, float v3) {
	crossAndAngle(r1, r2, r3, cos_theta, 1, 0, 0, v1, v2, v3);
}

__device__ void calcHalfBallItem(float* pBufferBase, float* halfBall,
	float x, float y, float z, float size, float r, float g, float b,
	float normX, float normY, float normZ, float alpha, float colorScale) {
	float axisX, axisY, axisZ, cos_theta;
	crossAndAngle(axisX, axisY, axisZ, cos_theta, normX, normY, normZ);
	float sin_theta = sqrtf(1 - cos_theta * cos_theta);

	for (size_t i = 0; i < kHalfBallPoints; ++i) {
		if (threadIdx.x == 0) {
			deviceDebugPrint("%llu, %f, %f, %f", i, halfBall[3 * i], halfBall[3 * i + 1], halfBall[3 * i + 2]);
		}
		pBufferBase[7 * i] = halfBall[3 * i];
		pBufferBase[7 * i + 1] = halfBall[3 * i + 1];
		pBufferBase[7 * i + 2] = halfBall[3 * i + 2];
		pBufferBase[7 * i + 3] = r + colorScale;
		pBufferBase[7 * i + 4] = g + colorScale;
		pBufferBase[7 * i + 5] = b + colorScale;
		pBufferBase[7 * i + 6] = alpha;
		rotate(axisX, axisY, axisZ, cos_theta, sin_theta,
			pBufferBase[7 * i], pBufferBase[7 * i + 1], pBufferBase[7 * i + 2]);
		resizeAndTrans(
			pBufferBase[7 * i], pBufferBase[7 * i + 1], pBufferBase[7 * i + 2],
			size, x, y, z);
	}
}

__device__ void fillHalfBallIndices(size_t indexOffset,
		uint32_t* pIndexBase, uint32_t* indices, uint32_t bufferStart) {
	for (int i = 0; i < kHalfBallIndices; ++i) {
		pIndexBase[i] = indices[i] + indexOffset + bufferStart;
	}
}

__global__ void calcLeftHalfBall(
		const float* dPointsIn, const float* dSizesIn, const float* dColorsIn,
		const size_t* groupOffsets, const size_t* bufferOffsets,
		const size_t* indicesOffsets, float* buffer, uint32_t* dIndicesOut, 
		float alpha, float colorScale, float sizeScale, uint32_t bufferStart) {
	size_t idx = threadIdx.x;
	float* pBufferBase = buffer + bufferOffsets[idx];
	uint32_t* pIndicesBase = dIndicesOut + indicesOffsets[idx];
	size_t indexOffset = bufferOffsets[idx] / 7;
	float size = dSizesIn[groupOffsets[idx]];
	const float* color = dColorsIn + (groupOffsets[idx]) * 3;
	const float* pos = dPointsIn + (groupOffsets[idx]) * 3;

	float normX, normY, normZ;
	if (groupOffsets[idx + 1] - groupOffsets[idx] > 1) {
		normX = *(pos + 3)-*(pos);
		normY = *(pos + 4) - *(pos + 1);
		normZ = *(pos + 5) - *(pos + 2);
	} else {
		normX = 1;
		normY = normZ = 0;
	}

	calcHalfBallItem(
		pBufferBase, halfBallLeft, pos[0], pos[1], pos[2], size * sizeScale,
		color[0], color[1], color[2], normX, normY, normZ, alpha, colorScale);
	fillHalfBallIndices(indexOffset, pIndicesBase, halfBallIndicesLeft, bufferStart);
}

__global__ void calcRightHalfBall(
		const float* dPointsIn, const float* dSizesIn, const float* dColorsIn,
		const size_t* groupOffsets, const size_t* bufferOffsets,
		const size_t* indicesOffsets, float* buffer, uint32_t* dIndicesOut,
		float alpha, float colorScale, float sizeScale, uint32_t bufferStart) {
	size_t idx = threadIdx.x;
	int64_t num_circles = static_cast<int64_t>(
		groupOffsets[idx + 1] - groupOffsets[idx]) - 2;

	int64_t bufferOffset = static_cast<int64_t>(bufferOffsets[idx]) +
		(kCirclePoints * num_circles + kHalfBallPoints) * 7;
	float* pBufferBase = buffer + bufferOffset;
	int64_t indicesOffset = static_cast<int64_t>(indicesOffsets[idx]) +
		kHalfBallIndices + (num_circles + 1) * kCircleIndices;
	uint32_t* pIndicesBase = dIndicesOut + indicesOffset;

	int64_t indexOffset = static_cast<int64_t>(bufferOffsets[idx] / 7) +
		kHalfBallPoints + num_circles * kCirclePoints;
	deviceDebugPrint("%llu: %lld, %lld, %lld, %lld\n",
		idx, num_circles, bufferOffset, indicesOffset, indexOffset);
	float size = dSizesIn[groupOffsets[idx + 1] - 1];
	const float* color = dColorsIn + (groupOffsets[idx + 1] - 1) * 3;
	const float* pos = dPointsIn + (groupOffsets[idx + 1] - 1) * 3;

	float normX, normY, normZ;
	if (groupOffsets[idx + 1] - groupOffsets[idx] > 1) {
		normX = *(pos)-*(pos - 3);
		normY = *(pos + 1) - *(pos - 2);
		normZ = *(pos + 2) - *(pos - 1);
	} else {
		normX = 1;
		normY = normZ = 0;
	}
	calcHalfBallItem(
		pBufferBase, halfBallRight, pos[0], pos[1], pos[2], size * sizeScale,
		color[0], color[1], color[2], normX, normY, normZ, alpha, colorScale);
	fillHalfBallIndices(
		indexOffset, pIndicesBase, halfBallIndicesRight, bufferStart);
}

void calcHalfBall(
		const float* dPointsIn, const float* dSizesIn, const float* dColorsIn,
		const size_t* groupOffsets, size_t nGroups, const size_t* bufferOffsets,
		const size_t* indicesOffsets, float* buffer, uint32_t* dIndicesOut,
		float alpha, float colorScale, float sizeScale, uint32_t bufferStart) {

	// leftBall
	calcLeftHalfBall << <1, nGroups >> > (dPointsIn, dSizesIn, dColorsIn,
		groupOffsets, bufferOffsets, indicesOffsets, buffer, dIndicesOut,
		alpha, colorScale, sizeScale, bufferStart);
	CUDACHECK(cudaGetLastError());
	CUDACHECK(cudaDeviceSynchronize());

	// rightBall
	calcRightHalfBall << <1, nGroups >> > (dPointsIn, dSizesIn, dColorsIn,
		groupOffsets, bufferOffsets, indicesOffsets, buffer, dIndicesOut,
		alpha, colorScale, sizeScale, bufferStart);
	CUDACHECK(cudaGetLastError());
	CUDACHECK(cudaDeviceSynchronize());
}

__device__ void calcCircularTruncatedConeItem(float* pBufferBase,
		size_t bufferOffset, uint32_t* pIndexBase, const float* circle,
		const uint32_t* indices, float x, float y, float z, float size,
		float r, float g, float b, float normX, float normY, float normZ,
		float alpha, float colorScale, uint32_t bufferStart) {
	float axisX, axisY, axisZ, cos_theta;
	crossAndAngle(axisX, axisY, axisZ, cos_theta, normX, normY, normZ);
	float sin_theta = sqrtf(1 - cos_theta * cos_theta);
	deviceDebugPrint("|||--- %f, %f, %f, %f, %f ---|||\n",
		axisX, axisY, axisZ, cos_theta, sin_theta);
	for (size_t i = 0; i < kCirclePoints; ++i) {
		pBufferBase[7 * i] = circle[3 * i];
		pBufferBase[7 * i + 1] = circle[3 * i + 1];
		pBufferBase[7 * i + 2] = circle[3 * i + 2];
		pBufferBase[7 * i + 3] = r + colorScale;
		pBufferBase[7 * i + 4] = g + colorScale;
		pBufferBase[7 * i + 5] = b + colorScale;
		pBufferBase[7 * i + 6] = alpha;
		rotate(axisX, axisY, axisZ, cos_theta, sin_theta,
			pBufferBase[7 * i], pBufferBase[7 * i + 1], pBufferBase[7 * i + 2]);
		resizeAndTrans(
			pBufferBase[7 * i], pBufferBase[7 * i + 1], pBufferBase[7 * i + 2],
			size, x, y, z);
		uint32_t baseIndex = bufferOffset - kCirclePoints + bufferStart;
		pIndexBase[6 * i] = baseIndex + indices[6 * i];
		pIndexBase[6 * i + 1] = baseIndex + indices[6 * i + 1];
		pIndexBase[6 * i + 2] = baseIndex + indices[6 * i + 2];
		pIndexBase[6 * i + 3] = baseIndex + indices[6 * i + 3];
		pIndexBase[6 * i + 4] = baseIndex + indices[6 * i + 4];
		pIndexBase[6 * i + 5] = baseIndex + indices[6 * i + 5];
	}
}

using trans_func_t = size_t(*)(size_t);

__global__ void calcCircularTruncatedConeGroup(
		const float* dPointsIn, const float* dSizesIn, const float* dColorsIn,
		const size_t* groupOffsets, const size_t* bufferOffsets,
		const size_t* indicesOffsets, float* buffer, uint32_t* dIndicesOut,
		const float* circle, const uint32_t* indices, trans_func_t trans,
		float alpha, float colorScale, float sizeScale, uint32_t bufferStart) {
	deviceDebugPrint("in : %u, %u\n", blockIdx.x, threadIdx.x);
	size_t idx = trans(threadIdx.x);
	size_t bidx = blockIdx.x;
	size_t totalNum = groupOffsets[bidx + 1] - groupOffsets[bidx];
	deviceDebugPrint("here~~ : %llu, %llu\n", idx, totalNum);
	if ((idx + 1) < totalNum) {
		deviceDebugPrint("idx less than totalNum : %llu, %llu\n", idx, totalNum);
		size_t bufferOffset = bufferOffsets[bidx] / 7 +
			(kHalfBallPoints + kCirclePoints * (idx - 1));
		float* pBufferBase = buffer + bufferOffset * 7;
		uint32_t* pIndicesBase = dIndicesOut + indicesOffsets[bidx] +
			kHalfBallIndices + kCircleIndices * (idx - 1);
		float size = dSizesIn[groupOffsets[bidx] + idx];
		const float* color = dColorsIn + (groupOffsets[bidx] + idx) * 3;
		const float* pos = dPointsIn + (groupOffsets[bidx] + idx) * 3;
		float normX = *(pos + 3) - *(pos - 3);
		float normY = *(pos + 4) - *(pos - 2);
		float normZ = *(pos + 5) - *(pos - 1);
		calcCircularTruncatedConeItem(
			pBufferBase, bufferOffset, pIndicesBase, circle, indices,
			*pos, *(pos + 1), *(pos + 2), size * sizeScale, *color,
			*(color + 1), *(color + 2), normX, normY, normZ, alpha,
			colorScale, bufferStart);
	}
}

__global__ void calcFinalIndices(
		const size_t* groupOffsets, const size_t* indicesOffsets,
		const size_t* bufferOffsets, uint32_t* dIndicesOut,
		uint32_t bufferStart) {
	size_t idx = threadIdx.x;
	if (groupOffsets[idx + 1] - groupOffsets[idx] == 1) return;
	size_t offset = groupOffsets[idx + 1] - groupOffsets[idx] - 2;
	uint32_t* pIndicesBase = dIndicesOut + indicesOffsets[idx] +
		kHalfBallIndices + kCircleIndices * offset;
	uint32_t baseIndex = bufferOffsets[idx] / 7 +
		kHalfBallPoints + kCirclePoints * (offset - 1) + bufferStart;
	deviceDebugPrint("fill final : %llu, offset: %llu, off: %llu\n",
		idx, offset, baseIndex);
	for (int i = 0; i < kCircleIndices; ++i) {
		pIndicesBase[i] = baseIndex + indices2[i];
	}
}

__device__ size_t oddTrans(size_t x) {
	return x * 2 + 1;
}
__device__ trans_func_t d_odd = oddTrans;

__device__ size_t evenTrans(size_t x) {
	return (x + 1) * 2;
}
__device__ trans_func_t d_even = evenTrans;

void calcCircularTruncatedCone(
		const float* dPointsIn, const float* dSizesIn, const float* dColorsIn,
		const size_t* groupOffsets, size_t maxSize, size_t nGroups,
		const size_t* bufferOffsets, const size_t* indicesOffsets,
		float* buffer, uint32_t* dIndicesOut, float alpha,
		float colorScale, float sizeScale, uint32_t bufferStart) {
	float *circle = nullptr;
	uint32_t *indices = nullptr;
	trans_func_t odd;
	CUDACHECK(cudaMemcpyFromSymbol(&odd, d_odd, sizeof(trans_func_t)));

	// 使用__constant__地址的方法来自于
	// https://devtalk.nvidia.com/default/topic/487853/can-i-use-__constant__-memory-with-pointer-to-it-as-kernel-arg/
	CUDACHECK(cudaGetSymbolAddress((void**)&circle, circle2));
	CUDACHECK(cudaGetSymbolAddress((void**)&indices, indices1));
	calcCircularTruncatedConeGroup << <nGroups, maxSize >> > (
		dPointsIn, dSizesIn, dColorsIn, groupOffsets,
		bufferOffsets, indicesOffsets, buffer, dIndicesOut,
		circle, indices, odd, alpha, colorScale, sizeScale, bufferStart);
	CUDACHECK(cudaGetLastError());

	trans_func_t even;
	cudaMemcpyFromSymbol(&even, d_even, sizeof(trans_func_t));
	cudaGetSymbolAddress((void**)&circle, circle1);
	cudaGetSymbolAddress((void**)&indices, indices2);
	calcCircularTruncatedConeGroup << <nGroups, maxSize >> > (
		dPointsIn, dSizesIn, dColorsIn, groupOffsets,
		bufferOffsets, indicesOffsets, buffer, dIndicesOut,
		circle, indices, even, alpha, colorScale, sizeScale, bufferStart);
	CUDACHECK(cudaGetLastError());
	// 填充最后一个indices
	calcFinalIndices << <1, nGroups >> > (
		groupOffsets, indicesOffsets, bufferOffsets, dIndicesOut, bufferStart);
	CUDACHECK(cudaGetLastError());
}

__global__ void calcOffsets(const size_t* groupOffsets,
		size_t* bufferOffsets, size_t* indicesOffsets) {
	size_t idx = threadIdx.x;
	bufferOffsets[idx] = groupOffsets[idx] * kCirclePoints * 7 +
		14 * idx * (kHalfBallPoints - kCirclePoints);
	indicesOffsets[idx] = (groupOffsets[idx] - idx) * kCircleIndices +
		2 * idx * kHalfBallIndices;
	deviceDebugPrint("%llu - %llu : %llu, %llu\n", idx, groupOffsets[idx],
		bufferOffsets[idx], indicesOffsets[idx]);
}

size_t pointToLine(
		const float* dPointsIn,
		const float* dSizesIn,
		const float* dColorsIn,
		size_t maxSizePerGroup,
		size_t* const dGroupOffsets,
		size_t nGroups,
		float* dBuffer,
		uint32_t* dIndicesOut,
		float outterAlpha,
		float innerSize,
	    float innerColor
		/*,float innerColorRate*/) {
	size_t *bufferOffsets, *indicesOffsets;
	//float *dInnerColorRatesTemp, *dInnerColorRates;
	CUDACHECK(cudaMallocAlign(&bufferOffsets, (nGroups + 1) * sizeof(size_t)));
	CUDACHECK(cudaMallocAlign(&indicesOffsets, (nGroups + 1) * sizeof(size_t)));
	//CUDACHECK(cudaMallocAlign(&dInnerColorRatesTemp, maxSizePerGroup * sizeof(float)));
	//CUDACHECK(cudaMallocAlign(&dInnerColorRates, maxSizePerGroup * sizeof(float)));
	//fill(dInnerColorRatesTemp, innerColorRate, maxSizePerGroup);
	//cuMul(dInnerColorRates, dInnerColorRatesTemp, maxSizePerGroup);

	calcOffsets << <1, nGroups + 1 >> > (dGroupOffsets,
		bufferOffsets, indicesOffsets);
	CUDACHECK(cudaGetLastError());
	calcHalfBall(dPointsIn, dSizesIn, dColorsIn, dGroupOffsets,
		nGroups, bufferOffsets, indicesOffsets, dBuffer, dIndicesOut, 1.0f, innerColor, innerSize, 0);
	calcCircularTruncatedCone(
		dPointsIn, dSizesIn, dColorsIn, dGroupOffsets, maxSizePerGroup,
		nGroups, bufferOffsets, indicesOffsets, dBuffer, dIndicesOut, 1.0f, innerColor, innerSize, 0);

	size_t totalBuffers, totalIndices;
	CUDACHECK(cudaMemcpy(&totalBuffers, bufferOffsets + nGroups,
		sizeof(size_t), cudaMemcpyDeviceToHost));
	CUDACHECK(cudaMemcpy(&totalIndices, indicesOffsets + nGroups,
		sizeof(size_t), cudaMemcpyDeviceToHost));
	
	calcHalfBall(dPointsIn, dSizesIn, dColorsIn, dGroupOffsets,
		nGroups, bufferOffsets, indicesOffsets, dBuffer + totalBuffers,
		dIndicesOut + totalIndices, outterAlpha, 0.0f, 1.0f, totalBuffers / 7);
	calcCircularTruncatedCone(
		dPointsIn, dSizesIn, dColorsIn, dGroupOffsets, maxSizePerGroup,
		nGroups, bufferOffsets, indicesOffsets, dBuffer + totalBuffers,
		dIndicesOut + totalIndices, outterAlpha, 0.0f, 1.0f, totalBuffers / 7);
	
	CUDACHECK(cudaFree(bufferOffsets));
	CUDACHECK(cudaFree(indicesOffsets));

	return 2 * totalIndices;
}

}
