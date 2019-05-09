#include "kernel.h"
#include "kernels.h"
#include "cuda_runtime.h"
#include "corecrt_math.h"
#include "utils.h"
#include "test.h"
#include <corecrt_math_defines.h>
#include <chrono>
#include <mutex>

// 为了让__syncthreads()通过语法检查
#ifndef __CUDACC__
#define __CUDACC__
#endif // !__CUDACC__
#include "device_launch_parameters.h"
#include "device_functions.h"
#include "curand_kernel.h" 
#include <cstdio>

namespace cudaKernel {

__global__ void kernel_set_random(
		curandState *curandStates, long seed) {
	int idx = threadIdx.x + blockIdx.x * blockDim.x;
	curand_init(seed, idx, 0, curandStates + idx);
}

namespace normalFw{

__global__ void getNumbersPerGroup(size_t *sizes, float *angles, size_t n) {
	angles[threadIdx.x] =
		static_cast<float>(threadIdx.x) * M_PI * 2 / static_cast<float>(n);
	sizes[threadIdx.x] =
		static_cast<size_t>(fmaxf(1.0f, n * sinf(angles[threadIdx.x])));
}

__global__ void getDirections(float *directions, const float* angles,
		const size_t* sizes, const size_t* offsets, float xStretch,
		float xRate, float yStretch, float yRate, float zStretch,
		float zRate, curandState *devStates) {
	size_t bid = blockIdx.x;
	size_t tid = threadIdx.x;
	size_t idx = offsets[bid] + tid;
	size_t n = sizes[bid];
	float* basePtr = directions + idx * 3;
	if (tid < n) {
		float angle1 = angles[bid];
		float angle2 =
			static_cast<float>(threadIdx.x) * M_PI * 2 / static_cast<float>(n);
		xRate = (2 * abs(curand_uniform(devStates + idx)) - 1) * xRate + xStretch;
		yRate = (2 * abs(curand_uniform(devStates + idx)) - 1) * yRate + yStretch;
		zRate = (2 * abs(curand_uniform(devStates + idx)) - 1) * zRate + zStretch;
		
		basePtr[0] = cosf(angle1) * xRate;
		basePtr[1] = sinf(angle1) * cosf(angle2) * yRate;
		basePtr[2] = sinf(angle1) * sinf(angle2) * zRate;
	}
}

}

void initFixedRand(curandState *dev_states, size_t size) {
	static clock_t seed;
	static std::once_flag inited;
	std::call_once(inited, [] { seed = clock(); });
	size_t blockDim = 256;
	size_t gridDim = ceilAlign(size, blockDim);
	kernel_set_random<<<gridDim, blockDim>>>(dev_states, seed);
	CUDACHECK(cudaGetLastError());
}

void initRand(curandState *dev_states, size_t size) {
	clock_t seed = clock();
	size_t blockDim = 256;
	size_t gridDim = ceilAlign(size, blockDim);
	kernel_set_random << <gridDim, blockDim >> > (dev_states, seed);
	CUDACHECK(cudaGetLastError());
}

size_t normalFireworkDirections(float* dDirections,
		size_t nIntersectingSurfaceParticle,
		float xRate, float yRate, float zRate,
		float xStretch, float yStretch, float zStretch) {
	size_t nGroups = nIntersectingSurfaceParticle / 2 + 1;
	size_t *dSizes, *dOffsets;
	float* dAngles;
	cudaMallocAlign(&dSizes, nGroups * sizeof(size_t));
	cudaMallocAlign(&dOffsets, (nGroups + 1) * sizeof(size_t));
	cudaMallocAlign(&dAngles, (nGroups) * sizeof(float));
	normalFw::getNumbersPerGroup<<<1, nGroups>>>(
		dSizes, dAngles, nIntersectingSurfaceParticle);
	CUDACHECK(cudaGetLastError());
	CUDACHECK(cudaMemset(dOffsets, 0, (nGroups + 1) * sizeof(size_t)));
	cuSum(dOffsets + 1, dSizes, nGroups);
	CUDACHECK(cudaGetLastError());
	size_t numDirections;
	CUDACHECK(cudaMemcpy(&numDirections, dOffsets + nGroups,
		sizeof(size_t), cudaMemcpyDeviceToHost));
	curandState *devStates;
	CUDACHECK(cudaMallocAlign(&devStates, sizeof(curandState) * numDirections));
	initFixedRand(devStates, numDirections);

	normalFw::getDirections << <nGroups, nIntersectingSurfaceParticle >> > (
		dDirections, dAngles, dSizes, dOffsets, xStretch, xRate,
		yStretch, yRate, zStretch, zRate, devStates);
	cudaFreeAll(dSizes, dOffsets, dAngles, devStates);
	return numDirections;
}

namespace circleFw {
__device__ void crossAndAngle(float& r1, float& r2, float& r3, float& cos_theta,
	float u1, float u2, float u3, float v1, float v2, float v3) {
	r1 = u2 * v3 - u3 * v2;
	r2 = u3 * v1 - u1 * v3;
	r3 = u1 * v2 - u2 * v1;
	cos_theta = (u1 * v1 + u2 * v2 + u3 * v3) / (
		sqrtf(u1*u1 + u2 * u2 + u3 * u3) * sqrtf(v1 * v1 + v2 * v2 + v3 * v3));
}

__device__ void crossAndAngle(float& r1, float& r2, float& r3, float& cos_theta,
	float v1, float v2, float v3) {
	crossAndAngle(r1, r2, r3, cos_theta, v1, v2, v3, 0, 1, 0);
}

__device__ void rotate(float u, float v, float w, float cos_theta,
	float sin_theta, float& a, float& b, float& c)
{
	if (fabsf(cos_theta - 1.0f) < 1e-6) {
		deviceDebugPrint("No need to rotate!");
		return;
	}
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

__global__ void circleFireworkDirections(float* directions, float angle1,
		float angle2, float xStretch, float xRate, float yStretch,
		float yRate, float zStretch, float zRate, curandState *devStates,
		float normX, float normY, float normZ) {
	size_t idx = threadIdx.x;
	xRate = (2 * abs(curand_uniform(devStates + idx)) - 1) * xRate + xStretch;
	yRate = (2 * abs(curand_uniform(devStates + idx)) - 1) * yRate + yStretch;
	zRate = (2 * abs(curand_uniform(devStates + idx)) - 1) * zRate + zStretch;
	directions[3 * idx] = sinf(angle1) * cosf(angle2 * idx);
	directions[3 * idx + 1] = cosf(angle1);
	directions[3 * idx + 2] = sinf(angle1) * sinf(angle2 * idx);
	float axisX, axisY, axisZ, cos_theta;
	crossAndAngle(axisX, axisY, axisZ, cos_theta, normX, normY, normZ);
	float sin_theta = sqrtf(1 - cos_theta * cos_theta);
	rotate(axisX, axisY, axisZ, cos_theta, sin_theta,
		directions[3 * idx], directions[3 * idx + 1], directions[3 * idx + 2]);
}

void normalize(float& a, float& b, float& c) {
	FW_ASSERT(a != 0 || b != 0 || c != 0);
	float temp = 1 / std::sqrt(a * a + b * b + c * c);
	a = a * temp;
	b = b * temp;
	c = c * temp;
}
}

size_t circleFireworkDirections(
		float* dDirections, size_t nIntersectingSurfaceParticle,
		float* norm, float angleFromNormal, float xRate, float yRate,
		float zRate, float xStretch, float yStretch, float zStretch) {
	float angle2 =  M_PI * 2 / static_cast<float>(nIntersectingSurfaceParticle);
	curandState *devStates;
	CUDACHECK(cudaMallocAlign(&devStates, sizeof(curandState) *
		nIntersectingSurfaceParticle));
	initFixedRand(devStates, nIntersectingSurfaceParticle);

	float normX = norm[0], normY = norm[1], normZ = norm[2];
	circleFw::normalize(normX, normY, normZ);
	circleFw::circleFireworkDirections <<<1, nIntersectingSurfaceParticle >>> (
		dDirections, M_PI * angleFromNormal / 180.0f, angle2, xStretch, xRate,
		yStretch, yRate, zStretch, zRate, devStates, normX, normY, normZ);
	CUDACHECK(cudaGetLastError());
	cudaFreeAll(devStates);
	return nIntersectingSurfaceParticle;
}

namespace starfeFw{
__global__ void strafeFireworkDirections(float* directions) {
	size_t bidx = blockIdx.x;
	size_t tidx = threadIdx.x;
	size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
	float theta = M_PI * (30.0f + static_cast<float>(bidx) * 5.0f +
		(30.0f - 2.5f * static_cast<float>(bidx)) *
		static_cast<float>(tidx)) / 180.0f;
	directions[3 * idx] = cosf(theta);
	directions[3 * idx + 1] = sinf(theta);
	directions[3 * idx + 2] = 0;
}
}

size_t strafeFireworkDirections(
		float* dDirections, size_t nGroups, size_t size) {
	starfeFw::strafeFireworkDirections <<<nGroups, size >>> (dDirections);
	return nGroups * size;
}

}  // end namespace cudaKernel