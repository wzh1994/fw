#include "hostmethods.hpp"
#include "../fw/exceptions.h"
#include <random>
#include <ctime>
#include <mutex>
#include <cmath>
#include <corecrt_math_defines.h>

namespace hostMethod {
	size_t normalFireworkDirections(float* directions,
			size_t nIntersectingSurfaceParticle,
			float xRate, float yRate, float zRate,
			float xStretch, float yStretch, float zStretch) {
		static std::once_flag inited;
		static time_t seed;
		std::call_once(inited, []() {
			seed = time(0);
		});
		srand(seed);
		size_t nGroups = nIntersectingSurfaceParticle / 2 + 1;
		int num = nIntersectingSurfaceParticle / 2 + 1;
		float angle = 2 * M_PI / nIntersectingSurfaceParticle;
		int numDirections = 0;
		for (int i = 0; i < num; i++) {
			float theta = i * angle;
			int num2 = nIntersectingSurfaceParticle * sin(theta);
			if (num2 <= 1) num2 = 1;
			float angle2 = 2 * M_PI / num2;
			for (int j = 0; j < num2; j++) {
				float gamma = j * angle2;
				float x = sin(theta)*sin(gamma);
				float y = cos(theta);
				float z = sin(theta)*cos(gamma);
				float rdx = 1 + xStretch + ((float)rand() /
					(float)RAND_MAX - 0.5f)* xRate;
				float rdy = 1 + yStretch + ((float)rand() /
					(float)RAND_MAX - 0.5f)* yRate;
				float rdz = 1 + zStretch + ((float)rand() /
					(float)RAND_MAX - 0.5f)* zRate;
				directions[3 * numDirections] = x * rdx;
				directions[3 * numDirections + 1] = y * rdy;
				directions[3 * numDirections + 2] = z * rdz;
				++numDirections;
			}
		}
		return numDirections;
	}
	namespace circleFw {
		void crossAndAngle(float& r1, float& r2, float& r3, float& cos_theta,
			float u1, float u2, float u3, float v1, float v2, float v3) {
			r1 = u2 * v3 - u3 * v2;
			r2 = u3 * v1 - u1 * v3;
			r3 = u1 * v2 - u2 * v1;
			cos_theta = (u1 * v1 + u2 * v2 + u3 * v3) / (
				sqrt(u1*u1 + u2 * u2 + u3 * u3) * sqrt(
					v1 * v1 + v2 * v2 + v3 * v3));
		}

		void crossAndAngle(float& r1, float& r2, float& r3, float& cos_theta,
			float v1, float v2, float v3) {
			crossAndAngle(r1, r2, r3, cos_theta, v1, v2, v3, 0, 1, 0);
		}

		void rotate(float u, float v, float w, float cos_theta,
			float sin_theta, float& a, float& b, float& c)
		{
			if (fabsf(cos_theta - 1.0f) < 1e-6) {
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

		void rotate(float u, float v, float w, float theta,
			float& a, float& b, float& c) {
			rotate(u, v, w, cos(theta), sin(theta), a, b, c);
		}

		void circleFireworkDirections(float* directions, float angle1,
			float angle2, float xStretch, float xRate, float yStretch,
			float yRate, float zStretch, float zRate, curandState *devStates,
			float normX, float normY, float normZ) {
			
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
			float* directions, size_t nIntersectingSurfaceParticle,
			float* norm, float angleFromNormal,
			float xRate, float yRate, float zRate,
			float xStretch, float yStretch, float zStretch) {
		static time_t t = time(0);
		srand(t);
		float angle1 = M_PI * angleFromNormal / 180.0f;
		float angle2 = M_PI * 2 / static_cast<float>(
			nIntersectingSurfaceParticle);
		float normX = norm[0], normY = norm[1], normZ = norm[2];
		circleFw::normalize(normX, normY, normZ);
		for (size_t idx = 0; idx < nIntersectingSurfaceParticle; ++idx) {
			float rx = 1 + xStretch + ((float)rand() /
				(float)RAND_MAX - 0.5f)* xRate;
			float ry = 1 + yStretch + ((float)rand() /
				(float)RAND_MAX - 0.5f)* yRate;
			float rz = 1 + zStretch + ((float)rand() /
				(float)RAND_MAX - 0.5f)* zRate;
			directions[3 * idx] = sin(angle1) * cos(angle2 * idx) * rx;
			directions[3 * idx + 1] = cos(angle1) * ry;
			directions[3 * idx + 2] = sin(angle1) * sin(angle2 * idx) * rz;
			float axisX, axisY, axisZ, cos_theta;
			circleFw::crossAndAngle(
				axisX, axisY, axisZ, cos_theta, normX, normY, normZ);
			float sin_theta = sqrt(1 - cos_theta * cos_theta);
			circleFw::rotate(
				axisX, axisY, axisZ, cos_theta, sin_theta, directions[3 * idx],
				directions[3 * idx + 1], directions[3 * idx + 2]);
		}
		return nIntersectingSurfaceParticle;
	}

	size_t strafeFireworkDirections(
			float* directions, size_t nGroups, size_t size) {
		for (size_t bidx = 0; bidx < nGroups; ++bidx) {
			for (size_t tidx = 0; tidx < size; ++tidx) {
				size_t idx = bidx * size + tidx;
				float theta = M_PI * (30.0f + static_cast<float>(bidx) * 5.0f +
					(30.0f - 2.5f * static_cast<float>(bidx)) *
					static_cast<float>(tidx)) / 180.0f;
				directions[3 * idx] = cosf(theta);
				directions[3 * idx + 1] = sinf(theta);
				directions[3 * idx + 2] = 0;
			}
		}
		return nGroups * size;
	}

	void getSubFireworkPositions(float* startPoses, float* directions,
			const float* subDirs, size_t nDirs, size_t nSubDirs,
			size_t nSubGroups, const float* centrifugalPos_, size_t startFrame,
			size_t kShift, const float* shiftX_, const float* shiftY_) {
		size_t stride = nDirs / nSubGroups;
		const float* relativePos = centrifugalPos_ + startFrame;
		for (size_t bid = 0; bid < nSubGroups; ++bid) {
			for (size_t tid = 0; tid < nSubDirs; ++tid) {
				size_t idx = bid * nSubDirs + tid;
				const float* dir = directions + bid * stride * 3;
				float* targetDir =
					directions + (nDirs + bid * nSubDirs + tid) * 3;
				startPoses[3 * idx] = dir[0] * *relativePos + shiftX_[kShift];
				startPoses[3 * idx + 1] =
					dir[1] * *relativePos + shiftY_[kShift];
				startPoses[3 * idx + 2] = dir[2] * *relativePos;
				targetDir[0] = subDirs[tid * 3];
				targetDir[1] = subDirs[tid * 3 + 1];
				targetDir[2] = subDirs[tid * 3 + 2];
			}
		}

	}
	
}
