#include "hostmethods.hpp"
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
				float rdx = 1 + xStretch + ((float)rand() / (float)RAND_MAX - 0.5f)* xRate;
				float rdy = 1 + yStretch + ((float)rand() / (float)RAND_MAX - 0.5f)* yRate;
				float rdz = 1 + zStretch + ((float)rand() / (float)RAND_MAX - 0.5f)* zRate;
				directions[3 * numDirections] = x * rdx;
				directions[3 * numDirections + 1] = y * rdy;
				directions[3 * numDirections + 2] = z * rdz;
				++numDirections;
			}
		}
		return numDirections;
	}
	
}
