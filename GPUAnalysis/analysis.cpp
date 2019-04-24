#include <cstdio>
#include <cuda_runtime.h>
#include "kernel.h"

inline int _ConvertSMVer2Cores(int major, int minor) {
	// Defines for GPU Architecture types (using the SM version to determine
	// the # of cores per SM
	typedef struct {
		int SM;  // 0xMm (hexidecimal notation), M = SM Major version,
				 // and m = SM minor version
		int Cores;
	} sSMtoCores;

	sSMtoCores nGpuArchCoresPerSM[] = {
		{ 0x30, 192 },
		{ 0x32, 192 },
		{ 0x35, 192 },
		{ 0x37, 192 },
		{ 0x50, 128 },
		{ 0x52, 128 },
		{ 0x53, 128 },
		{ 0x60,  64 },
		{ 0x61, 128 },
		{ 0x62, 128 },
		{ 0x70,  64 },
		{ 0x72,  64 },
		{ 0x75,  64 },
		{ -1, -1 }
	};

	int index = 0;

	while (nGpuArchCoresPerSM[index].SM != -1) {
		if (nGpuArchCoresPerSM[index].SM == ((major << 4) + minor)) {
			return nGpuArchCoresPerSM[index].Cores;
		}

		index++;
	}

	// If we don't find the values, we default use the previous one
	// to run properly
	printf(
		"MapSMtoCores for SM %d.%d is undefined."
		"  Default to use %d Cores/SM\n",
		major, minor, nGpuArchCoresPerSM[index - 1].Cores);
	return nGpuArchCoresPerSM[index - 1].Cores;
}

int main() {
	cudaDeviceProp deviceProp;
	size_t dev = 0;
	CUDACHECK(cudaSetDevice(dev));
	CUDACHECK(cudaGetDeviceProperties(&deviceProp, dev));

	printf("\nDevice %d: \"%s\"\n", dev, deviceProp.name);
	printf("GPU Max Clock rate: %0.2f (GHz)\n", deviceProp.clockRate * 1e-6f);

	float freq = deviceProp.clockRate * 1e-6f; // GHz
	int nCore = _ConvertSMVer2Cores(deviceProp.major, deviceProp.minor);
	int nSM = deviceProp.multiProcessorCount;
	printf("(%2d) Multiprocessors, (%2d) CUDA Cores/MP: %d CUDA Cores\n",
		nSM,
		nCore,
		nSM*nCore);

	float peakPerf = nSM * nCore * freq * 2;
	printf("GPU Peak Performance is %0.2f GFlops.\n", peakPerf);
	printf("Maximum number of threads per block: %d - (%d, %d, %d)\n",
		deviceProp.maxThreadsPerBlock,
		deviceProp.maxThreadsDim[0],
		deviceProp.maxThreadsDim[1],
		deviceProp.maxThreadsDim[2]);
	printf("Maximum number of blocks per grid: (%d, %d, %d)\n",
		deviceProp.maxGridSize[0],
		deviceProp.maxGridSize[1],
		deviceProp.maxGridSize[2]);
	printf("GPU SharedMemPerBlock: %llu\n", deviceProp.sharedMemPerBlock);
}