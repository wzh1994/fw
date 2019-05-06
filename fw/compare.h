#pragma once
#include <unordered_map>
#include <string>
#include "utils.h"
#include "test.h"
#include "exceptions.h"
using namespace cudaKernel;
namespace compare {

using string_t = std::string;
template<typename T>
bool itemClose(T a, T b) {
	return  a == b;
}

template<>
bool itemClose(float a, float b) {
	return abs(a - b) < 1e-5f;
}
template<>
bool itemClose(double a, double b) {
	return abs(a - b) < 1e-5;
}

template<typename T>
bool allClose(const T* dA, const T* dB, size_t size, bool debug=false) {
	T *hA = new T[size], *hB = new T[size];
	CUDACHECK(cudaMemcpy(hA, dA, size * sizeof(T), cudaMemcpyDeviceToHost));
	CUDACHECK(cudaMemcpy(hB, dB, size * sizeof(T), cudaMemcpyDeviceToHost));
	for (size_t i = 0; i < size; ++i) {
		if (!itemClose(hA[i], hB[i])) {
			//if (debug) {
				printSplitLine("Error occur at" + std::to_string(i));
				show(dA, size);
				show(dB, size);
				printSplitLine("Error message echo done!");
			//}
			return false;
		}
	}
	return true;
}
class Compare {
	struct Pointer {
		void* p;
		size_t size;
		Pointer(size_t size) :p(nullptr), size(size){};
	};
	std::unordered_map<string_t, Pointer> contents_;
public:
	template <typename T>
	bool compare(
			string_t name, const T* p, size_t size, bool debug=false) {
		if (contents_.count(name) > 0) {
			auto res = contents_.find(name);
			FW_ASSERT(res != contents_.end());
			return allClose(static_cast<T*>(res->second.p), p, size, debug);
		} else {
			float* temp;
			auto res = contents_.emplace(std::piecewise_construct,
				std::forward_as_tuple(name), std::forward_as_tuple(size));
			cudaMallocAndCopy(
				reinterpret_cast<T*&>(res.first->second.p), p, size);
			if (debug) {
				show(static_cast<T*>(res.first->second.p), size);
			}
		}
		return true;
	}

	template<typename T>
	bool compare(string_t name, size_t idx, const T* p,
			size_t size, bool debug = false) {
		return compare(name + "@" + std::to_string(idx), p, size, debug);
	}

	~Compare() {
		for (auto it = contents_.begin(); it != contents_.end(); ++it) {
			cudaFreeAll(it->second.p);
		}
	}
};

}