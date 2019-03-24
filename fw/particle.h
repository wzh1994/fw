#pragma once

// particle.h: 描述粒子系统的类
//

#include <memory>
#include "exceptions.h"

class FwBase;
class Particle {
	friend class FwBase;
private:
	using upt_t = std::unique_ptr<float>;
	int count_;
	int size_;
	upt_t pStartTimes_;
	upt_t pDirections_;
	upt_t pSpeeds_;

public:
	Particle(int n = 0) :count_(n), size_(0) {
		if (count_ > 0) {
			pStartTimes_.reset(new float[count_]);
			pDirections_.reset(new float[3 * count_]);
			pSpeeds_.reset(new float[count_]);
		}
	}
	void reserve(int n) {
		FW_ASSERT(n >= count_);
		if (n == count_) return;
		upt_t pStartTimes(new float[n]);
		upt_t pDirections(new float[3 * n]);
		upt_t pStartSpeeds(new float[n]);
		if (count_ > 0) {
			memcpy(pStartTimes.get(), pStartTimes_.get(), count_ * sizeof(float));
			memcpy(pDirections.get(), pDirections_.get(), 3 * count_ * sizeof(float));
			memcpy(pStartSpeeds.get(), pSpeeds_.get(), count_ * sizeof(float));
			pStartTimes_.reset(pStartTimes.release());
			pDirections_.reset(pDirections.release());
			pSpeeds_.reset(pStartSpeeds.release());
		}
		count_ = n;
	}

};

