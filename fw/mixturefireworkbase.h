#pragma once

#include "firework.h"
#include <memory>
#include <vector>

namespace firework {
class MixtureFireworkBase : public FwBase {
private:
	bool* showFlags_;

protected:
	std::vector<std::unique_ptr<FwBase>> fws;
	size_t nSubFw_;

	MixtureFireworkBase(float* args, size_t nSubFw)
		: nSubFw_(nSubFw), FwBase(args) {
		showFlags_ = new bool[nSubFw];
		for (size_t i = 0; i < nSubFw; ++i) {
			showFlags_[i] = true;
		}
	}

public:
	// 仅被调用一次
	void initialize() final {
		for (size_t i = 0; i < nSubFw_; ++i) {
			fws[i]->initialize();
		}
	}

	bool isMixture() const noexcept final {
		return true;
	}

	size_t nSubFw() const noexcept final {
		return nSubFw_;
	}

	bool* showFlags() final {
		return showFlags_;
	}

	const bool* showFlags() const final {
		return showFlags_;
	}

	void GetParticles(int currFrame) final {
		for (size_t i = 0; i < nSubFw_; ++i) {
			if (showFlags_[i]) {
				fws[i]->GetParticles(currFrame);
			}
		}
	}

	void RenderScene(const Camera& camera) final {
		for (size_t i = 0; i < nSubFw_; ++i) {
			if (showFlags_[i]) {
				fws[i]->RenderScene(camera);
			}
		}
	}

	void prepare() final {
		for (size_t i = 0; i < nSubFw_; ++i) {
			fws[i]->prepare();
		}
	}

	~MixtureFireworkBase() {
		delete[] showFlags_;
	}
};
}