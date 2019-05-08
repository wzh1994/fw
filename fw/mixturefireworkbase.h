#pragma once

#include "firework.h"
#include <memory>
#include <vector>

namespace firework {
class MixtureFireworkBase : public FwBase {
protected:
	std::vector<std::unique_ptr<FwBase>> fws;
	size_t nSubFw_;

	MixtureFireworkBase(float* args, size_t nSubFw) : nSubFw_(nSubFw), FwBase(args) {}

public:
	// 仅被调用一次
	void initialize() final {
		for (size_t i = 0; i < nSubFw_; ++i) {
			fws[i]->initialize();
		}
	}

	void GetParticles(int currFrame) final {
		for (size_t i = 0; i < nSubFw_; ++i) {
			fws[i]->GetParticles(currFrame);
		}
	}

	void RenderScene(const Camera& camera) final {
		for (size_t i = 0; i < nSubFw_; ++i) {
			fws[i]->RenderScene(camera);
		}
	}

	void prepare() final {
		for (size_t i = 0; i < nSubFw_; ++i) {
			fws[i]->prepare();
		}
	}
};
}