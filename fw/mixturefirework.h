#pragma once
#include "mixturefireworkbase.h"
#include <memory>
#include <vector>

namespace firework {

class NormalMixtureFirework final : public MixtureFireworkBase {
	friend FwBase* getFirework(FireWorkType, float*, bool, size_t);

private:
	NormalMixtureFirework(float* args, size_t nSubfw)
		: MixtureFireworkBase(args, nSubfw) {
		nFrames_ = kDefaultFrames;
		fws.emplace_back(getFirework(FireWorkType::Normal,
			args, false));
		size_t nArgsPerSubFw = kDefaultNormalArgs;
		for (size_t i = 1; i < nSubFw_; ++i) {
			fws.emplace_back(getFirework(FireWorkType::Normal,
				args + nArgsPerSubFw * i, false, 20000000));
		}
		for (size_t i = 0; i < nSubFw_; ++i) {
			NORMAL_RULE_GROUP(std::to_wstring(i + 1));
			NORMAL_RULE_VALUE(std::to_wstring(i + 1));
		}
	}

public:
	~NormalMixtureFirework() override {}
};

class MultiKindMixtureFirework final : public MixtureFireworkBase {
	friend FwBase* getFirework(FireWorkType, float*, bool, size_t);
	MultiKindMixtureFirework(float* args, std::vector<FireWorkType> types,
			size_t bufferSize = 80000000)
		: MixtureFireworkBase(args, types.size()) {
		nFrames_ = kDefaultFrames;
		size_t offset = 0;
		size_t index = 1;
		for (auto it = types.begin(); it != types.end(); ++it, ++index) {
			fws.emplace_back(getFirework(*it, args + offset, false, bufferSize));
			switch (*it) {
			case FireWorkType::Normal:
				NORMAL_RULE_GROUP(std::to_wstring(index));
				NORMAL_RULE_VALUE(std::to_wstring(index));
				offset += kDefaultNormalArgs;
				break;
			case FireWorkType::Circle:
				CIRCLE_RULE_GROUP(std::to_wstring(index));
				CIRCLE_RULE_VALUE(std::to_wstring(index));
				offset += kDefaultCircleArgs;
				break;
			case FireWorkType::Strafe:
				STRAFE_RULE_GROUP(std::to_wstring(index));
				STRAFE_RULE_VALUE(std::to_wstring(index));
				offset += kDefaultStrafeArgs;
				break;
			case FireWorkType::Twinkle:
				TWINKLE_RULE_GROUP(std::to_wstring(index));
				TWINKLE_RULE_VALUE(std::to_wstring(index));
				offset += kDefaultTwinkleArgs;
				break;
			case FireWorkType::MultiExplosion:
				MULTI_EXPLOSION_RULE_GROUP(std::to_wstring(index));
				MULTI_EXPLOSION_RULE_VALUE(std::to_wstring(index));
				offset += kDefaultMultiExplosionArgs;
				break;
			default:
				FW_NOTSUPPORTED << "Invalid firework type";
			}
		}
	}

public:
	~MultiKindMixtureFirework() override {}

};

}