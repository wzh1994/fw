
// firework.cpp: 实现文件 实现firework.h中的函数
//


#include "stdafx.h"
#include "firework.h"
#include "normalfirework.h"
#include "multiexplosionfirework.hpp"
#include "mixturefirework.h"
#include "strafefirework.h"
#include "circlefirework.h"
#include "twinklefirework.h"

namespace firework {
	size_t FwBase::Attr::idx = 0;
	size_t FwBase::Attr::groupOffset = 0;
	size_t FwBase::Attr::groupNums = 0;
	size_t FwBase::Attr::groupStep = 0;
	size_t FwBase::Attr::nFrames = 0;


FwBase* getFirework(FireWorkType type, float* args, bool initAttr, size_t bufferSize) {
	switch (type) {
	case FireWorkType::Normal:
		return new NormalFirework(args, initAttr, bufferSize);
		break;
	case FireWorkType::DualMixture:
		FW_ASSERT(initAttr);
		return new NormalMixtureFirework(args, 2);
		break;
	case FireWorkType::TriplicateMixture:
		FW_ASSERT(initAttr);
		return new NormalMixtureFirework(args, 3);
		break;
	case FireWorkType::MultiExplosion:
		return new MultiExplosionFirework(args, initAttr);
		break;
	case FireWorkType::Strafe:
		return new StrafeFirework(args, initAttr);
		break;
	case FireWorkType::Circle:
		return new CircleFirework(args, initAttr);
		break;
	case FireWorkType::Twinkle:
		return new TwinkleFirework(args, initAttr);
		break;
	case FireWorkType::NormalCircleAndTwinkle:
		return new MultiKindMixtureFirework(args, std::vector<FireWorkType>{
			FireWorkType::Normal,
			FireWorkType::Circle,
			FireWorkType::Twinkle});
	case FireWorkType::FiveNormal:
		return new MultiKindMixtureFirework(args, std::vector<FireWorkType>(
			5, FireWorkType::Normal));
	case FireWorkType::SixCircle:
		return new MultiKindMixtureFirework(args, std::vector<FireWorkType>(
			6, FireWorkType::Circle));
	case FireWorkType::ThreeNormalAndTwinkle:
		return new MultiKindMixtureFirework(args, std::vector<FireWorkType>{
			FireWorkType::Normal,
			FireWorkType::Normal,
			FireWorkType::Normal,
			FireWorkType::Twinkle});
	case FireWorkType::ThreeNormalCircleTwinkle:
		return new MultiKindMixtureFirework(args, std::vector<FireWorkType>{
			FireWorkType::Normal,
			FireWorkType::Normal,
			FireWorkType::Normal,
			FireWorkType::Circle,
			FireWorkType::Twinkle});
		break;
	default:
		FW_NOTSUPPORTED << "Invalid firework type!";
		return nullptr;
	}
}

}