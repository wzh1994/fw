#pragma once
#include "mixturefireworkbase.h"
#include <memory>
#include <vector>

namespace firework {

class NormalMixtureFirework final : public MixtureFireworkBase {
	friend FwBase* getFirework(FireWorkType, float*, bool, size_t);

private:
	NormalMixtureFirework(float* args, size_t nSubfw, bool initAttr = true)
		: MixtureFireworkBase(args, nSubfw) {
		fws.emplace_back(getFirework(FireWorkType::Normal,
			args, false));
		nFrames_ = 49;
		size_t nArgsPerSubFw = nFrames_ * 9 + 8;
		for (size_t i = 1; i < nSubFw_; ++i) {
			fws.emplace_back(getFirework(FireWorkType::Normal,
				args + nArgsPerSubFw * i, false, 20000000));
		}
		if (initAttr) {
			for (size_t i = 0; i < nSubFw_; ++i) {
				BeginGroup(1, 3);
					AddColorGroup("初始颜色" + std::to_wstring(i + 1));
				EndGroup();
				BeginGroup(1, 1);
					AddScalarGroup("初始尺寸" + std::to_wstring(i + 1));
				EndGroup();
				BeginGroup(1, 1);
					AddScalarGroup("X方向加速度" + std::to_wstring(i + 1));
				EndGroup();
				BeginGroup(1, 1);
					AddScalarGroup("Y方向加速度" + std::to_wstring(i + 1));
				EndGroup();
				BeginGroup(1, 1);
					AddScalarGroup("离心速度" + std::to_wstring(i + 1));
				EndGroup();
				BeginGroup(1, 1);
					AddScalarGroup("内环尺寸" + std::to_wstring(i + 1));
				EndGroup();
				BeginGroup(1, 1);
					AddScalarGroup("内环色彩增强" + std::to_wstring(i + 1));
				EndGroup();
				AddValue("颜色衰减率" + std::to_wstring(i + 1));
				AddValue("尺寸衰减率" + std::to_wstring(i + 1));
				AddVec3("初始位置" + std::to_wstring(i + 1));
				AddValue("横截面粒子数量" + std::to_wstring(i + 1));
				AddValue("随机比率");
				AddValue("寿命" + std::to_wstring(i + 1));
			}
		}
	}

public:
	~NormalMixtureFirework() override {}
};

}