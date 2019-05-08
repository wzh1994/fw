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
					AddColorGroup("��ʼ��ɫ" + std::to_wstring(i + 1));
				EndGroup();
				BeginGroup(1, 1);
					AddScalarGroup("��ʼ�ߴ�" + std::to_wstring(i + 1));
				EndGroup();
				BeginGroup(1, 1);
					AddScalarGroup("X������ٶ�" + std::to_wstring(i + 1));
				EndGroup();
				BeginGroup(1, 1);
					AddScalarGroup("Y������ٶ�" + std::to_wstring(i + 1));
				EndGroup();
				BeginGroup(1, 1);
					AddScalarGroup("�����ٶ�" + std::to_wstring(i + 1));
				EndGroup();
				BeginGroup(1, 1);
					AddScalarGroup("�ڻ��ߴ�" + std::to_wstring(i + 1));
				EndGroup();
				BeginGroup(1, 1);
					AddScalarGroup("�ڻ�ɫ����ǿ" + std::to_wstring(i + 1));
				EndGroup();
				AddValue("��ɫ˥����" + std::to_wstring(i + 1));
				AddValue("�ߴ�˥����" + std::to_wstring(i + 1));
				AddVec3("��ʼλ��" + std::to_wstring(i + 1));
				AddValue("�������������" + std::to_wstring(i + 1));
				AddValue("�������");
				AddValue("����" + std::to_wstring(i + 1));
			}
		}
	}

public:
	~NormalMixtureFirework() override {}
};

}