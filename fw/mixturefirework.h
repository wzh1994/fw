#pragma once
#include "firework.h"
#include <cuda_runtime.h>
#include <cuda_gl_interop.h>
#include <memory>
#include <vector>
#include "test.h"
#include "compare.h"

namespace firework {

	class MixtureFirework final : public FwBase {
		friend FwBase* getFirework(FireWorkType type, float* args, bool initAttr);

	private:
		std::vector<std::unique_ptr<FwBase>> fws;
		size_t nSubFw_;
	private:
		MixtureFirework(float* args, int nSubFw, bool initAttr = true)
			: nSubFw_(nSubFw)
			, FwBase(args) {
			nFrames_ = 49;
			size_t nArgsPerSubFw = nFrames_ * 7 + 8;
			for (size_t i = 0; i < nSubFw_; ++i) {
				fws.emplace_back(getFirework(FireWorkType::Normal,
					args + nArgsPerSubFw * i, false));
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
					AddValue("��ɫ˥����" + std::to_wstring(i + 1));
					AddValue("�ߴ�˥����" + std::to_wstring(i + 1));
					AddVec3("��ʼλ��" + std::to_wstring(i + 1));
					AddValue("�������������" + std::to_wstring(i + 1));
					AddValue("��Ȧ����" + std::to_wstring(i + 1));
					AddValue("����" + std::to_wstring(i + 1));
				}
			}
		}

	public:
		// ��������һ��
		void initialize() override {
			for (size_t i = 0; i < nSubFw_; ++i) {
				fws[i]->initialize();
			}
		}

		void GetParticles(int currFrame) {
			for (size_t i = 0; i < nSubFw_; ++i) {
				fws[i]->GetParticles(currFrame);
			}
		}

		void RenderScene(const Camera& camera) {
			for (size_t i = 0; i < nSubFw_; ++i) {
				fws[i]->RenderScene(camera);
			}
		}

		void prepare() {
			for (size_t i = 0; i < nSubFw_; ++i) {
				fws[i]->prepare();
			}
		}

	public:
		~MixtureFirework() override {}
	};

}