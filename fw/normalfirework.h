#pragma once
#include "firework.h"
#include "fireworkrenderbase.h"
#include <cuda_runtime.h>
#include <cuda_gl_interop.h>
#include "test.h"
#include "compare.h"

namespace firework{
class NormalFirework final : public FwRenderBase {
	friend FwBase* getFirework(FireWorkType type, float* args, bool initAttr);

private:
	float* pStartColors_;
	float* pStartSizes_;
	float* pXAcc_;
	float* pYAcc_;
	float* pSpeed_;
	float* pColorDecay_;
	float* pSizeDecay_;
	float* pStartPos_;
	float* pCrossSectionNum_;
private:
	NormalFirework(float* args, bool initAttr = true) : FwRenderBase(args) {
		nFrames_ = 49;
		nInterpolation_ = 15;
		scaleRate_ = 0.025f;
		if (initAttr) {
			pStartColors_ = args_;
			BeginGroup(1, 3);
				AddColorGroup("��ʼ��ɫ");
			EndGroup();

			pStartSizes_ = pStartColors_ + 3 * nFrames_;
			BeginGroup(1, 1);
				AddScalarGroup("��ʼ�ߴ�");
			EndGroup();
			
			pXAcc_ = pStartSizes_ + nFrames_;
			BeginGroup(1, 1);
				AddScalarGroup("X������ٶ�");
			EndGroup();
			
			pYAcc_ = pXAcc_ + nFrames_;
			BeginGroup(1, 1);
				AddScalarGroup("Y������ٶ�");
			EndGroup();
			
			pSpeed_ = pYAcc_ + nFrames_;
			BeginGroup(1, 1);
				AddScalarGroup("�����ٶ�");
			EndGroup();
			
			pColorDecay_ = pSpeed_ + nFrames_;
			AddValue("��ɫ˥����");
			
			pSizeDecay_ = pColorDecay_ + 1;
			AddValue("�ߴ�˥����");

			pStartPos_ = pSizeDecay_ + 1;
			AddVec3("��ʼλ��");

			pCrossSectionNum_ = pStartPos_ + 3;
			AddValue("�������������");
		}
	}
	
	size_t initDirections() {
		// �Ȼ�ȡ���еķ���, ��dDirections_��ֵ
		size_t n = static_cast<size_t>(*pCrossSectionNum_);
		nParticleGroups_ = normalFireworkDirections(dDirections_, n);
		return nParticleGroups_;
	}

	// ���������dColorMatrix_, dSizeMatrix_, dSpeeds_, dStartPoses_��
	// dStartFrames_�����ֵ
	void prepare() override {
		if (initDirections() > maxNParticleGroups_) {
			maxNParticleGroups_ = nParticleGroups_;
			releaseDynamicResources();
			allocDynamicResources();
			maxNParticleGroups_ = nParticleGroups_;
		}
		// ��ȡ��ɫ�ͳߴ�仯�������
		getColorAndSizeMatrix(pStartColors_, pStartSizes_, nFrames_,
			*pColorDecay_, *pSizeDecay_, dColorMatrix_, dSizeMatrix_);

		scale(dSizeMatrix_, scaleRate_ / 2, nFrames_ * nFrames_);

		// ��ȡ���ӵ��ٶȣ� ���ٶ�
		cudaMemset(dSpeed_, 0, (nFrames_ + 1) * sizeof(float));
		cudaMemcpy(dSpeed_ + 1, pSpeed_, nFrames_ * sizeof(float), cudaMemcpyHostToDevice);
		cuSum(dCentrifugalPos_, dSpeed_, nFrames_ + 1);
		scale(dCentrifugalPos_, scaleRate_ * kFrameTime, nFrames_ + 1);

		fill(dStartPoses_, pStartPos_, nParticleGroups_, 3);
		scale(dStartPoses_, scaleRate_, 3 * nParticleGroups_);

		fill(dStartFrames_, 0, nParticleGroups_);
		fill(dLifeTime_, nFrames_, nParticleGroups_);

		CUDACHECK(cudaMemcpy(dShiftX_, pXAcc_,
			nFrames_ * sizeof(float), cudaMemcpyHostToDevice));
		CUDACHECK(cudaMemcpy(dShiftY_, pYAcc_,
			nFrames_ * sizeof(float), cudaMemcpyHostToDevice));

		calcShiftingByOutsideForce(dShiftX_, nFrames_, nInterpolation_);
		calcShiftingByOutsideForce(dShiftY_, nFrames_, nInterpolation_);
		size_t shiftSize = nInterpolation_ * (nFrames_ - 1) + nFrames_;
		shiftSize *= shiftSize;

		scale(dShiftX_, scaleRate_, shiftSize);
		scale(dShiftY_, scaleRate_, shiftSize);
	}

	void getPoints(int currFrame) override {
		particleSystemToPoints(
			dPoints_, dColors_, dSizes_, dGroupStarts_, dStartFrames_,
			dLifeTime_, nParticleGroups_, dDirections_, dCentrifugalPos_,
			dStartPoses_, currFrame, nFrames_, dColorMatrix_, dSizeMatrix_);
	}

	void allocAppendixResource() override {};
	void releaseAppendixResource() override {};
	
public:
	// ��������һ��
	void initialize() override {
		// ���ø���ĳ�ʼ��
		FwRenderBase::initialize();
		allocStaticResources();

		maxNParticleGroups_ = initDirections();

		// �ڵ���initDirections֮��nParticleGroups_ ����ֵ
		allocDynamicResources();

		// �����Դ涼������֮��ſ��Ե���prepare
		prepare();
	}

public:
	~NormalFirework() override {
		releaseStaticResources();
		releaseDynamicResources();
	}
};

}