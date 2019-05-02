#pragma once
#include "firework.h"
#include "fireworkrenderbase.h"
#include <cuda_runtime.h>
#include <cuda_gl_interop.h>
#include "test.h"
#include "compare.h"

namespace firework {

class MultiExplosionFirework final : public FwRenderBase {
	friend FwBase* getFirework(FireWorkType type, float* args, bool initAttr);

private:
	size_t nDirs_;
	size_t nSubGroups_;
	size_t* dColorAndSizeStarts_;

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
	float* pDualExplosionTime_;
	float* pDualExplosionRate_;

private:
	MultiExplosionFirework(float* args, bool initAttr = true)
		: FwRenderBase(args) {
		nFrames_ = 49;
		nInterpolation_ = 15;
		scaleRate_ = 0.0025f;
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

			pDualExplosionTime_ = pCrossSectionNum_ + 1;
			AddValue("���α�ըʱ��");

			pDualExplosionRate_ = pDualExplosionTime_ + 1;
			AddValue("���α�ը����");
		}
	}

private:
	void allocAppendixResource() override {
		CUDACHECK(cudaMallocAlign(
			&dColorAndSizeStarts_, nParticleGroups_ * sizeof(size_t)));
	};
	void releaseAppendixResource() override {
		CUDACHECK(cudaFree(dColorAndSizeStarts_));
	};

	size_t initDirections() {
		// �Ȼ�ȡ���еķ���, ��dDirections_��ֵ
		size_t n = static_cast<size_t>(*pCrossSectionNum_);
		nDirs_ = normalFireworkDirections(dDirections_, n);
		nSubGroups_ = static_cast<size_t>(
			static_cast<float>(nDirs_) * *pDualExplosionRate_);
		nParticleGroups_ = nDirs_ + nDirs_ * nSubGroups_;
		cout << nSubGroups_ << endl;
		FW_ASSERT(nParticleGroups_ < kMaxParticleGroup) << sstr(
			"We can only support ", kMaxParticleGroup,
			" particles, however ", nParticleGroups_, "is given");
		return nParticleGroups_;
	}

	void getPoints(int currFrame) override {
		particleSystemToPoints(dPoints_, dColors_, dSizes_, dGroupStarts_,
			dStartFrames_, dLifeTime_, nParticleGroups_, dDirections_,
			dCentrifugalPos_, dStartPoses_, currFrame, nFrames_,
			dColorAndSizeStarts_, dColorMatrix_, dSizeMatrix_);
	}

public:
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

		// ��ȡ���ٶȶ�λ��Ӱ��ľ���
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

		// ��ȡ���ӵ���ʼʱ��
		fill(dStartFrames_, 0, nDirs_);
		fill(dStartFrames_ + nDirs_, static_cast<size_t>(
			*pDualExplosionTime_), nParticleGroups_ - nDirs_);
		fill(dColorAndSizeStarts_, 0, nDirs_);
		fill(dColorAndSizeStarts_ + nDirs_, static_cast<size_t>(
			*pDualExplosionTime_), nParticleGroups_ - nDirs_);
		fill(dLifeTime_, static_cast<size_t>(*pDualExplosionTime_), nDirs_);
		fill(dLifeTime_ + nDirs_, nFrames_, nParticleGroups_ - nDirs_);

		// ��ȡ���ӵ�λ��
		cudaMemset(dSpeed_, 0, (nFrames_ + 1) * sizeof(float));
		cudaMemcpy(dSpeed_ + 1, pSpeed_, nFrames_ * sizeof(float), cudaMemcpyHostToDevice);
		cuSum(dCentrifugalPos_, dSpeed_, nFrames_ + 1);
		scale(dCentrifugalPos_, scaleRate_ * kFrameTime, nFrames_ + 1);

		fill(dStartPoses_, pStartPos_, nDirs_, 3);
		scale(dStartPoses_, scaleRate_, 3 * nDirs_);
		getSubFireworkPositions(dStartPoses_ + 3 * nDirs_, dDirections_,
			nDirs_, nSubGroups_, dCentrifugalPos_, *pDualExplosionTime_,
			nFrames_ * (nInterpolation_ + 1), dShiftX_, dShiftY_);
		//show(dStartPoses_, 3 * nParticleGroups_, 3 * nDirs_);
	}

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
	~MultiExplosionFirework() override {
		releaseStaticResources();
		releaseDynamicResources();
	}
};

}