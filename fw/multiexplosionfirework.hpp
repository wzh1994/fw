#pragma once
#include "firework.h"
#include "fireworkrenderbase.h"
#include <cuda_runtime.h>
#include <cuda_gl_interop.h>
#include "test.h"
#include "compare.h"

namespace firework {

class MultiExplosionFirework final : public FwRenderBase {
	friend FwBase* getFirework(FireWorkType, float*, bool, size_t);

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
	float* pInnerSize_;
	float* pInnerColor_;
	float* pColorDecay_;
	float* pSizeDecay_;
	float* pStartPos_;
	float* pCrossSectionNum_;
	float* pRandomRate_;
	float* pMaxLifeTime_;
	float* pDualExplosionTime_;
	float* pDualExplosionRate_;

private:
	MultiExplosionFirework(float* args, bool initAttr = true,
			size_t bufferSize = 200000000)
		: FwRenderBase(args) {
		nEboToInit_ = bufferSize;
		nVboToInit_ = bufferSize;
		nFrames_ = 49;
		nInterpolation_ = 15;
		scaleRate_ = 0.025f;
		pStartColors_ = args_;
		pStartSizes_ = pStartColors_ + 3 * nFrames_;
		pXAcc_ = pStartSizes_ + nFrames_;
		pYAcc_ = pXAcc_ + nFrames_;
		pSpeed_ = pYAcc_ + nFrames_;
		pInnerSize_ = pSpeed_ + nFrames_;
		pInnerColor_ = pInnerSize_ + nFrames_;
		pColorDecay_ = pInnerColor_ + nFrames_;
		pSizeDecay_ = pColorDecay_ + 1;
		pStartPos_ = pSizeDecay_ + 1;
		pCrossSectionNum_ = pStartPos_ + 3;
		pRandomRate_ = pCrossSectionNum_ + 1;
		pMaxLifeTime_ = pRandomRate_ + 1;
		pDualExplosionTime_ = pMaxLifeTime_ + 1;
		pDualExplosionRate_ = pDualExplosionTime_ + 1;
		if (initAttr) {
			BeginGroup(1, 3);
				AddColorGroup("��ʼ��ɫ");
			EndGroup();
			BeginGroup(1, 1);
				AddScalarGroup("��ʼ�ߴ�");
			EndGroup();
			BeginGroup(1, 1);
				AddScalarGroup("X������ٶ�");
			EndGroup();
			BeginGroup(1, 1);
				AddScalarGroup("Y������ٶ�");
			EndGroup();
			BeginGroup(1, 1);
				AddScalarGroup("�����ٶ�");
			EndGroup();
			BeginGroup(1, 1);
				AddScalarGroup("�ڻ��ߴ�");
			EndGroup();
			BeginGroup(1, 1);
				AddScalarGroup("�ڻ�ɫ����ǿ");
			EndGroup();
			AddValue("��ɫ˥����");
			AddValue("�ߴ�˥����");
			AddVec3("��ʼλ��");
			AddValue("�������������");
			AddValue("�������");
			AddValue("����");		
			AddValue("���α�ըʱ��");		
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
		printf("%llu\n", n);
		nDirs_ = normalFireworkDirections(dDirections_, n,
			*pRandomRate_, *pRandomRate_, *pRandomRate_);
		printf("%llu %f\n", nDirs_, *pDualExplosionRate_);
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
		innerSize_ = pInnerSize_[currFrame];
		innerColor_ = pInnerColor_[currFrame];
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

		//show(dCentrifugalPos_, nFrames_ + 1);

		fill(dStartPoses_, pStartPos_, nDirs_, 3);
		scale(dStartPoses_, scaleRate_, 3 * nDirs_);
		getSubFireworkPositions(dStartPoses_ + 3 * nDirs_, dDirections_,
			nDirs_, nSubGroups_, dCentrifugalPos_, *pDualExplosionTime_,
			nFrames_ * (nInterpolation_ + 1) - nInterpolation_, dShiftX_, dShiftY_);
		//show(dStartPoses_, nParticleGroups_ * 3, nDirs_ * 3);
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