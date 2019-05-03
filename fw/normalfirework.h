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
	float* pInnerSize_;
	float* pMaxLifeTime_;
private:
	NormalFirework(float* args, bool initAttr = true, size_t lifeTime = 50)
		: FwRenderBase(args) {
		nFrames_ = 49;
		nInterpolation_ = 15;
		scaleRate_ = 0.025f;
		pStartColors_ = args_;
		pStartSizes_ = pStartColors_ + 3 * nFrames_;
		pXAcc_ = pStartSizes_ + nFrames_;
		pYAcc_ = pXAcc_ + nFrames_;
		pSpeed_ = pYAcc_ + nFrames_;
		pColorDecay_ = pSpeed_ + nFrames_;
		pSizeDecay_ = pColorDecay_ + 1;
		pStartPos_ = pSizeDecay_ + 1;
		pCrossSectionNum_ = pStartPos_ + 3;
		pInnerSize_ = pCrossSectionNum_ + 1;
		pMaxLifeTime_ = pInnerSize_ + 1;
		
		if (initAttr) {
			BeginGroup(1, 3);
				AddColorGroup("初始颜色");
			EndGroup();
			BeginGroup(1, 1);
				AddScalarGroup("初始尺寸");
			EndGroup();
			BeginGroup(1, 1);
				AddScalarGroup("X方向加速度");
			EndGroup();
			BeginGroup(1, 1);
				AddScalarGroup("Y方向加速度");
			EndGroup();
			BeginGroup(1, 1);
				AddScalarGroup("离心速度");
			EndGroup();
			AddValue("颜色衰减率");
			AddValue("尺寸衰减率");
			AddVec3("初始位置");
			AddValue("横截面粒子数量");
			AddValue("内环尺寸")
			AddValue("寿命");
		}
	}
	
	size_t initDirections() {
		// 先获取所有的方向, 给dDirections_赋值
		size_t n = static_cast<size_t>(*pCrossSectionNum_);
		nParticleGroups_ = normalFireworkDirections(dDirections_, n);
		return nParticleGroups_;
	}

	// 本方法会给dColorMatrix_, dSizeMatrix_, dSpeeds_, dStartPoses_和
	// dStartFrames_赋予初值
	void prepare() override {
		if (initDirections() > maxNParticleGroups_) {
			maxNParticleGroups_ = nParticleGroups_;
			releaseDynamicResources();
			allocDynamicResources();
			maxNParticleGroups_ = nParticleGroups_;
		}
		// 获取颜色和尺寸变化情况矩阵
		getColorAndSizeMatrix(pStartColors_, pStartSizes_, nFrames_,
			*pColorDecay_, *pSizeDecay_, dColorMatrix_, dSizeMatrix_);

		scale(dSizeMatrix_, scaleRate_ / 2, nFrames_ * nFrames_);

		// 获取粒子的速度， 加速度
		cudaMemset(dSpeed_, 0, (nFrames_ + 1) * sizeof(float));
		cudaMemcpy(dSpeed_ + 1, pSpeed_, nFrames_ * sizeof(float), cudaMemcpyHostToDevice);
		cuSum(dCentrifugalPos_, dSpeed_, nFrames_ + 1);
		scale(dCentrifugalPos_, scaleRate_ * kFrameTime, nFrames_ + 1);

		fill(dStartPoses_, pStartPos_, nParticleGroups_, 3);
		scale(dStartPoses_, scaleRate_, 3 * nParticleGroups_);

		fill(dStartFrames_, 0, nParticleGroups_);
		fill(dLifeTime_, *pMaxLifeTime_, nParticleGroups_);

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
		
		innerSize_ = *pInnerSize_;
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
	// 仅被调用一次
	void initialize() override {
		// 调用父类的初始化
		FwRenderBase::initialize();
		allocStaticResources();

		maxNParticleGroups_ = initDirections();

		// 在调用initDirections之后nParticleGroups_ 才有值
		allocDynamicResources();

		// 所有显存都被分配之后才可以调用prepare
		prepare();
	}

public:
	~NormalFirework() override {
		releaseStaticResources();
		releaseDynamicResources();
	}
};

}