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
			BeginGroup(1, 1);
				AddScalarGroup("内环尺寸");
			EndGroup();
			BeginGroup(1, 1);
				AddScalarGroup("内环色彩增强");
			EndGroup();
			AddValue("颜色衰减率");
			AddValue("尺寸衰减率");
			AddVec3("初始位置");
			AddValue("横截面粒子数量");
			AddValue("随机比率");
			AddValue("寿命");		
			AddValue("二次爆炸时间");		
			AddValue("二次爆炸比率");
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
		// 先获取所有的方向, 给dDirections_赋值
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

		// 获取加速度对位移影响的矩阵
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

		// 获取粒子的起始时间
		fill(dStartFrames_, 0, nDirs_);
		fill(dStartFrames_ + nDirs_, static_cast<size_t>(
			*pDualExplosionTime_), nParticleGroups_ - nDirs_);
		fill(dColorAndSizeStarts_, 0, nDirs_);
		fill(dColorAndSizeStarts_ + nDirs_, static_cast<size_t>(
			*pDualExplosionTime_), nParticleGroups_ - nDirs_);
		fill(dLifeTime_, static_cast<size_t>(*pDualExplosionTime_), nDirs_);
		fill(dLifeTime_ + nDirs_, nFrames_, nParticleGroups_ - nDirs_);

		// 获取粒子的位置
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
	~MultiExplosionFirework() override {
		releaseStaticResources();
		releaseDynamicResources();
	}
};

}