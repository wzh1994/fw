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
				AddColorGroup("初始颜色");
			EndGroup();

			pStartSizes_ = pStartColors_ + 3 * nFrames_;
			BeginGroup(1, 1);
				AddScalarGroup("初始尺寸");
			EndGroup();

			pXAcc_ = pStartSizes_ + nFrames_;
			BeginGroup(1, 1);
				AddScalarGroup("X方向加速度");
			EndGroup();

			pYAcc_ = pXAcc_ + nFrames_;
			BeginGroup(1, 1);
				AddScalarGroup("Y方向加速度");
			EndGroup();

			pSpeed_ = pYAcc_ + nFrames_;
			BeginGroup(1, 1);
				AddScalarGroup("离心速度");
			EndGroup();

			pColorDecay_ = pSpeed_ + nFrames_;
			AddValue("颜色衰减率");

			pSizeDecay_ = pColorDecay_ + 1;
			AddValue("尺寸衰减率");

			pStartPos_ = pSizeDecay_ + 1;
			AddVec3("初始位置");

			pCrossSectionNum_ = pStartPos_ + 3;
			AddValue("横截面粒子数量");

			pDualExplosionTime_ = pCrossSectionNum_ + 1;
			AddValue("二次爆炸时间");

			pDualExplosionRate_ = pDualExplosionTime_ + 1;
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

		fill(dStartPoses_, pStartPos_, nDirs_, 3);
		scale(dStartPoses_, scaleRate_, 3 * nDirs_);
		getSubFireworkPositions(dStartPoses_ + 3 * nDirs_, dDirections_,
			nDirs_, nSubGroups_, dCentrifugalPos_, *pDualExplosionTime_,
			nFrames_ * (nInterpolation_ + 1), dShiftX_, dShiftY_);
		//show(dStartPoses_, 3 * nParticleGroups_, 3 * nDirs_);
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