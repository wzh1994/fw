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
	MultiExplosionFirework(float* args, bool initAttr = true)
		: FwRenderBase(args) {
		nFrames_ = 49;
		nInterpolation_ = 15;
		scaleRate_ = 0.0025f;
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
			AddValue("颜色衰减率");
			AddValue("尺寸衰减率");
			AddValue("初始速度");
			AddVec3("初始位置");
			AddValue("横截面粒子数量");
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
		size_t n = static_cast<size_t>(args_[6 * nFrames_ + 6]);
		nDirs_ = normalFireworkDirections(dDirections_, n);
		nSubGroups_ = static_cast<size_t>(
			static_cast<float>(nDirs_) * args_[6 * nFrames_ + 8]);
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
			dSpeeds_, dStartPoses_, currFrame, nFrames_,
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
		getColorAndSizeMatrix(args_, args_ + 3 * nFrames_, nFrames_,
			args_[6 * nFrames_], args_[6 * nFrames_ + 1],
			dColorMatrix_, dSizeMatrix_);

		scale(dSizeMatrix_, scaleRate_, nFrames_ * nFrames_);

		// 获取加速度对位移影响的矩阵
		CUDACHECK(cudaMemcpy(dShiftX_, args_ + 4 * nFrames_,
			nFrames_ * sizeof(float), cudaMemcpyHostToDevice));
		CUDACHECK(cudaMemcpy(dShiftY_, args_ + 5 * nFrames_,
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
			args_[6 * nFrames_ + 7]), nParticleGroups_ - nDirs_);
		fill(dColorAndSizeStarts_, 0, nDirs_);
		fill(dColorAndSizeStarts_ + nDirs_, static_cast<size_t>(
			args_[6 * nFrames_ + 7]), nParticleGroups_ - nDirs_);
		fill(dLifeTime_, static_cast<size_t>(args_[6 * nFrames_ + 7]), nDirs_);
		fill(dLifeTime_ + nDirs_, nFrames_, nParticleGroups_ - nDirs_);

		// 获取粒子的位置
		fill(dSpeeds_, args_[6 * nFrames_ + 2] * scaleRate_, nParticleGroups_);
		fill(dStartPoses_, args_ + 6 * nFrames_ + 3, nDirs_, 3);
		scale(dStartPoses_, scaleRate_, 3 * nDirs_);
		getSubFireworkPositions(dStartPoses_ + 3 * nDirs_, dDirections_,
			nDirs_, nSubGroups_, args_[6 * nFrames_ + 2] * scaleRate_,
			static_cast<size_t>(args_[6 * nFrames_ + 7]),
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