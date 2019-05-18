#pragma once
#include "fireworkrenderbase.h"


namespace firework {
	class TwinkleFirework final : public FwRenderBase {
		friend FwBase* getFirework(FireWorkType, float*, bool, size_t);

	private:
		float* pStartColors_;
		float* pStartSizes_;
		float* pXAcc_;
		float* pYAcc_;
		float* pSpeed_;
		float* pInnerSize_;
		float* pInnerColor_;
		float* pVisivbleRate_;
		float* pColorDecay_;
		float* pSizeDecay_;
		float* pStartPos_;
		float* pCrossSectionNum_;
		float* pRandomRate_;
		float* pMaxLifeTime_;

	private:
		TwinkleFirework(float* args, bool initAttr = true,
			size_t bufferSize = 200000000)
			: FwRenderBase(args) {
			nEboToInit_ = bufferSize;
			nVboToInit_ = bufferSize;
			nFrames_ = kDefaultFrames;
			nInterpolation_ = kDefaultInterpolation;
			scaleRate_ = kDefaultScaleRate;
			pStartColors_ = args_;
			pStartSizes_ = pStartColors_ + 3 * nFrames_;
			pXAcc_ = pStartSizes_ + nFrames_;
			pYAcc_ = pXAcc_ + nFrames_;
			pSpeed_ = pYAcc_ + nFrames_;
			pInnerSize_ = pSpeed_ + nFrames_;
			pInnerColor_ = pInnerSize_ + nFrames_;
			pVisivbleRate_ = pInnerColor_ + nFrames_;
			pColorDecay_ = pVisivbleRate_ + nFrames_;
			pSizeDecay_ = pColorDecay_ + 1;
			pStartPos_ = pSizeDecay_ + 1;
			pCrossSectionNum_ = pStartPos_ + 3;
			pRandomRate_ = pCrossSectionNum_ + 1;
			pMaxLifeTime_ = pRandomRate_ + 1;

			if (initAttr) {
				TWINKLE_RULE_GROUP(std::wstring(L""));
				TWINKLE_RULE_VALUE(std::wstring(L""));
			}
		}

		size_t initDirections() {
			// �Ȼ�ȡ���еķ���, ��dDirections_��ֵ
			size_t n = static_cast<size_t>(*pCrossSectionNum_);
			nParticleGroups_ = normalFireworkDirections(dDirections_, n,
				*pRandomRate_, *pRandomRate_, *pRandomRate_);
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
				initRand(devStates_, nParticleGroups_);
			}
			// ��ȡ��ɫ�ͳߴ�仯�������
			getColorAndSizeMatrix(pStartColors_, pStartSizes_, nFrames_,
				*pColorDecay_, *pSizeDecay_, dColorMatrix_, dSizeMatrix_);

			scale(dSizeMatrix_, scaleRate_ / 2, nFrames_ * nFrames_);

			// ��ȡ���ӵ��ٶȣ� ���ٶ�
			cudaMemset(dSpeed_, 0, (nFrames_ + 1) * sizeof(float));
			cudaMemcpy(dSpeed_ + 1,
				pSpeed_, nFrames_ * sizeof(float), cudaMemcpyHostToDevice);
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
		}

		void getPoints(int currFrame) override {
			innerSize_ = pInnerSize_[currFrame];
			innerColor_ = pInnerColor_[currFrame];
			particleSystemToPoints(dPoints_, dColors_, dSizes_,
				dGroupStarts_, dStartFrames_, dLifeTime_, nParticleGroups_,
				dDirections_, dCentrifugalPos_, dStartPoses_,
				currFrame, nFrames_, dColorMatrix_, dSizeMatrix_);
			visibleRate_ = pVisivbleRate_[currFrame];
		}

		void allocAppendixResource() override {
			CUDACHECK(cudaMallocAlign(
				&devStates_, sizeof(curandState) * nParticleGroups_));
		};
		void releaseAppendixResource() override {
			cudaFreeAll(devStates_);
		};

	public:
		// ��������һ��
		void initialize() override {
			// ���ø���ĳ�ʼ��
			FwRenderBase::initialize();
			allocStaticResources();

			maxNParticleGroups_ = initDirections();

			// �ڵ���initDirections֮��nParticleGroups_ ����ֵ
			allocDynamicResources();
			initRand(devStates_, nParticleGroups_);
			// �����Դ涼������֮��ſ��Ե���prepare
			prepare();
		}

	public:
		~TwinkleFirework() override {
			releaseStaticResources();
			releaseDynamicResources();
		}
	};

}