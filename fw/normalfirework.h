#pragma once
#include "firework.h"
#include <cuda_runtime.h>
#include <cuda_gl_interop.h>
#include "test.h"
#include "compare.h"

namespace firework{
class NormalFirework final : public FwBase {
	friend FwBase* getFirework(FireWorkType type, float* args);

private:
	NormalFirework(float* args) : FwBase(args) {
		nFrames_ = 49;
		nInterpolation_ = 15;
		scaleRate_ = 0.0025f;
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
		AddValue("��ɫ˥����");
		AddValue("�ߴ�˥����");
		AddValue("��ʼ�ٶ�");
		AddVec3("��ʼλ��");
		AddValue("�������������");
	}
	
	size_t initDirections() {
		// �Ȼ�ȡ���еķ���, ��dDirections_��ֵ
		size_t n = static_cast<size_t>(args_[6 * nFrames_ + 6]);
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
		getColorAndSizeMatrix(args_, args_ + 3 * nFrames_, nFrames_,
			args_[6 * nFrames_], args_[6 * nFrames_ + 1],
			dColorMatrix_, dSizeMatrix_);

		scale(dSizeMatrix_, scaleRate_, nFrames_ * nFrames_);

		// ��ȡ���ӵ��ٶȣ� ���ٶ�
		fill(dSpeeds_, args_[6 * nFrames_ + 2] * scaleRate_, nParticleGroups_);
		fill(dStartPoses_, args_ + 6 * nFrames_ + 3, nParticleGroups_, 3);
		scale(dStartPoses_, scaleRate_, 3 * nParticleGroups_);

		fill(dStartFrames_, 0, nParticleGroups_);

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
	}

public:
	// ��������һ��
	void initialize() override {
		// ���ø���ĳ�ʼ��
		FwBase::initialize();
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