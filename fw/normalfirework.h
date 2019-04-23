#pragma once
#include "firework.h"
#include "kernels.h"
#include <cuda_runtime.h>
#include <cuda_gl_interop.h>
#include "test.h"

class NormalFirework : public FwBase {
	friend FwBase* getFirework(FireWorkType type, float* args);

private:
	float *dColorMatrix_, *dSizeMatrix_;
	float *dPoints_, *dColors_, *dSizes_;
	size_t nInterpolation_ = 15;
	
	// ����ϵͳ���
	size_t nParticleGroups_;
	float *dDirections_, *dSpeeds_, *dStartPoses_;
	size_t *dStartFrames_, *dGroupStarts_, *dGroupOffsets_;
	size_t realNGroups_;

	// �������
	float *dShiftX_, *dShiftY_;

private:
	NormalFirework(float* args) : FwBase(args) {
		nFrames_ = 5;
		nInterpolation_ = 15;
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
	}

	// ��������һ��
	void initialize() override {
		// ���ø���ĳ�ʼ��
		FwBase::initialize();
		genBuffer(5000000, 5000000);
		CUDACHECK(cudaMalloc(&dColorMatrix_, 3 * nFrames_ * nFrames_ * sizeof(float)));
		CUDACHECK(cudaMalloc(&dSizeMatrix_, nFrames_ * nFrames_ * sizeof(float)));

		// �ڵ���initDirections֮��nParticleGroups_ ����ֵ
		initDirections();

		// Ϊ���ٶȣ���ʼλ�õȷ���ռ�
		CUDACHECK(cudaMalloc(&dSpeeds_, nParticleGroups_ * sizeof(float)));
		CUDACHECK(cudaMalloc(&dStartPoses_, 3 * nParticleGroups_ * sizeof(float)));
		CUDACHECK(cudaMalloc(&dStartFrames_, nParticleGroups_ * sizeof(size_t)));

		size_t maxSize = (nInterpolation_ + 1) * nParticleGroups_ * nFrames_;
		CUDACHECK(cudaMalloc(&dPoints_, 3 * maxSize * sizeof(float)));
		CUDACHECK(cudaMalloc(&dColors_, 3 * maxSize * sizeof(float)));
		CUDACHECK(cudaMalloc(&dSizes_, maxSize * sizeof(float)));
		CUDACHECK(cudaMalloc(&dGroupStarts_, nParticleGroups_ * sizeof(size_t)));
		CUDACHECK(cudaMalloc(&dGroupOffsets_, nParticleGroups_ * sizeof(size_t)));

		size_t shiftSize = (nParticleGroups_ + 1) * nFrames_;
		shiftSize *= shiftSize;

		CUDACHECK(cudaMalloc(&dShiftX_, shiftSize * sizeof(float)));
		CUDACHECK(cudaMalloc(&dShiftY_, shiftSize * sizeof(float)));
		
		// �����Դ涼������֮��ſ��Ե���prepare
		prepare();
	}
	
	void initDirections() {
		// �Ȼ�ȡ���еķ���, ��dDirections_��ֵ
		nParticleGroups_ = 5;
		float* directions = new float[3 * nParticleGroups_] {
			1, 1, 1,
			1, 0, 1,
			0, 0, 1,
			1, 0, -1,
			0, 1, 1
		};
		CUDACHECK(cudaMalloc(&dDirections_, 3 * nParticleGroups_ * sizeof(float)));
		CUDACHECK(cudaMemcpy(dDirections_, directions,
			3 * nParticleGroups_ * sizeof(float), cudaMemcpyHostToDevice));
		normalize(dDirections_, nParticleGroups_);
		delete directions;
	}

	void releaseResources() {
		CUDACHECK(cudaFree(dColorMatrix_));
		CUDACHECK(cudaFree(dSizeMatrix_));

		CUDACHECK(cudaFree(dDirections_));
		CUDACHECK(cudaFree(dSpeeds_));
		CUDACHECK(cudaFree(dStartPoses_));
		CUDACHECK(cudaFree(dStartFrames_));

		CUDACHECK(cudaFree(dPoints_));
		CUDACHECK(cudaFree(dColors_));
		CUDACHECK(cudaFree(dSizes_));
		CUDACHECK(cudaFree(dGroupStarts_));
		CUDACHECK(cudaFree(dGroupOffsets_));
	}

public:
	// ÿ�β����ı�֮����Ҫ���ñ�����
	// ���������dColorMatrix_, dSizeMatrix_, dSpeeds_, dStartPoses_��
	// dStartFrames_�����ֵ
	void prepare() final {
		// ��ȡ��ɫ�ͳߴ�仯�������
		getColorAndSizeMatrix(args_, args_ + 3 * nFrames_, nFrames_,
			args_[6 * nFrames_], args_[6 * nFrames_ + 1],
			dColorMatrix_, dSizeMatrix_);

		// ��ȡ���ӵ��ٶȣ� ���ٶ�
		fill(dSpeeds_, args_[6 * nFrames_ + 2], nParticleGroups_);
		fill(dStartPoses_, args_ + 6 * nFrames_ + 3, nParticleGroups_, 3);
		fill(dStartFrames_, 0, nParticleGroups_);

		CUDACHECK(cudaMemcpy(dShiftX_, args_ + 4 * nFrames_,
			nFrames_ * sizeof(float), cudaMemcpyHostToDevice));
		CUDACHECK(cudaMemcpy(dShiftY_, args_ + 5 * nFrames_,
			nFrames_ * sizeof(float), cudaMemcpyHostToDevice));

		calcShiftingByOutsideForce(dShiftX_, nFrames_, nInterpolation_);
		calcShiftingByOutsideForce(dShiftY_, nFrames_, nInterpolation_);
	}

	void GetParticles(int currFrame) override {
		// �˴���dPoints_, dColors_, dSizes_, dGroupStarts_��ֵ
		particleSystemToPoints(dPoints_, dColors_, dSizes_, dGroupStarts_,
			dStartFrames_, nParticleGroups_, dDirections_, dSpeeds_, 
			dStartPoses_, currFrame, nFrames_, dColorMatrix_, dSizeMatrix_);

		realNGroups_ = compress(dPoints_, dColors_, dSizes_,
			nParticleGroups_, nFrames_, dGroupOffsets_, dGroupStarts_);

		if (realNGroups_ > 0) {

			interpolation(dPoints_, dColors_, dSizes_, dGroupOffsets_,
				realNGroups_, nFrames_, nInterpolation_);

			size_t shiftSize = nFrames_ * (nInterpolation_ + 1) - nInterpolation_;
			calcFinalPosition(dPoints_, realNGroups_, nFrames_ * nInterpolation_,
				nInterpolation_, currFrame, dGroupOffsets_, dGroupStarts_, dShiftX_,
				dShiftY_, shiftSize);

			// ӳ��buffer���ڴ�ָ��
			CUDACHECK(cudaGraphicsMapResources(1, &cuda_vbo_resource_, 0));
			CUDACHECK(cudaGraphicsMapResources(1, &cuda_ebo_resource_, 0));
			void *pVboData, *pEboData;
			size_t sizeVbo, sizeEbo;
			CUDACHECK(cudaGraphicsResourceGetMappedPointer(
				&pVboData, &sizeVbo, cuda_vbo_resource_));
			CUDACHECK(cudaGraphicsResourceGetMappedPointer(
				&pEboData, &sizeEbo, cuda_ebo_resource_));

			eboSize_ = pointToLine(dPoints_, dSizes_, dColors_,
				nFrames_ * (nInterpolation_ + 1), dGroupOffsets_, realNGroups_,
				static_cast<float*>(pVboData), static_cast<GLuint*>(pEboData));

			// �ͷŶ�buffer���ڴ�ָ��ӳ��
			CUDACHECK(cudaGraphicsUnmapResources(1, &cuda_vbo_resource_, 0));
			CUDACHECK(cudaGraphicsUnmapResources(1, &cuda_ebo_resource_, 0));

			glDeleteVertexArrays(1, &vao);
			glGenVertexArrays(1, &vao);
			glBindVertexArray(vao);
			glBindBuffer(GL_ARRAY_BUFFER, vbo);
			glEnableVertexAttribArray(0);
			glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE,
				6 * sizeof(GLfloat), (GLvoid*)0);
			glEnableVertexAttribArray(1);
			glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE,
				6 * sizeof(GLfloat), (GLvoid*)(3 * sizeof(GLfloat)));
			glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, ebo);
			glBindVertexArray(0);
			glBindBuffer(GL_ARRAY_BUFFER, 0);
			glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0);
		} else {
			eboSize_ = 0;
		}
	}

public:
	~NormalFirework() {
		deleteBuffer();
		releaseResources();
	}
};