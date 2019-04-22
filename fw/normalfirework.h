#pragma once
#include "firework.h"
#include "kernels.h"
#include <cuda_runtime.h>
#include <cuda_gl_interop.h>

class NormalFirework : public FwBase {
	friend FwBase* getFirework(FireWorkType type, float* args);

private:
	float *dColorMatrix_, *dSizeMatrix_;
	float *dPoints_, *dColors_, *dSizes_;
	size_t nInterpolation_ = 15;
	
	// ��ʼ������ϵͳ���
	size_t nParticleGroups_;
	float *dDirections_, *dSpeeds_, *dStartPoses_;
	size_t *dStartFrames_, *dGroupStarts_;

private:
	NormalFirework(float* args) : FwBase(args) {
		nFrames_ = 49;
		nInterpolation_ = 15;
		BeginGroup(4, 3);
			AddVec3Group("����1");
			AddVec3Group("����2");
			AddVec3Group("����3");
			AddVec3Group("����4");
		EndGroup(12);
		BeginGroup(4, 3);
			AddVec3Group("��ɫ1");
			AddVec3Group("��ɫ2");
			AddVec3Group("��ɫ3");
			AddVec3Group("��ɫ4");
		EndGroup(12);
	}

	// ��������һ��
	void initialize() override {
		// ���ø���ĳ�ʼ��
		FwBase::initialize();
		genBuffer(5000000, 5000000);
		cudaMalloc(&dColorMatrix_, 3 * nFrames_ * nFrames_ * sizeof(float));
		cudaMalloc(&dSizeMatrix_, nFrames_ * nFrames_ * sizeof(float));

		// �ڵ���initParticleSystem֮��nParticleGroups_ ����ֵ

		initParticleSystem();

		size_t maxSize = (nInterpolation_ + 1) * nParticleGroups_ * nFrames_;
		cudaMalloc(&dPoints_, 3 * maxSize * sizeof(float));
		cudaMalloc(&dColors_, 3 * maxSize * sizeof(float));
		cudaMalloc(&dSizes_, maxSize * sizeof(float));
		cudaMalloc(&dGroupStarts_, nParticleGroups_ * sizeof(size_t));
		
		prepare();
	}
	
	void initParticleSystem() {
		nParticleGroups_ = 10;
		cudaMalloc(&dDirections_, 3 * nParticleGroups_ * sizeof(float));
		cudaMalloc(&dSpeeds_, 3 * nParticleGroups_ * sizeof(float));
		cudaMalloc(&dStartPoses_, 3 * nParticleGroups_ * sizeof(float));
		cudaMalloc(&dStartFrames_, nParticleGroups_ * sizeof(size_t));
	}

	void releaseResources() {
		cudaFree(dColorMatrix_);
		cudaFree(dSizeMatrix_);

		cudaFree(dDirections_);
		cudaFree(dSpeeds_);
		cudaFree(dStartPoses_);
		cudaFree(dStartFrames_);

		cudaFree(dPoints_);
		cudaFree(dColors_);
		cudaFree(dSizes_);
		cudaFree(dGroupStarts_);
	}



public:
	// ÿ�β����ı�֮����Ҫ���ñ�����
	void prepare() {
		getColorAndSizeMatrix(args_, args_ + 3 * nFrames_, nFrames_,
			args_[4 * nFrames_], args_[4 * nFrames_ + 1],
			dColorMatrix_, dSizeMatrix_);
	}

	void GetParticles(int currFrame) override {
		// ӳ��buffer���ڴ�ָ��
		cudaGraphicsMapResources(1, &cuda_vbo_resource, 0);
		cudaGraphicsMapResources(1, &cuda_ebo_resource, 0);
		void *pVboData, *pEboData;
		size_t sizeVbo, sizeEbo;
		cudaGraphicsResourceGetMappedPointer(
			&pVboData, &sizeVbo, cuda_vbo_resource);
		cudaGraphicsResourceGetMappedPointer(
			&pEboData, &sizeEbo, cuda_ebo_resource);

		particleSystemToPoints(dPoints_, dColors_, dSizes_, dGroupStarts_,
			dStartFrames_, nParticleGroups_, dDirections_, dSpeeds_, 
			dStartPoses_, currFrame, nFrames_, dColorMatrix_, dSizeMatrix_);

	
		//static_cast<float*>(pVboData)
		//static_cast<GLuint*>(pEboData)

		// �ͷŶ�buffer���ڴ�ָ��ӳ��
		cudaGraphicsUnmapResources(1, &cuda_vbo_resource, 0);
		cudaGraphicsUnmapResources(1, &cuda_ebo_resource, 0);

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
	}

public:
	~NormalFirework() {
		deleteBuffer();
		releaseResources();
	}
};