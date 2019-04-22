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
	
	// 初始化粒子系统相关
	size_t nParticleGroups_;
	float *dDirections_, *dSpeeds_, *dStartPoses_;
	size_t *dStartFrames_, *dGroupStarts_;

private:
	NormalFirework(float* args) : FwBase(args) {
		nFrames_ = 49;
		nInterpolation_ = 15;
		BeginGroup(4, 3);
			AddVec3Group("顶点1");
			AddVec3Group("顶点2");
			AddVec3Group("顶点3");
			AddVec3Group("顶点4");
		EndGroup(12);
		BeginGroup(4, 3);
			AddVec3Group("颜色1");
			AddVec3Group("颜色2");
			AddVec3Group("颜色3");
			AddVec3Group("颜色4");
		EndGroup(12);
	}

	// 仅被调用一次
	void initialize() override {
		// 调用父类的初始化
		FwBase::initialize();
		genBuffer(5000000, 5000000);
		cudaMalloc(&dColorMatrix_, 3 * nFrames_ * nFrames_ * sizeof(float));
		cudaMalloc(&dSizeMatrix_, nFrames_ * nFrames_ * sizeof(float));

		// 在调用initParticleSystem之后nParticleGroups_ 才有值

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
	// 每次参数改变之后，需要调用本方法
	void prepare() {
		getColorAndSizeMatrix(args_, args_ + 3 * nFrames_, nFrames_,
			args_[4 * nFrames_], args_[4 * nFrames_ + 1],
			dColorMatrix_, dSizeMatrix_);
	}

	void GetParticles(int currFrame) override {
		// 映射buffer的内存指针
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

		// 释放对buffer的内存指针映射
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