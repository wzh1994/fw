#pragma once
#include "firework.h"
#include <cuda_runtime.h>
#include <cuda_gl_interop.h>
#include "kernels.h"
#include "Shader.h"
#include <utils.h>
#include "test.h"

#ifdef USE_CUDA_KERNEL //������vs���̵�Ԥ��������
using namespace cudaKernel;
#else
using namespace hostFunction;
#endif

namespace firework {

static constexpr size_t kMaxParticleGroup = 2000;

class FwRenderBase : public FwBase {
protected:
	GLuint vbo = 0;
	GLuint ebo = 0;
	GLuint vao = 0;
	size_t eboSize_ = 0;
	float scaleRate_ = 0;
	struct cudaGraphicsResource *cuda_vbo_resource_, *cuda_ebo_resource_;
	size_t nVboToInit_;
	size_t nEboToInit_;

protected:
	// �����������̻�����ʱ����ʹ�õ�ָ��ͱ���
	float *dColorMatrix_, *dSizeMatrix_;
	float *dPoints_, *dColors_, *dSizes_;
	size_t nInterpolation_;

	// ����ϵͳ���
	size_t nParticleGroups_, maxNParticleGroups_;
	float *dDirections_, *dSpeed_, *dCentrifugalPos_, *dStartPoses_;
	size_t *dStartFrames_, *dGroupStarts_, *dGroupOffsets_, *dLifeTime_;
	size_t realNGroups_;
	float innerSize_, innerColor_;

	// �������
	float *dShiftX_, *dShiftY_;

	// ��������
	curandState *devStates_;
	float visibleRate_;

	// ��ɫ��
	std::unique_ptr<Shader> shader_;

protected:
	FwRenderBase(float* args)
		: FwBase(args)
		, devStates_(nullptr)
		, visibleRate_(1.0f) {}

public:
	// ��opengl�йصĳ�ʼ����������Ա�֤����glewInit֮��ִ��
	// �����дʱ�����������ʽ�ĵ��ñ�������
	void initialize() override{
		FwBase::initialize();
		shader_.reset(new Shader("fw.vs", "fw.fs"));

	}

	/* ==========================================
	 * ������ʵ�����������ڲ�����
	 * ==========================================
	 */
private:
	virtual void getPoints(int currFrame) = 0;
	//������ڴ������ͷ�
	virtual void allocAppendixResource() = 0;
	virtual void releaseAppendixResource() = 0;

protected:
	void allocStaticResources() {
		// ��vbo��ebo��Ԥ�ȷ���һ���ϴ��Ŀռ䣬�Ժ󷴸�ʹ��
		// Ŀǰÿ��buffer��������800MB�Դ棨�ڲ���һ��sizeof(float)��
		genBuffer(nVboToInit_, nEboToInit_);
		CUDACHECK(cudaDeviceSynchronize());
		CUDACHECK(cudaMallocAlign(
			&dColorMatrix_, 3 * nFrames_ * nFrames_ * sizeof(float)));
		CUDACHECK(cudaMallocAlign(
			&dSizeMatrix_, nFrames_ * nFrames_ * sizeof(float)));

		CUDACHECK(cudaMallocAlign(
			&dDirections_, 3 * kMaxParticleGroup * sizeof(float)));

		size_t shiftSize = (nInterpolation_ + 1) * nFrames_;
		shiftSize *= shiftSize;

		CUDACHECK(cudaMallocAlign(&dShiftX_, shiftSize * sizeof(float)));
		CUDACHECK(cudaMallocAlign(&dShiftY_, shiftSize * sizeof(float)));
	}

	void releaseStaticResources() {
		deleteBuffer();
		cudaFreeAll(dColorMatrix_, dSizeMatrix_, dDirections_,
			dShiftX_, dShiftY_);
	}

	void allocDynamicResources() {

		// Ϊ���ٶȣ���ʼλ�õȷ���ռ�
		CUDACHECK(cudaMallocAlign(
			&dSpeed_, (nFrames_ + 1) * sizeof(float)));
		CUDACHECK(cudaMallocAlign(
			&dCentrifugalPos_, (nFrames_ + 1) * sizeof(float)));
		CUDACHECK(cudaMallocAlign(
			&dStartPoses_, 3 * nParticleGroups_ * sizeof(float)));
		CUDACHECK(cudaMallocAlign(
			&dStartFrames_, nParticleGroups_ * sizeof(size_t)));

		size_t maxSize = (nInterpolation_ + 1) * nParticleGroups_ * nFrames_;
		CUDACHECK(cudaMallocAlign(&dPoints_, 3 * maxSize * sizeof(float)));
		CUDACHECK(cudaMallocAlign(&dColors_, 3 * maxSize * sizeof(float)));
		CUDACHECK(cudaMallocAlign(&dSizes_, maxSize * sizeof(float)));
		CUDACHECK(cudaMallocAlign(
			&dGroupStarts_, nParticleGroups_ * sizeof(size_t)));
		CUDACHECK(cudaMallocAlign(
			&dGroupOffsets_, (nParticleGroups_ + 1) * sizeof(size_t)));
		CUDACHECK(cudaMallocAlign(
			&dLifeTime_, nParticleGroups_ * sizeof(size_t)));
		allocAppendixResource();
	}

	void releaseDynamicResources() {
		cudaFreeAll(dSpeed_, dCentrifugalPos_, dStartPoses_, dStartFrames_,
			dPoints_, dColors_, dSizes_, dGroupStarts_, dGroupOffsets_, 
			dLifeTime_);

		releaseAppendixResource();
	}

protected:
	void genBuffer(size_t vboSize, size_t eboSize) {
		glGenBuffers(1, &vbo);
		glBindBuffer(GL_ARRAY_BUFFER, vbo);
		glBufferData(GL_ARRAY_BUFFER,
			vboSize * sizeof(float), nullptr, GL_DYNAMIC_DRAW);
		glBindBuffer(GL_ARRAY_BUFFER, 0);
		cudaGraphicsGLRegisterBuffer(&cuda_vbo_resource_, vbo,
			cudaGraphicsMapFlagsWriteDiscard);

		glGenBuffers(1, &ebo);
		glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, ebo);
		glBufferData(GL_ELEMENT_ARRAY_BUFFER,
			eboSize * sizeof(int), nullptr, GL_DYNAMIC_DRAW);
		glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0);
		cudaGraphicsGLRegisterBuffer(&cuda_ebo_resource_, ebo,
			cudaGraphicsMapFlagsWriteDiscard);
	}

	void deleteBuffer() {
		glDeleteBuffers(1, &vbo);
		glDeleteBuffers(1, &ebo);
		glDeleteVertexArrays(1, &vao);
	}


/* ==========================================
 * �ⲿ�ӿ�
 * ==========================================
 */
public:

	// ��ȡһ��֡����������
	void GetParticles(int currFrame) final{

		// �˴���dPoints_, dColors_, dSizes_, dGroupStarts_��ֵ
		getPoints(currFrame);


		CUDACHECK(cudaDeviceSynchronize());
		if (visibleRate_ < 1.0f) {
			realNGroups_ = compress(dPoints_, dColors_,
				dSizes_, nParticleGroups_, nFrames_, dGroupOffsets_,
				dGroupStarts_, visibleRate_, devStates_);
		} else {
			realNGroups_ = compress(
				dPoints_, dColors_, dSizes_, nParticleGroups_,
				nFrames_, dGroupOffsets_, dGroupStarts_);
		}
		CUDACHECK(cudaDeviceSynchronize());
		if (realNGroups_ > 0) {
			//printSplitLine("before");
			//show(dPoints_, dGroupOffsets_, realNGroups_, 3);
			//printSplitLine("after");
			interpolation(dPoints_, dColors_, dSizes_, dGroupOffsets_,
				realNGroups_, nFrames_, nInterpolation_);
			//show(dPoints_, dGroupOffsets_, realNGroups_, 3);
			CUDACHECK(cudaDeviceSynchronize());
			size_t shiftSize =
				nFrames_ * (nInterpolation_ + 1) - nInterpolation_;
			calcFinalPosition(dPoints_, realNGroups_,
				shiftSize, nInterpolation_, currFrame, dGroupOffsets_,
				dGroupStarts_, dStartFrames_, dShiftX_, dShiftY_, shiftSize);

			CUDACHECK(cudaDeviceSynchronize());
			// ӳ��buffer���ڴ�ָ��
			CUDACHECK(cudaGraphicsMapResources(1, &cuda_vbo_resource_, 0));
			CUDACHECK(cudaGraphicsMapResources(1, &cuda_ebo_resource_, 0));
			void *pVboData, *pEboData;
			size_t sizeVbo, sizeEbo;
			CUDACHECK(cudaGraphicsResourceGetMappedPointer(
				&pVboData, &sizeVbo, cuda_vbo_resource_));
			CUDACHECK(cudaGraphicsResourceGetMappedPointer(
				&pEboData, &sizeEbo, cuda_ebo_resource_));
			CUDACHECK(cudaDeviceSynchronize());

			eboSize_ = pointToLine(dPoints_, dSizes_, dColors_,
				nFrames_ * (nInterpolation_ + 1), dGroupOffsets_, realNGroups_,
				static_cast<float*>(pVboData), static_cast<GLuint*>(pEboData),
				0.5, innerSize_, innerColor_);

			CUDACHECK(cudaDeviceSynchronize());

			// �ͷŶ�buffer���ڴ�ָ��ӳ��
			CUDACHECK(cudaGraphicsUnmapResources(1, &cuda_vbo_resource_, 0));
			CUDACHECK(cudaGraphicsUnmapResources(1, &cuda_ebo_resource_, 0));

			glDeleteVertexArrays(1, &vao);
			glGenVertexArrays(1, &vao);
			glBindVertexArray(vao);
			glBindBuffer(GL_ARRAY_BUFFER, vbo);
			glEnableVertexAttribArray(0);
			glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE,
				7 * sizeof(GLfloat), (GLvoid*)0);
			glEnableVertexAttribArray(1);
			glVertexAttribPointer(1, 4, GL_FLOAT, GL_FALSE,
				7 * sizeof(GLfloat), (GLvoid*)(3 * sizeof(GLfloat)));
			glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, ebo);
			glBindVertexArray(0);
			glBindBuffer(GL_ARRAY_BUFFER, 0);
			glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0);
		}
		else {
			eboSize_ = 0;
		}
	}

	void RenderScene(const Camera& camera) override {
		shader_->use();
		shader_->setMat4("view", camera.GetViewMatrix());
		shader_->setMat4("projection", camera.GetProjectionMatrix());
		if (eboSize_ > 0) {
			glBindVertexArray(vao);
			// draw points 0-3 from the currently bound VAO with current in-use shader;
			glEnable(GL_BLEND);
			glBlendFunc(GL_SRC_ALPHA, GL_DST_ALPHA);
			glBlendEquation(GL_MAX);
			glDepthMask(GL_FALSE);
			glDrawElements(GL_TRIANGLES, eboSize_, GL_UNSIGNED_INT, 0);
			glDepthMask(GL_TRUE);
			glDisable(GL_BLEND);
			glBindVertexArray(0);
		}
	}
};

}