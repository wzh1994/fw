#pragma once
#include <vector>
#include <glm.hpp>
#include <memory>
#include "particle.h"
#include "Shader.h"
#include "Camera.h" 
#include <cuda_runtime.h>
#include <cuda_gl_interop.h>
#include "kernels.h"
#ifdef USE_CUDA_KERNEL //������vs���̵�Ԥ��������
using namespace cudaKernel;
#else
using namespace hostFunction;
#endif

namespace firework{

static constexpr size_t kMaxParticleGroup = 2000;

class CfwDlg;
enum class ArgType {
	Scalar = 0,
	Color,
	Vector3,
	ScalarGroup,
	ColorGroup,
	Vec3Group
};

class FwBase {
protected:
	struct Attr {
		ArgType type;
		std::wstring name;
		size_t start;
		size_t offset;
		size_t stride;
		static size_t nFrames;
		static size_t idx;
		static size_t groupOffset;
		static size_t groupNums;
		static size_t groupStep;
		Attr(ArgType type, std::wstring name)
			: type(type)
			, name(name)
			, start(idx)
			, offset(0)
			, stride(0) {
			switch (type) {
			case ArgType::Scalar:
				idx += 1;
				break;
			case ArgType::Color:
			case ArgType::Vector3:
				idx += 3;
				break;
			case ArgType::ScalarGroup:
			case ArgType::ColorGroup:
			case ArgType::Vec3Group:
				offset = groupOffset;
				groupOffset += groupStep;
				stride = groupNums * groupStep;
				break;
			default:
				FW_NOTSUPPORTED << "Unexpected attr type!";
			}
		}

		static void startGroup(int n, int step, int nFrames) {
			FW_ASSERT(n > 0 && step > 0);
			groupOffset = 0;
			groupNums = n;
			groupStep = step;
			Attr::nFrames = nFrames;
		}

		static void stopGroup() {
			FW_ASSERT(groupNums * groupStep == groupOffset) << "Check whether"
					"the number of Attr is the same as set in startGroup";
			idx += nFrames * groupNums * groupStep;
		}
	};

protected:
	std::vector<Attr> attrs_;
	float* args_;
	// nFrames_������ֱ�BeginGroup����ʹ�ã��޸ĵ�ʱ����Ҫע��һ��
	size_t nFrames_;
	std::unique_ptr<Shader> shader_;

protected:
	GLuint vbo = 0;
	GLuint ebo = 0;
	GLuint vao = 0;
	size_t eboSize_ = 0;
	float scaleRate_ = 0;
	struct cudaGraphicsResource *cuda_vbo_resource_, *cuda_ebo_resource_;

protected:
	// �����������̻�����ʱ����ʹ�õ�ָ��ͱ���
	float *dColorMatrix_, *dSizeMatrix_;
	float *dPoints_, *dColors_, *dSizes_;
	size_t nInterpolation_;

	// ����ϵͳ���
	size_t nParticleGroups_, maxNParticleGroups_;
	float *dDirections_, *dSpeeds_, *dStartPoses_;
	size_t *dStartFrames_, *dGroupStarts_, *dGroupOffsets_, *dLifeTime_;
	size_t realNGroups_;

	// �������
	float *dShiftX_, *dShiftY_;

protected:
	FwBase(float* args) : args_(args) {}

/* ==========================================
 * ������ʵ�����������ӿڷ���
 * ==========================================
 */
public:
	// ��opengl�йصĳ�ʼ����������Ա�֤����glewInit֮��ִ��
	// �����дʱ�����������ʽ�ĵ��ñ�������
	virtual void initialize() {
		static bool isInited = false;
		FW_ASSERT(!isInited) << "Cannot init fw class more than once!";
		shader_.reset(new Shader("fw.vs", "fw.fs"));
		isInited = true;
	}



	// ÿ�β����ı�֮����Ҫ���ñ�����
	virtual void prepare() = 0;

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
		genBuffer(200000000, 200000000);
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
		CUDACHECK(cudaFree(dColorMatrix_));
		CUDACHECK(cudaFree(dSizeMatrix_));
		CUDACHECK(cudaFree(dDirections_));
		CUDACHECK(cudaFree(dShiftX_));
		CUDACHECK(cudaFree(dShiftY_));
	}

	void allocDynamicResources() {

		// Ϊ���ٶȣ���ʼλ�õȷ���ռ�
		CUDACHECK(cudaMallocAlign(
			&dSpeeds_, nParticleGroups_ * sizeof(float)));
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
		CUDACHECK(cudaFree(dSpeeds_));
		CUDACHECK(cudaFree(dStartPoses_));
		CUDACHECK(cudaFree(dStartFrames_));

		CUDACHECK(cudaFree(dPoints_));
		CUDACHECK(cudaFree(dColors_));
		CUDACHECK(cudaFree(dSizes_));
		CUDACHECK(cudaFree(dGroupStarts_));
		CUDACHECK(cudaFree(dGroupOffsets_));
		CUDACHECK(cudaFree(dLifeTime_));

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
	void GetParticles(int currFrame) {

		// �˴���dPoints_, dColors_, dSizes_, dGroupStarts_��ֵ
		getPoints(currFrame);


		CUDACHECK(cudaDeviceSynchronize());

		realNGroups_ = compress(dPoints_, dColors_, dSizes_,
			nParticleGroups_, nFrames_, dGroupOffsets_, dGroupStarts_);

		CUDACHECK(cudaDeviceSynchronize());
		if (realNGroups_ > 0) {
			interpolation(dPoints_, dColors_, dSizes_, dGroupOffsets_,
				realNGroups_, nFrames_, nInterpolation_);

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
				static_cast<float*>(pVboData), static_cast<GLuint*>(pEboData));

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
				6 * sizeof(GLfloat), (GLvoid*)0);
			glEnableVertexAttribArray(1);
			glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE,
				6 * sizeof(GLfloat), (GLvoid*)(3 * sizeof(GLfloat)));
			glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, ebo);
			glBindVertexArray(0);
			glBindBuffer(GL_ARRAY_BUFFER, 0);
			glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0);
		}
		else {
			eboSize_ = 0;
		}
	}

	size_t getTotalFrame() {
		return nFrames_;
	}

	std::vector<Attr>& attrs() {
		return attrs_;
	}

	float* getArgs(size_t idx, size_t frame) {
		return args_ + attrs_[idx].start +
			attrs_[idx].offset + frame * attrs_[idx].stride;
	}

	void RenderScene(const Camera& camera) {
		shader_->use();
		shader_->setMat4("view", camera.GetViewMatrix());
		shader_->setMat4("projection", camera.GetProjectionMatrix());
		if (eboSize_ > 0) {
			glBindVertexArray(vao);
			// draw points 0-3 from the currently bound VAO with current in-use shader;
			glDrawElements(GL_TRIANGLES, eboSize_, GL_UNSIGNED_INT, 0);
			glBindVertexArray(0);
		}
	}

	virtual ~FwBase() = default;
};

enum class FireWorkType {
	Normal = 0,
	Mixture,
	MultiExplosion
};


// ��������FireWork���Ψһ��ں���
FwBase* getFirework(FireWorkType type, float* args);

}

// useful definations used in sub classes
#define AddValue(_name) \
	attrs_.push_back(Attr(ArgType::Scalar, L##_name));

#define AddColor(_name) \
	attrs_.push_back(Attr(ArgType::Color, L##_name));

#define AddVec3(_name) \
	attrs_.push_back(Attr(ArgType::Vector3, L##_name));

#define BeginGroup(_n, _step) Attr::startGroup((_n), (_step), nFrames_)
#define EndGroup() Attr::stopGroup()

#define AddColorGroup(_name) \
	attrs_.push_back(Attr(ArgType::ColorGroup, L##_name));

#define AddVec3Group(_name) \
	attrs_.push_back(Attr(ArgType::Vec3Group, L##_name));

#define AddScalarGroup(_name) \
	attrs_.push_back(Attr(ArgType::ScalarGroup, L##_name));