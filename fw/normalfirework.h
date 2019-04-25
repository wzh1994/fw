#pragma once
#include "firework.h"
#include "kernels.h"
#include <cuda_runtime.h>
#include <cuda_gl_interop.h>
#include "test.h"
#include "compare.h"
#ifdef USE_CUDA_KERNEL //������vs���̵�Ԥ��������
using namespace cudaKernel;
#else
using namespace hostFunction;
#endif
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
	}

	// ��������һ��
	void initialize() override {
		// ���ø���ĳ�ʼ��
		FwBase::initialize();
		CUDACHECK(cudaMallocAlign(&dColorMatrix_, 3 * nFrames_ * nFrames_ * sizeof(float)));
		CUDACHECK(cudaMallocAlign(&dSizeMatrix_, nFrames_ * nFrames_ * sizeof(float)));

		// �ڵ���initDirections֮��nParticleGroups_ ����ֵ
		initDirections();
		
		// ��vbo��ebo��Ԥ�ȷ���һ���ϴ��Ŀռ䣬�Ժ󷴸�ʹ��
		// Ŀǰÿ��buffer��������80MB�Դ棨�ڲ���һ��sizeof(float)��
		genBuffer(20000000, 20000000);
		// Ϊ���ٶȣ���ʼλ�õȷ���ռ�
		CUDACHECK(cudaMallocAlign(&dSpeeds_, nParticleGroups_ * sizeof(float)));
		CUDACHECK(cudaMallocAlign(&dStartPoses_, 3 * nParticleGroups_ * sizeof(float)));
		CUDACHECK(cudaMallocAlign(&dStartFrames_, nParticleGroups_ * sizeof(size_t)));

		size_t maxSize = (nInterpolation_ + 1) * nParticleGroups_ * nFrames_;
		CUDACHECK(cudaMallocAlign(&dPoints_, 3 * maxSize * sizeof(float)));
		CUDACHECK(cudaMallocAlign(&dColors_, 3 * maxSize * sizeof(float)));
		CUDACHECK(cudaMallocAlign(&dSizes_, maxSize * sizeof(float)));
		CUDACHECK(cudaMallocAlign(&dGroupStarts_, nParticleGroups_ * sizeof(size_t)));
		CUDACHECK(cudaMallocAlign(&dGroupOffsets_, (nParticleGroups_ + 1) * sizeof(size_t)));

		size_t shiftSize = (nInterpolation_ + 1) * nFrames_;
		shiftSize *= shiftSize;

		CUDACHECK(cudaMallocAlign(&dShiftX_, shiftSize * sizeof(float)));
		CUDACHECK(cudaMallocAlign(&dShiftY_, shiftSize * sizeof(float)));
		
		// �����Դ涼������֮��ſ��Ե���prepare
		prepare();
	}
	
	void initDirections() {
		// �Ȼ�ȡ���еķ���, ��dDirections_��ֵ
		nParticleGroups_ = 100;
		float* directions = new float[3 * nParticleGroups_] {};
		for (int i = 0; i < nParticleGroups_; ++i) {
			directions[3 * i] = 1 - 0.02 * i;
			directions[3 * i + 1] = 0.02 * i;
			directions[3 * i + 2] = sin(i);
		}
		CUDACHECK(cudaMallocAlign(&dDirections_, 3 * nParticleGroups_ * sizeof(float)));
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

		scale(dSizeMatrix_, scaleRate_, nFrames_ * nFrames_);

		/*show(dColorMatrix_, 3 * nFrames_ * nFrames_, 3 * nFrames_);
		printSplitLine();
		show(dSizeMatrix_, nFrames_ * nFrames_, nFrames_);
		printSplitLine();*/

		// ��ȡ���ӵ��ٶȣ� ���ٶ�
		fill(dSpeeds_, args_[6 * nFrames_ + 2] * scaleRate_, nParticleGroups_);
		//show(dSpeeds_, nParticleGroups_);
		//printSplitLine();

		fill(dStartPoses_, args_ + 6 * nFrames_ + 3, nParticleGroups_, 3);
		scale(dStartPoses_, scaleRate_, 3 * nParticleGroups_);
		/*show(dStartPoses_, 3 * nParticleGroups_);
		printSplitLine();*/

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

		//show(dShiftX_, shiftSize, nInterpolation_ * (nFrames_ - 1) + nFrames_);
		//show(dShiftY_, shiftSize, nInterpolation_ * (nFrames_ - 1) + nFrames_);
	}

	void GetParticles(int currFrame) override {
		static compare::Compare comp;
		// �˴���dPoints_, dColors_, dSizes_, dGroupStarts_��ֵ
		particleSystemToPoints(dPoints_, dColors_, dSizes_, dGroupStarts_,
			dStartFrames_, nParticleGroups_, dDirections_, dSpeeds_, 
			dStartPoses_, currFrame, nFrames_, dColorMatrix_, dSizeMatrix_);

		// �۲�particleSystemToPoints���� ������ͬ֡���������Ƿ�һ��
		/*FW_ASSERT(comp.compare("points", currFrame, dPoints_, 3 * nFrames_ * nParticleGroups_));
		FW_ASSERT(comp.compare("colors", currFrame, dColors_, 3 * nFrames_ * nParticleGroups_));
		FW_ASSERT(comp.compare("point", currFrame, dSizes_, nFrames_ * nParticleGroups_));
		FW_ASSERT(comp.compare("group_start", currFrame, dGroupStarts_, nParticleGroups_));*/
		

		CUDACHECK(cudaDeviceSynchronize());
		/*show(dPoints_, 3 * nParticleGroups_ * nFrames_, 3 * nFrames_);
		printSplitLine();
		show(dColors_, 3 * nParticleGroups_ * nFrames_, 3 * nFrames_);
		printSplitLine();
		show(dSizes_, nParticleGroups_ * nFrames_, nFrames_);
		printSplitLine();
		show(dGroupStarts_, nParticleGroups_);*/

		realNGroups_ = compress(dPoints_, dColors_, dSizes_,
			nParticleGroups_, nFrames_, dGroupOffsets_, dGroupStarts_);
		// �۲�compress���� ������ͬ֡���������Ƿ�һ��
		/*{
			size_t *dRealNGroup;
			cudaMallocAndCopy(dRealNGroup, &realNGroups_, 1);
			FW_ASSERT(comp.compare("realGroup", currFrame, dRealNGroup, 1));
			FW_ASSERT(comp.compare("groupOffsets", currFrame, dGroupOffsets_, realNGroups_ + 1));
			FW_ASSERT(comp.compare("groupStarts", currFrame, dGroupStarts_, realNGroups_));
			size_t resultSize;
			CUDACHECK(cudaMemcpy(&resultSize, dGroupOffsets_ + realNGroups_, sizeof(size_t), cudaMemcpyDeviceToHost));
			FW_ASSERT(comp.compare("pointsAfterCompress", currFrame, dPoints_, 3 * resultSize));
			FW_ASSERT(comp.compare("colorsAfterCompress", currFrame, dColors_, 3 * resultSize));
			FW_ASSERT(comp.compare("pointAfterCompress", currFrame, dSizes_, resultSize));
		}*/

		CUDACHECK(cudaDeviceSynchronize());
		if (realNGroups_ > 0) {
			// �۲�interpolation֮ǰ�����Ƿ�һ��
			/*{
				show(dGroupOffsets_, realNGroups_ + 1);
				size_t resultSize;
				CUDACHECK(cudaMemcpy(&resultSize, dGroupOffsets_ + realNGroups_,
					sizeof(size_t), cudaMemcpyDeviceToHost));
				printf("input size %llu\n", resultSize);
				FW_ASSERT(comp.compare("pointsBeforeInterpolation",
					currFrame, dPoints_, 3 * resultSize));
				FW_ASSERT(comp.compare("colorsBeforeInterpolation",
					currFrame, dColors_, 3 * resultSize));
				FW_ASSERT(comp.compare("pointBeforeInterpolation",
					currFrame, dSizes_, resultSize));
				show(dPoints_, dGroupOffsets_, realNGroups_, 3);
			}*/
			interpolation(dPoints_, dColors_, dSizes_, dGroupOffsets_,
				realNGroups_, nFrames_, nInterpolation_);
			// �۲�interpolation���� ������ͬ֡���������Ƿ�һ��
			/*{
				show(dGroupOffsets_, realNGroups_ + 1);
				size_t resultSize;
				CUDACHECK(cudaMemcpy(&resultSize, dGroupOffsets_ + realNGroups_,
					sizeof(size_t), cudaMemcpyDeviceToHost));
				printf("resultSize %llu\n", resultSize);
				FW_ASSERT(comp.compare("pointsAfterInterpolation",
					currFrame, dPoints_, 3 * resultSize));
				FW_ASSERT(comp.compare("colorsAfterInterpolation",
					currFrame, dColors_, 3 * resultSize));
				FW_ASSERT(comp.compare("pointAfterInterpolation",
					currFrame, dSizes_, resultSize));
			}*/
			CUDACHECK(cudaDeviceSynchronize());
			size_t shiftSize = nFrames_ * (nInterpolation_ + 1) - nInterpolation_;
			calcFinalPosition(dPoints_, realNGroups_, nFrames_ * nInterpolation_,
				nInterpolation_, currFrame, dGroupOffsets_, dGroupStarts_, dShiftX_,
				dShiftY_, shiftSize);
			// �۲�finalPosition���� ������ͬ֡���������Ƿ�һ��
			/*{
				size_t resultSize;
				CUDACHECK(cudaMemcpy(&resultSize, dGroupOffsets_ + realNGroups_,
					sizeof(size_t), cudaMemcpyDeviceToHost));
				FW_ASSERT(comp.compare("pointsAfterInterpolation",
					currFrame, dPoints_, 3 * resultSize));
			}*/
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
			{
				size_t *dEboSize;
				cudaMallocAndCopy(dEboSize, &eboSize_, 1);
				FW_ASSERT(comp.compare("eboSize", currFrame, dEboSize, 1));
			}
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