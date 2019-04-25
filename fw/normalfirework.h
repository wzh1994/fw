#pragma once
#include "firework.h"
#include "kernels.h"
#include <cuda_runtime.h>
#include <cuda_gl_interop.h>
#include "test.h"
#include "compare.h"
#ifdef USE_CUDA_KERNEL //定义在vs工程的预处理器中
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
	
	// 粒子系统相关
	size_t nParticleGroups_;
	float *dDirections_, *dSpeeds_, *dStartPoses_;
	size_t *dStartFrames_, *dGroupStarts_, *dGroupOffsets_;
	size_t realNGroups_;

	// 外力相关
	float *dShiftX_, *dShiftY_;

private:
	NormalFirework(float* args) : FwBase(args) {
		nFrames_ = 49;
		nInterpolation_ = 15;
		scaleRate_ = 0.0025f;
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
	}

	// 仅被调用一次
	void initialize() override {
		// 调用父类的初始化
		FwBase::initialize();
		CUDACHECK(cudaMallocAlign(&dColorMatrix_, 3 * nFrames_ * nFrames_ * sizeof(float)));
		CUDACHECK(cudaMallocAlign(&dSizeMatrix_, nFrames_ * nFrames_ * sizeof(float)));

		// 在调用initDirections之后nParticleGroups_ 才有值
		initDirections();
		
		// 给vbo和ebo都预先分配一个较大点的空间，以后反复使用
		// 目前每个buffer被分配了80MB显存（内部有一个sizeof(float)）
		genBuffer(20000000, 20000000);
		// 为初速度，初始位置等分配空间
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
		
		// 所有显存都被分配之后才可以调用prepare
		prepare();
	}
	
	void initDirections() {
		// 先获取所有的方向, 给dDirections_赋值
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
	// 每次参数改变之后，需要调用本方法
	// 本方法会给dColorMatrix_, dSizeMatrix_, dSpeeds_, dStartPoses_和
	// dStartFrames_赋予初值
	void prepare() final {
		// 获取颜色和尺寸变化情况矩阵
		getColorAndSizeMatrix(args_, args_ + 3 * nFrames_, nFrames_,
			args_[6 * nFrames_], args_[6 * nFrames_ + 1],
			dColorMatrix_, dSizeMatrix_);

		scale(dSizeMatrix_, scaleRate_, nFrames_ * nFrames_);

		/*show(dColorMatrix_, 3 * nFrames_ * nFrames_, 3 * nFrames_);
		printSplitLine();
		show(dSizeMatrix_, nFrames_ * nFrames_, nFrames_);
		printSplitLine();*/

		// 获取粒子的速度， 加速度
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
		// 此处给dPoints_, dColors_, dSizes_, dGroupStarts_赋值
		particleSystemToPoints(dPoints_, dColors_, dSizes_, dGroupStarts_,
			dStartFrames_, nParticleGroups_, dDirections_, dSpeeds_, 
			dStartPoses_, currFrame, nFrames_, dColorMatrix_, dSizeMatrix_);

		// 观察particleSystemToPoints操作 对于相同帧的输出结果是否一致
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
		// 观察compress操作 对于相同帧的输出结果是否一致
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
			// 观察interpolation之前输入是否一致
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
			// 观察interpolation操作 对于相同帧的输出结果是否一致
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
			// 观察finalPosition操作 对于相同帧的输出结果是否一致
			/*{
				size_t resultSize;
				CUDACHECK(cudaMemcpy(&resultSize, dGroupOffsets_ + realNGroups_,
					sizeof(size_t), cudaMemcpyDeviceToHost));
				FW_ASSERT(comp.compare("pointsAfterInterpolation",
					currFrame, dPoints_, 3 * resultSize));
			}*/
			CUDACHECK(cudaDeviceSynchronize());
			// 映射buffer的内存指针
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

			// 释放对buffer的内存指针映射
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