#ifndef FW_KERNEL_KERNELS_HPP
#define FW_KERNEL_KERNELS_HPP

#include "kernel.h"
#include "cuda_runtime.h"

/*
 * ����Ŀ����.lib�ļ���fw����cuda��صĺ�����
 * ���ļ�����������kernel�����Ľӿ�
 * ʹ�ñ���Ŀ��kernel��������Ҫ���뱾ͷ�ļ�
 * �����ļ��еĺ�����Ϊ������Ķ���ӿ�
 */

namespace cudaKernel {

static const float kFrameTime = 0.08333333333f;

/*
 * ����cuda������Դ棬��ALIGN����
 */
template <class T>
cudaError_t cudaMallocAlign(T** ptr, size_t size) {
	size = ceilAlign(size) * kernelAlign;
	return cudaMalloc(ptr, size);
};

/* ==================================
 * ����ͨ����ѧ����
 * ==================================
 */

/* 
 * ������
 */
void fill(float* dArray, float data, size_t size);
void fill(size_t* dArray, size_t data, size_t size);
void fill(float* dArray, const float* data, size_t size, size_t step);
void fill(size_t* dArray, const size_t* data, size_t size, size_t step);

/*
 * ɨ�����
 */
// �ۼӺͲ��� ÿ��������������������kMmaxBlockDim * kMmaxBlockDim
void cuSum(float* out, const float* in, size_t size, size_t numGroup = 1);
void cuSum(size_t* out, const size_t* in, size_t size, size_t numGroup = 1);

// �ۼ� ���ֵ����
void cuMax(float* out, float* in, size_t size, size_t numGroup = 1);

// �ۼ� �˻�����
void cuMul(float* out, float* in, size_t size, size_t numGroup = 1);

/*
 * ����
 */
void normalize(float* vectors, size_t size);

/*
 * ����
 */
void scale(float* dArray, float rate, size_t size);

/*
 * ��Լ
 */
enum class ReduceOption {
	sum,
	min
};
// ��Լ��ֱ�ӵ���Ĭ�����
void reduce(size_t *dMatrix, size_t* dResult,
	size_t nGroups, size_t size, ReduceOption op = ReduceOption::sum);
void reduce(float *dMatrix, float* dResult,
	size_t nGroups, size_t size, ReduceOption op = ReduceOption::sum);
void reduce2(float *dMatrix, float* dResult,
	size_t nGroups, size_t size, ReduceOption op = ReduceOption::sum);
// ����Сֵ
void reduceMin(float *dMatrix, float* dResult, size_t nGroups, size_t size);
void reduceMin(size_t *dMatrix, size_t* dResult, size_t nGroups, size_t size);

/*
 * �������ÿһ�е�һ����0Ԫ�ص�λ��
 */
void argFirstNoneZero(size_t* dMatrix, size_t* result,
	size_t nGroups, size_t size);

/* ==================================
 * ͨ���̻�������ط���
 * ==================================
 */

 /*
  * ����ĳһ������ÿһʱ�̵����������һʱ�����ɵ������ڵ��ڸ��������µ�λ��
  */
// Ҫ�� size * nInterpolation ���ܳ���kMmaxBlockDim
void calcShiftingByOutsideForce(
	float* dIn,
	size_t size, // ������������ÿһ֡��Ӧһ����������˵�ͬ��֡��
	size_t nInterpolation, // ��ֵ������
	float time = kFrameTime // ��֡�����ʱ��
);

/*
 * ��ȡ��ɫ�ͳߴ�仯�ľ���
 */
// Ҫ��nFrames���ܳ���kMmaxBlockDim
void getColorAndSizeMatrix(
	const float* startColors, // ���� ��ʼ��ɫ
	const float* startSizes, // ���� ��ʼ�ߴ�
	size_t nFrames, // �ܼ�֡��
	float colorDecay, // ��ɫ˥����
	float sizeDecay, // �ߴ�˥����
	float* dColorMatrix, // �������ɫ��֡���仯����
	float* dSizeMatrix // ������ߴ���֡���仯����
);

void getColorAndSizeMatrixDevInput(
	const float* startColors, // ���� ��ʼ��ɫ
	const float* startSizes, // ���� ��ʼ�ߴ�
	size_t nFrames, // �ܼ�֡��
	float colorDecay, // ��ɫ˥����
	float sizeDecay, // �ߴ�˥����
	float* dColorMatrix, // �������ɫ��֡���仯����
	float* dSizeMatrix // ������ߴ���֡���仯����
);

/*
 * ��N������ϵͳ����N��㣬���ÿ�����Ӧ��λ�ã���ɫ���ߴ�
 */ 
// Ҫ��nFrames���ܴ���kMmaxBlockDim

// �˷����������Ŷ༶��ը���̻�����
void particleSystemToPoints(float* dPoints, float* dColors, float* dSizes,
	size_t* dGroupStarts, const size_t* dStartFrames, const size_t* dLifeTime,
	size_t nGroups, const float* dDirections, const float* dCentrifugalPos,
	const float* dStartPoses, size_t currFrame, size_t nFrames,
	const size_t* dColorAndSizeStarts, const float* dColorMatrix,
	const float* dSizeMatrix, float time = kFrameTime);

// �˷�������û�ж༶��ը���̻����棬��ʱdColorAndSizeStartsĬ��Ϊ0
void particleSystemToPoints(
	float* dPoints, // ��� ��ʼλ�� 
	float* dColors, // ��� ��ʼ��ɫ
	float* dSizes, // ��� ��ʼ�ߴ�
	size_t* dGroupStarts, // ��� �˷�����dStartFrames��������
	const size_t* dStartFrames, // ÿһ�����ӵ���ʼ֡
	const size_t* dLifeTime, // ÿһ�����ӵ�����
	size_t nGroups, // �ܼƵ���������
	const float* dDirections, // ÿһ�����ӵķ���
	const float* dCentrifugalPos, // ÿһ�����ӵĳ�ʼ�ٶ�
	const float* dStartPoses, // ÿһ֡��������ʱ�������λ��
	size_t currFrame, // ��ǰ֡��
	size_t nFrames, // ��֡��
	const float* dColorMatrix, // ��ɫ��֡���仯�ľ���
	const float* dSizeMatrix, // �ߴ���֡���仯�ľ���
	float time = kFrameTime // ��֡�����ʱ��
);

/*
 * ���̻������ӽ��пռ�ѹ������ȥ���еĲ��ɼ�����
 */
// Ҫ��nGroups��size���ܳ���kMmaxBlockDim
// ����ֵ��ѹ����ʵ����Ч����������
size_t compress(
	float* dPoints, // ����&��� ���ӵ�λ��
	float* dColors, // ����&��� ���ӵ���ɫ
	float* dSizes, // ����&��� ���ӵĳߴ�
	size_t nGroups, // ��������
	size_t size, // ÿ�����ӵĸ������˷��������ÿ������������ͬ
	size_t* dGroupOffsets, // ��� ѹ����ÿ������λ���������ʼλ�õ�ƫ��
	size_t* dGroupStarts // ��� ѹ�����ÿ�����ӵ���ʼ֡
);

/*
 * ��ֵ�㷨 ��N����ͬ���ȵ���������ֵ
 */
// Ҫ��ÿ���ֵ�Ľ�����Ȳ��ܳ���kMmaxBlockDim
// ���� ��49֡�����ӣ� ÿ���ֵ15���� ����48 * 15 + 49������

// �˷������ڲ�ֵN�飬ÿ���������һ�µ����
void interpolation(
	float* dArray, size_t nGroups, size_t size, size_t nInterpolation);

// �˷������ڶԵ��λ�ã���ɫ���ߴ���в�ֵ��ÿ�����������һ��
void interpolation(
	float* dPoints, // ����&��� ���ӵ�λ��
	float* dColors, // ����&��� ���ӵ���ɫ
	float* dSizes, // ����&��� ���ӵĳߴ�
	size_t* dGroupOffsets, // ����&��� ��ֵ��ÿ������λ�õ�ƫ��
	size_t nGroups, // ��������
	size_t maxSize, // ��ֵ֮ǰ��ÿ���������ĸ�����һ��Ϊ֡��
	size_t nInterpolation // ÿ��������֮��������������
);

/*
 * ��������ÿ�����λ��
 */
// Ҫ�� ÿ�����ӵ�����������ܳ���kMmaxBlockDim
void calcFinalPosition(
	float* dPoints, // ����&��� ���ӵ�λ��
	size_t nGroups, // ��������
	size_t maxSize, // ÿ���������ĸ�����һ��Ϊ֡����nInterpolation+1�ĳ˻�
	size_t nInterpolation, // ÿ��������֮��������������
	size_t frame, // ��ǰ֡
	const size_t* dGroupOffsets, // ÿ������λ���������ʼλ�õ�ƫ��
	const size_t* dGroupStarts, // ÿ������ѹ�������ʼ֡
	const size_t* dStartFrames, // ÿ������ѹ��֮ǰ����ʼ֡
	const float* dXShiftMatrix, // ��ɫ��֡���仯�ľ���
	const float* dYShiftMatrix, // �ߴ���֡���仯�ľ���
	size_t shiftsize // shiftMatrixÿһ�еĳߴ�
);

/*
 * �ѵ������ߣ�������һ�����������������Ƭ
 */
// ����Ҫ�� ÿ�����ӵ�����������ܳ���kMmaxBlockDim�� ÿ�����ӵĵ�Ϊ������
// Ӧ�ó����� ���ɵ�ÿ������֮�����������15�����㣬�Ա�֤����Ϊ����
// ����ֵ�� indices�ĳߴ磬��ӦҪ���Ƶĵ������
size_t pointToLine(
	const float* dPointsIn, // ���յ�����λ��
	const float* dSizesIn, // ���յ����ӳߴ�
	const float* dColorsIn, // ���յ�������ɫ
	size_t maxSizePerGroup, // ÿһ�����ӵ����ߴ�
	size_t* const dGroupOffsets, // ÿ������λ���������ʼλ�õ�ƫ��
	size_t nGroups, // ��������
	float* dBuffer, // �������ݻ��� vbo
	uint32_t* dIndicesOut, // �������л��� ebo
	float outterAlpha = 0.5, // ��Ȧ�Ĳ�͸����
	float innerSize = 0.25, // ��Ȧ�ĳߴ�
	float innerColor = 0.8
	/*,float innerColorRate = 1*/
);

/* ==================================
 * NormalFirework��ط���
 * ==================================
 */

// ��ȡnormalFirework�����̻��ĳ�ʼ���򣬷����䷽��������������ӵ�����
size_t normalFireworkDirections(
	float* dDirections, // �������ȡ��ÿ�����ӵķ���
	size_t nIntersectingSurfaceParticle, // �������������
	float xRate = 0.1, float yRate = 0.1, float zRate = 0.1,
	float xStretch = 1, float yStretch = 1, float zStretch = 1
);

/* ==================================
 * StrafeFirework��ط���
 * ==================================
 */

 // ��ȡstrafeFirework�����̻��ĳ�ʼ���򣬷����䷽��������������ӵ�����
size_t strafeFireworkDirections(
	float* dDirections, size_t nGroups, size_t size);

/* ==================================
 * MultiExplosionFirework��ط���
 * ==================================
 */
// ��ȡ���α�ը�����ӵĳ�λ��
void getSubFireworkPositions(
	float* dStartPoses, float* dDirections, size_t nDirs,
	size_t nSubGroups, const float* dCentrifugalPos_, size_t startFrame,
	size_t kShift, const float* dShiftX_, const float* dShiftY_);
}// end namespace cudaKernel
#endif