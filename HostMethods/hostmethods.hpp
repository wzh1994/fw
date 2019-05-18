#ifndef FW_HOSTMETHODS_HOSTMETHODS_HPP
#define FW_HOSTMETHODS_HOSTMETHODS_HPP

#include "hostmethod.h"

/*
 * ����Ŀ����.lib�ļ���fw����host��صĺ�����
 * ���ļ�����������host�����Ľӿ�
 * ʹ�ñ���Ŀ��host��������Ҫ���뱾ͷ�ļ�
 * �����ļ��еĺ�����Ϊ������Ķ���ӿ�
 */


namespace hostMethod {

	static const float kFrameTime = 0.08333333333f;
/*
 * ����host������Դ棬��ALIGN����
 * �˴���Ϊcuda��Ϊ�˷�������߲��Ĵ���������ֱ��ʹ�øú�����
 */
	template <class T>
	void cudaMallocAlign(T** ptr, size_t size) {
		size = ceilAlign(size) * kernelAlign / sizeof(T);
		*ptr = new T[size];
	}

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
	void cuSum(float* dOut, const float* dIn, size_t size, size_t numGroup = 1);
	void cuSum(size_t* dOut, const size_t* dIn, size_t size, size_t numGroup = 1);

	// �ۼ� ���ֵ����
	void cuMax(float* dOut, float* dIn, size_t size, size_t numGroup = 1);

	// �ۼ� �˻�����
	void cuMul(float* dOut, float* dIn, size_t size, size_t numGroup = 1);

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

	// ����Сֵ
	void reduceMin(float *dMatrix, float* dResult, size_t nGroups, size_t size);
	void reduceMin(size_t *dMatrix, size_t* dResult, size_t nGroups, size_t size);

/* ==================================
 * ͨ���̻�������ط���
 * ==================================
 */

	void getColorAndSizeMatrix(
		const float* startColors, // ���� ��ʼ��ɫ
		const float* startSizes, // ���� ��ʼ�ߴ�
		size_t nFrames, // �ܼ�֡��
		float colorDecay, // ��ɫ˥����
		float sizeDecay, // �ߴ�˥����
		float* dColorMatrix, // �������ɫ��֡���仯����
		float* dSizeMatrix // ������ߴ���֡���仯����
	);

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
		const size_t* dColorAndSizeStarts_,
		const float* dColorMatrix, // ��ɫ��֡���仯�ľ���
		const float* dSizeMatrix, // �ߴ���֡���仯�ľ���
		float time = kFrameTime // ��֡�����ʱ��
	);

	void argFirstNoneZero(size_t* matrix, size_t* result,
		size_t nGroups, size_t size);

	size_t compress(
		float* dPoints, // ����&��� ���ӵ�λ��
		float* dColors, // ����&��� ���ӵ���ɫ
		float* dSizes, // ����&��� ���ӵĳߴ�
		size_t nGroups, // ��������
		size_t size, // ÿ�����ӵĸ������˷��������ÿ������������ͬ
		size_t* dGroupOffsets, // ��� ѹ����ÿ������λ���������ʼλ�õ�ƫ��
		size_t* dGroupStarts, // ��� ѹ�����ÿ�����ӵ���ʼ֡
		float rate = 1.0,
		curandState* devStates = nullptr
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

	void calcFinalPosition(float* dPoints, size_t nGroups, size_t maxSize,
		size_t nInterpolation, size_t frame, const size_t* dGroupOffsets,
		const size_t* dGroupStarts, const size_t* dStartFrames,
		const float* dXShiftMatrix, const float* dYShiftMatrix, size_t shiftsize);

	size_t normalFireworkDirections(float* directions,
		size_t nIntersectingSurfaceParticle,
		float xRate=0, float yRate=0, float zRate=0,
		float xStretch=0, float yStretch=0, float zStretch=0);

	size_t pointToLine(
		const float* dPointsIn,
		const float* dSizesIn,
		const float* dColorsIn,
		size_t maxSizePerGroup,
		size_t* const dGroupOffsets,
		size_t nGroups,
		float* dBuffer,
		uint32_t* dIndicesOut,
		float outterAlpha,
		float innerSize,
		float innerColor
	);

	/* ==================================
	 * CircleFirework��ط���
	 * ==================================
	 */

	// ��ȡnormalFirework�����̻��ĳ�ʼ���򣬷����䷽�������
	size_t circleFireworkDirections(
		float* dDirections, size_t nIntersectingSurfaceParticle,
		float* norm, float angleFromNormal = 0,
		float xRate = 0.1, float yRate = 0.1, float zRate = 0.1,
		float xStretch = 1, float yStretch = 1, float zStretch = 1);

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
	void getSubFireworkPositions(float* dStartPoses,
		float* dDirections, const float* dSubDirs, size_t nDirs, size_t nSubDirs,
		size_t nSubGroups, const float* dCentrifugalPos_, size_t startFrame,
		size_t kShift, const float* dShiftX_, const float* dShiftY_);

	inline void initRand(curandState *dev_states, size_t size) {};

}

#endif  // FW_HOSTMETHODS_HOSTMETHODS_HPP