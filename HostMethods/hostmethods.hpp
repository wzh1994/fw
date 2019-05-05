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
 * ����hosy������Դ棬��ALIGN����
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

	size_t normalFireworkDirections(float* directions,
		size_t nIntersectingSurfaceParticle,
		float xRate=0, float yRate=0, float zRate=0,
		float xStretch=0, float yStretch=0, float zStretch=0);

}

#endif  // FW_HOSTMETHODS_HOSTMETHODS_HPP