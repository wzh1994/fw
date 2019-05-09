#pragma once
#include <vector>
#include <glm.hpp>
#include <memory>
#include "exceptions.h"
#include "Camera.h" 
#include "fwrule.h"

namespace firework{

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

private:
	bool isInited_;

protected:
	std::vector<Attr> attrs_;
	float* args_;
	// nFrames_这个名字被BeginGroup宏所使用，修改的时候需要注意一下
	size_t nFrames_;

protected:
	FwBase(float* args) :args_(args), isInited_(false){}
public:

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

	virtual void initialize() {
		FW_ASSERT(!isInited_) << "Cannot init fw class more than once!";
		isInited_ = true;
	};
	virtual void GetParticles(int currFrame) = 0;
	virtual void RenderScene(const Camera& camera) = 0;
	virtual void prepare() = 0;

	virtual ~FwBase() = default;
};

enum class FireWorkType {
	Normal = 0,
	MultiExplosion = 1,
	Strafe = 2,
	CircleFirework = 3,
	TwinkleFirework = 4,
	DualMixture = 5,
	TriplicateMixture = 6
};


// 构造所有FireWork类的唯一入口函数
__declspec(dllexport)
FwBase* getFirework(FireWorkType, float*,
	bool initAttr = true, size_t bufferSize = 200000000);

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