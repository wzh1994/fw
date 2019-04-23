
// firework.cpp: 实现文件 实现firework.h中的函数
//


#include "stdafx.h"
#include "firework.h"
#include "normalfirework.h"

size_t FwBase::Attr::idx = 0;
size_t FwBase::Attr::groupOffset = 0;
size_t FwBase::Attr::groupNums = 0;
size_t FwBase::Attr::groupStep = 0;
size_t FwBase::Attr::nFrames = 0;

FwBase* getFirework(FireWorkType type, float* args) {
	switch (type) {
	case FireWorkType::Normal:
		return new NormalFirework(args);
		break;
	default:
		return nullptr;
	}
}