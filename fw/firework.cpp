
// firework.cpp: ʵ���ļ� ʵ��firework.h�еĺ���
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