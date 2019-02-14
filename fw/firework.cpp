
// firework.cpp: ʵ���ļ� ʵ��firework.h�еĺ���
//


#include "stdafx.h"
#include "firework.h"
#include "normalfirework.h"



FwBase* getFirework(FireWorkType type, float* args) {
	switch (type) {
	case FireWorkType::Normal:
		return new NormalFirework(args);
		break;
	default:
		return nullptr;
	}
}