
// firework.cpp: 实现文件 实现firework.h中的函数
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