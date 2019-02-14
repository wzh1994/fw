#pragma once
#include "firework.h"


// useful definations
#define AddValue(_name, _value) \
	attrs_.push_back(Attr(L##_name, _value));
#define AddColor(_name, _r, _g, _b) \
	attrs_.push_back(Attr(L##_name, glm::vec3(_r, _g, _b)));

class NormalFirework : public FwBase {
	friend FwBase* getFirework(FireWorkType type, float* args);
private:
	NormalFirework(float* args) {
		AddValue("³õËÙ¶È", 0.1f);
		AddColor("ÑÕÉ«", 1.0f, 0.2f, 0.5f);
	}

	void GetParticles() override {

	}

	void RenderParticles() override {

	}

};