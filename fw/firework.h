#pragma once
#include <vector>
#include <glm.hpp>
#include <memory>
#include "Shader.h"
#include "Camera.h"

class CfwDlg;
enum class ArgType {
	Scalar = 0,
	Color,
	Vector3

};
class FwBase {
	friend class CfwDlg;
protected:
	struct Attr {
		ArgType type;
		std::wstring name;
		glm::vec3 value;
		Attr(std::wstring name, float value)
			: type(ArgType::Scalar)
			, name(name)
			, value(value, 0, 0) {}
		Attr(ArgType type, std::wstring name, glm::vec3 value)
			: type(type)
			, name(name)
			, value(value) {}
	};

protected:
	std::vector<Attr> attrs_;
	FwBase() = default;

protected:
	std::unique_ptr<Shader> shader_;

private:
	/* ==========================================
	 * ������ʵ�����¼�������
	 * ==========================================
	 */

	// ��opengl�йصĳ�ʼ����������Ա�֤����glewInit֮��ִ��
	virtual void initialize() {
		shader_.reset(new Shader("fw.vs", "fw.fs"));
		GetParticles();
	}

	virtual void GetParticles() = 0;
	virtual void RenderParticles(const Camera& camera) = 0;

public:
	void RenderScene(const Camera& camera) {
		RenderParticles(camera);
	}

	virtual ~FwBase() = default;
};

enum class FireWorkType {
	Normal = 0
};

// ��������FireWork���Ψһ��ں���
FwBase* getFirework(FireWorkType type, float* args);

// useful definations used in sub classes
#define AddValue(_name, _value) \
	attrs_.push_back(Attr(L##_name, _value));

#define AddColor(_name, _r, _g, _b) \
	attrs_.push_back(Attr(ArgType::Color, L##_name, glm::vec3(_r, _g, _b)));

#define AddVec3(_name, _x, _y, _z) \
	attrs_.push_back(Attr(ArgType::Vector3, L##_name, glm::vec3(_x, _y, _z)));