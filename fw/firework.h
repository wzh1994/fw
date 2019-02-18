#pragma once
#include <vector>
#include <glm.hpp>
#include <memory>
#include "Shader.h"

class CfwDlg;
enum class ArgType {
	Value = 0,
	Color
};
class FwBase {
	friend class CfwDlg;
protected:
	struct Attr {
		std::wstring name;
		ArgType type;
		float value;
		glm::vec3 color;
		Attr(std::wstring name, float value)
			: name(name)
			, type(ArgType::Value)
			, value(value) {}
		Attr(std::wstring name, glm::vec3 color)
			: name(name)
			, type(ArgType::Color)
			, color(color) {}
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
	virtual void RenderParticles() = 0;

public:
	void RenderScene() {
		RenderParticles();
	}

	virtual ~FwBase() = default;
};

enum class FireWorkType {
	Normal = 0
};

// ��������FireWork���Ψһ��ں���
FwBase* getFirework(FireWorkType type, float* args);