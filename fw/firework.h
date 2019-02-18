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
	 * 子类需实现以下几个方法
	 * ==========================================
	 */

	// 与opengl有关的初始化放在这里，以保证其在glewInit之后执行
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

// 构造所有FireWork类的唯一入口函数
FwBase* getFirework(FireWorkType type, float* args);