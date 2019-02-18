#pragma once
#include "OpenGLWidget.h"
#include "firework.h"

class OpenGLWindow : public OpenGLWidget {
private:
	FwBase& fw_;
	struct StartPoint {
		bool valid;
		LONG x;
		LONG y;
		StartPoint() : valid(false), x(0), y(0) {}
		StartPoint(LONG x, LONG y) : valid(true), x(x), y(y) {}
		void reset() {
			valid = false;
		}
		void set(LONG x, LONG y) {
			valid = true;
			this->x = x;
			this->y = y;
		}

		void set(CPoint p) {
			set(p.x, p.y);
		}
	} startPoint_;
public:
	OpenGLWindow(FwBase& fw) : fw_(fw) {}
	~OpenGLWindow() = default;

	/* ============================================
	 * 定义Opengl窗口中的鼠标移动事件
	 * 用于调整Opengl窗口的视角
	 * ============================================
	 */
	afx_msg void OnLButtonDown(UINT nFlags, CPoint point) override {
		startPoint_.set(point);
		CWnd::OnLButtonDown(nFlags, point);
	};

	afx_msg void OnLButtonUp(UINT nFlags, CPoint point) override {
		if (startPoint_.valid) {
			LONG dx = point.x - startPoint_.x;
			LONG dy = point.y - startPoint_.y;
			printf("%d, %d\n", dx, dy);
		}
		startPoint_.reset();
		CWnd::OnLButtonDown(nFlags, point);
	};

	// TODO: 遗留问题，此事件并不响应，
	// 导致鼠标从内移出，再从外移入仍然响应OnLButtonUp事件
	afx_msg void OnMouseLeave() override {
		printf("leave\n");
		startPoint_.reset();
		CWnd::OnMouseLeave();
	};

public:
	bool Initialize() override {

		return true;
	}

	void RenderScene() override {
		glClearColor(1, 0.2, 1.0, 1.0);
		glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
		fw_.RenderScene();
	}
};