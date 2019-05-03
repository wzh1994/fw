#pragma once
#include "OpenGLWidget.h"
#include "firework.h"
#include "Camera.h"

class OpenGLWindow : public OpenGLWidget {
private:
	firework::FwBase& fw_;
	std::unique_ptr<Camera> camera_;
	size_t width_, height_;
	unsigned int rbo_;
	unsigned int framebuffer_;
	unsigned int intermediateFBO_;
	unsigned int screenTexture_;
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
	OpenGLWindow(firework::FwBase& fw, int width, int height)
		: fw_(fw)
		, camera_(new Camera(width, height))
		, width_(width)
		, height_(height){}

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
		using Dir = Camera::RotateDirection;
		if (startPoint_.valid) {
			LONG dx = point.x - startPoint_.x;
			LONG dy = point.y - startPoint_.y;
			if (abs(dx) > abs(dy))
				camera_->rotate<Dir::Horizontal>(static_cast<float>(-dx / 10));
			else
				camera_->rotate<Dir::Vertical>(static_cast<float>(-dy / 10));
			Invalidate();
		}
		startPoint_.reset();
		CWnd::OnLButtonDown(nFlags, point);
	};

	// TODO: 遗留问题，此事件并不响应，
	// 导致鼠标从内移出，再从外移入仍然响应OnLButtonUp事件
	afx_msg void OnMouseLeave() override {
		startPoint_.reset();
		CWnd::OnMouseLeave();
	};

public:
	bool Initialize() override {
		// configure MSAA framebuffer
		// --------------------------
		glGenFramebuffers(1, &framebuffer_);
		glBindFramebuffer(GL_FRAMEBUFFER, framebuffer_);
		// create a multisampled color attachment texture
		unsigned int textureColorBufferMultiSampled;
		glGenTextures(1, &textureColorBufferMultiSampled);
		glBindTexture(GL_TEXTURE_2D_MULTISAMPLE, textureColorBufferMultiSampled);
		glTexImage2DMultisample(GL_TEXTURE_2D_MULTISAMPLE, 4, GL_RGB, width_, height_, GL_TRUE);
		glBindTexture(GL_TEXTURE_2D_MULTISAMPLE, 0);
		glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D_MULTISAMPLE, textureColorBufferMultiSampled, 0);
		// create a (also multisampled) renderbuffer object for depth and stencil attachments
		
		glGenRenderbuffers(1, &rbo_);
		glBindRenderbuffer(GL_RENDERBUFFER, rbo_);
		glRenderbufferStorageMultisample(GL_RENDERBUFFER, 4, GL_DEPTH24_STENCIL8, width_, height_);
		glBindRenderbuffer(GL_RENDERBUFFER, 0);
		glFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_DEPTH_STENCIL_ATTACHMENT, GL_RENDERBUFFER, rbo_);

		if (glCheckFramebufferStatus(GL_FRAMEBUFFER) != GL_FRAMEBUFFER_COMPLETE)
			cout << "ERROR::FRAMEBUFFER:: Framebuffer is not complete!" << endl;
		glBindFramebuffer(GL_FRAMEBUFFER, 0);

		// configure second post-processing framebuffer
		
		glGenFramebuffers(1, &intermediateFBO_);
		glBindFramebuffer(GL_FRAMEBUFFER, intermediateFBO_);
		// create a color attachment texture
		
		glGenTextures(1, &screenTexture_);
		glBindTexture(GL_TEXTURE_2D, screenTexture_);
		glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, width_, height_, 0, GL_RGB, GL_UNSIGNED_BYTE, NULL);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
		glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, screenTexture_, 0);	// we only need a color buffer

		if (glCheckFramebufferStatus(GL_FRAMEBUFFER) != GL_FRAMEBUFFER_COMPLETE)
			cout << "ERROR::FRAMEBUFFER:: Intermediate framebuffer is not complete!" << endl;
		glBindFramebuffer(GL_FRAMEBUFFER, 0);
		return true;
	}

	void RenderScene() override {
		glClearColor(0, 0, 0, 1.0);
		glEnable(GL_MULTISAMPLE);
		glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
		fw_.RenderScene(*camera_);
	}

	~OpenGLWindow() {
		glDeleteFramebuffers(1, &framebuffer_);
		glDeleteRenderbuffers(1, &rbo_);
		glDeleteFramebuffers(1, &intermediateFBO_);
		glDeleteTextures(1, &screenTexture_);
	}
};