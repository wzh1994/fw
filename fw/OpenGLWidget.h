#pragma once
#include <GL/glew.h>
#include <GL/wglew.h>
#include "stdafx.h"
// OpenGLWidget

class OpenGLWidget : public CWnd {
	DECLARE_DYNAMIC(OpenGLWidget)

	HDC m_hDC; HGLRC m_hRC;
public:
	OpenGLWidget();
	virtual ~OpenGLWidget();
	virtual bool Initialize() = 0;
	virtual void RenderScene() = 0;
protected:
	//{{AFX_MSG(COpenGL)
	afx_msg int OnCreate(LPCREATESTRUCT lpCreateStruct);
	afx_msg void OnPaint();
	//}}AFX_MSG
	DECLARE_MESSAGE_MAP()
};
