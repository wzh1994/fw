// OpenGLWidget.cpp : 实现文件
//

#include "stdafx.h"
#include "OpenGLWidget.h"


// OpenGLWidget

IMPLEMENT_DYNAMIC(OpenGLWidget, CWnd)

OpenGLWidget::OpenGLWidget() {}

OpenGLWidget::~OpenGLWidget() {}

BEGIN_MESSAGE_MAP(OpenGLWidget, CWnd)
	//{{AFX_MSG_MAP(COpenGL)
	ON_WM_CREATE()
	ON_WM_PAINT()
	ON_WM_MOUSELEAVE()
	ON_WM_LBUTTONDOWN()
	ON_WM_LBUTTONUP()
	//}}AFX_MSG_MAP
END_MESSAGE_MAP()

// OpenGLWidget 消息处理程序
int OpenGLWidget::OnCreate(LPCREATESTRUCT lpCreateStruct) {
	if (CWnd::OnCreate(lpCreateStruct) == -1)
		return -1;

	// TODO: Add your specialized creation code here

	{	//set pixel-format & initialize glew
		m_hDC = ::GetDC(m_hWnd);
		PIXELFORMATDESCRIPTOR pfd;
		memset(&pfd, 0, sizeof(PIXELFORMATDESCRIPTOR));
		pfd.nSize = sizeof(PIXELFORMATDESCRIPTOR);
		pfd.nVersion = 1;
		pfd.dwFlags = PFD_DOUBLEBUFFER | PFD_SUPPORT_OPENGL | PFD_DRAW_TO_WINDOW;
		pfd.iPixelType = PFD_TYPE_RGBA;
		pfd.cColorBits = 32;
		pfd.cDepthBits = 32;
		pfd.iLayerType = PFD_MAIN_PLANE;

		int nPixelFormat = ChoosePixelFormat(m_hDC, &pfd);

		if (nPixelFormat == 0) return -1;

		BOOL bResult = SetPixelFormat(m_hDC, nPixelFormat, &pfd);

		if (!bResult) return -1;

		//initialize glew
		HGLRC tempContext = wglCreateContext(m_hDC);
		wglMakeCurrent(m_hDC, tempContext);

		GLenum err = glewInit();
		if (GLEW_OK != err) {
			::MessageBox(NULL, _T("glew initialize failed"), NULL, MB_OK);
			return -1;
		}

		//Get a GL 4,2 context
		int attribs[] = {
			WGL_CONTEXT_MAJOR_VERSION_ARB, 4,
			WGL_CONTEXT_MINOR_VERSION_ARB, 2,
			WGL_CONTEXT_FLAGS_ARB, 0,
			0
		};

		if (wglewIsSupported("WGL_ARB_create_context") == 1) {
			m_hRC = wglCreateContextAttribsARB(m_hDC, 0, attribs);
			wglMakeCurrent(NULL, NULL);
			wglDeleteContext(tempContext);
			wglMakeCurrent(m_hDC, m_hRC);
			glEnable(GL_DEPTH_TEST);
		}
		else {    //It's not possible to make a GL 4.x context. Use the old style context (GL 2.1 and before)
			m_hRC = tempContext;
		}
		if (!m_hRC) return -1;
	}

	if (!Initialize()) return -1;
	return 0;
}

void OpenGLWidget::OnPaint() {
	CPaintDC dc(this); // device context for painting

	// TODO: Add your message handler code here
	if (wglMakeCurrent(m_hDC, m_hRC)) {
		//render here
		RenderScene();
		::glFlush();
	 	SwapBuffers(m_hDC);
	};
	// Do not call CWnd::OnPaint() for painting messages
}
