
// fwDlg.cpp: 实现文件
//

#include "stdafx.h"
#ifndef __AFXWIN_H__
#error "在包含此文件之前包含“stdafx.h”以生成 PCH 文件"
#endif

#include "resource.h"		// 主符号
#include "fwDlg.h"
#include "afxdialogex.h"

#ifdef _DEBUG
#define new DEBUG_NEW
#endif

namespace {
	const char opencvWindow[] = "IDC_OPENCV_OUTPUT";
}

// CfwDlg 对话框

extern "C" __declspec(dllexport) void ShowDialog()
{
	AFX_MANAGE_STATE(AfxGetStaticModuleState());
	CfwDlg dlg;
	dlg.DoModal();
}

FwBase* getFirework(FireWorkType type, float* args);
CfwDlg::CfwDlg(CWnd* pParent /*=nullptr*/)
	: CDialogEx(IDD_FW_DIALOG, pParent)
	, m_edit_value(0)
{
	m_hIcon = AfxGetApp()->LoadIcon(IDR_MAINFRAME);
}

/* ==================================================
 * 初始化，包括opencv窗口，opengl窗口，和一些控件位置
 * ==================================================
 */

void CfwDlg::setSliderPos(int pos) {
	m_sliderc.SetPos(pos);//当前停留的位置
	onSliderChange();
}

void CfwDlg::myInitialize() {
	/*------- opengl --------*/
	MoveWindow(100, 100, 800, 800);
	pOpenGLWindow = new OpenGLWindow();
	pOpenGLWindow->Create(
		NULL, NULL, WS_CHILD | WS_CLIPSIBLINGS | WS_CLIPCHILDREN | WS_VISIBLE,
		CRect(50, 50, 349, 349),
		this,   //this is the parent
		0);

	/*------- opencv --------*/
	//CWnd是MFC窗口类的基类,提供了微软基础类库中所有窗口类的基本功能。
	CWnd  *pWnd = GetDlgItem(IDC_PIC);
	pWnd->SetWindowPos(NULL, 450, 50, 300, 300, SWP_NOZORDER);
	cvNamedWindow(opencvWindow, 0);//设置窗口名
	cvResizeWindow(opencvWindow, 300, 300);
	//hWnd 表示窗口句柄,获取窗口句柄
	HWND hWnd = (HWND)cvGetWindowHandle("IDC_OPENCV_OUTPUT");
	//GetParent函数一个指定子窗口的父窗口句柄
	HWND hParent = ::GetParent(hWnd);
	::SetParent(hWnd, pWnd->m_hWnd);
	//ShowWindow指定窗口中显示
	::ShowWindow(hParent, SW_HIDE);

	/*------- firework -----*/
	// 放在这里，已保证FireWork实例化的时候，glew已经初始化完成
	float* args = nullptr;
	fw.reset(getFirework(FireWorkType::Normal, args));

	/*------- combo --------*/
	for (auto it = fw->attrs_.begin(); it != fw->attrs_.end(); ++it) {
		m_combo.AddString(it->name.c_str());
	}

	/*------- slider --------*/
	// 因为setSliderPos会触发onSliderChange， 而该函数需要使用fw变量
	m_sliderc.SetRange(0, 20);//设置范围
	m_sliderc.SetTicFreq(2);//设置显示刻度的间隔
	setSliderPos(5);
	m_sliderc.SetLineSize(10);//一行的大小，对应键盘的方向键
}

/* ========================================
 * 系统生成的映射函数
 * ========================================
 */
 
void CfwDlg::DoDataExchange(CDataExchange* pDX)
{
	CDialogEx::DoDataExchange(pDX);
	DDX_Control(pDX, IDC_SLIDER1, m_sliderc);
	DDX_Control(pDX, IDC_BUTTON1, m_pic);
	DDX_Control(pDX, IDC_COMBO1, m_combo);
	DDX_Control(pDX, IDC_EDIT1, m_edit);
	DDX_Text(pDX, IDC_EDIT1, m_edit_value);
	DDX_Control(pDX, IDC_BUTTON2, m_bn_reset);
	DDX_Control(pDX, IDC_BUTTON3, m_bn_conform);
	DDX_Control(pDX, IDC_BUTTON5, m_bn_color);
}

BEGIN_MESSAGE_MAP(CfwDlg, CDialogEx)
	ON_WM_PAINT()
	ON_WM_QUERYDRAGICON()
	ON_WM_HSCROLL()
	ON_BN_CLICKED(IDC_BUTTON1, &CfwDlg::OnBnClickedButtonMinus)
	ON_BN_CLICKED(IDC_BUTTON4, &CfwDlg::OnBnClickedButtonPlus)
	ON_CBN_SELCHANGE(IDC_COMBO1, &CfwDlg::OnArgComboChange)
	ON_BN_CLICKED(IDC_BUTTON2, &CfwDlg::resetArgValue)
	ON_BN_CLICKED(IDC_BUTTON3, &CfwDlg::OnBnClickedConform)
	ON_BN_CLICKED(IDC_BUTTON5, &CfwDlg::OnBnClickedColorBtn)
END_MESSAGE_MAP()

// CfwDlg 消息处理程序

BOOL CfwDlg::OnInitDialog()
{
	CDialogEx::OnInitDialog();

	// 设置此对话框的图标。  当应用程序主窗口不是对话框时，框架将自动
	//  执行此操作
	SetIcon(m_hIcon, TRUE);			// 设置大图标
	SetIcon(m_hIcon, FALSE);		// 设置小图标

	// TODO: 在此添加额外的初始化代码
	myInitialize();

	return TRUE;  // 除非将焦点设置到控件，否则返回 TRUE
}

// 如果向对话框添加最小化按钮，则需要下面的代码
//  来绘制该图标。  对于使用文档/视图模型的 MFC 应用程序，
//  这将由框架自动完成。

void CfwDlg::OnPaint()
{
	if (IsIconic())
	{
		CPaintDC dc(this); // 用于绘制的设备上下文

		SendMessage(WM_ICONERASEBKGND, reinterpret_cast<WPARAM>(dc.GetSafeHdc()), 0);

		// 使图标在工作区矩形中居中
		int cxIcon = GetSystemMetrics(SM_CXICON);
		int cyIcon = GetSystemMetrics(SM_CYICON);
		CRect rect;
		GetClientRect(&rect);
		int x = (rect.Width() - cxIcon + 1) / 2;
		int y = (rect.Height() - cyIcon + 1) / 2;

		// 绘制图标
		dc.DrawIcon(x, y, m_hIcon);
	}
	else
	{
		CDialogEx::OnPaint();
	}
}

//当用户拖动最小化窗口时系统调用此函数取得光标
//显示。
HCURSOR CfwDlg::OnQueryDragIcon()
{
	return static_cast<HCURSOR>(m_hIcon);
}

/* ========================================
 * 自定义鼠标移动响应事件
 * ========================================
 */
void CfwDlg::OnHScroll(UINT nSBCode, UINT nPos, CScrollBar* pScrollBar)
{
	CSliderCtrl* pSlider = reinterpret_cast<CSliderCtrl*>(pScrollBar);

	// Check which slider sent the notification  
	if (pSlider == &m_sliderc)
	{
		// Check what happened  
		switch (nSBCode)
		{
		case TB_ENDTRACK:
			onSliderChange();
			break;
		default:
			break;
		}	
	}
}

/* ========================================
 * 自定义控件响应事件
 * ========================================
 */

void CfwDlg::onSliderChange() {
	int pos = m_sliderc.GetPos();
	cv::Mat photo(300, 300, CV_8UC3, cv::Scalar(pos * 10, 0, 255));
	cv::imshow(opencvWindow, photo);
	fw->GetParticles();
}

void CfwDlg::OnBnClickedButtonMinus(){
	int pos = m_sliderc.GetPos();
	int mx = m_sliderc.GetRangeMax();
	pos += 1;
	if (pos > mx) return;
	setSliderPos(pos);
}

void CfwDlg::OnBnClickedButtonPlus(){
	int pos = m_sliderc.GetPos();
	int mn = m_sliderc.GetRangeMin();
	pos -= 1;
	if (pos < mn) return;
	setSliderPos(pos);
}

namespace {
	// 全局变量，构造颜色对话框 
	CColorDialog colorDlg;
}

void CfwDlg::resetArgValue(){
	int r = m_combo.GetCurSel();
	switch (fw->attrs_[r].type) {
	case ArgType::Value:
		m_edit_value = fw->attrs_[r].value;
		UpdateData(false);
		m_edit.ShowWindow(true);
		m_bn_color.ShowWindow(false);
		break;
	case ArgType::Color:
		glm::vec3 color = fw->attrs_[r].color;
		{
			int r = static_cast<int>(color.r * 255);
			int g = static_cast<int>(color.g * 255);
			int b = static_cast<int>(color.b * 255);
			colorDlg.SetCurrentColor(RGB(r, g, b));
			m_bn_color.SetFaceColor(RGB(r, g, b));
		}
		m_edit.ShowWindow(false);
		m_bn_color.ShowWindow(true);
		break;
	default:
		printf("Unexpected Arg Type!\n");
	}
}

void CfwDlg::OnArgComboChange() {
	resetArgValue();
	m_bn_reset.ShowWindow(true);
	m_bn_conform.ShowWindow(true);
}

void CfwDlg::OnBnClickedConform(){
	int r = m_combo.GetCurSel();
	switch (fw->attrs_[r].type) {
	case ArgType::Value:
		UpdateData(true);
		fw->attrs_[r].value = m_edit_value;
		break;
	case ArgType::Color: {
			COLORREF color = colorDlg.GetColor();
			fw->attrs_[r].color = glm::vec3(
				GetRValue(color) / 255.0,
				GetGValue(color) / 255.0,
				GetBValue(color) / 255.0);
		}
		break;
	default:
		printf("Unexpected Arg Type!\n");
	}
}

void CfwDlg::OnBnClickedColorBtn(){
	// 显示颜色对话框，并判断是否点击了“确定”
	if (IDOK == colorDlg.DoModal()) {
		COLORREF color = colorDlg.GetColor();  
		m_bn_color.SetFaceColor(color);
	}
}
