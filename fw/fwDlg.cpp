﻿
// fwDlg.cpp: 实现文件
//

#include "stdafx.h"
#ifndef __AFXWIN_H__
#error "在包含此文件之前包含“stdafx.h”以生成 PCH 文件"
#endif

#include "resource.h"		// 主符号
#include "fwDlg.h"
#include "afxdialogex.h"
#include <ctime>
#include <thread>
#include <iostream>
#include "firework.h"

#ifdef _DEBUG
#define new DEBUG_NEW
#endif

// 与窗口有关的变量在这里，限定本cpp文件内部使用
namespace {
	const char opencvWindow[] = "IDC_OPENCV_OUTPUT";

	// 子窗口大小
	const int subWindowWidth = 400;
	const int subWindowHeight = 400;

	// 外边距
	const int leftMargin = 50;
	const int rightMargin = 50;
	const int topMargin = 50;
	const int bottomMargin = 50;

	// 内边距
	const int windowMargin = 100;
	const int rowMargin = 20;

	// 控件边距
	const int widgetMargin = 15;

	// 控件大小
	const int widgetWidth = 70;
	const int widgetHeight = 25;
	const int comboWidth = 100;
	const int sliderWidth = 600;
	const int sliderHeight = 50;
	const int autoPlaySide = 50;

	//窗口大小
	const int windowWidth =
		leftMargin + subWindowWidth * 2 + rightMargin + windowMargin;
	const int windowHeight = topMargin + subWindowHeight +
		rowMargin * 3 + widgetHeight * 3 + sliderHeight + bottomMargin;
}

// 其他实用常量
namespace {
	const UINT_PTR autoPlayId = 0; // MFC计时器id
}

using firework::FireWorkType;
using firework::ArgType;

// CfwDlg 对话框

extern "C" __declspec(dllexport)
void ShowDialog(
		FireWorkType type, float* args, const char* pMovieName, size_t len) {
	AFX_MANAGE_STATE(AfxGetStaticModuleState());
	FW_ASSERT(args) << "invalid args!";
	FW_ASSERT(pMovieName) << "invalid movie name!";
	std::string movieName(pMovieName, len);
	CfwDlg dlg(type, args, movieName);
	dlg.DoModal();
}

CfwDlg::CfwDlg(FireWorkType type, float* args,
		string_t movieName, CWnd* pParent /*=nullptr*/)
	: CDialogEx(IDD_FW_DIALOG, pParent), movieName_(movieName)
{
	m_hIcon = AfxGetApp()->LoadIcon(IDR_MAINFRAME);

	/* -------------------------------
	 * firework
	 * 实例化fw对象的时候，不执行任何opengl相关的初始化工作
	 * 后续在初始化opengl之后，通过调用initialize来初始化opengl相关的东西
	 * -------------------------------
	 */
	
	fw.reset(firework::getFirework(type, args));
	sliderLen_ = fw->getTotalFrame();
}


/* ==================================================
 * 初始化，包括opencv窗口，opengl窗口，和一些控件位置
 * ==================================================
 */
void CfwDlg::setSliderPos(int pos) {
	m_sliderc.SetPos(pos);//当前停留的位置
	onSliderChange();
}

// 初始化opencv窗口时，用于拾色器的回调函数
void opencv_mouse_callback(int event, int x, int y, int flags, void* ustc) {
	CfwDlg* dlg = static_cast<CfwDlg*>(ustc);
	if (dlg->bColorSelecting_) {
		if (flags == 2) {
			dlg->changeGetColorStatus();
			dlg->resetArgValue();
		} else {
			int pos = dlg->m_sliderc.GetPos();
			cv::Mat& m = dlg->pPhotos_[pos];
			// 横纵坐标分别对应列和行
			auto re = m.at<cv::Vec3b>(y, x);
			// opencv 的颜色是BGR，而opengl的颜色是RGB
			auto color = RGB(re[2], re[1], re[0]);
			cout << "(" << ((float)re[2]) / 255.0 << ", " << ((float)re[1])/255.0 << ", " << ((float)re[0])/255.0 << ")" << endl;
			if (flags == 1) {
				dlg->colorDlg.m_cc.rgbResult = color;
				dlg->changeGetColorStatus();
			}
			dlg->m_btn_color.SetFaceColor(color);
		}
	}
}

void CfwDlg::myInitialize() {

	// 初始化整个窗口
	MoveWindow(300, 100, windowWidth, windowHeight);

	/* -------------------------------
	 * opengl
	 * -------------------------------
	 */
	// 由于OpenGLWindow需要使用FwBase类的引用，因此需要放在fw类的实例化之后
	// 以保证fw已经被实例化出具体的对象
	pOpenGLWindow = new OpenGLWindow(*fw, subWindowWidth, subWindowHeight);
	pOpenGLWindow->Create(
		NULL, NULL, WS_CHILD | WS_CLIPSIBLINGS | WS_CLIPCHILDREN | WS_VISIBLE,
		CRect(leftMargin, topMargin,
			leftMargin + subWindowWidth, topMargin + subWindowHeight),
		this,   //this is the parent
		0);

	/* -------------------------------
	 * initialize firework
	 * -------------------------------
	 */
	// 构造OpenGLWindow对象时候，初始化glew，因此与glew有关的初始化操作放在这里
	fw->initialize();

	/* -------------------------------
	 * opencv
	 * -------------------------------
	 */
	//CWnd是MFC窗口类的基类,提供了微软基础类库中所有窗口类的基本功能。
	//CWnd  *pPictureWnd = GetDlgItem(IDC_PIC);
	CWnd *pPictureWnd = new CStatic();
	// 创建窗口并设置位置
	pPictureWnd->Create(
		NULL, NULL, WS_CHILD | WS_CLIPSIBLINGS | WS_CLIPCHILDREN | WS_VISIBLE,
		CRect(leftMargin + subWindowWidth + windowMargin, topMargin,
			leftMargin + subWindowWidth * 2 + windowMargin,
			topMargin + subWindowHeight),
		this/*this is the parent */, 0);
	cvNamedWindow(opencvWindow, 0);//设置窗口名
	cvResizeWindow(opencvWindow, subWindowWidth, subWindowWidth);
	// 设置回调函数，用于拾色器鼠标移动事件
	cv::setMouseCallback(opencvWindow, opencv_mouse_callback, this);
	//hWnd 表示窗口句柄,获取窗口句柄
	HWND hWnd = (HWND)cvGetWindowHandle("IDC_OPENCV_OUTPUT");
	//GetParent函数一个指定子窗口的父窗口句柄
	HWND hParent = ::GetParent(hWnd);
	::SetParent(hWnd, pPictureWnd->m_hWnd);
	//ShowWindow指定窗口中显示
	::ShowWindow(hParent, SW_HIDE);

	/* -------------------------------
	 * 为opencv的框设置内容
	 * -------------------------------
	 */
	cv::VideoCapture cap;
	cap.open(movieName_);
	FW_ASSERT(cap.isOpened()) << "Error open movies given: " << movieName_;
	size_t totalFrame = cap.get(cv::CAP_PROP_FRAME_COUNT);
	for (size_t i = 0; i < std::min(totalFrame, fw->getTotalFrame()); ++i) {
		cv::Mat frame;
		cap >> frame;
		FW_ASSERT(!frame.empty()) << sstr("Error get frames: ", i);
		pPhotos_.emplace_back(subWindowWidth, subWindowWidth, CV_8UC3);
		cv::resize(
			frame, pPhotos_.back(), cv::Size(subWindowWidth, subWindowHeight));
	}

	/* -------------------------------
	 * combo
	 * -------------------------------
	 */
	// 调整位置
	CWnd *pComboWnd = GetDlgItem(IDC_COMBO1);
	pComboWnd->SetWindowPos(NULL, leftMargin,
		topMargin + subWindowHeight + rowMargin,
		comboWidth, widgetHeight, SWP_NOZORDER);
	// 将fw的所有attr放在选择框里面
	for (auto it = fw->attrs().begin(); it != fw->attrs().end(); ++it) {
		m_combo.AddString(it->name.c_str());
	}

	/* -------------------------------
	 * edit box and color button
	 * 设置位置
	 * -------------------------------
	 */
	CWnd  *pColorWnd = GetDlgItem(IDC_BUTTON5);
	CWnd  *pColorPickWnd = GetDlgItem(IDC_BUTTON6);
	CWnd  *pEditWnd1 = GetDlgItem(IDC_EDIT1);
	CWnd  *pEditWnd2 = GetDlgItem(IDC_EDIT2);
	CWnd  *pEditWnd3 = GetDlgItem(IDC_EDIT3);
	pColorWnd->SetWindowPos(NULL, leftMargin + comboWidth + widgetMargin,
		topMargin + subWindowHeight + rowMargin,
		widgetWidth, widgetHeight, SWP_NOZORDER);
	pColorPickWnd->SetWindowPos(NULL,
		leftMargin + comboWidth + widgetMargin * 2 + widgetWidth,
		topMargin + subWindowHeight + rowMargin,
		widgetWidth, widgetHeight, SWP_NOZORDER);
	pEditWnd1->SetWindowPos(NULL, leftMargin + comboWidth + widgetMargin,
		topMargin + subWindowHeight + rowMargin,
		widgetWidth, widgetHeight, SWP_NOZORDER);
	pEditWnd2->SetWindowPos(NULL,
		leftMargin + comboWidth + widgetMargin * 2 + widgetWidth,
		topMargin + subWindowHeight + rowMargin,
		widgetWidth, widgetHeight, SWP_NOZORDER);
	pEditWnd3->SetWindowPos(NULL,
		leftMargin + comboWidth + widgetMargin * 3 + widgetWidth * 2,
		topMargin + subWindowHeight + rowMargin,
		widgetWidth, widgetHeight, SWP_NOZORDER);

	/* -------------------------------
	 * autoplay button
	 * -------------------------------
	 */
	m_speed_combo.AddString(L"0.2倍速");
	m_speed_combo.AddString(L"0.5倍速");
	m_speed_combo.AddString(L"1.0倍速");
	//设置位置
	CWnd  *pAutoPlayComboWnd = GetDlgItem(IDC_COMBO2);
	pAutoPlayComboWnd->SetWindowPos(NULL, (windowWidth - autoPlaySide) / 2,
		topMargin + subWindowHeight + rowMargin,
		autoPlaySide, widgetHeight, SWP_NOZORDER);
	CWnd  *pAutoPlayWnd = GetDlgItem(IDC_BUTTON7);
	pAutoPlayWnd->SetWindowPos(NULL, (windowWidth - autoPlaySide) / 2,
		topMargin + subWindowHeight + rowMargin * 2 + widgetHeight,
		autoPlaySide, widgetHeight, SWP_NOZORDER);

	/* -------------------------------
	 * reset and conform button
	 * 设置位置
	 * -------------------------------
	 */
	CWnd  *pResetWnd = GetDlgItem(IDC_BUTTON2);
	CWnd  *pConformWnd = GetDlgItem(IDC_BUTTON3);
	pResetWnd->SetWindowPos(NULL, leftMargin + comboWidth + widgetMargin,
		topMargin + subWindowHeight + rowMargin * 2 + widgetHeight,
		widgetWidth, widgetHeight, SWP_NOZORDER);
	pConformWnd->SetWindowPos(NULL,
		leftMargin + comboWidth + widgetMargin * 2 + widgetWidth,
		topMargin + subWindowHeight + rowMargin * 2 + widgetHeight,
		widgetWidth, widgetHeight, SWP_NOZORDER);

	/* -------------------------------
	 * list
	 * -------------------------------
	 */
	CWnd  *pListWnd = GetDlgItem(IDC_LIST1);
	pListWnd->SetWindowPos(NULL, leftMargin,
		topMargin + subWindowHeight + rowMargin * 2 + widgetHeight,
		widgetWidth, widgetHeight * 3, SWP_NOZORDER);
	ListView_SetExtendedListViewStyle(
		m_list.GetSafeHwnd(), m_list.GetExStyle() | LVS_EX_CHECKBOXES );
	if (fw->isMixture()) {
		m_list.ShowWindow(true);
		const bool* showFlags = fw->showFlags();
		for (size_t i = 0; i < fw->nSubFw(); ++i) {
			BOOL fcheck = showFlags[i] ? 1 : 0;
			m_list.InsertItem(i, (L"烟花" + std::to_wstring(i + 1)).c_str());
			m_list.SetCheck(i, fcheck);
		}
	} else {
		m_list.ShowWindow(false);
	}


	/* -------------------------------
	 * slider and its control button
	 * -------------------------------
	 */
	// 设置位置
	CWnd  *pSliderWnd = GetDlgItem(IDC_SLIDER1);
	CWnd  *pSubWnd = GetDlgItem(IDC_BUTTON4);
	CWnd  *pAddWnd = GetDlgItem(IDC_BUTTON1);
	pSliderWnd->SetWindowPos(NULL, (windowWidth - sliderWidth)/ 2,
		topMargin + subWindowHeight + rowMargin * 3 + widgetHeight * 2,
		sliderWidth, sliderHeight, SWP_NOZORDER);
	pAddWnd->SetWindowPos(NULL, (windowWidth + sliderWidth) / 2 + widgetMargin,
		topMargin + subWindowHeight + rowMargin * 3 + widgetHeight * 2,
		sliderHeight, sliderHeight, SWP_NOZORDER);
	pSubWnd->SetWindowPos(NULL,
		(windowWidth - sliderWidth) / 2 - widgetMargin - sliderHeight,
		topMargin + subWindowHeight + rowMargin * 3 + widgetHeight * 2,
		sliderHeight, sliderHeight, SWP_NOZORDER);
	// 因为setSliderPos会触发onSliderChange， 该函数需要在fw被实例化之后调用
	m_sliderc.SetRange(0, sliderLen_ - 1);//设置范围
	m_sliderc.SetTicFreq(1);//设置显示刻度的间隔
	setSliderPos(0);
	m_sliderc.SetLineSize(2);//一行的大小，对应键盘的方向键
}

/* ========================================
 * 事件映射函数
 * ========================================
 */
 
void CfwDlg::DoDataExchange(CDataExchange* pDX)
{
	CDialogEx::DoDataExchange(pDX);
	DDX_Control(pDX, IDC_SLIDER1, m_sliderc);
	DDX_Control(pDX, IDC_COMBO1, m_combo);
	DDX_Control(pDX, IDC_COMBO2, m_speed_combo);
	DDX_Control(pDX, IDC_EDIT1, m_edit1);
	DDX_Control(pDX, IDC_EDIT2, m_edit3);
	DDX_Control(pDX, IDC_EDIT3, m_edit2);
	DDX_Text(pDX, IDC_EDIT1, m_edit_value1);
	DDX_Text(pDX, IDC_EDIT2, m_edit_value2);
	DDX_Text(pDX, IDC_EDIT3, m_edit_value3);
	DDX_Control(pDX, IDC_BUTTON2, m_btn_reset);
	DDX_Control(pDX, IDC_BUTTON3, m_btn_conform);
	DDX_Control(pDX, IDC_BUTTON5, m_btn_color);
	DDX_Control(pDX, IDC_BUTTON6, m_btn_get_color);
	DDX_Control(pDX, IDC_LIST1, m_list);
}

BEGIN_MESSAGE_MAP(CfwDlg, CDialogEx)
	ON_WM_PAINT()
	ON_WM_QUERYDRAGICON()
	ON_WM_HSCROLL()
	ON_WM_TIMER()
	ON_BN_CLICKED(IDC_BUTTON1, &CfwDlg::OnBnClickedButtonMinus)
	ON_BN_CLICKED(IDC_BUTTON4, &CfwDlg::OnBnClickedButtonPlus)
	ON_CBN_SELCHANGE(IDC_COMBO1, &CfwDlg::OnArgComboChange)
	ON_BN_CLICKED(IDC_BUTTON2, &CfwDlg::resetArgValue)
	ON_BN_CLICKED(IDC_BUTTON3, &CfwDlg::OnBnClickedConform)
	ON_BN_CLICKED(IDC_BUTTON5, &CfwDlg::OnBnClickedColorBtn)
	ON_BN_CLICKED(IDC_BUTTON6, &CfwDlg::changeGetColorStatus)
	ON_BN_CLICKED(IDC_BUTTON7, &CfwDlg::OnBnClickAutoPlay)
	ON_CBN_SELCHANGE(IDC_COMBO2, &CfwDlg::OnCbnSelchangeCombo2)
	ON_NOTIFY(LVN_ITEMCHANGED, IDC_LIST1, &CfwDlg::OnLvnItemchangedList1)
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
	static std::mutex mtx;
	std::unique_lock<std::mutex> l(mtx);
	int pos = m_sliderc.GetPos();
	cv::imshow(opencvWindow, pPhotos_[pos]);
	resetArgValue();
	fw->GetParticles(pos);
	pOpenGLWindow->RedrawWindow();
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

void CfwDlg::resetArgValue(){
	int r = m_combo.GetCurSel();
	int pos = m_sliderc.GetPos();
	if (r == -1) {
		m_edit1.ShowWindow(false);
		m_btn_color.ShowWindow(false);
		return;
	}
	size_t idx = r;
	switch (fw->attrs()[idx].type) {
	case ArgType::Scalar:
	case ArgType::ScalarGroup:
		m_edit_value1 = *fw->getArgs(idx, pos);
		UpdateData(false);
		m_edit1.ShowWindow(true);
		m_edit2.ShowWindow(false);
		m_edit3.ShowWindow(false);
		m_btn_color.ShowWindow(false);
		m_btn_get_color.ShowWindow(false);
		break;
	case ArgType::Vector3:
	case ArgType::Vec3Group:
		m_edit_value1 = fw->getArgs(idx, pos)[0];
		m_edit_value2 = fw->getArgs(idx, pos)[1];
		m_edit_value3 = fw->getArgs(idx, pos)[2];
		UpdateData(false);
		m_edit1.ShowWindow(true);
		m_edit2.ShowWindow(true);
		m_edit3.ShowWindow(true);
		m_btn_color.ShowWindow(false);
		m_btn_get_color.ShowWindow(false);
		break;
	case ArgType::Color:
	case ArgType::ColorGroup:
		{
			int r = static_cast<int>(fw->getArgs(idx, pos)[0] * 255);
			int g = static_cast<int>(fw->getArgs(idx, pos)[1] * 255);
			int b = static_cast<int>(fw->getArgs(idx, pos)[2] * 255);
			auto color = RGB(r, g, b);
			// colorDlg.SetCurrentColor()无效，必须直接修改其值
			colorDlg.m_cc.rgbResult = color;
			m_btn_color.SetFaceColor(color);
		}
		m_edit1.ShowWindow(false);
		m_edit2.ShowWindow(false);
		m_edit3.ShowWindow(false);
		m_btn_color.ShowWindow(true);
		m_btn_get_color.ShowWindow(true);
		break;
	}
}

void CfwDlg::OnArgComboChange() {
	resetArgValue();
	m_btn_reset.ShowWindow(true);
	m_btn_conform.ShowWindow(true);
}

void CfwDlg::OnBnClickedConform(){
	int r = m_combo.GetCurSel();
	int pos = m_sliderc.GetPos();
	size_t idx = r;
	switch (fw->attrs()[idx].type) {
	case ArgType::Scalar:
	case ArgType::ScalarGroup:
		UpdateData(true);
		*fw->getArgs(idx, pos) = m_edit_value1;
		break;
	case ArgType::Vector3:
	case ArgType::Vec3Group:
		UpdateData(true);
		fw->getArgs(idx, pos)[0] = m_edit_value1;
		fw->getArgs(idx, pos)[1] = m_edit_value2;
		fw->getArgs(idx, pos)[2] = m_edit_value3;
		break;
	case ArgType::Color:
	case ArgType::ColorGroup:
		{
			// 一个新的作用域 用于定义临时变量color
			COLORREF color = colorDlg.GetColor();
			fw->getArgs(idx, pos)[0] = GetRValue(color) / 255.0;
			fw->getArgs(idx, pos)[1] = GetGValue(color) / 255.0;
			fw->getArgs(idx, pos)[2] = GetBValue(color) / 255.0;
		}
		break;
	default:
		FW_NOTSUPPORTED << "Unexpected data type!";
	}
	fw->prepare();
	fw->GetParticles(pos);
	pOpenGLWindow->Invalidate();
}

void CfwDlg::OnBnClickedColorBtn(){
	// 显示颜色对话框，并判断是否点击了“确定”
	if (IDOK == colorDlg.DoModal()) {
		COLORREF color = colorDlg.GetColor();  
		m_btn_color.SetFaceColor(color);
	}
}

void CfwDlg::changeGetColorStatus() {
	bColorSelecting_ = !bColorSelecting_;
	if (bColorSelecting_) {
		GetDlgItem(IDC_BUTTON6)->EnableWindow(FALSE);
	} else {
		GetDlgItem(IDC_BUTTON6)->EnableWindow(TRUE);
	}
}

/* ========================================
 * 自动播放
 * ========================================
 */
// 播放速率调整
void CfwDlg::OnCbnSelchangeCombo2(){
	static float speeds[3]{0.2, 0.5, 1.0};
	autoPlaySpeed_ = speeds[m_speed_combo.GetCurSel()];
}

void CfwDlg::autoPlay() {
	int pos = m_sliderc.GetPos();
	if (++pos < sliderLen_) {
		setSliderPos(pos);
	} else {
		KillTimer(autoPlayId);
		changeAllControlStatus(true);
	}
}

void CfwDlg::changeAllControlStatus(BOOL bEnable) {
	GetDlgItem(IDC_BUTTON1)->EnableWindow(bEnable);
	GetDlgItem(IDC_BUTTON2)->EnableWindow(bEnable);
	GetDlgItem(IDC_BUTTON3)->EnableWindow(bEnable);
	GetDlgItem(IDC_BUTTON4)->EnableWindow(bEnable);
	GetDlgItem(IDC_BUTTON5)->EnableWindow(bEnable);
	GetDlgItem(IDC_BUTTON6)->EnableWindow(bEnable);
	GetDlgItem(IDC_BUTTON7)->EnableWindow(bEnable);
	GetDlgItem(IDC_COMBO1)->EnableWindow(bEnable);
	GetDlgItem(IDC_COMBO2)->EnableWindow(bEnable);
	GetDlgItem(IDC_EDIT1)->EnableWindow(bEnable);
	GetDlgItem(IDC_EDIT2)->EnableWindow(bEnable);
	GetDlgItem(IDC_EDIT3)->EnableWindow(bEnable);
	GetDlgItem(IDC_SLIDER1)->EnableWindow(bEnable);
}

void CfwDlg::OnBnClickAutoPlay() {
	changeAllControlStatus(false);
	float freq = 1000 / (36.0 * autoPlaySpeed_);
	SetTimer(autoPlayId, freq, nullptr);
};

void CfwDlg::OnTimer(UINT_PTR nIDEvent) {
	switch (nIDEvent) {
	case autoPlayId:
		autoPlay();
		break;
	default:
		break;
	}
}


void CfwDlg::OnLvnItemchangedList1(NMHDR *pNMHDR, LRESULT *pResult)
{
	LPNMLISTVIEW pNMLV = reinterpret_cast<LPNMLISTVIEW>(pNMHDR);
	*pResult = 0;

	int nItem, i;
	nItem = m_list.GetItemCount();
	for (i = 0; i < nItem; ++i) {
		bool* showFlags = fw->showFlags();
		showFlags[i] = m_list.GetCheck(i);
	}
	int pos = m_sliderc.GetPos();
	fw->prepare();
	fw->GetParticles(pos);
	pOpenGLWindow->Invalidate();
}
