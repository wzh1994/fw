
// fwDlg.h: 头文件
//

#pragma once

#ifndef __AFXWIN_H__
#error "在包含此文件之前包含“stdafx.h”以生成 PCH 文件"
#endif

#include <opencv2/opencv.hpp>
#include <opencv2/highgui/highgui_c.h>
#include <memory>
#include "resource.h"
#include "OpenGLWindow.h"
#include "firework.h"
// CfwDlg 对话框
class CfwDlg : public CDialogEx
{
	OpenGLWindow *pOpenGLWindow = nullptr;
	std::unique_ptr<FwBase> fw;
	void myInitialize();
	
	// slider change
	void setSliderPos(int pos);
	void onSliderChange();
// 构造
public:
	CfwDlg(CWnd* pParent = nullptr);	// 标准构造函数

// 对话框数据
#ifdef AFX_DESIGN_TIME
	enum { IDD = IDD_FW_DIALOG };
#endif

	protected:
	virtual void DoDataExchange(CDataExchange* pDX);	// DDX/DDV 支持


// 实现
protected:
	HICON m_hIcon;

	// 生成的消息映射函数
	virtual BOOL OnInitDialog();
	afx_msg void OnPaint();
	afx_msg HCURSOR OnQueryDragIcon();
	DECLARE_MESSAGE_MAP()
	void CfwDlg::OnHScroll(UINT nSBCode, UINT nPos, CScrollBar* pScrollBar);
public:
	CSliderCtrl m_sliderc;
	CButton m_pic;
	// 关闭combo的自动排序
	CComboBox m_combo;
	CEdit m_edit;
	CButton m_bn_reset;
	CButton m_bn_conform;
	CMFCButton m_bn_color;
	float m_edit_value;
	afx_msg void OnBnClickedButtonMinus();
	afx_msg void OnBnClickedButtonPlus();
	afx_msg void OnArgComboChange();
	afx_msg void resetArgValue();
	afx_msg void OnBnClickedConform();
	afx_msg void OnBnClickedColorBtn();

};
