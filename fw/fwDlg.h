
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
#include "exceptions.h"
#include <string>

// CfwDlg 对话框
class CfwDlg : public CDialogEx
{
	using string_t = std::string;
	OpenGLWindow *pOpenGLWindow = nullptr;
	std::unique_ptr<firework::FwBase> fw;
	void myInitialize();
	
	// slider change
	void setSliderPos(int pos);
	void onSliderChange();
// 构造
public:
	CfwDlg(firework::FireWorkType type, float* args,
		string_t movieName, CWnd* pParent = nullptr);

// 对话框数据
#ifdef AFX_DESIGN_TIME
	enum { IDD = IDD_FW_DIALOG };
#endif

	protected:
	virtual void DoDataExchange(CDataExchange* pDX);	// DDX/DDV 支持

private:
	int sliderLen_;
	string_t movieName_;
	float autoPlaySpeed_ = 1.0;

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
	bool bColorSelecting_ = false;
	CColorDialog colorDlg;
	std::vector<cv::Mat> pPhotos_;

	/*控件成员变量*/
public:
	CSliderCtrl m_sliderc;
	// 关闭combo的自动排序
	CComboBox m_combo;
	CComboBox m_speed_combo;
	CEdit m_edit1;
	CEdit m_edit2;
	CEdit m_edit3;
	CButton m_btn_reset;
	CButton m_btn_conform;
	CMFCButton m_btn_color;
	CButton m_btn_get_color;
	float m_edit_value1;
	float m_edit_value2;
	float m_edit_value3;

	// 按钮事件函数
	afx_msg void OnBnClickedButtonMinus();
	afx_msg void OnBnClickedButtonPlus();
	afx_msg void OnArgComboChange();
	afx_msg void resetArgValue();
	afx_msg void OnBnClickedConform();
	afx_msg void OnBnClickedColorBtn();
	afx_msg void changeGetColorStatus();
	afx_msg void OnBnClickAutoPlay();
	afx_msg void OnTimer(UINT_PTR nIDEvent);

private:
	// 其他实用函数
	void autoPlay();
	void changeAllControlStatus(BOOL bEnable);
public:
	afx_msg void OnCbnSelchangeCombo2();
	
	afx_msg void OnLvnItemchangedList1(NMHDR *pNMHDR, LRESULT *pResult);
	CListCtrl m_list;
};
