#pragma once
#include <iostream>
#include <string>
#include <functional>
#include <exception>
#include "format.h"
#include <windows.h>

// 方法来自https://bbs.csdn.net/topics/60176846
#include <dbghelp.h>
#pragma comment(lib, "Dbghelp.lib")

#define MAX_EXCEPTION_MESSAGE_LEN 50
#define FW_CODELOC __FILE__ " (" FW_MSTRING(__LINE__) ")"

enum class ErrorCode : int32_t {
	Success = 0,
	AssertionFailed,
	NotSupported,
	UnexpectedNull,
	InvalidArgs,
	OutOfBound,
	AllocFailed,
	ExecutionFailed,
	NotInitialized,
	AlreadyInitialized,
	KeyNotFound,
	DuplicatedKey,
};



inline std::string errorCode2str(ErrorCode c) noexcept {
	switch (c) {
	case ErrorCode::Success: return "Success";
	case ErrorCode::AssertionFailed: return "Assertion failed";
	case ErrorCode::NotSupported: return "Not supported";
	case ErrorCode::UnexpectedNull: return "Unexpected null pointer";
	case ErrorCode::InvalidArgs: return "Invalid arguments";
	case ErrorCode::OutOfBound: return "Index out of bound";
	case ErrorCode::AllocFailed: return "Memory allocation failed";
	case ErrorCode::ExecutionFailed: return "Task execution failed";
	case ErrorCode::NotInitialized: return "Not initialized";
	case ErrorCode::AlreadyInitialized: return "Already initialized";
	case ErrorCode::KeyNotFound: return "Key not found";
	case ErrorCode::DuplicatedKey: return "Duplicated key";
	}
	return "Unknown Parrots Error";
}


// 方法来自https://blog.csdn.net/windpenguin/article/details/80382344
// 改为调用栈打印成了20层，并删掉了不需要的输出
inline std::string TraceStack()
{
	static constexpr int MAX_STACK_FRAMES = 20;

	void *pStack[MAX_STACK_FRAMES];

	HANDLE process = GetCurrentProcess();
	SymInitialize(process, NULL, TRUE);
	WORD frames = CaptureStackBackTrace(0, MAX_STACK_FRAMES, pStack, NULL);

	std::ostringstream oss;
	oss << "stack traceback: " << std::endl;
	for (WORD i = 0; i < frames; ++i) {
		DWORD64 address = (DWORD64)(pStack[i]);

		DWORD64 displacementSym = 0;
		char buffer[sizeof(SYMBOL_INFO) + MAX_SYM_NAME * sizeof(TCHAR)];
		PSYMBOL_INFO pSymbol = (PSYMBOL_INFO)buffer;
		pSymbol->SizeOfStruct = sizeof(SYMBOL_INFO);
		pSymbol->MaxNameLen = MAX_SYM_NAME;

		DWORD displacementLine = 0;
		IMAGEHLP_LINE64 line;
		//SymSetOptions(SYMOPT_LOAD_LINES);
		line.SizeOfStruct = sizeof(IMAGEHLP_LINE64);

		if (SymFromAddr(process, address, &displacementSym, pSymbol) &&
			SymGetLineFromAddr64(process, address, &displacementLine, &line)) {
			oss << "\t" << pSymbol->Name << " at " << line.FileName << ":" <<
				line.LineNumber << "(0x" << std::hex << pSymbol->Address <<
				std::dec << ")" << std::endl;
		}  /* else {
			oss << "\terror: " << GetLastError() << std::endl;
		} */
	}
	return oss.str();
}

struct FwExceptionBase : public std::exception
{
	const char* what() const noexcept override {
		return getErrorMsg();
	}

	virtual const char* getErrorMsg() const = 0;
};

template <ErrorCode e>
struct FwException : public FwExceptionBase
{
	std::string str_;
	FwException(std::string s = "") :
		str_(sstr(errorCode2str(e), " : ", s)) {}
	const char* getErrorMsg() const override {
		return str_.c_str();
	}
	FwException& operator<<(std::string s) {
		str_ += " " + s;
		return *this;
	}
	~FwException() override{
		std::cout << str_ << std::endl << TraceStack() << std::endl;
		terminate();
	}
};

#define FW_THROW(ExceptionType) \
    throw FwException<ErrorCode::ExceptionType>(\
            "Assertion failed at " FW_CODELOC)

#define FW_NO_THROW(ExceptionType) \
    throw FwException<ErrorCode::ExceptionType>(\
            "Assertion failed at " FW_CODELOC)

#define FW_CHECK(ExceptionType, cond) \
    if (!(cond)) FwException<ErrorCode::ExceptionType>(\
            "Assertion " #cond " failed at " FW_CODELOC)

#define FW_NOTSUPPORTED FW_NO_THROW(NotSupported)

#define FW_ASSERT(cond) FW_CHECK(AssertionFailed, cond)

