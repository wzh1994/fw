#pragma once
#include <iostream>
#include <string>
#include <functional>
#include <exception>
#include "format.h"

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
};

#define FW_THROW(ExceptionType) \
    throw FwException<ErrorCode::ExceptionType>(\
            "Assertion failed at " FW_CODELOC)

#define FW_CHECK(ExceptionType, cond) \
    if (!(cond)) throw FwException<ErrorCode::ExceptionType>(\
            "Assertion " #cond " failed at " FW_CODELOC)

#define FW_NOTSUPPORTED FW_THROW(NotSupported)

#define FW_ASSERT(cond) FW_CHECK(AssertionFailed, cond)

