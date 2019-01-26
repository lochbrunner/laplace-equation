#pragma once

#include <fstream>

class file_wrapper
{
public:
	__declspec(dllexport) file_wrapper(const char * filename, std::ios_base::openmode);
	__declspec(dllexport) ~file_wrapper(void);

	__declspec(dllexport) bool eof();
	__declspec(dllexport) void getline(char* _Str, std::streamsize _Count);
	__declspec(dllexport) void close();

private:
	std::fstream m_file;
};

