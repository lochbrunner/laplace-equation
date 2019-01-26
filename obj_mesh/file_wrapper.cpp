#include "file_wrapper.h"


#include <fstream>

file_wrapper::file_wrapper(const char * filename, std::ios_base::openmode mode) : m_file(filename, std::ios::in)
{
	
}


file_wrapper::~file_wrapper(void)
{

}


bool file_wrapper::eof()
{
	return m_file.eof();
}

void file_wrapper::getline(char* _Str, std::streamsize _Count)
{
	//char buffer[128];
	m_file.getline(_Str, _Count);
}

void file_wrapper::close()
{
	m_file.close();
}