#pragma once

#include <map>
#include <string>


#ifndef GLuint		// Die Header-Datei wird hier nicht eingebunden um die Kompilierzeut zu verkürzen

typedef unsigned int GLuint;

#endif


using std::string;
using std::ifstream;
using std::map;

class GLSLProgram
{
public:
    struct GLSLShader
    {
        unsigned int id;
        string filename;
        string source;
    };

	__declspec(dllexport) GLSLProgram(const std::string& vertexShader, const std::string& fragmentShader);

    __declspec(dllexport) virtual ~GLSLProgram();

    __declspec(dllexport) void Unload();

    __declspec(dllexport) bool Initialize();
	__declspec(dllexport) void LinkProgram();

    __declspec(dllexport) GLuint GetUniformLocation(const string& name);
    __declspec(dllexport) GLuint GetAttribLocation(const string& name);


	__declspec(dllexport) void SendUniform(const string& name, const int id);
	__declspec(dllexport) void SendUniform4x4(const string& name, const float* matrix, bool transpose=false);
	__declspec(dllexport) void SendUniform3x3(const string& name, const float* matrix, bool transpose=false);
	__declspec(dllexport) void SendUniform2x2(const string& name, const float* matrix, bool transpose=false);
	__declspec(dllexport) void SendUniform(const string& name, const float red, const float green,
			 const float blue, const float alpha);
	__declspec(dllexport) void SendUniform(const string& name, const float x, const float y, const float z);
	__declspec(dllexport) void SendUniform(const string& name, const float x, const float y);
	__declspec(dllexport) void SendUniform(const string& name, const float scalar);

    __declspec(dllexport) void BindAttrib(unsigned int index, const string& attribName);
    __declspec(dllexport) void BindShader();

private:
    string ReadFile(const string& filename);
    bool CompileShader(const GLSLShader& shader);
    void OutputShaderLog(const GLSLShader& shader);

    GLSLShader m_vertexShader;
    GLSLShader m_fragmentShader;
    unsigned int m_programID;

    map<string, GLuint> m_uniformMap;
    map<string, GLuint> m_attribMap;

	//string m_ShaderName1;
	//string m_ShaderName2;
};
