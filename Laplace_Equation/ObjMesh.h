#pragma once

//#include <windows.h>
#include <vector>
#include <string>
//
#include <GL/glew.h>
#include "GLSL_Shader.h"
#include <GL/glu.h>
//
#include "AlgebraStuff.h"



class ObjMesh
{
public:
	ObjMesh(Vector3 size = Vector3(1.0f, 1.0f, 1.0f));
	virtual ~ObjMesh(void);

	bool Init(std::string &filename);
	void Render(float *modelviewMatrix, float *projectionMatrix);

	bool HitPoint(Vector3 &orig, Vector &dir, Vector3 &hit);

	

private:

	void GetNextWord(int SourceSize, const char* const SourceBuffer, int DestSize, char* const DestBuffer, int &EndIndex, bool &EndOfFile);
	bool CompareString(int StingSize, const char* String1, const char* String2);
	void GetIndexTrippel(int StringSize, const char* SourceString, CustomeVector<3, int> &trippel);


	GLuint m_vertexBuffer;
	GLuint m_indexBuffer;
	//GLuint m_texCoordBuffer;
	GLuint m_normalBuffer;
	
	//GLuint m_logoTexID;
	//GLuint m_logoNormalTexID;

	Vector3 m_Size;
	int cVertices;

	static GLSLProgram* m_GLSLProgram;
	static bool m_bShaderInitiated;

	//std::vector<float> calculateNormalMatrix(const float* modelviewMatrix);

	
	void calculateNormalMatrix(const float* modelviewMatrix, float* N);
	bool IntersectTriangle(Vector orig, Vector dir, 
                        Vector v0, Vector v1, Vector v2, float &t);

	std::vector<GLuint> m_indices;
	std::vector<Vector3> m_vertices_sorted;
};

