#pragma once

#include <vector_types.h>
#include <list>
#include <string>
#include "ObjMesh.h"
#include "globals.h"



class SceneLoader
{
public:
	struct Object{
		Object(){ pMesh = new ObjMesh(Vector3(1.0f, 1.0f, 1.0f));}
		ObjMesh *pMesh;
		float value;
	};
	SceneLoader();
	~SceneLoader(void);
	void LoadScene(std::string &filename);

	void Render(float *modelviewMatrix, float *projectionMatrix);

	VolumeType *GetInitFieldPtr() {return m_pField;}

	VolumeType *LoadField(cudaExtent volumeSize);

private:
	void GetNextWord(int SourceSize, const char* const SourceBuffer, int DestSize, char* const DestBuffer, int &EndIndex, bool &EndOfFile);
	bool CompareString(int StingSize, const char* String1, const char* String2);
	void GetIndexTrippel(int StringSize, const char* SourceString, CustomeVector<3, int> &trippel);

	void LoadBoundaryValue(VolumeType &value, int &currentIndex, const int size, char* buffer, const int wordSize, char* wordBuffer, bool &bEndOfLine);

	VolumeType* m_pField;
	std::list<Object> m_ObjectList;

	struct _BoundaryCondition{
		_BoundaryCondition() : br(false), bl(false), bt(false), bb(false), bf(false), bh(false), bdef(false) {}
		VolumeType r, l, t, b, f, h, def;
		bool br, bl, bt, bb, bf, bh, bdef;
	} m_Boundary;
};

