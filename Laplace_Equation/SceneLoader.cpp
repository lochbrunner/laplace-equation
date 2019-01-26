#include "SceneLoader.h"
#include "../obj_mesh/file_wrapper.h"



SceneLoader::SceneLoader()
{
	m_pField = NULL;
	m_Boundary.def.x = 0.0f;
	m_Boundary.def.y = 0.0f;
	m_Boundary.def.z = 0.0f;
	m_Boundary.def.w = 0.0f;
}


SceneLoader::~SceneLoader(void)
{
	for(auto it = m_ObjectList.begin(); it != m_ObjectList.end(); ++it)
	{
		delete it->pMesh;
	}
	m_ObjectList.clear();
	if(m_pField) delete[] m_pField;
}



VolumeType *SceneLoader::LoadField(cudaExtent volumeSize)
{
	if(m_pField) delete[] m_pField;

	m_pField = new VolumeType[volumeSize.width*volumeSize.height*volumeSize.depth];

	Vector3 dir = Vector3(1.0f, 0.0f, 0.0f);
	Vector3 hit;

	float ddw = 2.0f/static_cast<float>(volumeSize.width);
	float ddh = 2.0f/static_cast<float>(volumeSize.height);
	float ddd = 2.0f/static_cast<float>(volumeSize.depth);

	float swf = -0.5f*ddw*static_cast<float>(volumeSize.width);
	float shf = -0.5f*ddh*static_cast<float>(volumeSize.height);
	float sdf = -0.5f*ddd*static_cast<float>(volumeSize.depth);

	for(unsigned int w = 0; w < volumeSize.width; w++){
		for(unsigned int h = 0; h < volumeSize.height; h++){
			for(unsigned int d = 0; d < volumeSize.depth; d++){
				int index = (w*volumeSize.height+h)*volumeSize.depth+d;
				if(w==0 && m_Boundary.bl) m_pField[index] = m_Boundary.l;
				else if(w==volumeSize.width-1 && m_Boundary.br) m_pField[index] = m_Boundary.r;
				else if(h==0 && m_Boundary.bb) m_pField[index] = m_Boundary.b;
				else if(h==volumeSize.height-1 && m_Boundary.bt) m_pField[index] = m_Boundary.t;
				else if(d==0 && m_Boundary.bh) m_pField[index] = m_Boundary.h;
				else if(d==volumeSize.depth-1 && m_Boundary.bf) m_pField[index] = m_Boundary.f;
				else if(m_Boundary.bdef) m_pField[index] = m_Boundary.def;
			}
		}
	}
	
	bool bFirst = true;
	for(auto it = m_ObjectList.begin(); it != m_ObjectList.end(); ++it){
		float cwf = swf;
		for(unsigned int w = 0; w < volumeSize.width; w++){
			float chf = shf;
			for(unsigned int h = 0; h < volumeSize.height; h++){
				float cdf = sdf;
				bool bInGeomatry = false;
				for(unsigned int d = 0; d < volumeSize.depth; d++){

					Vector3 orig = Vector3(cdf-ddd*0.5f, chf, cwf);

					if(it->pMesh->HitPoint(orig, dir, hit))
					{
						if(hit.x < cdf+ddd*0.5f)
						{
							orig = hit + dir*0.001f;
							if(it->pMesh->HitPoint(orig, dir, hit))	//Double hitting at sharp edges?
							{
								if(hit.x < cdf+ddd*0.5f);
								else bInGeomatry ^= true;
							}
							else bInGeomatry ^= true;
						}
					}
					int index = (w*volumeSize.height+h)*volumeSize.depth+d;
					//if(w==0 && m_Boundary.bl) m_pField[index] = m_Boundary.l;
					//else if(w==volumeSize.width-1 && m_Boundary.br) m_pField[index] = m_Boundary.r;
					//else if(h==0 && m_Boundary.bb) m_pField[index] = m_Boundary.b;
					//else if(h==volumeSize.height-1 && m_Boundary.bt) m_pField[index] = m_Boundary.t;
					//else if(d==0 && m_Boundary.bh) m_pField[index] = m_Boundary.h;
					//else if(d==volumeSize.depth-1 && m_Boundary.bf) m_pField[index] = m_Boundary.f;
					//else
					//{
					//	if(bFirst)
					//	{
						//	m_pField[index].x = bInGeomatry ?  it->value : m_Boundary.def.x;	// initial Value
						//	m_pField[index].y = bInGeomatry ?  it->value : m_Boundary.def.y;	// boundary Value
						//	m_pField[index].z = bInGeomatry ?  1.0f : m_Boundary.def.z;			// Boundary condition
						//	m_pField[index].w = bInGeomatry ?  1.0f : m_Boundary.def.w;			// Transparency									// Transparency
						//}
						//else 
					if(bInGeomatry)
					{
						m_pField[index].x = it->value;		// inital Value
						m_pField[index].y = it->value;		// boundary Value
						m_pField[index].z = 1.0f;			// Boundary condition
						m_pField[index].w = 1.0f;			// Transparency
					}
					//}

					cdf+=ddd;
				}
				chf+=ddh;
			}
			cwf+=ddw;
		}
		bFirst = false;
	}
	return m_pField;
}

void SceneLoader::LoadScene(std::string &filename)
{
	file_wrapper file(filename.c_str(), std::ios::in);

	bool bEndOfLine;

	const int size = 128;
	char *buffer = new char[size];

	const int wordSize = size;
	char wordBuffer[wordSize];

	while(!file.eof())
	{
		bEndOfLine = false;

		file.getline(buffer, size);

		int currentIndex=0;

		GetNextWord(size, buffer, wordSize, wordBuffer, currentIndex,bEndOfLine);

		if(CompareString(size, "Object\0", wordBuffer))
		{
			Object newObject;
			
			if(currentIndex!= size && !bEndOfLine)
			{
				GetNextWord(size, buffer, wordSize, wordBuffer, currentIndex,bEndOfLine);

				newObject.value = static_cast<float>(atof(wordBuffer));
				
			}
			
			if(currentIndex!= size && !bEndOfLine)
			{
				GetNextWord(size, buffer, wordSize, wordBuffer, currentIndex,bEndOfLine);

				newObject.pMesh->Init(std::string(wordBuffer));
				
			}

			m_ObjectList.push_back(newObject);
		}
		else if(CompareString(size, "default\0", wordBuffer))				//right
		{
			m_Boundary.bdef = true;
			LoadBoundaryValue(m_Boundary.def, currentIndex, size, buffer, wordSize, wordBuffer, bEndOfLine);
		}
		else if(CompareString(size, "r\0", wordBuffer))				//right
		{
			m_Boundary.br = true;
			LoadBoundaryValue(m_Boundary.r, currentIndex, size, buffer, wordSize, wordBuffer, bEndOfLine);
		}
		else if(CompareString(size, "l\0", wordBuffer))				//right
		{
			m_Boundary.bl = true;
			LoadBoundaryValue(m_Boundary.l, currentIndex, size, buffer, wordSize, wordBuffer, bEndOfLine);
		}
		else if(CompareString(size, "t\0", wordBuffer))				//right
		{
			m_Boundary.bt = true;
			LoadBoundaryValue(m_Boundary.t, currentIndex, size, buffer, wordSize, wordBuffer, bEndOfLine);
		}
		else if(CompareString(size, "b\0", wordBuffer))				//right
		{
			m_Boundary.bb = true;
			LoadBoundaryValue(m_Boundary.b, currentIndex, size, buffer, wordSize, wordBuffer, bEndOfLine);
		}
		else if(CompareString(size, "f\0", wordBuffer))				//right
		{
			m_Boundary.bf = true;
			LoadBoundaryValue(m_Boundary.f, currentIndex, size, buffer, wordSize, wordBuffer, bEndOfLine);
		}
		else if(CompareString(size, "h\0", wordBuffer))				//right
		{
			m_Boundary.bh = true;
			LoadBoundaryValue(m_Boundary.h, currentIndex, size, buffer, wordSize, wordBuffer, bEndOfLine);
		}
	}


	file.close();
}

void SceneLoader::LoadBoundaryValue(VolumeType &value, int &currentIndex, const int size, char* buffer, const int wordSize, char* wordBuffer, bool &bEndOfLine)
{
	if(currentIndex!= size && !bEndOfLine)
	{
		GetNextWord(size, buffer, wordSize, wordBuffer, currentIndex,bEndOfLine);

		value.x = static_cast<float>(atof(wordBuffer));
	}
	if(currentIndex!= size && !bEndOfLine)
	{
		GetNextWord(size, buffer, wordSize, wordBuffer, currentIndex,bEndOfLine);

		value.y = static_cast<float>(atof(wordBuffer));
	}
	if(currentIndex!= size && !bEndOfLine)
	{
		GetNextWord(size, buffer, wordSize, wordBuffer, currentIndex,bEndOfLine);

		value.z = static_cast<float>(atof(wordBuffer));
	}
	if(currentIndex!= size && !bEndOfLine)
	{
		GetNextWord(size, buffer, wordSize, wordBuffer, currentIndex,bEndOfLine);

		value.w = static_cast<float>(atof(wordBuffer));
	}
}

void SceneLoader::Render(float *modelviewMatrix, float *projectionMatrix)
{
	for(auto it = m_ObjectList.begin(); it != m_ObjectList.end(); ++it)
	{
		it->pMesh->Render(modelviewMatrix, projectionMatrix);
	}
}


void SceneLoader::GetNextWord(int SourceSize, const char* const SourceBuffer, int DestSize, char* const DestBuffer, int &CurrentIndex, bool &EndOfFile){
	if(CurrentIndex < 0)
	{
		return;
	}

	bool bFirstSpace = true;
	EndOfFile=false;

	int DestIndex=0;
	for(int i = CurrentIndex; i < SourceSize; i++)
	{
		CurrentIndex=i;
		if(SourceBuffer[i]=='\0' )
		{
			EndOfFile=true;
			break;			// Wir sind bereits am Ende angekommen
		}

		if(SourceBuffer[i]==' ' && !bFirstSpace) break;		// Das Wort scheint zu Ende zu sein
		
		if(SourceBuffer[i]!=' ')
		{
			bFirstSpace = false;
			DestBuffer[DestIndex]=SourceBuffer[i];
			DestIndex++;
			if(DestIndex == DestSize) break;		// Ziel Speicher zu klein !!
		}

	}
	DestBuffer[DestIndex]='\0';
}

bool SceneLoader::CompareString(int StingSize, const char* String1, const char* String2)
{
	for(int i=0; i < StingSize; i++)
	{
		if(String1[i] != String2[i]) return false;
		if(String1[i] == '\0') break;
	}
	return true;
}

void SceneLoader::GetIndexTrippel(int StringSize, const char* SourceString, CustomeVector<3, int> &trippel)
{
	int IndexOfNumber=0;
	int WordIndexOfNumber=0;

	char Numberbuffer[64];
	for(int i=0; i < StringSize;i++)
	{
		if(SourceString[i]=='\0' ) 
		{
			Numberbuffer[WordIndexOfNumber] ='\0';
			trippel[IndexOfNumber]=atoi(Numberbuffer);
			return;
		}
		if(SourceString[i]=='/')
		{
			Numberbuffer[WordIndexOfNumber] ='\0';
			trippel[IndexOfNumber]=atoi(Numberbuffer);
			WordIndexOfNumber=0;
			IndexOfNumber++;
		}
		else
		{
			Numberbuffer[WordIndexOfNumber]=SourceString[i];
			WordIndexOfNumber++;
		}
	}
}