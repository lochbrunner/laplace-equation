

#include "ObjMesh.h"
#include "../obj_mesh/file_wrapper.h"


//std::vector<float> N(3 * 3);

GLSLProgram* ObjMesh::m_GLSLProgram = NULL;
bool ObjMesh::m_bShaderInitiated = false;

ObjMesh::ObjMesh(Vector3 size)
{
	m_vertexBuffer = m_indexBuffer = 0;
	if(m_GLSLProgram == NULL) m_GLSLProgram = new GLSLProgram("../shader/obj.vert", "../shader/obj.frag");
	m_Size = size;
}

ObjMesh::~ObjMesh(void)
{
	glDeleteBuffers(1, &m_vertexBuffer); 
	//glDeleteBuffers(1, &m_colorBuffer); 
	//glDeleteBuffers(1, &m_texCoordBuffer); 
	glDeleteBuffers(1, &m_normalBuffer); 
	//glDeleteBuffers(1, &m_biNormalBuffer); 
	glDeleteBuffers(1, &m_indexBuffer);

	//glDeleteTextures(1, &m_logoTexID);
	//glDeleteTextures(1, &m_logoNormalTexID);

	m_indices.clear();
	m_vertices_sorted.clear();
}

void ObjMesh::GetNextWord(int SourceSize, const char* const SourceBuffer, int DestSize, char* const DestBuffer, int &CurrentIndex, bool &EndOfFile){
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

bool ObjMesh::CompareString(int StingSize, const char* String1, const char* String2)
{
	for(int i=0; i < StingSize; i++)
	{
		if(String1[i] != String2[i]) return false;
		if(String1[i] == '\0') break;
	}
	return true;
}

void ObjMesh::GetIndexTrippel(int StringSize, const char* SourceString, CustomeVector<3, int> &trippel)
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

bool ObjMesh::Init(std::string &filename)
{

	////std::fstream file(filename.c_str(), std::ios::in);

	file_wrapper file(filename.c_str(), std::ios::in);

	const int size = 128;
	char *buffer = new char[size];

	const int wordSize = size;
	char wordBuffer[wordSize];

	

	std::vector<Vector3> vertices;
	std::vector<Vector2> texCoords;
	std::vector<Vector3> normals;


	std::vector<CustomeVector<3, CustomeVector<3, int>>> triIndex;
	std::vector<CustomeVector<4, CustomeVector<3, int>>> quadIndex;

	std::vector<CustomeVector<3, int>>IndexTripleVector;

	bool bEndOfLine;
	
	while(!file.eof())
	{
		bEndOfLine = false;

		file.getline(buffer, size);

		int currentIndex=0;

		GetNextWord(size, buffer, wordSize, wordBuffer, currentIndex,bEndOfLine);

		if(CompareString(size, "v\0", wordBuffer))
		{
			Vector3 ver;
			int IndexOfComponent=0;
			while(currentIndex!= size && !bEndOfLine)
			{
				GetNextWord(size, buffer, wordSize, wordBuffer, currentIndex,bEndOfLine);

				ver[IndexOfComponent] = static_cast<float>(atof(wordBuffer));
				IndexOfComponent++;
			}

			vertices.push_back(m_Size*ver);
		}

		if(CompareString(size, "vn\0", wordBuffer))
		{
			Vector3 ver;
			int IndexOfComponent=0;
			while(currentIndex!= size && !bEndOfLine)
			{
				GetNextWord(size, buffer, wordSize, wordBuffer, currentIndex,bEndOfLine);

				ver[IndexOfComponent] = static_cast<float>(atof(wordBuffer));
				IndexOfComponent++;
			}

			normals.push_back(ver);
		}

		if(CompareString(size, "vt\0", wordBuffer))
		{
			Vector2 ver;
			int IndexOfComponent=0;
			while(currentIndex!= size && !bEndOfLine && IndexOfComponent<3)
			{
				GetNextWord(size, buffer, wordSize, wordBuffer, currentIndex,bEndOfLine);

				ver[IndexOfComponent] = static_cast<float>(atof(wordBuffer));
				IndexOfComponent++;
			}

			texCoords.push_back(ver);
		}

		if(CompareString(size, "f\0", wordBuffer))
		{
			//Neue Variante
			CustomeVector<3,int> IndexTuppel;
			//int IndexOfComponent=0;
			
			while(currentIndex!= size && !bEndOfLine)
			{

				GetNextWord(size, buffer, wordSize, wordBuffer, currentIndex,bEndOfLine);

				GetIndexTrippel(wordSize, wordBuffer, IndexTuppel);
				IndexTripleVector.push_back(IndexTuppel);
				//IndexOfComponent++;
			}
			CustomeVector<3,CustomeVector<3,int>> ver3;
			for(unsigned i = 0; i < IndexTripleVector.size()-2; ++i)
			{
				ver3[0]=IndexTripleVector[0];
				ver3[1]=IndexTripleVector[i+1];
				ver3[2]=IndexTripleVector[i+2];
				triIndex.push_back(ver3);
			}


			IndexTripleVector.clear();
			
		}
		
	}

	file.close();


	// Nun müssen wir die Positionen, Normalen und Texuturkoordinaten noch sortieren, da OpenGL keinen Multiindex unterstützt


	//std::vector<TexCoord> m_texCoords_sorted;
	std::vector<Vector3> normals_sorted;



	//int QuadToTriange[] = {0,1,2,0,2,3};

	int count =0;


	for(std::vector<CustomeVector<3, CustomeVector<3, int>>>::iterator it = triIndex.begin(); it != triIndex.end(); ++it)
	{
		for(unsigned int i=0; i < 3; i++)
		{
			m_indices.push_back(count);
			m_vertices_sorted.push_back(vertices[it->get(i).get(0)-1]);
			normals_sorted.push_back(normals[it->get(i).get(2)-1]);
			count++;
		}
	}

	vertices.clear();
	normals.clear();
	texCoords.clear();

	
	//m_texCoords.push_back(TexCoord(0.0f, 1.0f));
	

	cVertices = m_indices.size();


	glGenBuffers(1, &m_vertexBuffer); //Generate a buffer for the vertices
	glBindBuffer(GL_ARRAY_BUFFER, m_vertexBuffer); //Bind the vertex buffer
	glBufferData(GL_ARRAY_BUFFER, sizeof(GLfloat) * m_vertices_sorted.size() * 3, &m_vertices_sorted[0], GL_STATIC_DRAW); //Send the data to OpenGL

	glGenBuffers(1, &m_indexBuffer);
	glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, m_indexBuffer);
	glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof(GLuint) * m_indices.size(), &m_indices[0], GL_STATIC_DRAW);

	//glGenBuffers(1, &m_texCoordBuffer);
	//glBindBuffer(GL_ARRAY_BUFFER, m_texCoordBuffer); //Bind the vertex buffer
	//glBufferData(GL_ARRAY_BUFFER, sizeof(GLfloat) * m_texCoords_sorted.size() * 2, &m_texCoords_sorted[0], GL_STATIC_DRAW); //Send the data to OpenGL

	glGenBuffers(1, &m_normalBuffer);
	glBindBuffer(GL_ARRAY_BUFFER, m_normalBuffer); //Bind the normal buffer
	glBufferData(GL_ARRAY_BUFFER, sizeof(GLfloat) * normals_sorted.size() * 3, &normals_sorted[0], GL_STATIC_DRAW); //Send the data to OpenGL

	//glGenBuffers(1, &m_biNormalBuffer);
	//glBindBuffer(GL_ARRAY_BUFFER, m_biNormalBuffer); //Bind the BInormal buffer
	//glBufferData(GL_ARRAY_BUFFER, sizeof(GLfloat) * m_biNormals.size() * 3, &m_biNormals[0], GL_STATIC_DRAW); //Send the data to OpenGL




	normals_sorted.clear();


	if(!m_bShaderInitiated)
	{
		if (!m_GLSLProgram->Initialize()) 
		{
			//Could not initialize the shaders.
			return false;
		}
		//Bind the attribute locations
		m_GLSLProgram->BindAttrib(0, "a_Vertex");
		m_GLSLProgram->BindAttrib(1, "a_Normal");
	
		//Re link the program
		m_GLSLProgram->LinkProgram();
	}

	
	return true;
}

void ObjMesh::Render(float *modelviewMatrix, float *projectionMatrix)
{

	glEnable(GL_CULL_FACE);
    glEnable(GL_DEPTH_TEST);
	

	//Get the current matrices from OpenGL
	//float modelviewMatrix[16];
	//float projectionMatrix[16];


	//glLoadIdentity();

	//glTranslatef(0.0f, -10.0f, 0.0f);
	//glRotatef(25.0f, 1.0f, 0.0f, 0.0f);
	////Translate using our zPosition var
	//glTranslatef(0.0, 0.0, -30.0f);
	//glRotatef(rotate, 0.0f, 1.0f, 0.0f);
	//glRotatef(rotate, 1.0f, 0.0f, 0.0f);

	//glGetFloatv(GL_MODELVIEW_MATRIX, modelviewMatrix);
	//glGetFloatv(GL_PROJECTION_MATRIX, projectionMatrix);

	
	//std::vector<float> normalMatrix = calculateNormalMatrix(modelviewMatrix);

	float normalMatrix[9];

	calculateNormalMatrix(modelviewMatrix, normalMatrix);

	//Send the modelview and projection matrices to the shaders
	m_GLSLProgram->BindShader();
	m_GLSLProgram->SendUniform4x4("modelview_matrix", modelviewMatrix);
	m_GLSLProgram->SendUniform4x4("projection_matrix", projectionMatrix);
	m_GLSLProgram->SendUniform3x3("normal_matrix", normalMatrix);

	m_GLSLProgram->SendUniform("light0.diffuse", 1.0f, 1.0f, 1.0f, 1.0f);
	m_GLSLProgram->SendUniform("light0.position", 0.0f, 1.0f, 30.0f, -40.0f);

	glEnableVertexAttribArray(0); //Enable the vertex attribute
	//////glEnableVertexAttribArray(2); //Enable the tecCoord attribute
	glEnableVertexAttribArray(1); //Enable the normal
	////
	//////Bind the vertex array and set the vertex pointer to point at it
    glBindBuffer(GL_ARRAY_BUFFER, m_vertexBuffer);
    glVertexAttribPointer((GLint)0, 3, GL_FLOAT, GL_FALSE, 0, 0);
 //   
	//////// Bind the TexCoordiantes
	//////glBindBuffer(GL_ARRAY_BUFFER, m_texCoordBuffer);
	//////glVertexAttribPointer((GLint)2, 2, GL_FLOAT, GL_FALSE, 0, 0);

	// Normal Buffer
	glBindBuffer(GL_ARRAY_BUFFER, m_normalBuffer);
	glVertexAttribPointer((GLint)1, 3, GL_FLOAT, GL_FALSE, 0, 0);

    //Bind the index array
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, m_indexBuffer);

	//////glActiveTexture(GL_TEXTURE0);
	//////glBindTexture(GL_TEXTURE_2D, m_logoTexID);
	//////
	//////glActiveTexture(GL_TEXTURE1);
	//////glBindTexture(GL_TEXTURE_2D, m_logoNormalTexID);

 ////   //Draw the triangles
	glDrawElements(GL_TRIANGLES, cVertices, GL_UNSIGNED_INT, 0);

	glDisableVertexAttribArray(0); //Disable the vertex attribute
	////glDisableVertexAttribArray(2); //Disable the tecCoord attribute
	glDisableVertexAttribArray(1); //Disable normal attribute


	glDisable(GL_CULL_FACE);
    glDisable(GL_DEPTH_TEST);

	glUseProgram(0);
}


void ObjMesh::calculateNormalMatrix(const float* modelviewMatrix, float* N)
{
	float M[9];
	M[0] = modelviewMatrix[0];
    M[1] = modelviewMatrix[1];
    M[2] = modelviewMatrix[2];
    M[3] = modelviewMatrix[4];
    M[4] = modelviewMatrix[5];
    M[5] = modelviewMatrix[6];
    M[6] = modelviewMatrix[8];
    M[7] = modelviewMatrix[9];
    M[8] = modelviewMatrix[10];

    //Work out the determinate
    float determinate = M[0] * M[4] * M[8] + M[1] * M[5] * M[6] + M[2] * M[3] * M[7];
    determinate -= M[2] * M[4] * M[6] + M[0] * M[5] * M[7] + M[1] * M[3] * M[8];

    //One division is faster than several
    float oneOverDet = 1.0f / determinate;

    
    //Calculate the inverse and assign it to the transpose matrix positions
    N[0] = (M[4] * M[8] - M[5] * M[7]) * oneOverDet;
    N[3] = (M[2] * M[7] - M[1] * M[8]) * oneOverDet;
    N[6] = (M[1] * M[5] - M[2] * M[4]) * oneOverDet;

    N[1] = (M[5] * M[6] - M[3] * M[8]) * oneOverDet;
    N[4] = (M[0] * M[8] - M[2] * M[6]) * oneOverDet;
    N[7] = (M[2] * M[3] - M[0] * M[5]) * oneOverDet;

    N[2] = (M[3] * M[7] - M[4] * M[6]) * oneOverDet;
    N[5] = (M[1] * M[6] - M[8] * M[7]) * oneOverDet;
    N[8] = (M[0] * M[4] - M[1] * M[3]) * oneOverDet;
}



bool ObjMesh::HitPoint(Vector3 &orig, Vector &dir, Vector3 &hit)
{
	bool bHit = false;
	float dis = 99999.9f;
	for(int i = 0; i < cVertices; i+=3)
	{
		Vector3 v1 = m_vertices_sorted[m_indices[i]];
		Vector3 v2 = m_vertices_sorted[m_indices[i+1]];
		Vector3 v3 = m_vertices_sorted[m_indices[i+2]];
		float t;
		if(IntersectTriangle(orig, dir, v1, v2, v3, t))
		{
			bHit = true;
			if(t < dis && t > 0.0f) dis = t;
		}
	}

	hit = orig + dir*dis;

	return bHit;
}

bool ObjMesh::IntersectTriangle( Vector orig, Vector dir, 
                        Vector v0, Vector v1, Vector v2, float &t)
{
    // Find vectors for two edges sharing vert0
    Vector edge1 = v1 - v0;
    Vector edge2 = v2 - v0;

    // Begin calculating determinant - also used to calculate U parameter
    Vector pvec;
	pvec = Vector::CrossProduct(dir, edge2 );

    // If determinant is near zero, ray lies in plane of triangle
    float det = Vector::ScalarProduct(edge1, pvec);
	float u, v;

    Vector tvec;
    if( det > 0 )
    {
        tvec = orig - v0;
    }
    else
    {
        tvec = v0 - orig;
        det = -det;
    }

    if( det < 0.00001f )
        return false;

    // Calculate U parameter and test bounds
    u = Vector::ScalarProduct(tvec, pvec);
    if( u < 0.0f || u > det )
        return false;

    // Prepare to test V parameter
    Vector qvec;
    qvec = Vector::CrossProduct(tvec, edge1 );

    // Calculate V parameter and test bounds
    v = Vector::ScalarProduct( dir, qvec );
    if( v < 0.0f || u + v > det )
        return false;

    // Calculate t, scale parameters, ray intersects triangle
    t = Vector::ScalarProduct(edge2, qvec);
    float fInvDet = 1.0f / det;
    t *= fInvDet;
    u *= fInvDet;
    v *= fInvDet;

    return true;
}