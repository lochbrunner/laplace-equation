/*
* Copyright 1993-2010 NVIDIA Corporation.  All rights reserved.
*
* Please refer to the NVIDIA end user license agreement (EULA) associated
* with this source code for terms and conditions that govern your use of
* this software. Any use, reproduction, disclosure, or distribution of
* this software and related documentation outside the terms of the EULA
* is strictly prohibited.
*
*/

/*
Volume rendering sample

This sample loads a 3D volume from disk and displays it using
ray marching and 3D textures.

Note - this is intended to be an example of using 3D textures
in CUDA, not an optimized volume renderer.

Changes
sgg 22/3/2010
- updated to use texture for display instead of glDrawPixels.
- changed to render from front-to-back rather than back-to-front.
*/

// OpenGL Graphics includes
#include <GL/glew.h>
#if defined (__APPLE__) || defined(MACOSX)
#include <GLUT/glut.h>
#else
#include <GL/freeglut.h>
#endif

// CUDA utilities and system includes
#include <cutil_inline.h>    // includes cuda.h and cuda_runtime_api.h
#include <cutil_gl_inline.h> // includes cuda_gl_interop.h// includes cuda_gl_interop.h
#include <rendercheck_gl.h>

// CUDA Includes
#include <vector_types.h>
#include <vector_functions.h>
#include <driver_functions.h>

// Shared Library Test Functions
#include <shrUtils.h>

#include "globals.h"
#include "SceneLoader.h"

typedef unsigned int uint;
typedef unsigned char uchar;

#define MAX_EPSILON_ERROR 5.00f
#define THRESHOLD         0.30f


//cudaExtent volumeSize = make_cudaExtent(64, 64, 64);
cudaExtent volumeSize = make_cudaExtent(128, 128, 128);	//Absolute maximum: 2048 x 2048 x 1024


const uint width = 512, height = 512;
uint window_width = 512, window_height = 512;
dim3 blockSize(16, 16);
dim3 gridSize;

float3 viewRotation;
float3 viewTranslation = make_float3(0.0, 0.0, -4.0f);
float invViewMatrix[12];

float density = 0.05f;
float brightness = 1.0f;
bool linearFiltering = true;

GLuint pbo = 0;     // OpenGL pixel buffer object
GLuint tex = 0;     // OpenGL texture object
struct cudaGraphicsResource *cuda_pbo_resource; // CUDA Graphics Resource (to transfer PBO)

unsigned int timer = 0;

// Auto-Verification Code
const int frameCheckNumber = 2;
int fpsCount = 0;        // FPS count for averaging
int fpsLimit = 1;        // FPS limit for sampling
int g_Index = 0;
unsigned int frameCount = 0;
unsigned int g_TotalErrors = 0;


ObjMesh *pScene = NULL;
SceneLoader Scene;
GLfloat perspectivView[16];
GLuint depthBuffer = 0;
GLuint colorBuffer = 0;
GLuint depthTexture = 0;
GLuint fbo = 0;

bool bShowScene = true;
bool bShowField = true;

struct cudaGraphicsResource *cuda_depthBuffer;
struct cudaGraphicsResource *cuda_colorBuffer;

GLuint opengl_tex = 0;
struct cudaGraphicsResource *cuda_tex = NULL;
float4* cuda_dest_resource = NULL;
float fAnimation = 0.0f;
float fThickness = 0.08f;
bool bAnimation = false;
bool bShowSlice = false;
float g_blueOpacity = 0.0f;

// CheckFBO/BackBuffer class objects


#define MAX(a,b) ((a > b) ? a : b)

extern "C" void setTextureFilterMode(bool bLinearFilter);
extern "C" void freeCudaBuffers();
extern "C" void render_kernel(dim3 gridSize, dim3 blockSize, uint *d_output, cudaArray *source, /*cudaArray *color_array,*/ cudaArray *depth_array, uint imageW, uint imageH, 
	float density, float brightness, float min, float max, float blueOpacity);
extern "C" void copyInvViewMatrix(float *invViewMatrix, size_t sizeofMatrix);




extern "C" void setRenderSource(cudaArray *source);
//extern "C" void init_boundary_cuda(cudaArray *boundary_array);

extern "C" void launch_cudaLaplace(dim3 grid, dim3 block, cudaArray *g_data_array, VolumeType* g_odata, 
	int imgw, int imgh, int imgd, float speed);

void initPixelBuffer();

void createPBO(/*GLuint* pbo, struct cudaGraphicsResource **pbo_resource*/)
{
	// set up vertex data parameter
	int num_texels = width * height;
	int num_values = num_texels;
	int size_tex_data = sizeof(GLfloat) * num_values;
	void *data = malloc(size_tex_data);

	// create buffer object
	glGenBuffers(1, &depthBuffer);
	glBindBuffer(GL_ARRAY_BUFFER, depthBuffer);
	glBufferData(GL_ARRAY_BUFFER, size_tex_data, data, GL_DYNAMIC_DRAW);
	free(data);

	glBindBuffer(GL_ARRAY_BUFFER, 0);

	// register this buffer object with CUDA
	cutilSafeCall(cudaGraphicsGLRegisterBuffer(&cuda_depthBuffer, depthBuffer, cudaGraphicsMapFlagsNone));
}

void createTexture(GLuint* pTexture, cudaGraphicsResource **ppCudaResource, cudaExtent size, GLvoid* pixels = NULL)
{
	// create a texture
	glGenTextures(1, pTexture);
	glBindTexture(GL_TEXTURE_3D, *pTexture);

	// set basic parameters
	glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
	glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
	glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_WRAP_R, GL_CLAMP_TO_EDGE);
	glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
	glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);

	glTexImage3D(GL_TEXTURE_3D, 0, GL_RGBA32F_ARB, size.width, size.height, size.depth, 0, GL_RGBA, GL_FLOAT, pixels);
	// register this texture with CUDA
	cutilSafeCall(cudaGraphicsGLRegisterImage(ppCudaResource, *pTexture, 
		GL_TEXTURE_3D, cudaGraphicsMapFlagsReadOnly));
}
void deleteTexuture(GLuint* pTexture, cudaGraphicsResource **ppCudaResource)
{
	if(pTexture)
		glDeleteTextures(1,pTexture);
	if(*ppCudaResource)
		cutilSafeCall(cudaGraphicsUnregisterResource(*ppCudaResource));
}
void initCUDABuffers(cudaExtent size)
{
	// set up vertex data parameter
	int num_texels = size.width * size.height * size.depth;
	int num_values = num_texels;
	int size_tex_data = sizeof(VolumeType) * num_values;
	cutilSafeCall(cudaMalloc((void**)&cuda_dest_resource, size_tex_data));
}
void deleteCUDABuffers()
{
	cudaFree(cuda_dest_resource);
}
void copyBufferFormCudaResourceToTexture(cudaExtent size)
{
	cudaArray *texture_ptr;
	cutilSafeCall(cudaGraphicsMapResources(1, &cuda_tex, 0));
	cutilSafeCall(cudaGraphicsSubResourceGetMappedArray(&texture_ptr, cuda_tex, 0, 0));

	int num_texels = size.width * size.height * size.depth;
	int num_values = num_texels;
	int size_tex_data = sizeof(VolumeType) * num_values;
	cudaMemcpy3DParms copyParams = {0};
	copyParams.srcPtr   = make_cudaPitchedPtr((void*)cuda_dest_resource, size.width*sizeof(VolumeType), size.height, size.depth);
	copyParams.dstArray = texture_ptr;
	copyParams.extent   = size;
	copyParams.kind     = cudaMemcpyDeviceToDevice;
	cutilSafeCall(cudaMemcpy3D(&copyParams));

	cutilSafeCall(cudaGraphicsUnmapResources(1, &cuda_tex, 0));
}
void processCuda(cudaExtent size, float speed) 
{
	cudaArray *in_array; 
	VolumeType* out_data;

	out_data = cuda_dest_resource;

	// map buffer objects to get CUDA device pointers
	cutilSafeCall(cudaGraphicsMapResources(1, &cuda_tex, 0));
	cutilSafeCall(cudaGraphicsSubResourceGetMappedArray(&in_array, cuda_tex, 0, 0));
	//int sbytes = block.x*(block.y)*sizeof(unsigned int);

	// calculate grid size
	dim3 block(size.depth, 1, 1);					//Max:	1024x1024x64
	dim3 grid(size.width, size.height, 1);			//Max:	65535x65535x?65535?
	
	// execute CUDA kernel
	launch_cudaLaplace(grid, block, in_array, out_data,
		size.width, size.height, size.depth, speed);

	cutilSafeCall(cudaGraphicsUnmapResources(1, &cuda_tex, 0));
}


void createDepthBuffer()
{
	glGenFramebuffersEXT(1, &fbo);
	glBindFramebufferEXT(GL_FRAMEBUFFER_EXT, fbo);
	glGenRenderbuffersEXT(1, &depthBuffer);
	glBindRenderbufferEXT(GL_RENDERBUFFER_EXT, depthBuffer);
	glRenderbufferStorageEXT(GL_RENDERBUFFER_EXT, GL_DEPTH_COMPONENT, width, height);
	glFramebufferRenderbufferEXT(GL_FRAMEBUFFER_EXT, GL_DEPTH_ATTACHMENT_EXT,
		GL_RENDERBUFFER_EXT, depthBuffer);
	glGenTextures(1, &colorBuffer);
	glBindTexture(GL_TEXTURE_2D, colorBuffer);
	glTexParameteri(GL_TEXTURE_2D,GL_TEXTURE_MAG_FILTER, GL_NEAREST);
	glTexParameteri(GL_TEXTURE_2D,GL_TEXTURE_MIN_FILTER, GL_NEAREST);
	glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8,  width, height, 0, GL_RGBA, GL_UNSIGNED_BYTE, NULL);
	glFramebufferTexture2DEXT(GL_FRAMEBUFFER_EXT, GL_COLOR_ATTACHMENT0_EXT, 	GL_TEXTURE_2D, colorBuffer, 0);

//
//	glutReportErrors(); 
//
//	//glGenTextures(1, &depthTexture);
// //   glBindTexture(GL_TEXTURE_2D, depthTexture);
//
// //   // set basic parameters
// //   glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
// //   glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
// //   glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
// //   glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
//
//	//glutReportErrors(); 
//
//	//glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB32F_ARB, width, height, 0, GL_RGB, GL_FLOAT, 0);
//
//	glutReportErrors(); 
//
//	cutilSafeCall(cudaGraphicsGLRegisterImage(&cuda_depthBuffer, depthBuffer, 
//		GL_TEXTURE_2D, cudaGraphicsMapFlagsReadOnly));
//
//	cutilSafeCall(cudaGraphicsGLRegisterImage(&cuda_colorBuffer, colorBuffer, 
//		GL_TEXTURE_2D, cudaGraphicsMapFlagsReadOnly));
//
//	glBindFramebufferEXT(GL_FRAMEBUFFER_EXT, 0);
//
//
//	GLint bits;
//	glGetIntegerv(GL_DEPTH_BITS, &bits);
//
//	printf("Bit of depthbuffer %d\n", bits); 
//
//	
//	//glGenTextures(1, &depthBuffer);
//
//	//glBindTexture(GL_TEXTURE_2D, depthBuffer);
//	//glTexImage2D(GL_TEXTURE_2D, 0, GL_DEPTH_COMPONENT32, width, height, 0,  
//	//			 GL_DEPTH_COMPONENT, GL_UNSIGNED_INT, 0);
//	//glBindTexture(GL_TEXTURE_2D, 0);
//
//	//glGenFramebuffersEXT(1, &m_fbo);
//	//glBindFramebufferEXT( GL_FRAMEBUFFER, m_fbo);
//
//	//glFramebufferTexture2DEXT(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT,
//	//					   GL_TEXTURE_2D, depthBuffer, 0 );
//	//cudaError_t b;
//
//	//
	cutilSafeCall(cudaGraphicsGLRegisterImage(&cuda_depthBuffer, depthBuffer,
		GL_TEXTURE_2D, cudaGraphicsMapFlagsReadOnly));
	cutilSafeCall(cudaGraphicsGLRegisterImage(&cuda_colorBuffer, colorBuffer,
		GL_TEXTURE_2D, cudaGraphicsMapFlagsReadOnly));

	glBindFramebufferEXT(GL_FRAMEBUFFER_EXT, 0);

//	//printf("%s",  cudaGetErrorString(b));
//
}

void computeFPS()
{
	frameCount++;
	fpsCount++;
	if (fpsCount == fpsLimit) {
		char fps[256];
		float ifps = 1.f / (cutGetAverageTimerValue(timer) / 1000.f);
		sprintf(fps, "Laplace Equation: %3.1f fps", ifps);  

		glutSetWindowTitle(fps);
		fpsCount = 0; 

		cutilCheckError(cutResetTimer(timer));  
	}
}

// render image using CUDA
void render()
{

	float min, max;
	if(bShowSlice)
	{
		min = fAnimation - fThickness;
		max = fAnimation + fThickness;
	}
	else 
	{
		min = 0.0f;
		max = 1.0f;
	}
	cudaArray *depth_array;
	cudaArray *source_array;
	//void depth_array;
	//cudaArray *color_array;

	//cutilSafeCall(cudaGraphicsMapResources(1, &cuda_colorBuffer, 0));
	//printf("Mapping tex_in\n");
	//cutilSafeCall(cudaGraphicsSubResourceGetMappedArray(&color_array, cuda_colorBuffer, 0, 0));

	cutilSafeCall(cudaGraphicsMapResources(1, &cuda_depthBuffer, 0));
	//printf("Mapping tex_in\n");
	cutilSafeCall(cudaGraphicsSubResourceGetMappedArray(&depth_array, cuda_depthBuffer, 0, 0));


	//glReadBuffer(GL_DEPTH_COMPONENT);
	//glBindBuffer(GL_PIXEL_PACK_BUFFER, depthBuffer);
	//glReadPixels(0, 0, width, height, GL_DEPTH_COMPONENT, GL_FLOAT, 0); // NOTE: in linux GL_RGBA is necessary for fast pixel reads, in vista, both formats are slow
	//glBindBuffer(GL_PIXEL_UNPACK_BUFFER, depthBuffer);

	//cutilSafeCall(cudaGLMapBufferObject((void**)&depth_array, depthBuffer));

	//size_t num_bytes_depth;
	//cutilSafeCall(cudaGraphicsMapResources(1, &cuda_depthBuffer, 0));
	//cutilSafeCall(cudaGraphicsResourceGetMappedPointer((void **)&depth_array,  &num_bytes_depth, cuda_depthBuffer));

	cutilSafeCall(cudaGraphicsMapResources(1, &cuda_tex, 0));
	cutilSafeCall(cudaGraphicsSubResourceGetMappedArray(&source_array,  cuda_tex, 0, 0));


	copyInvViewMatrix(invViewMatrix, sizeof(float4)*3);

	// map PBO to get CUDA device pointer
	uint *d_output;
	// map PBO to get CUDA device pointer
	cutilSafeCall(cudaGraphicsMapResources(1, &cuda_pbo_resource, 0));
	size_t num_bytes; 
	cutilSafeCall(cudaGraphicsResourceGetMappedPointer((void **)&d_output, &num_bytes,  
		cuda_pbo_resource));
	//printf("CUDA mapped PBO: May access %ld bytes\n", num_bytes);

	// clear image
	cutilSafeCall(cudaMemset(d_output, 0, width*height*4));

	// call CUDA kernel, writing results to PBO
	render_kernel(gridSize, blockSize, d_output, source_array, /*color_array, */depth_array, width, height, density, brightness, min, max, g_blueOpacity);

	cutilCheckMsg("kernel failed");

	cutilSafeCall(cudaGraphicsUnmapResources(1, &cuda_pbo_resource, 0));

	cutilSafeCall(cudaGraphicsUnmapResources(1, &cuda_depthBuffer, 0));
	cutilSafeCall(cudaGraphicsUnmapResources(1, &cuda_tex, 0));
	//cutilSafeCall(cudaGraphicsUnmapResources(1, &cuda_colorBuffer, 0));
}

// display results using OpenGL (called by GLUT)
void display()
{
	cutilCheckError(cutStartTimer(timer));  

	// Calculate inverse Modelview Matrix
	GLfloat modelView[16];
	glMatrixMode(GL_MODELVIEW);
	glPushMatrix();
	glLoadIdentity();
	glRotatef(-viewRotation.y, 0.0, 1.0, 0.0);
	glRotatef(-viewRotation.x, 1.0, 0.0, 0.0);
	glTranslatef(viewTranslation.x, viewTranslation.y, viewTranslation.z);
	glGetFloatv(GL_MODELVIEW_MATRIX, modelView);
	glPopMatrix();

	invViewMatrix[0] = modelView[0]; invViewMatrix[1] = modelView[4]; invViewMatrix[2] = modelView[8]; invViewMatrix[3] = -modelView[12];
	invViewMatrix[4] = modelView[1]; invViewMatrix[5] = modelView[5]; invViewMatrix[6] = modelView[9]; invViewMatrix[7] = -modelView[13];
	invViewMatrix[8] = modelView[2]; invViewMatrix[9] = modelView[6]; invViewMatrix[10] = modelView[10]; invViewMatrix[11] = -modelView[14];

	// Calculate normal Modelview Matrix
	glMatrixMode(GL_MODELVIEW);
	glPushMatrix();
	glLoadIdentity();
	glTranslatef(viewTranslation.x, viewTranslation.y, viewTranslation.z);
	glRotatef(viewRotation.x, 1.0, 0.0, 0.0);
	glRotatef(viewRotation.y, 0.0, 1.0, 0.0);
	glGetFloatv(GL_MODELVIEW_MATRIX, modelView);
	glPopMatrix();


	//if(bShowScene)
	//{
	//	glViewport(0, 0, width, height);
	//	glBindRenderbufferEXT(GL_RENDERBUFFER_EXT, depthBuffer);
	//	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

	//	Scene.Render(modelView, perspectivView);

	//	glViewport(MAX(0,(window_width-window_height)/2), MAX(0,(window_height-window_width)/2),
	//		MIN(window_width,window_height), MIN(window_width,window_height));
	//	glBindRenderbufferEXT(GL_RENDERBUFFER_EXT, 0);

	//	if(bShowSlice&&bAnimation)
	//	{
	//		fAnimation -= 0.008f;
	//		if(fAnimation<0.0) fAnimation = 1.0f;
	//	}
	//}
	if(bShowSlice&&bAnimation)
	{
		fAnimation -= 0.008f;
		if(fAnimation<0.0) fAnimation = 1.0f;
	}
	if(bShowField) render();

	// display results
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);



	if(bShowScene) Scene.Render(modelView, perspectivView);
	
	/*GLint viewport[4]; 
	glGetIntegerv(GL_VIEWPORT, viewport); 

	GLfloat* zbuffer_float = new GLfloat[viewport[2] * viewport[3]]; 

	glPixelStorei(GL_PACK_ALIGNMENT, 1); 
	glReadPixels(0, 0, viewport[2], viewport[3], GL_DEPTH_COMPONENT, GL_FLOAT, zbuffer_float);


	delete []zbuffer_float;*/

	if(bShowField)
	{
		// draw image from PBO
		glDisable(GL_DEPTH_TEST);

		glPixelStorei(GL_UNPACK_ALIGNMENT, 1);
		// draw using texture

		// copy from pbo to texture
		glBindBufferARB(GL_PIXEL_UNPACK_BUFFER_ARB, pbo);
		glBindTexture(GL_TEXTURE_2D, tex);
		glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, width, height, GL_RGBA, GL_UNSIGNED_BYTE, 0);
		glBindBufferARB(GL_PIXEL_UNPACK_BUFFER_ARB, 0);


		glEnable(GL_BLEND);
		glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

		// draw textured quad
		glEnable(GL_TEXTURE_2D);
		glBegin(GL_QUADS);
		glTexCoord2f(0, 0); glVertex2f(0, 0);
		glTexCoord2f(1, 0); glVertex2f(1, 0);
		glTexCoord2f(1, 1); glVertex2f(1, 1);
		glTexCoord2f(0, 1); glVertex2f(0, 1);
		glEnd();

		glDisable(GL_TEXTURE_2D);
		glBindTexture(GL_TEXTURE_2D, 0);
	}


	glutSwapBuffers();

	glutReportErrors(); 
	cutilCheckError(cutStopTimer(timer));  

	computeFPS();
}

void idle()
{
	glutPostRedisplay();
}

void keyboard(unsigned char key, int x, int y)
{
	switch(key) {
	case 27:		// Esc
		exit(0);
		break;
	case 'l':
		linearFiltering = !linearFiltering;
		setTextureFilterMode(linearFiltering);
		break;
	case '+':
		density += 0.01f;
		break;
	case '-':
		density -= 0.01f;
		break;
	case ']':
		brightness += 0.1f;
		break;
	case '[':
		brightness -= 0.1f;
		break;
	case 'o':
		bShowScene ^= true;
		break;
	case 'f':
		bShowField ^= true;
		if(bShowField) fAnimation = 1.0f;
		break;
	case 'p':
		for(int i = 0; i < 200; i++)
		{
			processCuda(volumeSize, 0.15f);
			copyBufferFormCudaResourceToTexture(volumeSize);
		}
		break;
	case 'a':
		bAnimation ^= true;
		break;
	case 's':
		bShowSlice ^= true;
		if(bShowSlice) fAnimation = 1.0f;
		break;
	case '.':
		fThickness += 0.005f;
		break;
	case ',':
		fThickness -= 0.005f;
		fThickness = MAX(fThickness, 0.0f);
		break;
	case 'n':
		g_blueOpacity += 0.05f;
		g_blueOpacity = MIN(g_blueOpacity, 1.0f);
		break;
	case 'm':
		g_blueOpacity -= 0.05f;
		g_blueOpacity = MAX(g_blueOpacity, 0.0f);
		break;
		

	default:
		break;
	}
	shrLog("density = %.2f, brightness = %.2f, thickness = %.2f, blueOpacity = %.2f\n", density, brightness, fThickness, g_blueOpacity);
	glutPostRedisplay();
}

int ox, oy;
int buttonState = 0;

void mouse(int button, int state, int x, int y)
{
	if (state == GLUT_DOWN)
		buttonState  |= 1<<button;
	else if (state == GLUT_UP)
		buttonState = 0;

	ox = x; oy = y;
	glutPostRedisplay();
}

void motion(int x, int y)
{
	float dx, dy;
	dx = (float)(x - ox);
	dy = (float)(y - oy);

	if(glutGetModifiers() == GLUT_ACTIVE_CTRL)
	{
		fAnimation += static_cast<float>(dy+dx)*0.002f;
		fAnimation = MIN(1.0f, MAX(fAnimation, 0.0f));
	}
	else {
		if (buttonState == 4) {
			// right = zoom
			viewTranslation.z += dy / 100.0f;
		} 
		else if (buttonState == 2) {
			// middle = translate
			viewTranslation.x += dx / 100.0f;
			viewTranslation.y -= dy / 100.0f;
		}
		else if (buttonState == 1) {
			// left = rotate
			viewRotation.x += dy / 5.0f;
			viewRotation.y += dx / 5.0f;
		}
	}

	ox = x; oy = y;
	glutPostRedisplay();
}

int iDivUp(int a, int b){
	return (a % b != 0) ? (a / b + 1) : (a / b);
}

void reshape(int w, int h)
{
	window_width = w;
	window_height = h;
	initPixelBuffer();

	// calculate new grid size
	gridSize = dim3(iDivUp(width, blockSize.x), iDivUp(height, blockSize.y));

	glViewport(MAX(0,(window_width-window_height)/2), MAX(0,(window_height-window_width)/2),
		MIN(window_width,window_height), MIN(window_width,window_height));

	glMatrixMode(GL_MODELVIEW);
	glLoadIdentity();

	glMatrixMode(GL_PROJECTION);
	glLoadIdentity();
	glOrtho(0.0, 1.0, 0.0, 1.0, 0.0, 1.0);
}

void cleanup()
{
	cutilCheckError( cutDeleteTimer( timer));

	freeCudaBuffers();

	if (pbo) {
		cudaGraphicsUnregisterResource(cuda_pbo_resource);
		glDeleteBuffersARB(1, &pbo);
		glDeleteTextures(1, &tex);
	}

	glDeleteRenderbuffersEXT(1, &depthBuffer);

	deleteTexuture(&opengl_tex, &cuda_tex);
	deleteCUDABuffers();

}

void initGL(int *argc, char **argv)
{
	// initialize GLUT callback functions
	glutInitDisplayString("depth=32");
	//glutInitDisplayString("stencil~2 rgb double depth>=32 samples");
	glutInit(argc, argv);
	glutInitDisplayMode(GLUT_RGB | GLUT_DOUBLE | GLUT_DEPTH);
	glutInitWindowSize(width, height);
	glutCreateWindow("CUDA volume rendering");
	glewInit();
	if (!glewIsSupported("GL_VERSION_2_0 GL_ARB_pixel_buffer_object")) {
		printf("Required OpenGL extensions missing.");
		exit(-1);
	}

	GLint depthBits, stencilBits;
	glGetIntegerv(GL_DEPTH_BITS, &depthBits);
	glGetIntegerv(GL_STENCIL_BITS, &stencilBits);
	

	printf("Bit of depthbuffer %d Bits and of stencilbuffer %d\n", depthBits,stencilBits); 

	glMatrixMode(GL_PROJECTION);
	glLoadIdentity();
	gluPerspective(53.50, 1.0f, NearClipping, FarClipping);
	glGetFloatv(GL_PROJECTION_MATRIX, perspectivView);

	glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);
}

void initPixelBuffer()
{
	if (pbo) {
		// unregister this buffer object from CUDA C
		cutilSafeCall(cudaGraphicsUnregisterResource(cuda_pbo_resource));

		// delete old buffer
		glDeleteBuffersARB(1, &pbo);
		glDeleteTextures(1, &tex);
	}

	// create pixel buffer object for display
	glGenBuffersARB(1, &pbo);
	glBindBufferARB(GL_PIXEL_UNPACK_BUFFER_ARB, pbo);
	glBufferDataARB(GL_PIXEL_UNPACK_BUFFER_ARB, width*height*sizeof(GLubyte)*4, 0, GL_STREAM_DRAW_ARB);
	glBindBufferARB(GL_PIXEL_UNPACK_BUFFER_ARB, 0);

	// register this buffer object with CUDA
	cutilSafeCall(cudaGraphicsGLRegisterBuffer(&cuda_pbo_resource, pbo, cudaGraphicsMapFlagsWriteDiscard));	

	// create texture for display
	glGenTextures(1, &tex);
	glBindTexture(GL_TEXTURE_2D, tex);
	glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, width, height, 0, GL_RGBA, GL_UNSIGNED_BYTE, NULL);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
	glBindTexture(GL_TEXTURE_2D, 0);
}


VolumeType *loadScene(std::string &filename)
{
	Scene.LoadScene(filename);
	return Scene.LoadField(volumeSize);



	//if(pVolumeInitData) delete[] pVolumeInitData;

	//pVolumeInitData = new VolumeType[volumeSize.width*volumeSize.height*volumeSize.depth];

	//Vector3 dir = Vector3(1.0f, 0.0f, 0.0f);
	//Vector3 hit;

	//float ddw = 2.0f/static_cast<float>(volumeSize.width);
	//float ddh = 2.0f/static_cast<float>(volumeSize.height);
	//float ddd = 2.0f/static_cast<float>(volumeSize.depth);

	//float swf = -0.5f*ddw*static_cast<float>(volumeSize.width);
	//float shf = -0.5f*ddh*static_cast<float>(volumeSize.height);
	//float sdf = -0.5f*ddd*static_cast<float>(volumeSize.depth);
	//
	////bool isAlreadyGeomety[volumeSize.width*volumeSize.height*volumeSize.depth];
	////for(int i = 0; i < volumeSize.width*volumeSize.height*volumeSize.depth; i++) isAlreadyGeomety[i] = false;
	//bool bFirst = true;
	//for(auto it = Scene.m_ObjectList.begin(); it != Scene.m_ObjectList.end(); ++it){
	//	float cwf = swf;
	//	for(unsigned int w = 0; w < volumeSize.width; w++){
	//		float chf = shf;
	//		for(unsigned int h = 0; h < volumeSize.height; h++){
	//			float cdf = sdf;
	//			bool bInGeomatry = false;
	//			for(unsigned int d = 0; d < volumeSize.depth; d++){

	//				Vector3 orig = Vector3(cdf-ddd*0.5f, chf, cwf);

	//				if(it->pMesh->HitPoint(orig, dir, hit))
	//				{
	//					if(hit.x < cdf+ddd*0.5f)
	//					{
	//						bInGeomatry ^= true;
	//					}
	//				}

	//				int index = (w*volumeSize.height+h)*volumeSize.depth+d;
	//				//isAlreadyGeomety[index] = isAlreadyGeomety[index]  || bInGeomatry;
	//				if(bFirst)
	//				{
	//					pVolumeInitData[index].x = bInGeomatry ?  it->value : 0.0f;	// Value
	//					pVolumeInitData[index].y = bInGeomatry ?  1.0f : 0.0f;		// boundary Value
	//					pVolumeInitData[index].z = bInGeomatry ?  1.0f : 0.0f;		// Boundary condition
	//					pVolumeInitData[index].w = 0.0f;							// Not in use
	//				}
	//				else if(bInGeomatry)
	//				{
	//					pVolumeInitData[index].x = it->value;		// Value
	//					pVolumeInitData[index].y = it->value;		// boundary Value
	//					pVolumeInitData[index].z = 1.0f;			// Boundary condition
	//					pVolumeInitData[index].w = 0.0f;			// Not in use
	//				}


	//				//pVolumeInitData[(w*volumeSize.height+h)*volumeSize.depth+d] = static_cast<VolumeType>(powf(1.3f, -(w2+h2+d2)/2.0f))*255;
	//				//if((w*2 - volumeSize.width==0) || (h*2 - volumeSize.height==0) || (d*2 - volumeSize.depth==0)) pVolumeInitData[(w*volumeSize.height+h)*volumeSize.depth+d] = 255;
	//				//if(((w-h+d)*2-volumeSize.width ==0)||((w-h+d)*2-volumeSize.width ==2))pVolumeInitData[(w*volumeSize.height+h)*volumeSize.depth+d] = 255;
	//				//if((w==volumeSize.width-1)||(w==0)||((h)*2-volumeSize.height ==0))pVolumeInitData[(w*volumeSize.height+h)*volumeSize.depth+d] = 1.0f;
	//				//else pVolumeInitData[(w*volumeSize.height+h)*volumeSize.depth+d] = 0.0f;
	//				//pVolumeInitData[(w*volumeSize.height+h)*volumeSize.depth+d] = w*256/volumeSize.width;
	//				cdf+=ddd;
	//			}
	//			chf+=ddh;
	//		}
	//		cwf+=ddw;
	//	}
	//	bFirst = false;
	//}



	//return pVolumeInitData;
}

////////////////////////////////////////////////////////////////////////////////
// Program main
////////////////////////////////////////////////////////////////////////////////
int main( int argc, char** argv) 
{
	// First initialize OpenGL context, so we can properly set the GL for CUDA.
	// This is necessary in order to achieve optimal performance with OpenGL/CUDA interop.
	initGL( &argc, argv );


	// use command-line specified CUDA device, otherwise use device with highest Gflops/s
	cutilChooseCudaGLDevice(argc, argv);

	VolumeType *h_volume = loadScene(std::string("../scenes/scene_8.set"));

	createTexture(&opengl_tex, &cuda_tex, volumeSize, h_volume);
	initCUDABuffers(volumeSize);

	createPBO(/*&depthBuffer, &cuda_depthBuffer*/);

	cutilCheckError( cutCreateTimer( &timer));

	printf("Press '+' and '-' to change density (0.01 increments)\n"
		"      ']' and '[' to change brightness\n");

	// calculate new grid size
	gridSize = dim3(iDivUp(width, blockSize.x), iDivUp(height, blockSize.y));

	createDepthBuffer();

	// This is the normal rendering path for VolumeRender
	glutDisplayFunc(display);
	glutKeyboardFunc(keyboard);
	glutMouseFunc(mouse);
	glutMotionFunc(motion);
	glutReshapeFunc(reshape);
	glutIdleFunc(idle);

	initPixelBuffer();

	atexit(cleanup);

	glutMainLoop();


	cutilDeviceReset();
}
