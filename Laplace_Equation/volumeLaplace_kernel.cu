#include <shrUtils.h>
#include <cutil_inline.h>    // includes cuda.h and cuda_runtime_api.h
#include "globals.h"



texture<VolumeType, 3, cudaReadModeElementType> inTex;
texture<VolumeType, 3, cudaReadModeElementType> boundaryTex;



__global__ void
cudaLaplace(VolumeType* g_odata, int imgw, int imgh, int imgd,  
	    float speed)

{
//    int tx = threadIdx.x;			
//    int ty = threadIdx.y;
//    int tz = threadIdx.z;
//    int bw = blockDim.x;			// = 1
//    int bh = blockDim.y;			// = 1
//    int bd = blockDim.z;			// = size.depth
//    int x = blockIdx.x/*bw + tx*/;		
//    int y = blockIdx.y/*bh + ty*/;
//	int z = /*blockIdx.z*bd + */tx;

	int x = blockIdx.x;
	int y = blockIdx.y;
	int z = threadIdx.x;

	VolumeType boundary = tex3D(boundaryTex, x, y,z);
	VolumeType v000 = tex3D(inTex, x, y, z);

	VolumeType result = v000;
	VolumeType v0p0;
	VolumeType vp00;
	VolumeType v0m0;
	VolumeType vm00;
	VolumeType v00p;
	VolumeType v00m;

		// Define values
	if(x!=imgw-1) v0p0 =  tex3D(inTex, x+1, y, z);
	else v0p0 =  v000;
	if(y!=imgh-1) vp00 =  tex3D(inTex, x, y+1, z);
	else vp00 =  v000;
	if(z!=imgd-1) v00p =  tex3D(inTex, x, y, z+1);
	else v00p =  v000;
	if(x!=0) v0m0 =  tex3D(inTex, x-1, y, z);
	else v0m0 =  v000;
	if(y!=0) vm00 =  tex3D(inTex, x, y-1, z);
	else vm00 =  v000;
	if(z!=0) v00m =  tex3D(inTex, x, y, z-1);
	else v00m =  v000;

	__syncthreads();


	float tmp = (v0p0.x + vp00.x + v0m0.x + vm00.x + v00p.x + v00m.x)*speed + (1.0f - 6.0f*speed)*v000.x; 
	result.x = result.y*result.z + tmp*(1.0f-result.z);
	result.w = 1.0f;

	g_odata[z*imgw*imgh+y*imgw+x] = result;
}



extern "C"
void init_boundary_cuda(cudaArray *boundary_array)
{
	cutilSafeCall(cudaBindTextureToArray(boundaryTex, boundary_array));

	struct cudaChannelFormatDesc desc; 
    cutilSafeCall(cudaGetChannelDesc(&desc, boundary_array));
}


extern "C"
void launch_cudaLaplace(dim3 grid, dim3 block, 
		   cudaArray *g_data_array, VolumeType* g_odata, 
		   int imgw, int imgh, int imgd, float speed)
{
	cutilSafeCall(cudaBindTextureToArray(inTex, g_data_array));

	struct cudaChannelFormatDesc desc; 
    cutilSafeCall(cudaGetChannelDesc(&desc, g_data_array));


	cudaLaplace<<< grid, block>>> (g_odata, imgw, imgh, imgd, speed);

	cutilSafeCall(cudaUnbindTexture(boundaryTex));
}