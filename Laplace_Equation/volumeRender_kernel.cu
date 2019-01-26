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

// Simple 3D volume renderer

#ifndef _VOLUMERENDER_KERNEL_CU_
#define _VOLUMERENDER_KERNEL_CU_

#include <cutil_inline.h>    // includes cuda.h and cuda_runtime_api.h
#include <cutil_math.h>
#include "globals.h"

typedef unsigned int  uint;
typedef unsigned char uchar;

cudaArray *d_volumeArray = 0;
cudaArray *d_transferFuncArray;



texture<VolumeType, 3, cudaReadModeElementType> boundaryTex;         // 3D texture
texture<uint, 2, cudaReadModeElementType> depthTex;
texture<uchar4, 2, cudaReadModeElementType> colorTex;

typedef struct {
    float4 m[3];
} float3x4;

__constant__ float3x4 c_invViewMatrix;  // inverse view matrix

struct Ray {
	float3 o;	// origin
	float3 d;	// direction
};

// intersect ray with a box
// http://www.siggraph.org/education/materials/HyperGraph/raytrace/rtinter3.htm

__device__
int intersectBox(Ray r, float3 boxmin, float3 boxmax, float *tnear, float *tfar)
{
    // compute intersection of ray with all six bbox planes
    float3 invR = make_float3(1.0f) / r.d;
    float3 tbot = invR * (boxmin - r.o);
    float3 ttop = invR * (boxmax - r.o);

    // re-order intersections to find smallest and largest on each axis
    float3 tmin = fminf(ttop, tbot);
    float3 tmax = fmaxf(ttop, tbot);

    // find the largest tmin and the smallest tmax
    float largest_tmin = fmaxf(fmaxf(tmin.x, tmin.y), fmaxf(tmin.x, tmin.z));
    float smallest_tmax = fminf(fminf(tmax.x, tmax.y), fminf(tmax.x, tmax.z));

	*tnear = largest_tmin;
	*tfar = smallest_tmax;

	return smallest_tmax > largest_tmin;
}

// transform vector by matrix (no translation)
__device__
float3 mul(const float3x4 &M, const float3 &v)
{
    float3 r;
    r.x = dot(v, make_float3(M.m[0]));
    r.y = dot(v, make_float3(M.m[1]));
    r.z = dot(v, make_float3(M.m[2]));
    return r;
}

// transform vector by matrix with translation
__device__
float4 mul(const float3x4 &M, const float4 &v)
{
    float4 r;
    r.x = dot(v, M.m[0]);
    r.y = dot(v, M.m[1]);
    r.z = dot(v, M.m[2]);
    r.w = 1.0f;
    return r;
}

__device__ uint rgbaFloatToInt(float4 rgba)
{
    rgba.x = __saturatef(rgba.x);   // clamp to [0.0, 1.0]
    rgba.y = __saturatef(rgba.y);
    rgba.z = __saturatef(rgba.z);
    rgba.w = __saturatef(rgba.w);
    return (uint(rgba.w*255)<<24) | (uint(rgba.z*255)<<16) | (uint(rgba.y*255)<<8) | uint(rgba.x*255);
}

__device__ float calcDepth(uint depth)
{
	float ratio = (depth>>8)/(1<<24);
	return NearClipping + (FarClipping-NearClipping)*ratio;
}

__global__ void
d_render(uint *d_output, uint imageW, uint imageH,
         float density, float brightness,
         float min, float max, float blueOpacity)
{
    const int maxSteps = 500;
    const float tstep = 0.01f;
    const float opacityThreshold = 0.95f;
    const float3 boxMin = make_float3(-1.0f, -1.0f, -1.0f);
    const float3 boxMax = make_float3(1.0f, 1.0f, 1.0f);

	uint x = blockIdx.x*blockDim.x + threadIdx.x;
    uint y = blockIdx.y*blockDim.y + threadIdx.y;
    if ((x >= imageW) || (y >= imageH)) return;

    float u = (x / (float) imageW)*2.0f-1.0f;
    float v = (y / (float) imageH)*2.0f-1.0f;

    // calculate eye ray in world space
    Ray eyeRay;
    eyeRay.o = make_float3(mul(c_invViewMatrix, make_float4(0.0f, 0.0f, 0.0f, 1.0f)));
    eyeRay.d = normalize(make_float3(u, v, -2.0f));
    eyeRay.d = mul(c_invViewMatrix, eyeRay.d);

    // find intersection with box
	float tnear, tfar;
	int hit = intersectBox(eyeRay, boxMin, boxMax, &tnear, &tfar);
    if (!hit) return;
	if (tnear < 0.0f) tnear = 0.0f;     // clamp to near plane

    // march along ray from front to back, accumulating color
    float4 sum = make_float4(0.0f);
    float t = tnear;
    float3 pos = eyeRay.o + eyeRay.d*tnear;
    float3 step = eyeRay.d*tstep;

	//float depth = calcDepth(tex2D(depthTex, 0.0f, 0.0f));
	uint depthUchar = tex2D(depthTex, 0.0f, 0.0f);

    for(int i=0; i<maxSteps; i++) {
        float4 source = tex3D(boundaryTex, pos.x*0.5f+0.5f, pos.y*0.5f+0.5f, pos.z*0.5f+0.5f);
		float sample;
		//if(source.x>=min && source.x<=max) sample = source.x + blueOpacity*(1.0f-source.x);
		if(source.x>=min && source.x<=max) sample = abs(1.0f-2.0f*source.x);
		else sample = 0.0f;
      
        // lookup in transfer function texture
        float4 col = make_float4(source.x, 0.0f, 1.0f-source.x, sample);
        col.w *= density;

      
        // pre-multiply alpha
        col.x *= col.w;
        col.y *= col.w;
        col.z *= col.w;
        // "over" operator for front-to-back blending
        sum = sum + col*(1.0f - sum.w);
		//sum = sum*(1.0f - col.w) + col*col.w;

        // exit early if opaque
        if (sum.w > opacityThreshold)
            break;

        t += tstep;
        if (t > tfar) break;

        pos += step;
    }
    sum *= brightness;

    // write output color
    d_output[y*imageW + x] = rgbaFloatToInt(sum);
	//d_output[y*imageW + x] = depthUchar | 0xff000000;
}

extern "C"
void setTextureFilterMode(bool bLinearFilter)
{
    boundaryTex.filterMode = bLinearFilter ? cudaFilterModeLinear : cudaFilterModePoint;
}


extern "C"
void setRenderSource(cudaArray *source)
{
	cutilSafeCall(cudaBindTextureToArray(boundaryTex, source));

	struct cudaChannelFormatDesc desc; 
    cutilSafeCall(cudaGetChannelDesc(&desc, source));
}


extern "C" 
void freeCudaBuffers()
{
    cutilSafeCall(cudaFreeArray(d_volumeArray));
    cutilSafeCall(cudaFreeArray(d_transferFuncArray));
}


extern "C"
void render_kernel(dim3 gridSize, dim3 blockSize, uint *d_output, cudaArray *source,
			/*cudaArray *color_array,*/ cudaArray *depth_array, uint imageW, uint imageH, 
			float density, float brightness, float min, float max, float blueOpacity)
{
	struct cudaChannelFormatDesc desc;

	cutilSafeCall(cudaBindTextureToArray(boundaryTex, source));
	cutilSafeCall(cudaGetChannelDesc(&desc, source));

	boundaryTex.normalized = true;                      // access with normalized texture coordinates
    boundaryTex.filterMode = cudaFilterModeLinear;      // linear interpolation
    boundaryTex.addressMode[0] = cudaAddressModeClamp;  // clamp texture coordinates
    boundaryTex.addressMode[1] = cudaAddressModeClamp;

	cutilSafeCall(cudaBindTextureToArray(depthTex, depth_array));
	cutilSafeCall(cudaGetChannelDesc(&desc, depth_array));

	//cutilSafeCall(cudaBindTextureToArray(colorTex, color_array));
	//cutilSafeCall(cudaGetChannelDesc(&desc, color_array));

	d_render<<<gridSize, blockSize>>>( d_output, imageW, imageH, density, 
										brightness, min, max, blueOpacity);

	cutilSafeCall(cudaUnbindTexture(boundaryTex));
}

extern "C"
void copyInvViewMatrix(float *invViewMatrix, size_t sizeofMatrix)
{
    cutilSafeCall( cudaMemcpyToSymbol(c_invViewMatrix, invViewMatrix, sizeofMatrix) );
}


#endif // #ifndef _VOLUMERENDER_KERNEL_CU_
