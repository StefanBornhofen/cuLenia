#pragma once

#include <GL/glew.h>
#include <GLFW/glfw3.h>

#include <cuda_runtime.h>
#include <cuda_gl_interop.h>
#include <cufft.h>
#include <cufftXt.h>
#include <cuda_bf16.h>

#define MAX_CHANNELS 10
#define MAX_RULES 20
#define MAX_RANK 5

typedef struct {
    int source; // source channel index
    int destination; // destination channel index
    int radius; // kernel size

    // FP32 single precision 
    float mu; // growth center
    float sigma2; // growth width (2* and squared already)
    float weight; // kernel weight (the sum of all weights for a given destination channel should be 1.f)
    // FP64 double precision 
    double mu64;
    double sigma264;
    double weight64;
    // FP16 precision (bfloat)
    __nv_bfloat162 mu16;
    __nv_bfloat162 sigma216;
    __nv_bfloat162 weight16;
} Rule;

typedef struct {
    int nbChannels;
    int nbRules;
    int size;
    int T; // simulation time step
    bool toGridB;
    float threshold[MAX_CHANNELS];
    float4 color[MAX_CHANNELS];
    Rule rule[MAX_RULES];

    // FP32 single precision 
    float* gridA[MAX_CHANNELS];
    float* gridB[MAX_CHANNELS];
    float* kernel[MAX_RULES];
    // FP64 double precision 
    double* gridA64[MAX_CHANNELS];
    double* gridB64[MAX_CHANNELS];
    double* kernel64[MAX_RULES];
    // FP16 precision (bfloat)
    __nv_bfloat16* gridA16[MAX_CHANNELS];
    __nv_bfloat16* gridB16[MAX_CHANNELS];
    __nv_bfloat16* kernel16[MAX_RULES];
} LeniaWorld;

float* randomGrid(int size, float threshold);
float4* zeroRenderGrid(int size);
float* initKernel(int R, float relR, int rank, float alpha, float beta[MAX_RANK]);
__host__ __device__ void computeGrowth(int i, int j, float mu, float sigma2, float weight, float u_t, float* cell);
__host__ __device__ void computeLeniaStep(int i, int j, int radius, float mu, float sigma2, float weight, float* kernel, float* p1, int size, float* cell);
extern __constant__ Rule constantRule[MAX_RULES];
extern __constant__ float4 constantColor[MAX_CHANNELS];

#define checkCudaErrors(err) __checkCudaErrors(err, __FILE__, __LINE__)
inline void __checkCudaErrors(cudaError err, const char* file, const int line)
{
    if (err != cudaSuccess)
    {
        fprintf(stderr, "%s(%i) : CUDA Runtime API error %d: %s.\n",
            file, line, (int)err, cudaGetErrorString(err));
        system("pause");
        exit(1);
    }
}

