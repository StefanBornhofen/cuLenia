#include <stdio.h>
#include <stdlib.h>

#include "Device32noInteropMode.h"
#include "HostMode.h"
using namespace cuLenia;

__global__ void leniaKernelRendernoInterop(LeniaWorld lw, float4* cuPixels)
{
    int i = threadIdx.y + blockIdx.y * blockDim.y;
    int j = threadIdx.x + blockIdx.x * blockDim.x;

    if (i < lw.size && j < lw.size)
    {
        int index = i * lw.size + j;
        // apply rules
        for (int k = 0; k < lw.nbRules; k++)
        {
            Rule r = lw.rule[k];
            float* g1, * g2;
            if (lw.toGridB) {
                g1 = lw.gridA[r.source];
                g2 = lw.gridB[r.destination];
            }
            else {
                g1 = lw.gridB[r.source];
                g2 = lw.gridA[r.destination];
            }
            computeLeniaStep(i, j, r.radius, r.mu, r.sigma2, r.weight, lw.kernel[k], g1, lw.size, g2 + index);
        }
        // prepare renderGrid (rendering is nearly no-time)
        cuPixels[index].x = 0;
        cuPixels[index].y = 0;
        cuPixels[index].z = 0;
        cuPixels[index].w = 1;
        // finalize and report channels to rendergrid
        for (int k = 0; k < lw.nbChannels; k++)
        {
            float* gridA, * gridB;
            if (lw.toGridB) {
                gridA = lw.gridA[k]; gridB = lw.gridB[k];
            }
            else {
                gridA = lw.gridB[k]; gridB = lw.gridA[k];
            }
            gridB[index] /= lw.T;
            gridB[index] += gridA[index];
            if (gridB[index] < 0.0f) gridB[index] = 0.0f;
            if (gridB[index] > 1.0f) gridB[index] = 1.0f;
            cuPixels[index].x += gridB[index] * lw.color[k].x;
            cuPixels[index].y += gridB[index] * lw.color[k].y;
            cuPixels[index].z += gridB[index] * lw.color[k].z;
        }
    }
}

__global__ void leniaKernelnoInterop(LeniaWorld lw)
{
    int i = threadIdx.y + blockIdx.y * blockDim.y;
    int j = threadIdx.x + blockIdx.x * blockDim.x;

    if (i < lw.size && j < lw.size)
    {
        int index = i * lw.size + j;
        // apply rules
        for (int k = 0; k < lw.nbRules; k++)
        {
            Rule r = lw.rule[k];
            float* g1, * g2;
            if (lw.toGridB) {
                g1 = lw.gridA[r.source];
                g2 = lw.gridB[r.destination];
            }
            else {
                g1 = lw.gridB[r.source];
                g2 = lw.gridA[r.destination];
            }
            computeLeniaStep(i, j, r.radius, r.mu, r.sigma2, r.weight, lw.kernel[k], g1, lw.size, g2 + index);
        }
        for (int k = 0; k < lw.nbChannels; k++)
        {
            float* gridA, * gridB;
            if (lw.toGridB) {
                gridA = lw.gridA[k]; gridB = lw.gridB[k];
            }
            else {
                gridA = lw.gridB[k]; gridB = lw.gridA[k];
            }
            gridB[index] /= lw.T;
            gridB[index] += gridA[index];
            if (gridB[index] < 0.0f) gridB[index] = 0.0f;
            if (gridB[index] > 1.0f) gridB[index] = 1.0f;
        }
    }
}


Device32noInteropMode::Device32noInteropMode(int size, int T, int nbChannels, int nbRules, float channels[][4], float rules[][11])
{
    HostMode* lm = new HostMode(size, T, nbChannels, nbRules, channels, rules);
    lw = lm->lw;

    cudaMemcpyToSymbol(constantRule, lw.rule, MAX_RULES * sizeof(Rule));
    cudaMemcpyToSymbol(constantColor, lw.color, MAX_CHANNELS * sizeof(float4));

    for (int i = 0; i < lw.nbChannels; i++)
    {
        checkCudaErrors(cudaMalloc((void**)&lw.gridA[i], lw.size * lw.size * sizeof(float)));
        checkCudaErrors(cudaMalloc((void**)&lw.gridB[i], lw.size * lw.size * sizeof(float)));
        checkCudaErrors(cudaMemcpy(lw.gridA[i], lm->lw.gridA[i], lw.size * lw.size * sizeof(float), cudaMemcpyHostToDevice));
        checkCudaErrors(cudaMemset(lw.gridB[i], 0, lw.size * lw.size * sizeof(float)));
    }
    for (int i = 0; i < lw.nbRules; i++)
    {
        checkCudaErrors(cudaMalloc((void**)&lw.kernel[i], (2 * lw.rule[i].radius + 1) * (2 * lw.rule[i].radius + 1) * sizeof(float)));
        checkCudaErrors(cudaMemcpy(lw.kernel[i], lm->lw.kernel[i], (2 * lw.rule[i].radius + 1) * (2 * lw.rule[i].radius + 1) * sizeof(float), cudaMemcpyHostToDevice));
    }
    delete lm;

    cudaMalloc((void**)&cuPixels, lw.size * lw.size * sizeof(float4));
    cudaMallocHost((void**)&cuPixelsHost, lw.size * lw.size * sizeof(float4));

}

__host__ void Device32noInteropMode::compute(bool render, bool batchFFT = false)
{
    int dim = 16;
    dim3 dimBlock(dim, dim);
    dim3 dimGrid((lw.size + dim - 1) / dim, (lw.size + dim - 1) / dim);

    if (render)
        leniaKernelRendernoInterop << <dimGrid, dimBlock >> > (lw, cuPixels);
    else
        leniaKernelnoInterop << <dimGrid, dimBlock >> > (lw);

    for (int i = 0; i < lw.nbChannels; i++) // clear grid for next step
        if (lw.toGridB) checkCudaErrors(cudaMemset(lw.gridA[i], 0, lw.size * lw.size * sizeof(float)));
        else checkCudaErrors(cudaMemset(lw.gridB[i], 0, lw.size * lw.size * sizeof(float)));
    lw.toGridB = !lw.toGridB;
}

__host__ void Device32noInteropMode::render(int w, int h)
{
    glClear(GL_COLOR_BUFFER_BIT);
    glViewport(0, 0, w, h);

    checkCudaErrors(cudaMemcpy(cuPixelsHost, cuPixels, lw.size * lw.size * sizeof(float4), cudaMemcpyDeviceToHost));
    glPixelZoom(w / (float)lw.size, h / (float)lw.size);
    glDrawPixels(lw.size, lw.size, GL_RGBA, GL_FLOAT, cuPixelsHost);
}

Device32noInteropMode::~Device32noInteropMode()
{
    for (int i = 0; i < lw.nbRules; i++)
    {
        cudaFree(lw.kernel[i]);
    }

    for (int i = 0; i < lw.nbChannels; i++)
    {
        cudaFree(lw.gridA[i]);
        cudaFree(lw.gridB[i]);
    }
    cudaFree(cuPixels);
    cudaFreeHost(cuPixelsHost);
}

