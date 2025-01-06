#include <stdio.h>
#include <stdlib.h>

#include "Device32Mode.h"
#include "HostMode.h"
using namespace cuLenia;

__global__ void leniaKernelRender(LeniaWorld lw, float4* cuPixels)
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

__global__ void leniaKernel(LeniaWorld lw)
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


Device32Mode::Device32Mode(int size, int T, int nbChannels, int nbRules, float channels[][4], float rules[][11])
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

    // init Interop
    glGenBuffers(1, &renderGridBuffer); // create a buffer
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER_ARB, renderGridBuffer); // make it the active buffer
    glBufferData(GL_PIXEL_UNPACK_BUFFER_ARB, lw.size * lw.size * sizeof(float4), NULL, GL_STREAM_DRAW); // allocate memory, but dont copy data (NULL)
    glEnable(GL_TEXTURE_2D); // Enable texturing
    glGenTextures(1, &renderGridTexture); // Generate a texture ID
    glBindTexture(GL_TEXTURE_2D, renderGridTexture); // Set as the current texture
    // Allocate the texture memory. The last parameter is NULL: we only want to allocate memory, not initialize it
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, lw.size, lw.size, 0, GL_RGBA, GL_FLOAT, NULL);
    // Must set the filter mode: GL_LINEAR enables interpolation when scaling
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    // cudaGraphicsMapFlagsWriteDiscard: CUDA will only write and will not read from this resource
    cudaGraphicsGLRegisterBuffer(&cuBuffer, renderGridBuffer, cudaGraphicsMapFlagsWriteDiscard);
}

__host__ void Device32Mode::compute(bool render, bool batchFFT)
{
    int dim = 16;
    dim3 dimBlock(dim, dim);
    dim3 dimGrid((lw.size + dim - 1) / dim, (lw.size + dim - 1) / dim);

    if (render)
    {
        cudaGraphicsMapResources(1, &cuBuffer, 0);
        float4* cuPixels;
        size_t num_bytes;
        cudaGraphicsResourceGetMappedPointer((void**)&cuPixels, &num_bytes, cuBuffer);
        leniaKernelRender << <dimGrid, dimBlock >> > (lw, cuPixels);
        cudaGraphicsUnmapResources(1, &cuBuffer);
    }    
    else
        leniaKernel << <dimGrid, dimBlock >> > (lw);

    for (int i = 0; i < lw.nbChannels; i++) // clear grid for next step
        if (lw.toGridB) checkCudaErrors(cudaMemset(lw.gridA[i], 0, lw.size * lw.size * sizeof(float)));
        else checkCudaErrors(cudaMemset(lw.gridB[i], 0, lw.size * lw.size * sizeof(float)));
    lw.toGridB = !lw.toGridB;
}

__host__ void Device32Mode::render(int w, int h)
{
    glClear(GL_COLOR_BUFFER_BIT);
    glViewport(0, 0, w, h);

    glBindBuffer(GL_PIXEL_UNPACK_BUFFER, renderGridBuffer); // Select the appropriate buffer
    glBindTexture(GL_TEXTURE_2D, renderGridTexture); // Select the appropriate texture
    glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, lw.size, lw.size, GL_RGBA, GL_FLOAT, NULL); // Make a texture from the buffer

    glBegin(GL_QUADS);
    glVertex2f(-1.0f, -1.0f);
    glTexCoord2f(0.0f, 0.0f);
    glVertex2f(1.0f, -1.0f);
    glTexCoord2f(1.0f, 0.0f);
    glVertex2f(1.0f, 1.0f);
    glTexCoord2f(1.0f, 1.0f);
    glVertex2f(-1.0f, 1.0f);
    glTexCoord2f(0.0f, 1.0f);
    glEnd();
}

Device32Mode::~Device32Mode()
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
    
    // clear Interop
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER_ARB, 0);
    glBindTexture(GL_TEXTURE_2D, 0);
    glDisable(GL_TEXTURE_2D);
    cudaGraphicsUnregisterResource(cuBuffer);
    glDeleteTextures(1, &renderGridTexture);
    glDeleteBuffers(1, &renderGridBuffer);
}

