#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "DeviceFFT32Mode.h"
#include "HostMode.h"
using namespace cuLenia;

__global__ void convolveFFT32(int size, int nbRules, LeniaWorldFFT lwFFT)
{
    int i = threadIdx.y + blockIdx.y * blockDim.y;
    int j = threadIdx.x + blockIdx.x * blockDim.x;
    int index = i * size + j;
    int size2 = size * size;
    int sizeFFT = (size / 2 + 1) * size;

    if (i < (size / 2 + 1) && j < size)
    {
        for (int r = 0; r < nbRules; r++)
        {
            Rule rule = constantRule[r];
            cufftComplex s = lwFFT.sourceFFT[rule.source * sizeFFT + index];
            cufftComplex k = lwFFT.kernelFFT[r * sizeFFT + index];
            lwFFT.destFFT[r * sizeFFT + index].x = (s.x * k.x - s.y * k.y) / size2;
            lwFFT.destFFT[r * sizeFFT + index].y = (s.x * k.y + s.y * k.x) / size2;
        }
    }
}

__global__ void computeLeniaStepFFT32Render(int size, int nbRules, int nbChannels, int T, bool toGridB, float* gridA, float* gridB, cufftReal* dest, float4* cuPixels)
{
    int i = threadIdx.y + blockIdx.y * blockDim.y;
    int j = threadIdx.x + blockIdx.x * blockDim.x;
    int size2 = size * size;
    float value[MAX_CHANNELS] = { 0.f };

    if (i < size && j < size)
    {
        int index = i * size + j;
        for (int r = 0; r < nbRules; r++)
        {
            Rule rule = constantRule[r];
            // computeGrowth
            float ur = dest[r * size2 + index] - rule.mu;
            value[rule.destination] += rule.weight * (2.0f * expf(-ur * ur / rule.sigma2) - 1.0f);
        }
        // prepare renderGrid (rendering is nearly no-time)
        // finalize and report channels to rendergrid
        cuPixels[index].x = 0;
        cuPixels[index].y = 0;
        cuPixels[index].z = 0;
        cuPixels[index].w = 1;

        float* gA, * gB;
        float v;
        for (int k = 0; k < nbChannels; k++)
        {
            if (toGridB) {
                gA = gridA + k * size2; gB = gridB + k * size2;
            }
            else {
                gA = gridB + k * size2; gB = gridA + k * size2;
            }
            v = value[k] / T + gA[index];
            if (v < 0.0f) v = 0.0f;
            else if (v > 1.0f) v = 1.0f;
            cuPixels[index].x += v * constantColor[k].x;
            cuPixels[index].y += v * constantColor[k].y;
            cuPixels[index].z += v * constantColor[k].z;
            gB[index] = v;
        }
    }
}

__global__ void computeLeniaStepFFT32(int size, int nbRules, int nbChannels, int T, bool toGridB, float* gridA, float* gridB, cufftReal* dest)
{
    int i = threadIdx.y + blockIdx.y * blockDim.y;
    int j = threadIdx.x + blockIdx.x * blockDim.x;
    int size2 = size * size;
    float value[MAX_CHANNELS] = { 0.f };

    if (i < size && j < size)
    {
        int index = i * size + j;
        for (int r = 0; r < nbRules; r++)
        {
            Rule rule = constantRule[r];
            // computeGrowth
            float ur = dest[r * size2 + index] - rule.mu;
            value[rule.destination] += rule.weight * (2.0f * expf(-ur * ur / rule.sigma2) - 1.0f);
        }
        float* gA, * gB;
        float v;
        for (int k = 0; k < nbChannels; k++)
        {
            if (toGridB) {
                gA = gridA + k * size2; gB = gridB + k * size2;
            }
            else {
                gA = gridB + k * size2; gB = gridA + k * size2;
            }
            v = value[k] / T + gA[index];
            if (v < 0.0f) v = 0.0f;
            else if (v > 1.0f) v = 1.0f;
            gB[index] = v;
        }
    }
}

DeviceFFT32Mode::DeviceFFT32Mode(int size, int T, int nbChannels, int nbRules, float channels[][4], float rules[][11])
{
    HostMode* lm = new HostMode(size, T, nbChannels, nbRules, channels, rules);
    lw = lm->lw;

    cudaMemcpyToSymbol(constantRule, lw.rule, MAX_RULES * sizeof(Rule));
    cudaMemcpyToSymbol(constantColor, lw.color, MAX_CHANNELS * sizeof(float4));

    // all channel grids are consecutively stored in gridA/2[0] to allow for FFT batches
    checkCudaErrors(cudaMalloc((void**)&lw.gridA[0], lw.nbChannels * size * size * sizeof(float)));
    checkCudaErrors(cudaMalloc((void**)&lw.gridB[0], lw.nbChannels * size * size * sizeof(float)));
    for (int i = 0; i < lw.nbChannels; i++)
        checkCudaErrors(cudaMemcpy(lw.gridA[0] + i * size * size, lm->lw.gridA[i], size * size * sizeof(float), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemset(lw.gridB[0], 0, lw.nbChannels * lw.size * lw.size * sizeof(float)));
    checkCudaErrors(cudaMalloc((void**)&lwFFT.sourceFFT, lw.nbChannels * (size / 2 + 1) * size * sizeof(cufftComplex)));

    // create FFT plans
    int batchR2C = lw.nbChannels;
    int batchC2R = lw.nbRules;
    int rank = 2;
    int n[2] = { size, size };
    int idist = size * size;
    int odist = size * (size / 2 + 1);
    int inembed[] = { size, size };
    int onembed[] = { size, size / 2 + 1 };
    int istride = 1;
    int ostride = 1;
    cufftPlanMany(&planManyR2C, rank, n, inembed, istride, idist, onembed, ostride, odist, CUFFT_R2C, batchR2C);
    cufftPlanMany(&planManyC2R, rank, n, onembed, ostride, odist, inembed, istride, idist, CUFFT_C2R, batchC2R);
    cufftPlan2d(&planR2C, size, size, CUFFT_R2C);
    cufftPlan2d(&planC2R, size, size, CUFFT_C2R);

    checkCudaErrors(cudaMalloc((void**)&lwFFT.destFFT, lw.nbRules * (size / 2 + 1) * size * sizeof(cufftComplex)));
    checkCudaErrors(cudaMalloc((void**)&lwFFT.kernelFFT, lw.nbRules * (size / 2 + 1) * size * sizeof(cufftComplex)));
    checkCudaErrors(cudaMalloc((void**)&lwFFT.dest, lw.nbRules * size * size * sizeof(cufftReal)));
    for (int r = 0; r < lw.nbRules; r++)
    {
        float* kernelgs = initKernelGridSize(r); // copy the kernel into a size*size matrix
        checkCudaErrors(cudaMalloc((void**)&lw.kernel[r], size * size * sizeof(float)));
        checkCudaErrors(cudaMemcpy(lw.kernel[r], kernelgs, size * size * sizeof(float), cudaMemcpyHostToDevice));
        cufftExecR2C(planR2C, lw.kernel[r], lwFFT.kernelFFT + r * (size / 2 + 1) * size);
        free(kernelgs);
    }

    delete lm;

    // init Interop
    glGenBuffers(1, &renderGridBuffer); // create a buffer  
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER_ARB, renderGridBuffer); // make it the active buffer    
    glBufferData(GL_PIXEL_UNPACK_BUFFER_ARB, size * size * sizeof(float4), NULL, GL_STREAM_DRAW); // allocate memory, but dont copy data (NULL)
    glEnable(GL_TEXTURE_2D); // Enable texturing
    glGenTextures(1, &renderGridTexture); // Generate a texture ID
    glBindTexture(GL_TEXTURE_2D, renderGridTexture); // Set as the current texture
    // Allocate the texture memory. The last parameter is NULL: we only want to allocate memory, not initialize it
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, size, size, 0, GL_RGBA, GL_FLOAT, NULL);
    // Must set the filter mode: GL_LINEAR enables interpolation when scaling
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    // cudaGraphicsMapFlagsWriteDiscard: CUDA will only write and will not read from this resource
    cudaGraphicsGLRegisterBuffer(&cuBuffer, renderGridBuffer, cudaGraphicsMapFlagsWriteDiscard);
}

float* DeviceFFT32Mode::initKernelGridSize(int r)
{
    float* kernelgs = (float*)malloc(lw.size * lw.size * sizeof(float));
    memset(kernelgs, 0, lw.size * lw.size * sizeof(float));
    int R = lw.rule[r].radius;
    int w = 2 * R + 1;
    int S = lw.size;
    for (int i = -R;i <= R; i++)
        for (int j = -R;j <= R; j++)
            kernelgs[((i + S) % S) * S + ((j + S) % S)] = lw.kernel[r][((i + w) % w) * w + ((j + w) % w)];
    return kernelgs;
}

void DeviceFFT32Mode::compute(bool render, bool batchFFT)
{
    int dim = 16;
    dim3 dimBlock(dim, dim);
    dim3 dimGrid((lw.size + dim - 1) / dim, (lw.size + dim - 1) / dim);
    dim3 dimGridFFT((lw.size + dim - 1) / dim, ((lw.size / 2 + 1) + dim - 1) / dim);

    float* fromGrid;
    if (lw.toGridB) fromGrid = lw.gridA[0]; else fromGrid = lw.gridB[0];

    if (batchFFT) {
        cufftExecR2C(planManyR2C, fromGrid, lwFFT.sourceFFT);
        convolveFFT32 << <dimGridFFT, dimBlock >> > (lw.size, lw.nbRules, lwFFT);
        cufftExecC2R(planManyC2R, lwFFT.destFFT, lwFFT.dest);
    }
    else
    {
        for (int k = 0; k < lw.nbChannels; k++)
            cufftExecR2C(planR2C, fromGrid + k * lw.size * lw.size, lwFFT.sourceFFT + k * lw.size * (lw.size / 2 + 1));
        convolveFFT32 << <dimGridFFT, dimBlock >> > (lw.size, lw.nbRules, lwFFT);
        for (int k = 0; k < lw.nbRules; k++)
            cufftExecC2R(planC2R, lwFFT.destFFT + k * lw.size * (lw.size / 2 + 1), lwFFT.dest + k * lw.size * lw.size);
    }


    if (render)
    {
        cudaGraphicsMapResources(1, &cuBuffer, 0); // ********* OpenGL interop
        float4* cuPixels;
        size_t num_bytes;
        cudaGraphicsResourceGetMappedPointer((void**)&cuPixels, &num_bytes, cuBuffer);
        computeLeniaStepFFT32Render << <dimGrid, dimBlock >> > (lw.size, lw.nbRules, lw.nbChannels, lw.T, lw.toGridB, lw.gridA[0], lw.gridB[0], lwFFT.dest, cuPixels);
        cudaGraphicsUnmapResources(1, &cuBuffer);
    }
    else
        computeLeniaStepFFT32 << <dimGrid, dimBlock >> > (lw.size, lw.nbRules, lw.nbChannels, lw.T, lw.toGridB, lw.gridA[0], lw.gridB[0], lwFFT.dest);

    lw.toGridB = !lw.toGridB;

}

void DeviceFFT32Mode::render(int w, int h)
{
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

DeviceFFT32Mode::~DeviceFFT32Mode()
{
    cudaFree(lwFFT.kernelFFT);
    cudaFree(lwFFT.destFFT);
    cudaFree(lwFFT.dest);
    cudaFree(lwFFT.sourceFFT);

    cudaFree(lw.gridA[0]);
    cudaFree(lw.gridB[0]);
    for (int i = 0; i < lw.nbRules; i++)
        cudaFree(lw.kernel[i]);

    cufftDestroy(planR2C);
    cufftDestroy(planC2R);
    cufftDestroy(planManyR2C);
    cufftDestroy(planManyC2R);

    // Interop
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER_ARB, 0);
    glBindTexture(GL_TEXTURE_2D, 0);
    glDisable(GL_TEXTURE_2D);
    cudaGraphicsUnregisterResource(cuBuffer);
    glDeleteTextures(1, &renderGridTexture);
    glDeleteBuffers(1, &renderGridBuffer);

}


