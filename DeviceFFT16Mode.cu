#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "DeviceFFT16Mode.h"
#include "HostMode.h"
using namespace cuLenia;

__global__ void convolveFFT16(int size, int nbRules, LeniaWorldFFT16 lwFFT)
{
    int i = threadIdx.y + blockIdx.y * blockDim.y;
    int j = threadIdx.x + blockIdx.x * blockDim.x;
    int index = i * size + j;
    __nv_bfloat162 size2 = __bfloat162bfloat162(__int2bfloat16_rn(size * size));
    __nv_bfloat162 zero = __bfloat162bfloat162(CUDART_ZERO_BF16);
    int sizeFFT = (size / 2 + 1) * size;

    if (i < (size / 2 + 1) && j < size)
    {
        for (int r = 0; r < nbRules; r++)
            lwFFT.destFFT[r * sizeFFT + index] = __hcmadd(lwFFT.sourceFFT[constantRule[r].source * sizeFFT + index], lwFFT.kernelFFT[r * sizeFFT + index], zero) / size2;
    }
}

__global__ void computeLeniaStepFFT16Render(int size, int nbRules, int nbChannels, int T, bool toGridB, __nv_bfloat162* gridA2, __nv_bfloat162* gridB2, __nv_bfloat162* dest2, float4* cuPixels)
{
    int i = threadIdx.y + blockIdx.y * blockDim.y;
    int j = threadIdx.x + blockIdx.x * blockDim.x;
    int size2 = size * size / 2;
    __nv_bfloat162 value[MAX_CHANNELS] = { {0.f,0.f} };
    __nv_bfloat162 CUDART_ONE_BF162 = __bfloat162bfloat162(CUDART_ONE_BF16);
    __nv_bfloat162 CUDART_TWO_BF162 = CUDART_ONE_BF162 + CUDART_ONE_BF162;
    __nv_bfloat162 bT2 = __bfloat162bfloat162(__float2bfloat16(1.f / (float)T));

    __nv_bfloat162 ur;
    if (i < size && 2 * j < size)
    {
        int index = i * size / 2 + j;
        int index_ = 2 * index;
        for (int r = 0; r < nbRules; r++)
        {
            Rule rule = constantRule[r]; // computeGrowth
            ur = dest2[r * size2 + index] - rule.mu16;
            value[rule.destination] += rule.weight16 * (CUDART_TWO_BF162 * h2exp(-ur * ur / rule.sigma216) - CUDART_ONE_BF162);
        }
        // prepare renderGrid (rendering is nearly no-time)
        // finalize and report channels to rendergrid
        cuPixels[index_].x = 0;
        cuPixels[index_].y = 0;
        cuPixels[index_].z = 0;
        cuPixels[index_].w = 1;
        cuPixels[index_ + 1].x = 0;
        cuPixels[index_ + 1].y = 0;
        cuPixels[index_ + 1].z = 0;
        cuPixels[index_ + 1].w = 1;

        __nv_bfloat162 v, * gA2, * gB2;
        float2 vf2;
        for (int k = 0; k < nbChannels; k++)
        {
            if (toGridB) { gA2 = gridA2 + k * size2; gB2 = gridB2 + k * size2; }
            else { gA2 = gridB2 + k * size2; gB2 = gridA2 + k * size2; }
            v = __hfma2_sat(value[k], bT2, gA2[index]);
            gB2[index] = v;

            vf2 = __bfloat1622float2(v);
            cuPixels[index_].x += vf2.x * constantColor[k].x;
            cuPixels[index_].y += vf2.x * constantColor[k].y;
            cuPixels[index_].z += vf2.x * constantColor[k].z;
            cuPixels[index_ + 1].x += vf2.y * constantColor[k].x;
            cuPixels[index_ + 1].y += vf2.y * constantColor[k].y;
            cuPixels[index_ + 1].z += vf2.y * constantColor[k].z;
        }
    }
}

__global__ void computeLeniaStepFFT16(int size, int nbRules, int nbChannels, int T, bool toGridB, __nv_bfloat162* gridA2, __nv_bfloat162* gridB2, __nv_bfloat162* dest2)
{
    int i = threadIdx.y + blockIdx.y * blockDim.y;
    int j = threadIdx.x + blockIdx.x * blockDim.x;
    int size2 = size * size / 2;
    __nv_bfloat162 value[MAX_CHANNELS] = { {0.f,0.f} };
    __nv_bfloat162 CUDART_ONE_BF162 = __bfloat162bfloat162(CUDART_ONE_BF16);
    __nv_bfloat162 CUDART_TWO_BF162 = CUDART_ONE_BF162 + CUDART_ONE_BF162;
    __nv_bfloat162 bT2 = __bfloat162bfloat162(__float2bfloat16(1.f / (float)T));

    __nv_bfloat162 ur;
    if (i < size && 2 * j < size)
    {
        int index = i * size / 2 + j;
        for (int r = 0; r < nbRules; r++)
        {
            Rule rule = constantRule[r]; // computeGrowth
            ur = dest2[r * size2 + index] - rule.mu16;
            value[rule.destination] += rule.weight16 * (CUDART_TWO_BF162 * h2exp(-ur * ur / rule.sigma216) - CUDART_ONE_BF162);
        }
        if (toGridB) {
            for (int k = 0; k < nbChannels; k++)
                gridB2[k * size2 + index] = __hfma2_sat(value[k], bT2, gridA2[k * size2 + index]);
        }
        else {
            for (int k = 0; k < nbChannels; k++)
                gridA2[k * size2 + index] = __hfma2_sat(value[k], bT2, gridB2[k * size2 + index]);
        }
    }
}

DeviceFFT16Mode::DeviceFFT16Mode(int size, int T, int nbChannels, int nbRules, float channels[][4], float rules[][11])
{
    HostMode* lm = new HostMode(size, T, nbChannels, nbRules, channels, rules);
    lm->init16();
    lw = lm->lw;

    cudaMemcpyToSymbol(constantRule, lw.rule, MAX_RULES * sizeof(Rule));
    cudaMemcpyToSymbol(constantColor, lw.color, MAX_CHANNELS * sizeof(float4));

    // all channel grids are consecutively stored in grid1/2[0] to allow for FFT batches
    checkCudaErrors(cudaMalloc((void**)&lw.gridA16[0], lw.nbChannels * size * size * sizeof(__nv_bfloat16)));
    checkCudaErrors(cudaMalloc((void**)&lw.gridB16[0], lw.nbChannels * size * size * sizeof(__nv_bfloat16)));
    for (int i = 0; i < lw.nbChannels; i++)
        checkCudaErrors(cudaMemcpy(lw.gridA16[0] + i * size * size, lm->lw.gridA16[i], size * size * sizeof(__nv_bfloat16), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemset(lw.gridB16[0], 0, lw.nbChannels * lw.size * lw.size * sizeof(__nv_bfloat16)));
    checkCudaErrors(cudaMalloc((void**)&lwFFT.sourceFFT, lw.nbChannels * (size / 2 + 1) * size * sizeof(__nv_bfloat162)));

    // create FFT plans
    int batchR2C = lw.nbChannels;
    int batchC2R = lw.nbRules;
    int rank = 2;
    long long int n[2] = { size, size };
    long long int idist = size * size;
    long long int odist = size * (size / 2 + 1);
    long long int inembed[] = { size, size };
    long long int onembed[] = { size, size / 2 + 1 };
    long long int istride = 1;
    long long int ostride = 1;
    size_t workSize = 0;

    cufftCreate(&planManyR2C); 
    cufftXtMakePlanMany(planManyR2C, rank, n, inembed, istride, idist, CUDA_R_16BF, onembed, ostride, odist, CUDA_C_16BF, batchR2C, &workSize, CUDA_C_16BF);

    cufftCreate(&planManyC2R);
    cufftXtMakePlanMany(planManyC2R, rank, n, onembed, ostride, odist, CUDA_C_16BF, inembed, istride, idist, CUDA_R_16BF, batchC2R, &workSize, CUDA_R_16BF);

    cufftCreate(&planR2C);
    cufftXtMakePlanMany(planR2C, rank, n, inembed, istride, idist, CUDA_R_16BF, onembed, ostride, odist, CUDA_C_16BF, 1, &workSize, CUDA_C_16BF);

    cufftCreate(&planC2R);
    cufftXtMakePlanMany(planC2R, rank, n, onembed, ostride, odist, CUDA_C_16BF, inembed, istride, idist, CUDA_R_16BF, 1, &workSize, CUDA_R_16BF);

    checkCudaErrors(cudaMalloc((void**)&lwFFT.destFFT, lw.nbRules * (size / 2 + 1) * size * sizeof(__nv_bfloat162)));
    checkCudaErrors(cudaMalloc((void**)&lwFFT.kernelFFT, lw.nbRules * (size / 2 + 1) * size * sizeof(__nv_bfloat162)));
    checkCudaErrors(cudaMalloc((void**)&lwFFT.dest, lw.nbRules * size * size * sizeof(__nv_bfloat16)));
    for (int r = 0; r < lw.nbRules; r++)
    {
        __nv_bfloat16* kernelgs = initKernelGridSize(r); // copy the kernel into a size*size matrix
        checkCudaErrors(cudaMalloc((void**)&lw.kernel16[r], size * size * sizeof(__nv_bfloat16)));
        checkCudaErrors(cudaMemcpy(lw.kernel16[r], kernelgs, size * size * sizeof(__nv_bfloat16), cudaMemcpyHostToDevice));
        cufftXtExec(planR2C, lw.kernel16[r], lwFFT.kernelFFT + r * (size / 2 + 1) * size, CUFFT_FORWARD);
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
    
    dimBlock = dim3(DIM, DIM);
    dimGrid2 = dim3(((lw.size + DIM - 1) / DIM) / 2, (lw.size + DIM - 1) / DIM);
    dimGridFFT = dim3((lw.size + DIM - 1) / DIM, ((lw.size / 2 + 1) + DIM - 1) / DIM);
}

__nv_bfloat16* DeviceFFT16Mode::initKernelGridSize(int r)
{
    __nv_bfloat16* kernelgs = (__nv_bfloat16*)malloc(lw.size * lw.size * sizeof(__nv_bfloat16));
    memset(kernelgs, 0, lw.size * lw.size * sizeof(__nv_bfloat16));
    int R = lw.rule[r].radius;
    int w = 2 * R + 1;
    int S = lw.size;
    for (int i = -R;i <= R; i++)
        for (int j = -R;j <= R; j++)
            kernelgs[((i + S) % S) * S + ((j + S) % S)] = lw.kernel16[r][((i + w) % w) * w + ((j + w) % w)];
    return kernelgs;
}

void DeviceFFT16Mode::compute(bool render, bool batchFFT)
{
    __nv_bfloat16* fromGrid;
    if (lw.toGridB)
        fromGrid = lw.gridA16[0];
    else
        fromGrid = lw.gridB16[0];

    if (batchFFT) {
        cufftXtExec(planManyR2C, fromGrid, lwFFT.sourceFFT, CUFFT_FORWARD);
        convolveFFT16 << <dimGridFFT, dimBlock >> > (lw.size, lw.nbRules, lwFFT);
        cufftXtExec(planManyC2R, lwFFT.destFFT, lwFFT.dest, CUFFT_INVERSE);
    }
    else
    {
        for (int k = 0; k < lw.nbChannels; k++)
            cufftXtExec(planR2C, fromGrid + k * lw.size * lw.size, lwFFT.sourceFFT + k * lw.size * (lw.size / 2 + 1), CUFFT_FORWARD);
        convolveFFT16 << <dimGridFFT, dimBlock >> > (lw.size, lw.nbRules, lwFFT);
        for (int k = 0; k < lw.nbRules; k++)
            cufftXtExec(planC2R, lwFFT.destFFT + k * lw.size * (lw.size / 2 + 1), lwFFT.dest + k * lw.size * lw.size, CUFFT_INVERSE);
    }

    if (render)
    {
        cudaGraphicsMapResources(1, &cuBuffer, 0); // ********* OpenGL interop
        float4* cuPixels;
        size_t num_bytes;
        cudaGraphicsResourceGetMappedPointer((void**)&cuPixels, &num_bytes, cuBuffer);
        computeLeniaStepFFT16Render << <dimGrid2, dimBlock >> > (lw.size, lw.nbRules, lw.nbChannels, lw.T, lw.toGridB,
            reinterpret_cast<__nv_bfloat162*>(lw.gridA16[0]),
            reinterpret_cast<__nv_bfloat162*>(lw.gridB16[0]),
            reinterpret_cast<__nv_bfloat162*>(lwFFT.dest), cuPixels);
        cudaGraphicsUnmapResources(1, &cuBuffer);
    }
    else
        computeLeniaStepFFT16 << <dimGrid2, dimBlock >> > (lw.size, lw.nbRules, lw.nbChannels, lw.T, lw.toGridB,
            reinterpret_cast<__nv_bfloat162*>(lw.gridA16[0]),
            reinterpret_cast<__nv_bfloat162*>(lw.gridB16[0]),
            reinterpret_cast<__nv_bfloat162*>(lwFFT.dest));

    lw.toGridB = !lw.toGridB;
}

void DeviceFFT16Mode::render(int w, int h)
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

DeviceFFT16Mode::~DeviceFFT16Mode()
{
    cudaFree(lwFFT.kernelFFT);
    cudaFree(lwFFT.destFFT);
    cudaFree(lwFFT.dest);
    cudaFree(lwFFT.sourceFFT);

    cudaFree(lw.gridA16[0]);
    cudaFree(lw.gridB16[0]);
    for (int i = 0; i < lw.nbRules; i++)
        cudaFree(lw.kernel16[i]);

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


