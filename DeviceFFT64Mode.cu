#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "DeviceFFT64Mode.h"
#include "HostMode.h"
using namespace cuLenia;

__device__ void computeGrowth64(int i, int j, double mu, double sigma2, double weight, double u_t, double* cell) {
    double ur = u_t - mu;
    double g_t = 2.0 * exp(-ur * ur / sigma2) - 1.0;
    double c_t = weight * g_t;
    *cell += c_t;
}

__global__ void convolveFFT64(int size, int nbRules, LeniaWorldFFT64 lwFFT)
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
            cufftDoubleComplex s = lwFFT.sourceFFT[rule.source * sizeFFT + index];
            cufftDoubleComplex k = lwFFT.kernelFFT[r * sizeFFT + index];
            lwFFT.destFFT[r * sizeFFT + index].x = (s.x * k.x - s.y * k.y) / size2;
            lwFFT.destFFT[r * sizeFFT + index].y = (s.x * k.y + s.y * k.x) / size2;
        }
    }
}

__global__ void computeLeniaStepFFT64Render(int size, int nbRules, int nbChannels, int T, bool toGridB, double* gridA, double* gridB, cufftDoubleReal* dest, float4* cuPixels)
{
    int i = threadIdx.y + blockIdx.y * blockDim.y;
    int j = threadIdx.x + blockIdx.x * blockDim.x;
    int size2 = size * size;
    double value[MAX_CHANNELS] = { 0. };

    if (i < size && j < size)
    {
        int index = i * size + j;
        for (int r = 0; r < nbRules; r++)
        {
            Rule rule = constantRule[r];
            // computeGrowth
            double ur = dest[r * size2 + index] - rule.mu64;
            value[rule.destination] += rule.weight64 * (2.0 * expf(-ur * ur / rule.sigma264) - 1.0);
        }
        // prepare renderGrid (rendering is nearly no-time)
        // finalize and report channels to rendergrid
        cuPixels[index].x = 0;
        cuPixels[index].y = 0;
        cuPixels[index].z = 0;
        cuPixels[index].w = 1;

        double* gA, * gB;
        double v;
        for (int k = 0; k < nbChannels; k++)
        {
            if (toGridB) {
                gA = gridA + k * size2; gB = gridB + k * size2;
            }
            else {
                gA = gridB + k * size2; gB = gridA + k * size2;
            }
            v = value[k] / T + gA[index];
            if (v < 0.0) v = 0.0;
            else if (v > 1.0) v = 1.0;
            cuPixels[index].x += v * constantColor[k].x;
            cuPixels[index].y += v * constantColor[k].y;
            cuPixels[index].z += v * constantColor[k].z;
            gB[index] = v;
        }
    }
}

__global__ void computeLeniaStepFFT64(int size, int nbRules, int nbChannels, int T, bool toGridB, double* gridA, double* gridB, cufftDoubleReal* dest)
{
    int i = threadIdx.y + blockIdx.y * blockDim.y;
    int j = threadIdx.x + blockIdx.x * blockDim.x;
    int size2 = size * size;
    double value[MAX_CHANNELS] = { 0. };

    if (i < size && j < size)
    {
        int index = i * size + j;
        for (int r = 0; r < nbRules; r++)
        {
            Rule rule = constantRule[r];
            // computeGrowth
            double ur = dest[r * size2 + index] - rule.mu64;
            value[rule.destination] += rule.weight64 * (2.0 * expf(-ur * ur / rule.sigma264) - 1.0);
        }
        double* gA, * gB;
        double v;
        for (int k = 0; k < nbChannels; k++)
        {
            if (toGridB) {
                gA = gridA + k * size2; gB = gridB + k * size2;
            }
            else {
                gA = gridB + k * size2; gB = gridA + k * size2;
            }
            v = value[k] / T + gA[index];
            if (v < 0.0) v = 0.0;
            else if (v > 1.0) v = 1.0;
            gB[index] = v;
        }
    }
}


DeviceFFT64Mode::DeviceFFT64Mode(int size, int T, int nbChannels, int nbRules, float channels[][4], float rules[][11])
{
    HostMode* lm = new HostMode(size, T, nbChannels, nbRules, channels, rules);
    lm->init64();
    lw = lm->lw;

    cudaMemcpyToSymbol(constantRule, lw.rule, MAX_RULES * sizeof(Rule));
    cudaMemcpyToSymbol(constantColor, lw.color, MAX_CHANNELS * sizeof(float4));

    // all channel grids are consecutively stored in grid1/2[0] to allow for FFT batches
    checkCudaErrors(cudaMalloc((void**)&lw.gridA64[0], lw.nbChannels * size * size * sizeof(double)));
    checkCudaErrors(cudaMalloc((void**)&lw.gridB64[0], lw.nbChannels * size * size * sizeof(double)));
    for (int i = 0; i < lw.nbChannels; i++)
        checkCudaErrors(cudaMemcpy(lw.gridA64[0] + i * size * size, lm->lw.gridA64[i], size * size * sizeof(double), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemset(lw.gridB64[0], 0, lw.nbChannels * lw.size * lw.size * sizeof(double)));
    checkCudaErrors(cudaMalloc((void**)&lwFFT.sourceFFT, lw.nbChannels * (size / 2 + 1) * size * sizeof(cufftDoubleComplex)));

    // create FFT plans
    int batchD2Z = lw.nbChannels;
    int batchZ2D = lw.nbRules;
    int rank = 2;
    int n[2] = { size, size };
    int idist = size * size;
    int odist = size * (size / 2 + 1);
    int inembed[] = { size, size };
    int onembed[] = { size, size / 2 + 1 };
    int istride = 1;
    int ostride = 1;

    cufftPlanMany(&planManyD2Z, rank, n, inembed, istride, idist, onembed, ostride, odist, CUFFT_D2Z, batchD2Z);
    cufftPlanMany(&planManyZ2D, rank, n, onembed, ostride, odist, inembed, istride, idist, CUFFT_Z2D, batchZ2D);
    cufftPlan2d(&planD2Z, size, size, CUFFT_D2Z);
    cufftPlan2d(&planZ2D, size, size, CUFFT_Z2D);

    checkCudaErrors(cudaMalloc((void**)&lwFFT.destFFT, lw.nbRules * (size / 2 + 1) * size * sizeof(cufftDoubleComplex)));
    checkCudaErrors(cudaMalloc((void**)&lwFFT.kernelFFT, lw.nbRules * (size / 2 + 1) * size * sizeof(cufftDoubleComplex)));
    checkCudaErrors(cudaMalloc((void**)&lwFFT.dest, lw.nbRules * size * size * sizeof(cufftDoubleReal)));
    for (int r = 0; r < lw.nbRules; r++)
    {
        double* kernelgs = initKernelGridSize(r); // copy the kernel into a size*size matrix
        checkCudaErrors(cudaMalloc((void**)&lw.kernel64[r], size * size * sizeof(double)));
        checkCudaErrors(cudaMemcpy(lw.kernel64[r], kernelgs, size * size * sizeof(double), cudaMemcpyHostToDevice));
        cufftExecD2Z(planD2Z, lw.kernel64[r], lwFFT.kernelFFT + r * (size / 2 + 1) * size);
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

double* DeviceFFT64Mode::initKernelGridSize(int r)
{
    double* kernelgs = (double*)malloc(lw.size * lw.size * sizeof(double));
    memset(kernelgs, 0, lw.size * lw.size * sizeof(double));
    int R = lw.rule[r].radius;
    int w = 2 * R + 1;
    int S = lw.size;
    for (int i = -R;i <= R; i++)
        for (int j = -R;j <= R; j++)
            kernelgs[((i + S) % S) * S + ((j + S) % S)] = lw.kernel64[r][((i + w) % w) * w + ((j + w) % w)];
    return kernelgs;
}

void DeviceFFT64Mode::compute(bool render, bool batchFFT)
{
    int dim = 16;
    dim3 dimBlock(dim, dim);
    dim3 dimGrid((lw.size + dim - 1) / dim, (lw.size + dim - 1) / dim);
    dim3 dimGridFFT((lw.size + dim - 1) / dim, ((lw.size / 2 + 1) + dim - 1) / dim);

    double* fromGrid;
    if (lw.toGridB) fromGrid = lw.gridA64[0]; else fromGrid = lw.gridB64[0];

    if (batchFFT) {
        cufftExecD2Z(planManyD2Z, fromGrid, lwFFT.sourceFFT);
        convolveFFT64 << <dimGridFFT, dimBlock >> > (lw.size, lw.nbRules, lwFFT);
        cufftExecZ2D(planManyZ2D, lwFFT.destFFT, lwFFT.dest);
    }
    else
    {
        for (int k = 0; k < lw.nbChannels; k++)
            cufftExecD2Z(planD2Z, fromGrid + k * lw.size * lw.size, lwFFT.sourceFFT + k * lw.size * (lw.size / 2 + 1));
        convolveFFT64 << <dimGridFFT, dimBlock >> > (lw.size, lw.nbRules, lwFFT);
        for (int k = 0; k < lw.nbRules; k++)
            cufftExecZ2D(planZ2D, lwFFT.destFFT + k * lw.size * (lw.size / 2 + 1), lwFFT.dest + k * lw.size * lw.size);
    }


    if (render)
    {
        cudaGraphicsMapResources(1, &cuBuffer, 0); // ********* OpenGL interop
        float4* cuPixels;
        size_t num_bytes;
        cudaGraphicsResourceGetMappedPointer((void**)&cuPixels, &num_bytes, cuBuffer);
        computeLeniaStepFFT64Render << <dimGrid, dimBlock >> > (lw.size, lw.nbRules, lw.nbChannels, lw.T, lw.toGridB, lw.gridA64[0], lw.gridB64[0], lwFFT.dest, cuPixels);
        cudaGraphicsUnmapResources(1, &cuBuffer);
    }
    else
        computeLeniaStepFFT64 << <dimGrid, dimBlock >> > (lw.size, lw.nbRules, lw.nbChannels, lw.T, lw.toGridB, lw.gridA64[0], lw.gridB64[0], lwFFT.dest);

    lw.toGridB = !lw.toGridB;
}

void DeviceFFT64Mode::render(int w, int h)
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

DeviceFFT64Mode::~DeviceFFT64Mode()
{
    cudaFree(lwFFT.kernelFFT);
    cudaFree(lwFFT.destFFT);
    cudaFree(lwFFT.dest);
    cudaFree(lwFFT.sourceFFT);

    cudaFree(lw.gridA64[0]);
    cudaFree(lw.gridB64[0]);
    for (int i = 0; i < lw.nbRules; i++)
        cudaFree(lw.kernel64[i]);

    cufftDestroy(planD2Z);
    cufftDestroy(planZ2D);
    cufftDestroy(planManyD2Z);
    cufftDestroy(planManyZ2D);

    // Interop
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER_ARB, 0);
    glBindTexture(GL_TEXTURE_2D, 0);
    glDisable(GL_TEXTURE_2D);
    cudaGraphicsUnregisterResource(cuBuffer);
    glDeleteTextures(1, &renderGridTexture);
    glDeleteBuffers(1, &renderGridBuffer);
}


