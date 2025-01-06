#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#include "lenia.h"

__constant__ Rule constantRule[MAX_RULES];
__constant__ float4 constantColor[MAX_CHANNELS];


float* randomGrid(int size, float threshold) {
    float* g = (float*)malloc(size * size * sizeof(float));
    for (int i = 0; i < size * size; i++) if (g) {
        float test_alive = rand() / (float)RAND_MAX;
        if (test_alive >= threshold) g[i] = 0.0f; else g[i] = rand() / (float)RAND_MAX;
    }
    return g;
}

float4* zeroRenderGrid(int size) {
    float4* g = (float4*)malloc(size * size * sizeof(float4));
    for (int i = 0; i < size * size;i++) {
        g[i].x = 0.0f;
        g[i].y = 0.0f;
        g[i].z = 0.0f;
        g[i].w = 1.0f;
    }
    return g;
}

float kernel_shell(float r, int B, float alpha, float beta[MAX_RANK])
{
    float Br = B * r;
    int index = floor(Br); // r in [0... 1[, so index in 0..B-1
    float u = fmod(Br, 1.0f);
//    float k_core_exp = exp(-(u - 0.5f)*(u - 0.5f) / 0.15f / 0.15f / 2.f); // shadertoy version
    float k_core_exp = exp(alpha * (1.0f - 1.0f / (4.0f * u * (1.0f - u))));
    return beta[index] * k_core_exp;
}

float* initKernel(int R, float relR, int rank, float alpha, float beta[MAX_RANK])
{
    int w = 2 * R + 1;
    float* kernel = (float*)malloc(w * w * sizeof(float));

    float ktotal = 0.0f;
    for (int i = -R; i <= R; i++)
        for (int j = -R; j <= R; j++) if (kernel)
        {
            int ni = (i + w) % w;
            int nj = (j + w) % w;
            kernel[ni * w + nj] = 0;
            float r = sqrtf(i * i + j * j) / R;
            if (r < relR) kernel[ni * w + nj] = kernel_shell(r / relR, rank, alpha, beta); // within the radius
            ktotal += kernel[ni * w + nj];
        }

    for (int i = 0; i < w; i++)
        for (int j = 0; j < w; j++)
            kernel[i * w + j] /= ktotal;

    return kernel;
}

__host__ __device__ float convolve(int y, int x, float* grid, int size, float* kernel, int R) {
    // exploit kernel symmetry
    int w = 2 * R + 1;
    int R2 = R * R;
    int xi, xi_, yj, yj_, yi, yi_, xj, xj_;
    float sum = kernel[0] * grid[y * size + x]; // ******* i=0 and j=0
    for (int i = 1; i <= R; i++)
    {
        xi = x + i; if (xi >= size) xi -= size;
        xi_ = x - i; if (xi_ < 0) xi_ += size;
        yi = y + i; if (yi >= size) yi -= size;
        yi_ = y - i; if (yi_ < 0) yi_ += size;

        sum += kernel[i] * (grid[y * size + xi] + grid[y * size + xi_] + grid[yi * size + x] + grid[yi_ * size + x]); // horizontals i=0 or j=0
        if (2 * i * i <= R2)
            sum += kernel[i * w + i] * (grid[yi * size + xi] + grid[yi_ * size + xi] + grid[yi * size + xi_] + grid[yi_ * size + xi_]); // diagonals i=j

        for (int j = 1; j < i; j++) // ********** i>0 and j>0
        {
            yj = y + j; if (yj >= size) yj -= size;
            yj_ = y - j; if (yj_ < 0) yj_ += size;
            xj = x + j;  if (xj >= size) xj -= size;
            xj_ = x - j; if (xj_ < 0) xj_ += size;

            if (i * i + j * j <= R2) { // dont use the slower sqrtf(i * i + j * j) <= R
                sum += kernel[i * w + j] * (grid[yj * size + xi] + grid[yj * size + xi_] + grid[yj_ * size + xi] + grid[yj_ * size + xi_]
                    + grid[yi * size + xj] + grid[yi * size + xj_] + grid[yi_ * size + xj] + grid[yi_ * size + xj_]);
            }
            else break;
        }
    }
    return sum;
}

__host__ __device__ void computeGrowth(int i, int j, float mu, float sigma2, float weight, float u_t, float* cell) {
    float ur = u_t - mu;
    float g_t = 2.0f * expf(-ur * ur / sigma2) - 1.0f;
    float c_t = weight * g_t; // field C at time step t + delta(t) ; (delta (t) = 1/T)
    *cell += c_t;
}

__host__ __device__ void computeLeniaStep(int i, int j, int radius, float mu, float sigma2, float weight, float* kernel, float* p1, int size, float* cell) {
    float u_t = convolve(i, j, p1, size, kernel, radius);
    computeGrowth(i, j, mu, sigma2, weight, u_t, cell);
}
