#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#include "HostMode.h"
using namespace cuLenia;


HostMode::HostMode(int size, int T, int nbChannels, int nbRules, float channels[][4], float rules[][11])
{
    lw.toGridB = true;
    lw.nbChannels = nbChannels;
    lw.nbRules = nbRules;
    lw.size = size;
    lw.T = T;

    for (int i = 0;i < lw.nbChannels;i++)
    {
        lw.color[i] = float4();
        lw.color[i].x = channels[i][0]; lw.color[i].y = channels[i][1]; lw.color[i].z = channels[i][2]; lw.color[i].w = 1.f;
        lw.gridA[i] = randomGrid(lw.size, channels[i][3]);
        lw.gridB[i] = randomGrid(lw.size, 0.0f);
        lw.gridA64[i] = 0;
        lw.gridB64[i] = 0;
        lw.gridA16[i] = 0;
        lw.gridB16[i] = 0;
    }

    for (int i = 0;i < lw.nbRules;i++)
    {
        lw.rule[i].source = rules[i][0];
        lw.rule[i].destination = rules[i][1];
        lw.rule[i].radius = rules[i][2];
        lw.rule[i].mu = rules[i][8];
        lw.rule[i].sigma2 = 2 * rules[i][9] * rules[i][9];
        lw.rule[i].weight = rules[i][10];
        float b[MAX_RULES];
        b[0] = rules[i][6]; b[1] = rules[i][7];
        lw.kernel[i] = initKernel(rules[i][2], rules[i][3], rules[i][4], rules[i][5], b);
        lw.kernel64[i] = 0;
        lw.kernel16[i] = 0;
    }
    renderGrid = zeroRenderGrid(size);
}

void HostMode::init64()
{
    for (int i = 0;i < lw.nbChannels;i++) {
        lw.gridA64[i] = (double*)malloc(lw.size * lw.size * sizeof(double));
        lw.gridB64[i] = (double*)malloc(lw.size * lw.size * sizeof(double));
        for (int j = 0; j < lw.size * lw.size; j++)
        {
            lw.gridA64[i][j] = (double)lw.gridA[i][j];
            lw.gridB64[i][j] = (double)lw.gridB[i][j];
        }
    }
    for (int i = 0;i < lw.nbRules;i++) {
        lw.rule[i].mu64 = (double)lw.rule[i].mu;
        lw.rule[i].sigma264 = (double)lw.rule[i].sigma2;
        lw.rule[i].weight64 = (double)lw.rule[i].weight;
        lw.kernel64[i] = (double*)malloc((2 * lw.rule[i].radius + 1) * (2 * lw.rule[i].radius + 1) * sizeof(double));
        for (int j = 0; j < (2 * lw.rule[i].radius + 1) * (2 * lw.rule[i].radius + 1); j++) lw.kernel64[i][j] = (double)lw.kernel[i][j];
    }
}

void HostMode::init16()
{
    for (int i = 0;i < lw.nbChannels;i++) {
        lw.gridA16[i] = (__nv_bfloat16*)malloc(lw.size * lw.size * sizeof(__nv_bfloat16));
        lw.gridB16[i] = (__nv_bfloat16*)malloc(lw.size * lw.size * sizeof(__nv_bfloat16));
        for (int j = 0; j < lw.size * lw.size; j++)
        {
            lw.gridA16[i][j] = __float2bfloat16(lw.gridA[i][j]);
            lw.gridB16[i][j] = __float2bfloat16(lw.gridB[i][j]);
        }
    }
    for (int i = 0;i < lw.nbRules;i++) {
        lw.rule[i].mu16 = __float2bfloat162_rn(lw.rule[i].mu);
        lw.rule[i].sigma216 = __float2bfloat162_rn(lw.rule[i].sigma2);
        lw.rule[i].weight16 = __float2bfloat162_rn(lw.rule[i].weight);
        lw.kernel16[i] = (__nv_bfloat16*)malloc((2 * lw.rule[i].radius + 1) * (2 * lw.rule[i].radius + 1) * sizeof(__nv_bfloat16));
        for (int j = 0; j < (2 * lw.rule[i].radius + 1) * (2 * lw.rule[i].radius + 1); j++) lw.kernel16[i][j] = __float2bfloat16(lw.kernel[i][j]);
    }
}

void HostMode::compute(bool render, bool batchFFT)
{
    for (int k = 0; k < lw.nbRules; k++)
    {
        // apply rules
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
        for (int i = 0; i < lw.size; i++) {
            for (int j = 0; j < lw.size; j++) {
                computeLeniaStep(i, j, r.radius, r.mu, r.sigma2, r.weight, lw.kernel[k], g1, lw.size, g2 + i * lw.size + j);
            }
        }
    } // rules

    for (int i = 0; i < lw.size; i++) {
        for (int j = 0; j < lw.size; j++) {
            // clear renderGrid
            int index = i * lw.size + j;
            renderGrid[index].x = 0;
            renderGrid[index].y = 0;
            renderGrid[index].z = 0;
            renderGrid[index].w = 1;
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
                gridA[index] = 0;
                if (gridB[index] < 0.0f) gridB[index] = 0.0f;
                if (gridB[index] > 1.0f) gridB[index] = 1.0f;
                renderGrid[index].x += gridB[index] * lw.color[k].x;
                renderGrid[index].y += gridB[index] * lw.color[k].y;
                renderGrid[index].z += gridB[index] * lw.color[k].z;
            }
        } // j
    } // i
    lw.toGridB = !lw.toGridB;
}

void HostMode::render(int w, int h)
{
    glClear(GL_COLOR_BUFFER_BIT);
    glViewport(0, 0, w, h);
    glPixelZoom(w / (float)lw.size, h / (float)lw.size);
    glDrawPixels(lw.size, lw.size, GL_RGBA, GL_FLOAT, renderGrid);
}

HostMode::~HostMode()
{
    free(renderGrid);

    for (int i = 0; i < lw.nbRules;i++)
    {
        free(lw.kernel[i]);
        if (lw.kernel64[i]) { free(lw.kernel64[i]); lw.kernel64[i] = 0; }
        if (lw.kernel16[i]) { free(lw.kernel16[i]); lw.kernel16[i] = 0; }
    }
    for (int i = 0; i < lw.nbChannels;i++)
    {
        free(lw.gridA[i]);
        free(lw.gridB[i]);
        if (lw.gridA64[i]) { free(lw.gridA64[i]); lw.gridA64[i] = 0; }
        if (lw.gridB64[i]) { free(lw.gridB64[i]); lw.gridB64[i] = 0; }
        if (lw.gridA16[i]) { free(lw.gridA16[i]); lw.gridA16[i] = 0; }
        if (lw.gridB16[i]) { free(lw.gridB16[i]); lw.gridB16[i] = 0; }
    }
}