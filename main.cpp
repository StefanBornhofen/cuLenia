#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#include "Device32noInteropMode.h"
#include "Device32Mode.h"
#include "DeviceFFT32Mode.h"
#include "DeviceFFT64Mode.h"
#include "DeviceFFT16Mode.h"
using namespace cuLenia;
using namespace std;

#define TITLE "Lenia"
#define SCREEN_X 1024
#define SCREEN_Y 1024
#define FPS_UPDATE 1.0

#define DEVICE_NO_INTEROP_MODE 1
#define DEVICE_MODE 2
#define DEVICE_FFT_MODE 3
#define DEVICE_FFT64_MODE 4
#define DEVICE_FFT16_MODE 5

#define ORBIUM 1
#define TESSELLATIUM 2

char mode_str[20] = "";
int MODE = 0;
int SIZE = 1024;
int SCENE = TESSELLATIUM;
int RENDER_UPDATE = 25;
bool batchFFT = true;

Mode* mode;

void initOrbium()
{
    float r[1][11] = { // rules: { source,dest,radius,relRadius,rank,alpha,beta0,beta1,mu,sigma,weight}
        { 0,0,15,1.f,1,4.f,1.f,0.f,0.14f,0.014f,1.f }
    };
    float c[1][4] = { // channels : { red, green, blue, init threshold}
         { 0.f, 1.f, 0.f, 0.2 } };
    if (MODE == DEVICE_NO_INTEROP_MODE) mode = new Device32noInteropMode(SIZE, 16, 1, 1, c, r);
    if (MODE == DEVICE_MODE) mode = new Device32Mode(SIZE, 16, 1, 1, c, r);
    if (MODE == DEVICE_FFT_MODE) mode = new DeviceFFT32Mode(SIZE, 16, 1, 1, c, r);
    if (MODE == DEVICE_FFT64_MODE) mode = new DeviceFFT64Mode(SIZE, 16, 1, 1, c, r);
    if (MODE == DEVICE_FFT16_MODE) mode = new DeviceFFT16Mode(SIZE, 16, 1, 1, c, r);
}

void initTesselatium()
{
    float c[3][4] = { // channels : { red, green, blue, init threshold}
         { 1.f, 0.f, 0.f, 0.5 },
         { 0.f, 1.f, 0.f, 0.5 },
         { 0.f, 0.f, 1.f, 0.5 } };

    float r[15][11] = {  // rules: { source,dest,radius,relRadius,rank,alpha,beta0,beta1,mu,sigma,weight}
        { 0,0,12,0.91,1,4.f,1.f,0.f,0.272,0.0595,0.19 },
        { 0,0,12,0.62,1,4.f,1.f,0.f,0.349,0.1585,0.66 },
        { 0,0,12,0.5,2,4.f,1.f,1.f / 4,0.2,0.0332,0.39 },
        { 1,1,12,0.97,2,4.f,0.f,1.f,0.114,0.0528,0.38 },
        { 1,1,12,0.72,1,4.f,1.f,0.f,0.447,0.0777,0.74 },
        { 1,1,12,0.8,2,4.f,5.f / 6,1.f,0.247,0.0342,0.92 },
        { 2,2,12,0.96,1,4.f,1.f,0.f,0.21,0.0617,0.59 },
        { 2,2,12,0.56,1,4.f,1.f,0.f,0.462,0.1192,0.37 },
        { 2,2,12,0.78,1,4.f,1.f,0.f,0.446,0.1793,0.94 },
        { 0,1,12,0.79,2,4.f,11.f / 12,1.f,0.327,0.1408,0.51 },
        { 0,2,12,0.5,2,4.f, 3.f / 4,1.f,0.476,0.0995,0.77 },
        { 1,0,12,0.72,2,4.f,11.f / 12,1.f,0.379,0.0697,0.92 },
        { 1,2,12,0.68,1,4.f,1.f,0.f,0.262,0.0877,0.71 },
        { 2,0,12,0.55,2,4.f,1.f / 6,1.f,0.412,0.1101,0.59 },
        { 2,1,12,0.82,1,4.f,1.f,0.f,0.201,0.0786,0.41 } };

    if (MODE == DEVICE_NO_INTEROP_MODE) mode = new Device32noInteropMode(SIZE, 8, 3, 15, c, r);
    if (MODE == DEVICE_MODE) mode = new Device32Mode(SIZE, 8, 3, 15, c, r);
    if (MODE == DEVICE_FFT_MODE) mode = new DeviceFFT32Mode(SIZE, 8, 3, 15, c, r);
    if (MODE == DEVICE_FFT64_MODE) mode = new DeviceFFT64Mode(SIZE, 8, 3, 15, c, r);
    if (MODE == DEVICE_FFT16_MODE) mode = new DeviceFFT16Mode(SIZE, 8, 3, 15, c, r);
}

void initLenia(GLFWwindow* window, int m)
{
    MODE = m;
    switch (SCENE)
    {
    case ORBIUM: initOrbium(); break;
    case TESSELLATIUM: initTesselatium(); break;
    }

    switch (MODE)
    {
    case DEVICE_NO_INTEROP_MODE: sprintf_s(mode_str, "GPU-noInterop32"); break;
    case DEVICE_MODE: sprintf_s(mode_str, "GPU-Interop32"); break;
    case DEVICE_FFT_MODE: sprintf_s(mode_str, "GPU-FFT32"); break;
    case DEVICE_FFT64_MODE: sprintf_s(mode_str, "GPU-FFT64"); break;
    case DEVICE_FFT16_MODE: sprintf_s(mode_str, "GPU-FFT16"); break;
    }

    char t[200];
    sprintf_s(t, "%s %s %dx%d T=%d", TITLE, mode_str, mode->lw.size, mode->lw.size, mode->lw.T);
    glfwSetWindowTitle(window, t);
}

void toggle(GLFWwindow* window, int m)
{
    delete mode;
   // srand(12345);
    srand(time(NULL));
    initLenia(window, m);
}

void checkFPS(GLFWwindow* window, double* lastTime, int* fpsCount)
{
    double currentTime = glfwGetTime();
    double delta = currentTime - *lastTime;
    if (delta >= FPS_UPDATE) {
        double fps = double(*fpsCount) / delta;
        char t[200];
        int gsize = mode->lw.size * mode->lw.size;
        sprintf_s(t, "%s %s %dx%d T=%d, renderrate=%d batchFFT=%s %.2f FPS (TP=%.2f Mpx/s)", TITLE, mode_str, mode->lw.size, mode->lw.size, mode->lw.T, RENDER_UPDATE, batchFFT? "yes": "no", fps, fps * gsize / (1000000.f));
        glfwSetWindowTitle(window, t);
        *fpsCount = 0;
        *lastTime = currentTime;
    }
}

void checkRender(GLFWwindow* window, int* renderCount)
{
    int w, h;
    if (*renderCount == RENDER_UPDATE)
    {
        glfwGetFramebufferSize(window, &w, &h);
        mode->render(w, h);
        glfwSwapBuffers(window);
        *renderCount = 0;
    }
}

static void key_callback(GLFWwindow* window, int key, int scancode, int action, int mods)
{
    if (action == GLFW_PRESS)
    {
        if (key == GLFW_KEY_ESCAPE) glfwSetWindowShouldClose(window, GLFW_TRUE);
        if (key == 'O') { SCENE = ORBIUM; toggle(window, MODE); }
        if (key == 'T') { SCENE = TESSELLATIUM; toggle(window, MODE); }
        if (key == '1') toggle(window, DEVICE_NO_INTEROP_MODE);
        if (key == '2') toggle(window, DEVICE_MODE);
        if (key == '3') toggle(window, DEVICE_FFT_MODE);
        if (key == '4') toggle(window, DEVICE_FFT64_MODE);
        if (key == '5') toggle(window, DEVICE_FFT16_MODE);
        if (key == 'R') if (RENDER_UPDATE < 5) RENDER_UPDATE += 1; else RENDER_UPDATE += 5;
        if (key == 'F') if (RENDER_UPDATE > 1) { if (RENDER_UPDATE < 10) RENDER_UPDATE -= 1; else RENDER_UPDATE -= 5; }
        if (key == 'B') batchFFT = !batchFFT;
        if (key == GLFW_KEY_ENTER) toggle(window, MODE);
        if (key == GLFW_KEY_UP) { SIZE *= 2; toggle(window, MODE); }
        if (key == GLFW_KEY_DOWN) if (SIZE > 64) { SIZE /= 2; toggle(window, MODE); }
        if (key == GLFW_KEY_RIGHT) if (mode->lw.T < 4) mode->lw.T += 1; else { mode->lw.T *= 2; }
        if (key == GLFW_KEY_LEFT) if (mode->lw.T >= 2) { if (mode->lw.T <= 4) mode->lw.T -= 1; else mode->lw.T /= 2; }
    }
}

static void error_callback(int error, const char* description)
{
    fprintf(stderr, "GLFW error: %s\n", description);
}

int main(void)
{
    srand(time(NULL));
    if (!glfwInit()) return -1;
 
    GLFWwindow* window = glfwCreateWindow(SCREEN_X, SCREEN_Y, TITLE, NULL, NULL);
    if (!window)
    {
        glfwTerminate();
        return -1;
    }

    glfwMakeContextCurrent(window);
    glfwSwapInterval(0); // disable vsync

    GLint GlewInitResult = glewInit();
    if (GlewInitResult != GLEW_OK) {
        printf("ERROR: %s\n", glewGetErrorString(GlewInitResult));
    }

    glfwSetKeyCallback(window, key_callback);
    glfwSetErrorCallback(error_callback);

    toggle(window, DEVICE_FFT16_MODE);

    double lastTime = glfwGetTime();
    int fpsCount = 0;
    int renderCount = 0;

    while (!glfwWindowShouldClose(window))
    {
        fpsCount++;
        renderCount++;
        mode->compute(renderCount == RENDER_UPDATE, batchFFT);
        checkFPS(window, &lastTime, &fpsCount);
        checkRender(window, &renderCount);
        glfwPollEvents();
    }

    toggle(window, 0);
    glfwDestroyWindow(window);
    glfwTerminate();
    return 0;
}
