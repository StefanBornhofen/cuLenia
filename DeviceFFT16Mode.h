#pragma once
#include "Mode.h"

#define DIM 16

namespace cuLenia {
	typedef struct {
		__nv_bfloat162* kernelFFT;
		__nv_bfloat162* sourceFFT;
		__nv_bfloat162* destFFT;
		__nv_bfloat16* dest;
	} LeniaWorldFFT16; // FFT extension to LeniaWorld

	class DeviceFFT16Mode : public Mode {

	public:
		DeviceFFT16Mode(int size, int T, int nbChannels, int nbRules, float channels[][4], float rules[][11]);
		~DeviceFFT16Mode();
		void compute(bool render, bool batchFFT);
		void render(int w, int h);
	private:
		LeniaWorldFFT16 lwFFT;
		cufftHandle planR2C, planC2R, planManyR2C, planManyC2R;
		GLuint renderGridBuffer;
		GLuint renderGridTexture;
		struct cudaGraphicsResource* cuBuffer;
		__nv_bfloat16* initKernelGridSize(int r);

		dim3 dimBlock, dimGrid2, dimGridFFT;
	};
}
