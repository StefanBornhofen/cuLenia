#pragma once
#include "Mode.h"

namespace cuLenia {
	typedef struct {
		cufftComplex* kernelFFT;
		cufftComplex* sourceFFT;
		cufftComplex* destFFT;
		cufftReal* dest;
	} LeniaWorldFFT; // FFT extension to LeniaWorld

	class DeviceFFT32Mode : public Mode {

	public:
		DeviceFFT32Mode(int size, int T, int nbChannels, int nbRules, float channels[][4], float rules[][11]);
		~DeviceFFT32Mode();
		void compute(bool render, bool batchFFT);
		void render(int w, int h);
	private:
		LeniaWorldFFT lwFFT;
		cufftHandle planR2C, planC2R, planManyR2C, planManyC2R;
		GLuint renderGridBuffer;
		GLuint renderGridTexture;
		struct cudaGraphicsResource* cuBuffer;
		float* initKernelGridSize(int r);
	};
}
