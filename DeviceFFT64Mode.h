#pragma once
#include "Mode.h"

namespace cuLenia {
	typedef struct {
		cufftDoubleComplex* kernelFFT;
		cufftDoubleComplex* sourceFFT;
		cufftDoubleComplex* destFFT;
		cufftDoubleReal* dest;
	} LeniaWorldFFT64; // FFT extension to LeniaWorld

	class DeviceFFT64Mode : public Mode {

	public:
		DeviceFFT64Mode(int size, int T, int nbChannels, int nbRules, float channels[][4], float rules[][11]);
		~DeviceFFT64Mode();
		void compute(bool render, bool batchFFT);
		void render(int w, int h);
	private:
		LeniaWorldFFT64 lwFFT;
		cufftHandle planD2Z, planZ2D, planManyD2Z, planManyZ2D;
		GLuint renderGridBuffer;
		GLuint renderGridTexture;
		struct cudaGraphicsResource* cuBuffer;
		double* initKernelGridSize(int r);
	};
}

