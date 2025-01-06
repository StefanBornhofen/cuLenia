#pragma once

#include "lenia.h"


namespace cuLenia {
	class Mode {
	public:
		virtual ~Mode() {};
		virtual void compute(bool render, bool batchFFT = false) = 0;
		virtual void render(int w, int h) = 0;
		LeniaWorld lw; // single precision with host or device Lenia pointers
	};
}

