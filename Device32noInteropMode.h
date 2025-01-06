#pragma once
#include "Mode.h"

namespace cuLenia {
	class Device32noInteropMode :public Mode {
	public:
		Device32noInteropMode(int size, int T, int nbChannels, int nbRules, float channels[][4], float rules[][11]);
		~Device32noInteropMode();
		void compute(bool render, bool batchFFT);
		void render(int w, int h);
	private:
		float4* cuPixelsHost;
		float4* cuPixels;
	};
}
#pragma once
