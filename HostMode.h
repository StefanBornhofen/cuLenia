#pragma once
#include "Mode.h"

namespace cuLenia {
	class HostMode :public Mode {
	public:
		HostMode(int size, int T, int nbChannels, int nbRules, float channels[][4], float rules[][11]);
		~HostMode();
		void compute(bool render, bool batchFFT);
		void render(int w, int h);
		void init64();
		void init16();
	private:
		float4* renderGrid;
	};
}



