#pragma once
#include "Mode.h"

namespace cuLenia {
	class Device32Mode :public Mode {
	public:
		Device32Mode(int size, int T, int nbChannels, int nbRules, float channels[][4], float rules[][11]);
		~Device32Mode();
		void compute(bool render, bool batchFFT);
		void render(int w, int h);
	private:
		GLuint renderGridBuffer;
		GLuint renderGridTexture;
		struct cudaGraphicsResource* cuBuffer;
	};
}
#pragma once
