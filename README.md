# cuLenia

cuLenia is a new implementation of the Lenia algorithm developed in the CUDA C++ programming language. The visualization part is done with the open-source library GLFW which represents a lightweight toolkit for
the management of windows and OpenGL contexts; however, it could easily be adapted to other rendering frameworks.

cuLenia allows configuring all standard parameters such as grid size, time resolution, growth function, and kernel shape. It simulates the simple and the expanded versions of Lenia with the possibility to define multiple kernels and channels. Focusing on performance rather than rich user interactions, cuLenia is not meant to be a finished and self-contained executable but rather a high-performance base code providing a valuable starting point for researchers who intend to explore the Lenia universe through automated strategies such as computer vision or machine learning algorithms.

Dependencies

CUDA: https://developer.nvidia.com/cuda-toolkit 

GLFW: https://www.glfw.org
