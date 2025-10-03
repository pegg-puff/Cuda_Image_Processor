# CUDA path (default location)
CUDA_PATH ?= /usr/local/cuda

# OpenCV flags using pkg-config
OPENCV_FLAGS := `pkg-config --cflags --libs opencv4`

# Target
all: image_proc

# Compile
image_proc: src/main.cu
	nvcc src/main.cu -o image_proc $(OPENCV_FLAGS)

# Clean
clean:
	rm -f image_proc
