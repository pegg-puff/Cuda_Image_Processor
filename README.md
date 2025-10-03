# CUDA Image Processor

## Description
This project performs GPU-accelerated image processing on multiple images.  
It applies:
1. Color Quantization (reduce color levels)
2. Sobel Edge Detection

## How to Run
1. Place input images in `data/`.
2. Compile: `make`
3. Run: `./run.sh`
4. Output images are saved in `output/`

## Dependencies
- CUDA toolkit
- OpenCV
