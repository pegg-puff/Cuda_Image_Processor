#include <opencv2/opencv.hpp>
#include <iostream>
#include <filesystem>
#include <cmath>

namespace fs = std::filesystem;

// ---------------- CUDA KERNELS ----------------

// Color Quantization Kernel
__global__ void colorQuantizationKernel(unsigned char* input, unsigned char* output, int width, int height, int channels, int levels) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < width && y < height) {
        int idx = (y * width + x) * channels;
        for (int c = 0; c < channels; c++) {
            float scale = 255.0f / (levels - 1);
            output[idx + c] = roundf(input[idx + c] / scale) * scale;
        }
    }
}

// Sobel Edge Detection Kernel (grayscale input)
__global__ void sobelKernel(unsigned char* input, unsigned char* output, int width, int height) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x > 0 && y > 0 && x < width - 1 && y < height - 1) {
        int Gx[3][3] = {{-1,0,1},{-2,0,2},{-1,0,1}};
        int Gy[3][3] = {{1,2,1},{0,0,0},{-1,-2,-1}};

        float sumX = 0;
        float sumY = 0;

        for(int i=-1;i<=1;i++){
            for(int j=-1;j<=1;j++){
                int pixel = input[(y+i)*width + (x+j)];
                sumX += Gx[i+1][j+1] * pixel;
                sumY += Gy[i+1][j+1] * pixel;
            }
        }

        // Compute gradient magnitude
        float val = sqrtf(sumX*sumX + sumY*sumY);

        // Normalize to 0-255 for visibility
        val = val / 8.0f; // divide by max possible gradient (for 8-bit image)
        if (val > 255.0f) val = 255.0f;

        output[y*width + x] = (unsigned char)val;
    }
}

// ---------------- MAIN PROGRAM ----------------
int main() {
    std::string input_folder = "data/";
    std::string output_folder = "output/";
    int quant_levels = 8;

    // Create output folder if not exists
    cv::utils::fs::createDirectory(output_folder);

    for (const auto & entry : fs::directory_iterator(input_folder)) {
        std::string path = entry.path().string();
        // Skip hidden files like .DS_Store
        if(entry.path().filename().string()[0] == '.') continue;

        cv::Mat input = cv::imread(path);
        if (input.empty()) {
            std::cout << "Failed to load " << path << std::endl;
            continue;
        }

        int width = input.cols;
        int height = input.rows;
        int channels = input.channels();

        cv::Mat quant_output(height, width, input.type());
        cv::Mat gray(height, width, CV_8UC1);
        cv::Mat edge_output(height, width, CV_8UC1);

        // Allocate CUDA memory
        unsigned char *d_input, *d_quant, *d_edge;
        cudaMalloc(&d_input, width * height * channels * sizeof(unsigned char));
        cudaMalloc(&d_quant, width * height * channels * sizeof(unsigned char));
        cudaMalloc(&d_edge, width * height * sizeof(unsigned char));

        // Copy image to device
        cudaMemcpy(d_input, input.data, width * height * channels * sizeof(unsigned char), cudaMemcpyHostToDevice);

        // Launch color quantization kernel
        dim3 block(16,16);
        dim3 grid((width + block.x - 1)/block.x, (height + block.y - 1)/block.y);
        colorQuantizationKernel<<<grid, block>>>(d_input, d_quant, width, height, channels, quant_levels);
        cudaDeviceSynchronize();

        // Copy quantized image back
        cudaMemcpy(quant_output.data, d_quant, width * height * channels * sizeof(unsigned char), cudaMemcpyDeviceToHost);

        // Convert to grayscale
        cv::cvtColor(quant_output, gray, cv::COLOR_BGR2GRAY);
        cudaMemcpy(d_edge, gray.data, width * height * sizeof(unsigned char), cudaMemcpyHostToDevice);

        // Launch Sobel kernel
        sobelKernel<<<grid, block>>>(d_edge, d_edge, width, height);
        cudaDeviceSynchronize();

        // Copy edge image back to host
        cudaMemcpy(edge_output.data, d_edge, width * height * sizeof(unsigned char), cudaMemcpyDeviceToHost);

        // Save output
        std::string filename = output_folder + entry.path().filename().string();
        cv::imwrite(filename, edge_output);

        // Free memory
        cudaFree(d_input);
        cudaFree(d_quant);
        cudaFree(d_edge);

        std::cout << "Processed " << path << std::endl;
    }

    return 0;
}
