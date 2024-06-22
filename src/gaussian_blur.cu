#include <iostream>
#include <opencv2/opencv.hpp>
#include <cuda_runtime.h>
#include <filesystem>
#include <vector>
#include <string>

namespace fs = std::filesystem;

using namespace std;
using namespace cv;

#define BLOCK_SIZE 16

__global__ void gaussianBlurKernel(uchar *input, uchar *output, int width, int height, int *filter, int filterWidth)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int idx = y * width + x;

    if (x < width && y < height)
    {
        int halfFilterWidth = filterWidth / 2;
        int sum = 0;
        int filterSum = 0;

        for (int ky = -halfFilterWidth; ky <= halfFilterWidth; ky++)
        {
            for (int kx = -halfFilterWidth; kx <= halfFilterWidth; kx++)
            {
                int pixelX = min(max(x + kx, 0), width - 1);
                int pixelY = min(max(y + ky, 0), height - 1);

                int filterValue = filter[(ky + halfFilterWidth) * filterWidth + (kx + halfFilterWidth)];
                sum += input[pixelY * width + pixelX] * filterValue;
                filterSum += filterValue;
            }
        }

        output[idx] = sum / filterSum;
    }
}

__host__ void allocateDeviceMemory(uchar **d_input, uchar **d_output, int **d_filter, int imageSize, int filterSize)
{
    cudaMalloc(d_input, imageSize);
    cudaMalloc(d_output, imageSize);
    cudaMalloc(d_filter, filterSize);
}

__host__ void copyToDevice(uchar *d_input, uchar *input, int *d_filter, int *filter, int imageSize, int filterSize)
{
    cudaMemcpy(d_input, input, imageSize, cudaMemcpyHostToDevice);
    cudaMemcpy(d_filter, filter, filterSize, cudaMemcpyHostToDevice);
}

__host__ void copyFromDevice(uchar *output, uchar *d_output, int imageSize)
{
    cudaMemcpy(output, d_output, imageSize, cudaMemcpyDeviceToHost);
}

__host__ void freeDeviceMemory(uchar *d_input, uchar *d_output, int *d_filter)
{
    cudaFree(d_input);
    cudaFree(d_output);
    cudaFree(d_filter);
}

__host__ void executeKernel(uchar *d_input, uchar *d_output, int *d_filter, int width, int height, int filterWidth, int threadsPerBlock)
{
    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);
    dim3 dimGrid((width + BLOCK_SIZE - 1) / BLOCK_SIZE, (height + BLOCK_SIZE - 1) / BLOCK_SIZE);

    gaussianBlurKernel<<<dimGrid, dimBlock>>>(d_input, d_output, width, height, d_filter, filterWidth);
    cudaError_t err = cudaGetLastError();

    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to launch gaussianBlurKernel (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
}

__host__ float compareImages(uchar *image1, uchar *image2, int size)
{
    int diff = 0;

    for (int i = 0; i < size; ++i)
    {
        diff += abs(image1[i] - image2[i]);
    }

    float meanDiff = static_cast<float>(diff) / size;
    return (meanDiff / 255.0f) * 100.0f;
}

int main(int argc, char **argv)
{
    if (argc != 3)
    {
        cerr << "Usage: " << argv[0] << " <input_folder> <output_folder>" << endl;
        return -1;
    }

    string inputFolder = argv[1];
    string outputFolder = argv[2];

    namespace fs = std::filesystem;

    vector<string> imageFiles;
    for (const auto &entry : fs::directory_iterator(inputFolder))
    {
        if (entry.path().extension() == ".tiff")
        {
            imageFiles.push_back(entry.path().string());
        }
    }

    for (const string &imageFile : imageFiles)
    {
        Mat inputImage = imread(imageFile, IMREAD_GRAYSCALE);
        if (inputImage.empty())
        {
            cerr << "Error: Unable to open input image " << imageFile << endl;
            continue;
        }

        int width = inputImage.cols;
        int height = inputImage.rows;
        int imageSize = width * height * sizeof(uchar);

        uchar *input = inputImage.data;
        uchar *output = new uchar[width * height];

        // Define Gaussian filter
        int filterWidth = 5;
        int filter[] = {
            1, 4, 7, 4, 1,
            4, 16, 26, 16, 4,
            7, 26, 41, 26, 7,
            4, 16, 26, 16, 4,
            1, 4, 7, 4, 1
        };
        int filterSize = filterWidth * filterWidth * sizeof(int);

        uchar *d_input, *d_output;
        int *d_filter;

        // Allocate device memory
        allocateDeviceMemory(&d_input, &d_output, &d_filter, imageSize, filterSize);

        // Copy data to device
        copyToDevice(d_input, input, d_filter, filter, imageSize, filterSize);

        // Execute kernel
        executeKernel(d_input, d_output, d_filter, width, height, filterWidth, BLOCK_SIZE);

        // Copy result back to host
        copyFromDevice(output, d_output, imageSize);

        // Compare images
        float diffPercentage = compareImages(input, output, width * height);
        cout << "Image: " << imageFile << " - Difference percentage: " << diffPercentage << "%" << endl;

        // Save output image
        fs::path outputPath = fs::path(outputFolder) / fs::path(imageFile).filename();
        Mat outputImage(height, width, CV_8UC1, output);
        imwrite(outputPath.string(), outputImage);

        // Free device memory
        freeDeviceMemory(d_input, d_output, d_filter);

        delete[] output;
    }

    return 0;
}
