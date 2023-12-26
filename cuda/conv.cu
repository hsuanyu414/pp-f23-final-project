#include "kernel.h"

__global__ void conv8_kernel(
    uint8_t *input, float *kernel, uint8_t *output, int start_width, 
    int end_width, int start_height, int end_height, int kernel_size){

    int this_x = blockIdx.x * blockDim.x + threadIdx.x;
    int this_y = blockIdx.y * blockDim.y + threadIdx.y;

    if(this_x >= end_width || this_y >= end_height){
        return;
    }

    int height = end_height - start_height;
    int width = end_width - start_width;

    int kernel_half = kernel_size / 2;

    float sum = 0;
    for(int i = 0; i < kernel_size; i++){
        for(int j = 0; j < kernel_size; j++){
            int x = this_x + i - kernel_half;
            int y = this_y + j - kernel_half;
            if(x < 0 || x >= width || y < 0 || y >= height){
                sum += 0;
            }
            else{
                sum += kernel[i * kernel_size + j] * float(input[x * width + y]);
            }
        }
    }
    sum = sum < 0 ? 0 : sum;
    sum = sum > 255 ? 255 : sum;
    output[this_x * width + this_y] = (uint8_t)sum;
}

__global__ void conv32_kernel(
    uint8_t *input, float *kernel, int32_t *output, int start_width, 
    int end_width, int start_height, int end_height, int kernel_size){

    int this_x = blockIdx.x * blockDim.x + threadIdx.x;
    int this_y = blockIdx.y * blockDim.y + threadIdx.y;

    if(this_x >= end_width || this_y >= end_height){
        return;
    }

    int height = end_height - start_height;
    int width = end_width - start_width;

    int kernel_half = kernel_size / 2;

    float sum = 0;
    for(int i = 0; i < kernel_size; i++){
        for(int j = 0; j < kernel_size; j++){
            int x = this_x + i - kernel_half;
            int y = this_y + j - kernel_half;
            if(x < 0 || x >= width || y < 0 || y >= height)
                sum += 0;
            else
                sum += kernel[i * kernel_size + j] * float(input[x * width + y]);
        }
    }

    output[this_x * width + this_y] = (int32_t)sum;
}

void conv8(
    uint8_t *input, float *kernel, uint8_t *output, int start_width, 
    int end_width, int start_height, int end_height, int kernel_size){
    
    int width = end_width - start_width;
    int height = end_height - start_height;
    int img_size = width * height;
    int k_size = kernel_size * kernel_size;

    uint8_t *d_input, *d_output;
    float *d_kernel;

    cudaMalloc((void **)&d_input, img_size * sizeof(uint8_t));
    cudaMalloc((void **)&d_output, img_size * sizeof(uint8_t));
    cudaMalloc((void **)&d_kernel, k_size * sizeof(float));

    cudaMemcpy(d_input, input, img_size * sizeof(uint8_t), cudaMemcpyHostToDevice);
    cudaMemcpy(d_kernel, kernel, k_size * sizeof(float), cudaMemcpyHostToDevice);

    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);
    dim3 dimGrid(width / dimBlock.x, height / dimBlock.y);

    if (dimBlock.x * dimBlock.y > MAX_THREADS_PER_BLOCK) {
        printf("The thread number exceeds the maximum thread number per block.\n");
        exit(1);
    }

    conv8_kernel<<<dimGrid, dimBlock>>>(d_input, d_kernel, d_output, start_width, end_width, start_height, end_height, kernel_size);

    cudaError_t err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        printf("Error: %s\n", cudaGetErrorString(err));
        exit(1);
    }

    cudaMemcpy(output, d_output, img_size * sizeof(uint8_t), cudaMemcpyDeviceToHost);

    cudaFree(d_input);
    cudaFree(d_output);
    cudaFree(d_kernel);
}

void conv32(
    uint8_t *input, float *kernel, int32_t *output, int start_width, 
    int end_width, int start_height, int end_height, int kernel_size){
    
    int width = end_width - start_width;
    int height = end_height - start_height;
    int img_size = width * height;
    int k_size = kernel_size * kernel_size;

    uint8_t *d_input;
    int32_t *d_output;
    float *d_kernel;

    cudaMalloc((void **)&d_input, img_size * sizeof(uint8_t));
    cudaMalloc((void **)&d_output, img_size * sizeof(int32_t));
    cudaMalloc((void **)&d_kernel, k_size * sizeof(float));

    cudaMemcpy(d_input, input, img_size * sizeof(uint8_t), cudaMemcpyHostToDevice);
    cudaMemcpy(d_kernel, kernel, k_size * sizeof(float), cudaMemcpyHostToDevice);

    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);
    dim3 dimGrid(width / dimBlock.x, height / dimBlock.y);

    if (dimBlock.x * dimBlock.y > MAX_THREADS_PER_BLOCK) {
        printf("The thread number exceeds the maximum thread number per block.\n");
        exit(1);
    }

    conv32_kernel<<<dimGrid, dimBlock>>>(d_input, d_kernel, d_output, start_width, end_width, start_height, end_height, kernel_size);

    cudaError_t err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        printf("Error: %s\n", cudaGetErrorString(err));
        exit(1);
    }

    cudaMemcpy(output, d_output, img_size * sizeof(int32_t), cudaMemcpyDeviceToHost);

    cudaFree(d_input);
    cudaFree(d_output);
    cudaFree(d_kernel);
}