#include "kernel.h"

__global__ void grad_cal_kernel(
    int32_t *gx, int32_t *gy, int32_t *output, int start_width, 
    int end_width, int start_height, int end_height){

    int this_x = blockIdx.x * blockDim.x + threadIdx.x;
    int this_y = blockIdx.y * blockDim.y + threadIdx.y;

    int width = end_width - start_width;

    int32_t tmp_idx = this_x * width + this_y;
    int32_t grad_mag = int32_t(sqrt(gx[tmp_idx] * gx[tmp_idx] + 
                                    gy[tmp_idx] * gy[tmp_idx]));
    output[tmp_idx] = grad_mag;
}

void grad_cal(
    int32_t *gx, int32_t *gy, int32_t *output, int start_width, 
    int end_width, int start_height, int end_height){

    int width = end_width - start_width;
    int height = end_height - start_height;
    int img_size = width * height;

    int32_t *d_gx, *d_gy, *d_output;
    cudaMalloc((void **)&d_gx, img_size * sizeof(int32_t));
    cudaMalloc((void **)&d_gy, img_size * sizeof(int32_t));
    cudaMalloc((void **)&d_output, img_size * sizeof(int32_t));

    cudaMemcpy(d_gx, gx, img_size * sizeof(int32_t), cudaMemcpyHostToDevice);
    cudaMemcpy(d_gy, gy, img_size * sizeof(int32_t), cudaMemcpyHostToDevice);

    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);
    dim3 dimGrid(width / BLOCK_SIZE, height / BLOCK_SIZE);

    if (dimBlock.x * dimBlock.y > MAX_THREADS_PER_BLOCK) {
        printf("The thread number exceeds the maximum thread number per block.\n");
        exit(1);
    }

    grad_cal_kernel<<<dimGrid, dimBlock>>>(
        d_gx, d_gy, d_output, start_width, end_width, start_height, end_height);

    cudaError_t err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        printf("Error: %s\n", cudaGetErrorString(err));
        exit(1);
    }

    cudaMemcpy(output, d_output, img_size * sizeof(int32_t), cudaMemcpyDeviceToHost);

    cudaFree(d_gx);
    cudaFree(d_gy);
    cudaFree(d_output);
}