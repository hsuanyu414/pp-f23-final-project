#include "kernel.h"

__global__ void theta_cal_kernel(
    int32_t *gx, int32_t *gy, double *output, int start_width,
    int end_width, int start_height, int end_height){

    int this_x = blockIdx.x * blockDim.x + threadIdx.x;
    int this_y = blockIdx.y * blockDim.y + threadIdx.y;

    int width = end_width - start_width;

    double theta = 0.0;
    /* TODO: cannot call atan2 (host function) from kernel function */
    theta = atan2(gy[this_x * width + this_y], gx[this_x * width + this_y]);
    theta = theta * 180 / PI;

    if(theta < 0) theta += 180;
    theta = 180 - theta;

    if(theta >= 0 && theta < 22.5) theta = 0;
    else if(theta >= 22.5 && theta < 67.5) theta = 45;
    else if(theta >= 67.5 && theta < 112.5) theta = 90;
    else if(theta >= 112.5 && theta < 157.5) theta = 135;
    else if(theta >= 157.5 && theta <= 180) theta = 0;

    output[this_x * width + this_y] = theta;
}

void theta_cal(
    int32_t *gx, int32_t *gy, double *output, int start_width,
    int end_width, int start_height, int end_height){

    int width = end_width - start_width;
    int height = end_height - start_height;

    int img_size = width * height;
    int32_t *d_gx, *d_gy;
    double *d_output;

    cudaMalloc((void**)&d_gx, img_size * sizeof(int32_t));
    cudaMalloc((void**)&d_gy, img_size * sizeof(int32_t));
    cudaMalloc((void**)&d_output, img_size * sizeof(double));

    cudaMemcpy(d_gx, gx, img_size * sizeof(int32_t), cudaMemcpyHostToDevice);
    cudaMemcpy(d_gy, gy, img_size * sizeof(int32_t), cudaMemcpyHostToDevice);

    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);
    dim3 dimGrid(width / BLOCK_SIZE, height / BLOCK_SIZE);
    
    if (dimBlock.x * dimBlock.y > MAX_THREADS_PER_BLOCK) {
        printf("The thread number exceeds the maximum thread number per block.\n");
        exit(1);
    }

    theta_cal_kernel<<<dimGrid, dimBlock>>>(
        d_gx, d_gy, d_output, start_width, end_width, start_height, end_height);

    cudaError_t err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        printf("Error: %s\n", cudaGetErrorString(err));
        exit(1);
    }

    cudaMemcpy(output, d_output, img_size * sizeof(double), cudaMemcpyDeviceToHost);

    cudaFree(d_gx);
    cudaFree(d_gy);
    cudaFree(d_output);
}
