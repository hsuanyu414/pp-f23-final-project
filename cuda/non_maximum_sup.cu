#include "kernel.h"

__global__ void non_maximum_sup_kernel(
    int32_t *input, int32_t *output, double *theta,
    int start_width, int end_width, int start_height, int end_height){

    int this_x = blockIdx.x * blockDim.x + threadIdx.x;
    int this_y = blockIdx.y * blockDim.y + threadIdx.y;

    int width = end_width - start_width;

    int32_t indexMa, indexMb;
    int32_t Ma, Mb;
    double theta_tmp;

    theta_tmp = theta[this_x * width + this_y];
    if(theta_tmp == 0){
        indexMa = this_x * width + this_y -1;
        indexMb = this_x * width + this_y +1;
    }
    else if(theta_tmp == 45){
        indexMa = (this_x -1) * width + this_y +1;
        indexMb = (this_x +1) * width + this_y -1;
    }
    else if(theta_tmp == 90){
        indexMa = (this_x -1) * width + this_y;
        indexMb = (this_x +1) * width + this_y;
    }
    else if(theta_tmp == 135){
        indexMa = (this_x -1) * width + this_y -1;
        indexMb = (this_x +1) * width + this_y +1;
    }

    if(indexMa < 0 || indexMa >= width * width){
        Ma = 0;
    }
    else{
        Ma = input[indexMa];
    }

    if(indexMb < 0 || indexMb >= width * width){
        Mb = 0;
    }
    else{
        Mb = input[indexMb];
    }

    if(input[this_x * width + this_y] >= Ma && input[this_x * width + this_y] >= Mb){
        output[this_x * width + this_y] = input[this_x * width + this_y];
    }
    else{
        output[this_x * width + this_y] = 0;
    }
}

void non_maximum_sup(
    int32_t *input, int32_t *output, double *theta,
    int start_width, int end_width, int start_height, int end_height){

    int width = end_width - start_width;
    int img_size = width * width;

    int32_t *d_input, *d_output;
    double *d_theta;

    cudaMalloc((void**)&d_input, img_size * sizeof(int32_t));
    cudaMalloc((void**)&d_output, img_size * sizeof(int32_t));
    cudaMalloc((void**)&d_theta, img_size * sizeof(double));

    cudaMemcpy(d_input, input, img_size * sizeof(int32_t), cudaMemcpyHostToDevice);
    cudaMemcpy(d_theta, theta, img_size * sizeof(double), cudaMemcpyHostToDevice);

    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);
    dim3 dimGrid(width / BLOCK_SIZE, width / BLOCK_SIZE);

    if (dimBlock.x * dimBlock.y > MAX_THREADS_PER_BLOCK) {
        printf("The thread number exceeds the maximum thread number per block.\n");
        exit(1);
    }

    non_maximum_sup_kernel<<<dimGrid, dimBlock>>>(
        d_input, d_output, d_theta, start_width, end_width, start_height, end_height);

    cudaError_t err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        printf("Error: %s\n", cudaGetErrorString(err));
        exit(1);
    }

    cudaMemcpy(output, d_output, img_size * sizeof(int32_t), cudaMemcpyDeviceToHost);

    cudaFree(d_input);
    cudaFree(d_output);
    cudaFree(d_theta);
}