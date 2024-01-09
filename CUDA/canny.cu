#include "kernel.h"

__global__ void conv8_kernel(
    uint8_t *input, float *kernel, uint8_t *output, int start_width, 
    int end_width, int start_height, int end_height, int kernel_size){

    __shared__ float shared_kernel[9];
    for(int i = 0; i < 9; i++){
        shared_kernel[i] = kernel[i];
    }

    __syncthreads();

    int this_x = blockIdx.x * blockDim.x + threadIdx.x;
    int this_y = blockIdx.y * blockDim.y + threadIdx.y;

    if(this_x < 0 || this_x >= end_width || this_y < 0 || this_y >= end_height){
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
            /* zero padding */
            if(x < 0 || x >= width || y < 0 || y >= height){
                sum += 0;
            }
            else{
                sum += shared_kernel[i + kernel_size * j] * float(input[x + width * y]);
            }
        }
    }
    /* clipping */
    sum = sum < 0 ? 0 : sum;
    sum = sum > 255 ? 255 : sum;
    output[this_x + width * this_y] = (uint8_t)sum;
}

__global__ void conv32_kernel(
    uint8_t *input, float *kernel, int32_t *output, int start_width, 
    int end_width, int start_height, int end_height, int kernel_size){

    __shared__ float shared_kernel[9];
    for(int i = 0; i < 9; i++){
        shared_kernel[i] = kernel[i];
    }

    __syncthreads();

    int this_x = blockIdx.x * blockDim.x + threadIdx.x;
    int this_y = blockIdx.y * blockDim.y + threadIdx.y;

    if(this_x < 0 || this_y < 0 || this_x >= end_width || this_y >= end_height){
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
                sum += shared_kernel[i + kernel_size * j] * float(input[x + width * y]);
        }
    }

    output[this_x + width * this_y] = (int32_t)sum;
}

__global__ void grad_cal_kernel(
    int32_t *gx, int32_t *gy, int32_t *output, int start_width, 
    int end_width, int start_height, int end_height){

    int this_x = blockIdx.x * blockDim.x + threadIdx.x;
    int this_y = blockIdx.y * blockDim.y + threadIdx.y;

    if(this_x >= end_width || this_y >= end_height){
        return;
    }

    int width = end_width - start_width;

    int32_t tmp_idx = this_x + width * this_y;
    int32_t grad_mag = int32_t(sqrt(gx[tmp_idx] * gx[tmp_idx] + 
                                    gy[tmp_idx] * gy[tmp_idx]));
    output[tmp_idx] = grad_mag;
}

__global__ void theta_cal_kernel(
    int32_t *gx, int32_t *gy, double *output, int start_width,
    int end_width, int start_height, int end_height){

    int this_x = blockIdx.x * blockDim.x + threadIdx.x;
    int this_y = blockIdx.y * blockDim.y + threadIdx.y;

    if(this_x >= end_width || this_y >= end_height){
        return;
    }

    int width = end_width - start_width;

    double theta = 0.0;
    theta = atan2(gy[this_x + width * this_y], gx[this_x + width * this_y]);
    theta = theta * 180 / PI;

    if(theta < 0) theta += 180;
    theta = 180 - theta;
    /* avoid negative zero problem */
    theta = abs(theta);

    /* classify theta angle */
    if(theta >= 0 && theta < 22.5) theta = 0;
    else if(theta >= 22.5 && theta < 67.5) theta = 45;
    else if(theta >= 67.5 && theta < 112.5) theta = 90;
    else if(theta >= 112.5 && theta < 157.5) theta = 135;
    else if(theta >= 157.5 && theta <= 180) theta = 0;

    output[this_x + width * this_y] = theta;
}

__global__ void non_maximum_sup_kernel(
    int32_t *input, int32_t *output, double *theta,
    int start_width, int end_width, int start_height, int end_height){

    int this_x = blockIdx.x * blockDim.x + threadIdx.x;
    int this_y = blockIdx.y * blockDim.y + threadIdx.y;

    int width = end_width - start_width;
    int height = end_height - start_height;

    int32_t indexMa, indexMb;
    int32_t Ma, Mb;
    double theta_tmp;
    int32_t this_index = this_x + this_y * width;

    if(this_x < 0 || this_x >= width || this_y < 0 || this_y >= height){
        return;
    }

    /* set index of corresponding theta angle */
    theta_tmp = theta[this_index];
    if(theta_tmp == 90){
        indexMa = this_x + (this_y -1) * width;
        indexMb = this_x + (this_y +1) * width;
    }
    else if(theta_tmp == 135){
        indexMa = (this_x -1) + (this_y -1) * width;
        indexMb = (this_x +1) + (this_y +1) * width;
    }
    else if(theta_tmp == 0){
        indexMa = (this_x -1) + this_y * width;
        indexMb = (this_x +1) + this_y * width;
    }
    else if(theta_tmp == 45){
        indexMa = (this_x -1) + (this_y +1) * width;
        indexMb = (this_x +1) + (this_y -1) * width;
    }

    if(indexMa < 0 || indexMa >= width * height){
        Ma = 0;
    }
    else{
        Ma = input[indexMa];
    }

    if(indexMb < 0 || indexMb >= width * height){
        Mb = 0;
    }
    else{
        Mb = input[indexMb];
    }

    /* check the magnitude of the pixel on corresponding direction */
    if(input[this_index] >= Ma && input[this_index] >= Mb){
        output[this_index] = input[this_index];
    }
    else{
        output[this_index] = 0;
    }
}

void canny_edge_detection(
    uint8_t *input, int32_t *output, int start_width, 
    int end_width, int start_height, int end_height
){
    int width = end_width - start_width;
    int height = end_height - start_height;

    int32_t img_size = width * height;

    uint8_t *d_input;
    cudaMalloc((void **)&d_input, img_size * sizeof(uint8_t));
    cudaMemcpy(d_input, input, img_size * sizeof(uint8_t), cudaMemcpyHostToDevice);

    /* 
     * step1: smoothing 
     * use 3x3 Gaussian filter
     */
    uint8_t *d_smoothed_img;
    float *d_kernel;
    float G[9] = {1.0/16, 2.0/16, 1.0/16, 2.0/16, 4.0/16, 2.0/16, 1.0/16, 2.0/16, 1.0/16};
    cudaMalloc((void **)&d_smoothed_img, img_size * sizeof(uint8_t));
    cudaMalloc((void **)&d_kernel, 9 * sizeof(float));
    cudaMemcpy(d_kernel, G, 9 * sizeof(float), cudaMemcpyHostToDevice);

    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);
    dim3 dimGrid(width / dimBlock.x + 1, height / dimBlock.y + 1);

    if (dimBlock.x * dimBlock.y > MAX_THREADS_PER_BLOCK) {
        printf("The thread number exceeds the maximum thread number per block.\n");
        exit(1);
    }

    conv8_kernel<<<dimGrid, dimBlock>>>(
        d_input, d_kernel, d_smoothed_img, start_width, end_width, start_height, end_height, 3
    );

    cudaError_t err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        printf("Error: %s\n", cudaGetErrorString(err));
        exit(1);
    }

    cudaFree(d_input);
    cudaFree(d_kernel);

    /* 
     * step2: gradient 
     * use 3x3 Sobel filter
     */
    int32_t *gx, *gy;
    cudaMalloc((void **)&gx, img_size * sizeof(int32_t));
    cudaMalloc((void **)&gy, img_size * sizeof(int32_t));

    float Sx[9] = {-1.0, 0.0, 1.0, -2.0, 0.0, 2.0, -1.0, 0.0, 1.0};
    float Sy[9] = {-1.0, -2.0, -1.0, 0.0, 0.0, 0.0, 1.0, 2.0, 1.0};

    float *d_Sx, *d_Sy;
    cudaMalloc((void **)&d_Sx, 9 * sizeof(float));
    cudaMalloc((void **)&d_Sy, 9 * sizeof(float));
    cudaMemcpy(d_Sx, Sx, 9 * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_Sy, Sy, 9 * sizeof(float), cudaMemcpyHostToDevice);

    conv32_kernel<<<dimGrid, dimBlock>>>(
        d_smoothed_img, d_Sx, gx, start_width, end_width, start_height, end_height, 3
    );

    conv32_kernel<<<dimGrid, dimBlock>>>(
        d_smoothed_img, d_Sy, gy, start_width, end_width, start_height, end_height, 3
    );

    err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        printf("Error: %s\n", cudaGetErrorString(err));
        exit(1);
    }

    cudaFree(d_smoothed_img);

    /* 
     * step3: gradient magnitude 
     * M = sqrt(gx^2 + gy^2)
     */
    int32_t *d_grad_mag;
    cudaMalloc((void **)&d_grad_mag, img_size * sizeof(int32_t));

    grad_cal_kernel<<<dimGrid, dimBlock>>>(
        gx, gy, d_grad_mag, start_width, end_width, start_height, end_height
    );

    /* 
     * step4: theta 
     * theta = arctan(gy / gx)
     * classify theta angle into 0, 45, 90, 135 degree
     */
    double *d_theta;
    cudaMalloc((void **)&d_theta, img_size * sizeof(double));

    theta_cal_kernel<<<dimGrid, dimBlock>>>(
        gx, gy, d_theta, start_width, end_width, start_height, end_height
    );

    err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        printf("Error: %s\n", cudaGetErrorString(err));
        exit(1);
    }

    cudaFree(gx);
    cudaFree(gy);

    /* 
     * step5: non-maximum suppression 
     * if the pixel is not the maximum in its neighborhood, set it to 0
     * otherwise, keep it
     */
    int32_t *d_suppressed_img;
    cudaHostRegister(output, img_size * sizeof(int32_t), cudaHostRegisterDefault);
    cudaHostGetDevicePointer(&d_suppressed_img, output, 0);

    non_maximum_sup_kernel<<<dimGrid, dimBlock>>>(
        d_grad_mag, d_suppressed_img, d_theta, start_width, end_width, start_height, end_height
    );

    err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        printf("Error: %s\n", cudaGetErrorString(err));
        exit(1);
    }

    cudaMemcpy(output, d_suppressed_img, img_size * sizeof(int32_t), cudaMemcpyDeviceToHost);

    cudaFree(d_grad_mag);
    cudaFree(d_theta);
    cudaHostUnregister(d_suppressed_img);
}