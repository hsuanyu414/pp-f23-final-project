#include <iostream>
#include <stdlib.h>
#include <stdio.h>
#include "bmp.h"
#include <cmath>
#include <algorithm>
#include <queue>
#include "../common/CycleTimer.h"

#include <omp.h>
// read an rgb bmp image and transfer it to gray image

#define PI 3.14159265

using namespace std;

int width, height;
int thread_used;

void conv(
    uint8_t *input, 
    float *kernel, 
    uint8_t *output, 
    int start_width, int end_width, 
    int start_height, int end_height, 
    int kernel_size){

    float temp_pixel = 0;
    int indexi, indexj;
    printf("convolution start!\n");
    // conv with 0 padding
    
    #pragma omp parallel for private(temp_pixel, indexi, indexj) schedule(dynamic)
    for(int i = start_height ; i < end_height ; i++){
        for(int j = start_width; j < end_width ; j++){
            temp_pixel = 0;
            for(int k = 0 ; k < kernel_size ; k++){
                for(int l = 0 ; l < kernel_size ; l++){
                    indexi = i - 1 + k;
                    indexj = j - 1 + l;
                    if(indexi < 0 || indexi >= height || indexj < 0 || indexj >= width)
                        temp_pixel += 0.0;
                    else
                        temp_pixel += float(input[(indexi) * width + (indexj)] * kernel[k * kernel_size + l]);
                }
            }
            // to clamp the value to 0~255 to prevent overflow
            if(temp_pixel < 0)
                temp_pixel = 0;
            else if(temp_pixel > 255)
                temp_pixel = 255;
            // return the result back to output
            output[i * width + j] = uint8_t(temp_pixel);
        }
    }
    printf("convolution done!\n");
}

void conv2(
    uint8_t *input, 
    float *kernel, 
    int32_t *output, 
    int start_width, int end_width, 
    int start_height, int end_height, 
    int kernel_size){
    // conv with int32_t output

    float temp_pixel = 0;
    printf("convolution start!\n");
    int indexi, indexj;

    #pragma omp parallel for private(temp_pixel, indexi, indexj) schedule(dynamic)
    for(int i = start_height ; i < end_height ; i++){
        for(int j = start_width; j < end_width ; j++){
            temp_pixel = 0;
            for(int k = 0 ; k < kernel_size ; k++){
                for(int l = 0 ; l < kernel_size ; l++){
                    indexi = i - 1 + k;
                    indexj = j - 1 + l;
                    if(indexi < 0 || indexi >= height || indexj < 0 || indexj >= width)
                        temp_pixel += 0.0;
                    else
                    {
                        temp_pixel += float((input[(indexi) * width + (indexj)]) * kernel[k * kernel_size + l]);
                    }
                }
            }
            // return the result back to output
            output[i * width + j] = int32_t(temp_pixel);
        }
    }       
    printf("convolution done!\n");
}

void grad_cal(
        int32_t *gx, int32_t *gy, 
        int32_t *output, 
        int start_width, int end_width, 
        int start_height, int end_height){
    int32_t temp_pixel = 0, temp_index = 0;
    printf("gradient calculation start!\n");

    #pragma omp parallel for private(temp_pixel, temp_index) schedule(dynamic)
    for(int i = start_height ; i < end_height ; i++){
        for(int j = start_width; j < end_width ; j++){
            temp_index = i * width + j;
            temp_pixel = int32_t(sqrt(gx[temp_index] * gx[temp_index] + gy[temp_index] * gy[temp_index]));
            output[temp_index] = temp_pixel;
        }
    }
    printf("gradient calculation done!\n");
}

void theta_cal(
        int32_t *gx, int32_t *gy, 
        double *output, 
        int start_width, int end_width, 
        int start_height, int end_height){
    double temp_pixel = 0;
    printf("theta calculation start!\n");

    #pragma omp parallel for private(temp_pixel) schedule(dynamic)
    for(int i = start_height ; i < end_height ; i++){
        for(int j = start_width; j < end_width ; j++){
            temp_pixel = atan2(gy[i * width + j], gx[i * width + j]);
            // turn the radian to degree
            temp_pixel = temp_pixel * 180 / PI;

            // realign the theta offset due to direction of sobel
            if(temp_pixel < 0)
                temp_pixel += 180;
            temp_pixel = 180 - temp_pixel;

            if(temp_pixel >= 0 && temp_pixel < 22.5)
                temp_pixel = 0;
            else if(temp_pixel >= 22.5 && temp_pixel < 67.5)
                temp_pixel = 45;
            else if(temp_pixel >= 67.5 && temp_pixel < 112.5)
                temp_pixel = 90;
            else if(temp_pixel >= 112.5 && temp_pixel < 157.5)
                temp_pixel = 135;
            else if(temp_pixel >= 157.5 && temp_pixel < 180)
                temp_pixel = 0;
            output[i * width + j] = temp_pixel;
        }
    }
    printf("theta calculation done!\n");
}


void non_maximum_sup(
        int32_t *input, int32_t* output, 
        double* theta, 
        int start_width, int end_width, 
        int start_height, int end_height){
    int32_t indexMa, indexMb;
    int32_t Ma, Mb;
    double theta_temp;

    #pragma omp parallel for private(indexMa, indexMb, Ma, Mb, theta_temp) schedule(dynamic)
    for(int i = start_height ; i < end_height ; i++){
        for(int j = start_width; j < end_width ; j++){
            theta_temp = theta[i * width + j];
            if(theta_temp == 0){
                indexMa = i * width + j - 1;
                indexMb = i * width + j + 1;
            }
            else if(theta_temp == 45){
                indexMa = (i - 1) * width + j + 1;
                indexMb = (i + 1) * width + j - 1;
            }
            else if(theta_temp == 90){
                indexMa = (i - 1) * width + j;
                indexMb = (i + 1) * width + j;
            }
            else if(theta_temp == 135){
                indexMa = (i - 1) * width + j - 1;
                indexMb = (i + 1) * width + j + 1;
            }
            if(indexMa < 0 || indexMa >= height * width)
                Ma = 0;
            else
                Ma = input[indexMa];
            if(indexMb < 0 || indexMb >= height * width)
                Mb = 0;
            else
                Mb = input[indexMb];
            if(input[i * width + j] >= Ma && input[i * width + j] >= Mb)
                output[i * width + j] = input[i * width + j];
            else
                output[i * width + j] = 0;
        }
    }
    printf("non-maximum suppression done!\n");
}

int32_t Th, Tl;
queue<int32_t> q;
int32_t visit_count[4] = {0};

void edge_linking(
        int32_t *input, 
        int32_t* output,
        int32_t* visited, 
        int start_width, int end_width, 
        int start_height, int end_height){
    int32_t index;
    int32_t temp;
    // queue of each thread
    queue<int32_t> q_omp[thread_used];
    while(!q.empty()){
        // distribute the queue to each thread
        int count = 0;
        while(!q.empty()){
            q_omp[count].push(q.front());
            q.pop();
            count = (count + 1) % thread_used;
        }
        #pragma omp parallel for private(index) schedule(dynamic)
        for(int i = 0 ; i < thread_used ; i++){
            while(!q_omp[i].empty()){
                index = q_omp[i].front();
                q_omp[i].pop();
                if(visited[index] == 0){
                    // visit_count[i] += 1;
                    visited[index] = 1;
                    if(input[index] >= Tl){
                        // since the origin q only push the pixel with value >= Th, 
                        // any pixel in queue must be visited after an strong edge pixel
                        // so can be seen as a weak edge pixel connected to an strong edge pixel
                        output[index] = 255;
                            // up
                        if(index - width >= 0)
                            q_omp[i].push(index - width);
                        // down
                        if(index + width < width * height)
                            q_omp[i].push(index + width);
                        // left
                        if(index % width != 0)
                            q_omp[i].push(index - 1);
                        // right
                        if(index % width != width - 1)
                            q_omp[i].push(index + 1);
                        // up left
                        if(index - width - 1 >= 0)
                            q_omp[i].push(index - width - 1);
                        // up right
                        if(index - width + 1 >= 0)
                            q_omp[i].push(index - width + 1);
                        // down left
                        if(index + width - 1 < width * height)
                            q_omp[i].push(index + width - 1);
                        // down right
                        if(index + width + 1 < width * height)
                            q_omp[i].push(index + width + 1);
                    }
                    else
                        output[index] = 0;
                }
            }
        }

    }
    printf("edge linking done!\n");
    int total_visit_count = 0;
    for(int i = 0 ; i < thread_used ; i++)
        total_visit_count += visit_count[i];
    // printf("total visit count: %d\n", total_visit_count);
}

int main(int argc, char *argv[]){
    // the input file path
    char filename[100]="../common/data/1024.bmp";
    
    if (argc != 2){
        printf("Usage: ./omp <thread_used>\n");
        exit(1);
    }
    thread_used = atoi(argv[1]);
    omp_set_num_threads(thread_used);

    // variables for recording time
    double startTime, endTime;
    double totalTime = 0.0;

    // read the bmp file (only 24-bit rgb bmp image is supported)
    FILE *fp = fopen(filename, "rb");
    if(fp == NULL){
        printf("Error: cannot open the file!\n");
        exit(1);
    }

    // read the bmp header
    sBmpHeader header = {0};
    fread(&header, sizeof(sBmpHeader), 1, fp);
    width = header.width;
    height = header.height;
    // print_bmp_header(&header);

    // move the file pointer to the beginning of image data
    fseek(fp, header.offset, SEEK_SET);
    pixel24 *p = (pixel24 *)malloc(sizeof(pixel24) * width * height);
    fread(p, sizeof(pixel24), width * height, fp);
    fclose(fp);

    // turn rgb image to one channel gray 
    uint8_t *p1_gray = (uint8_t *)malloc(sizeof(uint8_t) * (width) * (height));
    for(int i = 0 ; i < (width) * (height) ; i ++){
        p1_gray[i] = 0;
    }
    int temp_index, temp_int_pixel;
    for(int i = 0 ; i < height; i++){
        for(int j = 0; j < width; j++){
            temp_index = i * (width) + j;
            temp_int_pixel = (p[temp_index].r + p[temp_index].g + p[temp_index].b) / 3;
            p1_gray[temp_index] = uint8_t(temp_int_pixel);
        }
    }
    
    // step 1: Smoothing
    // fs(x, y) = f(x, y) * G(x, y)
    uint8_t *fs = (uint8_t *)malloc(sizeof(uint8_t) * (width) * (height));
    for(int i = 0 ; i < (width) * (height) ; i ++){
        fs[i] = 0;
    }
    
    float G[9] = {1.0/16, 2.0/16, 1.0/16, 
                  2.0/16, 4.0/16, 2.0/16, 
                  1.0/16, 2.0/16, 1.0/16};
    
    startTime = CycleTimer::currentSeconds();
    conv(p1_gray, G, fs, 0, width, 0, height, 3);
    endTime = CycleTimer::currentSeconds();
    totalTime += (endTime - startTime);
    printf("convolution time: %.3f ms\n", (endTime - startTime) * 1000);

    // step 2: Gradient Computation
    // gx(x, y) = fs(x, y) * Sx(x, y)
    // gy(x, y) = fs(x, y) * Sy(x, y)
    float Sx[9] = {
        -1.0,  0.0,  1.0, 
        -2.0,  0.0,  2.0, 
        -1.0,  0.0,  1.0};
    float Sy[9] = {
        -1.0, -2.0, -1.0, 
         0.0,  0.0,  0.0, 
         1.0,  2.0,  1.0};

    int32_t *gx = (int32_t *)malloc(sizeof(int32_t) * (width) * (height));
    startTime = CycleTimer::currentSeconds();
    conv2(fs, Sx, gx, 0, width, 0, height, 3);
    endTime = CycleTimer::currentSeconds();
    totalTime += (endTime - startTime);
    printf("convolution time: %.3f ms\n", (endTime - startTime) * 1000);
    
    int32_t *gy = (int32_t *)malloc(sizeof(int32_t) * (width) * (height));
    startTime = CycleTimer::currentSeconds();
    conv2(fs, Sy, gy, 0, width, 0, height, 3);
    endTime = CycleTimer::currentSeconds();
    totalTime += (endTime - startTime);
    printf("convolution time: %.3f ms\n", (endTime - startTime) * 1000);
    free(fs);

    // M(x, y) = sqrt(gx(x, y)^2 + gy(x, y)^2)
    int32_t *M = (int32_t *)malloc(sizeof(int32_t) * (width) * (height));
    
    
    startTime = CycleTimer::currentSeconds();
    grad_cal(gx, gy, M, 0, width, 0, height);
    endTime = CycleTimer::currentSeconds();
    totalTime += (endTime - startTime);
    printf("gradient calculation time: %.3f ms\n", (endTime - startTime) * 1000);
    
    // α(x, y) = arctan(gy(x, y) / gx(x, y))
    double *theta = (double *)malloc(sizeof(double) * (width) * (height));
    
    startTime = CycleTimer::currentSeconds();
    theta_cal(gx, gy, theta, 0, width, 0, height);
    endTime = CycleTimer::currentSeconds();
    totalTime += (endTime - startTime);
    printf("theta calculation time: %.3f ms\n", (endTime - startTime) * 1000);


    // step 3: Non-maximum Suppression
    // fN(x, y) = M(x, y) if M(x, y) >= M(x + α(x, y), y + β(x, y)) and M(x, y) >= M(x - α(x, y), y - β(x, y))
    // where α, β are the neighboring pixels in the direction of the gradient
    int32_t *fN = (int32_t *)malloc(sizeof(int32_t) * (width) * (height));

    startTime = CycleTimer::currentSeconds();
    non_maximum_sup(M, fN, theta, 0, width, 0, height);
    endTime = CycleTimer::currentSeconds();
    totalTime += (endTime - startTime);
    printf("non-maximum suppression time: %.3f ms\n", (endTime - startTime) * 1000);

    // step 4: Double Thresholding
    // strong edge: fN(x, y) >= Th
    // weak edge: Tl <= fN(x, y) < Th
    // non-edge: fN(x, y) < Tl
    // Th defined as the 10% of the maximum value of fN
    // Tl defined as the 10% of Th
    int32_t max_fN_index = std::max_element(fN, fN + width * height) - fN;
    Th = fN[max_fN_index] * 0.1;
    Tl = Th * 0.1;

    // step 5: Edge Tracking by Hysteresis
    // edge tracking by bfs
    int32_t *visited = (int32_t *)malloc(sizeof(int32_t) * (width) * (height));
    int32_t *fN_linked = (int32_t *)malloc(sizeof(int32_t) * (width) * (height));
    // init parameters before edge linking
    for(int i = 0 ; i < height ; i += 1){
        for(int j = 0 ; j < width ; j += 1){
            temp_index = i * width + j;
            visited[temp_index] = 0;
            if (fN[temp_index] >= Th)
                q.push(temp_index);
            fN_linked[temp_index] = 0;
        }
    }
    startTime = CycleTimer::currentSeconds();
    edge_linking(fN, fN_linked, visited, 0, width, 0, height);
    endTime = CycleTimer::currentSeconds();
    totalTime += (endTime - startTime);
    printf("edge linking time: %.3f ms\n", (endTime - startTime) * 1000);
    

    // normalize the result to 0~255
    uint8_t *fN_u8 = (uint8_t *)malloc(sizeof(uint8_t) * (width) * (height));
    int32_t max=0, min=1000000;
    for(int i = 0 ; i < height ; i += 1){
        for(int j = 0 ; j < width ; j += 1){
            temp_index = i * width + j;
            max = max > fN_linked[temp_index] ? max : fN_linked[temp_index];
            min = min < fN_linked[temp_index] ? min : fN_linked[temp_index];
        }
    }
    // printf("max: %d, min: %d\n", max, min);
    // printf("====================================\n");
    for(int i = 0 ; i < height ; i += 1){
        for(int j = 0 ; j < width ; j += 1){
            temp_index = i * width + j;
            fN_u8[temp_index] = uint8_t((fN_linked[temp_index] - min) * 255 / (max - min));
        }
    }

    uint8_t *final_result = fN_u8;

    // back to three dimention gray
    pixel24 *p1 = (pixel24 *)malloc(sizeof(pixel24) * width * height);
    for(int i = 0 ; i < height; i++){
        for(int j = 0; j < width; j++){
            temp_index = i * (width) + j;
            p1[temp_index].r = final_result[temp_index];
            p1[temp_index].g = final_result[temp_index];
            p1[temp_index].b = final_result[temp_index];
        }
    }


    // write to a new file
    FILE *fp2 = fopen("output.bmp", "wb");
    if(fp2 == NULL){
        printf("Error: cannot open the file!\n");
        exit(1);
    }
    fwrite(&header, sizeof(sBmpHeader), 1, fp2);
    fwrite(p1, sizeof(pixel24), width * height, fp2);
    fclose(fp2);
    free(p);
    printf("done!\n");

    printf("total time: %.3f ms\n", totalTime * 1000);

    return 0;

}

