#include <iostream>
#include <stdlib.h>
#include <stdio.h>
#include "bmp.h"
#include <cmath>
#include <algorithm>
#include <queue>
#include <pthread.h>
#include <fstream>
// read an rgb bmp image and transfer it to gray image

#define PI 3.14159265
#define THREAD_NUM 2
pthread_mutex_t output_mutex;

using namespace std;

int width, height;

struct edge_link_args{
    int32_t *input;
    int32_t* output;
    int32_t* visited;
    int start_width, end_width;
    int start_height, end_height;
    int thread_id;
};

struct conv_args{
    uint8_t *input; 
    float *kernel;
    uint8_t *output; 
    int start_width;
    int end_width; 
    int start_height; 
    int end_height; 
    int kernel_size;
};

void* conv(void* args){
    struct conv_args *conv_arg = (struct conv_args*)args;
    uint8_t *input  = conv_arg->input;
    float *kernel   = conv_arg->kernel;
    uint8_t *output = conv_arg->output;
    int start_width = conv_arg->start_width;
    int end_width   = conv_arg->end_width;
    int start_height= conv_arg->start_height;
    int end_height  = conv_arg->end_height;
    int kernel_size = conv_arg->kernel_size;

    float temp_pixel = 0;
    printf("convolution start!\n");

    ofstream myfile;
    myfile.open ("conv.txt");

    // TODO: boundary check due to padding, modify to the version without padding
    int indexi, indexj;
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
            myfile << temp_pixel;
        }
    }
    printf("convolution done!\n");
    myfile.close();
    pthread_exit(EXIT_SUCCESS);
}

void conv2(
    uint8_t *input, 
    float *kernel, 
    int32_t *output, 
    int start_width, int end_width, 
    int start_height, int end_height, 
    int kernel_size){

    float temp_pixel = 0;
    printf("convolution2 start!\n");
    // TODO: boundary check due to padding, modify to the version without padding
    int indexi, indexj;
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
    printf("convolution2 done!\n");
}

void grad_cal(
        int32_t *gx, int32_t *gy, 
        int32_t *output, 
        int start_width, int end_width, 
        int start_height, int end_height){
    int32_t temp_pixel = 0, temp_index = 0;
    printf("gradient calculation start!\n");
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
    for(int i = start_height ; i < end_height ; i++){
        for(int j = start_width; j < end_width ; j++){
            temp_pixel = atan2(gy[i * width + j], gx[i * width + j]);
            temp_pixel = temp_pixel * 180 / PI;
            if(temp_pixel < 0)
                temp_pixel += 180;
            temp_pixel = 180 - temp_pixel;
            // realign the theta offset due to direction of sobel

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
queue<int32_t> q[THREAD_NUM];

void *edge_linking(void *args){
    int32_t index;
    int32_t temp;
    int32_t *input = ((edge_link_args *)args)->input;
    int32_t *output = ((edge_link_args *)args)->output;
    int32_t *visited = ((edge_link_args *)args)->visited;
    int start_width = ((edge_link_args *)args)->start_width;
    int end_width = ((edge_link_args *)args)->end_width;
    int start_height = ((edge_link_args *)args)->start_height;
    int end_height = ((edge_link_args *)args)->end_height;
    int thread_id = ((edge_link_args *)args)->thread_id;
    queue<int32_t> q_th = q[thread_id];
    while(!q_th.empty()){
        index = q_th.front();
        q_th.pop();
        if(visited[index] == 0){
            visited[index] = 1;
            if(input[index] >= Tl){
            // since the origin q only push the pixel with value >= Th, 
            // any pixel in queue must be visited after an strong edge pixel
            // so can be seen as a weak edge pixel connected to an strong edge pixel
                pthread_mutex_lock(&output_mutex);
                output[index] = 255;
                pthread_mutex_unlock(&output_mutex);
                // up
                if(index - width >= 0)
                    q_th.push(index - width);
                // down
                if(index + width < width * height)
                    q_th.push(index + width);
                // left
                if(index % width != 0)
                    q_th.push(index - 1);
                // right
                if(index % width != width - 1)
                    q_th.push(index + 1);
                // up left
                if(index - width - 1 >= 0)
                    q_th.push(index - width - 1);
                // up right
                if(index - width + 1 >= 0)
                    q_th.push(index - width + 1);
                // down left
                if(index + width - 1 < width * height)
                    q_th.push(index + width - 1);
                // down right
                if(index + width + 1 < width * height)
                    q_th.push(index + width + 1);
            }
            else{
                pthread_mutex_lock(&output_mutex);
                output[index] = 0;
                pthread_mutex_unlock(&output_mutex);
            }
        }

    }
    printf("edge linking done!\n");
}

int main(){
    char filename[100] = "izuna24.bmp";

    FILE *fp = fopen(filename, "rb");
    if(fp == NULL){
        printf("Error: cannot open the file!\n");
        exit(1);
    }
    // read the header
    sBmpHeader header = {0};
    fread(&header, sizeof(sBmpHeader), 1, fp);
    width = header.width;
    height = header.height;
    // print_bmp_header(&header);

    fseek(fp, header.offset, SEEK_SET);
    // read each pixel (for 24-bit bmp)
    pixel24 *p = (pixel24 *)malloc(sizeof(pixel24) * width * height);
    fread(p, sizeof(pixel24), width * height, fp);
    fclose(fp);

    // to one dimention gray
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
    
    // convolution
    uint8_t *fs = (uint8_t *)malloc(sizeof(uint8_t) * (width) * (height));
    
    for(int i = 0 ; i < (width) * (height) ; i ++){
        fs[i] = 0;
    }
    
    // step 1: Smoothing
    pthread_t t[THREAD_NUM];
    // int width_per_thread[THREAD_NUM], height_per_thread[THREAD_NUM];
    conv_args conv_arg[THREAD_NUM];

    int start_width = 0, start_height = 0;
    int width_per_thread  = width / THREAD_NUM;
    // int height_per_thread = height / THREAD_NUM;
    float G[9] = {1.0/16, 2.0/16, 1.0/16, 2.0/16, 4.0/16, 2.0/16, 1.0/16, 2.0/16, 1.0/16};
    
    for(int i = 0; i < 1; i++) {
        conv_arg[i].input  = p1_gray;
        conv_arg[i].kernel = G;
        conv_arg[i].output = fs;
        conv_arg[i].start_width = 0;//start_width
        // start_width += width_per_thread;
        conv_arg[i].end_width = width;//(i != THREAD_NUM - 1) ? start_width : 
        // conv_arg[i].start_height = start_height;
        conv_arg[i].start_height = 0;
        // start_height += height_per_thread;
        // conv_arg[i].end_height = (i != THREAD_NUM - 1) ? start_height : height;
        conv_arg[i].end_height = height;
        conv_arg[i].kernel_size = 3;
        // std::cout << width << " " << height <<"\n";
        std::cout << conv_arg[i].start_width << " " << conv_arg[i].end_width << "\n";
        std::cout << conv_arg[i].start_height << " " << conv_arg[i].end_height << "\n";

    }
    for(int i = 0; i < 1; i++) {
        pthread_create(&t[i], NULL, conv, &conv_arg[i]);
    }
    // conv(p1_gray, G, fs, 0, width, 0, height, 3);
    for(int i = 0; i < 1; i++) {
        pthread_join(t[i], NULL);
    }
    // conv(p1_gray, G, fs, 0, width/2, 0, height/2, 3);
    // conv(p1_gray, G, fs, width/2, width, 0, height/2, 3);
    // conv(p1_gray, G, fs, width/2, width, width/2, height, 3);
    // conv(p1_gray, G, fs, 0, width/2, width/2, height, 3);
    
    // step 2: Gradient Computation

    float Sx[9] = {
        -1.0,  0.0,  1.0, 
        -2.0,  0.0,  2.0, 
        -1.0,  0.0,  1.0};
    float Sy[9] = {
        -1.0, -2.0, -1.0, 
         0.0,  0.0,  0.0, 
         1.0,  2.0,  1.0};

    // TODO: gx gy fN modify to int32_t
    int32_t *gx = (int32_t *)malloc(sizeof(int32_t) * (width) * (height));
    conv2(fs, Sx, gx, 0, width, 0, height, 3);
    int32_t *gy = (int32_t *)malloc(sizeof(int32_t) * (width) * (height));
    conv2(fs, Sy, gy, 0, width, 0, height, 3);
    free(fs);
    int32_t *M = (int32_t *)malloc(sizeof(int32_t) * (width) * (height));
    grad_cal(gx, gy, M, 0, width, 0, height);
    // grad_cal(gx, gy, M, 0, width, 0, height/2);
    // grad_cal(gx, gy, M, 0, width, height/2, height);


    double theta_temp = 0.0;
    double *theta = (double *)malloc(sizeof(double) * (width) * (height));
    theta_cal(gx, gy, theta, 0, width, 0, height);



    // step 3: Non-maximum Suppression
    int32_t *fN = (int32_t *)malloc(sizeof(int32_t) * (width) * (height));
    non_maximum_sup(M, fN, theta, 0, width, 0, height);

    // step 4: Double Thresholding
    // get the max and min of fN
    int32_t max_fN_index = std::max_element(fN, fN + width * height) - fN;
    Th = fN[max_fN_index] * 0.1;
    Tl = Th * 0.1;

    // step 5: Edge Tracking by Hysteresis
    // for(int i = 0 ; i < height ; i += 1){
    //     for(int j = 0 ; j < width ; j += 1){
    //         if(fN[i * width + j] >= Th)
    //             fN[i * width + j] = 255;
    //         else if(fN[i * width + j] <= Tl)
    //             fN[i * width + j] = 0;
    //         else{
    //             if(fN[(i - 1) * width + j] >= Th || fN[(i + 1) * width + j] >= Th || fN[i * width + j - 1] >= Th || fN[i * width + j + 1] >= Th)
    //                 fN[i * width + j] = 255;
    //             else
    //                 fN[i * width + j] = 0;
    //         }
    //     }
    // }

    /* 
     * parallelize the bfs edge tracking
     * split the queue into array with size THREAD_NUM
     * each thread will take one queue to do the bfs
     * and mutex lock will handle the writing of critical section (output)
     */
    // edge tracking by bfs
    int32_t *visited = (int32_t *)malloc(sizeof(int32_t) * (width) * (height));
    int32_t *fN_linked = (int32_t *)malloc(sizeof(int32_t) * (width) * (height));
    int queue_index = 0;
    // init parameters before edge linking
    for(int i = 0 ; i < height ; i += 1){
        for(int j = 0 ; j < width ; j += 1){
            temp_index = i * width + j;
            visited[temp_index] = 0;
            if (fN[temp_index] >= Th)
                q[queue_index].push(temp_index);
                queue_index = (queue_index + 1) % THREAD_NUM;
            fN_linked[temp_index] = 0;
        }
    }
    pthread_t *thread_pool = (pthread_t*)malloc(THREAD_NUM * sizeof(pthread_t));
    pthread_mutex_init(&output_mutex, NULL);
    
    for(int i = 0 ; i < THREAD_NUM ; i++){
        edge_link_args input_args = {fN, fN_linked, visited, 0, width, 0, height, i};
        pthread_create(&thread_pool[i], NULL, edge_linking, (void *)&input_args);
    }

    for(int i = 0; i < THREAD_NUM; i++){
        pthread_join(thread_pool[i], NULL);
    }

    free(thread_pool);
    pthread_mutex_destroy(&output_mutex);
    
    // edge_linking(fN, fN_linked, visited, 0, width, 0, height);
    // for(int i = 0 ; i < height ; i += 1){
    //     for(int j = 0 ; j < width ; j += 1){
    //         temp_index = i * width + j;
    //         if(visited[temp_index] == 0)
    //             fN[temp_index] = 0;
    //     }
    // }

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
    FILE *fp2 = fopen("output_with_bfs.bmp", "wb");
    if(fp2 == NULL){
        printf("Error: cannot open the file!\n");
        exit(1);
    }
    fwrite(&header, sizeof(sBmpHeader), 1, fp2);
    fwrite(p1, sizeof(pixel24), width * height, fp2);
    fclose(fp2);
    free(p);
    printf("done!\n");
    return 0;


}

