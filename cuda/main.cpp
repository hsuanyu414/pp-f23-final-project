#include <iostream>
#include <stdlib.h>
#include <stdio.h>
#include <queue>
#include <algorithm>
#include "bmp.h"

using namespace std;

int width, height;

extern void conv8(uint8_t *input, float *kernel, uint8_t *output, 
                 int start_width, int end_width, int start_height, 
                 int end_height, int kernel_size);
extern void conv32(uint8_t *input, float *kernel, int32_t *output, 
                 int start_width, int end_width, int start_height, 
                 int end_height, int kernel_size);
extern void grad_cal(int32_t *gx, int32_t *gy, int32_t *output, int start_width, 
                     int end_width, int start_height, int end_height);
extern void theta_cal(int32_t *gx, int32_t *gy, double *output, int start_width, 
                     int end_width, int start_height, int end_height);
extern void non_maximum_sup(int32_t *input, int32_t *output, double *theta,
                            int start_width, int end_width, int start_height, int end_height);

int32_t Th, Tl;
queue<int32_t> q; 

int main(int argc, char *argv[]){
    /* read the image file from prompt */
    if(argc != 2){
        fprintf(stderr, "Usage: ./program <file_path>\n");
        exit(1);
    }
    FILE *f = fopen(argv[1], "rb");
    if(f == NULL){
        fprintf(stderr, "Error: Invalid file\n");
        exit(1);
    }
    /* read the header */
    sBmpHeader header = {0};
    if(!fread(&header, sizeof(sBmpHeader), 1, f)){
        fprintf(stderr, "Error: Invalid BMP file\n");
        exit(1);
    }
    width = header.width;
    height = header.height;

    /* read the image */
    fseek(f, header.offset, SEEK_SET);
    pixel24 *p = (pixel24 *)malloc(width * height * sizeof(pixel24));
    if(!fread(p, sizeof(pixel24), width * height, f)){
        fprintf(stderr, "Error: Invalid BMP file\n");
        exit(1);
    }
    fclose(f);

    /* convert to 1-d array */
    uint8_t *pi_gray = (uint8_t *)calloc(width * height, sizeof(uint8_t));
    int temp_index, temp_int_pixel;
    for(int i = 0; i < height; i++){
        for(int j = 0; j < width; j++){
            temp_index = i * width + j;
            temp_int_pixel = (p[temp_index].r + p[temp_index].g + p[temp_index].b) / 3;
            pi_gray[temp_index] = (uint8_t)temp_int_pixel;
        }
    }

    /* step1: smoothing */
    uint8_t *fs = (uint8_t *)calloc(width * height, sizeof(uint8_t));
    float G[9] = {1.0/16, 2.0/16, 1.0/16, 2.0/16, 4.0/16, 2.0/16, 1.0/16, 2.0/16, 1.0/16};
    printf("conv start\n");
    conv8(pi_gray, G, fs, 0, width, 0, height, 3);
    printf("conv end\n");

    /* step2: gradient computation */
    float Sx[9] = {-1.0, 0.0, 1.0, -2.0, 0.0, 2.0, -1.0, 0.0, 1.0};
    float Sy[9] = {-1.0, -2.0, -1.0, 0.0, 0.0, 0.0, 1.0, 2.0, 1.0};

    int32_t *gx = (int32_t *)malloc(width * height * sizeof(int32_t));
    int32_t *gy = (int32_t *)malloc(width * height * sizeof(int32_t));
    printf("gradient start\n");
    conv32(fs, Sx, gx, 0, width, 0, height, 3);
    conv32(fs, Sy, gy, 0, width, 0, height, 3);
    printf("gradient end\n");

    int32_t *M = (int32_t *)malloc(width * height * sizeof(int32_t));
    grad_cal(gx, gy, M, 0, width, 0, height);

    double *theta = (double *)malloc(width * height * sizeof(double));
    theta_cal(gx, gy, theta, 0, width, 0, height);

    /* step3: Non-Maximum Suppression */
    int32_t * fN = (int32_t *)malloc(width * height * sizeof(int32_t));
    non_maximum_sup(M, fN, theta, 0, width, 0, height);

    /* step4: Double Thresholding */
    int32_t max_fN_index = std::max_element(fN, fN + width * height) - fN;
    Th = fN[max_fN_index] * 0.1;
    Tl = Th * 0.1;

    /* normalize to uint8 */
    int32_t *nor_img = fN;
    uint8_t *fN_u8 = (uint8_t *)malloc(width * height * sizeof(uint8_t));
    int32_t max=0, min=1000000;
    for(int i = 0 ; i < height ; i++){
        for(int j = 0 ; j < width ; j++){
            temp_index = i * width + j;
            max = max > nor_img[temp_index] ? max : nor_img[temp_index];
            min = min < nor_img[temp_index] ? min : nor_img[temp_index];
        }
    }

    printf("max: %d, min: %d\n", max, min);

    for(int i = 0; i < height; i++){
        for(int j = 0; j < width; j++){
            int tmp_index = i * width + j;
            fN_u8[tmp_index] = uint8_t((nor_img[tmp_index] - min) * 255 / (max - min));
        }
    }

    uint8_t *final_temp = fN_u8;

    /* convert to 3-d array */
    pixel24 *final_result = (pixel24 *)malloc(width * height * sizeof(pixel24));
    for(int i = 0; i < height; i++){
        for(int j = 0; j < width; j++){
            temp_index = i * width + j;
            final_result[temp_index].r = final_temp[temp_index];
            final_result[temp_index].g = final_temp[temp_index];
            final_result[temp_index].b = final_temp[temp_index];
        }
    }

    /* write to file */
    FILE *f2 = fopen("output.bmp", "wb");
    if(f2 == NULL){
        fprintf(stderr, "Error: Invalid file\n");
        exit(1);
    }
    fwrite(&header, sizeof(sBmpHeader), 1, f2);
    fwrite(final_result, sizeof(pixel24), width * height, f2);
    fclose(f2);
    free(p);
    free(pi_gray);
    free(fs);
    free(final_result);
    return 0;
}