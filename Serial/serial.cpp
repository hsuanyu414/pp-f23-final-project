#include <iostream>
#include <stdlib.h>
#include <stdio.h>
#include "bmp.h"
// read an rgb bmp image and transfer it to gray image

using namespace std;

void conv(uint8_t *input, float *kernel, uint8_t *output, int width, int height, int kernel_size){
    // takes an gray image array (already padded by 1) and a kernel, return the convoluted image array
    float temp_pixel = 0;
    for(int i = 1 ; i < height - 1; i++){
        for(int j = 1; j < width - 1; j++){
            temp_pixel = 0;
            for(int k = 0 ; k < kernel_size ; k++){
                for(int l = 0 ; l < kernel_size ; l++){
                    temp_pixel += float(input[(i + k - 1) * width + j + l - 1] * kernel[k * kernel_size + l]);
                }
            }
            output[i * width + j] = uint8_t(temp_pixel);
        }
    }
}

int main(){
    FILE *fp = fopen("test_gray.bmp", "rb");
    if(fp == NULL){
        printf("Error: cannot open the file!\n");
        exit(1);
    }
    // read the header
    sBmpHeader header = {0};
    fread(&header, sizeof(sBmpHeader), 1, fp);
    // print_bmp_header(&header);

    fseek(fp, header.offset, SEEK_SET);
    // read each pixel (for 24-bit bmp)
    pixel24 *p = (pixel24 *)malloc(sizeof(pixel24) * header.width * header.height);
    fread(p, sizeof(pixel24), header.width * header.height, fp);
    fclose(fp);

    // padding
    pixel24 *p1_padding = (pixel24 *)malloc(sizeof(pixel24) * (header.width + 2) * (header.height + 2));
    for(int i = 1; i< header.height + 1; i++){
        for(int j = 1; j < header.width + 1; j++){
            p1_padding[i * (header.width + 2) + j] = p[(i - 1) * header.width + j - 1];
        }
    }

    // to one dimention gray
    uint8_t *p1_gray = (uint8_t *)malloc(sizeof(uint8_t) * (header.width + 2) * (header.height + 2));
    for(int i = 0 ; i < (header.width + 2) * (header.height + 2) ; i ++){
        p1_gray[i] = 0;
    }
    for(int i = 1 ; i < header.height + 1; i++){
        for(int j = 1; j < header.width + 1; j++){
            int gray = (p1_padding[i * (header.width + 2) + j].r + p1_padding[i * (header.width + 2) + j].g + p1_padding[i * (header.width + 2) + j].b) / 3;
            p1_gray[i * (header.width + 2) + j] = gray;
        }
    }

    // convolution
    uint8_t *p1_gray_after_gaussian = (uint8_t *)malloc(sizeof(uint8_t) * (header.width + 2) * (header.height + 2));
    for(int i = 0 ; i < (header.width + 2) * (header.height + 2) ; i ++){
        p1_gray_after_gaussian[i] = 0;
    }
    float gaussian_kernel[9] = {1.0/16, 2.0/16, 1.0/16, 2.0/16, 4.0/16, 2.0/16, 1.0/16, 2.0/16, 1.0/16};
    conv(p1_gray, gaussian_kernel, p1_gray_after_gaussian, header.width + 2, header.height + 2, 3);


    uint8_t *final_result = p1_gray_after_gaussian;

    // back to three dimention gray
    for(int i = 1 ; i < header.height + 1; i++){
        for(int j = 1; j < header.width + 1; j++){
            p1_padding[i * (header.width + 2) + j].r = final_result[i * (header.width + 2) + j];
            p1_padding[i * (header.width + 2) + j].g = final_result[i * (header.width + 2) + j];
            p1_padding[i * (header.width + 2) + j].b = final_result[i * (header.width + 2) + j];
        }
    }

    // reverse padding
    pixel24 *p1 = (pixel24 *)malloc(sizeof(pixel24) * header.width * header.height);
    for(int i = 1; i< header.height + 1; i++){
        for(int j = 1; j < header.width + 1; j++){
            p1[(i - 1) * header.width + j - 1] = p1_padding[i * (header.width + 2) + j];
        }
    }   


    // write to a new file
    FILE *fp2 = fopen("test_gray_output.bmp", "wb");
    if(fp2 == NULL){
        printf("Error: cannot open the file!\n");
        exit(1);
    }
    fwrite(&header, sizeof(sBmpHeader), 1, fp2);
    fwrite(p1, sizeof(pixel24), header.width * header.height, fp2);
    fclose(fp2);
    free(p);
    return 0;


}

