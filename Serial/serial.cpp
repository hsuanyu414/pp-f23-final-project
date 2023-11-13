#include <iostream>
#include <stdlib.h>
#include <stdio.h>
#include "bmp.h"
// read an rgb bmp image and transfer it to gray image

using namespace std;

void conv(uint8_t *input, float *kernel, uint8_t *output, int width, int height, int kernel_size){
    float temp_pixel = 0;
    printf("convolution start!\n");
    // TODO: boundary check due to padding, modify to the version without padding
    int indexi, indexj;
    for(int i = 0 ; i < height ; i++){
        for(int j = 0; j < width ; j++){
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
            output[i * width + j] = uint8_t(temp_pixel);
        }
    }
    printf("convolution done!\n");
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

    // to one dimention gray
    uint8_t *p1_gray = (uint8_t *)malloc(sizeof(uint8_t) * (header.width) * (header.height));
    for(int i = 0 ; i < (header.width) * (header.height) ; i ++){
        p1_gray[i] = 0;
    }
    for(int i = 0 ; i < header.height; i++){
        for(int j = 0; j < header.width; j++){
            int gray = (p[i * (header.width) + j].r + p[i * (header.width) + j].g + p[i * (header.width) + j].b) / 3;
            p1_gray[i * (header.width ) + j] = gray;
        }
    }
    
    // convolution
    uint8_t *p1_gray_after_gaussian = (uint8_t *)malloc(sizeof(uint8_t) * (header.width) * (header.height));
    
    for(int i = 0 ; i < (header.width) * (header.height) ; i ++){
        p1_gray_after_gaussian[i] = 0;
    }
    
    float gaussian_kernel[9] = {1.0/16, 2.0/16, 1.0/16, 2.0/16, 4.0/16, 2.0/16, 1.0/16, 2.0/16, 1.0/16};
    conv(p1_gray, gaussian_kernel, p1_gray_after_gaussian, header.width , header.height , 3);


    uint8_t *final_result = p1_gray_after_gaussian;

    // back to three dimention gray
    pixel24 *p1 = (pixel24 *)malloc(sizeof(pixel24) * header.width * header.height);
    for(int i = 0 ; i < header.height; i++){
        for(int j = 0; j < header.width; j++){
            p1[i * header.width + j].r = final_result[i * (header.width ) + j];
            p1[i * header.width + j].g = final_result[i * (header.width ) + j];
            p1[i * header.width + j].b = final_result[i * (header.width ) + j];
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

