#include <iostream>
#include <stdlib.h>
#include <stdio.h>
#include "bmp.h"
// read an rgb bmp image and transfer it to gray image

using namespace std;

int main(){
    FILE *fp = fopen("test24.bmp", "rb");
    if(fp == NULL){
        printf("Error: cannot open the file!\n");
        exit(1);
    }
    // read the header
    sBmpHeader header = {0};
    fread(&header, sizeof(sBmpHeader), 1, fp);
    print_bmp_header(&header);

    fseek(fp, header.offset, SEEK_SET);
    // read each pixel (for 32-bit bmp)
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

    // gray
    

    // reverse padding
    pixel24 *p1 = (pixel24 *)malloc(sizeof(pixel24) * header.width * header.height);
    for(int i = 1; i< header.height + 1; i++){
        for(int j = 1; j < header.width + 1; j++){
            p1[(i - 1) * header.width + j - 1] = p1_padding[i * (header.width + 2) + j];
        }
    }   


    // write to a new file
    FILE *fp2 = fopen("test_gray.bmp", "wb");
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