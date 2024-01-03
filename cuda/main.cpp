#include <iostream>
#include <stdlib.h>
#include <stdio.h>
#include <queue>
#include <algorithm>
#include <cmath>
#include "bmp.h"
#include "../common/CycleTimer.h"

using namespace std;

int width, height;

extern void canny_edge_detection(uint8_t *input, int32_t *output, int start_width, 
    int end_width, int start_height, int end_height);

extern void edge_link_bfs(int32_t *input, int32_t *output, int start_width, 
                         int end_width, int start_height, int end_height, int32_t Th, int32_t Tl);

int32_t Th, Tl;
queue<int32_t> q;

void edge_linking(
        int32_t *input, 
        int32_t* output,
        int32_t* visited, 
        int start_width, int end_width, 
        int start_height, int end_height){
    int32_t index;
    int32_t temp;
    while(!q.empty()){
        index = q.front();
        q.pop();
        if(visited[index] == 0){
            visited[index] = 1;
            if(input[index] >= Tl){
            // since the origin q only push the pixel with value >= Th, 
            // any pixel in queue must be visited after an strong edge pixel
            // so can be seen as a weak edge pixel connected to an strong edge pixel
                output[index] = 255;
                // up
                if(index - width >= 0)
                    q.push(index - width);
                // down
                if(index + width < width * height)
                    q.push(index + width);
                // left
                if(index % width != 0)
                    q.push(index - 1);
                // right
                if(index % width != width - 1)
                    q.push(index + 1);
                // up left
                if(index - width - 1 >= 0)
                    q.push(index - width - 1);
                // up right
                if(index - width + 1 >= 0)
                    q.push(index - width + 1);
                // down left
                if(index + width - 1 < width * height)
                    q.push(index + width - 1);
                // down right
                if(index + width + 1 < width * height)
                    q.push(index + width + 1);
            }
            else
                output[index] = 0;
        }

    }
    printf("edge linking done!\n");
}

int main(int argc, char *argv[]){
    /* read the image file from prompt */
    if(argc > 3){
        fprintf(stderr, "Usage: ./program <file_path>\n");
        exit(1);
    }

    /* if cuda_bfs is true, use cuda edge linking */
    bool cuda_bfs = false;
    if(argc == 3){
        if(strcmp(argv[2], "cuda_bfs") == 0)
            cuda_bfs = true;
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

    double startTime, endTime;

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

    /* step1~5 */
    int32_t * fN = (int32_t *)malloc(width * height * sizeof(int32_t));

    startTime = CycleTimer::currentSeconds();
    canny_edge_detection(pi_gray, fN, 0, width, 0, height);
    endTime = CycleTimer::currentSeconds();
    double non_maximum_sup_time = (endTime - startTime) * 1000;
    printf("step1~5 time: %.3f ms\n", non_maximum_sup_time);

    /* step6: Double Thresholding */
    int32_t max_fN_index = std::max_element(fN, fN + width * height) - fN;
    Th = fN[max_fN_index] * 0.1;
    Tl = Th * 0.1;

    /* step7: Edge Linking by BFS */
    int32_t *fN_linked = (int32_t *)malloc(sizeof(int32_t) * (width) * (height));
    double edge_link_bfs_time;

    if(cuda_bfs){
        startTime = CycleTimer::currentSeconds();
        edge_link_bfs(fN, fN_linked, 0, width, 0, height, Th, Tl);
        endTime = CycleTimer::currentSeconds();
        edge_link_bfs_time = (endTime - startTime) * 1000;
        printf("edge_link_bfs time: %.3f ms\n", edge_link_bfs_time);
        printf("edge_link_bfs end\n");
    }

    else{
        /* serial edge linking */
        int32_t *visited = (int32_t *)malloc(sizeof(int32_t) * (width) * (height));
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
        edge_link_bfs_time = (endTime - startTime) * 1000;
        printf("edge_link_bfs time: %.3f ms\n", edge_link_bfs_time);
    }
    free(fN);

    /* compare with golden */
    // int32_t *golden = (int32_t *)malloc(width * height * sizeof(int32_t));
    // char *file_name = strrchr(argv[1], '/');
    // file_name++;
    // char *dot = strrchr(file_name, '.');
    // *dot = '\0';
    // char golden_file_name[100];
    // sprintf(golden_file_name, "%s_golden.txt", file_name);
    // printf("golden_file_name: %s\n", golden_file_name);
    // FILE *f_golden = fopen(golden_file_name, "r");
    // for(int i = 0 ; i < height ; i++){
    //     for(int j = 0 ; j < width ; j++){
    //         temp_index = i * width + j;
    //         fscanf(f_golden, "%d", &golden[temp_index]);
    //     }
    // }
    // fclose(f_golden);

    // int32_t error = 0;
    // for(int i = 0 ; i < height ; i++){
    //     for(int j = 0 ; j < width ; j++){
    //         temp_index = i * width + j;
    //         if(fN_linked[temp_index] != golden[temp_index])
    //             error++;
    //     }
    // }
    // printf("error: %d\n", error);
    // printf("error rate: %.3f\n", (double)error / (width * height) * 100);
    // printf("correct rate: %.3f\n", 100 - (double)error / (width * height) * 100);

    /* total time */
    double total_time = non_maximum_sup_time + edge_link_bfs_time;
    printf("total time: %f ms\n", total_time);

    /* normalize to uint8 */
    int32_t *nor_img = fN_linked;
    uint8_t *fN_u8 = (uint8_t *)malloc(width * height * sizeof(uint8_t));
    int32_t max=0, min=1000000;
    for(int i = 0 ; i < height ; i++){
        for(int j = 0 ; j < width ; j++){
            temp_index = i * width + j;
            max = max > nor_img[temp_index] ? max : nor_img[temp_index];
            min = min < nor_img[temp_index] ? min : nor_img[temp_index];
        }
    }

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
    free(fN_linked);
    free(final_result);
    return 0;
}