#ifndef KERNEL_H
#define KERNEL_H

#include <cuda.h>
#include <stdio.h>
#include <stdlib.h>
#include <cmath>
#include "cuda_runtime.h"

#define MAX_THREADS_PER_BLOCK 1024
#define BLOCK_SIZE 32

#define PI 3.14159265

// void conv8(
//     uint8_t *input, float *kernel, uint8_t *output, int start_width, 
//     int end_width, int start_height, int end_height, int kernel_size);

// void conv32(
//     uint8_t *input, float *kernel, int32_t *output, int start_width, 
//     int end_width, int start_height, int end_height, int kernel_size);

// void grad_cal(
//     int32_t *gx, int32_t *gy, int32_t *output, int start_width, 
//     int end_width, int start_height, int end_height);

// void theta_cal(
//     int32_t *gx, int32_t *gy, double *output, int start_width, 
//     int end_width, int start_height, int end_height);

// void non_maximum_sup(int32_t *input, int32_t *output, double *theta,
//     int start_width, int end_width, int start_height, int end_height);

void canny_edge_detection(uint8_t *input, int32_t *output, int start_width, 
    int end_width, int start_height, int end_height);

void edge_link_bfs(int32_t *input, int32_t *output, int start_width, 
    int end_width, int start_height, int end_height, int32_t Th, int32_t Tl);

#endif