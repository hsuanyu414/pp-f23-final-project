#include "kernel.h"

#define QUEUE_SIZE 10000
#define BLOCK_SIZE_BFS 32
#define ITER 4

__device__ void enqueue(
    int32_t *block_queue, int *front, int *rear, int32_t value
){
    /* queue full */
    if((*rear + 1) == *front){
        return;
    }
    block_queue[*rear] = value;
    *rear = (*rear + 1);
}

__device__ int32_t dequeue(
    int32_t *block_queue, int *front, int *rear
){
    /* queue empty */
    if(*front == *rear){
        return -1;
    }
    int32_t value = block_queue[*front];
    *front = (*front + 1);
    return value;
}

__device__ int queue_size(
    int *front, int *rear
){
    return (*rear - *front);
}

__global__ void edge_link_bfs_kernel(
    int32_t *input, int32_t *output, int start_width, int end_width,
    int start_height, int end_height, int32_t Th, int32_t Tl
){

    __shared__ int front, rear;
    __shared__ int count;
    __shared__ int32_t block_queue[QUEUE_SIZE];
    __shared__ int sub_fN[(BLOCK_SIZE_BFS + 2) * (BLOCK_SIZE_BFS + 2)];

    /* 
     * 1. preprocessing 
     *    mark the pixel as 
     *     -2 : definite edge (push to shared queue)
     *     -1 : candidate edge
     *      0 : non edge
     * 2. executing BFS
     *    for each thread in block, take one pixel index from queue and check its 8 neighbors
     */

    int thread_start_width = blockIdx.x * blockDim.x;
    int thread_end_width = thread_start_width + blockDim.x;
    int thread_start_height = blockIdx.y * blockDim.y;
    int thread_end_height = thread_start_height + blockDim.y;

    thread_end_width = thread_end_width > end_width ? end_width : thread_end_width;
    thread_end_height = thread_end_height > end_height ? end_height : thread_end_height;

    if(thread_start_width < 0 || thread_end_width > end_width || 
       thread_start_height < 0 || thread_end_height > end_height){
        return;
    }

    /* Padding the apron */
    int thread_start_width_apron = thread_start_width - 1;
    int thread_end_width_apron = thread_end_width + 1;
    int thread_start_height_apron = thread_start_height - 1;
    int thread_end_height_apron = thread_end_height + 1;

    int thread_width = thread_end_width_apron - thread_start_width_apron;
    int thread_height = thread_end_height_apron - thread_start_height_apron;

    /* Check the paron boundary */
    thread_start_width_apron = thread_start_width_apron < 0 ? 0 : thread_start_width_apron;
    thread_end_width_apron = thread_end_width_apron > end_width ? end_width : thread_end_width_apron;
    thread_start_height_apron = thread_start_height_apron < 0 ? 0 : thread_start_height_apron;
    thread_end_height_apron = thread_end_height_apron > end_height ? end_height : thread_end_height_apron;

    int32_t width = end_width - start_width;
    int32_t height = end_height - start_height;

    /* 
     * initialize the queue for each thread block
     * mark the visited pixel
     */
    if(threadIdx.x == 0 && threadIdx.y == 0){
        front = 0;
        rear = 0;
        count = 0;
        for(int i = thread_start_height_apron; i < thread_end_height_apron; i++){
            for(int j = thread_start_width_apron; j < thread_end_width_apron; j++){
                int32_t index = i * width + j;
                int32_t sub_index = (i - (thread_start_height_apron)) * thread_width + (j - (thread_start_width_apron));
                /* If pixel (including apron) exceed the high threshold, enqueue and marked as -2 */
                if(input[index] >= Th || output[index] == 255){
                    sub_fN[sub_index] = -2;
                    enqueue(block_queue, &front, &rear, index);
                    if(i >= thread_start_height && i < thread_end_height && j >= thread_start_width && j < thread_end_width){
                        output[index] = 255;
                    }
                }
                /* If pixel exceed low threshold, marked as -1 (strong edge candidate) */
                else if(input[index] >= Tl){
                    sub_fN[sub_index] = -1;
                }
                else{
                    sub_fN[sub_index] = 0;
                }
            }
        }
    }
    __syncthreads();

    /* 
     * BFS operation
     * for each thread in block, take one strong pixel index from queue and check its 8 neighbors
     * if the neighbor is weak/strong edge candidate, enqueue and marked as -2 
     */
    while(queue_size(&front, &rear) > 0){
        int32_t index = dequeue(block_queue, &front, &rear);
        int32_t i = index / width;
        int32_t j = index % width;
        if(index < 0 || index >= width * height){
            continue;
        }
        output[index] = 255;
        /* up */
        if(i - 1 >= thread_start_height){
            int32_t up_index = (i - 1) * width + j;
            int32_t sub_up_index = (i - 1 - (thread_start_height_apron)) * thread_width + (j - (thread_start_width_apron));
            if(sub_fN[sub_up_index] == -1){
                sub_fN[sub_up_index] = -2;
                enqueue(block_queue, &front, &rear, up_index);
            }
        }
        /* down */
        if(i + 1 < thread_end_height){
            int32_t down_index = (i + 1) * width + j;
            int32_t sub_down_index = (i + 1 - (thread_start_height_apron)) * thread_width + (j - (thread_start_width_apron));
            if(sub_fN[sub_down_index] == -1){
                sub_fN[sub_down_index] = -2;
                enqueue(block_queue, &front, &rear, down_index);
            }
        }
        /* left */
        if(j - 1 >= thread_start_width){
            int32_t left_index = i * width + (j - 1);
            int32_t sub_left_index = (i - (thread_start_height_apron)) * thread_width + (j - 1 - (thread_start_width_apron));
            if(sub_fN[sub_left_index] == -1){
                sub_fN[sub_left_index] = -2;
                enqueue(block_queue, &front, &rear, left_index);
            }
        }
        /* right */
        if(j + 1 < thread_end_width){
            int32_t right_index = i * width + (j + 1);
            int32_t sub_right_index = (i - (thread_start_height_apron)) * thread_width + (j + 1 - (thread_start_width_apron));
            if(sub_fN[sub_right_index] == -1){
                sub_fN[sub_right_index] = -2;
                enqueue(block_queue, &front, &rear, right_index);
            }
        }
        /* up left */
        if(i - 1 >= thread_start_height && j - 1 >= thread_start_width){
            int32_t up_left_index = (i - 1) * width + (j - 1);
            int32_t sub_up_left_index = (i - 1 - (thread_start_height_apron)) * thread_width + (j - 1 - (thread_start_width_apron));
            if(sub_fN[sub_up_left_index] == -1){
                sub_fN[sub_up_left_index] = -2;
                enqueue(block_queue, &front, &rear, up_left_index);
            }
        }
        /* up right */
        if(i - 1 >= thread_start_height && j + 1 < thread_end_width){
            int32_t up_right_index = (i - 1) * width + (j + 1);
            int32_t sub_up_right_index = (i - 1 - (thread_start_height_apron)) * thread_width + (j + 1 - (thread_start_width_apron));
            if(sub_fN[sub_up_right_index] == -1){
                sub_fN[sub_up_right_index] = -2;
                enqueue(block_queue, &front, &rear, up_right_index);
            }
        }
        /* down left */
        if(i + 1 < thread_end_height && j - 1 >= thread_start_width){
            int32_t down_left_index = (i + 1) * width + (j - 1);
            int32_t sub_down_left_index = (i + 1 - (thread_start_height_apron)) * thread_width + (j - 1 - (thread_start_width_apron));
            if(sub_fN[sub_down_left_index] == -1){
                sub_fN[sub_down_left_index] = -2;
                enqueue(block_queue, &front, &rear, down_left_index);
            }
        }
        /* down right */
        if(i + 1 < thread_end_height && j + 1 < thread_end_width){
            int32_t down_right_index = (i + 1) * width + (j + 1);
            int32_t sub_down_right_index = (i + 1 - (thread_start_height_apron)) * thread_width + (j + 1 - (thread_start_width_apron));
            if(sub_fN[sub_down_right_index] == -1){
                sub_fN[sub_down_right_index] = -2;
                enqueue(block_queue, &front, &rear, down_right_index);
            }
        }
    }
    __syncthreads();

}


void edge_link_bfs(
    int32_t *input, int32_t *output, int start_width, int end_width,
    int start_height, int end_height, int32_t Th, int32_t Tl
){
    int width = end_width - start_width;
    int height = end_height - start_height;
    int size = width * height;
    int32_t *d_input, *d_output;
    cudaMalloc((void **)&d_input, size * sizeof(int32_t));
    cudaMalloc((void **)&d_output, size * sizeof(int32_t));
    cudaMemcpy(d_input, input, size * sizeof(int32_t), cudaMemcpyHostToDevice);
    cudaMemcpy(d_output, output, size * sizeof(int32_t), cudaMemcpyHostToDevice);
    
    dim3 dimBlock(BLOCK_SIZE_BFS, BLOCK_SIZE_BFS);
    dim3 dimGrid(width / dimBlock.x + 1, height / dimBlock.y + 1);

    for(int i = 0; i < ITER; i++){
        edge_link_bfs_kernel<<<dimGrid, dimBlock>>>(
            d_input, d_output, start_width, end_width, start_height, end_height, Th, Tl
        );

        cudaError_t cudaerr = cudaDeviceSynchronize();
        if (cudaerr != cudaSuccess)
            printf("kernel launch failed with error \"%s\".\n",
                cudaGetErrorString(cudaerr));
    }
    
    cudaMemcpy(output, d_output, size * sizeof(int32_t), cudaMemcpyDeviceToHost);
    cudaFree(d_input);
    cudaFree(d_output);
}