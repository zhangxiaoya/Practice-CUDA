#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include "squareSum.h"

#include <cuda_runtime.h>

#define DATA_SiZE 1048576

int data[DATA_SiZE];

__global__ static void squareSum(int *data, int *sum, clock_t *time)
{
    int sumT = 0;
    clock_t start = clock();
    for (int i = 0; i < DATA_SiZE; ++i)
    {
        sumT += data[i] * data[i];
    }
    *sum = sumT;
    *time = clock() - start;
}

void generateDat(int *data, int size)
{
    for (int i = 0; i < size; ++i)
    {
        data[i] = rand() % 10;
    }
}

int squareSum()
{
    generateDat(data, DATA_SiZE);
    int *gpuData;
    int *sum;
    clock_t *time;
    cudaMalloc((void **) &gpuData, sizeof(int) * DATA_SiZE);
    cudaMalloc((void **) &sum, sizeof(int));
    cudaMalloc((void **) &time, sizeof(clock_t));
    cudaMemcpy(gpuData, data, sizeof(int) * DATA_SiZE, cudaMemcpyHostToDevice);

    squareSum <<< 1, 1, 0 >>> (gpuData, sum, time);
    int result;
    clock_t time_used;
    cudaMemcpy(&result, sum, sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(&time_used, time, sizeof(clock_t), cudaMemcpyDeviceToHost);

    cudaFree(gpuData);
    cudaFree(sum);
    cudaFree(time);

    printf("(GPU) sum : %d time: %ld\n", result, time_used);

    result = 0;
    return 0;
}