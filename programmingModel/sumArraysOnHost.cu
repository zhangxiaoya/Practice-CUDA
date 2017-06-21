#include <stdio.h>
#include <string.h>
#include <sys/time.h>
#include <cuda_runtime.h>

#define CHECK(call)                                                         \
{                                                                           \
    const cudaError_t error = call;                                         \
    if(error != cudaSuccess)                                                \
    {                                                                       \
        printf("Error: %s:%d, ", __FILE__, __LINE);                         \
        printf("code: %d, reason: %s\n", error, cudaGetErrorString(error)); \
        exit(1);                                                            \
    }                                                                       \
}                                                                           \

double cpuSecond()
{
    struct timeval tp;
    gettimeofday(&tp,NULL);
    return ((double)tp.tv_sec + (double)tp.tv_usec*1.e-6);
}

__global__ void sumArraysOnDevice(float* A, float* B, float* C)
{
    C[threadIdx.x] = A[threadIdx.x] + B[threadIdx.x];
}

void sumArraysOnHost(float* A, float* B, float* C, int N)
{
    for(int i=0;i<N;++i)
    {
        C[i] = A[i] + B[i];
    }
}

void initialData(float* ip, int size)
{
    time_t t;
    srand((unsigned int) time(&t));
    for(int i=0;i<size;++i)
    {
        ip[i] = (float)(rand() & 0xFF) / 10.0f;
    }
}

void checkResult(float* hostResult, float* deviceResult, const int N)
{
    double epsilon = 1.0E-8;
    int match = 1;
    for(int i =0;i<N;++i)
    {
        if(abs(hostResult[i] - deviceResult[i]) > epsilon)
        {
            match = 0;
            printf("Array do not match\n");
            printf("host %5.2f gpu %5.2f at current %d\n", hostResult[i], deviceResult[i], i);
            break;
        }
    }
    if(match)
        printf("Array match\n");

    return;
}

int main(void)
{
    // int nElem = 1024;
    int nElem = 1 << 10;
    size_t nBytes = nElem * sizeof(float);

    float *A, *B, *C;
    A = (float*)malloc(nBytes);
    B = (float*)malloc(nBytes);
    C = (float*)malloc(nBytes);

    float *dA, *dB, *dC;
    float *gpuResult;
    cudaMalloc((float**)&dA, nBytes);
    cudaMalloc((float**)&dB, nBytes);
    cudaMalloc((float**)&dC, nBytes);
    gpuResult = (float*)malloc(nBytes);

    initialData(A, nElem);
    initialData(B, nElem);

    cudaMemcpy(dA,A,nBytes,cudaMemcpyHostToDevice);
    cudaMemcpy(dB,B,nBytes,cudaMemcpyHostToDevice);

    dim3 block(nElem);
    dim3 grid(nElem / block.x);

    double iStart = cpuSecond();
    sumArraysOnDevice<<<grid,block>>>(dA,dB,dC);
    cudaDeviceSynchronize();
    double iElaps = cpuSecond() - iStart;

    printf("GPU time is %f\n",iElaps);

    cudaMemcpy(gpuResult,dC,nBytes,cudaMemcpyDeviceToHost);

    sumArraysOnHost(A,B,C,nElem);

    checkResult(C,gpuResult,nElem);
    free(A);
    free(B);
    free(C);
    free(gpuResult);

    cudaFree(dA);
    cudaFree(dB);
    cudaFree(dC);

    return 0;
}