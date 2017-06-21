#include <stdio.h>
#include <cuda_runtime.h>
#include <string.h>
#include <sys/time.h>
#include <time.h>

#define CHECK(call)                                                         \
{                                                                           \
    const cudaError_t error = call;                                         \
    if(error != cudaSuccess)                                                \
    {                                                                       \
        printf("Error: %s:%d, ", __FILE__, __LINE__);                       \
        printf("code: %d, reason: %s\n", error, cudaGetErrorString(error)); \
        exit(1);                                                            \
    }                                                                       \
}                                                                           \

void initialData(float* ip, int size);

double cpuSecond();

void sumArraysOnHost(float* A, float* B, float* C, int N);

__global__ void sumArraysOnDevice(float* A, float* B, float* C, const int N);

void checkResult(float* hostResult, float* deviceResult, const int N);

int main(int argc, char** argv)
{
    printf("%s Starting....\n", argv[0]);

    int dev = 0;
    cudaDeviceProp deviceProp;
    CHECK(cudaGetDeviceProperties(&deviceProp, dev));
    printf("Using Device %d: %s\n", dev, deviceProp.name);
    CHECK(cudaSetDevice(dev));

    int nElem = 1 << 24;
    printf("Vector size %d\n", nElem);

    size_t nBytes = nElem * sizeof(float);

    float *hA, *hB, *hostResult, * gpuResult;
    hA = (float*)malloc(nBytes);
    hB = (float*)malloc(nBytes);
    hostResult = (float*)malloc(nBytes);
    gpuResult = (float*)malloc(nBytes);

    double iStart, iEnd, iElaps;

    iStart = cpuSecond();
    initialData(hA, nElem);
    initialData(hB, nElem);
    iElaps = cpuSecond() - iStart;

    memset(hostResult, 0, nBytes);
    memset(gpuResult, 0, nBytes);

    iStart = cpuSecond();
    sumArraysOnHost(hA, hB,hostResult,nElem);
    iElaps = cpuSecond() - iStart;
    printf("SumArrayOnCPU Time elapsed %fsec\n", iElaps);

    float *dA, *dB, *dC;
    cudaMalloc((float**)&dA, nBytes);
    cudaMalloc((float**)&dB, nBytes);
    cudaMalloc((float**)&dC, nBytes);

    cudaMemcpy(dA,hA,nBytes,cudaMemcpyHostToDevice);
    cudaMemcpy(dB,hB,nBytes,cudaMemcpyHostToDevice);

    int iLen = 1024;
    dim3 block(iLen);
    dim3 grid((nElem + block.x-1)/block.x);

    iStart = cpuSecond();
    printf("Start: %f\n", iStart);
    sumArraysOnDevice<<<grid, block>>>(dA, dB, dC,nElem);
    cudaDeviceSynchronize();
    iEnd = cpuSecond();
    printf("End: %f\n", iEnd);
    iElaps =  iEnd - iStart;
    printf("SumArrayOnGPU <<<%d, %d>>> Time elapsed %fsec\n", grid.x, block.x, iElaps);

    cudaMemcpy(gpuResult, dC, nBytes, cudaMemcpyDeviceToHost);
    

    checkResult(hostResult, gpuResult,nElem);

    cudaFree(dA);
    cudaFree(dB);
    cudaFree(dC);

    free(hA);
    free(hB);
    free(hostResult);
    free(gpuResult);

    return(0);
}

double cpuSecond()
{
    struct timeval tp;
    gettimeofday(&tp, NULL);
    return ((double)tp.tv_sec + (double)tp.tv_sec * 1.e-6);
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

void sumArraysOnHost(float* A, float* B, float* C, int N)
{
    for(int i=0;i<N;++i)
    {
        C[i] = A[i] + B[i];
    }
}

__global__ void sumArraysOnDevice(float* A, float* B, float* C, const int N)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if( i < N)
        C[i] = A[i] + B[i];
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