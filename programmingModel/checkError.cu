#include <stdio.h>
#include <cuda_runtime.h>
#include <time.h>

#define CHECK(call)                                                          \
{                                                                            \
    const cudaError_t error = call;                                          \
    if(error 1= cudaSuccess)                                                 \
    {                                                                        \
        printf("Error: %s : %d, ", __FILE__, __LINE__);                      \
        printf("code: %d, reason: %s\n", error, cudaGetErrorString(error));  \
        exit(1);                                                             \
    }                                                                        \
}

void CheckResult(float* hostResult, float* deviceResult, const int N)
{
    double epsilon = 1.0E-8;
    int match = 1;
    for(int i = 0; i < N; ++i)
    {
        if(abs(hostResult[i] - deviceResult[i]) > epsilon)
        {
            match = 0;
            printf("Array do not match!\n");
            printf("Host %5.2gf GPU %5.2f at current %d \n", hostResult[i], deviceResult[i], i);
            break;
        }
    }
    if(match == 1)
    {
        printf("Array match.\n\n");
    }
    return;
}

void InitData(float* data, const int size)
{
    time_t t;
    srand((unsigned)time(&t));
    for(int i =0; i < size; ++i)
    {
        data[i] = (float)(rand() & 0xFF) /10.0f;
    }
}

void SumArrayOnHost(float* A, float* B, float* C, const int size)
{
    for(int i =0; i< size;++i)
    {
        C[i] = A[i] + B[i];
    }
}

__global__ void SumArrayOnDevice(float* A, float* B, float* C)
{
    int i = threadIdx.x;
    C[i] = A[i] + B[i];
}

int main(int argc, char const *argv[])
{
    printf("%s Starting ... \n", argv[0]);

    int dev = 0;
    cudaSetDevice(dev);

    int nElem = 128;
    printf("Array size is %d\n", nElem);

    size_t nBytes = nElem * sizeof(float);

    float* hA;
    float* hB;
    float* hostResult;
    float* deviceResult;
    hA = (float*)malloc(nBytes);
    hB = (float*)malloc(nBytes);
    hostResult = (float*)malloc(nBytes);
    deviceResult = (float*)malloc(nBytes);

    InitData(hA, nElem);
    InitData(hB, nElem);

    memset(hostResult, 0, nBytes);
    memset(deviceResult, 0, nBytes);

    float* dA;
    float* dB;
    float* dC;
    cudaMalloc((float**)&dA, nBytes);
    cudaMalloc((float**)&dB, nBytes);
    cudaMalloc((float**)&dC, nBytes);

    cudaMemcpy(dA, hA, nBytes, cudaMemcpyHostToDevice);
    cudaMemcpy(dB, hB, nBytes, cudaMemcpyHostToDevice);

    dim3 block(nElem);
    dim3 grid((nElem + block.x - 1) / block.x);

    SumArrayOnDevice<<<grid,block>>>(dA, dB, dC);
    printf("Execution configuration <<<%d, %d>>>\n", grid.x, block.x);

    cudaMemcpy(deviceResult, dC, nBytes, cudaMemcpyDeviceToHost);

    SumArrayOnHost(hA, hB,hostResult,nElem);

    CheckResult(hostResult, deviceResult,nElem);

    cudaFree(dA);
    cudaFree(dB);
    cudaFree(dC);

    free(hA);
    free(hB);
    free(hostResult);
    free(deviceResult);

    return 0;
}