#include <stdio.h>
#include <string.h>
#include <time.h>
#include <cuda_runtime.h>

__global__ void sumArraysOnDevice(float* A, float* B, float* C, int N)
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
    int nElem = 1024;
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

    cudaMemCpy(dA,A,nBytes,cudaMemcpyHostToDevice);
    cudaMemCpy(dB,B,nBytes,cudaMemcpyHostToDevice);

    cudaMemCpy(gpuResult,dC,nBytes,cudaMemCpyDeviceToHost);

    sumArraysOnHost(A,B,C,nElem);
    sumArraysOnDevice<<<1,1024>>>(dA,dB,dC,nElem);

    checkResult(C,gpuResult,nElem);
    free(A);
    free(B);
    free(C);

    cudaFree(dA);
    cudaFree(dB);
    cudaFree(dC);

    return 0;
}