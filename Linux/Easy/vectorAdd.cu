#include <stdio.h>
#include <cuda_runtime.h>

__global__ void VectorAdd(int* A, int* B, int* C, int N)
{
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if(idx > N )
        return;
    C[idx] = A[idx] + B[idx];
}

void GenerateData(int* data, int N)
{
    for(int i = 0; i< N; ++i)
        (*(data+i)) = i;
}
void VectorAddOnDevice(int N)
{
    int* HA = (int*)malloc(sizeof(int) * N);
    int* HB = (int*)malloc(sizeof(int) * N);
    int* HC = (int*)malloc(sizeof(int) * N);

    GenerateData(HA, N);
    GenerateData(HB, N);

    int nBytes = N * sizeof(int);
    int* DA;
    int* DB;
    int* DC;
    cudaMalloc((int**)&DA, nBytes);
    cudaMalloc((int**)&DB, nBytes);
    cudaMalloc((int**)&DC, nBytes);

    cudaMemcpy(DA, HA,nBytes, cudaMemcpyHostToDevice);
    cudaMemcpy(DB, HB,nBytes, cudaMemcpyHostToDevice);

    dim3 block(32);
    dim3 grid((N + block.x -1) / block.x);

    VectorAdd<<<grid, block>>>(DA, DB,DC,N);
    cudaMemcpy(HC, DC, nBytes, cudaMemcpyDeviceToHost);

    for(int i = 0; i < N;++i)
        printf("%d ", HC[i]);
    printf("\n");

    cudaFree(DA);
    cudaFree(DB);
    cudaFree(DC);

    free(HA);
    free(HB);
    free(HC);
}