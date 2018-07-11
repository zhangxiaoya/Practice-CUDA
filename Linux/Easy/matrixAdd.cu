#include <cuda_runtime.h>
#include <stdio.h>

void Generate(int* M, int width, int height)
{
    for(int r = 0; r < height; r ++)
    {
        for(int c = 0;c< width; c++)
        {
            *(M + (r * width) + c) = r*width +c;
        }
    }
}

__global__ void MatrixAdd(int* A, int * B, int* C, int width, int height)
{
    int r = threadIdx.x + blockIdx.x * blockDim.x;
    int c = threadIdx.y + blockIdx.y * blockDim.y;
    if(r >= height || c >= width)
        return;
    C[r * width + c] = A[r * width + c] + B[r * width + c];

    printf("row: %d, col: %d, sum = %d\n", r,c,C[r * width + c]);
}

void MatrixAddOnDevice(int width, int height)
{
    int nBytes = sizeof(int) * width * height;
    int* A = (int*)malloc(nBytes);
    int* B = (int*)malloc(nBytes);
    int* C = (int*)malloc(nBytes);

    Generate(A, width, height);
    Generate(B, width, height);

    int* DA;
    int* DB;
    int* DC;
    cudaMalloc((int**)&DA, nBytes);
    cudaMalloc((int**)&DB, nBytes);
    cudaMalloc((int**)&DC, nBytes);

    cudaMemcpy(DA, A, nBytes, cudaMemcpyHostToDevice);
    cudaMemcpy(DB, B, nBytes, cudaMemcpyHostToDevice);

    dim3 block(32,32);
    dim3 grid((width + block.x - 1) / block.x,(height + block.y -1)/ block.y);

    MatrixAdd<<<grid,block>>>(DA,DB,DC,width,height);
    
    cudaMemcpy(C,DC,nBytes,cudaMemcpyDeviceToHost);
    for(int r = 0; r < width; ++r)
    {
        for(int c= 0 ; c < height; ++c)
        {
            printf("%d ", C[r * width + c]);
        }
        printf("\n");
    }

    cudaFree(DA);
    cudaFree(DB);
    cudaFree(DC);
    free(A);
    free(B);
    free(C);
}