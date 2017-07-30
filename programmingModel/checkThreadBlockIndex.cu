#include <stdio.h>
#include <cuda_runtime.h>
#include <time.h>
#include <sys/time.h>

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

void printMatrix(int* data, const int nx, const int ny)
{
    int* pdata = data;
    printf("\nMatrix : (%d, %d)\n", nx, ny);
    for(int i=0;i<ny;++i)
    {
        for(int j =0;j<nx;++j)
        {
            printf("%3d", pdata[j]);
        }
        pdata += nx;
        printf("\n");
    }
    printf("\n");
}

__global__ void PrintThreadBlockIndex(int* data, const int nx, const int ny)
{
    int ix = threadIdx.x + blockIdx.x * blockDim.x;
    int iy = threadIdx.y + blockIdx.y * blockDim.y;
    unsigned int idex = ix + iy * nx;

    printf("thread index (%d, %d), block index (%d, %d) coordinate (%d, %d), global index %d and value %d\n", 
            threadIdx.x, threadIdx.y, blockIdx.x, blockIdx.y, ix, iy,idex, data[idex]);
}

void InitData(int* data, const int size)
{
    time_t t;
    srand((unsigned)time(&t));
    for(int i =0; i < size; ++i)
    {
        data[i] = (int)(rand() & 0xFF) /10.0f;
    }
}

int main(int argc, char const *argv[])
{
    printf("%s Starting ... \n", argv[0]);

    int dev = 0;
    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp,dev);
    printf("Using Device %d, %s\n", dev, deviceProp.name);
    cudaSetDevice(dev);

    int nx = 8;
    int ny = 6;
    int nxy = nx * ny;

    size_t nBytes = nxy * sizeof(int);

    int* hA;
    hA = (int*)malloc(nBytes);

    InitData(hA, nxy);
    printMatrix(hA, nx, ny);

    int* dMat;
    cudaMalloc((void**)&dMat, nBytes);

    cudaMemcpy(dMat, hA, nBytes, cudaMemcpyHostToDevice);

    dim3 block(4,2);
    dim3 grid((nx + block.x -1) / block.x, (ny + block.y - 1) / block.y);

    PrintThreadBlockIndex<<<grid, block>>>(dMat, nx, ny);

    cudaDeviceSynchronize();
    cudaFree(dMat);
    free(hA);

    cudaDeviceReset();
    return 0;
}