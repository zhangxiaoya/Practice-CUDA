#include <stdio.h>
#include <cuda_runtime.h>
#include <sys/time.h>

void sumMatrixOnHost(float* A, float* B, float* C, const int nx, const int ny)
{
    float* pA = A;
    float* pB = B;
    float* pC = C;

    for(int iy = 0;iy < ny; ++iy)
    {
        for(int ix = 0;ix < nx; ++ix)
        {
            pC[ix] = pA[ix] + pB[ix];
        }
        pA += nx;
        pB += nx;
        pC += nx;
    }
}

__global__ void sumMatrixOnDevice(float* MatA, float* MatB, float* MatC, const int nx, const int ny)
{
    unsigned int ix = threadIdx.x + blockDim.x * blockIdx.x;
    unsigned int iy = threadIdx.y + blockDim.y * blockIdx.y;

    unsigned int idx = ix + iy * nx;

    if(ix < nx && iy < ny)
    {
        MatC[idx] = MatA[idx] + MatB[idx];
    }
}

double cpuSecond()
{
    timeval tp;
    gettimeofday(&tp, NULL);
    return ((double)tp.tv_sec + (double)tp.tv_usec*1.e-6);
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

int main(int argc, char const *argv[])
{
    int dev = 0;
    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp,dev);
    printf("Using Device %d, %s\n", dev, deviceProp.name);
    cudaSetDevice(dev);   

    int nx = 1 << 13;
    int ny = 1 << 13;

    int nxy = nx * ny;
    int nBytes = nxy * sizeof(float);
    printf("Matrix size : (%d, %d)", nx,ny);


    float* hA;
    float* hB;
    float* resultHost;
    float* resultDevice;

    hA = (float*)malloc(nBytes);
    hB = (float*)malloc(nBytes);
    resultDevice = (float*)malloc(nBytes);
    resultHost = (float*)malloc(nBytes);

    InitData(hA, nxy);
    InitData(hB, nxy);

    memset(resultDevice, 0, nBytes);
    memset(resultHost, 0, nBytes);

    double iStart, iElaps;

    iStart = cpuSecond();
    sumMatrixOnHost(hA, hB,resultHost,nx, ny);
    iElaps = cpuSecond() - iStart;

    printf("Sum Matrix on host time elapsed %f\n", iElaps);

    float* dMatA;
    float* dMatB;
    float* dMatC;
    cudaMalloc((void**)&dMatA,nBytes);
    cudaMalloc((void**)&dMatB,nBytes);
    cudaMalloc((void**)&dMatC,nBytes);

    cudaMemcpy(dMatA, hA, nBytes, cudaMemcpyHostToDevice);
    cudaMemcpy(dMatB, hB, nBytes, cudaMemcpyHostToDevice);

    int dimx = 32;
    int dimy = 32;
    dim3 block(dimx, dimy);
    dim3 grid((nx + block.x -1) / block.x, (ny + block.y - 1) / block.y);

    iStart = cpuSecond();
    sumMatrixOnDevice<<<grid, block>>>(dMatA, dMatB, dMatC,nx,ny);
    cudaDeviceSynchronize();
    iElaps = cpuSecond() - iStart;
    printf("sumMatrixOnDevice <<<(%d,%d), (%d,%d)>>> elapsed %f sec\n", grid.x,grid.y, block.x, block.y, iElaps);

    cudaMemcpy(resultDevice, dMatC, nBytes, cudaMemcpyDeviceToHost);

    CheckResult(resultHost,resultDevice, nxy);

    cudaFree(dMatA);
    cudaFree(dMatB);
    cudaFree(dMatC);

    free(hA);
    free(hB);
    free(resultHost);
    free(resultDevice);

    cudaDeviceReset();
    return 0;
}