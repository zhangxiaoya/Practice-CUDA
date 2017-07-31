#include <stdio.h>
#include <cuda_runtime.h>

int main(int argc, char const *argv[])
{
    int deviceCount = 0;
    cudaError_t errorId = cudaGetDeviceCount(&deviceCount);

    if (errorId != cudaSuccess) 
    {
        printf("cudaGetDeviceCount returned %d\n-> %s\n", (int)errorId, cudaGetErrorString(errorId));
        printf("Result = FAIL\n");
        exit(EXIT_FAILURE);
    }

    if (deviceCount == 0) 
    {
        printf("There are no available device(s) that support CUDA\n");
    }
    else
    {
        printf("Detected %d CUDA Capable device(s)\n", deviceCount);
    }

    int dev = 0;
    int driverVersion =0;
    int runtimeVersion = 0;

    dev = 0;
    cudaSetDevice(dev);
    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, dev);
    printf("Device %d : %s\n", dev, deviceProp.name);

    cudaDriverGetVersion(&driverVersion);
    cudaRuntimeGetVersion(&runtimeVersion);

    printf(" CUDA Driver Version / Runtime Version %d.%d / %d.%d\n",
            driverVersion/1000,
            (driverVersion%100)/10,
            runtimeVersion/1000,
            (runtimeVersion%100)/10);


    return 0;
}