#include <stdio.h>

__global__ void helloFromGPU()
{
    printf("Hello World From GPU!\n");
}

int main()
{
    // helloFromGPU
    printf("Hello World From CPU!\n");

    // helloFromGPU
    helloFromGPU<<<1, 10>>>();
    cudaDeviceReset();
    return 0;
}