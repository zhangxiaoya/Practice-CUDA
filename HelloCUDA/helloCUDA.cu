#include <stdio.h>

__global__ void helloFromGPU()
{
    if(threadIdx.x == 5)
        printf("Hello World From GPU %d!\n",threadIdx);
}

int main()
{
    // helloFromGPU
    printf("Hello World From CPU!\n");

    // helloFromGPU
    helloFromGPU<<<1, 10>>>();
    cudaDeviceReset();
    //cudaDeviceSynchronize();
    return 0;
}