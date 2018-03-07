
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <iostream>

#include <stdio.h>

__device__ float devData;

__global__ void checkGlobalVariable()
{
	printf("Device: The value of global variable is %f\n", devData);
	devData += 2.0;
}

int main()
{
	float value = 3.14f;
	cudaMemcpyToSymbol(&devData, &value, sizeof(value));
	printf("Host: copied %f to the global variable\n", value);

	checkGlobalVariable<<<1, 1 >>> ();

	cudaMemcpyFromSymbol(&value, &devData, sizeof(value));
	printf("Host: Value changed by kernel to %f\n", value);

	cudaDeviceReset();

	system("Pause");
    return 0;
}