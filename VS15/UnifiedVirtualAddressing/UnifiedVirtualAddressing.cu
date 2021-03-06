
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <iostream>
#include <ctime>


__global__ void sumArraysZeroCopyUVA(float* A, float* B, float* C, int N)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx < N)
		C[idx] = A[idx] + B[idx];
}

void sumArraysOnHost(float* A, float* B, float* C, int N)
{
	for (int i = 0; i<N; ++i)
	{
		C[i] = A[i] + B[i];
	}
}

void initialData(float* ip, int size)
{
	time_t t;
	srand((unsigned int)time(&t));
	for (int i = 0; i<size; ++i)
	{
		ip[i] = (float)(rand() & 0xFF) / 10.0f;
	}
}

void checkResult(float* hostResult, float* deviceResult, const int N)
{
	double epsilon = 1.0E-8;
	int match = 1;
	for (int i = 0; i<N; ++i)
	{
		if (abs(hostResult[i] - deviceResult[i]) > epsilon)
		{
			match = 0;
			printf("Array do not match\n");
			printf("host %5.2f gpu %5.2f at current %d\n", hostResult[i], deviceResult[i], i);
			break;
		}
	}
	if (match)
		printf("Array match\n");

	return;
}

int main(int argc, char* argv[])
{
	int dev = 0;
	cudaSetDevice(dev);

	cudaDeviceProp deviceProp;
	cudaGetDeviceProperties(&deviceProp, dev);

	if (deviceProp.canMapHostMemory == false)
	{
		printf("Device %d dose not support mapping CPU host memory!\n", dev);
		cudaDeviceReset();
		return -1;
	}

	printf("Using Device %d, %s\n", dev, deviceProp.name);

	int iPower = 10;
	if (argc > 1)
	{
		iPower = atoi(argv[1]);
	}
	int nElem = 1 << iPower;
	size_t nBytes = nElem * sizeof(float);

	// part 1 use device memory
	float* h_a;
	float* h_b;
	float* d_c;
	float* hostRef;
	float* gpuRef;

	unsigned int flags = cudaHostAllocMapped;
	cudaHostAlloc((void**)&h_a, nBytes, flags);
	cudaHostAlloc((void**)&h_b, nBytes, flags);
	cudaHostAlloc((void**)&d_c, nBytes, flags);

	hostRef = (float*)malloc(nBytes);

	initialData(h_a, nElem);
	initialData(h_b, nElem);
	memset(hostRef, 0, nBytes);

	sumArraysOnHost(h_a, h_b, hostRef, nElem);

	int nLen = 32;
	dim3 block(nLen);
	dim3 grid((nElem + block.x - 1) / block.x);

	sumArraysZeroCopyUVA<<<grid, block >>>(h_a, h_b, d_c, nElem);

	checkResult(hostRef, d_c, nElem);

	cudaFreeHost(d_c);
	cudaFreeHost(h_a);
	cudaFreeHost(h_b);

	free(hostRef);

	cudaDeviceReset();

	system("Pause");
	return 0;
}
