
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <iostream>
#include <ctime>


__global__ void sumArraysOnDevice(float* A, float* B, float* C, int N)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if(idx < N)
		C[idx] = A[idx] + B[idx];
}

__global__ void sumArraysZeroCopy(float* A, float* B, float* C, int N)
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

	if(deviceProp.canMapHostMemory == false)
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
	float* hostRef;
	float* gpuRef;
	h_a = (float*)malloc(nBytes);
	h_b = (float*)malloc(nBytes);
	hostRef = (float*)malloc(nBytes);
	gpuRef = (float*)malloc(nBytes);

	initialData(h_a, nElem);
	initialData(h_b, nElem);
	memset(hostRef, 0, nBytes);
	memset(gpuRef, 0, nBytes);

	sumArraysOnHost(h_a, h_b, hostRef, nElem);

	float* d_a;
	float* d_b;
	float* d_c;
	cudaMalloc((float**)&d_a, nBytes);
	cudaMalloc((float**)&d_b, nBytes);
	cudaMalloc((float**)&d_c, nBytes);

	cudaMemcpy(d_a, h_a, nBytes, cudaMemcpyHostToDevice);
	cudaMemcpy(d_b, h_b, nBytes, cudaMemcpyHostToDevice);
	int nLen = 32;
	dim3 block(nLen);
	dim3 grid((nElem + block.x - 1) / block.x);
	sumArraysOnDevice<<<grid, block >>>(d_a, d_b, d_c,nElem);
	cudaMemcpy(gpuRef, d_c, nBytes, cudaMemcpyDeviceToHost);

	checkResult(hostRef, gpuRef, nElem);

	cudaFree(d_a);
	cudaFree(d_b);
	free(h_a);
	free(h_b);

	// part 2 pass the pointer to device
	unsigned int flags = cudaHostAllocMapped;
	cudaHostAlloc((void**)&h_a, nBytes, flags);
	cudaHostAlloc((void**)&h_b, nBytes, flags);

	initialData(h_a, nElem);
	initialData(h_b, nElem);
	memset(hostRef, 0, nBytes);
	memset(gpuRef, 0, nBytes);

	cudaHostGetDevicePointer((void**)&d_a, (void*)h_a, 0);
	cudaHostGetDevicePointer((void**)&d_b, (void*)h_b, 0);


	sumArraysOnHost(h_a, h_b, hostRef, nElem);
	sumArraysZeroCopy<<<grid, block >>>(d_a, d_b, d_c, nElem);
	cudaMemcpy(gpuRef, d_c, nBytes, cudaMemcpyDeviceToHost);
	checkResult(hostRef, gpuRef, nElem);

	cudaFree(d_c);
	cudaFreeHost(h_a);
	cudaFreeHost(h_b);

	free(gpuRef);
	free(hostRef);

	cudaDeviceReset();

	system("Pause");
	return 0;
}
