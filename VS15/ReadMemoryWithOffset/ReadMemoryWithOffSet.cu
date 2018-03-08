
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <chrono>
#include <windows.h>

using namespace  std;

void checkResult(float *hostRef, float *gpuRef, const int N)
{
	double epsilon = 1.0E-8;
	bool match = 1;

	for (int i = 0; i < N; i++)
	{
		if (abs(hostRef[i] - gpuRef[i]) > epsilon)
		{
			match = 0;
			printf("different on %dth element: host %f gpu %f\n", i, hostRef[i],
				gpuRef[i]);
			break;
		}
	}

	if (!match)  printf("Arrays do not match.\n\n");
	else printf("Matched!\n");
}

void initialData(float* ip, int size)
{
	time_t t;
	srand(static_cast<unsigned int>(time(&t)));
	for (auto i = 0; i<size; ++i)
	{
		ip[i] = static_cast<float>(rand() & 0xFF) / 10.0f;
	}
}

void sumArraysOnHost(float *A, float *B, float *C, const int n, int offset)
{
	for (auto idx = offset, k = 0; idx < n; idx++, k++)
	{
		C[k] = A[idx] + B[idx];
	}
}

__global__ void readOffset(float *A, float *B, float *C, const int n, int offset) 
{
	auto i = blockIdx.x * blockDim.x + threadIdx.x;
	auto k = i + offset;
	if (k < n) C[i] = A[k] + B[k];
}

__global__ void warmup(float *A, float *B, float *C, const int n, int offset)
{
	auto i = blockIdx.x * blockDim.x + threadIdx.x;
	auto k = i + offset;

	if (k < n) C[i] = A[k] + B[k];
}


int gettimeofday(struct timeval * tp)
{
	// Note: some broken versions only have 8 trailing zero's, the correct epoch has 9 trailing zero's
	// This magic number is the number of 100 nanosecond intervals since January 1, 1601 (UTC)
	// until 00:00:00 January 1, 1970 
	static const uint64_t EPOCH = ((uint64_t)116444736000000000ULL);

	SYSTEMTIME  system_time;
	FILETIME    file_time;
	uint64_t    time;

	GetSystemTime(&system_time);
	SystemTimeToFileTime(&system_time, &file_time);
	time = ((uint64_t)file_time.dwLowDateTime);
	time += ((uint64_t)file_time.dwHighDateTime) << 32;

	tp->tv_sec = (long)((time - EPOCH) / 10000000L);
	tp->tv_usec = (long)(system_time.wMilliseconds * 1000);
	return 0;
}

inline double seconds()
{
	timeval tp;
	auto i = gettimeofday(&tp);
	return (static_cast<double>(tp.tv_sec) + static_cast<double>(tp.tv_usec) * 1.e-6);
}

int main(int argc, char** argv)
{
	// Init Device
	auto dev = 0;
	cudaSetDevice(dev);

	cudaDeviceProp deviceProp;
	cudaGetDeviceProperties(&deviceProp, dev);
	printf("%s Starting at ", argv[0]);
	printf("device %d:%s ", dev, deviceProp.name);

	// Setup array size
	unsigned int nElem = 1 << 20;
	printf("with array size %d\n", nElem);
	size_t nBytes = nElem * sizeof(float);

	auto blockSize = 512;
	auto offset = 0;
	if (argc > 1) offset = atoi(argv[1]);
	if (argc > 2) blockSize = atoi(argv[2]);

	dim3 block(blockSize,1);
	dim3 grid((nElem + block.x - 1) / block.x, 1);

	auto h_a = static_cast<float*>(malloc(nBytes));
	auto h_b = static_cast<float*>(malloc(nBytes));
	auto hostRef = static_cast<float *>(malloc(nBytes));
	auto gpuRef = static_cast<float *>(malloc(nBytes));

	// initialize host array
	initialData(h_a, nElem);
	memcpy(h_b, h_a, nBytes);

	// summary at host side
	sumArraysOnHost(h_a, h_b, hostRef, nElem, offset);

	// allocate device memory
	float* d_a;
	float* d_b;
	float* d_c;
	cudaMalloc(reinterpret_cast<void**>(&d_a), nBytes);
	cudaMalloc(reinterpret_cast<void**>(&d_b), nBytes);
	cudaMalloc(reinterpret_cast<void**>(&d_c), nBytes);

	cudaMemcpy(d_a, h_a, nBytes, cudaMemcpyHostToDevice);
	cudaMemcpy(d_b, h_b, nBytes, cudaMemcpyHostToDevice);

	// kernel 1 
	double start = seconds();
	warmup<<<grid, block>>>(d_a, d_b, d_c, nElem, offset);
	cudaDeviceSynchronize();
	double iElaps = seconds() - start;
	printf("warmup <<< %4d, %4d >>> offset %4d elapsed %f sec\n",
		grid.x, block.x,
		offset, iElaps);

	start = seconds();
	readOffset<<<grid, block>>>(d_a, d_b, d_c, nElem, offset);
	cudaDeviceSynchronize();
	iElaps = seconds() - start;
	printf("readOffset <<< %4d, %4d >>> offset %4d elapsed %f sec\n",
		grid.x, block.x,
		offset, iElaps);

	// copy kernel result back to host side and check device results
	cudaMemcpy(gpuRef, d_c, nBytes, cudaMemcpyDeviceToHost);
	checkResult(hostRef, gpuRef, nElem - offset);
	// copy kernel result back to host side and check device results
	cudaMemcpy(gpuRef, d_c, nBytes, cudaMemcpyDeviceToHost);
	checkResult(hostRef, gpuRef, nElem - offset);
	// copy kernel result back to host side and check device results
	cudaMemcpy(gpuRef, d_c, nBytes, cudaMemcpyDeviceToHost);
	checkResult(hostRef, gpuRef, nElem - offset);
	// free host and device memory
	cudaFree(d_a);
	cudaFree(d_b);
	cudaFree(d_c);
	free(h_a);
	free(h_b);
		// reset device
		cudaDeviceReset();

	system("pause");
	return 0;
}