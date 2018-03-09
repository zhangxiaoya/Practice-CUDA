#include <cuda_runtime.h>
#include <stdio.h>
#include <cmath>
#include <device_launch_parameters.h>
#include <windows.h>
/*
* A simple example of using an array of structures to store data on the device.
* This example is used to study the impact on performance of data layout on the
* GPU.
*
* AoS: one contiguous 64-bit read to get x and y (up to 300 cycles)
*/

#define CHECK(call)                                                            \
{                                                                              \
    const cudaError_t error = call;                                            \
    if (error != cudaSuccess)                                                  \
    {                                                                          \
        fprintf(stderr, "Error: %s:%d, ", __FILE__, __LINE__);                 \
        fprintf(stderr, "code: %d, reason: %s\n", error,                       \
                cudaGetErrorString(error));                                    \
    }                                                                          \
}

typedef unsigned long long uint64_t;

#define LEN 1<<22

struct innerStruct
{
	float x;
	float y;
};

struct innerArray
{
	float x[LEN];
	float y[LEN];
};

void initialInnerStruct(innerStruct *ip, int size)
{
	for (int i = 0; i < size; i++)
	{
		ip[i].x = (float)(rand() & 0xFF) / 100.0f;
		ip[i].y = (float)(rand() & 0xFF) / 100.0f;
	}

	return;
}

void testInnerStructHost(innerStruct *A, innerStruct *C, const int n)
{
	for (int idx = 0; idx < n; idx++)
	{
		C[idx].x = A[idx].x + 10.f;
		C[idx].y = A[idx].y + 20.f;
	}

	return;
}

void checkInnerStruct(innerStruct *hostRef, innerStruct *gpuRef, const int N)
{
	double epsilon = 1.0E-8;
	bool match = 1;

	for (int i = 0; i < N; i++)
	{
		if (abs(hostRef[i].x - gpuRef[i].x) > epsilon)
		{
			match = 0;
			printf("different on %dth element: host %f gpu %f\n", i,
				hostRef[i].x, gpuRef[i].x);
			break;
		}

		if (abs(hostRef[i].y - gpuRef[i].y) > epsilon)
		{
			match = 0;
			printf("different on %dth element: host %f gpu %f\n", i,
				hostRef[i].y, gpuRef[i].y);
			break;
		}
	}

	if (!match)  printf("Arrays do not match.\n\n");
}

__global__ void testInnerStruct(innerStruct *data, innerStruct * result,
	const int n)
{
	unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;

	if (i < n)
	{
		innerStruct tmp = data[i];
		tmp.x += 10.f;
		tmp.y += 20.f;
		result[i] = tmp;
	}
}

__global__ void warmup(innerStruct *data, innerStruct * result, const int n)
{
	unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;

	if (i < n)
	{
		innerStruct tmp = data[i];
		tmp.x += 10.f;
		tmp.y += 20.f;
		result[i] = tmp;
	}
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

int main(int argc, char **argv)
{
	// set up device
	int dev = 0;
	cudaDeviceProp deviceProp;
	CHECK(cudaGetDeviceProperties(&deviceProp, dev));
	printf("%s test struct of array at ", argv[0]);
	printf("device %d: %s \n", dev, deviceProp.name);
	CHECK(cudaSetDevice(dev));

	// allocate host memory
	int nElem = LEN;
	size_t nBytes = nElem * sizeof(innerStruct);
	innerStruct     *h_A = (innerStruct *)malloc(nBytes);
	innerStruct *hostRef = (innerStruct *)malloc(nBytes);
	innerStruct *gpuRef = (innerStruct *)malloc(nBytes);

	// initialize host array
	initialInnerStruct(h_A, nElem);
	testInnerStructHost(h_A, hostRef, nElem);

	// allocate device memory
	innerStruct *d_A, *d_C;
	CHECK(cudaMalloc((innerStruct**)&d_A, nBytes));
	CHECK(cudaMalloc((innerStruct**)&d_C, nBytes));

	// copy data from host to device
	CHECK(cudaMemcpy(d_A, h_A, nBytes, cudaMemcpyHostToDevice));

	// set up offset for summaryAU: It is blocksize not offset. Thanks.CZ
	int blocksize = 128;

	if (argc > 1) blocksize = atoi(argv[1]);

	// execution configuration
	dim3 block(blocksize, 1);
	dim3 grid((nElem + block.x - 1) / block.x, 1);

	// kernel 1: warmup
	double iStart = seconds();
	warmup<<<grid, block >>>(d_A, d_C, nElem);
	CHECK(cudaDeviceSynchronize());
	double iElaps = seconds() - iStart;
	printf("warmup      <<< %3d, %3d >>> elapsed %f sec\n", grid.x, block.x,
		iElaps);
	CHECK(cudaMemcpy(gpuRef, d_C, nBytes, cudaMemcpyDeviceToHost));
	checkInnerStruct(hostRef, gpuRef, nElem);
	CHECK(cudaGetLastError());

	// kernel 2: testInnerStruct
	iStart = seconds();
	testInnerStruct<<<grid, block >>>(d_A, d_C, nElem);
	CHECK(cudaDeviceSynchronize());
	iElaps = seconds() - iStart;
	printf("innerstruct <<< %3d, %3d >>> elapsed %f sec\n", grid.x, block.x,
		iElaps);
	CHECK(cudaMemcpy(gpuRef, d_C, nBytes, cudaMemcpyDeviceToHost));
	checkInnerStruct(hostRef, gpuRef, nElem);
	CHECK(cudaGetLastError());

	// free memories both host and device
	CHECK(cudaFree(d_A));
	CHECK(cudaFree(d_C));
	free(h_A);
	free(hostRef);
	free(gpuRef);

	// reset device
	CHECK(cudaDeviceReset());
	return EXIT_SUCCESS;
}