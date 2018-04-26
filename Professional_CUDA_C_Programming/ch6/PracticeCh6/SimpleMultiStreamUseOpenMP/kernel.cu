#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <cmath>
#include <cstdio>
#include <omp.h>

#define N 300
#define NSTREAM 4

__global__ void kernel_1()
{
	double sum = 0.0;

	for (int i = 0; i < N; i++)
	{
		sum = sum + tan(0.1) * tan(0.1);
	}
}

__global__ void kernel_2()
{
	double sum = 0.0;

	for (int i = 0; i < N; i++)
	{
		sum = sum + tan(0.1) * tan(0.1);
	}
}

__global__ void kernel_3()
{
	double sum = 0.0;

	for (int i = 0; i < N; i++)
	{
		sum = sum + tan(0.1) * tan(0.1);
	}
}

__global__ void kernel_4()
{
	double sum = 0.0;

	for (int i = 0; i < N; i++)
	{
		sum = sum + tan(0.1) * tan(0.1);
	}
}

int main(int argc, char **argv)
{
	int n_streams = NSTREAM;
	int isize = 1;
	int iblock = 1;

	float elapsed_time;


	int dev = 0;
	cudaDeviceProp deviceProp;
	cudaGetDeviceProperties(&deviceProp, dev);
	printf("> Using Device %d: %s with num_streams=%d\n", dev, deviceProp.name, n_streams);
	cudaSetDevice(dev);

	// check if device support hyper-q
	if (deviceProp.major < 3 || (deviceProp.major == 3 && deviceProp.minor < 5))
	{
		if (deviceProp.concurrentKernels == 0)
		{
			printf("> GPU does not support concurrent kernel execution (SM 3.5 or higher required)\n");
			printf("> CUDA kernel runs will be serialized\n");
		}
		else
		{
			printf("> GPU does not support HyperQ\n");
			printf("> CUDA kernel runs will have limited concurrency\n");
		}
	}

	printf("> Compute Capability %d.%d hardware with %d multi-processors\n", deviceProp.major, deviceProp.minor, deviceProp.multiProcessorCount);

	// Allocate and initialize an array of stream handles
	cudaStream_t *streams = (cudaStream_t *)malloc(n_streams * sizeof(cudaStream_t));

	for (int i = 0; i < n_streams; i++)
	{
		cudaStreamCreate(&(streams[i]));
	}

	// set up execution configuration
	dim3 block(iblock);
	dim3 grid(isize / iblock);
	printf("> grid %d block %d\n", grid.x, block.x);

	// creat events
	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);

	// record start event
	cudaEventRecord(start, 0);

	// dispatch job with depth first ordering
	omp_set_num_threads(n_streams);
	for (int i = 0; i < n_streams; ++i)
#pragma omp parallel
	{
		kernel_1 <<<grid, block, 0, streams[i] >>>();
		kernel_2 <<<grid, block, 0, streams[i] >>>();
		kernel_3 <<<grid, block, 0, streams[i] >>>();
		kernel_4 <<<grid, block, 0, streams[i] >>>();
	}

	// record stop event
	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);

	// calculate elapsed time
	cudaEventElapsedTime(&elapsed_time, start, stop);
	printf("Measured time for parallel execution = %.3fs\n", elapsed_time / 1000.0f);

	// release all stream
	for (int i = 0; i < n_streams; i++)
	{
		cudaStreamDestroy(streams[i]);
	}

	free(streams);

	// destroy events
	cudaEventDestroy(start);
	cudaEventDestroy(stop);

	// reset device
	cudaDeviceReset();
	system("Pause");
	return 0;
}
