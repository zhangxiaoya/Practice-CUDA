# CUDA 流与事件、并行

[TOC]

## 学习目标
之前接触的CUDA变成都是进行kernel级别的并行，一个Kernel执行一个简单的任务，每一个kernel都分布在不同的线程在不同的core里面运行，调整kernel运行的效率分别可以从下面的几个角度做修改：
- 编程模型
- 执行模型
- 存储器的管理和分配

接触到流以后，并行能力不再是仅仅在kernel级别，有了Grid级别。所谓的Grid的级别并行，是指在不同的kernel运行在不同的stream中，每个stream相当于更高级的“进程”，每个这样的“进程”又包含多个"线程"，每个设备可以运行多个stream，也就是多个kernel并行在一个设备上运行。


## Stream和Event的简介

一个stream就是一个异步操作的序列，这些操作包括数据的传输和kernel的启动,等等。多个stream流之间同步控制是需要认为的控制，但是每个流之间没有交互，通过运行多个流，实现Grid 级别的并行。

使用Stream编程，类似于在CPU上使用多线程编程，将任务分解为可以overlap的子任务，每个线程处理其中的一个子任务，利用overlap的时间降低总体时间，提高效率。

### CUDA Stream
所有的CUDA操作显示或者隐式的运行在stream中，在之前的例子中，没有显示的使用stream，kernel都是运行在隐式的流中，因此有两种流：
- 隐式的创建流（NULL Stream）
- 显示的创建流（non-NULL Stream）

要是需要控制多个kernel并行执行，overlap不同的CUDA操作，就必须显示的创建stream。一个粗粒度的并行如下所示：
- host计算和device计算的overlap
- host计算和host-device数据传输overlap
- host-device数据传输overlap和device计算之间的overlap
- device 计算之间的overlap

一个典型的overlap：
```
cudaMemcpy(..., cudaMemcpyHostToDevice);
kernel<<<grid,block>>>(...);
cudaMemcpy(...,cudaMemcpyDeviceToHost);
```

- 第一行，将数据从host转移到device，这个过程是同步的，host阻塞等待数据传输完成；
- 第二行，触发一个kernel，默认kernel触发是异步的， 触发kernel后，host进程立刻执行下面的操作；
- 第三行，数据传输，从device到host，也就是触发了kernel后，不管kernel有没有执行结束，都将数据从device转移到host

#### 异步数据传输
上面的代码的问题是，数据传输是同步的，触发一个kernel是异步的，kernel可能还没有计算完成，就把没有完成的结果传输到host。
数据传输任务也可以是异步的，比如使用下面这个API：
```
cudaError_t cudaMemcpyAsync(void* dst, const void* src, size_t count, cudaMemcpyKind kind, cudaStream_t stream=0);
```
stream作为该API的第5个参数，如果没有提供显示的stream，那么就是使用默认的stream。

#### 创建一个流
创建一个显示的stream如下所示：
```
cudaError_t cudaStreamCreate(cudaStream_t* pStream);
```

#### 分配pinned的内存
使用异步传输数据的时候，必须使用pinned 内存，因为这些内存是物理内存上直接分配的，不是虚拟内存，防止异步传输数据与操作系统的页面置换造成的冲突，引起资源竞争问题。
```
cudaError_t cudaMallocHost(void** ptr, size_t size);
cudaError_t cudaHostAlloc(void** pHost, size_t size, unsigned int flags);
```

#### 在流中触发一个kernel
如何在流中触发一个kernel？触发kernel的函数还有这样一个”重载“：
```
kernel_name<<<grid, block, shareMemSize, stream>>>(argument list);
```
其中，stream作为第四个参数。

#### 销毁一个Stream

```
cudaError_t cudaStreamDestroy(cudaStream_t stream);
```
如果，当前的stream中仍然有任务没有结束，函数立刻返回，流所占用的资源，在所有任务都结束后自动释放。

#### 查询操作
由于流中所有的操作都是异步的，CUDA提供两个API，用来查询是否所有的任务都完成了。

```
cudaError_t cudaStreamSynchronize(cudaStream_t stream);
cudaError_t cudaStreamQuery(cudaStream_t stream);
```
其中第一个API，强制阻塞host进程，直到在stream中的所有任务都结束了；第二个不会阻塞host进程，但是如果所有的任务都结束了，那么返回cudaSuccess，否则返回cudaErrorNotReady。

#### 一个简单的流处理的例子
```
for(int i = 0; i < nStream; ++i)
{
    int offset = i * bytePerStream;
    cudaMemcpyAsync(&d_a[offset], &a[offset], bytePerStream, streams[i]);
    kernel<<<grid, block, 0, streams[i]>>>(&d_a[offset]);
    cudaMemcpyAsync(&a[offset], &d_a[offset], bytePerStream, streams[i])
}
for(int i = 0; i < nStreams; ++i)
{
    cudaStreamSynchronize(streams[i]);
}
```
由于PCIe是共享资源，数据传输没有重叠，只有不同的流并且数据流方向不同的时候才能两个流并行（全双工pcie）。

stream并行的程度是跟硬件相关的.

### Stream日程表

Stream日程表是管理Stream运行的机制，虽然有了Stream，但是也不能理想化的并行执行所有的stream，这跟硬件有很大的关系。

#### False-Dependencies

只有一个pipeline的设备可能会有这样的问题，存在一个任务必须等待其他的任务完成才能运行，这样不同stream的任务也必须等待。

#### Hyper Q
超级队列是用来降低false dependencies的，用多个队列真正允许多个stream并行执行。但是Hyper Q也有硬件的限制，超过队列个数的stream也会出现false dependencies的问题。

### Stream属性
类似与线程属性，给每个stream设定一个优先级，但是对于数据传输操作没有影响。
```
cudaError_t cudaStreamCreateWithPriority(cudaStream_t* pStream, unsigned int flags, int priority);
```
优先级是有上限和下限的，如果超过限制，会自定设定为最高值或者最低值。也可以写一段代码，获取当前设备支持的stream优先级范围。
```
cudaError_t cudaDeviceGetStreamPriorityRange(int* leastPriority, int* greatestPriority);
```

### CUDA Event
event是一个与一个stream关联，是一些操作流的一个点，类似于一个标签，event的作用有两个：
- 同步stream的执行
- 监控设备的进程

Event可以被创建，可以被删除，只有到达该点的所有操作都结束，Event才被记录。

#### Event的创建和销毁

1. 创建
```
cudaEvent_t event;

cudaError_t cudaEventCreate(cudaEvent_t* event);
```

2. 销毁
```
cudaError_t cudaEventDestroy(cudaEvent_t event);
```
如果没有满足条件的Event被删除，函数立即返回，直到Event满足条件，然后自动释放所占用的资源。

#### 记录Event

记录Event相当于在stream中插入一个操作，这个操作只会向host返回一个响应，对应的操作如下：
```
cudaError_t cudaEventRecord(cudaEvent_t event, cudaStream_t stream=0);
```
同步Event，类似于同步stream，不同的是，这个API只是强制阻塞等到当前Event的执行，并不阻塞整个stream：
```
cudaError_t cudaEventSynchronize(cudaEvent_t envet);
```

也可以查询Event的完成状态，与Stream的查询类似：
```
cudaError_t cudaEventQuery(cudaEvent_t event);
```

查询两个事件的间隔时间：
```
cudaError_t cudaEventElapsedTime(float* ms, cudaEvent_t start, cudaEvent_t stop);
```
需要注意的是，这两个事件并不一定需要在一个stream中，但是，两个事件都在非NULL的stream中，那么整个操作返回的时间要比期待的长。

#### 一个完整的事件测量事件的过程
```
// 在默认stream上创建两个事件
cudaEvent_t start;
cudaEvent_t stop;
cudaCreateEvent(&start);
cudaCreateEvent(&stop);

// 记录第一个开始事件
cudaEventRecord(&start);
// 触发核函数
kernel<<<grid, block>>>(arguments);

// 记录结束事件
cudaEventRecord(&stop);
// 强制等待结束事件完成
cudaEventSynchronize(stop);

// 计算事件间隔时间
float time;
cudaEventElapsedTime(&time, start, stop);

// 销毁事件
cudaEventDestroy(start);
cudaEventDestory(stop);
```

### 流同步
所有在非空stream中的操作相对于host线程都是非阻塞的，在一些场景中需要同步host进程与一个流中操作。从内存的角度考虑，CUDA操作可以分为两类：
- 内存相关的
- 计算相关的

计算相关操作都是异步的，许多内存操作本质上是同步的，但是CUDA的运行时对这些内存操作提供了异步操作功能。

有两种类型的stream，从同步和异步的角度划分：
- 异步stream（非空的stream）
- 同步stream（默认的stream）

对于非空的steam相对于host进程都是异步的，对于默认stream，是同步stream，其中的大部分操作都会引host进程阻塞。


非空stream有可以分为两个类别：
- 阻塞stream
- 非阻塞stream

对于阻塞的非空stream，NULL stream可以阻塞其中的操作，杜宇非阻塞的非空stream，NULL Stream不能阻塞其中的操作。

#### 阻塞和非阻塞stream
使用cudaStreamCreate创建的stream是阻塞stream，其中所有的操作必须等待比它早创建的NULL Stream中的操作都完成，它才能执行。NULL stream是默认的stream，与其他的阻塞stream是保持同步的，即，触发一个在默认stream中的操作，如果前面有阻塞stream的操作正在执行，那么它将暂停，后续如果创建阻塞stream的操作，这个阻塞stream中的操作也会等待。

```
kernel_1<<<1, 1, 0, stream_1>>>();
kernel_2<<<1, 1>>>();
kernel_3<<<1, 1, 0, stream_2>>>();
```
上面的这三个kernel，一次是顺序执行的。

创建一个非阻塞的stream使用下面的API：
```
cudaError_t cudaStreamCreateWithFlags(cudaStream_t* pStream, unsigned int flags);
```
其中flag有两个选项：
```
cudaStreamDefault: 阻塞stream
cudaStreamNonBlocking: 非阻塞stream
```

#### 隐式同步
在CUDA中有两种host-device之间的同步方法：显式同步和隐式同步：
其中显式同步已经接触了很多：
- cudaDeviceSynchronize
- cudaStreamSynchronize
- cudaEventSynchronize

隐式的同步，比如:
- cudaMemcpy

#### 显式同步
- 同步device
- 同步stream
- 同步Event
- 使用Event同步不同的stream

#### 配置Event

使用下面的API配置Event的行为
```
cudaError_t cudaEventCreateWithFlags(cudaEvent_t* event, unsigned int flags);
```
其中flag可以是下面的参数：
- cudaEventDefault;
- cudaEventBlockingSync;
- cudaEventDisableTiming;
- cudaEventInterprocess;

## 核函数的并行执行
前面的内容介绍了stream和Event的基本概念，以及并发执行的基本概念，这节内容介绍如何并发执行kernel：
- 使用深度优先和广度优先的方法分配job
- 调整硬件队列
- 避免false-dependence
- 检查默认stream的阻塞行为
- 在不同的非默认stream之间添加依赖
- 实验不同的资源使用对并发的影响

### 在非空stream上并发执行kernel

一个简单的实验，创建4个kernel函数，每个kernel进行大量的数据计算，创建4个stream，每个stream中多进行着4个kernel操作，一个最简单的例子如下：
```
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <cmath>
#include <cstdio>

#define N 3000
#define NSTREAM 4

__global__ void kernel_1()
{
	auto sum = 0.0;

	for (auto i = 0; i < N; i++)
	{
		sum = sum + tan(0.1) * tan(0.1);
	}
}

__global__ void kernel_2()
{
	auto sum = 0.0;

	for (auto i = 0; i < N; i++)
	{
		sum = sum + tan(0.1) * tan(0.1);
	}
}

__global__ void kernel_3()
{
	auto sum = 0.0;

	for (auto i = 0; i < N; i++)
	{
		sum = sum + tan(0.1) * tan(0.1);
	}
}

__global__ void kernel_4()
{
	auto sum = 0.0;

	for (auto i = 0; i < N; i++)
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

	auto dev = 0;
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
	cudaStream_t *streams = static_cast<cudaStream_t *>(malloc(n_streams * sizeof(cudaStream_t)));

	for (auto i = 0; i < n_streams; i++)
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
	for (int i = 0; i < n_streams; i++)
	{
		kernel_1<<<grid, block, 0, streams[i] >>>();
		kernel_2<<<grid, block, 0, streams[i] >>>();
		kernel_3<<<grid, block, 0, streams[i] >>>();
		kernel_4<<<grid, block, 0, streams[i] >>>();
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
```

### 广度优先分配任务
上面这是一个“深度优先”分配task的例子，即，一次性触发一个stream中的所有task kernel函数。下面是“广度优先”分配的例子，每次触发一个stream的一个task。
```
	for (int i = 0; i < n_streams; i++)
	{
		kernel_1 <<<grid, block, 0, streams[i] >>>();
	}
	for (int i = 0; i < n_streams; i++)
	{
		kernel_2 <<<grid, block, 0, streams[i] >>>();
	}
	for (int i = 0; i < n_streams; i++)
	{
		kernel_3 <<<grid, block, 0, streams[i] >>>();
	}
	for (int i = 0; i < n_streams; i++)
	{
		kernel_4 <<<grid, block, 0, streams[i] >>>();
	}
```

### 使用OpenMP分配任务
```
	omp_set_num_threads(n_streams);
	for (int i = 0; i < n_streams; ++i)
#pragma omp parallel
	{
		kernel_1 <<<grid, block, 0, streams[i] >>>();
		kernel_2 <<<grid, block, 0, streams[i] >>>();
		kernel_3 <<<grid, block, 0, streams[i] >>>();
		kernel_4 <<<grid, block, 0, streams[i] >>>();
	}
```
这段代码是使用的openmp，需要在工程中打开openmp的开关，然后在当前文件中包含<omp.h>文件。

### 利用默认stream的任务阻塞其他任务的特性
默认stream中的task会阻塞非默认stream的task，并且，如果有非默认stream的task在运行，也会阻塞自己。
```
for (int i = 0; i < n_streams; i++)
	{
		kernel_1 <<<grid, block, 0, streams[i] >>>();
		kernel_2 <<<grid, block, 0, streams[i] >>>();
		kernel_3 <<<grid, block>>>();
		kernel_4 <<<grid, block, 0, streams[i] >>>();
	}
```
kernel_3会在前两个kernel都运行结束后才运行，kernel_4会在kernel_3运行结束后才能运行。

### 添加内部依赖
使用事件来同步stream之间的操作，比如只有一个stream结束后，才能开始下一个stream。
```
for (int i = 0; i < n_streams; i++)
	{
		kernel_1 <<<grid, block, 0, streams[i] >>>();
		kernel_2 <<<grid, block, 0, streams[i] >>>();
		kernel_3 <<<grid, block, 0, streams[i] >>>();
		kernel_4 <<<grid, block, 0, streams[i] >>>();

		cudaEventRecord(events[i], streams[i]);
		cudaStreamWaitEvent(streams[i], events[i], 0);
	}
```
先创建于stream一样多的event，然后用用它们来同步stream之间的顺序执行。

## kernel 与 数据传输的overlap

有两种方式，一种是类似于“深度优先的方式”，另外一种类似 与广度优先的方式。

### 深度优先
将整个任务分成4个子任务，然后在每个stream中一次性触发数据传输和kernel计算，这样在不同stream之间会产生overlap。
```
#include <stdio.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cmath>
#include <cstring>

#define NSTREAM 4
#define BDIM 128

void initialData(float *ip, int size)
{
	for (int i = 0; i < size; i++)
	{
		ip[i] = (float)(rand() & 0xFF) / 10.0f;
	}
}

void sumArraysOnHost(float *A, float *B, float *C, const int N)
{
	for (int idx = 0; idx < N; idx++)
		C[idx] = A[idx] + B[idx];
}

__global__ void sumArrays(float *A, float *B, float *C, const int N)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;

	if (idx < N)
	{
		for (int i = 0; i < N; ++i)
		{
			C[idx] = A[idx] + B[idx];
		}
	}
}

void checkResult(float *hostRef, float *gpuRef, const int N)
{
	double epsilon = 1.0E-8;
	bool match = true;

	for (int i = 0; i < N; i++)
	{
		if (abs(hostRef[i] - gpuRef[i]) > epsilon)
		{
			match = false;
			printf("Arrays do not match!\n");
			printf("host %5.2f gpu %5.2f at %d\n", hostRef[i], gpuRef[i], i);
			break;
		}
	}

	if (match)
		printf("Arrays match.\n\n");
}

int main(int argc, char **argv)
{
	printf("> %s Starting...\n", argv[0]);

	int dev = 0;
	cudaDeviceProp deviceProp;
	cudaGetDeviceProperties(&deviceProp, dev);
	printf("> Using Device %d: %s\n", dev, deviceProp.name);
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

	// set up data size of vectors
	int nElem = 1 << 16;
	printf("> vector size = %d\n", nElem);
	size_t nBytes = nElem * sizeof(float);

	// malloc pinned host memory for async memcpy
	float *h_A, *h_B, *hostRef, *gpuRef;
	cudaHostAlloc((void**)&h_A, nBytes, cudaHostAllocDefault);
	cudaHostAlloc((void**)&h_B, nBytes, cudaHostAllocDefault);
	cudaHostAlloc((void**)&gpuRef, nBytes, cudaHostAllocDefault);
	cudaHostAlloc((void**)&hostRef, nBytes, cudaHostAllocDefault);

	// initialize data at host side
	initialData(h_A, nElem);
	initialData(h_B, nElem);
	memset(hostRef, 0, nBytes);
	memset(gpuRef, 0, nBytes);

	// add vector at host side for result checks
	sumArraysOnHost(h_A, h_B, hostRef, nElem);

	// malloc device global memory
	float *d_A, *d_B, *d_C;
	cudaMalloc((float**)&d_A, nBytes);
	cudaMalloc((float**)&d_B, nBytes);
	cudaMalloc((float**)&d_C, nBytes);

	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);

	// invoke kernel at host side
	dim3 block(BDIM);
	dim3 grid((nElem + block.x - 1) / block.x);
	printf("> grid (%d, %d) block (%d, %d)\n", grid.x, grid.y, block.x,block.y);

	// sequential operation
	cudaEventRecord(start, 0);
	cudaMemcpy(d_A, h_A, nBytes, cudaMemcpyHostToDevice);
	cudaMemcpy(d_B, h_B, nBytes, cudaMemcpyHostToDevice);
	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	float memcpy_h2d_time;
	cudaEventElapsedTime(&memcpy_h2d_time, start, stop);

	cudaEventRecord(start, 0);
	sumArrays <<<grid, block >>>(d_A, d_B, d_C, nElem);
	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);

	float kernel_time;
	cudaEventElapsedTime(&kernel_time, start, stop);
	cudaEventRecord(start, 0);
	cudaMemcpy(gpuRef, d_C, nBytes, cudaMemcpyDeviceToHost);
	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	float memcpy_d2h_time;
	cudaEventElapsedTime(&memcpy_d2h_time, start, stop);
	float itotal = kernel_time + memcpy_h2d_time + memcpy_d2h_time;

	printf("\n");
	printf("Measured timings (throughput):\n");
	printf(" Memcpy host to device\t: %f ms (%f GB/s)\n", memcpy_h2d_time, (nBytes * 1e-6) / memcpy_h2d_time);
	printf(" Memcpy device to host\t: %f ms (%f GB/s)\n", memcpy_d2h_time, (nBytes * 1e-6) / memcpy_d2h_time);
	printf(" Kernel\t\t\t: %f ms (%f GB/s)\n", kernel_time, (nBytes * 2e-6) / kernel_time);
	printf(" Total\t\t\t: %f ms (%f GB/s)\n", itotal, (nBytes * 2e-6) / itotal);

	// grid parallel operation
	int iElem = nElem / NSTREAM;
	size_t iBytes = iElem * sizeof(float);
	grid.x = (iElem + block.x - 1) / block.x;

	cudaStream_t stream[NSTREAM];

	for (int i = 0; i < NSTREAM; ++i)
	{
		cudaStreamCreate(&stream[i]);
	}

	cudaEventRecord(start, 0);

	// initiate all work on the device asynchronously in depth-first order
	for (int i = 0; i < NSTREAM; ++i)
	{
		int ioffset = i * iElem;
		cudaMemcpyAsync(&d_A[ioffset], &h_A[ioffset], iBytes, cudaMemcpyHostToDevice, stream[i]);
		cudaMemcpyAsync(&d_B[ioffset], &h_B[ioffset], iBytes, cudaMemcpyHostToDevice, stream[i]);
		sumArrays <<<grid, block, 0, stream[i] >>>(&d_A[ioffset], &d_B[ioffset], &d_C[ioffset], iElem);
		cudaMemcpyAsync(&gpuRef[ioffset], &d_C[ioffset], iBytes, cudaMemcpyDeviceToHost, stream[i]);
	}

	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	float execution_time;
	cudaEventElapsedTime(&execution_time, start, stop);

	printf("\n");
	printf("Actual results from overlapped data transfers:\n");
	printf(" overlap with %d streams : %f ms (%f GB/s)\n", NSTREAM, execution_time, (nBytes * 2e-6) / execution_time);
	printf(" speedup                : %f \n", ((itotal - execution_time) * 100.0f) / itotal);

	// check kernel error
	cudaGetLastError();

	// check device results
	checkResult(hostRef, gpuRef, nElem);

	// free device global memory
	cudaFree(d_A);
	cudaFree(d_B);
	cudaFree(d_C);

	// free host memory
	cudaFreeHost(h_A);
	cudaFreeHost(h_B);
	cudaFreeHost(hostRef);
	cudaFreeHost(gpuRef);

	// destroy events
	cudaEventDestroy(start);
	cudaEventDestroy(stop);

	// destroy streams
	for (int i = 0; i < NSTREAM; ++i)
	{
		cudaStreamDestroy(stream[i]);
	}

	cudaDeviceReset();

	system("Pause");
	return(0);
}
```

### 广度优先
还是把整个任务分割成若干子任务，每个stream先进行数据传输，然后在依次进行计算，最后结果回传：
```
for (int i = 0; i < NSTREAM; ++i)
	{
		int ioffset = i * iElem;
		cudaMemcpyAsync(&d_A[ioffset], &h_A[ioffset], iBytes, cudaMemcpyHostToDevice, stream[i]);
		cudaMemcpyAsync(&d_B[ioffset], &h_B[ioffset], iBytes, cudaMemcpyHostToDevice, stream[i]);
	}
	for (int i = 0; i < NSTREAM; ++i)
	{
		int ioffset = i * iElem;
		sumArrays <<<grid, block, 0, stream[i] >>>(&d_A[ioffset], &d_B[ioffset], &d_C[ioffset], iElem);
	}
	for (int i = 0; i < NSTREAM; ++i)
	{
		int ioffset = i * iElem;
		cudaMemcpyAsync(&gpuRef[ioffset], &d_C[ioffset], iBytes, cudaMemcpyDeviceToHost, stream[i]);
	}
```

## Stream 回调函数

就是在stream调用，在host执行的函数。

```
cudaError_t cudaStreamAddCallback(cudaStream_t stream,cudaStreamCallback_t callback, void *userData, unsigned int flags);
```