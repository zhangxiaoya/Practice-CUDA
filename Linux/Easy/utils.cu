#include "utils.h"
#include <cuda_runtime.h>
#include <iostream>
using std::cout;
using std::endl;

void printDeviceProp(const cudaDeviceProp& prop)
{
    cout << "Device Name : " << prop.name << endl;
    cout << "Total Global mem : " << prop.totalGlobalMem << endl;
    cout << "Shared Mem per block : " << prop.sharedMemPerBlock << endl;
    cout << "regsPerBlock : " << prop.regsPerBlock << endl;
    cout << "warpSize : " << prop.warpSize << endl;
    cout << "memPitch : " << prop.memPitch << endl;
    cout << "maxThreadsPerBlock" << prop.maxThreadsPerBlock << endl;
    cout << "maxThreadsDim[0-2]" << prop.maxThreadsDim[0] << " " << prop.maxThreadsDim[1] << " "<<prop.maxThreadsDim[2] <<endl;
    cout << "maxGridSize[0-2]" << prop.maxGridSize[0] << " "<< prop.maxGridSize[1] << " " << prop.maxGridSize[2]  << endl;
    cout << "totalConstMem " << prop.totalConstMem << endl;
    cout << "major.minor  " << prop.major << " " << prop.minor << endl;
    cout << "clockRate " << prop.clockRate << endl;
    cout << "textureAlignment " << prop.textureAlignment << endl;
    cout << "deviceoverlap" << prop.deviceOverlap << endl;
    cout << "multiProcessorCount " << prop.multiProcessorCount << endl;
}

void PrintGPUProps()
{
    cudaDeviceProp prop;
    if(cudaSuccess == cudaGetDeviceProperties(&prop, 0))
    {
        cout << "Device Name : " << prop.name << endl;
        cout << "Total Global mem : " << prop.totalGlobalMem << endl;
        cout << "Shared Mem per block : " << prop.sharedMemPerBlock << endl;
        cout << "regsPerBlock : " << prop.regsPerBlock << endl;
        cout << "warpSize : " << prop.warpSize << endl;
        cout << "memPitch : " << prop.memPitch << endl;
        cout << "maxThreadsPerBlock" << prop.maxThreadsPerBlock << endl;
        cout << "maxThreadsDim[0-2]" << prop.maxThreadsDim[0] << " " << prop.maxThreadsDim[1] << " "<<prop.maxThreadsDim[2] <<endl;
        cout << "maxGridSize[0-2]" << prop.maxGridSize[0] << " "<< prop.maxGridSize[1] << " " << prop.maxGridSize[2]  << endl;
        cout << "totalConstMem " << prop.totalConstMem << endl;
        cout << "major.minor  " << prop.major << " " << prop.minor << endl;
        cout << "clockRate " << prop.clockRate << endl;
        cout << "textureAlignment " << prop.textureAlignment << endl;
        cout << "deviceoverlap" << prop.deviceOverlap << endl;
        cout << "multiProcessorCount " << prop.multiProcessorCount << endl;
    }
    else
    {
        cout << "Get properties failed" << endl;
    }
}

bool InitCUDA()
{
    int count;
    cudaGetDeviceCount(&count);
    cout <<  count << endl;
    if(count ==0)
    {
        cout << "There is no device" << endl;
        return false;
    }
    cudaSetDevice(0);
    return true;
}