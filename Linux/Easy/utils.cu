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
}