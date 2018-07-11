#include <iostream>
#include "utils.h"
#include <iostream>
#include "squareSum.h"
#include "vectorAdd.cuh"

#include "matrixAdd.cuh"

using namespace std;

int main()
{
    if(InitCUDA())
    {
        cout << "Found CUDA" << endl;
        PrintGPUProps();
        squareSum();
        VectorAddOnDevice(1024);
        MatrixAddOnDevice(100,100);
    }
    return 0;
}