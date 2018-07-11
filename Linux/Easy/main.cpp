#include <iostream>
#include "utils.h"
#include <iostream>
#include "squareSum.h"
#include "vectorAdd.cuh"

using namespace std;

int main()
{
    if(InitCUDA())
    {
        cout << "Found CUDA" << endl;
        PrintGPUProps();
        squareSum();
        VectorAddOnDevice(1024);
    }
    return 0;
}