#include <stdio.h>
#include <cuda_runtime.h>

int main(int argc, char const *argv[])
{
    int nElem = 1024;

    // first init
    dim3 block(1024);
    dim3 grid((nElem  + block.x - 1) / block.x);
    printf("grid.x : %d, block.x : %d\n", grid.x, block.x);

    // reset fist time
    block = 512;
    grid= (nElem + block.x - 1) / block.x;
    printf("grid.x : %d, block.x : %d\n", grid.x, block.x);

    // reset second time
    block = 256;
    grid = (nElem + block.x - 1) / block.x;
    printf("grid.x : %d, block.x : %d\n", grid.x, block.x);

    // reset third time
    block = 256;
    grid = (nElem + block.x - 1) / block.x;
    printf("grid.x : %d, block.x : %d\n", grid.x, block.x);

    // reset fourth time
    block = 128;
    grid = (nElem + block.x - 1) / block.x;
    printf("grid.x : %d, block.x : %d\n", grid.x, block.x);

    // reset fivth time
    block = 64;
    grid = (nElem + block.x - 1) / block.x;
    printf("grid.x : %d, block.x : %d\n", grid.x, block.x);

    // reset sixth time
    block = 32;
    grid =(nElem + block.x - 1) / block.x;
    printf("grid.x : %d, block.x : %d\n", grid.x, block.x);

    // reset seventh time
    block = 16;
    grid = (nElem + block.x - 1) / block.x;
    printf("grid.x : %d, block.x : %d\n", grid.x, block.x);

    return 0;
}