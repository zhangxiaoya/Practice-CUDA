#include <stdio.h>
#include <string.h>
#include <time.h>

void sumArraysOnHost(float* A, float* B, float* C, int N)
{
    for(int i=0;i<N;++i)
    {
        C[i] = A[i] + B[i];
    }
}

void initialData(float* ip, int size)
{
    time_t t;
    srand((unsigned int) time(&t));
    for(int i=0;i<size;++i)
    {
        ip[i] = (float)(rand() & 0xFF) / 10.0f;
    }
}

int main(void)
{
    int nElem = 1024;
    size_t nBytes = nElem * sizeof(float);

    float *A, *B, *C;
    A = (float*)malloc(nBytes);
    B = (float*)malloc(nBytes);
    C = (float*)malloc(nBytes);

    initialData(A, nElem);
    initialData(B, nElem);

    sumArraysOnHost(A,B,C,nElem);

    for(int i = 0; i < nElem; ++i)
    {
        printf("%f ", C[i]);
    }

    free(A);
    free(B);
    free(C);

    return 0;
}