
#ifndef EASY_UTILS_H
#define EASY_UTILS_H

#include <stdio.h>

void PrintGPUProps();

extern "C"
{
    bool InitCUDA();
};

#endif //EASY_UTILS_H
