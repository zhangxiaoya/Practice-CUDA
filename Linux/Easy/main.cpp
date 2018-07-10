#include <iostream>
#include "utils.h"
#include <iostream>

using namespace std;

int main()
{
    if(InitCUDA())
        cout << "Found CUDA" << endl;
    return 0;
}