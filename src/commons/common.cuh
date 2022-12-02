#ifndef COMMON_CUH
#define COMMON_CUH

#define WIN32_LEAN_AND_MEAN
#include <Windows.h>
#include <stdint.h>
#include <stdio.h>

double cpuMilliSeconds () {
    LARGE_INTEGER li;
    QueryPerformanceCounter(&li);
    return li.QuadPart / 10000.0;
} 

#define checkCudaError(val) check((val), __FILE__, __LINE__)
inline void check(cudaError_t err, const char *const file, int const line) {
  if (cudaSuccess != err) {
    fprintf(stderr, "CUDA error: %s %s %d \n", cudaGetErrorString(err), file, line);
    exit(-1);
  }
}

#endif