#include <cooperative_groups.h>
namespace cg = cooperative_groups;

#include "../../gpu/gpu_core_headers.h"
#include "../../gpu/gpu_indexing_functions.h"

#include "quick_test_gpu.h"

__device__ int sum(int* x, int n) {
    int sum = 0;
    for ( int i = 0; i < n; i++ ) {
        sum += x[i];
    }
    __syncthreads( );
    return sum;
}

__global__ void pk(float* x, int n) {

    float sum = 0;
    if ( threadIdx.x < blockDim.x / 2 ) {
        x[threadIdx.x] += x[threadIdx.x + blockDim.x / 2];
    }
}

void QuickTestGPU::callHelloFromGPU(int idx) {
}
