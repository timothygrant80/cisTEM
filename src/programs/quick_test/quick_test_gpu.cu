#include <cooperative_groups.h>
namespace cg = cooperative_groups;

#include "../../gpu/gpu_core_headers.h"
#include "../../gpu/gpu_indexing_functions.h"

#include "quick_test_gpu.h"

// __global__ void helloWorldKernel( ) {
//     printf("Hello from GPU\n");
// }

// #define FULLMASK 0xffffffff
// constexpr unsigned int nffts          = 4;
// constexpr unsigned int nWarps         = 1;
// constexpr unsigned int active_threads = 16 / nffts;

// __device__ __forceinline__ unsigned int get_lane_id( ) {
//     unsigned ret;
//     asm volatile("mov.u32 %0, %laneid;"
//                  : "=r"(ret));
//     return ret;
// }

// __device__ __forceinline__ unsigned int get_warp_id( ) {
//     unsigned ret;
//     asm volatile("mov.u32 %0, %warpid;"
//                  : "=r"(ret));
//     return ret;
// }

__global__ void testWarpShuffle(const int printidx) {
    // if ( threadIdx.x == 0 )
    //     printf("Hello from GPU\n");

    // float         c{256.0};
    // __half        h{256.0};
    __nv_bfloat16 b{256.0};
    printf("%3.12f\n", __bfloat162float(b));
    b++;
    printf("%3.12f\n", __bfloat162float(b));

    // printf("fl: %3.12f bf %3.12f hf %3.12f\n", c, __bfloat162float(b), __half2float(h));

    // c += 1;
    // b += 1;
    // h += 1;
}

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
