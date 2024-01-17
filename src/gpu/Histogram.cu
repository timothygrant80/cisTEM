/*
 * Histogram.cu
 *
 *  Created on: Aug 29, 2019
 *      Author: himesb
 */

#include "gpu_core_headers.h"
#include "gpu_indexing_functions.h"

#include "GpuImage.h"
#include "Histogram.h"
#include "../constants/constants.h"

constexpr int y_grid_divisor = 32;

__global__ void
histogram_smem_atomics(const __half* __restrict__ in, int4 dims, float* out, const __half bin_min, const __half bin_inc, const int max_padding);

__global__ void histogram_smem_atomics(const __half* __restrict__ in, int4 dims, float* out, const __half bin_min, const __half bin_inc, const int max_padding) {
    // pixel coordinates assuming a 2d image
    int x = physical_X( );
    int y = physical_Y( );

    //     threadsPerBlock_img = dim3(cistem::match_template::histogram_number_of_points, 1, 1);
    // gridDims_img        = dim3((input_image.dims.x + threadsPerBlock_img.x - 1) / threadsPerBlock_img.x,
    //                            (input_image.dims.y + (y_grid_divisor + threadsPerBlock_img.y) - 1) / (y_grid_divisor - 1 + threadsPerBlock_img.y), 1);

    // initialize temporary accumulation array in shared memory, this is equal to the number of bins in the histogram,
    // which may  be more or less than the number of threads in a block
    __shared__ int smem[cistem::match_template::histogram_number_of_points];

    // Each block has it's own copy of the histogram stored in global memory, found at the linear block index
    float* stored_array = &out[LinearBlockIdx_2dGrid( ) * cistem::match_template::histogram_number_of_points];

    // Since the number of x-threads is enforced to be equal to the number of bins, we can just copy the bins to shared memory
    // We could write if (threadIdx.x < cistem::match_template::histogram_number_of_points)
    for ( int i = threadIdx.x; i < cistem::match_template::histogram_number_of_points; i += BlockDimension_2d( ) )
        smem[i] = int(stored_array[i]);
    __syncthreads( );

    // __half       pixel_idx;
    int pixel_idx;
    // process pixels
    // updates our block's partial histogram in shared memory
    for ( int j = max_padding + y; j < dims.y - max_padding; j += blockDim.y * gridDim.y ) {
        for ( int i = max_padding + x; i < dims.x - max_padding; i += blockDim.x * gridDim.x ) {
            pixel_idx = __half2int_rd((in[j * dims.w + i] - bin_min) / bin_inc);
            if ( pixel_idx >= 0 && pixel_idx < cistem::match_template::histogram_number_of_points ) {
                atomicAdd(&smem[pixel_idx], 1);
            }
        }
    }
    __syncthreads( );

    // write partial histogram into the global memory
    // Converting to long was super slow. Given that I don't care about representing the number exactly,
    // but do care about overflow, just switch the bins to flaot
    for ( int i = threadIdx.x; i < cistem::match_template::histogram_number_of_points; i += blockDim.x * blockDim.y )
        stored_array[i] = float(smem[i]);
}

__global__ void histogram_final_accum(float* in, float* out, int n_bins, int n_blocks);

__global__ void histogram_final_accum(float* in, float* out, int n_bins, int n_blocks) {

    int lIDX = blockIdx.x * blockDim.x + threadIdx.x;

    if ( lIDX < n_bins ) {
        float total = 0.0f;
        for ( int j = 0; j < n_blocks; j++ ) {
            total += in[lIDX + n_bins * j];
        }
        out[lIDX] += total;
    }
}

Histogram::Histogram( ) {
    SetInitialValues( );
    wxPrintf("\n\tInit histogram\n");
}

Histogram::Histogram(int ignored_n_bins, float histogram_min, float histogram_step) {

    SetInitialValues( );
    Init(cistem::match_template::histogram_number_of_points, histogram_min, histogram_step);
}

Histogram::~Histogram( ) {

    if ( is_allocated_histogram ) {
        cudaErr(cudaFree(histogram));
        cudaErr(cudaFree(cummulative_histogram));
    }
}

//FIXME

void Histogram::Init(int ignored_n_bins, float histogram_min, float histogram_step) {

    this->histogram_min  = __float2half(histogram_min);
    this->histogram_step = __float2half(histogram_step);
    this->max_padding    = 2;
}

void Histogram::SetInitialValues( ) {
    is_allocated_histogram = false;
    histogram_min          = (__half)0.0;
    histogram_step         = (__half)0.0;
}

void Histogram::BufferInit(GpuImage& input_image) {

    // Set up grids for the kernels
    // To achieve best occupancy we can optimize the histogram to match the number of thread in each block
    static_assert(cistem::match_template::histogram_number_of_points <= 1024, "The histogram kernel assumes <= 1024 threads per block");
    static_assert(cistem::match_template::histogram_number_of_points % cistem::gpu::warp_size == 0, "The histogram kernel assumes a multiple of 32 threads per block");

    // Note: threads per block y,z are assume == 1 in the kernels
    // Note: a full histogram is assumed to fit on one block (cistem::match_template::histogram_number_of_points <= 1024
    constexpr int n_threads_in_y_or_z = 1;

    threadsPerBlock_img = dim3(cistem::match_template::histogram_number_of_points, n_threads_in_y_or_z, n_threads_in_y_or_z);

    gridDims_img = dim3((input_image.dims.x + threadsPerBlock_img.x - 1) / threadsPerBlock_img.x,
                        (input_image.dims.y + (y_grid_divisor + threadsPerBlock_img.y) - 1) / (y_grid_divisor - 1 + threadsPerBlock_img.y), 1);

    threadsPerBlock_accum_array = dim3(32, 1, 1);
    gridDims_accum_array        = dim3((cistem::match_template::histogram_number_of_points + threadsPerBlock_accum_array.x - 1) / threadsPerBlock_accum_array.x, 1, 1);

    // Every block will have a shared memory array of the size of the number of bins and aggregate those into their own
    // temp arrays. Only at the end of the search will these be added together
    size_of_temp_hist = (gridDims_img.x * gridDims_img.y * cistem::match_template::histogram_number_of_points * sizeof(float));

    // Array of temporary storage to accumulate the shared mem to
    cudaErr(cudaMalloc(&histogram, size_of_temp_hist));
    cudaErr(cudaMalloc(&cummulative_histogram, cistem::match_template::histogram_number_of_points * sizeof(float)));

    // could bring in the context and then put this to an async op
    cudaErr(cudaMemset(histogram, 0, size_of_temp_hist));
    cudaErr(cudaMemset(cummulative_histogram, 0, (cistem::match_template::histogram_number_of_points) * sizeof(float)));

    is_allocated_histogram = true;
}

void Histogram::AddToHistogram(GpuImage& input_image) {
    MyDebugAssertTrue(input_image.is_in_memory_gpu, "The image to add to the histogram is not in gpu memory.");

    precheck;
    histogram_smem_atomics<<<gridDims_img, threadsPerBlock_img, 0, cudaStreamPerThread>>>(
            (const __half*)input_image.real_values_16f, input_image.dims, histogram, histogram_min, histogram_step, max_padding);
    postcheck;
}

void Histogram::Accumulate(GpuImage& input_image) {
    cudaErr(cudaStreamSynchronize(cudaStreamPerThread));
    precheck;
    histogram_final_accum<<<gridDims_accum_array, threadsPerBlock_accum_array, 0, cudaStreamPerThread>>>(histogram, cummulative_histogram, cistem::match_template::histogram_number_of_points, gridDims_img.x * gridDims_img.y);
    postcheck;
}

void Histogram::CopyToHostAndAdd(long* array_to_add_to) {

    // Make a temporary copy of the cummulative histogram on the host and then add on the host. TODO errorchecking
    float* tmp_array;
    cudaErr(cudaMallocHost(&tmp_array, cistem::match_template::histogram_number_of_points * sizeof(float)));
    cudaErr(cudaMemcpy(tmp_array, this->cummulative_histogram, cistem::match_template::histogram_number_of_points * sizeof(float), cudaMemcpyDeviceToHost));

    for ( int iBin = 0; iBin < cistem::match_template::histogram_number_of_points; iBin++ ) {
        array_to_add_to[iBin] += (long)tmp_array[iBin];
    }

    cudaErr(cudaFreeHost(tmp_array));
}
