#ifndef ENABLEGPU
#error "GPU is not enabled"
#endif

#include "../../gpu/gpu_core_headers.h"
#include "../../gpu/GpuImage.h"
#include "batched_search.h"

BatchedSearch::BatchedSearch( ) {
    _is_initialized = false;
}

BatchedSearch::~BatchedSearch( ) {
    Deallocate( );
}

void BatchedSearch::Deallocate( ) {
    if ( _is_initialized ) {
        cudaErr(cudaFreeHost(_peak_buffer));
        cudaErr(cudaFree(_d_peak_buffer));
    }
}

void BatchedSearch::SetDeviceBuffer( ) {
    for ( int iBatch = 0; iBatch < _batch_size; iBatch++ ) {
        _peak_buffer[iBatch].value = float(iBatch);
    }
    cudaErr(cudaMemcpy(_d_peak_buffer, _peak_buffer, _batch_size * sizeof(IntegerPeak), cudaMemcpyHostToDevice));
}

void BatchedSearch::Init(GpuImage& reference_image, int wanted_number_search_images, int wanted_batch_size, bool test_mirror, int max_pix_x, int max_pix_y) {
    _n_search_images = wanted_number_search_images;
    _batch_size      = wanted_batch_size;
    _test_mirror     = test_mirror;
    _n_batches       = (_n_search_images + _batch_size - 1) / _batch_size;
    _n_in_last_batch = _n_search_images - ((_n_batches - 1) * _batch_size);
    _intra_loop_inc  = (_test_mirror) ? 2 : 1;

    _max_pixel_radius_x = (max_pix_x == 0) ? reference_image.dims.x / 2 : max_pix_x;
    _max_pixel_radius_y = (max_pix_y == 0) ? reference_image.dims.y / 2 : max_pix_y;

    _min_pixel_radius_x_y = 0;

    _stride = reference_image.dims.w * reference_image.dims.y;

    cudaErr(cudaMallocHost(&_peak_buffer, _batch_size * sizeof(IntegerPeak)));
    cudaErr(cudaMalloc(&_d_peak_buffer, _batch_size * sizeof(IntegerPeak)));
    _is_initialized = true;
}
