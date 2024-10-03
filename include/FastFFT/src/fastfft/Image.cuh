// Collection of helper functions for test.cu

#ifndef SRC_CPP_IMAGE_CUH_
#define SRC_CPP_IMAGE_CUH_

#include <iostream>
#include <vector>
#include <chrono>
#include <cmath>
#include <string>

// sudo apt-get install libfftw3-dev libfftw3-doc
#include <fftw3.h>

#include <cuda_runtime_api.h>
#include "../../include/cufftdx/include/cufftdx.hpp"
#include <cufft.h>
#include <cufftXt.h>

#include "../../include/ieee-754-half/half.hpp"

// A simple class to represent image objects needed for testing FastFFT.

template <class wanted_real_type, class wanted_complex_type>
class Image {

  public:
    Image( );
    Image(short4 wanted_size);
    ~Image( );

    wanted_real_type*    real_values;
    wanted_complex_type* complex_values;
    bool*                clipIntoMask;

    short4 size;
    int    real_memory_allocated;
    size_t n_bytes_allocated;
    int    padding_jump_value;

    float fftw_epsilon;

    bool is_in_memory;
    bool is_fftw_planned;
    bool is_in_real_space;
    bool is_cufft_planned;

    void Allocate( );
    void Allocate(bool plan_fftw);
    void FwdFFT( );
    void InvFFT( );

    // Make FFTW plans for comparing CPU to GPU xforms.
    // This is nearly verbatim from cisTEM::Image::Allocate - I do not know if FFTW_ESTIMATE is the best option.
    // In cisTEM we almost always use MKL, so this might be worth testing. I always used exhaustive in Matlab/emClarity.
    fftwf_plan plan_fwd = NULL;
    fftwf_plan plan_bwd = NULL;

    cufftHandle cuda_plan_forward;
    cufftHandle cuda_plan_inverse;
    size_t      cuda_plan_worksize_forward;
    size_t      cuda_plan_worksize_inverse;

    cudaEvent_t startEvent{nullptr};
    cudaEvent_t stopEvent{nullptr};
    float       elapsed_gpu_ms{ };

    inline void create_timing_events( ) {
        cudaEventCreate(&startEvent, cudaEventBlockingSync);
        cudaEventCreate(&stopEvent, cudaEventBlockingSync);
    }

    inline void record_start( ) { cudaEventRecord(startEvent); }

    inline void record_stop( ) { cudaEventRecord(stopEvent); }

    inline void synchronize( ) { cudaEventSynchronize(stopEvent); }

    inline void print_time(std::string msg, bool print_out = true) {
        cudaEventElapsedTime(&elapsed_gpu_ms, startEvent, stopEvent);
        if ( print_out ) {
            std::cout << "Time on " << msg << " " << elapsed_gpu_ms << " ms" << std::endl;
        }
    }

    void MakeCufftPlan( );
    void MakeCufftPlan3d( );

    void SetClipIntoMask(short4 input_size, short4 output_size);
    bool is_set_clip_into_mask = false;
    // void SetClipIntoCallback(cufftReal* image_to_insert, int image_to_insert_size_x, int image_to_insert_size_y,int image_to_insert_pitch);
    void SetComplexConjMultiplyAndLoadCallBack(cufftComplex* search_image_FT, cufftReal FT_normalization_factor);
    void MultiplyConjugateImage(wanted_complex_type* other_image);
    void print_values_complex(float* input, std::string msg, int n_to_print);
    // float ReturnSumOfReal(float* input, short4 size, bool print_val = false);
    template <typename T>
    float ReturnSumOfReal(T* input, short4 size, bool print_val = false);

    float2 ReturnSumOfComplex(float2* input, int n_to_print);
    float  ReturnSumOfComplexAmplitudes(float2* input, int n_to_print);
    void   ClipInto(const float* array_to_paste, float* array_to_paste_into, short4 size_from, short4 size_into, short4 wanted_center, float wanted_padding_value);

    bool data_is_fp16;
    void ConvertFP32ToFP16( );
    void ConvertFP16ToFP32( );

  private:
    // Note; this is not thread safe
    bool is_registered;

    void RegisterPageLockedMemory( );
    void UnRegisterPageLockedMemory( );
};

#endif // SRC_CPP_IMAGE_CUH_