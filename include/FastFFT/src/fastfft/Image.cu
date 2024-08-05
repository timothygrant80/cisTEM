#include "Image.cuh"
#include "../../include/FastFFT.cuh"

template <class wanted_real_type, class wanted_complex_type>
Image<wanted_real_type, wanted_complex_type>::Image(short4 wanted_size) {

    size = wanted_size;

    if ( wanted_size.x % 2 == 0 )
        padding_jump_value = 2;
    else
        padding_jump_value = 1;

    size.w = (size.x + padding_jump_value) / 2;

    is_in_memory          = false;
    is_in_real_space      = true;
    is_cufft_planned      = false;
    is_fftw_planned       = false;
    data_is_fp16          = false;
    real_memory_allocated = size.w * size.y * size.z * 2;
    n_bytes_allocated     = real_memory_allocated * sizeof(wanted_real_type);
    is_registered         = false;
}

template <class wanted_real_type, class wanted_complex_type>
Image<wanted_real_type, wanted_complex_type>::~Image( ) {

    UnRegisterPageLockedMemory( );

    if ( is_in_memory ) {
        delete[] real_values;
        // fftw_free((wanted_real_type *)real_values);
        is_in_memory = false;
    }

    if ( is_fftw_planned ) {
        fftwf_destroy_plan(plan_fwd);
        fftwf_destroy_plan(plan_bwd);
        is_fftw_planned = false;
    }

    if ( is_cufft_planned ) {
        cudaErr(cufftDestroy(cuda_plan_inverse));
        cudaErr(cufftDestroy(cuda_plan_forward));
        is_cufft_planned = false;
    }

    if ( is_set_clip_into_mask ) {
        cudaErr(cudaFree(clipIntoMask));
        is_set_clip_into_mask = false;
    }
}

template <class wanted_real_type, class wanted_complex_type>
void Image<wanted_real_type, wanted_complex_type>::SetClipIntoMask(short4 input_size, short4 output_size) {
    // Allocate the mask
    int   pjv;
    int   address  = 0;
    int   n_values = output_size.w * 2 * output_size.y;
    bool* tmpMask  = new bool[n_values];

    precheck;
    cudaErr(cudaMalloc(&clipIntoMask, (n_values) * sizeof(bool)));
    postcheck;

    if ( output_size.x % 2 == 0 )
        pjv = 2;
    else
        pjv = 1;

    for ( int j = 0; j < output_size.y; j++ ) {
        for ( int i = 0; i < output_size.x; i++ ) {
            if ( i < input_size.x && j < input_size.y )
                tmpMask[address] = true;
            else
                tmpMask[address] = false;
            address++;
        }
        tmpMask[address] = false;
        address++;
        if ( pjv > 1 ) {
            tmpMask[address] = false;
            address++;
        }
    }

    cudaErr(cudaMemcpyAsync(clipIntoMask, tmpMask, n_values * sizeof(bool), cudaMemcpyHostToDevice, cudaStreamPerThread));
    cudaStreamSynchronize(cudaStreamPerThread);

    delete[] tmpMask;
    is_set_clip_into_mask = true;
}

// template < class wanted_real_type, class wanted_complex_type >
// Image<class wanted_real_type, class wanted_complex_type >::Image()
// {

// }

template <class wanted_real_type, class wanted_complex_type>
void Image<wanted_real_type, wanted_complex_type>::Allocate( ) {
    // This overload for complex inputs is probably a bad idea. Either figure out the FFTW for complex (should be easy) or a static check on bool set plan.
    // FIXME. Also not sure why I neet to pad the array by 2, I'm assuming it is some alignment issue?
    real_values = new wanted_real_type[real_memory_allocated + 2];

    // real_values = (wanted_real_type *) fftw_malloc(sizeof(wanted_real_type) * real_memory_allocated);
    complex_values  = (wanted_complex_type*)real_values; // Set the complex_values to point at the newly allocated real values;
    is_fftw_planned = false;
    is_in_memory    = true;

    RegisterPageLockedMemory( );
}

template <class wanted_real_type, class wanted_complex_type>
void Image<wanted_real_type, wanted_complex_type>::Allocate(bool set_fftw_plan) {
    real_values = new wanted_real_type[real_memory_allocated + 2];
    // real_values = (wanted_real_type *) fftw_malloc(sizeof(wanted_real_type) * real_memory_allocated);
    complex_values = (wanted_complex_type*)real_values; // Set the complex_values to point at the newly allocated real values;

    // This will only work for single precision, should probably add a check on this, but for now rely on the user to make sure they are using single precision.
    if ( set_fftw_plan ) {
        plan_fwd        = fftwf_plan_dft_r2c_3d(size.z, size.y, size.x, real_values, reinterpret_cast<fftwf_complex*>(complex_values), FFTW_ESTIMATE);
        plan_bwd        = fftwf_plan_dft_c2r_3d(size.z, size.y, size.x, reinterpret_cast<fftwf_complex*>(complex_values), real_values, FFTW_ESTIMATE);
        is_fftw_planned = true;
    }

    is_in_memory = true;

    RegisterPageLockedMemory( );
}

template <class wanted_real_type, class wanted_complex_type>
void Image<wanted_real_type, wanted_complex_type>::FwdFFT( ) {
    if ( is_fftw_planned ) {
        // Now let's do the forward FFT on the host and check that the result is correct.
        fftwf_execute_dft_r2c(plan_fwd, real_values, reinterpret_cast<fftwf_complex*>(complex_values));
    }
    else {
        std::cout << "Error: FFTW plan not set up." << std::endl;
        exit(1);
    }

    is_in_real_space = false;
}

template <class wanted_real_type, class wanted_complex_type>
void Image<wanted_real_type, wanted_complex_type>::InvFFT( ) {
    if ( is_fftw_planned ) {
        // Now let's do the forward FFT on the host and check that the result is correct.
        fftwf_execute_dft_c2r(plan_bwd, reinterpret_cast<fftwf_complex*>(complex_values), real_values);
    }
    else {
        std::cout << "Error: FFTW plan not set up." << std::endl;
        exit(1);
    }

    is_in_real_space = true;
}

template <class wanted_real_type, class wanted_complex_type>
void Image<wanted_real_type, wanted_complex_type>::MultiplyConjugateImage(wanted_complex_type* other_image_complex_values) {
    wanted_complex_type tmp;
    for ( int iPixel = 0; iPixel < real_memory_allocated / 2; iPixel++ ) {

        tmp.x                  = (complex_values[iPixel].x * other_image_complex_values[iPixel].x + complex_values[iPixel].y * other_image_complex_values[iPixel].y);
        tmp.y                  = (complex_values[iPixel].y * other_image_complex_values[iPixel].x - complex_values[iPixel].x * other_image_complex_values[iPixel].y);
        complex_values[iPixel] = tmp;
    }
}

template <class wanted_real_type, class wanted_complex_type>
void Image<wanted_real_type, wanted_complex_type>::MakeCufftPlan( ) {

    // TODO for alternate precisions.

    cufftCreate(&cuda_plan_forward);
    cufftCreate(&cuda_plan_inverse);

    cufftSetStream(cuda_plan_forward, cudaStreamPerThread);
    cufftSetStream(cuda_plan_inverse, cudaStreamPerThread);

    int            rank    = 2;
    int            iBatch  = 1;
    long long int* fftDims = new long long int[rank];
    long long int* inembed = new long long int[rank];
    long long int* onembed = new long long int[rank];

    fftDims[0] = size.y;
    fftDims[1] = size.x;

    inembed[0] = size.y;
    inembed[1] = size.w;

    onembed[0] = size.y;
    onembed[1] = size.w;

    (cufftXtMakePlanMany(cuda_plan_forward, rank, fftDims,
                         NULL, 1, 1, CUDA_R_32F,
                         NULL, 1, 1, CUDA_C_32F, iBatch, &cuda_plan_worksize_forward, CUDA_C_32F));
    (cufftXtMakePlanMany(cuda_plan_inverse, rank, fftDims,
                         NULL, 1, 1, CUDA_C_32F,
                         NULL, 1, 1, CUDA_R_32F, iBatch, &cuda_plan_worksize_inverse, CUDA_R_32F));

    delete[] fftDims;
    delete[] inembed;
    delete[] onembed;

    is_cufft_planned = true;
}

template <class wanted_real_type, class wanted_complex_type>
void Image<wanted_real_type, wanted_complex_type>::MakeCufftPlan3d( ) {

    // TODO for alternate precisions.

    cufftCreate(&cuda_plan_forward);
    cufftCreate(&cuda_plan_inverse);

    cufftSetStream(cuda_plan_forward, cudaStreamPerThread);
    cufftSetStream(cuda_plan_inverse, cudaStreamPerThread);

    int            rank    = 3;
    int            iBatch  = 1;
    long long int* fftDims = new long long int[rank];
    long long int* inembed = new long long int[rank];
    long long int* onembed = new long long int[rank];

    fftDims[0] = size.z;
    fftDims[1] = size.y;
    fftDims[2] = size.x;

    inembed[0] = size.z;
    inembed[1] = size.y;
    inembed[2] = size.w;

    onembed[0] = size.z;
    onembed[1] = size.y;
    onembed[2] = size.w;

    (cufftXtMakePlanMany(cuda_plan_forward, rank, fftDims,
                         NULL, 1, 1, CUDA_R_32F,
                         NULL, 1, 1, CUDA_C_32F, iBatch, &cuda_plan_worksize_forward, CUDA_C_32F));
    (cufftXtMakePlanMany(cuda_plan_inverse, rank, fftDims,
                         NULL, 1, 1, CUDA_C_32F,
                         NULL, 1, 1, CUDA_R_32F, iBatch, &cuda_plan_worksize_inverse, CUDA_R_32F));

    delete[] fftDims;
    delete[] inembed;
    delete[] onembed;

    is_cufft_planned = true;
}

typedef struct _CB_realLoadAndClipInto_params {
    bool*      mask;
    cufftReal* target;

} CB_realLoadAndClipInto_params;

static __device__ cufftReal CB_realLoadAndClipInto(void* dataIn, size_t offset, void* callerInfo, void* sharedPtr);

static __device__ cufftReal CB_realLoadAndClipInto(void* dataIn, size_t offset, void* callerInfo, void* sharedPtr) {

    CB_realLoadAndClipInto_params* my_params = (CB_realLoadAndClipInto_params*)callerInfo;

    if ( my_params->mask[offset] ) {
        return my_params->target[offset];
    }
    else {
        return 0.0f;
    }
}

__device__ cufftCallbackLoadR d_realLoadAndClipInto = CB_realLoadAndClipInto;

// template < class wanted_real_type, class wanted_complex_type >
// void Image<wanted_real_type, wanted_complex_type>::SetClipIntoCallback(cufftReal* image_to_insert, int image_to_insert_size_x, int image_to_insert_size_y,int image_to_insert_pitch)
// {

//   // // First make the mask
//   short4 wanted_size = make_short4(image_to_insert_size_x, image_to_insert_size_y, 1, image_to_insert_pitch);
//   SetClipIntoMask(wanted_size, size );

//   if (!is_cufft_planned) {std::cout << "Cufft plan must be made before setting callback function." << std::endl; exit(-1);}

//   cufftCallbackLoadR h_realLoadAndClipInto;
//   CB_realLoadAndClipInto_params* d_params;
//   CB_realLoadAndClipInto_params h_params;

//   precheck;
//   h_params.target = (cufftReal *)image_to_insert;
//   h_params.mask = (bool*) clipIntoMask;
//   cudaErr(cudaMalloc((void **)&d_params,sizeof(CB_realLoadAndClipInto_params)));
//   postcheck;

//   precheck;
//   cudaErr(cudaMemcpyAsync(d_params, &h_params, sizeof(CB_realLoadAndClipInto_params), cudaMemcpyHostToDevice, cudaStreamPerThread));
//   postcheck;

//   precheck;
//   cudaErr(cudaMemcpyFromSymbol(&h_realLoadAndClipInto,d_realLoadAndClipInto, sizeof(h_realLoadAndClipInto)));
//   postcheck;

//   precheck;
//   cudaErr(cudaStreamSynchronize(cudaStreamPerThread));
//   postcheck;

//   precheck;
//   cudaErr(cufftXtSetCallback(cuda_plan_forward, (void **)&h_realLoadAndClipInto, CUFFT_CB_LD_REAL, (void **)&d_params));
//   postcheck;

// }

struct CB_complexConjMulLoad_params {
    cufftComplex* target;
    cufftReal     scale;
};

static __device__ cufftComplex CB_complexConjMulLoad_32f(void* dataIn, size_t offset, void* callerInfo, void* sharedPtr) {
    CB_complexConjMulLoad_params* my_params = (CB_complexConjMulLoad_params*)callerInfo;
    return (cufftComplex)FastFFT::ComplexConjMulAndScale<cufftComplex, cufftReal>(my_params->target[offset], ((cufftComplex*)dataIn)[offset], my_params->scale);
};

__device__ cufftCallbackLoadC d_complexConjMulLoad_32f = CB_complexConjMulLoad_32f;

template <class wanted_real_type, class wanted_complex_type>
void Image<wanted_real_type, wanted_complex_type>::SetComplexConjMultiplyAndLoadCallBack(cufftComplex* search_image_FT,
                                                                                         cufftReal     FT_normalization_factor) {
    cufftCallbackStoreC           h_complexConjMulLoad;
    CB_complexConjMulLoad_params* d_params;
    CB_complexConjMulLoad_params  h_params;

    h_params.scale  = FT_normalization_factor * FT_normalization_factor;
    h_params.target = (cufftComplex*)search_image_FT;

    cudaErr(cudaMalloc((void**)&d_params, sizeof(CB_complexConjMulLoad_params)));
    cudaErr(cudaMemcpyAsync(d_params, &h_params, sizeof(CB_complexConjMulLoad_params), cudaMemcpyHostToDevice, cudaStreamPerThread));
    cudaErr(cudaMemcpyFromSymbol(&h_complexConjMulLoad, d_complexConjMulLoad_32f, sizeof(h_complexConjMulLoad)));

    cudaErr(cudaStreamSynchronize(cudaStreamPerThread));
    cudaErr(cufftXtSetCallback(cuda_plan_inverse, (void**)&h_complexConjMulLoad, CUFFT_CB_LD_COMPLEX, (void**)&d_params));
}

// To print a message and some number n_to_print complex values to stdout
template <class wanted_real_type, class wanted_complex_type>
void Image<wanted_real_type, wanted_complex_type>::print_values_complex(float* input, std::string msg, int n_to_print) {
    for ( int i = 0; i < n_to_print * 2; i += 2 ) {
        std::cout << msg << i / 2 << "  " << input[i] << " " << input[i + 1] << std::endl;
    }
}

// // Return sum of real values
// template <class wanted_real_type, class wanted_complex_type>
// float Image<wanted_real_type, wanted_complex_type>::ReturnSumOfReal(float* input, short4 size, bool print_val) {
//     double temp_sum         = 0;
//     long   address          = 0;
//     int    padding_jump_val = size.w * 2 - size.x;
//     for ( int k = 0; k < size.z; k++ ) {
//         for ( int j = 0; j < size.y; j++ ) {
//             for ( int i = 0; i < size.x; i++ ) {

//                 temp_sum += double(input[address]);
//                 address++;
//             }
//             address += padding_jump_val;
//         }
//     }

//     return float(temp_sum);
// }

template <class wanted_real_type, class wanted_complex_type>
template <typename T>
float Image<wanted_real_type, wanted_complex_type>::ReturnSumOfReal(T* input, short4 size, bool print_val) {
    double temp_sum         = 0;
    long   address          = 0;
    int    padding_jump_val = size.w * 2 - size.x;
    for ( int k = 0; k < size.z; k++ ) {
        for ( int j = 0; j < size.y; j++ ) {
            for ( int i = 0; i < size.x; i++ ) {

                temp_sum += double(input[address]);
                address++;
            }
            address += padding_jump_val;
        }
    }

    return float(temp_sum);
}

template float Image<float, float2>::ReturnSumOfReal<float>(float* input, short4 size, bool print_val);
template float Image<float, float2>::ReturnSumOfReal<half_float::half>(half_float::half* input, short4 size, bool print_val);
template float Image<float, float2>::ReturnSumOfReal<__half>(__half* input, short4 size, bool print_val);

// Return the sum of the complex values

template <class wanted_real_type, class wanted_complex_type>
float2 Image<wanted_real_type, wanted_complex_type>::ReturnSumOfComplex(float2* input, int n_to_print) {
    double sum_x = 0;
    double sum_y = 0;

    for ( int i = 0; i < n_to_print; i++ ) {
        sum_x += input[i].x;
        sum_y += input[i].y;
    }

    return make_float2(float(sum_x), float(sum_y));
}

// Return the sum of the complex values
template <class wanted_real_type, class wanted_complex_type>
float Image<wanted_real_type, wanted_complex_type>::ReturnSumOfComplexAmplitudes(float2* input, int n_to_print) {
    // We want to asses the error in the FFT at single/half precision, but to not add
    // extra error for the use double here.
    double sum = 0;
    double x;
    double y;

    for ( int i = 0; i < n_to_print; i++ ) {
        x = double(input[i].x);
        y = double(input[i].y);
        sum += sqrt(x * x + y * y);
    }

    return sum;
}

template <class wanted_real_type, class wanted_complex_type>
void Image<wanted_real_type, wanted_complex_type>::ClipInto(const float* array_to_paste, float* array_to_paste_into, short4 size_from, short4 size_into, short4 wanted_center, float wanted_padding_value) {

    long pixel_counter = 0;

    int kk;
    int k;
    int kk_logi;

    int jj;
    int jj_logi;
    int j;

    int ii;
    int ii_logi;
    int i;

    short4 center_to_paste_into = make_short4(size_into.x / 2, size_into.y / 2, size_into.z / 2, 0);
    short4 center_to_paste      = make_short4(size_from.x / 2, size_from.y / 2, size_from.z / 2, 0);
    int    padding_jump_value;

    if ( size_into.x % 2 == 0 )
        padding_jump_value = 2;
    else
        padding_jump_value = 1;

    for ( kk = 0; kk < size_into.z; kk++ ) {
        kk_logi = kk - center_to_paste_into.z;
        k       = center_to_paste.z + wanted_center.z + kk_logi;

        for ( jj = 0; jj < size_into.y; jj++ ) {
            jj_logi = jj - center_to_paste_into.y;
            j       = center_to_paste.y + wanted_center.y + jj_logi;

            for ( ii = 0; ii < size_into.x; ii++ ) {
                ii_logi = ii - center_to_paste_into.x;
                i       = center_to_paste.x + wanted_center.x + ii_logi;

                if ( k < 0 || k >= size_from.z || j < 0 || j >= size_from.y || i < 0 || i >= size_from.x ) {
                    array_to_paste_into[pixel_counter] = wanted_padding_value;
                }
                else {
                    array_to_paste_into[pixel_counter] = array_to_paste[k * (size_from.w * 2 * size_from.y) + j * (size_from.x + padding_jump_value) + i];
                }

                pixel_counter++;
            }

            pixel_counter += padding_jump_value;
        }
    }

} // end of clip into

template <class wanted_real_type, class wanted_complex_type>
void Image<wanted_real_type, wanted_complex_type>::ConvertFP32ToFP16( ) {
    if ( data_is_fp16 ) {
        std::cerr << "Error: Image is already in FP16." << std::endl;
        exit(1);
    }
    if ( ! is_in_memory ) {
        std::cerr << "Error: Image is not in memory." << std::endl;
        exit(1);
    }
    // We can just do this in place as the new values are smaller than the old ones.
    for ( int i = 0; i < real_memory_allocated; i++ ) {
        reinterpret_cast<half_float::half*>(real_values)[i] = (half_float::half)real_values[i];
    }
    data_is_fp16 = true;
}

template <class wanted_real_type, class wanted_complex_type>
void Image<wanted_real_type, wanted_complex_type>::ConvertFP16ToFP32( ) {
    if ( ! data_is_fp16 ) {
        std::cerr << "Error: Image is not already in FP16." << std::endl;
        exit(1);
    }
    if ( ! is_in_memory ) {
        std::cerr << "Error: Image is not in memory." << std::endl;
        exit(1);
    }
    // We can just do this in place as the new values are smaller than the old ones.
    float* tmp = new float[real_memory_allocated];
    for ( int i = 0; i < real_memory_allocated; i++ ) {
        tmp[i] = float(reinterpret_cast<half_float::half*>(real_values)[i]);
    }
    for ( int i = 0; i < real_memory_allocated; i++ ) {
        real_values[i] = tmp[i];
    }
    delete[] tmp;
    data_is_fp16 = false;
}

template <class wanted_real_type, class wanted_complex_type>
void Image<wanted_real_type, wanted_complex_type>::RegisterPageLockedMemory( ) {
    if ( ! is_registered ) {
        cudaErr(cudaHostRegister(real_values, sizeof(wanted_real_type) * real_memory_allocated, cudaHostRegisterDefault));
        is_registered = true;
    }
}

template <class wanted_real_type, class wanted_complex_type>
void Image<wanted_real_type, wanted_complex_type>::UnRegisterPageLockedMemory( ) {
    if ( is_registered ) {
        cudaErr(cudaHostUnregister(real_values));
    }
}

template class Image<float, float2>;
// template Image<float, float2>::Image(short4);