// Insert some license stuff here

// #include <string>
#include <iostream>
#include <string>
#include <stdio.h>

#include "../../include/FastFFT.cuh"

#ifndef FFT_DEBUG_STAGE
#error "FFT_DEBUG_STAGE must be defined"
#endif

#ifndef FFT_DEBUG_LEVEL
#error "FFT_DEBUG_LEVEL must be defined"
#endif

namespace FastFFT {

template <class ExternalImage_t, class FFT, class invFFT, class ComplexData_t, class PreOpType, class IntraOpType, class PostOpType>
__launch_bounds__(FFT::max_threads_per_block) __global__
        void block_fft_kernel_C2C_FWD_INCREASE_OP_INV_NONE(const ExternalImage_t* __restrict__ image_to_search, const ComplexData_t* __restrict__ input_values, ComplexData_t* __restrict__ output_values,
                                                           Offsets mem_offsets, int apparent_Q, typename FFT::workspace_type workspace_fwd, typename invFFT::workspace_type workspace_inv,
                                                           PreOpType pre_op_functor, IntraOpType intra_op_functor, PostOpType post_op_functor) {

    // Initialize the shared memory, assuming everyting matches the input data X size in
    using complex_compute_t = typename FFT::value_type;
    using scalar_compute_t  = typename complex_compute_t::value_type;

    // __shared__ complex_compute_t shared_mem[invFFT::shared_memory_size/sizeof(complex_compute_t)]; // Storage for the input data that is re-used each blcok
    extern __shared__ complex_compute_t shared_mem[]; // Storage for the input data that is re-used each blcok

    complex_compute_t thread_data[FFT::storage_size];

    // For simplicity, we explicitly zeropad the input data to the size of the FFT.
    // It may be worth trying to use threadIdx.y as in the DECREASE methods.
    // Until then, this

    io<FFT>::load(&input_values[Return1DFFTAddress(size_of<FFT>::value / apparent_Q)], thread_data, size_of<FFT>::value / apparent_Q, pre_op_functor);

    // In the first FFT the modifying twiddle factor is 1 so the data are reeal
    FFT( ).execute(thread_data, shared_mem, workspace_fwd);

#if FFT_DEBUG_STAGE > 3
    //  * apparent_Q
    io<invFFT>::load_shared(&image_to_search[Return1DFFTAddress(size_of<FFT>::value)], thread_data, intra_op_functor);
#endif

#if FFT_DEBUG_STAGE > 4
    invFFT( ).execute(thread_data, shared_mem, workspace_inv);
    io<invFFT>::store(thread_data, &output_values[Return1DFFTAddress(size_of<FFT>::value)], post_op_functor);
#else
    // Do not do the post op lambda if the invFFT is not used.
    io<invFFT>::store(thread_data, &output_values[Return1DFFTAddress(size_of<FFT>::value)]);
#endif
}

template <class ComputeBaseType, class InputType, class OtherImageType, int Rank>
FourierTransformer<ComputeBaseType, InputType, OtherImageType, Rank>::FourierTransformer( ) {
    SetDefaults( );
    GetCudaDeviceProps(device_properties);
    // FIXME: assert on OtherImageType being a complex type
    static_assert(std::is_same_v<ComputeBaseType, float>, "Compute base type must be float");
    static_assert(Rank == 2 || Rank == 3, "Only 2D and 3D FFTs are supported");
    static_assert(std::is_same_v<InputType, __half> || std::is_same_v<InputType, float> || std::is_same_v<InputType, __half2> || std::is_same_v<InputType, float2>,
                  "Input base type must be either __half or float");

    // exit(0);
    // This assumption precludes the use of a packed _half2 that is really RRII layout for two arrays of __half.
    static_assert(IsAllowedRealType<InputType> || IsAllowedComplexType<InputType>, "Input type must be either float or __half");

    // Make sure an explicit specializtion for the device pointers is available
    static_assert(! std::is_same_v<decltype(d_ptr.buffer_1), std::nullptr_t>, "Device pointer type not specialized");
}

template <class ComputeBaseType, class InputType, class OtherImageType, int Rank>
FourierTransformer<ComputeBaseType, InputType, OtherImageType, Rank>::~FourierTransformer( ) {
    Deallocate( );
    SetDefaults( );
}

template <class ComputeBaseType, class InputType, class OtherImageType, int Rank>
void FourierTransformer<ComputeBaseType, InputType, OtherImageType, Rank>::SetDefaults( ) {

    // booleans to track state, could be bit fields but that seem opaque to me.
    current_buffer            = fastfft_external_input;
    transform_stage_completed = 0;

    implicit_dimension_change = false;

    is_fftw_padded_input  = false; // Padding for in place r2c transforms
    is_fftw_padded_output = false; // Currently the output state will match the input state, otherwise it is an error.

    is_set_input_params  = false; // Yes, yes, "are" set.
    is_set_output_params = false;
    is_size_validated    = false; // Defaults to false, set after both input/output dimensions are set and checked.

    input_data_is_on_device     = false;
    output_data_is_on_device    = false;
    external_image_is_on_device = false;

    compute_memory_wanted_ = 0;
}

template <class ComputeBaseType, class InputType, class OtherImageType, int Rank>
void FourierTransformer<ComputeBaseType, InputType, OtherImageType, Rank>::Deallocate( ) {

    if ( is_pointer_in_device_memory(d_ptr.buffer_1) ) {
        precheck;
        cudaErr(cudaFreeAsync(d_ptr.buffer_1, cudaStreamPerThread));
        postcheck;
    }
}

/**
 * @brief Create a forward FFT plan. 
 * Buffer memory is allocated on the latter of creating forward/inverse plans.
 * Data may be copied to this buffer and used directly 
 * 
 * @tparam ComputeBaseType 
 * @tparam InputType 
 * @tparam OtherImageType 
 * @tparam Rank 
 * @param input_logical_x_dimension 
 * @param input_logical_y_dimension 
 * @param input_logical_z_dimension 
 * @param output_logical_x_dimension 
 * @param output_logical_y_dimension 
 * @param output_logical_z_dimension 
 * @param is_padded_input 
 */
template <class ComputeBaseType, class InputType, class OtherImageType, int Rank>
void FourierTransformer<ComputeBaseType, InputType, OtherImageType, Rank>::SetForwardFFTPlan(size_t input_logical_x_dimension, size_t input_logical_y_dimension, size_t input_logical_z_dimension,
                                                                                             size_t output_logical_x_dimension, size_t output_logical_y_dimension, size_t output_logical_z_dimension,
                                                                                             bool is_padded_input) {
    MyFFTDebugAssertTrue(input_logical_x_dimension > 0, "Input logical x dimension must be > 0");
    MyFFTDebugAssertTrue(input_logical_y_dimension > 0, "Input logical y dimension must be > 0");
    MyFFTDebugAssertTrue(input_logical_z_dimension > 0, "Input logical z dimension must be > 0");
    MyFFTDebugAssertTrue(output_logical_x_dimension > 0, "output logical x dimension must be > 0");
    MyFFTDebugAssertTrue(output_logical_y_dimension > 0, "output logical y dimension must be > 0");
    MyFFTDebugAssertTrue(output_logical_z_dimension > 0, "output logical z dimension must be > 0");

    fwd_dims_in  = make_short4(input_logical_x_dimension, input_logical_y_dimension, input_logical_z_dimension, 0);
    fwd_dims_out = make_short4(output_logical_x_dimension, output_logical_y_dimension, output_logical_z_dimension, 0);

    is_fftw_padded_input = is_padded_input; // Note: Must be set before ReturnPaddedMemorySize
    MyFFTRunTimeAssertTrue(is_fftw_padded_input, "Support for input arrays that are not FFTW padded needs to be implemented."); // FIXME

    // ReturnPaddedMemorySize also sets FFTW padding etc.
    input_memory_wanted_ = ReturnPaddedMemorySize(fwd_dims_in);
    // sets .w and also increases compute_memory_wanted_ if needed.
    fwd_output_memory_wanted_ = ReturnPaddedMemorySize(fwd_dims_out);

    // The compute memory allocated is the max of all possible sizes.

    this->input_origin_type = OriginType::natural;
    is_set_input_params     = true;

    if ( is_set_output_params )
        AllocateBufferMemory( ); // TODO:
}

/**
 * @brief Create an inverse FFT plan. 
 * Buffer memory is allocated on the latter of creating forward/inverse plans.
 * Data may be copied to this buffer and used directly 
 * 
 * @tparam ComputeBaseType 
 * @tparam InputType 
 * @tparam OtherImageType 
 * @tparam Rank 
 * @param input_logical_x_dimension 
 * @param input_logical_y_dimension 
 * @param input_logical_z_dimension 
 * @param output_logical_x_dimension 
 * @param output_logical_y_dimension 
 * @param output_logical_z_dimension 
 * @param is_padded_output 
 */
template <class ComputeBaseType, class InputType, class OtherImageType, int Rank>
void FourierTransformer<ComputeBaseType, InputType, OtherImageType, Rank>::SetInverseFFTPlan(size_t input_logical_x_dimension, size_t input_logical_y_dimension, size_t input_logical_z_dimension,
                                                                                             size_t output_logical_x_dimension, size_t output_logical_y_dimension, size_t output_logical_z_dimension,
                                                                                             bool is_padded_output) {
    MyFFTDebugAssertTrue(output_logical_x_dimension > 0, "output logical x dimension must be > 0");
    MyFFTDebugAssertTrue(output_logical_y_dimension > 0, "output logical y dimension must be > 0");
    MyFFTDebugAssertTrue(output_logical_z_dimension > 0, "output logical z dimension must be > 0");
    MyFFTDebugAssertTrue(is_fftw_padded_input == is_padded_output, "If the input data are FFTW padded, so must the output.");

    inv_dims_in  = make_short4(input_logical_x_dimension, input_logical_y_dimension, input_logical_z_dimension, 0);
    inv_dims_out = make_short4(output_logical_x_dimension, output_logical_y_dimension, output_logical_z_dimension, 0);

    ReturnPaddedMemorySize(inv_dims_in); // sets .w and also increases compute_memory_wanted_ if needed.
    inv_output_memory_wanted_ = ReturnPaddedMemorySize(inv_dims_out);
    // The compute memory allocated is the max of all possible sizes.

    this->output_origin_type = OriginType::natural;
    is_set_output_params     = true;
    if ( is_set_input_params )
        AllocateBufferMemory( ); // TODO:
}

/**
 * @brief Private method to allocate memory for the internal FastFFT buffer.
 * 
 * @tparam ComputeBaseType 
 * @tparam InputType 
 * @tparam OtherImageType 
 * @tparam Rank 
 */
template <class ComputeBaseType, class InputType, class OtherImageType, int Rank>
void FourierTransformer<ComputeBaseType, InputType, OtherImageType, Rank>::AllocateBufferMemory( ) {
    MyFFTDebugAssertTrue(is_set_input_params && is_set_output_params, "Input and output parameters must be set before allocating buffer memory");

    MyFFTDebugAssertTrue(compute_memory_wanted_ > 0, "Compute memory already allocated");

    // Allocate enough for the out of place buffer as well.
    constexpr size_t compute_memory_scalar = 2;
    // To get the address of the second buffer we want half of the number of ComputeType, not ComputeBaseType elements
    constexpr size_t buffer_address_scalar = 2;
    precheck;
    cudaErr(cudaMallocAsync(&d_ptr.buffer_1, compute_memory_scalar * compute_memory_wanted_ * sizeof(ComputeBaseType), cudaStreamPerThread));
    postcheck;

    // cudaMallocAsync returns the pointer immediately, even though the allocation has not yet completed, so we
    // should be fine to go on and point our secondary buffer to the correct location.
    d_ptr.buffer_2 = &d_ptr.buffer_1[compute_memory_wanted_ / buffer_address_scalar];
}

template <class ComputeBaseType, class InputType, class OtherImageType, int Rank>
void FourierTransformer<ComputeBaseType, InputType, OtherImageType, Rank>::SetInputPointerFromPython(long input_pointer) {

    MyFFTRunTimeAssertFalse(true, "This needs to be re-implemented.");
    //         MyFFTDebugAssertTrue(is_set_input_params, "Input parameters not set");

    // // The assumption for now is that access from python wrappers have taken care of device/host xfer
    // // and the passed pointer is in device memory.
    // // TODO: I should probably have a state variable to track is_python_call
    // d_ptr.position_space        = reinterpret_cast<InputType*>(input_pointer);

    // // These are normally set on CopyHostToDevice
    // SetDevicePointers( );
}

// FIXME: see header file for comments
template <class ComputeBaseType, class InputType, class OtherImageType, int Rank>
void FourierTransformer<ComputeBaseType, InputType, OtherImageType, Rank>::CopyHostToDeviceAndSynchronize(InputType* input_pointer, int n_elements_to_copy) {
    CopyHostToDevice(input_pointer, n_elements_to_copy);
    cudaErr(cudaStreamSynchronize(cudaStreamPerThread));
}

// FIXME: see header file for comments
template <class ComputeBaseType, class InputType, class OtherImageType, int Rank>
void FourierTransformer<ComputeBaseType, InputType, OtherImageType, Rank>::CopyHostToDevice(InputType* input_pointer, int n_elements_to_copy) {
    MyFFTDebugAssertFalse(input_data_is_on_device, "External input pointer is on device, cannot copy from host");
    MyFFTRunTimeAssertTrue(false, "This method is being removed.");
    SetDimensions(DimensionCheckType::CopyFromHost);

    precheck;
    cudaErr(cudaMemcpyAsync(d_ptr.buffer_1, input_pointer, memory_size_to_copy_ * sizeof(InputType), cudaMemcpyHostToDevice, cudaStreamPerThread));
    postcheck;
}

template <class ComputeBaseType, class InputType, class OtherImageType, int Rank>
template <class PreOpType,
          class IntraOpType>
void FourierTransformer<ComputeBaseType, InputType, OtherImageType, Rank>::FwdFFT(InputType*  input_ptr,
                                                                                  InputType*  output_ptr,
                                                                                  PreOpType   pre_op,
                                                                                  IntraOpType intra_op) {

    transform_stage_completed = 0;
    current_buffer            = fastfft_external_input;
    // Keep track of the device side pointer used when called
    d_ptr.external_input  = input_ptr;
    d_ptr.external_output = output_ptr;
    Generic_Fwd(pre_op, intra_op);
}

template <class ComputeBaseType, class InputType, class OtherImageType, int Rank>
template <class IntraOpType,
          class PostOpType>
void FourierTransformer<ComputeBaseType, InputType, OtherImageType, Rank>::InvFFT(InputType*  input_ptr,
                                                                                  InputType*  output_ptr,
                                                                                  IntraOpType intra_op,
                                                                                  PostOpType  post_op) {
    transform_stage_completed = 4;
    current_buffer            = fastfft_external_input;
    // Keep track of the device side pointer used when called
    d_ptr.external_input  = input_ptr;
    d_ptr.external_output = output_ptr;

    Generic_Inv(intra_op, post_op);
}

template <class ComputeBaseType, class InputType, class OtherImageType, int Rank>

template <class PreOpType,
          class IntraOpType,
          class PostOpType>
void FourierTransformer<ComputeBaseType, InputType, OtherImageType, Rank>::FwdImageInvFFT(InputType*      input_ptr,
                                                                                          OtherImageType* image_to_search,
                                                                                          InputType*      output_ptr,
                                                                                          PreOpType       pre_op,
                                                                                          IntraOpType     intra_op,
                                                                                          PostOpType      post_op) {
    transform_stage_completed = 0;
    current_buffer            = fastfft_external_input;
    // Keep track of the device side pointer used when called
    d_ptr.external_input  = input_ptr;
    d_ptr.external_output = output_ptr;

    Generic_Fwd_Image_Inv<PreOpType, IntraOpType, PostOpType>(image_to_search, pre_op, intra_op, post_op);
}

template <class ComputeBaseType, class InputType, class OtherImageType, int Rank>
template <class PreOpType,
          class IntraOpType>
EnableIf<IsAllowedInputType<InputType>>
FourierTransformer<ComputeBaseType, InputType, OtherImageType, Rank>::Generic_Fwd(PreOpType   pre_op_functor,
                                                                                  IntraOpType intra_op_functor) {

    SetDimensions(DimensionCheckType::FwdTransform);

    // TODO: extend me
    MyFFTRunTimeAssertFalse(implicit_dimension_change, "Implicit dimension change not yet supported for FwdFFT");

    // All placeholders
    constexpr bool use_thread_method = false;
    // const bool     swap_real_space_quadrants = false;
    // const bool transpose_output = true;

    // SetPrecisionAndExectutionMethod<Generic_Fwd_FFT>(<Generic_Inv_FFT, KernelType kernel_type, bool  bool use_thread_method)
    if constexpr ( Rank == 1 ) {
        // FIXME there is some redundancy in specifying _decomposed and use_thread_method
        // Note: the only time the non-transposed method should be used is for 1d data.
        if constexpr ( use_thread_method ) {
            if constexpr ( IsAllowedRealType<InputType> ) {
                SetPrecisionAndExectutionMethod<Generic_Fwd_FFT, true>(nullptr, r2c_decomposed, pre_op_functor, intra_op_functor); //FFT_R2C_decomposed(transpose_output);
                transform_stage_completed = 1;
            }
            else {
                SetPrecisionAndExectutionMethod<Generic_Fwd_FFT, true>(nullptr, c2c_fwd_decomposed, pre_op_functor, intra_op_functor);
                transform_stage_completed = 1;
            }
        }
        else {
            if constexpr ( IsAllowedRealType<InputType> ) {
                switch ( fwd_size_change_type ) {
                    case SizeChangeType::no_change: {
                        SetPrecisionAndExectutionMethod<Generic_Fwd_FFT>(nullptr, r2c_none_XY, pre_op_functor, intra_op_functor);
                        transform_stage_completed = 1;
                        break;
                    }
                    case SizeChangeType::decrease: {
                        SetPrecisionAndExectutionMethod<Generic_Fwd_FFT>(nullptr, r2c_decrease_XY, pre_op_functor, intra_op_functor);
                        transform_stage_completed = 1;
                        break;
                    }
                    case SizeChangeType::increase: {
                        SetPrecisionAndExectutionMethod<Generic_Fwd_FFT>(nullptr, r2c_increase_XY, pre_op_functor, intra_op_functor);
                        transform_stage_completed = 1;
                        break;
                    }
                    default: {
                        MyFFTDebugAssertTrue(false, "Invalid size change type");
                    }
                }
            }
            else {
                switch ( fwd_size_change_type ) {
                    case SizeChangeType::no_change: {
                        MyFFTDebugAssertTrue(false, "Complex input images are not yet supported"); // FIXME:
                        SetPrecisionAndExectutionMethod<Generic_Fwd_FFT>(nullptr, c2c_fwd_none, pre_op_functor, intra_op_functor);
                        transform_stage_completed = 1;
                        break;
                    }
                    case SizeChangeType::decrease: {
                        SetPrecisionAndExectutionMethod<Generic_Fwd_FFT>(nullptr, c2c_fwd_decrease, pre_op_functor, intra_op_functor);
                        transform_stage_completed = 1;
                        break;
                    }
                    case SizeChangeType::increase: {
                        SetPrecisionAndExectutionMethod<Generic_Fwd_FFT>(nullptr, c2c_fwd_increase, pre_op_functor, intra_op_functor);
                        transform_stage_completed = 1;
                        break;
                    }
                    default: {
                        MyFFTDebugAssertTrue(false, "Invalid size change type");
                    }
                }
            }
        }
    }
    else if constexpr ( Rank == 2 ) {
        switch ( fwd_size_change_type ) {
            case SizeChangeType::no_change: {
                // FIXME there is some redundancy in specifying _decomposed and use_thread_method
                // Note: the only time the non-transposed method should be used is for 1d data.
                if ( use_thread_method ) {
                    SetPrecisionAndExectutionMethod<Generic_Fwd_FFT, true>(nullptr, r2c_decomposed_transposed, pre_op_functor, intra_op_functor);
                    transform_stage_completed = 1;
                    SetPrecisionAndExectutionMethod<Generic_Fwd_FFT, true>(nullptr, c2c_fwd_decomposed, pre_op_functor, intra_op_functor);
                    transform_stage_completed = 3;
                }
                else {
                    SetPrecisionAndExectutionMethod<Generic_Fwd_FFT>(nullptr, r2c_none_XY, pre_op_functor, intra_op_functor);
                    transform_stage_completed = 1;
                    SetPrecisionAndExectutionMethod<Generic_Fwd_FFT>(nullptr, c2c_fwd_none, pre_op_functor, intra_op_functor);
                    transform_stage_completed = 3;
                }
                break;
            }
            case SizeChangeType::increase: {
                SetPrecisionAndExectutionMethod<Generic_Fwd_FFT>(nullptr, r2c_increase_XY, pre_op_functor, intra_op_functor);
                transform_stage_completed = 1;
                SetPrecisionAndExectutionMethod<Generic_Fwd_FFT>(nullptr, c2c_fwd_increase, pre_op_functor, intra_op_functor);
                transform_stage_completed = 3;
                break;
            }
            case SizeChangeType::decrease: {
                SetPrecisionAndExectutionMethod<Generic_Fwd_FFT>(nullptr, r2c_decrease_XY, pre_op_functor, intra_op_functor);
                transform_stage_completed = 1;
                SetPrecisionAndExectutionMethod<Generic_Fwd_FFT>(nullptr, c2c_fwd_decrease, pre_op_functor, intra_op_functor);
                transform_stage_completed = 3;
                break;
            }
        }
    }
    else if constexpr ( Rank == 3 ) {
        switch ( fwd_size_change_type ) {
            case SizeChangeType::no_change: {
                SetPrecisionAndExectutionMethod<Generic_Fwd_FFT>(nullptr, r2c_none_XZ, pre_op_functor, intra_op_functor);
                transform_stage_completed = 1;
                SetPrecisionAndExectutionMethod<Generic_Fwd_FFT>(nullptr, c2c_fwd_none_XYZ, pre_op_functor, intra_op_functor);
                transform_stage_completed = 2;
                SetPrecisionAndExectutionMethod<Generic_Fwd_FFT>(nullptr, c2c_fwd_none, pre_op_functor, intra_op_functor);
                transform_stage_completed = 3;
                break;
            }
            case SizeChangeType::increase: {
                SetPrecisionAndExectutionMethod<Generic_Fwd_FFT>(nullptr, r2c_increase_XZ, pre_op_functor, intra_op_functor);
                transform_stage_completed = 1;
                SetPrecisionAndExectutionMethod<Generic_Fwd_FFT>(nullptr, c2c_fwd_increase_XYZ, pre_op_functor, intra_op_functor);
                transform_stage_completed = 2;
                SetPrecisionAndExectutionMethod<Generic_Fwd_FFT>(nullptr, c2c_fwd_increase, pre_op_functor, intra_op_functor);
                transform_stage_completed = 3;
                break;
            }
            case SizeChangeType::decrease: {
                // Not yet supported
                MyFFTRunTimeAssertTrue(false, "3D FFT fwd decrease not yet supported");
                break;
            }
        }
    }
    else {
        MyFFTDebugAssertTrue(false, "Invalid rank");
    }
}

template <class ComputeBaseType, class InputType, class OtherImageType, int Rank>
template <class IntraOpType,
          class PostOpType>
EnableIf<IsAllowedInputType<InputType>>
FourierTransformer<ComputeBaseType, InputType, OtherImageType, Rank>::Generic_Inv(IntraOpType intra_op,
                                                                                  PostOpType  post_op) {

    SetDimensions(DimensionCheckType::InvTransform);
    MyFFTRunTimeAssertFalse(implicit_dimension_change, "Implicit dimension change not yet supported for InvFFT");

    // All placeholders
    constexpr bool use_thread_method = false;
    // const bool     swap_real_space_quadrants = false;
    // const bool     transpose_output          = true;

    switch ( transform_dimension ) {
        case 1: {
            // FIXME there is some redundancy in specifying _decomposed and use_thread_method
            // Note: the only time the non-transposed method should be used is for 1d data.
            if constexpr ( use_thread_method ) {
                if constexpr ( IsAllowedRealType<InputType> ) {
                    SetPrecisionAndExectutionMethod<Generic_Inv_FFT, true>(nullptr, c2r_decomposed, intra_op, post_op); //FFT_R2C_decomposed(transpose_output);
                    transform_stage_completed = 5;
                }
                else {
                    SetPrecisionAndExectutionMethod<Generic_Inv_FFT, true>(nullptr, c2c_inv_decomposed, intra_op, post_op);
                    transform_stage_completed = 5;
                }
            }
            else {
                if constexpr ( IsAllowedRealType<InputType> ) {
                    switch ( inv_size_change_type ) {
                        case SizeChangeType::no_change: {
                            SetPrecisionAndExectutionMethod<Generic_Inv_FFT>(nullptr, c2r_none_XY, intra_op, post_op);
                            transform_stage_completed = 5;
                            break;
                        }
                        case SizeChangeType::decrease: {
                            SetPrecisionAndExectutionMethod<Generic_Inv_FFT>(nullptr, c2r_decrease_XY, intra_op, post_op);
                            transform_stage_completed = 5;
                            break;
                        }
                        case SizeChangeType::increase: {
                            SetPrecisionAndExectutionMethod<Generic_Inv_FFT>(nullptr, c2r_increase, intra_op, post_op);
                            transform_stage_completed = 5;
                            break;
                        }
                        default: {
                            MyFFTDebugAssertTrue(false, "Invalid size change type");
                        }
                    }
                }
                else {
                    switch ( inv_size_change_type ) {
                        case SizeChangeType::no_change: {
                            SetPrecisionAndExectutionMethod<Generic_Inv_FFT>(nullptr, c2c_inv_none, intra_op, post_op);
                            transform_stage_completed = 5;
                            break;
                        }
                        case SizeChangeType::decrease: {
                            SetPrecisionAndExectutionMethod<Generic_Inv_FFT>(nullptr, c2c_inv_decrease, intra_op, post_op);
                            transform_stage_completed = 5;
                            break;
                        }
                        case SizeChangeType::increase: {
                            SetPrecisionAndExectutionMethod<Generic_Inv_FFT>(nullptr, c2c_inv_increase, intra_op, post_op);
                            transform_stage_completed = 5;
                            break;
                        }
                        default: {
                            MyFFTDebugAssertTrue(false, "Invalid size change type");
                        }
                    }
                }
            }
            break;
        }
        case 2: {
            switch ( inv_size_change_type ) {
                case SizeChangeType::no_change: {
                    // FIXME there is some redundancy in specifying _decomposed and use_thread_method
                    // Note: the only time the non-transposed method should be used is for 1d data.
                    if ( use_thread_method ) {
                        SetPrecisionAndExectutionMethod<Generic_Inv_FFT, true>(nullptr, c2c_inv_decomposed, intra_op, post_op);
                        transform_stage_completed = 5;
                        SetPrecisionAndExectutionMethod<Generic_Inv_FFT, true>(nullptr, c2r_decomposed_transposed, intra_op, post_op);
                        transform_stage_completed = 7;
                    }
                    else {
                        SetPrecisionAndExectutionMethod<Generic_Inv_FFT>(nullptr, c2c_inv_none, intra_op, post_op);
                        transform_stage_completed = 5;
                        SetPrecisionAndExectutionMethod<Generic_Inv_FFT>(nullptr, c2r_none_XY, intra_op, post_op);
                        transform_stage_completed = 7;
                    }
                    break;
                }
                case SizeChangeType::increase: {
                    SetPrecisionAndExectutionMethod<Generic_Inv_FFT>(nullptr, c2c_inv_increase, intra_op, post_op);
                    transform_stage_completed = 5;
                    SetPrecisionAndExectutionMethod<Generic_Inv_FFT>(nullptr, c2r_increase, intra_op, post_op);
                    transform_stage_completed = 7;
                    break;
                }
                case SizeChangeType::decrease: {
                    SetPrecisionAndExectutionMethod<Generic_Inv_FFT>(nullptr, c2c_inv_decrease, intra_op, post_op);
                    transform_stage_completed = 5;
                    SetPrecisionAndExectutionMethod<Generic_Inv_FFT>(nullptr, c2r_decrease_XY, intra_op, post_op);
                    transform_stage_completed = 7;
                    break;
                }
                default: {
                    MyFFTDebugAssertTrue(false, "Invalid size change type");
                    break;
                }
            } // switch on inv size change type
            break; // case 2
        }
        case 3: {
            switch ( inv_size_change_type ) {
                case SizeChangeType::no_change: {
                    SetPrecisionAndExectutionMethod<Generic_Inv_FFT>(nullptr, c2c_inv_none_XZ, intra_op, post_op);
                    transform_stage_completed = 5;
                    SetPrecisionAndExectutionMethod<Generic_Inv_FFT>(nullptr, c2c_inv_none_XYZ, intra_op, post_op);
                    transform_stage_completed = 6;
                    SetPrecisionAndExectutionMethod<Generic_Inv_FFT>(nullptr, c2r_none, intra_op, post_op);
                    transform_stage_completed = 7;
                    break;
                }
                case SizeChangeType::increase: {
                    MyFFTRunTimeAssertFalse(true, "3D FFT inv increase not yet supported");
                    SetPrecisionAndExectutionMethod<Generic_Inv_FFT>(nullptr, r2c_increase_XY, intra_op, post_op);
                    // SetPrecisionAndExectutionMethod<Generic_Inv_FFT>( nullptr, c2c_fwd_increase_XYZ);
                    break;
                }
                case SizeChangeType::decrease: {
                    // Not yet supported
                    MyFFTRunTimeAssertTrue(false, "3D FFT inv no decrease not yet supported");
                    break;
                }
                default: {
                    MyFFTDebugAssertTrue(false, "Invalid dimension");
                    break;
                }
            } // switch on inv size change type
        }
    }
}

template <class ComputeBaseType, class InputType, class OtherImageType, int Rank>
template <class PreOpType,
          class IntraOpType,
          class PostOpType>
EnableIf<HasIntraOpFunctor<IntraOpType> && IsAllowedInputType<InputType, OtherImageType>>
FourierTransformer<ComputeBaseType, InputType, OtherImageType, Rank>::Generic_Fwd_Image_Inv(OtherImageType* image_to_search_ptr,
                                                                                            PreOpType       pre_op_functor,
                                                                                            IntraOpType     intra_op_functor,
                                                                                            PostOpType      post_op_functor) {

    // Set the member pointer to the passed pointer
    SetDimensions(DimensionCheckType::FwdTransform);

    switch ( transform_dimension ) {
        case 1: {
            MyFFTRunTimeAssertTrue(false, "1D FFT Cross correlation not yet supported");
            break;
        }
        case 2: {
            switch ( fwd_size_change_type ) {
                case SizeChangeType::no_change: {
                    SetPrecisionAndExectutionMethod<Generic_Fwd_FFT>(image_to_search_ptr, r2c_none_XY, pre_op_functor, intra_op_functor, post_op_functor);
                    transform_stage_completed = 1;
                    switch ( inv_size_change_type ) {
                        case SizeChangeType::no_change: {
                            MyFFTRunTimeAssertTrue(false, "2D FFT generic lambda no change/nochange not yet supported");
                            break;
                        }
                        case SizeChangeType::increase: {
                            MyFFTRunTimeAssertTrue(false, "2D FFT generic lambda no change/increase not yet supported");
                            break;
                        }
                        case SizeChangeType::decrease: {
                            SetPrecisionAndExectutionMethod<Generic_Fwd_Image_Inv_FFT>(image_to_search_ptr, xcorr_fwd_none_inv_decrease, pre_op_functor, intra_op_functor, post_op_functor);
                            transform_stage_completed = 5;
                            SetPrecisionAndExectutionMethod<Generic_Inv_FFT>(image_to_search_ptr, c2r_decrease_XY, pre_op_functor, intra_op_functor, post_op_functor);
                            transform_stage_completed = 7;
                            break;
                        }
                        default: {
                            MyFFTDebugAssertTrue(false, "Invalid size change type");
                            break;
                        }
                    } // switch on inv size change type
                    break;
                } // case fwd no change
                case SizeChangeType::increase: {
                    SetPrecisionAndExectutionMethod<Generic_Fwd_FFT>(image_to_search_ptr, r2c_increase_XY, pre_op_functor, intra_op_functor, post_op_functor);
                    transform_stage_completed = 1;
                    switch ( inv_size_change_type ) {
                        case SizeChangeType::no_change: {
                            SetPrecisionAndExectutionMethod<Generic_Fwd_Image_Inv_FFT>(image_to_search_ptr, generic_fwd_increase_op_inv_none, pre_op_functor, intra_op_functor, post_op_functor);
                            transform_stage_completed = 5;
                            SetPrecisionAndExectutionMethod<Generic_Inv_FFT>(image_to_search_ptr, c2r_none_XY, pre_op_functor, intra_op_functor, post_op_functor);
                            transform_stage_completed = 7;
                            break;
                        }

                        case SizeChangeType::increase: {
                            // I don't see where increase increase makes any sense
                            // FIXME add a check on this in the validation step.
                            MyFFTRunTimeAssertTrue(false, "2D FFT Cross correlation with fwd and inv size increase is not supported");
                            break;
                        }
                        case SizeChangeType::decrease: {
                            // with FwdTransform set, call c2c
                            // Set InvTransform
                            // Call new kernel that handles the conj mul inv c2c trimmed, and inv c2r in one go.
                            MyFFTRunTimeAssertTrue(false, "2D FFT Cross correlation with fwd increase and inv size decrease is a work in progress");

                            break;
                        }
                        default: {
                            MyFFTRunTimeAssertTrue(false, "Invalid size change type");
                        }
                    } // switch on inv size change type
                    break;
                }
                case SizeChangeType::decrease: {

                    SetPrecisionAndExectutionMethod<Generic_Fwd_FFT>(image_to_search_ptr, r2c_decrease_XY, pre_op_functor, intra_op_functor, post_op_functor);
                    transform_stage_completed = 1;
                    switch ( inv_size_change_type ) {
                        case SizeChangeType::no_change: {
                            SetPrecisionAndExectutionMethod<Generic_Fwd_Image_Inv_FFT>(image_to_search_ptr, generic_fwd_increase_op_inv_none, pre_op_functor, intra_op_functor, post_op_functor);
                            transform_stage_completed = 5;
                            SetPrecisionAndExectutionMethod<Generic_Inv_FFT>(image_to_search_ptr, c2r_none_XY, pre_op_functor, intra_op_functor, post_op_functor);
                            transform_stage_completed = 7;
                            break;
                        }
                        case SizeChangeType::increase: {

                            MyFFTRunTimeAssertTrue(false, "2D FFT Cross correlation with fwd and inv size increase is not supported");
                            break;
                        }
                        case SizeChangeType::decrease: {

                            MyFFTRunTimeAssertTrue(false, "2D FFT Cross correlation with fwd decrease and inv size decrease is a work in progress");
                            break;
                        }
                        default: {
                            MyFFTRunTimeAssertTrue(false, "Invalid inv size change type");
                        } break;
                    }
                    break;
                } // case decrease
                default: {
                    MyFFTRunTimeAssertTrue(false, "Invalid fwd size change type");
                }

            } // switch on fwd size change type
            break; // case dimension 2
        }
        case 3: {
            switch ( fwd_size_change_type ) {
                case SizeChangeType::no_change: {
                    MyFFTDebugAssertTrue(false, "3D FFT Cross correlation fwd no change not yet supported");
                    switch ( inv_size_change_type ) {
                        case SizeChangeType::no_change: {
                            break;
                        }
                        case SizeChangeType::increase: {
                            MyFFTDebugAssertTrue(false, "3D FFT Cross correlation with fwd and inv size increase is not supported");
                            break;
                        }
                        case SizeChangeType::decrease: {
                            MyFFTDebugAssertTrue(false, "3D FFT Cross correlation with fwd and inv size decrease is not supported");
                            break;
                        }
                    }
                    break;
                }
                case SizeChangeType::increase: {
                    SetPrecisionAndExectutionMethod<Generic_Fwd_FFT>(image_to_search_ptr, r2c_increase_XZ, pre_op_functor, intra_op_functor, post_op_functor);
                    transform_stage_completed = 1;
                    SetPrecisionAndExectutionMethod<Generic_Fwd_FFT>(image_to_search_ptr, c2c_fwd_increase_XYZ, pre_op_functor, intra_op_functor, post_op_functor);
                    transform_stage_completed = 2;
                    switch ( inv_size_change_type ) {
                        case SizeChangeType::no_change: {
                            // TODO: will need a kernel for generic_fwd_increase_op_inv_none_XZ
                            SetPrecisionAndExectutionMethod<Generic_Fwd_Image_Inv_FFT>(image_to_search_ptr, generic_fwd_increase_op_inv_none, pre_op_functor, intra_op_functor, post_op_functor);
                            transform_stage_completed = 5;
                            // SetPrecisionAndExectutionMethod<Generic_Fwd_Image_Inv_FFT>( image_to_search_ptr, c2c_inv_none_XZ);
                            SetPrecisionAndExectutionMethod<Generic_Inv_FFT>(image_to_search_ptr, c2c_inv_none_XYZ, pre_op_functor, intra_op_functor, post_op_functor);
                            transform_stage_completed = 6;
                            SetPrecisionAndExectutionMethod<Generic_Inv_FFT>(image_to_search_ptr, c2r_none, pre_op_functor, intra_op_functor, post_op_functor);
                            transform_stage_completed = 7;
                            break;
                        }
                        case SizeChangeType::increase: {
                            MyFFTDebugAssertTrue(false, "3D FFT Cross correlation with fwd and inv size increase is not supported");
                            break;
                        }
                        case SizeChangeType::decrease: {
                            MyFFTDebugAssertTrue(false, "3D FFT Cross correlation with fwd and inv size decrease is not supported");
                            break;
                        }
                        default: {
                            MyFFTRunTimeAssertTrue(false, "Invalid inv size change type");
                        }
                    }
                    break;
                }
                case SizeChangeType::decrease: {
                    MyFFTDebugAssertTrue(false, "3D FFT Cross correlation fwd decrease not yet supported");
                    switch ( inv_size_change_type ) {
                        case SizeChangeType::no_change: {
                            break;
                        }
                        case SizeChangeType::increase: {
                            MyFFTDebugAssertTrue(false, "3D FFT Cross correlation with fwd and inv size increase is not supported");
                            break;
                        }
                        case SizeChangeType::decrease: {
                            MyFFTDebugAssertTrue(false, "3D FFT Cross correlation with fwd and inv size decrease is not supported");
                            break;
                        }
                    }
                    break;
                }
                default: {
                    MyFFTRunTimeAssertTrue(false, "Invalid fwd size change type");
                }
            }
        }
        default: {
            MyFFTRunTimeAssertTrue(false, "Invalid dimension");
        }
    } // switch on transform dimension
}

////////////////////////////////////////////////////
/// END PUBLIC METHODS
////////////////////////////////////////////////////
template <class ComputeBaseType, class InputType, class OtherImageType, int Rank>
void FourierTransformer<ComputeBaseType, InputType, OtherImageType, Rank>::ValidateDimensions( ) {
    // TODO - runtime asserts would be better as these are breaking errors that are under user control.
    // check to see if there is any measurable penalty for this.

    MyFFTDebugAssertTrue(is_set_input_params, "Input parameters not set");
    MyFFTDebugAssertTrue(is_set_output_params, "Output parameters not set");

    MyFFTRunTimeAssertTrue(fwd_dims_out.x == inv_dims_in.x &&
                                   fwd_dims_out.y == inv_dims_in.y &&
                                   fwd_dims_out.z == inv_dims_in.z,
                           "Error in validating the dimension: Currently all fwd out should match inv in.");

    // Validate the forward transform
    if ( fwd_dims_out.x > fwd_dims_in.x || fwd_dims_out.y > fwd_dims_in.y || fwd_dims_out.z > fwd_dims_in.z ) {
        // For now we must pad in all dimensions, this is not needed and should be lifted. FIXME
        MyFFTDebugAssertTrue(fwd_dims_out.x >= fwd_dims_in.x, "If padding, all dimensions must be >=, x out < x in");
        MyFFTDebugAssertTrue(fwd_dims_out.y >= fwd_dims_in.y, "If padding, all dimensions must be >=, y out < y in");
        MyFFTDebugAssertTrue(fwd_dims_out.z >= fwd_dims_in.z, "If padding, all dimensions must be >=, z out < z in");

        fwd_size_change_type = SizeChangeType::increase;
    }
    else if ( fwd_dims_out.x < fwd_dims_in.x || fwd_dims_out.y < fwd_dims_in.y || fwd_dims_out.z < fwd_dims_in.z ) {
        // For now we must pad in all dimensions, this is not needed and should be lifted. FIXME
        MyFFTDebugAssertTrue(fwd_dims_out.x <= fwd_dims_in.x, "If padding, all dimensions must be <=, x out > x in");
        MyFFTDebugAssertTrue(fwd_dims_out.y <= fwd_dims_in.y, "If padding, all dimensions must be <=, y out > y in");
        MyFFTDebugAssertTrue(fwd_dims_out.z <= fwd_dims_in.z, "If padding, all dimensions must be <=, z out > z in");

        fwd_size_change_type = SizeChangeType::decrease;
    }
    else if ( fwd_dims_out.x == fwd_dims_in.x && fwd_dims_out.y == fwd_dims_in.y && fwd_dims_out.z == fwd_dims_in.z ) {
        fwd_size_change_type = SizeChangeType::no_change;
    }
    else {
        // TODO: if this is relaxed, the dimensionality check below will be invalid.
        MyFFTRunTimeAssertTrue(false, "Error in validating fwd plan: Currently all dimensions must either increase, decrease or stay the same.");
    }

    // Validate the inverse transform
    if ( inv_dims_out.x > inv_dims_in.x || inv_dims_out.y > inv_dims_in.y || inv_dims_out.z > inv_dims_in.z ) {
        // For now we must pad in all dimensions, this is not needed and should be lifted. FIXME
        MyFFTDebugAssertTrue(inv_dims_out.x >= inv_dims_in.x, "If padding, all dimensions must be >=, x out < x in");
        MyFFTDebugAssertTrue(inv_dims_out.y >= inv_dims_in.y, "If padding, all dimensions must be >=, y out < y in");
        MyFFTDebugAssertTrue(inv_dims_out.z >= inv_dims_in.z, "If padding, all dimensions must be >=, z out < z in");

        inv_size_change_type = SizeChangeType::increase;
    }
    else if ( inv_dims_out.x < inv_dims_in.x || inv_dims_out.y < inv_dims_in.y || inv_dims_out.z < inv_dims_in.z ) {
        inv_size_change_type = SizeChangeType::decrease;
    }
    else if ( inv_dims_out.x == inv_dims_in.x && inv_dims_out.y == inv_dims_in.y && inv_dims_out.z == inv_dims_in.z ) {
        inv_size_change_type = SizeChangeType::no_change;
    }
    else {
        // TODO: if this is relaxed, the dimensionality check below will be invalid.
        MyFFTRunTimeAssertTrue(false, "Error in validating inv plan: Currently all dimensions must either increase, decrease or stay the same.");
    }

    // check for dimensionality
    // Note: this is predicated on the else clause ensuring all dimensions behave the same way w.r.t. size change.
    if ( fwd_dims_in.z == 1 && fwd_dims_out.z == 1 ) {
        MyFFTRunTimeAssertTrue(inv_dims_in.z == 1 && inv_dims_out.z == 1, "Fwd/Inv dimensionality may not change from 1d,2d,3d (z dimension)");
        if ( fwd_dims_in.y == 1 && fwd_dims_out.y == 1 ) {
            MyFFTRunTimeAssertTrue(inv_dims_in.y == 1 && inv_dims_out.y == 1, "Fwd/Inv dimensionality may not change from 1d,2d,3d (y dimension)");
            transform_dimension = 1;
        }
        else {
            transform_dimension = 2;
        }
    }
    else {
        transform_dimension                = 3;
        constexpr unsigned int max_3d_size = 512;
        MyFFTRunTimeAssertFalse(fwd_dims_in.z > max_3d_size ||
                                        fwd_dims_out.z > max_3d_size ||
                                        inv_dims_in.z > max_3d_size ||
                                        inv_dims_out.z > max_3d_size ||
                                        fwd_dims_in.y > max_3d_size ||
                                        fwd_dims_out.y > max_3d_size ||
                                        inv_dims_in.y > max_3d_size ||
                                        inv_dims_out.y > max_3d_size ||
                                        fwd_dims_in.x > max_3d_size ||
                                        fwd_dims_out.x > max_3d_size ||
                                        inv_dims_in.x > max_3d_size ||
                                        inv_dims_out.x > max_3d_size,
                                "Error in validating the dimension: Currently all dimensions must be <= 512 for 3d transforms.");
    }

    // Allow some non-power of 2 sizes if there is a size change type.
    // For now, just forward increase.
    // Note: initially we'll only allow this logic for round trip transforms (that don't write out fwd_dims_out or read inv_dims_in) where
    //       the caller may be "surprised" by the size change.
    if ( ! IsAPowerOfTwo(fwd_dims_in.x) ) {
        MyFFTRunTimeAssertTrue(fwd_size_change_type == SizeChangeType::increase, "Input x dimension must be a power of 2");
        implicit_dimension_change = true;
    }
    if ( ! IsAPowerOfTwo(fwd_dims_in.y) ) {
        MyFFTRunTimeAssertTrue(fwd_size_change_type == SizeChangeType::increase, "Input y dimension must be a power of 2");
        implicit_dimension_change = true;
    }
    if ( ! IsAPowerOfTwo(fwd_dims_in.z) ) {
        MyFFTRunTimeAssertTrue(fwd_size_change_type == SizeChangeType::increase, "Input z dimension must be a power of 2");
        implicit_dimension_change = true;
    }

    MyFFTRunTimeAssertTrue(IsAPowerOfTwo(fwd_dims_out.x), "Output x dimension must be a power of 2");
    MyFFTRunTimeAssertTrue(IsAPowerOfTwo(fwd_dims_out.y), "Output y dimension must be a power of 2");
    MyFFTRunTimeAssertTrue(IsAPowerOfTwo(fwd_dims_out.z), "Output z dimension must be a power of 2");

    MyFFTRunTimeAssertTrue(IsAPowerOfTwo(inv_dims_in.x), "Input x dimension must be a power of 2");
    MyFFTRunTimeAssertTrue(IsAPowerOfTwo(inv_dims_in.y), "Input y dimension must be a power of 2");
    MyFFTRunTimeAssertTrue(IsAPowerOfTwo(inv_dims_in.z), "Input z dimension must be a power of 2");

    if ( ! IsAPowerOfTwo(inv_dims_out.x) ) {
        MyFFTRunTimeAssertTrue(inv_size_change_type == SizeChangeType::decrease, "Output x dimension must be a power of 2");
        implicit_dimension_change = true;
    }
    if ( ! IsAPowerOfTwo(inv_dims_out.y) ) {
        MyFFTRunTimeAssertTrue(inv_size_change_type == SizeChangeType::decrease, "Output y dimension must be a power of 2");
        implicit_dimension_change = true;
    }
    if ( ! IsAPowerOfTwo(inv_dims_out.z) ) {
        MyFFTRunTimeAssertTrue(inv_size_change_type == SizeChangeType::decrease, "Output z dimension must be a power of 2");
        implicit_dimension_change = true;
    }

    is_size_validated = true;
}

template <class ComputeBaseType, class InputType, class OtherImageType, int Rank>
void FourierTransformer<ComputeBaseType, InputType, OtherImageType, Rank>::SetDimensions(DimensionCheckType::Enum check_op_type) {
    // This should be run inside any public method call to ensure things ar properly setup.
    if ( ! is_size_validated ) {
        ValidateDimensions( );
    }

    switch ( check_op_type ) {
        case DimensionCheckType::CopyFromHost: {
            // MyFFTDebugAssertTrue(transform_stage_completed == none, "When copying from host, the transform stage should be none, something has gone wrong.");
            // FIXME: is this the right thing to do? Maybe this should be explicitly "reset" when the input image is "refereshed."
            memory_size_to_copy_ = input_memory_wanted_;
            break;
        }

        case DimensionCheckType::CopyToHost: {
            if ( transform_stage_completed == 0 ) {
                memory_size_to_copy_ = input_memory_wanted_;
            }
            else if ( transform_stage_completed < 5 ) {
                memory_size_to_copy_ = fwd_output_memory_wanted_;
            }
            else {
                memory_size_to_copy_ = inv_output_memory_wanted_;
            }
        } // switch transform_stage_completed
        break;

    } // end switch on operation type
}

////////////////////////////////////////////////////
/// Transform kernels
////////////////////////////////////////////////////

// R2C_decomposed

template <class ComputeBaseType, class InputType, class OtherImageType, int Rank>
void FourierTransformer<ComputeBaseType, InputType, OtherImageType, Rank>::CopyDeviceToHostAndSynchronize(InputType* input_pointer, int n_elements_to_copy) {
    SetDimensions(DimensionCheckType::CopyToHost);
    int n_to_actually_copy = (n_elements_to_copy > 0) ? n_elements_to_copy : memory_size_to_copy_;

    MyFFTDebugAssertTrue(n_to_actually_copy > 0, "Error in CopyDeviceToHostAndSynchronize: n_elements_to_copy must be > 0");
    MyFFTDebugAssertTrue(is_pointer_in_memory_and_registered(input_pointer), "Error in CopyDeviceToHostAndSynchronize: input_pointer must be in memory and registered");

    switch ( current_buffer ) {
        case fastfft_external_input: {
            MyFFTDebugAssertTrue(is_pointer_in_device_memory(d_ptr.external_input), "Error in CopyDeviceToHostAndSynchronize: input_pointer must be in device memory");
            cudaErr(cudaMemcpyAsync(input_pointer, d_ptr.external_input, n_to_actually_copy * sizeof(InputType), cudaMemcpyDeviceToHost, cudaStreamPerThread));
            break;
        }
        case fastfft_external_output: {
            MyFFTDebugAssertTrue(is_pointer_in_device_memory(d_ptr.external_input), "Error in CopyDeviceToHostAndSynchronize: input_pointer must be in device memory");
            cudaErr(cudaMemcpyAsync(input_pointer, d_ptr.external_output, n_to_actually_copy * sizeof(InputType), cudaMemcpyDeviceToHost, cudaStreamPerThread));
            break;
        }
        // If we are in the internal buffers, our data is ComputeBaseType
        case fastfft_internal_buffer_1: {
            MyFFTDebugAssertTrue(is_pointer_in_device_memory(d_ptr.buffer_1), "Error in CopyDeviceToHostAndSynchronize: input_pointer must be in device memory");
            if ( sizeof(ComputeBaseType) != sizeof(InputType) )
                std::cerr << "\n\tWarning: CopyDeviceToHostAndSynchronize: sizeof(ComputeBaseType) != sizeof(InputType) - this may be a problem\n\n";
            cudaErr(cudaMemcpyAsync(input_pointer, d_ptr.buffer_1, n_to_actually_copy * sizeof(ComputeBaseType), cudaMemcpyDeviceToHost, cudaStreamPerThread));
            break;
        }
        case fastfft_internal_buffer_2: {
            MyFFTDebugAssertTrue(is_pointer_in_device_memory(d_ptr.buffer_2), "Error in CopyDeviceToHostAndSynchronize: input_pointer must be in device memory");
            if ( sizeof(ComputeBaseType) != sizeof(InputType) )
                std::cerr << "\n\tWarning: CopyDeviceToHostAndSynchronize: sizeof(ComputeBaseType) != sizeof(InputType) - this may be a problem\n\n";
            cudaErr(cudaMemcpyAsync(input_pointer, d_ptr.buffer_2, n_to_actually_copy * sizeof(ComputeBaseType), cudaMemcpyDeviceToHost, cudaStreamPerThread));
            break;
        }
        default: {
            MyFFTDebugAssertTrue(false, "Error in CopyDeviceToHostAndSynchronize: current_buffer must be one of fastfft_external_input, fastfft_internal_buffer_1, fastfft_internal_buffer_2");
        }
    }

    cudaErr(cudaStreamSynchronize(cudaStreamPerThread));
};

template <class FFT, class InputData_t, class OutputData_t>
__global__ void thread_fft_kernel_R2C_decomposed(const InputData_t* __restrict__ input_values, OutputData_t* __restrict__ output_values, Offsets mem_offsets, float twiddle_in, int Q) {

    using complex_compute_t = typename FFT::value_type;
    using scalar_compute_t  = typename complex_compute_t::value_type;
    // The data store is non-coalesced, so don't aggregate the data in shared mem.
    extern __shared__ complex_compute_t shared_mem[];

    // Memory used by FFT - for Thread() type, FFT::storage_size == FFT::elements_per_thread == size_of<FFT>::value
    complex_compute_t thread_data[FFT::storage_size];

    io_thread<FFT>::load_r2c(&input_values[Return1DFFTAddress(mem_offsets.physical_x_input)], thread_data, Q);

    // We then have Q FFTs of size size_of<FFT>::value (P in the paper)
    FFT( ).execute(thread_data);

    // Now we need to aggregate each of the Q transforms into each output block of size P
    io_thread<FFT>::remap_decomposed_segments(thread_data, shared_mem, twiddle_in, Q, mem_offsets.physical_x_output);

    io_thread<FFT>::store_r2c(shared_mem, &output_values[Return1DFFTAddress(mem_offsets.physical_x_output)], Q, mem_offsets.physical_x_output);
}

template <class FFT, class InputData_t, class OutputData_t>
__global__ void thread_fft_kernel_R2C_decomposed_transposed(const InputData_t* __restrict__ input_values, OutputData_t* __restrict__ output_values, Offsets mem_offsets, float twiddle_in, int Q) {

    using complex_compute_t = typename FFT::value_type;
    using scalar_compute_t  = typename complex_compute_t::value_type;
    // The data store is non-coalesced, so don't aggregate the data in shared mem.
    extern __shared__ complex_compute_t shared_mem[];

    // Memory used by FFT - for Thread() type, FFT::storage_size == FFT::elements_per_thread == size_of<FFT>::value
    complex_compute_t thread_data[FFT::storage_size];

    io_thread<FFT>::load_r2c(&input_values[Return1DFFTAddress(mem_offsets.physical_x_input)], thread_data, Q);

    // We then have Q FFTs of size size_of<FFT>::value (P in the paper)
    FFT( ).execute(thread_data);

    // Now we need to aggregate each of the Q transforms into each output block of size P
    io_thread<FFT>::remap_decomposed_segments(thread_data, shared_mem, twiddle_in, Q, mem_offsets.physical_x_output);

    io_thread<FFT>::store_r2c_transposed_xy(shared_mem, &output_values[ReturnZplane(blockDim.y, mem_offsets.physical_x_output)], Q, gridDim.y, mem_offsets.physical_x_output);
}

template <class FFT, class InputData_t, class OutputData_t>
__launch_bounds__(FFT::max_threads_per_block) __global__
        void block_fft_kernel_R2C_NONE_XY(const InputData_t* __restrict__ input_values, OutputData_t* __restrict__ output_values, Offsets mem_offsets, typename FFT::workspace_type workspace) {
    // Initialize the shared memory, assuming everyting matches the input data X size in
    using complex_compute_t = typename FFT::value_type;
    using scalar_compute_t  = typename complex_compute_t::value_type;

    // The data store is non-coalesced, so don't aggregate the data in shared mem.
    extern __shared__ complex_compute_t shared_mem[];

    // Memory used by FFT
    complex_compute_t thread_data[FFT::storage_size];

    // No need to __syncthreads as each thread only accesses its own shared mem anyway
    // multiply Q*fwd_dims_out.w because x maps to y in the output transposed FFT
    io<FFT>::load_r2c(&input_values[Return1DFFTAddress(mem_offsets.physical_x_input)], thread_data);
    // io<FFT>::load_r2c(&input_values[blockIdx.y*mem_offsets.physical_x_input], thread_data);

    // In the first FFT the modifying twiddle factor is 1 so the data are real
    FFT( ).execute(thread_data, shared_mem, workspace);

    io<FFT>::store_r2c_transposed_xy(thread_data, &output_values[ReturnZplane(gridDim.y, mem_offsets.physical_x_output)], gridDim.y);
}

// 2 ffts/block via threadIdx.x, notice launch bounds. Creates partial coalescing.

template <class FFT, class InputData_t, class OutputData_t>
__launch_bounds__(XZ_STRIDE* FFT::max_threads_per_block) __global__
        void block_fft_kernel_R2C_NONE_XZ(const InputData_t* __restrict__ input_values, OutputData_t* __restrict__ output_values, Offsets mem_offsets, typename FFT::workspace_type workspace) {
    // Initialize the shared memory, assuming everyting matches the input data X size in
    using complex_compute_t = typename FFT::value_type;
    using scalar_compute_t  = typename complex_compute_t::value_type;

    // The data store is non-coalesced, so don't aggregate the data in shared mem.
    extern __shared__ complex_compute_t shared_mem[];

    // Memory used by FFT
    complex_compute_t thread_data[FFT::storage_size];

    io<FFT>::load_r2c(&input_values[Return1DFFTAddress_strided_Z(mem_offsets.physical_x_input)], thread_data);

    constexpr const unsigned int n_compute_elements = FFT::shared_memory_size / sizeof(complex_compute_t);
    FFT( ).execute(thread_data, &shared_mem[threadIdx.y * n_compute_elements], workspace);
    __syncthreads( ); // TODO: is this needed?

    // memory is at least large enough to hold the output with padding. synchronizing
    io<FFT>::transpose_r2c_in_shared_XZ(shared_mem, thread_data);

    // Transpose XZ, so the proper Z dimension now comes from X
    io<FFT>::store_r2c_transposed_xz_strided_Z(shared_mem, output_values);
}

template <class FFT, class InputData_t, class OutputData_t>
__launch_bounds__(FFT::max_threads_per_block) __global__
        void block_fft_kernel_R2C_INCREASE_XY(const InputData_t* __restrict__ input_values, OutputData_t* __restrict__ output_values, Offsets mem_offsets, float twiddle_in, int Q, typename FFT::workspace_type workspace) {
    // Initialize the shared memory, assuming everyting matches the input data X size in
    using complex_compute_t = typename FFT::value_type;
    using scalar_compute_t  = typename complex_compute_t::value_type;

    // The data store is non-coalesced, so don't aggregate the data in shared mem.
    extern __shared__ scalar_compute_t shared_input[];
    complex_compute_t*                 shared_mem = (complex_compute_t*)&shared_input[mem_offsets.shared_input];

    // Memory used by FFT
    complex_compute_t twiddle;
    complex_compute_t thread_data[FFT::storage_size];

    // To re-map the thread index to the data ... these really could be short ints, but I don't know how that will perform. TODO benchmark
    // It is also questionable whether storing these vs, recalculating makes more sense.
    int   input_MAP[FFT::storage_size];
    int   output_MAP[FFT::storage_size];
    float twiddle_factor_args[FFT::storage_size];

    // No need to __syncthreads as each thread only accesses its own shared mem anyway
    // multiply Q*fwd_dims_out.w because x maps to y in the output transposed FFT
    io<FFT>::load_r2c_shared(&input_values[Return1DFFTAddress(mem_offsets.physical_x_input)], shared_input, thread_data, twiddle_factor_args, twiddle_in, input_MAP, output_MAP, Q);

    // We unroll the first and last loops.
    // In the first FFT the modifying twiddle factor is 1 so the data are real
    FFT( ).execute(thread_data, shared_mem, workspace);

    io<FFT>::store_r2c_transposed_xy(thread_data, &output_values[ReturnZplane(blockDim.y, mem_offsets.physical_x_output)], output_MAP, gridDim.y);

    // For the other fragments we need the initial twiddle
    for ( int sub_fft = 1; sub_fft < Q - 1; sub_fft++ ) {
        io<FFT>::copy_from_shared(shared_input, thread_data, input_MAP);
        for ( int i = 0; i < FFT::elements_per_thread; i++ ) {
            // Pre shift with twiddle
            SINCOS(twiddle_factor_args[i] * sub_fft, &twiddle.y, &twiddle.x);
            thread_data[i] *= twiddle;
            // increment the output mapping.
            output_MAP[i]++;
        }
        FFT( ).execute(thread_data, shared_mem, workspace);

        io<FFT>::store_r2c_transposed_xy(thread_data, &output_values[ReturnZplane(blockDim.y, mem_offsets.physical_x_output)], output_MAP, gridDim.y);
    }

    // For the last fragment we need to also do a bounds check.
    io<FFT>::copy_from_shared(shared_input, thread_data, input_MAP);
    for ( int i = 0; i < FFT::elements_per_thread; i++ ) {
        // Pre shift with twiddle
        SINCOS(twiddle_factor_args[i] * (Q - 1), &twiddle.y, &twiddle.x);
        thread_data[i] *= twiddle;
        // increment the output mapping.
        output_MAP[i]++;
    }

    FFT( ).execute(thread_data, shared_mem, workspace);

    io<FFT>::store_r2c_transposed_xy(thread_data, &output_values[ReturnZplane(blockDim.y, mem_offsets.physical_x_output)], output_MAP, gridDim.y, mem_offsets.physical_x_output);
}

template <class FFT, class InputData_t, class OutputData_t>
__launch_bounds__(XZ_STRIDE* FFT::max_threads_per_block) __global__
        void block_fft_kernel_R2C_INCREASE_XZ(const InputData_t* __restrict__ input_values, OutputData_t* __restrict__ output_values, Offsets mem_offsets, float twiddle_in, int Q, typename FFT::workspace_type workspace) {

    // Initialize the shared memory, assuming everyting matches the input data X size in
    using complex_compute_t = typename FFT::value_type;
    using scalar_compute_t  = typename complex_compute_t::value_type;

    // The data store is non-coalesced, so don't aggregate the data in shared mem.
    extern __shared__ scalar_compute_t shared_input[];
    complex_compute_t*                 shared_mem = (complex_compute_t*)&shared_input[XZ_STRIDE * mem_offsets.shared_input];

    // Memory used by FFT
    complex_compute_t twiddle;
    complex_compute_t thread_data[FFT::storage_size];

    float twiddle_factor_args[FFT::storage_size];
    // Note: Q is used to calculate the strided output, which in this use, will end up being an offest in Z, so
    // we multiply by the NXY physical mem size of the OUTPUT array (which will be ZY') Then in the sub_fft loop, instead of adding one
    // we add NXY
    io<FFT>::load_r2c_shared(&input_values[Return1DFFTAddress_strided_Z(mem_offsets.physical_x_input)],
                             &shared_input[threadIdx.y * mem_offsets.shared_input],
                             thread_data, twiddle_factor_args, twiddle_in);

    FFT( ).execute(thread_data, &shared_mem[threadIdx.y * FFT::shared_memory_size / sizeof(complex_compute_t)], workspace);
    __syncthreads( );
    // Now we have a partial strided output due to the transform decomposition. In the 2D case we either write it out, or coalsece it in to shared memory
    // until we have the full output. Here, we are working on a tile, so we can transpose the data, and write it out partially coalesced.

    io<FFT>::transpose_r2c_in_shared_XZ(shared_mem, thread_data);
    io<FFT>::store_r2c_transposed_xz_strided_Z(shared_mem, output_values, Q, 0);

    // Now we need to loop over the remaining fragments.
    // For the other fragments we need the initial twiddle
    for ( int sub_fft = 1; sub_fft < Q; sub_fft++ ) {
        io<FFT>::copy_from_shared(&shared_input[threadIdx.y * mem_offsets.shared_input], thread_data);
        for ( int i = 0; i < FFT::elements_per_thread; i++ ) {
            // Pre shift with twiddle
            SINCOS(twiddle_factor_args[i] * sub_fft, &twiddle.y, &twiddle.x);
            thread_data[i] *= twiddle;
            // increment the output mapping.
        }

        FFT( ).execute(thread_data, &shared_mem[threadIdx.y * FFT::shared_memory_size / sizeof(complex_compute_t)], workspace); // FIXME the workspace is probably not going to work with the batched, look at the examples to see what to do.
        __syncthreads( );
        io<FFT>::transpose_r2c_in_shared_XZ(shared_mem, thread_data);
        io<FFT>::store_r2c_transposed_xz_strided_Z(shared_mem, output_values, Q, sub_fft);
    }

    // // For the last fragment we need to also do a bounds check. FIXME where does this happen
    // io<FFT>::copy_from_shared(&shared_input[threadIdx.y * mem_offsets.shared_input], thread_data);
    // for (int i = 0; i < FFT::elements_per_thread; i++) {
    //     // Pre shift with twiddle
    //     SINCOS(twiddle_factor_args[i]*(Q-1),&twiddle.y,&twiddle.x);
    //     thread_data[i] *= twiddle;
    //     // increment the output mapping.
    // }

    // FFT().execute(thread_data, &shared_mem[threadIdx.y * FFT::shared_memory_size/sizeof(complex_compute_t)], workspace); // FIXME the workspace is not setup for tiled approach
    // __syncthreads();
    // io<FFT>::transpose_r2c_in_shared_XZ(shared_mem, thread_data);
    // io<FFT>::store_r2c_transposed_xz_strided_Z(shared_mem, output_values, Q, 0);
}

// __launch_bounds__(FFT::max_threads_per_block)  we don't know this because it is threadDim.x * threadDim.z - this could be templated if it affects performance significantly
template <class FFT, class InputData_t, class OutputData_t>
__global__ void block_fft_kernel_R2C_DECREASE_XY(const InputData_t* __restrict__ input_values, OutputData_t* __restrict__ output_values, Offsets mem_offsets, float twiddle_in, int Q, typename FFT::workspace_type workspace) {
    // Initialize the shared memory, assuming everyting matches the input data X size in
    using complex_compute_t = typename FFT::value_type;
    using scalar_compute_t  = typename complex_compute_t::value_type;

    // The shared memory is used for storage, shuffling and fft ops at different stages and includes room for bank padding.
    extern __shared__ complex_compute_t shared_mem[];

    complex_compute_t thread_data[FFT::storage_size];

    // Load in natural order
    io<FFT>::load_r2c_shared_and_pad(&input_values[Return1DFFTAddress(mem_offsets.physical_x_input)], shared_mem);

    // DIT shuffle, bank conflict free
    io<FFT>::copy_from_shared(shared_mem, thread_data, Q);

    // The FFT operator has no idea we are using threadIdx.y to get multiple sub transforms, so we need to
    // segment the shared memory it accesses to avoid conflicts.
    constexpr const unsigned int fft_shared_mem_num_elements = FFT::shared_memory_size / sizeof(complex_compute_t);
    FFT( ).execute(thread_data, &shared_mem[fft_shared_mem_num_elements * threadIdx.y], workspace);
    __syncthreads( );

    // Full twiddle multiply and store in natural order in shared memory
    io<FFT>::reduce_block_fft(thread_data, shared_mem, twiddle_in, Q);

    // Reduce from shared memory into registers, ending up with only P valid outputs.
    io<FFT>::store_r2c_reduced(thread_data, &output_values[mem_offsets.physical_x_output * threadIdx.y], gridDim.y, mem_offsets.physical_x_output);
}

template <class ExternalImage_t, class FFT, class invFFT, class ComplexData_t>
__global__ void thread_fft_kernel_C2C_decomposed_ConjMul(const ExternalImage_t* __restrict__ image_to_search, const ComplexData_t* __restrict__ input_values, ComplexData_t* __restrict__ output_values, Offsets mem_offsets, float twiddle_in, int Q) {

    using complex_compute_t = typename FFT::value_type;
    using scalar_compute_t  = typename complex_compute_t::value_type;
    // The data store is non-coalesced, so don't aggregate the data in shared mem.
    extern __shared__ complex_compute_t shared_mem[];

    // Memory used by FFT - for Thread() type, FFT::storage_size == FFT::elements_per_thread == size_of<FFT>::value
    complex_compute_t thread_data[FFT::storage_size];

    io_thread<FFT>::load_c2c(&input_values[Return1DFFTAddress(size_of<FFT>::value) * Q], thread_data, Q);

    // We then have Q FFTs of size size_of<FFT>::value (P in the paper)
    FFT( ).execute(thread_data);

    // Now we need to aggregate each of the Q transforms into each output block of size P
    io_thread<FFT>::remap_decomposed_segments(thread_data, shared_mem, twiddle_in, Q, size_of<FFT>::value * Q);

#if FFT_DEBUG_STAGE > 3
    io_thread<invFFT>::load_shared_and_conj_multiply(&image_to_search[Return1DFFTAddress(size_of<FFT>::value * Q)], shared_mem, thread_data, Q);
#endif

#if FFT_DEBUG_STAGE > 4
    invFFT( ).execute(thread_data);
    // Now we need to aggregate each of the Q transforms into each output block of size P
    io_thread<invFFT>::remap_decomposed_segments(thread_data, shared_mem, -twiddle_in, Q, size_of<FFT>::value * Q);
#endif

    io_thread<invFFT>::store_c2c(shared_mem, &output_values[Return1DFFTAddress(size_of<FFT>::value * Q)], Q);
}

template <class ExternalImage_t, class FFT, class invFFT, class ComplexData_t>
__global__ void block_fft_kernel_C2C_FWD_NONE_INV_DECREASE_ConjMul(const ExternalImage_t* __restrict__ image_to_search, const ComplexData_t* __restrict__ input_values, ComplexData_t* __restrict__ output_values,
                                                                   Offsets mem_offsets, float twiddle_in, int apparent_Q, typename FFT::workspace_type workspace_fwd, typename invFFT::workspace_type workspace_inv) {

    using complex_compute_t = typename FFT::value_type;
    using scalar_compute_t  = typename complex_compute_t::value_type;

    extern __shared__ complex_compute_t shared_mem[];

    complex_compute_t thread_data[FFT::storage_size];

    // Load in natural order
    io<FFT>::load(&input_values[Return1DFFTAddress(size_of<FFT>::value)], thread_data);

    // io<FFT>::load_c2c_shared_and_pad(&input_values[Return1DFFTAddress(mem_offsets.physical_x_input)], shared_mem);

    // // DIT shuffle, bank conflict free
    // io<FFT>::copy_from_shared(shared_mem, thread_data, Q);

    // constexpr const unsigned int fft_shared_mem_num_elements = FFT::shared_memory_size / sizeof(complex_compute_t);
    // FFT().execute(thread_data, &shared_mem[fft_shared_mem_num_elements * threadIdx.y], workspace_fwd);
    // __syncthreads();
    FFT( ).execute(thread_data, shared_mem, workspace_fwd);

    // // Full twiddle multiply and store in natural order in shared memory
    // io<FFT>::reduce_block_fft(thread_data, shared_mem, twiddle_in, Q);

#if FFT_DEBUG_STAGE > 3
    // Load in imageFFT to search
    io<invFFT>::load_shared_and_conj_multiply(&image_to_search[Return1DFFTAddress(size_of<FFT>::value)], thread_data);
#endif

#if FFT_DEBUG_STAGE > 4
    // Run the inverse FFT
    // invFFT().execute(thread_data, &shared_mem[fft_shared_mem_num_elements * threadIdx.y], workspace_inv);
    invFFT( ).execute(thread_data, shared_mem, workspace_inv);

#endif

// // The reduced store considers threadIdx.y to ignore extra threads
// io<invFFT>::store_c2c_reduced(thread_data, &output_values[blockIdx.y * gridDim.y]);
#if FFT_DEBUG_STAGE < 5
    // There is no size reduction for this debug stage, so we need to use the pixel_pitch of the input array.
    io<invFFT>::store(thread_data, &output_values[Return1DFFTAddress(size_of<FFT>::value)]);
#else
    // In the current simplified version of the kernel, I am not using any transform decomposition (this is because of the difficulties with resrved threadIdx.x/y in the cufftdx lib)
    // So the full thing is calculated and only truncated on output.
    io<invFFT>::store(thread_data, &output_values[Return1DFFTAddress(size_of<FFT>::value / apparent_Q)], size_of<FFT>::value / apparent_Q);
#endif
}

// C2C

template <class FFT, class ComplexData_t>
__launch_bounds__(FFT::max_threads_per_block) __global__
        void block_fft_kernel_C2C_NONE(const ComplexData_t* __restrict__ input_values, ComplexData_t* __restrict__ output_values, Offsets mem_offsets, typename FFT::workspace_type workspace) {
    // Initialize the shared memory, assuming everyting matches the input data X size in
    using complex_compute_t = typename FFT::value_type;
    using scalar_compute_t  = typename complex_compute_t::value_type;

    extern __shared__ complex_compute_t shared_mem[]; // Storage for the input data that is re-used each blcok

    // Memory used by FFT
    complex_compute_t thread_data[FFT::storage_size];

    // No need to __syncthreads as each thread only accesses its own shared mem anyway
    io<FFT>::load(&input_values[Return1DFFTAddress(size_of<FFT>::value)], thread_data);

    // Since the memory ops are super straightforward this is an okay compromise.
    FFT( ).execute(thread_data, shared_mem, workspace);

    io<FFT>::store(thread_data, &output_values[Return1DFFTAddress(size_of<FFT>::value)]);
}

template <class FFT, class ComplexData_t>
__launch_bounds__(XZ_STRIDE* FFT::max_threads_per_block) __global__
        void block_fft_kernel_C2C_NONE_XZ(const ComplexData_t* __restrict__ input_values, ComplexData_t* __restrict__ output_values, Offsets mem_offsets, typename FFT::workspace_type workspace) {

    //	// Initialize the shared memory, assuming everyting matches the input data X size in
    using complex_compute_t = typename FFT::value_type;
    using scalar_compute_t  = typename complex_compute_t::value_type;

    extern __shared__ complex_compute_t shared_mem[]; // Storage for the input data that is re-used each blcok

    // Memory used by FFT
    complex_compute_t thread_data[FFT::storage_size];

    // No need to __syncthreads as each thread only accesses its own shared mem anyway
    io<FFT>::load(&input_values[Return1DFFTAddress_strided_Z(size_of<FFT>::value)], thread_data);

    FFT( ).execute(thread_data, &shared_mem[threadIdx.y * FFT::shared_memory_size / sizeof(complex_compute_t)], workspace);
    __syncthreads( );

    // Now we need to transpose in shared mem, fix bank conflicts later. TODO
    {
        const unsigned int stride = io<FFT>::stride_size( );
        unsigned int       index  = threadIdx.x;
        for ( unsigned int i = 0; i < FFT::elements_per_thread; i++ ) {
            // return (XZ_STRIDE*blockIdx.z + threadIdx.y) + (XZ_STRIDE*gridDim.z) * ( blockIdx.y + X * gridDim.y );
            // XZ_STRIDE == blockDim.z
            shared_mem[threadIdx.y + index * XZ_STRIDE] = thread_data[i];
            index += stride;
        }
    }
    __syncthreads( );

    // Transpose XZ, so the proper Z dimension now comes from X
    io<FFT>::store_transposed_xz_strided_Z(shared_mem, output_values);
}

// __launch_bounds__(FFT::max_threads_per_block)  we don't know this because it is threadDim.x * threadDim.z - this could be templated if it affects performance significantly
template <class FFT, class ComplexData_t>
__global__ void block_fft_kernel_C2C_DECREASE(const ComplexData_t* __restrict__ input_values, ComplexData_t* __restrict__ output_values, Offsets mem_offsets, float twiddle_in, int Q, typename FFT::workspace_type workspace) {
    //	// Initialize the shared memory, assuming everyting matches the input data X size in
    using complex_compute_t = typename FFT::value_type;
    using scalar_compute_t  = typename complex_compute_t::value_type;

    extern __shared__ complex_compute_t shared_mem[];

    complex_compute_t thread_data[FFT::storage_size];

    // Load in natural order
    io<FFT>::load_c2c_shared_and_pad(&input_values[Return1DFFTAddress(size_of<FFT>::value * Q)], shared_mem);

    // DIT shuffle, bank conflict free
    io<FFT>::copy_from_shared(shared_mem, thread_data, Q);

    constexpr const unsigned int fft_shared_mem_num_elements = FFT::shared_memory_size / sizeof(complex_compute_t);
    FFT( ).execute(thread_data, &shared_mem[fft_shared_mem_num_elements * threadIdx.y], workspace);
    __syncthreads( );

    // Full twiddle multiply and store in natural order in shared memory
    io<FFT>::reduce_block_fft(thread_data, shared_mem, twiddle_in, Q);

    // Reduce from shared memory into registers, ending up with only P valid outputs.
    io<FFT>::store_c2c_reduced(thread_data, &output_values[Return1DFFTAddress(size_of<FFT>::value)]);
}

// __launch_bounds__(FFT::max_threads_per_block)  we don't know this because it is threadDim.x * threadDim.z - this could be templated if it affects performance significantly
template <class FFT, class ComplexData_t>
__global__ void block_fft_kernel_C2C_INCREASE(const ComplexData_t* __restrict__ input_values, ComplexData_t* __restrict__ output_values, Offsets mem_offsets, float twiddle_in, int Q, typename FFT::workspace_type workspace) {
    // Initialize the shared memory, assuming everyting matches the input data X size in

    using complex_compute_t = typename FFT::value_type;
    using scalar_compute_t  = typename complex_compute_t::value_type;

    extern __shared__ complex_compute_t shared_input_complex[]; // Storage for the input data that is re-used each blcok
    complex_compute_t*                  shared_output = (complex_compute_t*)&shared_input_complex[mem_offsets.shared_input]; // storage for the coalesced output data. This may grow too large,
    complex_compute_t*                  shared_mem    = (complex_compute_t*)&shared_output[mem_offsets.shared_output];

    // Memory used by FFT
    complex_compute_t thread_data[FFT::storage_size];
    float             twiddle_factor_args[FFT::storage_size];
    complex_compute_t twiddle;

    // No need to __syncthreads as each thread only accesses its own shared mem anyway
    io<FFT>::load_shared(&input_values[Return1DFFTAddress(size_of<FFT>::value)], shared_input_complex, thread_data, twiddle_factor_args, twiddle_in);

    FFT( ).execute(thread_data, shared_mem, workspace);
    io<FFT>::store(thread_data, shared_output, Q, 0);

    // For the other fragments we need the initial twiddle
    for ( int sub_fft = 1; sub_fft < Q; sub_fft++ ) {
        io<FFT>::copy_from_shared(shared_input_complex, thread_data);

        for ( int i = 0; i < FFT::elements_per_thread; i++ ) {
            // Pre shift with twiddle
            SINCOS(twiddle_factor_args[i] * sub_fft, &twiddle.y, &twiddle.x);
            thread_data[i] *= twiddle;
        }
        FFT( ).execute(thread_data, shared_mem, workspace);
        io<FFT>::store(thread_data, shared_output, Q, sub_fft);
    }
    __syncthreads( );

    // Now that the memory output can be coalesced send to global
    // FIXME: is this actually coalced?
    for ( int sub_fft = 0; sub_fft < Q; sub_fft++ ) {
        io<FFT>::store_coalesced(shared_output, &output_values[Return1DFFTAddress(size_of<FFT>::value * Q)], sub_fft * mem_offsets.shared_input);
    }
}

template <class FFT, class InputData_t, class OutputData_t>
__launch_bounds__(FFT::max_threads_per_block) __global__
        void block_fft_kernel_C2C_INCREASE_SwapRealSpaceQuadrants(const InputData_t* __restrict__ input_values, OutputData_t* __restrict__ output_values, Offsets mem_offsets, float twiddle_in, int Q, typename FFT::workspace_type workspace) {

    //	// Initialize the shared memory, assuming everyting matches the input data X size in
    using complex_compute_t = typename FFT::value_type;
    using scalar_compute_t  = typename complex_compute_t::value_type;

    extern __shared__ complex_compute_t shared_input_complex[]; // Storage for the input data that is re-used each blcok
    complex_compute_t*                  shared_output = (complex_compute_t*)&shared_input_complex[mem_offsets.shared_input]; // storage for the coalesced output data. This may grow too large,
    complex_compute_t*                  shared_mem    = (complex_compute_t*)&shared_output[mem_offsets.shared_output];

    // Memory used by FFT
    complex_compute_t twiddle;
    complex_compute_t thread_data[FFT::storage_size];

    // To re-map the thread index to the data
    int input_MAP[FFT::storage_size];
    // To re-map the decomposed frequency to the full output frequency
    int output_MAP[FFT::storage_size];
    // For a given decomposed fragment
    float twiddle_factor_args[FFT::storage_size];

    // No need to __syncthreads as each thread only accesses its own shared mem anyway
    io<FFT>::load_shared(&input_values[Return1DFFTAddress(size_of<FFT>::value)], shared_input_complex, thread_data, twiddle_factor_args, twiddle_in, input_MAP, output_MAP, Q);

    // In the first FFT the modifying twiddle factor is 1 so the data are reeal
    FFT( ).execute(thread_data, shared_mem, workspace);

    // FIXME I have not confirmed on switch to physical_x_output that this represents the index of the first negative frequency in Y as it should.
    io<FFT>::store_and_swap_quadrants(thread_data, shared_output, output_MAP, size_of<FFT>::value * Q);

    // For the other fragments we need the initial twiddle
    for ( int sub_fft = 1; sub_fft < Q; sub_fft++ ) {

        io<FFT>::copy_from_shared(shared_input_complex, thread_data, input_MAP);

        for ( int i = 0; i < FFT::elements_per_thread; i++ ) {
            // Pre shift with twiddle
            SINCOS(twiddle_factor_args[i] * sub_fft, &twiddle.y, &twiddle.x);
            thread_data[i] *= twiddle;
            // increment the output map. Note this only works for the leading non-zero case
            output_MAP[i]++;
        }

        FFT( ).execute(thread_data, shared_mem, workspace);
        io<FFT>::store_and_swap_quadrants(thread_data, shared_output, output_MAP, size_of<FFT>::value * Q);
    }

    // TODO confirm this is needed
    __syncthreads( );

    // Now that the memory output can be coalesced send to global
    // FIXME is this actually coalced?
    for ( int sub_fft = 0; sub_fft < Q; sub_fft++ ) {
        io<FFT>::store_coalesced(shared_output, &output_values[Return1DFFTAddress(size_of<FFT>::value * Q)], sub_fft * mem_offsets.shared_input);
    }
}

template <class FFT, class ComplexData_t>
__global__ void thread_fft_kernel_C2C_decomposed(const ComplexData_t* __restrict__ input_values, ComplexData_t* __restrict__ output_values, Offsets mem_offsets, float twiddle_in, int Q) {

    using complex_compute_t = typename FFT::value_type;
    using scalar_compute_t  = typename complex_compute_t::value_type;

    // The data store is non-coalesced, so don't aggregate the data in shared mem.
    extern __shared__ complex_compute_t shared_mem[];

    // Memory used by FFT - for Thread() type, FFT::storage_size == FFT::elements_per_thread == size_of<FFT>::value
    complex_compute_t thread_data[FFT::storage_size];

    io_thread<FFT>::load_c2c(&input_values[Return1DFFTAddress(size_of<FFT>::value)], thread_data, Q);

    // We then have Q FFTs of size size_of<FFT>::value (P in the paper)
    FFT( ).execute(thread_data);

    // Now we need to aggregate each of the Q transforms into each output block of size P
    io_thread<FFT>::remap_decomposed_segments(thread_data, shared_mem, twiddle_in, Q, size_of<FFT>::value * Q);

    io_thread<FFT>::store_c2c(shared_mem, &output_values[Return1DFFTAddress(size_of<FFT>::value * Q)], Q);
}

template <class FFT, class ComplexData_t>
__launch_bounds__(XZ_STRIDE* FFT::max_threads_per_block) __global__
        void block_fft_kernel_C2C_NONE_XYZ(const ComplexData_t* __restrict__ input_values, ComplexData_t* __restrict__ output_values, Offsets mem_offsets, typename FFT::workspace_type workspace) {

    //	// Initialize the shared memory, assuming everyting matches the input data X size in
    using complex_compute_t = typename FFT::value_type;
    using scalar_compute_t  = typename complex_compute_t::value_type;

    extern __shared__ complex_compute_t shared_mem[]; // Storage for the input data that is re-used each blcok

    // Memory used by FFT
    complex_compute_t thread_data[FFT::storage_size];

    // No need to __syncthreads as each thread only accesses its own shared mem anyway
    io<FFT>::load(&input_values[Return1DFFTColumn_XYZ_transpose(size_of<FFT>::value)], thread_data);

    FFT( ).execute(thread_data, &shared_mem[threadIdx.y * FFT::shared_memory_size / sizeof(complex_compute_t)], workspace);
    __syncthreads( );

    io<FFT>::transpose_in_shared_XZ(shared_mem, thread_data);

    io<FFT>::store_Z(shared_mem, output_values);
}

template <class FFT, class ComplexData_t>
__launch_bounds__(XZ_STRIDE* FFT::max_threads_per_block) __global__
        void block_fft_kernel_C2C_INCREASE_XYZ(const ComplexData_t* __restrict__ input_values, ComplexData_t* __restrict__ output_values, Offsets mem_offsets, float twiddle_in, int Q, typename FFT::workspace_type workspace) {

    using complex_compute_t = typename FFT::value_type;
    using scalar_compute_t  = typename complex_compute_t::value_type;

    extern __shared__ complex_compute_t shared_input_complex[]; // Storage for the input data that is re-used each blcok
    complex_compute_t*                  shared_mem = (complex_compute_t*)&shared_input_complex[XZ_STRIDE * mem_offsets.shared_input]; // storage for computation and transposition (alternating)

    // Memory used by FFT
    complex_compute_t thread_data[FFT::storage_size];
    complex_compute_t twiddle;
    float             twiddle_factor_args[FFT::storage_size];

    // No need to __syncthreads as each thread only accesses its own shared mem anyway
    io<FFT>::load_shared(&input_values[Return1DFFTColumn_XYZ_transpose(size_of<FFT>::value)],
                         &shared_input_complex[threadIdx.y * mem_offsets.shared_input], thread_data, twiddle_factor_args, twiddle_in);

    FFT( ).execute(thread_data, &shared_mem[threadIdx.y * FFT::shared_memory_size / sizeof(complex_compute_t)], workspace);
    __syncthreads( );

    io<FFT>::transpose_in_shared_XZ(shared_mem, thread_data);
    io<FFT>::store_Z(shared_mem, output_values, Q, 0);

    // For the other fragments we need the initial twiddle
    for ( int sub_fft = 1; sub_fft < Q; sub_fft++ ) {
        io<FFT>::copy_from_shared(&shared_input_complex[threadIdx.y * mem_offsets.shared_input], thread_data);
        for ( int i = 0; i < FFT::elements_per_thread; i++ ) {
            // Pre shift with twiddle
            SINCOS(twiddle_factor_args[i] * sub_fft, &twiddle.y, &twiddle.x);
            thread_data[i] *= twiddle;
        }
        FFT( ).execute(thread_data, &shared_mem[threadIdx.y * FFT::shared_memory_size / sizeof(complex_compute_t)], workspace);
        io<FFT>::transpose_in_shared_XZ(shared_mem, thread_data);
        io<FFT>::store_Z(shared_mem, output_values, Q, sub_fft);
    }
}

template <class FFT, class InputData_t, class OutputData_t>
__launch_bounds__(FFT::max_threads_per_block) __global__
        void block_fft_kernel_C2R_NONE(const InputData_t* __restrict__ input_values, OutputData_t* __restrict__ output_values, Offsets mem_offsets, typename FFT::workspace_type workspace) {

    using complex_compute_t = typename FFT::value_type;
    using scalar_compute_t  = typename complex_compute_t::value_type;

    extern __shared__ complex_compute_t shared_mem[];

    complex_compute_t thread_data[FFT::storage_size];

    io<FFT>::load_c2r(&input_values[Return1DFFTAddress(mem_offsets.physical_x_input)], thread_data);

    // For loop zero the twiddles don't need to be computed
    FFT( ).execute(thread_data, shared_mem, workspace);

    io<FFT>::store_c2r(thread_data, &output_values[Return1DFFTAddress(mem_offsets.physical_x_output)], size_of<FFT>::value);
}

template <class FFT, class InputData_t, class OutputData_t>
__launch_bounds__(FFT::max_threads_per_block) __global__
        void block_fft_kernel_C2R_NONE_XY(const InputData_t* __restrict__ input_values, OutputData_t* __restrict__ output_values, Offsets mem_offsets, typename FFT::workspace_type workspace) {

    using complex_compute_t = typename FFT::value_type;
    using scalar_compute_t  = typename complex_compute_t::value_type;

    extern __shared__ complex_compute_t shared_mem[];

    complex_compute_t thread_data[FFT::storage_size];

    io<FFT>::load_c2r_transposed(&input_values[ReturnZplane(gridDim.y, mem_offsets.physical_x_input)], thread_data, gridDim.y);

    // For loop zero the twiddles don't need to be computed
    FFT( ).execute(thread_data, shared_mem, workspace);

    io<FFT>::store_c2r(thread_data, &output_values[Return1DFFTAddress(mem_offsets.physical_x_output)], size_of<FFT>::value);
}

template <class FFT, class InputData_t, class OutputData_t>
__global__ void block_fft_kernel_C2R_DECREASE_XY(const InputData_t* __restrict__ input_values, OutputData_t* __restrict__ output_values, Offsets mem_offsets, const float twiddle_in, const unsigned int Q, typename FFT::workspace_type workspace) {

    using complex_compute_t = typename FFT::value_type;
    using scalar_compute_t  = typename complex_compute_t::value_type;

    extern __shared__ complex_compute_t shared_mem[];

    complex_compute_t thread_data[FFT::storage_size];

    io<FFT>::load_c2r_transposed(&input_values[ReturnZplane(gridDim.y, mem_offsets.physical_x_input)], thread_data, gridDim.y);

    // For loop zero the twiddles don't need to be computed
    FFT( ).execute(thread_data, shared_mem, workspace);

    io<FFT>::store_c2r(thread_data, &output_values[Return1DFFTAddress(mem_offsets.physical_x_output)], size_of<FFT>::value);

    // // Load transposed data into shared memory in natural order.
    // io<FFT>::load_c2r_shared_and_pad(&input_values[blockIdx.y], shared_mem, mem_offsets.physical_x_input);

    // // DIT shuffle, bank conflict free
    // io<FFT>::copy_from_shared(shared_mem, thread_data, Q);

    // constexpr const unsigned int fft_shared_mem_num_elements = FFT::shared_memory_size / sizeof(complex_compute_t);
    // FFT().execute(thread_data, &shared_mem[fft_shared_mem_num_elements * threadIdx.y], workspace);
    // __syncthreads();

    // // Full twiddle multiply and store in natural order in shared memory
    // io<FFT>::reduce_block_fft(thread_data, shared_mem, twiddle_in, Q);

    // // Reduce from shared memory into registers, ending up with only P valid outputs.
    // io<FFT>::store_c2r_reduced(thread_data, &output_values[Return1DFFTAddress(mem_offsets.physical_x_output)]);
}

template <class FFT, class InputData_t, class OutputData_t>
__global__ void thread_fft_kernel_C2R_decomposed(const InputData_t* __restrict__ input_values, OutputData_t* __restrict__ output_values, Offsets mem_offsets, float twiddle_in, int Q) {
    using complex_compute_t = typename FFT::value_type;
    using scalar_compute_t  = typename complex_compute_t::value_type;
    ;

    // The data store is non-coalesced, so don't aggregate the data in shared mem.
    extern __shared__ scalar_compute_t shared_mem_C2R_decomposed[];

    // Memory used by FFT - for Thread() type, FFT::storage_size == FFT::elements_per_thread == size_of<FFT>::value
    complex_compute_t thread_data[FFT::storage_size];

    io_thread<FFT>::load_c2r(&input_values[Return1DFFTAddress(mem_offsets.physical_x_input)], thread_data, Q, mem_offsets.physical_x_input);

    // We then have Q FFTs of size size_of<FFT>::value (P in the paper)
    FFT( ).execute(thread_data);

    // Now we need to aggregate each of the Q transforms into each output block of size P
    io_thread<FFT>::remap_decomposed_segments_c2r(thread_data, shared_mem_C2R_decomposed, twiddle_in, Q);

    io_thread<FFT>::store_c2r(shared_mem_C2R_decomposed, &output_values[Return1DFFTAddress(mem_offsets.physical_x_output)], Q);
}

template <class FFT, class InputData_t, class OutputData_t>
__global__ void thread_fft_kernel_C2R_decomposed_transposed(const InputData_t* __restrict__ input_values, OutputData_t* __restrict__ output_values, Offsets mem_offsets, float twiddle_in, int Q) {

    using complex_compute_t = typename FFT::value_type;
    using scalar_compute_t  = typename complex_compute_t::value_type;

    // The data store is non-coalesced, so don't aggregate the data in shared mem.
    extern __shared__ scalar_compute_t shared_mem_transposed[];

    // Memory used by FFT - for Thread() type, FFT::storage_size == FFT::elements_per_thread == size_of<FFT>::value
    complex_compute_t thread_data[FFT::storage_size];

    io_thread<FFT>::load_c2r_transposed(&input_values[ReturnZplane(blockDim.y, mem_offsets.physical_x_input)], thread_data, Q, gridDim.y, mem_offsets.physical_x_input);

    // We then have Q FFTs of size size_of<FFT>::value (P in the paper)
    FFT( ).execute(thread_data);

    // Now we need to aggregate each of the Q transforms into each output block of size P
    io_thread<FFT>::remap_decomposed_segments_c2r(thread_data, shared_mem_transposed, twiddle_in, Q);

    io_thread<FFT>::store_c2r(shared_mem_transposed, &output_values[Return1DFFTAddress(mem_offsets.physical_x_output)], Q);
}

// FIXME assumed FWD
template <class InputType, class OtherImageType>
__global__ void clip_into_top_left_kernel(InputType* input_values, InputType* output_values, const short4 dims) {

    int x = blockIdx.x * blockDim.x + threadIdx.x;
    if ( x > dims.w )
        return; // Out of bounds.

    // dims.w is the pitch of the output array
    if ( blockIdx.y > dims.y ) {
        output_values[blockIdx.y * dims.w + x] = OtherImageType(0);
        return;
    }

    if ( threadIdx.x > dims.x ) {
        output_values[blockIdx.y * dims.w + x] = OtherImageType(0);
        return;
    }
    else {
        // dims.z is the pitch of the output array
        output_values[blockIdx.y * dims.w + x] = input_values[blockIdx.y * dims.z + x];
        return;
    }
}

template <class ComputeBaseType, class InputType, class OtherImageType, int Rank>
void FourierTransformer<ComputeBaseType, InputType, OtherImageType, Rank>::ClipIntoTopLeft(InputType* input_ptr) {
    // TODO add some checks and logic.

    // Assuming we are calling this from R2C_Transposed and that the launch bounds are not set.
    dim3 local_threadsPerBlock = dim3(512, 1, 1);
    dim3 local_gridDims        = dim3((fwd_dims_out.x + local_threadsPerBlock.x - 1) / local_threadsPerBlock.x, 1, 1);

    const short4 area_to_clip_from = make_short4(fwd_dims_in.x, fwd_dims_in.y, fwd_dims_in.w * 2, fwd_dims_out.w * 2);

    precheck;
    clip_into_top_left_kernel<InputType, InputType><<<local_gridDims, local_threadsPerBlock, 0, cudaStreamPerThread>>>(input_ptr, (InputType*)d_ptr.buffer_1, area_to_clip_from);
    postcheck;
    current_buffer = fastfft_internal_buffer_1;
}

// Modified from GpuImage::ClipIntoRealKernel
template <typename InputType, typename OtherImageType>
__global__ void clip_into_real_kernel(InputType* real_values_gpu,
                                      InputType* other_image_real_values_gpu,
                                      short4     dims,
                                      short4     other_dims,
                                      int3       wanted_coordinate_of_box_center,
                                      InputType  wanted_padding_value) {
    int3 other_coord = make_int3(blockIdx.x * blockDim.x + threadIdx.x,
                                 blockIdx.y * gridDim.y + threadIdx.y,
                                 blockIdx.z);

    int3 coord = make_int3(0, 0, 0);

    if ( other_coord.x < other_dims.x &&
         other_coord.y < other_dims.y &&
         other_coord.z < other_dims.z ) {

        coord.z = dims.z / 2 + wanted_coordinate_of_box_center.z +
                  other_coord.z - other_dims.z / 2;

        coord.y = dims.y / 2 + wanted_coordinate_of_box_center.y +
                  other_coord.y - other_dims.y / 2;

        coord.x = dims.x + wanted_coordinate_of_box_center.x +
                  other_coord.x - other_dims.x;

        if ( coord.z < 0 || coord.z >= dims.z ||
             coord.y < 0 || coord.y >= dims.y ||
             coord.x < 0 || coord.x >= dims.x ) {
            other_image_real_values_gpu[d_ReturnReal1DAddressFromPhysicalCoord(other_coord, other_dims)] = wanted_padding_value;
        }
        else {
            other_image_real_values_gpu[d_ReturnReal1DAddressFromPhysicalCoord(other_coord, other_dims)] =
                    real_values_gpu[d_ReturnReal1DAddressFromPhysicalCoord(coord, dims)];
        }

    } // end of bounds check
}

template <class ComputeBaseType, class InputType, class OtherImageType, int Rank>
void FourierTransformer<ComputeBaseType, InputType, OtherImageType, Rank>::ClipIntoReal(InputType* input_ptr, int wanted_coordinate_of_box_center_x, int wanted_coordinate_of_box_center_y, int wanted_coordinate_of_box_center_z) {
    // TODO add some checks and logic.

    // Assuming we are calling this from R2C_Transposed and that the launch bounds are not set.
    dim3 threadsPerBlock;
    dim3 gridDims;
    int3 wanted_center = make_int3(wanted_coordinate_of_box_center_x, wanted_coordinate_of_box_center_y, wanted_coordinate_of_box_center_z);
    threadsPerBlock    = dim3(32, 32, 1);
    gridDims           = dim3((fwd_dims_out.x + threadsPerBlock.x - 1) / threadsPerBlock.x,
                              (fwd_dims_out.y + threadsPerBlock.y - 1) / threadsPerBlock.y,
                              1);

    const short4 area_to_clip_from    = make_short4(fwd_dims_in.x, fwd_dims_in.y, fwd_dims_in.w * 2, fwd_dims_out.w * 2);
    float        wanted_padding_value = 0.f;

    precheck;
    clip_into_real_kernel<InputType, InputType><<<gridDims, threadsPerBlock, 0, cudaStreamPerThread>>>(input_ptr, (InputType*)d_ptr.buffer_1, fwd_dims_in, fwd_dims_out, wanted_center, wanted_padding_value);
    postcheck;
    current_buffer = fastfft_internal_buffer_1;
}

template <class ComputeBaseType, class InputType, class OtherImageType, int Rank>
template <int FFT_ALGO_t, bool use_thread_method, class PreOpType, class IntraOpType, class PostOpType>
EnableIf<IfAppliesIntraOpFunctor_HasIntraOpFunctor<IntraOpType, FFT_ALGO_t>>
FourierTransformer<ComputeBaseType, InputType, OtherImageType, Rank>::SetPrecisionAndExectutionMethod(OtherImageType* other_image_ptr,
                                                                                                      KernelType      kernel_type,
                                                                                                      PreOpType       pre_op_functor,
                                                                                                      IntraOpType     intra_op_functor,
                                                                                                      PostOpType      post_op_functor) {
    // For kernels with fwd and inv transforms, we want to not set the direction yet.

    static const bool is_half  = std::is_same_v<ComputeBaseType, __half>; // FIXME: This should be done in the constructor
    static const bool is_float = std::is_same_v<ComputeBaseType, float>;
    static_assert(is_half || is_float, "FourierTransformer::SetPrecisionAndExectutionMethod: Unsupported ComputeBaseType");
    if constexpr ( FFT_ALGO_t == Generic_Fwd_Image_Inv_FFT ) {
        static_assert(IS_IKF_t<IntraOpType>( ), "FourierTransformer::SetPrecisionAndExectutionMethod: Unsupported IntraOpType");
    }

    if constexpr ( use_thread_method ) {
        using FFT = decltype(Thread( ) + Size<32>( ) + Precision<ComputeBaseType>( ));
        SetIntraKernelFunctions<FFT_ALGO_t, FFT, PreOpType, IntraOpType, PostOpType>(other_image_ptr, kernel_type, pre_op_functor, intra_op_functor, post_op_functor);
    }
    else {
        using FFT = decltype(Block( ) + Precision<ComputeBaseType>( ) + FFTsPerBlock<1>( ));
        SetIntraKernelFunctions<FFT_ALGO_t, FFT, PreOpType, IntraOpType, PostOpType>(other_image_ptr, kernel_type, pre_op_functor, intra_op_functor, post_op_functor);
    }
}

template <class ComputeBaseType, class InputType, class OtherImageType, int Rank>
template <int FFT_ALGO_t, class FFT_base, class PreOpType, class IntraOpType, class PostOpType>
void FourierTransformer<ComputeBaseType, InputType, OtherImageType, Rank>::SetIntraKernelFunctions(OtherImageType* other_image_ptr,
                                                                                                   KernelType      kernel_type,
                                                                                                   PreOpType       pre_op_functor,
                                                                                                   IntraOpType     intra_op_functor,
                                                                                                   PostOpType      post_op_functor) {

    if constexpr ( ! detail::has_any_block_operator<FFT_base>::value ) {
        // SelectSizeAndType<FFT, PreOpType, IntraOpType, PostOpType>(kernel_type,  pre_op_functor, intra_op_functor, post_op_functor);
    }
    else {
        if constexpr ( Rank == 3 ) {
            SelectSizeAndType<FFT_ALGO_t, FFT_base, PreOpType, IntraOpType, PostOpType, 16, 4, 32, 8, 64, 8, 128, 8, 256, 8, 512, 8>(other_image_ptr, kernel_type, pre_op_functor, intra_op_functor, post_op_functor);
        }
        else {
            // TODO: 8192 will fail for sm75 if wanted need some extra logic ... , 8192, 16
            SelectSizeAndType<FFT_ALGO_t, FFT_base, PreOpType, IntraOpType, PostOpType, 16, 4, 32, 8, 64, 8, 128, 8, 256, 8, 512, 8, 1024, 8, 2048, 8, 4096, 16>(other_image_ptr, kernel_type, pre_op_functor, intra_op_functor, post_op_functor);
        }
    }
}

template <class ComputeBaseType, class InputType, class OtherImageType, int Rank>
template <int FFT_ALGO_t, class FFT_base, class PreOpType, class IntraOpType, class PostOpType>
void FourierTransformer<ComputeBaseType, InputType, OtherImageType, Rank>::SelectSizeAndType(OtherImageType* other_image_ptr,
                                                                                             KernelType      kernel_type,
                                                                                             PreOpType       pre_op_functor,
                                                                                             IntraOpType     intra_op_functor,
                                                                                             PostOpType      post_op_functor) {
    // This provides both a termination point for the recursive version needed for the block transform case as well as the actual function for thread transform with fixed size 32
    GetTransformSize(kernel_type);
    if constexpr ( ! detail::has_any_block_operator<FFT_base>::value ) {
        SetEptForUseInLaunchParameters(8);
        switch ( device_properties.device_arch ) {
            case 700: {
                using FFT = decltype(FFT_base( ) + SM<700>( ) + ElementsPerThread<8>( ));
                SetAndLaunchKernel<FFT_ALGO_t, FFT, PreOpType, IntraOpType, PostOpType>(other_image_ptr, kernel_type, pre_op_functor, intra_op_functor, post_op_functor);
                break;
            }
            case 750: {
                using FFT = decltype(FFT_base( ) + SM<750>( ) + ElementsPerThread<8>( ));
                SetAndLaunchKernel<FFT_ALGO_t, FFT, PreOpType, IntraOpType, PostOpType>(other_image_ptr, kernel_type, pre_op_functor, intra_op_functor, post_op_functor);
                break;
            }
            case 800: {
                using FFT = decltype(FFT_base( ) + SM<800>( ) + ElementsPerThread<8>( ));
                SetAndLaunchKernel<FFT_ALGO_t, FFT, PreOpType, IntraOpType, PostOpType>(other_image_ptr, kernel_type, pre_op_functor, intra_op_functor, post_op_functor);
                break;
            }
            case 860: {
                using FFT = decltype(FFT_base( ) + SM<700>( ) + ElementsPerThread<8>( ));
                SetAndLaunchKernel<FFT_ALGO_t, FFT, PreOpType, IntraOpType, PostOpType>(other_image_ptr, kernel_type, pre_op_functor, intra_op_functor, post_op_functor);
                break;
            }
            case 890: {
                // FIXME: on migrating to cufftDx 1.1.1
                using FFT = decltype(FFT_base( ) + SM<700>( ) + ElementsPerThread<8>( ));
                SetAndLaunchKernel<FFT_ALGO_t, FFT, PreOpType, IntraOpType, PostOpType>(other_image_ptr, kernel_type, pre_op_functor, intra_op_functor, post_op_functor);
                break;
            }
            default: {
                MyFFTRunTimeAssertTrue(false, "Unsupported architecture" + std::to_string(device_properties.device_arch));
                break;
            }
        }
    }
}

// TODO: see if you can replace this with a fold expression?
template <class ComputeBaseType, class InputType, class OtherImageType, int Rank>
template <int FFT_ALGO_t, class FFT_base, class PreOpType, class IntraOpType, class PostOpType, unsigned int SizeValue, unsigned int Ept, unsigned int... OtherValues>
void FourierTransformer<ComputeBaseType, InputType, OtherImageType, Rank>::SelectSizeAndType(OtherImageType* other_image_ptr,
                                                                                             KernelType      kernel_type,
                                                                                             PreOpType       pre_op_functor,
                                                                                             IntraOpType     intra_op_functor,
                                                                                             PostOpType      post_op_functor) {
    // Use recursion to step through the allowed sizes.
    GetTransformSize(kernel_type);

    // Note: the size of the input/output may not match the size of the transform, i.e. transform_size.L <= transform_size.P
    if ( SizeValue == transform_size.P ) {
        switch ( device_properties.device_arch ) {
            case 700: {
                SetEptForUseInLaunchParameters(Ept);
                using FFT = decltype(FFT_base( ) + Size<SizeValue>( ) + SM<700>( ) + ElementsPerThread<Ept>( ));
                SetAndLaunchKernel<FFT_ALGO_t, FFT, PreOpType, IntraOpType, PostOpType>(other_image_ptr, kernel_type, pre_op_functor, intra_op_functor, post_op_functor);
                break;
            }
            case 750: {
                SetEptForUseInLaunchParameters(Ept);
                if constexpr ( SizeValue <= 4096 ) {
                    using FFT = decltype(FFT_base( ) + Size<SizeValue>( ) + SM<750>( ) + ElementsPerThread<Ept>( ));
                    SetAndLaunchKernel<FFT_ALGO_t, FFT, PreOpType, IntraOpType, PostOpType>(other_image_ptr, kernel_type, pre_op_functor, intra_op_functor, post_op_functor);
                }
                break;
            }
            case 800: {
                SetEptForUseInLaunchParameters(Ept);
                using FFT = decltype(FFT_base( ) + Size<SizeValue>( ) + SM<800>( ) + ElementsPerThread<Ept>( ));
                SetAndLaunchKernel<FFT_ALGO_t, FFT, PreOpType, IntraOpType, PostOpType>(other_image_ptr, kernel_type, pre_op_functor, intra_op_functor, post_op_functor);
                break;
            }
            case 860: {
                // TODO: confirm that this is needed (over 860) which at the time just redirects to 700
                //       if maintining this, we could save some time on compilation by combining with the 700 case
                SetEptForUseInLaunchParameters(Ept);
                using FFT = decltype(FFT_base( ) + Size<SizeValue>( ) + SM<700>( ) + ElementsPerThread<Ept>( ));
                SetAndLaunchKernel<FFT_ALGO_t, FFT, PreOpType, IntraOpType, PostOpType>(other_image_ptr, kernel_type, pre_op_functor, intra_op_functor, post_op_functor);
                break;
            }
            case 890: {
                // TODO: confirm that this is needed (over 860) which at the time just redirects to 700
                //       if maintining this, we could save some time on compilation by combining with the 700 case
                // FIXME: on migrating to cufftDx 1.1.1

                SetEptForUseInLaunchParameters(Ept);
                using FFT = decltype(FFT_base( ) + Size<SizeValue>( ) + SM<700>( ) + ElementsPerThread<Ept>( ));
                SetAndLaunchKernel<FFT_ALGO_t, FFT, PreOpType, IntraOpType, PostOpType>(other_image_ptr, kernel_type, pre_op_functor, intra_op_functor, post_op_functor);
                break;
            }
            default: {
                MyFFTRunTimeAssertTrue(false, "Unsupported architecture" + std::to_string(device_properties.device_arch));
                break;
            }
        }
    }

    SelectSizeAndType<FFT_ALGO_t, FFT_base, PreOpType, IntraOpType, PostOpType, OtherValues...>(other_image_ptr, kernel_type, pre_op_functor, intra_op_functor, post_op_functor);
}

template <class ComputeBaseType, class InputType, class OtherImageType, int Rank>
template <int FFT_ALGO_t, class FFT_base_arch, class PreOpType, class IntraOpType, class PostOpType>
void FourierTransformer<ComputeBaseType, InputType, OtherImageType, Rank>::SetAndLaunchKernel(OtherImageType* other_image_ptr,
                                                                                              KernelType      kernel_type,
                                                                                              PreOpType       pre_op_functor,
                                                                                              IntraOpType     intra_op_functor,
                                                                                              PostOpType      post_op_functor) {

    // Used to determine shared memory requirements
    using complex_compute_t = typename FFT_base_arch::value_type;
    using scalar_compute_t  = typename complex_compute_t::value_type;
    // Determined by InputType as complex version, i.e., half half2 or float float2
    using data_buffer_t = std::remove_pointer_t<decltype(d_ptr.buffer_1)>;
    // Allowed half, float (real type image) half2 float2 (complex type image) so with typical case
    // as real valued image, data_io_t != data_buffer_t
    using data_io_t = std::remove_pointer_t<decltype(d_ptr.external_input)>;
    // Could match data_io_t, but need not, will be converted in kernels to match complex_compute_t as needed.
    using external_image_t = OtherImageType;

    // If the user passed in a different exerternal pointer for the image, set it here, and if not,
    // we just alias the external input
    data_io_t*      external_output_ptr;
    buffer_location aliased_output_buffer;
    if ( d_ptr.external_output != nullptr ) {
        external_output_ptr   = d_ptr.external_output;
        aliased_output_buffer = fastfft_external_output;
    }
    else {
        external_output_ptr   = d_ptr.external_input;
        aliased_output_buffer = fastfft_external_input;
    }
    // if constexpr (detail::is_operator<fft_operator::thread, FFT_base_arch>::value) {
    if constexpr ( ! detail::has_any_block_operator<FFT_base_arch>::value ) {
        MyFFTRunTimeAssertFalse(true, "thread_fft_kernel is currently broken");
        switch ( kernel_type ) {

            case r2c_decomposed: {
                if constexpr ( FFT_ALGO_t == Generic_Fwd_FFT ) {
                    using FFT = decltype(FFT_base_arch( ) + Direction<fft_direction::forward>( ) + Type<fft_type::c2c>( ));

                    LaunchParams LP = SetLaunchParameters(r2c_decomposed);

                    int shared_mem = LP.mem_offsets.shared_output * sizeof(complex_compute_t);
                    CheckSharedMemory(shared_mem, device_properties);
#if FFT_DEBUG_STAGE > 0
                    if constexpr ( Rank == 1 ) {
                        MyFFTDebugAssertTrue(current_buffer == fastfft_external_input, "current_buffer != fastfft_external_input");
                        cudaErr(cudaFuncSetAttribute((void*)thread_fft_kernel_R2C_decomposed<FFT, data_io_t, data_buffer_t>, cudaFuncAttributeMaxDynamicSharedMemorySize, shared_mem));
                        precheck;
                        thread_fft_kernel_R2C_decomposed<FFT, data_io_t, data_buffer_t><<<LP.gridDims, LP.threadsPerBlock, shared_mem, cudaStreamPerThread>>>(
                                d_ptr.external_input, reinterpret_cast<data_buffer_t*>(external_output_ptr), LP.mem_offsets, LP.twiddle_in, LP.Q);
                        postcheck;
                        current_buffer = aliased_output_buffer;
                    }
#endif
                }

                break;
            }

            case r2c_decomposed_transposed: {
                if constexpr ( FFT_ALGO_t == Generic_Fwd_FFT ) {
                    using FFT = decltype(FFT_base_arch( ) + Direction<fft_direction::forward>( ) + Type<fft_type::c2c>( ));

                    LaunchParams LP = SetLaunchParameters(r2c_decomposed_transposed);

                    int shared_mem = LP.mem_offsets.shared_output * sizeof(complex_compute_t);
                    CheckSharedMemory(shared_mem, device_properties);
#if FFT_DEBUG_STAGE > 0
                    if constexpr ( Rank == 2 ) {
                        MyFFTDebugAssertTrue(current_buffer == fastfft_external_input, "current_buffer != fastfft_external_input");
                        cudaErr(cudaFuncSetAttribute((void*)thread_fft_kernel_R2C_decomposed_transposed<FFT, data_io_t, data_buffer_t>, cudaFuncAttributeMaxDynamicSharedMemorySize, shared_mem));
                        precheck;
                        thread_fft_kernel_R2C_decomposed_transposed<FFT, data_io_t, data_buffer_t><<<LP.gridDims, LP.threadsPerBlock, shared_mem, cudaStreamPerThread>>>(
                                d_ptr.external_input, d_ptr.buffer_1, LP.mem_offsets, LP.twiddle_in, LP.Q);
                        postcheck;
                        current_buffer = fastfft_internal_buffer_1;
                    }
#endif
                }
                break;
            }

            case c2r_decomposed: {
                if constexpr ( FFT_ALGO_t == Generic_Inv_FFT ) {
                    // Note that unlike the block C2R we require a C2C sub xform.
                    using FFT = decltype(FFT_base_arch( ) + Direction<fft_direction::inverse>( ) + Type<fft_type::c2c>( ));
                    // TODO add completeness check.

                    LaunchParams LP            = SetLaunchParameters(c2r_decomposed);
                    int          shared_memory = LP.mem_offsets.shared_output * sizeof(scalar_compute_t);
                    CheckSharedMemory(shared_memory, device_properties);
#if FFT_DEBUG_STAGE > 6
                    if constexpr ( Rank == 1 ) {
                        cudaErr(cudaFuncSetAttribute((void*)thread_fft_kernel_C2R_decomposed<FFT, data_io_t, data_buffer_t>, cudaFuncAttributeMaxDynamicSharedMemorySize, shared_memory));
                        precheck;
                        thread_fft_kernel_C2R_decomposed<FFT, data_io_t, data_buffer_t><<<LP.gridDims, LP.threadsPerBlock, shared_memory, cudaStreamPerThread>>>(
                                reinterpret_cast<data_buffer_t*>(d_ptr.external_input), external_output_ptr, LP.mem_offsets, LP.twiddle_in, LP.Q);
                        postcheck;
                        current_buffer = aliased_output_buffer;
                    }
#endif
                }
                break;
            }

            case c2r_decomposed_transposed: {
                if constexpr ( FFT_ALGO_t == Generic_Inv_FFT ) {
                    // Note that unlike the block C2R we require a C2C sub xform.
                    using FFT = decltype(FFT_base_arch( ) + Direction<fft_direction::inverse>( ) + Type<fft_type::c2c>( ));

                    LaunchParams LP            = SetLaunchParameters(c2r_decomposed_transposed);
                    int          shared_memory = LP.mem_offsets.shared_output * sizeof(scalar_compute_t);
                    CheckSharedMemory(shared_memory, device_properties);
#if FFT_DEBUG_STAGE > 6
                    cudaErr(cudaFuncSetAttribute((void*)thread_fft_kernel_C2R_decomposed_transposed<FFT, data_io_t, data_buffer_t>, cudaFuncAttributeMaxDynamicSharedMemorySize, shared_memory));
                    if constexpr ( Rank == 2 ) {
                        MyFFTDebugAssertTrue(current_buffer == fastfft_internal_buffer_1, "current_buffer != fastfft_internal_buffer_1");
                        precheck;
                        thread_fft_kernel_C2R_decomposed_transposed<FFT, data_io_t, data_buffer_t><<<LP.gridDims, LP.threadsPerBlock, shared_memory, cudaStreamPerThread>>>(
                                d_ptr.buffer_1, external_output_ptr, LP.mem_offsets, LP.twiddle_in, LP.Q);
                        postcheck;
                        current_buffer = aliased_output_buffer;
                    }
#endif
                }
                break;
            }

            case xcorr_decomposed: {
                // TODO: FFT_ALGO_t
                // TODO: unused
                //                 using FFT    = decltype(FFT_base_arch( ) + Type<fft_type::c2c>( ) + Direction<fft_direction::forward>( ));
                //                 using invFFT = decltype(FFT_base_arch( ) + Type<fft_type::c2c>( ) + Direction<fft_direction::inverse>( ));

                //                 LaunchParams LP = SetLaunchParameters( xcorr_decomposed);

                //                 int shared_memory = LP.mem_offsets.shared_output * sizeof(complex_compute_t);
                //                 CheckSharedMemory(shared_memory, device_properties);

                // #if FFT_DEBUG_STAGE > 2
                //                 cudaErr(cudaFuncSetAttribute((void*)thread_fft_kernel_C2C_decomposed_ConjMul<external_image_t, FFT, invFFT, data_io_t, data_buffer_t>, cudaFuncAttributeMaxDynamicSharedMemorySize, shared_memory));

                //                 // the image_to_search pointer is set during call to CrossCorrelate,
                //                 precheck;
                //                 thread_fft_kernel_C2C_decomposed_ConjMul<FFT, invFFT, data_io_t, data_buffer_t><<<LP.gridDims, LP.threadsPerBlock, shared_memory, cudaStreamPerThread>>>((external_image_t*)other_image_ptr, intra_complex_input, intra_complex_output, LP.mem_offsets, LP.twiddle_in, LP.Q);
                //                 postcheck;
                //                 is_in_buffer_memory = ! is_in_buffer_memory;
                // #endif

                //                 break;
            }

            case c2c_fwd_decomposed: {
                // TODO: FFT_ALGO_t
                using FFT_nodir = decltype(FFT_base_arch( ) + Type<fft_type::c2c>( ));
                LaunchParams LP = SetLaunchParameters(c2c_fwd_decomposed);

                using FFT         = decltype(FFT_nodir( ) + Direction<fft_direction::forward>( ));
                int shared_memory = LP.mem_offsets.shared_output * sizeof(complex_compute_t);
                CheckSharedMemory(shared_memory, device_properties);
#if FFT_DEBUG_STAGE > 2
                cudaErr(cudaFuncSetAttribute((void*)thread_fft_kernel_C2C_decomposed<FFT, data_io_t, data_buffer_t>, cudaFuncAttributeMaxDynamicSharedMemorySize, shared_memory));
                if constexpr ( Rank == 1 ) {
                    // Should only be used when d_ptr.external_input is complex valued input
                    precheck;
                    thread_fft_kernel_C2C_decomposed<FFT, data_io_t, data_buffer_t><<<LP.gridDims, LP.threadsPerBlock, shared_memory, cudaStreamPerThread>>>(
                            d_ptr.external_input, external_output_ptr, LP.mem_offsets, LP.twiddle_in, LP.Q);
                    postcheck;
                }
                else if constexpr ( Rank == 2 ) {
                    precheck;
                    thread_fft_kernel_C2C_decomposed<FFT, data_io_t, data_buffer_t><<<LP.gridDims, LP.threadsPerBlock, shared_memory, cudaStreamPerThread>>>(
                            d_ptr.buffer_1, external_output_ptr, LP.mem_offsets, LP.twiddle_in, LP.Q);
                    postcheck;
                }
                current_buffer = aliased_output_buffer;
#endif

                break;
            }
            case c2c_inv_decomposed: {

                using FFT_nodir = decltype(FFT_base_arch( ) + Type<fft_type::c2c>( ));
                LaunchParams LP = SetLaunchParameters(c2c_inv_decomposed);

                using FFT         = decltype(FFT_nodir( ) + Direction<fft_direction::inverse>( ));
                int shared_memory = LP.mem_offsets.shared_output * sizeof(complex_compute_t);
                CheckSharedMemory(shared_memory, device_properties);
#if FFT_DEBUG_STAGE > 4
                cudaErr(cudaFuncSetAttribute((void*)thread_fft_kernel_C2C_decomposed<FFT, data_io_t, data_buffer_t>, cudaFuncAttributeMaxDynamicSharedMemorySize, shared_memory));
                if constexpr ( Rank == 1 ) {
                    // This should only be called when d_ptr.external_input is complex valued input
                    precheck;
                    thread_fft_kernel_C2C_decomposed<FFT, data_io_t, data_buffer_t><<<LP.gridDims, LP.threadsPerBlock, shared_memory, cudaStreamPerThread>>>(
                            d_ptr.external_input, external_output_ptr, LP.mem_offsets, LP.twiddle_in, LP.Q);
                    postcheck;
                    current_buffer = aliased_output_buffer;
                }
                else if constexpr ( Rank == 2 ) {
                    precheck;
                    thread_fft_kernel_C2C_decomposed<FFT, data_io_t, data_buffer_t><<<LP.gridDims, LP.threadsPerBlock, shared_memory, cudaStreamPerThread>>>(
                            reinterpret_cast<data_io_t*>(d_ptr.external_input), d_ptr.buffer_1, LP.mem_offsets, LP.twiddle_in, LP.Q);
                    postcheck;
                    current_buffer = fastfft_internal_buffer_1;
                }
#endif

                break;
            }
        } // switch on (thread) kernel_type
    } // if constexpr on thread type
    else {
        switch ( kernel_type ) {
            case r2c_none_XY: {
                if constexpr ( FFT_ALGO_t == Generic_Fwd_FFT ) {
                    using FFT               = decltype(FFT_base_arch( ) + Direction<fft_direction::forward>( ) + Type<fft_type::c2c>( ));
                    cudaError_t  error_code = cudaSuccess;
                    auto         workspace  = make_workspace<FFT>(error_code); // std::cout << " EPT: " << FFT::elements_per_thread << "kernel " << KernelName[kernel_type] << std::endl;
                    LaunchParams LP         = SetLaunchParameters(r2c_none_XY);

                    int shared_memory = FFT::shared_memory_size;
                    CheckSharedMemory(shared_memory, device_properties);

#if FFT_DEBUG_STAGE > 0
                    cudaErr(cudaFuncSetAttribute((void*)block_fft_kernel_R2C_NONE_XY<FFT, data_io_t, data_buffer_t>, cudaFuncAttributeMaxDynamicSharedMemorySize, shared_memory));

                    // If impl a round trip, the output will need to be data_io_t
                    if constexpr ( Rank == 1 ) {
                        MyFFTDebugAssertTrue(current_buffer == fastfft_external_input, "current_buffer != fastfft_external_input");
                        precheck;
                        block_fft_kernel_R2C_NONE_XY<FFT, data_io_t, data_buffer_t><<<LP.gridDims, LP.threadsPerBlock, shared_memory, cudaStreamPerThread>>>(
                                d_ptr.external_input, reinterpret_cast<data_buffer_t*>(external_output_ptr), LP.mem_offsets, workspace);
                        postcheck;
                        current_buffer = aliased_output_buffer;
                    }
                    else if constexpr ( Rank == 2 ) {
                        MyFFTDebugAssertTrue(current_buffer == fastfft_external_input, "current_buffer != fastfft_external_input");
                        precheck;
                        block_fft_kernel_R2C_NONE_XY<FFT, data_io_t, data_buffer_t><<<LP.gridDims, LP.threadsPerBlock, shared_memory, cudaStreamPerThread>>>(
                                d_ptr.external_input, d_ptr.buffer_1, LP.mem_offsets, workspace);
                        postcheck;
                    }
                    current_buffer = fastfft_internal_buffer_1;
#endif
                }
                break;
            }

            case r2c_none_XZ: {
                if constexpr ( FFT_ALGO_t == Generic_Fwd_FFT ) {
                    if constexpr ( Rank == 3 ) {
                        MyFFTDebugAssertTrue(current_buffer == fastfft_external_input, "current_buffer != fastfft_external_input");

                        using FFT               = decltype(FFT_base_arch( ) + Direction<fft_direction::forward>( ) + Type<fft_type::c2c>( ));
                        cudaError_t  error_code = cudaSuccess;
                        auto         workspace  = make_workspace<FFT>(error_code); // std::cout << " EPT: " << FFT::elements_per_thread << "kernel " << KernelName[kernel_type] << std::endl;
                        LaunchParams LP         = SetLaunchParameters(r2c_none_XZ);

                        int shared_memory = std::max(LP.threadsPerBlock.y * FFT::shared_memory_size, LP.threadsPerBlock.y * LP.mem_offsets.physical_x_output * (unsigned int)sizeof(complex_compute_t));
                        CheckSharedMemory(shared_memory, device_properties);

#if FFT_DEBUG_STAGE > 0
                        cudaErr(cudaFuncSetAttribute((void*)block_fft_kernel_R2C_NONE_XZ<FFT, data_io_t, data_buffer_t>, cudaFuncAttributeMaxDynamicSharedMemorySize, shared_memory));
                        precheck;
                        block_fft_kernel_R2C_NONE_XZ<FFT, data_io_t, data_buffer_t><<<LP.gridDims, LP.threadsPerBlock, shared_memory, cudaStreamPerThread>>>(
                                d_ptr.external_input, d_ptr.buffer_1, LP.mem_offsets, workspace);
                        postcheck;
                        current_buffer = fastfft_internal_buffer_1;
#endif
                    }
                }
                break;
            }

            case r2c_decrease_XY: {
                if constexpr ( FFT_ALGO_t == Generic_Fwd_FFT ) {
                    using FFT               = decltype(FFT_base_arch( ) + Direction<fft_direction::forward>( ) + Type<fft_type::c2c>( ));
                    cudaError_t  error_code = cudaSuccess;
                    auto         workspace  = make_workspace<FFT>(error_code); // std::cout << " EPT: " << FFT::elements_per_thread << "kernel " << KernelName[kernel_type] << std::endl;
                    LaunchParams LP         = SetLaunchParameters(r2c_decrease_XY);

                    // the shared mem is mixed between storage, shuffling and FFT. For this kernel we need to add padding to avoid bank conlicts (N/32)
                    int shared_memory = std::max(FFT::shared_memory_size * LP.threadsPerBlock.y, (LP.mem_offsets.shared_input + LP.mem_offsets.shared_input / 32) * (unsigned int)sizeof(complex_compute_t));

                    CheckSharedMemory(shared_memory, device_properties);
#if FFT_DEBUG_STAGE > 0

                    cudaErr(cudaFuncSetAttribute((void*)block_fft_kernel_R2C_DECREASE_XY<FFT, data_io_t, data_buffer_t>, cudaFuncAttributeMaxDynamicSharedMemorySize, shared_memory));
                    // PrintState( );
                    // PrintLaunchParameters(LP);
                    // std::cerr << "shared mem req " << shared_memory << std::endl;
                    // std::cerr << "FFT max tbp " << FFT::max_threads_per_block << std::endl;
                    if constexpr ( Rank == 1 ) {
                        MyFFTDebugAssertTrue(current_buffer == fastfft_external_input, "current_buffer != fastfft_external_input");
                        precheck;
                        block_fft_kernel_R2C_DECREASE_XY<FFT, data_io_t, data_buffer_t><<<LP.gridDims, LP.threadsPerBlock, shared_memory, cudaStreamPerThread>>>(
                                d_ptr.external_input, reinterpret_cast<data_buffer_t*>(external_output_ptr), LP.mem_offsets, LP.twiddle_in, LP.Q, workspace);
                        postcheck;
                        current_buffer = aliased_output_buffer;
                    }
                    else if constexpr ( Rank == 2 ) {
                        MyFFTDebugAssertTrue(current_buffer == fastfft_external_input, "current_buffer != fastfft_external_input");
                        precheck;
                        // PrintState( );
                        // PrintLaunchParameters(LP);
                        // std::cerr << "shared mem req " << shared_memory << std::endl;
                        // int    numBlocks;
                        // size_t block_size = LP.threadsPerBlock.x * LP.threadsPerBlock.y * LP.threadsPerBlock.y;
                        // cudaErr(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&numBlocks, (void*)block_fft_kernel_R2C_DECREASE_XY<FFT, data_io_t, data_buffer_t>, block_size, size_t(shared_memory)));

                        // std::cerr << "BLock size " << block_size << " shared mem " << shared_memory << " numBlocks " << numBlocks << std::endl;
                        // std::cerr << "pointer " << d_ptr.external_input << " is in memory " << is_pointer_in_device_memory(d_ptr.external_input) << std::endl;
                        // std::cerr << "pointer " << d_ptr.buffer_1 << " is in memory " << is_pointer_in_device_memory(d_ptr.buffer_1) << std::endl;
                        // std::cerr << "FFT::shared_memory_size " << FFT::shared_memory_size << std::endl;
                        // std::cerr << "size_of<FFT>::value " << size_of<FFT>::value << std::endl;
                        // std::cerr << "batch_size_of<FFT>::value " << batch_size_of<FFT>::value << std::endl;
                        // std::cerr << "FFT::elements_per_thread " << FFT::elements_per_thread << std::endl;
                        block_fft_kernel_R2C_DECREASE_XY<FFT, data_io_t, data_buffer_t><<<LP.gridDims, LP.threadsPerBlock, shared_memory, cudaStreamPerThread>>>(
                                d_ptr.external_input, d_ptr.buffer_1, LP.mem_offsets, LP.twiddle_in, LP.Q, workspace);
                        postcheck;
                        current_buffer = fastfft_internal_buffer_1;
                    }
#endif
                }
                break;
            }

            case r2c_increase_XY: {
                if constexpr ( FFT_ALGO_t == Generic_Fwd_FFT ) {
                    using FFT               = decltype(FFT_base_arch( ) + Direction<fft_direction::forward>( ) + Type<fft_type::c2c>( ));
                    cudaError_t  error_code = cudaSuccess;
                    auto         workspace  = make_workspace<FFT>(error_code); // std::cout << " EPT: " << FFT::elements_per_thread << "kernel " << KernelName[kernel_type] << std::endl;
                    LaunchParams LP         = SetLaunchParameters(r2c_increase_XY);

                    int shared_memory = LP.mem_offsets.shared_input * sizeof(scalar_compute_t) + FFT::shared_memory_size;

                    CheckSharedMemory(shared_memory, device_properties);
#if FFT_DEBUG_STAGE > 0
                    cudaErr(cudaFuncSetAttribute((void*)block_fft_kernel_R2C_INCREASE_XY<FFT, data_io_t, data_buffer_t>, cudaFuncAttributeMaxDynamicSharedMemorySize, shared_memory));
                    if constexpr ( Rank == 1 ) {
                        MyFFTDebugAssertTrue(current_buffer == fastfft_external_input, "current_buffer != fastfft_external_input");
                        precheck;
                        block_fft_kernel_R2C_INCREASE_XY<FFT, data_io_t, data_buffer_t><<<LP.gridDims, LP.threadsPerBlock, shared_memory, cudaStreamPerThread>>>(
                                d_ptr.external_input, reinterpret_cast<data_buffer_t*>(external_output_ptr), LP.mem_offsets, LP.twiddle_in, LP.Q, workspace);
                        postcheck;
                        current_buffer = aliased_output_buffer;
                    }
                    else if constexpr ( Rank == 2 ) {
                        MyFFTDebugAssertTrue(current_buffer == fastfft_external_input, "current_buffer != fastfft_external_input");
                        precheck;
                        block_fft_kernel_R2C_INCREASE_XY<FFT, data_io_t, data_buffer_t><<<LP.gridDims, LP.threadsPerBlock, shared_memory, cudaStreamPerThread>>>(
                                d_ptr.external_input, d_ptr.buffer_1, LP.mem_offsets, LP.twiddle_in, LP.Q, workspace);
                        postcheck;
                        current_buffer = fastfft_internal_buffer_1;
                    }
#endif
                }
                break;
            }

            case r2c_increase_XZ: {
                if constexpr ( FFT_ALGO_t == Generic_Fwd_FFT ) {
                    if constexpr ( Rank == 3 ) {
                        MyFFTDebugAssertTrue(current_buffer == fastfft_external_input, "current_buffer != fastfft_external_input");
                        using FFT               = decltype(FFT_base_arch( ) + Direction<fft_direction::forward>( ) + Type<fft_type::c2c>( ));
                        cudaError_t  error_code = cudaSuccess;
                        auto         workspace  = make_workspace<FFT>(error_code); // FIXME: I don't think this is right when XZ_STRIDE is used
                        LaunchParams LP         = SetLaunchParameters(r2c_increase_XZ);

                        // We need shared memory to hold the input array(s) that is const through the kernel.
                        // We alternate using additional shared memory for the computation and the transposition of the data.
                        int shared_memory = std::max(XZ_STRIDE * FFT::shared_memory_size, LP.mem_offsets.physical_x_output / LP.Q * (unsigned int)sizeof(complex_compute_t));
                        shared_memory += XZ_STRIDE * LP.mem_offsets.shared_input * (unsigned int)sizeof(scalar_compute_t);

                        CheckSharedMemory(shared_memory, device_properties);
#if FFT_DEBUG_STAGE > 0
                        cudaErr(cudaFuncSetAttribute((void*)block_fft_kernel_R2C_INCREASE_XZ<FFT, data_io_t, data_buffer_t>, cudaFuncAttributeMaxDynamicSharedMemorySize, shared_memory));
                        precheck;
                        block_fft_kernel_R2C_INCREASE_XZ<FFT, data_io_t, data_buffer_t><<<LP.gridDims, LP.threadsPerBlock, shared_memory, cudaStreamPerThread>>>(
                                d_ptr.external_input, d_ptr.buffer_1, LP.mem_offsets, LP.twiddle_in, LP.Q, workspace);
                        postcheck;
                        current_buffer = fastfft_internal_buffer_1;
#endif
                    }
                }
                break;
            }

            case c2c_fwd_none: {
                if constexpr ( FFT_ALGO_t == Generic_Fwd_FFT ) {
                    using FFT = decltype(FFT_base_arch( ) + Direction<fft_direction::forward>( ) + Type<fft_type::c2c>( ));

                    LaunchParams LP = SetLaunchParameters(c2c_fwd_none);

                    cudaError_t      error_code    = cudaSuccess;
                    DebugUnused auto workspace     = make_workspace<FFT>(error_code); // std::cout << " EPT: " << FFT::elements_per_thread << "kernel " << KernelName[kernel_type] << std::endl;
                    DebugUnused int  shared_memory = FFT::shared_memory_size;

#if FFT_DEBUG_STAGE > 2
                    CheckSharedMemory(shared_memory, device_properties);
                    cudaErr(cudaFuncSetAttribute((void*)block_fft_kernel_C2C_NONE<FFT, data_buffer_t>, cudaFuncAttributeMaxDynamicSharedMemorySize, shared_memory));
                    if constexpr ( Rank == 1 ) {
                        MyFFTDebugAssertTrue(current_buffer == fastfft_external_input, "current_buffer != fastfft_external_input");
                        precheck;
                        block_fft_kernel_C2C_NONE<FFT, data_buffer_t><<<LP.gridDims, LP.threadsPerBlock, shared_memory, cudaStreamPerThread>>>(
                                d_ptr.external_input, reinterpret_cast<data_buffer_t*>(external_output_ptr), LP.mem_offsets, workspace);
                        postcheck;
                    }
                    else if constexpr ( Rank == 2 ) {
                        MyFFTDebugAssertTrue(current_buffer == fastfft_internal_buffer_1, "current_buffer != fastfft_internal_buffer_1");
                        precheck;
                        block_fft_kernel_C2C_NONE<FFT, data_buffer_t><<<LP.gridDims, LP.threadsPerBlock, shared_memory, cudaStreamPerThread>>>(
                                d_ptr.buffer_1, reinterpret_cast<data_buffer_t*>(external_output_ptr), LP.mem_offsets, workspace);
                        postcheck;
                    }
                    else if constexpr ( Rank == 3 ) {
                        MyFFTDebugAssertTrue(current_buffer == fastfft_internal_buffer_2, "current_buffer != fastfft_internal_buffer_2");
                        precheck;
                        block_fft_kernel_C2C_NONE<FFT, data_buffer_t><<<LP.gridDims, LP.threadsPerBlock, shared_memory, cudaStreamPerThread>>>(
                                d_ptr.buffer_2, reinterpret_cast<data_buffer_t*>(external_output_ptr), LP.mem_offsets, workspace);
                        postcheck;
                    }
                    current_buffer = aliased_output_buffer;
#endif
                }
                break;
            }

            case c2c_fwd_none_XYZ: {
                if constexpr ( FFT_ALGO_t == Generic_Fwd_FFT ) {
                    if constexpr ( Rank == 3 ) {
                        MyFFTDebugAssertTrue(current_buffer == fastfft_internal_buffer_1, "current_buffer != fastfft_internal_buffer_1");

                        using FFT = decltype(FFT_base_arch( ) + Direction<fft_direction::forward>( ) + Type<fft_type::c2c>( ));

                        LaunchParams LP = SetLaunchParameters(c2c_fwd_none_XYZ);

                        cudaError_t error_code    = cudaSuccess;
                        auto        workspace     = make_workspace<FFT>(error_code); // std::cout << " EPT: " << FFT::elements_per_thread << "kernel " << KernelName[kernel_type] << std::endl;
                        int         shared_memory = std::max(XZ_STRIDE * FFT::shared_memory_size, size_of<FFT>::value * (unsigned int)sizeof(complex_compute_t) * XZ_STRIDE);

#if FFT_DEBUG_STAGE > 1

                        CheckSharedMemory(shared_memory, device_properties);
                        cudaErr(cudaFuncSetAttribute((void*)block_fft_kernel_C2C_NONE_XYZ<FFT, data_buffer_t>, cudaFuncAttributeMaxDynamicSharedMemorySize, shared_memory));
                        precheck;
                        block_fft_kernel_C2C_NONE_XYZ<FFT, data_buffer_t><<<LP.gridDims, LP.threadsPerBlock, shared_memory, cudaStreamPerThread>>>(
                                d_ptr.buffer_1, d_ptr.buffer_2, LP.mem_offsets, workspace);
                        postcheck;
                        current_buffer = fastfft_internal_buffer_2;
#endif
                    }
                }
                break;
            }

            case c2c_fwd_decrease: {
                if constexpr ( FFT_ALGO_t == Generic_Fwd_FFT ) {
                    using FFT               = decltype(FFT_base_arch( ) + Direction<fft_direction::forward>( ) + Type<fft_type::c2c>( ));
                    cudaError_t  error_code = cudaSuccess;
                    auto         workspace  = make_workspace<FFT>(error_code); // std::cout << " EPT: " << FFT::elements_per_thread << "kernel " << KernelName[kernel_type] << std::endl;
                    LaunchParams LP         = SetLaunchParameters(c2c_fwd_decrease);

#if FFT_DEBUG_STAGE > 2
                    // the shared mem is mixed between storage, shuffling and FFT. For this kernel we need to add padding to avoid bank conlicts (N/32)
                    // For decrease methods, the shared_input > shared_output
                    int shared_memory = std::max(FFT::shared_memory_size * LP.threadsPerBlock.y, (LP.mem_offsets.shared_input + LP.mem_offsets.shared_input / 32) * (unsigned int)sizeof(complex_compute_t));

                    CheckSharedMemory(shared_memory, device_properties);
                    cudaErr(cudaFuncSetAttribute((void*)block_fft_kernel_C2C_DECREASE<FFT, data_buffer_t>, cudaFuncAttributeMaxDynamicSharedMemorySize, shared_memory));

                    if constexpr ( Rank == 1 ) {
                        MyFFTDebugAssertTrue(current_buffer == fastfft_external_input, "current_buffer != fastfft_external_input");
                        precheck;
                        block_fft_kernel_C2C_DECREASE<FFT, data_buffer_t><<<LP.gridDims, LP.threadsPerBlock, shared_memory, cudaStreamPerThread>>>(
                                d_ptr.external_input, reinterpret_cast<data_buffer_t*>(external_output_ptr), LP.mem_offsets, LP.twiddle_in, LP.Q, workspace);
                        postcheck;
                    }
                    else if constexpr ( Rank == 2 ) {
                        MyFFTDebugAssertTrue(current_buffer == fastfft_internal_buffer_1, "current_buffer != fastfft_internal_buffer_1");
                        precheck;
                        block_fft_kernel_C2C_DECREASE<FFT, data_buffer_t><<<LP.gridDims, LP.threadsPerBlock, shared_memory, cudaStreamPerThread>>>(
                                d_ptr.buffer_1, reinterpret_cast<data_buffer_t*>(external_output_ptr), LP.mem_offsets, LP.twiddle_in, LP.Q, workspace);
                        postcheck;
                    }
                    // Rank 3 not yet implemented
                    current_buffer = aliased_output_buffer;
#endif
                }
                break;
            }

            case c2c_fwd_increase_XYZ: {
                if constexpr ( FFT_ALGO_t == Generic_Fwd_FFT ) {
                    if constexpr ( Rank == 3 ) {
                        MyFFTDebugAssertTrue(current_buffer == fastfft_internal_buffer_1, "current_buffer != fastfft_internal_buffer_1");
                        using FFT = decltype(FFT_base_arch( ) + Direction<fft_direction::forward>( ) + Type<fft_type::c2c>( ));

                        LaunchParams LP = SetLaunchParameters(c2c_fwd_increase_XYZ);

                        cudaError_t error_code = cudaSuccess;
                        auto        workspace  = make_workspace<FFT>(error_code); // std::cout << " EPT: " << FFT::elements_per_thread << "kernel " << KernelName[kernel_type] << std::endl;

                        // We need shared memory to hold the input array(s) that is const through the kernel.
                        // We alternate using additional shared memory for the computation and the transposition of the data.
                        int shared_memory = std::max(XZ_STRIDE * FFT::shared_memory_size, XZ_STRIDE * LP.mem_offsets.physical_x_output / LP.Q * (unsigned int)sizeof(complex_compute_t));
                        shared_memory += XZ_STRIDE * LP.mem_offsets.shared_input * (unsigned int)sizeof(complex_compute_t);

#if FFT_DEBUG_STAGE > 1

                        CheckSharedMemory(shared_memory, device_properties);
                        cudaErr(cudaFuncSetAttribute((void*)block_fft_kernel_C2C_INCREASE_XYZ<FFT, data_buffer_t>, cudaFuncAttributeMaxDynamicSharedMemorySize, shared_memory));
                        precheck;
                        block_fft_kernel_C2C_INCREASE_XYZ<FFT, data_buffer_t><<<LP.gridDims, LP.threadsPerBlock, shared_memory, cudaStreamPerThread>>>(
                                d_ptr.buffer_1, d_ptr.buffer_2, LP.mem_offsets, LP.twiddle_in, LP.Q, workspace);
                        postcheck;
                        current_buffer = fastfft_internal_buffer_2;
#endif
                    }
                }
                break;
            }

            case c2c_fwd_increase: {
                if constexpr ( FFT_ALGO_t == Generic_Fwd_FFT ) {
                    using FFT = decltype(FFT_base_arch( ) + Direction<fft_direction::forward>( ) + Type<fft_type::c2c>( ));

                    LaunchParams LP = SetLaunchParameters(c2c_fwd_increase);

                    cudaError_t error_code    = cudaSuccess;
                    auto        workspace     = make_workspace<FFT>(error_code); // std::cout << " EPT: " << FFT::elements_per_thread << "kernel " << KernelName[kernel_type] << std::endl;
                    int         shared_memory = FFT::shared_memory_size + (unsigned int)sizeof(complex_compute_t) * (LP.mem_offsets.shared_input + LP.mem_offsets.shared_output);

#if FFT_DEBUG_STAGE > 2
                    CheckSharedMemory(shared_memory, device_properties);
                    cudaErr(cudaFuncSetAttribute((void*)block_fft_kernel_C2C_INCREASE<FFT, data_buffer_t>, cudaFuncAttributeMaxDynamicSharedMemorySize, shared_memory));

                    if constexpr ( Rank == 1 ) {
                        MyFFTDebugAssertTrue(current_buffer == fastfft_external_input, "current_buffer != fastfft_external_input");
                        precheck;
                        block_fft_kernel_C2C_INCREASE<FFT, data_buffer_t><<<LP.gridDims, LP.threadsPerBlock, shared_memory, cudaStreamPerThread>>>(
                                d_ptr.external_input, reinterpret_cast<data_buffer_t*>(external_output_ptr), LP.mem_offsets, LP.twiddle_in, LP.Q, workspace);
                        postcheck;
                    }
                    else if constexpr ( Rank == 2 ) {
                        MyFFTDebugAssertTrue(current_buffer == fastfft_internal_buffer_1, "current_buffer != fastfft_internal_buffer_1");
                        precheck;
                        block_fft_kernel_C2C_INCREASE<FFT, data_buffer_t><<<LP.gridDims, LP.threadsPerBlock, shared_memory, cudaStreamPerThread>>>(
                                d_ptr.buffer_1, reinterpret_cast<data_buffer_t*>(external_output_ptr), LP.mem_offsets, LP.twiddle_in, LP.Q, workspace);
                        postcheck;
                    }
                    else if constexpr ( Rank == 3 ) {
                        MyFFTDebugAssertTrue(current_buffer == fastfft_internal_buffer_2, "current_buffer != fastfft_internal_buffer_2");
                        precheck;
                        block_fft_kernel_C2C_INCREASE<FFT, data_buffer_t><<<LP.gridDims, LP.threadsPerBlock, shared_memory, cudaStreamPerThread>>>(
                                d_ptr.buffer_2, reinterpret_cast<data_buffer_t*>(external_output_ptr), LP.mem_offsets, LP.twiddle_in, LP.Q, workspace);
                        postcheck;
                    }
                    current_buffer = aliased_output_buffer;
#endif
                }
                break;
            }

            case c2c_inv_none: {
                if constexpr ( FFT_ALGO_t == Generic_Inv_FFT ) {
                    using FFT = decltype(FFT_base_arch( ) + Type<fft_type::c2c>( ) + Direction<fft_direction::inverse>( ));

                    LaunchParams LP = SetLaunchParameters(c2c_inv_none);

                    cudaError_t error_code = cudaSuccess;
                    auto        workspace  = make_workspace<FFT>(error_code); // std::cout << " EPT: " << FFT::elements_per_thread << "kernel " << KernelName[kernel_type] << std::endl;

                    int shared_memory = FFT::shared_memory_size;

                    CheckSharedMemory(shared_memory, device_properties);
#if FFT_DEBUG_STAGE > 4
                    cudaErr(cudaFuncSetAttribute((void*)block_fft_kernel_C2C_NONE<FFT, data_buffer_t>, cudaFuncAttributeMaxDynamicSharedMemorySize, shared_memory));
                    if constexpr ( Rank == 1 ) {
                        MyFFTDebugAssertTrue(current_buffer == fastfft_external_input, "current_buffer != fastfft_external_input");
                        precheck;
                        block_fft_kernel_C2C_NONE<FFT, data_buffer_t><<<LP.gridDims, LP.threadsPerBlock, shared_memory, cudaStreamPerThread>>>(
                                reinterpret_cast<data_buffer_t*>(d_ptr.external_input), external_output_ptr, LP.mem_offsets, workspace);
                        postcheck;
                        current_buffer = aliased_output_buffer;
                    }
                    else if constexpr ( Rank == 2 ) {
                        MyFFTDebugAssertTrue(current_buffer == fastfft_external_input, "current_buffer != fastfft_external_input");
                        precheck;
                        block_fft_kernel_C2C_NONE<FFT, data_buffer_t><<<LP.gridDims, LP.threadsPerBlock, shared_memory, cudaStreamPerThread>>>(
                                reinterpret_cast<data_buffer_t*>(d_ptr.external_input), d_ptr.buffer_1, LP.mem_offsets, workspace);
                        postcheck;
                        current_buffer = fastfft_internal_buffer_1;
                    }
#endif

                    // do something
                }
                break;
            }

            case c2c_inv_none_XZ: {
                if constexpr ( FFT_ALGO_t == Generic_Inv_FFT ) {
                    if constexpr ( Rank == 3 ) {
                        MyFFTDebugAssertTrue(current_buffer == fastfft_external_input, "current_buffer != fastfft_external_input");
                        using FFT = decltype(FFT_base_arch( ) + Type<fft_type::c2c>( ) + Direction<fft_direction::inverse>( ));

                        LaunchParams LP = SetLaunchParameters(c2c_inv_none_XZ);

                        cudaError_t error_code = cudaSuccess;
                        auto        workspace  = make_workspace<FFT>(error_code); // std::cout << " EPT: " << FFT::elements_per_thread << "kernel " << KernelName[kernel_type] << std::endl;

                        int shared_memory = std::max(FFT::shared_memory_size * XZ_STRIDE, size_of<FFT>::value * (unsigned int)sizeof(complex_compute_t) * XZ_STRIDE);

                        CheckSharedMemory(shared_memory, device_properties);
#if FFT_DEBUG_STAGE > 4
                        cudaErr(cudaFuncSetAttribute((void*)block_fft_kernel_C2C_NONE_XZ<FFT, data_buffer_t>, cudaFuncAttributeMaxDynamicSharedMemorySize, shared_memory));
                        precheck;
                        block_fft_kernel_C2C_NONE_XZ<FFT, data_buffer_t><<<LP.gridDims, LP.threadsPerBlock, shared_memory, cudaStreamPerThread>>>(
                                reinterpret_cast<data_buffer_t*>(d_ptr.external_input), d_ptr.buffer_1, LP.mem_offsets, workspace);
                        postcheck;
                        current_buffer = fastfft_internal_buffer_1;
#endif
                    }
                    // do something
                }
                break;
            }

            case c2c_inv_none_XYZ: {
                if constexpr ( FFT_ALGO_t == Generic_Inv_FFT ) {
                    if constexpr ( Rank == 3 ) {
                        MyFFTDebugAssertTrue(current_buffer == fastfft_internal_buffer_1, "current_buffer != fastfft_internal_buffer_1");
                        using FFT = decltype(FFT_base_arch( ) + Direction<fft_direction::inverse>( ) + Type<fft_type::c2c>( ));

                        LaunchParams LP = SetLaunchParameters(c2c_inv_none_XYZ);

                        cudaError_t error_code    = cudaSuccess;
                        auto        workspace     = make_workspace<FFT>(error_code); // std::cout << " EPT: " << FFT::elements_per_thread << "kernel " << KernelName[kernel_type] << std::endl;
                        int         shared_memory = std::max(XZ_STRIDE * FFT::shared_memory_size, size_of<FFT>::value * (unsigned int)sizeof(complex_compute_t) * XZ_STRIDE);

#if FFT_DEBUG_STAGE > 5
                        CheckSharedMemory(shared_memory, device_properties);
                        cudaErr(cudaFuncSetAttribute((void*)block_fft_kernel_C2C_NONE_XYZ<FFT, data_buffer_t>, cudaFuncAttributeMaxDynamicSharedMemorySize, shared_memory));
                        precheck;
                        block_fft_kernel_C2C_NONE_XYZ<FFT, data_buffer_t><<<LP.gridDims, LP.threadsPerBlock, shared_memory, cudaStreamPerThread>>>(
                                d_ptr.buffer_1, d_ptr.buffer_2, LP.mem_offsets, workspace);
                        postcheck;
                        current_buffer = fastfft_internal_buffer_2;
#endif
                    }
                }
                break;
            }

            case c2c_inv_decrease: {
                if constexpr ( FFT_ALGO_t == Generic_Inv_FFT ) {
                    using FFT               = decltype(FFT_base_arch( ) + Direction<fft_direction::inverse>( ) + Type<fft_type::c2c>( ));
                    cudaError_t  error_code = cudaSuccess;
                    auto         workspace  = make_workspace<FFT>(error_code); // std::cout << " EPT: " << FFT::elements_per_thread << "kernel " << KernelName[kernel_type] << std::endl;
                    LaunchParams LP         = SetLaunchParameters(c2c_inv_decrease);

#if FFT_DEBUG_STAGE > 4
                    // the shared mem is mixed between storage, shuffling and FFT. For this kernel we need to add padding to avoid bank conlicts (N/32)
                    // For decrease methods, the shared_input > shared_output
                    int shared_memory = std::max(FFT::shared_memory_size * LP.threadsPerBlock.y, (LP.mem_offsets.shared_input + LP.mem_offsets.shared_input / 32) * (unsigned int)sizeof(complex_compute_t));

                    CheckSharedMemory(shared_memory, device_properties);
                    cudaErr(cudaFuncSetAttribute((void*)block_fft_kernel_C2C_DECREASE<FFT, data_buffer_t>, cudaFuncAttributeMaxDynamicSharedMemorySize, shared_memory));

                    if constexpr ( Rank == 1 ) {
                        MyFFTDebugAssertTrue(current_buffer == fastfft_external_input, "current_buffer != fastfft_external_input");
                        precheck;
                        block_fft_kernel_C2C_DECREASE<FFT, data_buffer_t><<<LP.gridDims, LP.threadsPerBlock, shared_memory, cudaStreamPerThread>>>(
                                reinterpret_cast<data_buffer_t*>(d_ptr.external_input), external_output_ptr, LP.mem_offsets, LP.twiddle_in, LP.Q, workspace);
                        postcheck;
                        current_buffer = aliased_output_buffer;
                    }
                    else if constexpr ( Rank == 2 ) {
                        MyFFTDebugAssertTrue(current_buffer == fastfft_external_input, "current_buffer != fastfft_external_input");
                        precheck;
                        block_fft_kernel_C2C_DECREASE<FFT, data_buffer_t><<<LP.gridDims, LP.threadsPerBlock, shared_memory, cudaStreamPerThread>>>(
                                reinterpret_cast<data_buffer_t*>(d_ptr.external_input), d_ptr.buffer_1, LP.mem_offsets, LP.twiddle_in, LP.Q, workspace);
                        postcheck;
                        current_buffer = fastfft_internal_buffer_1;
                    }
                    // 3D not yet implemented
#endif
                }
                break;
            }

            case c2c_inv_increase: {
                if constexpr ( FFT_ALGO_t == Generic_Inv_FFT ) {
                    MyFFTRunTimeAssertTrue(false, "c2c_inv_increase is not yet implemented.");

#if FFT_DEBUG_STAGE > 4
// TODO;
#endif
                }
                break;
            }

            case c2r_none: {
                if constexpr ( FFT_ALGO_t == Generic_Inv_FFT ) {
                    using FFT = decltype(FFT_base_arch( ) + Direction<fft_direction::inverse>( ) + Type<fft_type::c2r>( ));

                    LaunchParams LP = SetLaunchParameters(c2r_none);

                    cudaError_t error_code = cudaSuccess;
                    auto        workspace  = make_workspace<FFT>(error_code); // std::cout << " EPT: " << FFT::elements_per_thread << "kernel " << KernelName[kernel_type] << std::endl;        cudaErr(error_code);

                    int shared_memory = FFT::shared_memory_size;

                    CheckSharedMemory(shared_memory, device_properties);
#if FFT_DEBUG_STAGE > 6
                    cudaErr(cudaFuncSetAttribute((void*)block_fft_kernel_C2R_NONE<FFT, data_buffer_t, data_io_t>, cudaFuncAttributeMaxDynamicSharedMemorySize, shared_memory));

                    if constexpr ( Rank == 1 ) {
                        // TODO:
                        // precheck;
                        // block_fft_kernel_C2R_NONE<FFT, data_buffer_t, data_io_t><<<LP.gridDims, LP.threadsPerBlock, shared_memory, cudaStreamPerThread>>>(external_input, external_data_ptr, LP.mem_offsets, workspace);
                        // postcheck;
                    }
                    else if constexpr ( Rank == 2 ) {
                        // TODO:
                        // precheck;
                        // block_fft_kernel_C2R_NONE<FFT, data_buffer_t, data_io_t><<<LP.gridDims, LP.threadsPerBlock, shared_memory, cudaStreamPerThread>>>(
                        //         intra_complex_input, external_data_ptr, LP.mem_offsets, workspace);
                        // postcheck;
                    }
                    else if constexpr ( Rank == 3 ) {
                        MyFFTDebugAssertTrue(current_buffer == fastfft_internal_buffer_2, "current_buffer != fastfft_internal_buffer_2");
                        precheck;
                        block_fft_kernel_C2R_NONE<FFT, data_buffer_t, data_io_t><<<LP.gridDims, LP.threadsPerBlock, shared_memory, cudaStreamPerThread>>>(
                                d_ptr.buffer_2, external_output_ptr, LP.mem_offsets, workspace);
                        postcheck;
                        current_buffer = aliased_output_buffer;
                    }

#endif
                }
                break;
            }

            case c2r_none_XY: {
                if constexpr ( FFT_ALGO_t == Generic_Inv_FFT ) {
                    using FFT = decltype(FFT_base_arch( ) + Direction<fft_direction::inverse>( ) + Type<fft_type::c2r>( ));

                    LaunchParams LP = SetLaunchParameters(c2r_none_XY);

                    cudaError_t error_code = cudaSuccess;
                    auto        workspace  = make_workspace<FFT>(error_code); // std::cout << " EPT: " << FFT::elements_per_thread << "kernel " << KernelName[kernel_type] << std::endl;        cudaErr(error_code);

                    int shared_memory = FFT::shared_memory_size;

                    CheckSharedMemory(shared_memory, device_properties);
#if FFT_DEBUG_STAGE > 6
                    cudaErr(cudaFuncSetAttribute((void*)block_fft_kernel_C2R_NONE_XY<FFT, data_buffer_t, data_io_t>, cudaFuncAttributeMaxDynamicSharedMemorySize, shared_memory));

                    if constexpr ( Rank == 1 ) {
                        MyFFTDebugAssertTrue(current_buffer == fastfft_external_input, "current_buffer != fastfft_external_input");
                        precheck;
                        block_fft_kernel_C2R_NONE_XY<FFT, data_buffer_t, data_io_t><<<LP.gridDims, LP.threadsPerBlock, shared_memory, cudaStreamPerThread>>>(
                                reinterpret_cast<data_buffer_t*>(d_ptr.external_input), external_output_ptr, LP.mem_offsets, workspace);
                        postcheck;
                    }
                    else if constexpr ( Rank == 2 ) {
                        // This could be the last step after a partial InvFFT or partial FwdImgInv, and right now, the only way to tell the difference is if the
                        // current buffer is set correctly.
                        MyFFTDebugAssertTrue(current_buffer == fastfft_internal_buffer_1 || current_buffer == fastfft_internal_buffer_2, "current_buffer != fastfft_internal_buffer_1/2");
                        if ( current_buffer == fastfft_internal_buffer_1 ) {
                            // Presumably this is intended to be the second step of an InvFFT
                            precheck;
                            block_fft_kernel_C2R_NONE_XY<FFT, data_buffer_t, data_io_t><<<LP.gridDims, LP.threadsPerBlock, shared_memory, cudaStreamPerThread>>>(
                                    d_ptr.buffer_1, external_output_ptr, LP.mem_offsets, workspace);
                            postcheck;
                        }
                        else if ( current_buffer == fastfft_internal_buffer_2 ) {
                            // Presumably this is intended to be the last step in a FwdImgInv
                            precheck;
                            block_fft_kernel_C2R_NONE_XY<FFT, data_buffer_t, data_io_t><<<LP.gridDims, LP.threadsPerBlock, shared_memory, cudaStreamPerThread>>>(
                                    d_ptr.buffer_2, external_output_ptr, LP.mem_offsets, workspace);
                            postcheck;
                        }
                        // DebugAssert will fail if current_buffer is not set correctly to end up in an else clause
                    }
                    current_buffer = aliased_output_buffer;
                    // 3D not yet implemented
#endif
                }
                break;
            }

            case c2r_decrease_XY: {
                if constexpr ( FFT_ALGO_t == Generic_Inv_FFT ) {
                    using FFT = decltype(FFT_base_arch( ) + Direction<fft_direction::inverse>( ) + Type<fft_type::c2r>( ));

                    LaunchParams LP = SetLaunchParameters(c2r_decrease_XY);

                    cudaError_t error_code = cudaSuccess;
                    auto        workspace  = make_workspace<FFT>(error_code); // std::cout << " EPT: " << FFT::elements_per_thread << "kernel " << KernelName[kernel_type] << std::endl;        cudaErr(error_code);

                    int shared_memory = std::max(FFT::shared_memory_size * LP.gridDims.z, (LP.mem_offsets.shared_input + LP.mem_offsets.shared_input / 32) * (unsigned int)sizeof(complex_compute_t));

                    CheckSharedMemory(shared_memory, device_properties);

#if FFT_DEBUG_STAGE > 6
                    cudaErr(cudaFuncSetAttribute((void*)block_fft_kernel_C2R_DECREASE_XY<FFT, data_buffer_t, data_io_t>, cudaFuncAttributeMaxDynamicSharedMemorySize, shared_memory));
                    if constexpr ( Rank == 1 ) {
                        precheck;
                        block_fft_kernel_C2R_DECREASE_XY<FFT, data_buffer_t, data_io_t><<<LP.gridDims, LP.threadsPerBlock, shared_memory, cudaStreamPerThread>>>(
                                d_ptr.external_input, external_output_ptr, LP.mem_offsets, LP.twiddle_in, LP.Q, workspace);
                        postcheck;
                    }
                    else if constexpr ( Rank == 2 ) {
                        // This could be the last step after a partial InvFFT or partial FwdImgInv, and right now, the only way to tell the difference is if the
                        // current buffer is set correctly.
                        MyFFTDebugAssertTrue(current_buffer == fastfft_internal_buffer_1 || current_buffer == fastfft_internal_buffer_2, "current_buffer != fastfft_internal_buffer_1/2");
                        if ( current_buffer == fastfft_internal_buffer_1 ) {
                            precheck;
                            block_fft_kernel_C2R_DECREASE_XY<FFT, data_buffer_t, data_io_t><<<LP.gridDims, LP.threadsPerBlock, shared_memory, cudaStreamPerThread>>>(
                                    d_ptr.buffer_1, external_output_ptr, LP.mem_offsets, LP.twiddle_in, LP.Q, workspace);
                            postcheck;
                        }
                        else if ( current_buffer == fastfft_internal_buffer_2 ) {
                            precheck;
                            block_fft_kernel_C2R_DECREASE_XY<FFT, data_buffer_t, data_io_t><<<LP.gridDims, LP.threadsPerBlock, shared_memory, cudaStreamPerThread>>>(
                                    d_ptr.buffer_2, external_output_ptr, LP.mem_offsets, LP.twiddle_in, LP.Q, workspace);
                            postcheck;
                        }
                    }
                    current_buffer = aliased_output_buffer;

#endif
                }
                break;
            }

            case c2r_increase: {
                if constexpr ( FFT_ALGO_t == Generic_Inv_FFT ) {
                    MyFFTRunTimeAssertTrue(false, "c2r_increase is not yet implemented.");
                }
                break;
            }

            case xcorr_fwd_none_inv_decrease: {
                if constexpr ( FFT_ALGO_t == Generic_Fwd_Image_Inv_FFT ) {
                    if constexpr ( Rank == 2 ) {
                        MyFFTDebugAssertTrue(current_buffer == fastfft_internal_buffer_1, "current_buffer != fastfft_internal_buffer_1");
                        using FFT    = decltype(FFT_base_arch( ) + Type<fft_type::c2c>( ) + Direction<fft_direction::forward>( ));
                        using invFFT = decltype(FFT_base_arch( ) + Type<fft_type::c2c>( ) + Direction<fft_direction::inverse>( ));

                        LaunchParams LP = SetLaunchParameters(xcorr_fwd_none_inv_decrease);

                        cudaError_t error_code    = cudaSuccess;
                        auto        workspace_fwd = make_workspace<FFT>(error_code); // presumably larger of the two
                        cudaErr(error_code);
                        error_code         = cudaSuccess;
                        auto workspace_inv = make_workspace<invFFT>(error_code); // presumably larger of the two
                        cudaErr(error_code);

                        // Max shared memory needed to store the full 1d fft remaining on the forward transform
                        unsigned int shared_memory = FFT::shared_memory_size + (unsigned int)sizeof(complex_compute_t) * LP.mem_offsets.physical_x_input;
                        // shared_memory = std::max( shared_memory, std::max( invFFT::shared_memory_size * LP.threadsPerBlock.y, (LP.mem_offsets.shared_input + LP.mem_offsets.shared_input/32) * (unsigned int)sizeof(complex_compute_t)));

                        CheckSharedMemory(shared_memory, device_properties);
#if FFT_DEBUG_STAGE > 2

                        cudaErr(cudaFuncSetAttribute((void*)block_fft_kernel_C2C_FWD_NONE_INV_DECREASE_ConjMul<external_image_t, FFT, invFFT, data_buffer_t>, cudaFuncAttributeMaxDynamicSharedMemorySize, shared_memory));
                        // Right now, because of the n_threads == size_of<FFT> requirement, we are explicitly zero padding, so we need to send an "apparent Q" to know the input size.
                        // Could send the actual size, but later when converting to use the transform decomp with different sized FFTs this will be a more direct conversion.
                        int apparent_Q = size_of<FFT>::value / inv_dims_out.y;
                        precheck;
                        block_fft_kernel_C2C_FWD_NONE_INV_DECREASE_ConjMul<external_image_t, FFT, invFFT, data_buffer_t><<<LP.gridDims, LP.threadsPerBlock, shared_memory, cudaStreamPerThread>>>(
                                (external_image_t*)other_image_ptr, d_ptr.buffer_1, d_ptr.buffer_2, LP.mem_offsets, LP.twiddle_in, apparent_Q, workspace_fwd, workspace_inv);
                        postcheck;
                        current_buffer = fastfft_internal_buffer_2;
#endif
                    }
                }
                // do something

                break;
            }
            case generic_fwd_increase_op_inv_none: {
                if constexpr ( FFT_ALGO_t == Generic_Fwd_Image_Inv_FFT ) {
                    // For convenience, we are explicitly zero-padding. This is lazy. FIXME
                    using FFT    = decltype(FFT_base_arch( ) + Type<fft_type::c2c>( ) + Direction<fft_direction::forward>( ));
                    using invFFT = decltype(FFT_base_arch( ) + Type<fft_type::c2c>( ) + Direction<fft_direction::inverse>( ));

                    LaunchParams LP = SetLaunchParameters(generic_fwd_increase_op_inv_none);

                    cudaError_t error_code    = cudaSuccess;
                    auto        workspace_fwd = make_workspace<FFT>(error_code); // presumably larger of the two
                    cudaErr(error_code);
                    error_code         = cudaSuccess;
                    auto workspace_inv = make_workspace<invFFT>(error_code); // presumably larger of the two
                    cudaErr(error_code);

                    int shared_memory = invFFT::shared_memory_size;
                    CheckSharedMemory(shared_memory, device_properties);

                    // __nv_is_extended_device_lambda_closure_type(type);
                    // __nv_is_extended_host_device_lambda_closure_type(type)
                    if constexpr ( IS_IKF_t<IntraOpType>( ) ) {
// FIXME
#if FFT_DEBUG_STAGE > 2
                        // Right now, because of the n_threads == size_of<FFT> requirement, we are explicitly zero padding, so we need to send an "apparent Q" to know the input size.
                        // Could send the actual size, but later when converting to use the transform decomp with different sized FFTs this will be a more direct conversion.
                        int apparent_Q = size_of<FFT>::value / fwd_dims_in.y;
                        cudaErr(cudaFuncSetAttribute((void*)block_fft_kernel_C2C_FWD_INCREASE_OP_INV_NONE<external_image_t, FFT, invFFT, data_buffer_t, PreOpType, IntraOpType, PostOpType>, cudaFuncAttributeMaxDynamicSharedMemorySize, shared_memory));
                        if constexpr ( Rank == 2 ) {
                            MyFFTDebugAssertTrue(current_buffer == fastfft_internal_buffer_1, "current_buffer != fastfft_internal_buffer_1");
                            precheck;
                            block_fft_kernel_C2C_FWD_INCREASE_OP_INV_NONE<external_image_t, FFT, invFFT, data_buffer_t, PreOpType, IntraOpType, PostOpType><<<LP.gridDims, LP.threadsPerBlock, shared_memory, cudaStreamPerThread>>>(
                                    (external_image_t*)other_image_ptr, d_ptr.buffer_1, d_ptr.buffer_2, LP.mem_offsets, apparent_Q, workspace_fwd, workspace_inv, pre_op_functor, intra_op_functor, post_op_functor);
                            postcheck;
                            current_buffer = fastfft_internal_buffer_2;
                        }
                        else if constexpr ( Rank == 3 ) {
                            MyFFTDebugAssertTrue(current_buffer == fastfft_internal_buffer_2, "current_buffer != fastfft_internal_buffer_2");
                            precheck;
                            block_fft_kernel_C2C_FWD_INCREASE_OP_INV_NONE<external_image_t, FFT, invFFT, data_buffer_t, PreOpType, IntraOpType, PostOpType><<<LP.gridDims, LP.threadsPerBlock, shared_memory, cudaStreamPerThread>>>(
                                    (external_image_t*)other_image_ptr, d_ptr.buffer_2, d_ptr.buffer_1, LP.mem_offsets, apparent_Q, workspace_fwd, workspace_inv, pre_op_functor, intra_op_functor, post_op_functor);
                            postcheck;
                            current_buffer = fastfft_internal_buffer_1;
                        }

#endif
                    }

                    // do something
                }
                break;
            }
            default: {
                MyFFTDebugAssertFalse(true, "Unsupported transform stage.");
                break;
            }

        } // kernel type switch.
    } // constexpr if on thread/block

    //
} // end set and launch kernel

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// Some helper functions that are annoyingly long to have in the header.
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
template <class ComputeBaseType, class InputType, class OtherImageType, int Rank>
void FourierTransformer<ComputeBaseType, InputType, OtherImageType, Rank>::GetTransformSize(KernelType kernel_type) {
    // Set member variable transform_size.N (.P .L .Q)

    if ( IsR2CType(kernel_type) ) {
        AssertDivisibleAndFactorOf2(kernel_type, std::max(fwd_dims_in.x, fwd_dims_out.x), std::min(fwd_dims_in.x, fwd_dims_out.x));
    }
    else if ( IsC2RType(kernel_type) ) {
        // FIXME
        if ( kernel_type == c2r_decrease_XY ) {
            AssertDivisibleAndFactorOf2(kernel_type, std::max(inv_dims_in.x, inv_dims_out.x), std::max(inv_dims_in.x, inv_dims_out.x));
        }
        else {
            AssertDivisibleAndFactorOf2(kernel_type, std::max(inv_dims_in.x, inv_dims_out.x), std::min(inv_dims_in.x, inv_dims_out.x));
        }
    }
    else {
        // C2C type
        if ( IsForwardType(kernel_type) ) {
            switch ( transform_dimension ) {
                case 1: {
                    AssertDivisibleAndFactorOf2(kernel_type, std::max(fwd_dims_in.x, fwd_dims_out.x), std::min(fwd_dims_in.x, fwd_dims_out.x));
                    break;
                }
                case 2: {
                    if ( kernel_type == generic_fwd_increase_op_inv_none ) {
                        // FIXME
                        AssertDivisibleAndFactorOf2(kernel_type, std::max(fwd_dims_in.y, fwd_dims_out.y), std::max(fwd_dims_in.y, fwd_dims_out.y));
                    }
                    else {
                        AssertDivisibleAndFactorOf2(kernel_type, std::max(fwd_dims_in.y, fwd_dims_out.y), std::min(fwd_dims_in.y, fwd_dims_out.y));
                    }
                    break;
                }
                case 3: {
                    if ( IsTransormAlongZ(kernel_type) ) {
                        AssertDivisibleAndFactorOf2(kernel_type, std::max(fwd_dims_in.z, fwd_dims_out.z), std::min(fwd_dims_in.z, fwd_dims_out.z));
                    }
                    else {
                        AssertDivisibleAndFactorOf2(kernel_type, std::max(fwd_dims_in.y, fwd_dims_out.y), std::min(fwd_dims_in.y, fwd_dims_out.y));
                    }

                    break;
                }

                default: {
                    MyFFTDebugAssertTrue(false, "ERROR: Invalid transform dimension for c2c fwd type.\n");
                }
            }
        }
        else {
            switch ( transform_dimension ) {
                case 1: {
                    AssertDivisibleAndFactorOf2(kernel_type, std::max(inv_dims_in.x, inv_dims_out.x), std::min(inv_dims_in.x, inv_dims_out.x));
                    break;
                }
                case 2: {
                    if ( kernel_type == xcorr_fwd_none_inv_decrease ) {
                        // FIXME, for now using full transform
                        AssertDivisibleAndFactorOf2(kernel_type, std::max(inv_dims_in.y, inv_dims_out.y), std::max(inv_dims_in.y, inv_dims_out.y));
                    }
                    else {
                        AssertDivisibleAndFactorOf2(kernel_type, std::max(inv_dims_in.y, inv_dims_out.y), std::min(inv_dims_in.y, inv_dims_out.y));
                    }
                    break;
                }
                case 3: {
                    if ( IsTransormAlongZ(kernel_type) ) {
                        AssertDivisibleAndFactorOf2(kernel_type, std::max(inv_dims_in.z, inv_dims_out.z), std::min(inv_dims_in.z, inv_dims_out.z));
                    }
                    else {
                        AssertDivisibleAndFactorOf2(kernel_type, std::max(inv_dims_in.y, inv_dims_out.y), std::min(inv_dims_in.y, inv_dims_out.y));
                    }

                    break;
                }

                default: {
                    MyFFTDebugAssertTrue(false, "ERROR: Invalid transform dimension for c2c inverse type.\n");
                }
            }
        }
    }

} // end GetTransformSize function

template <class ComputeBaseType, class InputType, class OtherImageType, int Rank>
void FourierTransformer<ComputeBaseType, InputType, OtherImageType, Rank>::GetTransformSize_thread(KernelType kernel_type, int thread_fft_size) {

    transform_size.P = thread_fft_size;

    switch ( kernel_type ) {
        case r2c_decomposed:
            transform_size.N = fwd_dims_in.x;
            break;
        case r2c_decomposed_transposed:
            transform_size.N = fwd_dims_in.x;
            break;
        case c2c_fwd_decomposed:
            // FIXME fwd vs inv
            if ( fwd_dims_in.y == 1 )
                transform_size.N = fwd_dims_in.x;
            else
                transform_size.N = fwd_dims_in.y;
            break;
        case c2c_inv_decomposed:
            // FIXME fwd vs inv
            if ( fwd_dims_in.y == 1 )
                transform_size.N = fwd_dims_in.x;
            else
                transform_size.N = fwd_dims_in.y;
            break;
        case c2r_decomposed:
            transform_size.N = inv_dims_out.x;
            break;
        case c2r_decomposed_transposed:
            transform_size.N = inv_dims_out.x;
            break;
        case xcorr_decomposed:
            // FIXME fwd vs inv
            if ( fwd_dims_in.y == 1 )
                transform_size.N = fwd_dims_out.x; // FIXME should probably throw an error for now.
            else
                transform_size.N = fwd_dims_out.y; // does fwd_dims_in make sense?

            break;
        default:
            std::cerr << "Function GetTransformSize_thread does not recognize the kernel type ( " << KernelName[kernel_type] << " )" << std::endl;
            exit(-1);
    }

    if ( transform_size.N % transform_size.P != 0 ) {
        std::cerr << "Thread based decompositions must factor by thread_fft_size (" << thread_fft_size << ") in the current implmentations." << std::endl;
        exit(-1);
    }
    transform_size.Q = transform_size.N / transform_size.P;
} // end GetTransformSize_thread function

template <class ComputeBaseType, class InputType, class OtherImageType, int Rank>
LaunchParams FourierTransformer<ComputeBaseType, InputType, OtherImageType, Rank>::SetLaunchParameters(KernelType kernel_type) {
    /*
    Assuming:
    1) r2c/c2r imply forward/inverse transform. 
       c2c_padded implies forward transform.
    2) for 2d or 3d transforms the x/y dimensions are transposed in momentum space during store on the 1st set of 1ds transforms.
    3) if 1d then z = y = 1.

    threadsPerBlock = size/threads_per_fft (for thread based transforms)
                    = size of fft ( for block based transforms ) NOTE: Something in cufftdx seems to be very picky about this. Launching > threads seem to cause problems.
    gridDims = number of 1d FFTs, placed on blockDim perpendicular
    shared_input/output = number of elements reserved in dynamic shared memory. TODO add a check on minimal (48k) and whether this should be increased (depends on Arch)
    physical_x_input/output = number of elements along the fast (x) dimension, depends on fftw padding && whether the memory is currently transposed in x/y
    twiddle_in = +/- 2*PI/Largest dimension : + for the inverse transform
    Q = number of sub-transforms
  */
    MyFFTDebugAssertTrue(elements_per_thread_complex > 1 && elements_per_thread_complex < 33 && IsAPowerOfTwo(elements_per_thread_complex), "elements_per_thead_complex must be a power of 2 and > 2.");
    const int    ept = elements_per_thread_complex;
    LaunchParams L;

    // This is the same for all kernels as set in AssertDivisibleAndFactorOf2()
    L.Q = transform_size.Q;

    // Set the twiddle factor, only differ in sign between fwd/inv transforms.
    // For mixed kernels (eg. xcorr_* the size type is defined by where the size change happens.
    // FIXME fwd_increase (oversampling) xcorr -> inv decrease (peak search) is a likely algorithm, that will not fit with this logic.
    SizeChangeType::Enum size_change_type;
    if ( IsForwardType(kernel_type) ) {
        size_change_type = fwd_size_change_type;
        L.twiddle_in = L.twiddle_in = -2 * pi_v<float> / transform_size.N;
    }
    else {
        size_change_type = inv_size_change_type;
        L.twiddle_in = L.twiddle_in = 2 * pi_v<float> / transform_size.N;
    }

    // Set the thread block dimensions
    if ( IsThreadType(kernel_type) ) {
        L.threadsPerBlock = dim3(transform_size.Q, 1, 1);
    }
    else {
        if ( size_change_type == SizeChangeType::decrease ) {
            L.threadsPerBlock = dim3(transform_size.P / ept, transform_size.Q, 1);
        }
        else {
            // In the current xcorr methods that have INCREASE, explicit zero padding is used, so this will be overridden (overrode?) with transform_size.N
            L.threadsPerBlock = dim3(transform_size.P / ept, 1, 1);
        }
    }

    // Set the shared mem sizes, which depend on the size_change_type
    switch ( size_change_type ) {
        case SizeChangeType::no_change: {
            // no shared memory is needed outside that for the FFT itself.
            // For C2C kernels of size_type increase, the shared output may be reset below in order to store for coalesced global writes.
            L.mem_offsets.shared_input  = 0;
            L.mem_offsets.shared_output = 0;
            break;
        }
        case SizeChangeType::decrease: {
            // Prior to reduction, we must be able to store the full transform. An alternate algorithm with multiple reads would relieve this dependency and
            // may be worth considering if L2 cache residence on Ampere is an effective way to reduce repeated Globabl memory access.
            // Note: that this shared memory is not static, in the sense that it is used both for temporory fast storage, as well as the calculation of the FFT. The max of those two requirments is calculated per kernel.
            L.mem_offsets.shared_input = transform_size.N;
            if ( IsR2CType(kernel_type) ) {
                L.mem_offsets.shared_output = 0;
            }
            else {
                L.mem_offsets.shared_output = transform_size.N;
            } // TODO this line is just from case increase, haven't thought about it.
            break;
        }
        case SizeChangeType::increase: {
            // We want to re-use the input memory as we loop over construction of the full FFT. This shared memory is independent of the
            // memory used for the FFT itself.
            L.mem_offsets.shared_input = transform_size.P;
            if ( IsR2CType(kernel_type) ) {
                L.mem_offsets.shared_output = 0;
            }
            else {
                L.mem_offsets.shared_output = transform_size.N;
            }
            // Note: This is overwritten in the C2C methods as it depends on 1d vs 2d and fwd vs inv.
            break;
        }
        default: {
            MyFFTDebugAssertTrue(false, "Unknown size_change_type ( " + std::to_string(size_change_type) + " )");
        }
    } // switch on size change

    // Set the grid dimensions and pixel pitch
    if ( IsR2CType(kernel_type) ) {
        L.gridDims                      = dim3(1, fwd_dims_in.y, fwd_dims_in.z);
        L.mem_offsets.physical_x_input  = fwd_dims_in.w * 2; // scalar type, natural
        L.mem_offsets.physical_x_output = fwd_dims_out.w;

        if ( kernel_type == r2c_none_XZ || kernel_type == r2c_increase_XZ ) {
            L.threadsPerBlock.y = XZ_STRIDE;
            L.gridDims.z /= XZ_STRIDE;
        }
    }
    else if ( IsC2RType(kernel_type) ) {
        // This is always the last op, so if there is a size change, it will have happened once on C2C, reducing the number of blocks
        L.gridDims                      = dim3(1, inv_dims_out.y, inv_dims_out.z);
        L.mem_offsets.physical_x_input  = inv_dims_in.w;
        L.mem_offsets.physical_x_output = inv_dims_out.w * 2;
    }
    else // C2C type
    {
        // All dimensions have the same physical x dims for C2C transforms
        if ( IsForwardType(kernel_type) ) {
            // If 1d, this is implicitly a complex valued input, s.t. fwd_dims_in.x = fwd_dims_in.w.) But if fftw_padding is allowed false this may not be true.
            L.mem_offsets.physical_x_input  = fwd_dims_in.w;
            L.mem_offsets.physical_x_output = fwd_dims_out.w;
        }
        else {
            L.mem_offsets.physical_x_input  = inv_dims_in.w;
            L.mem_offsets.physical_x_output = inv_dims_out.w;
        }

        switch ( transform_dimension ) {
            case 1: {
                L.gridDims = dim3(1, 1, 1);
                break;
            }
            case 2: {
                if ( IsForwardType(kernel_type) ) {
                    L.gridDims = dim3(1, fwd_dims_out.w, fwd_dims_out.z);
                }
                else {
                    L.gridDims = dim3(1, inv_dims_out.w, inv_dims_out.z);
                }
                break;
            }
            case 3: {
                if ( IsTransormAlongZ(kernel_type) ) {
                    // When transforming along the (logical) Z-dimension, The Z grid dimensions for a 3d kernel are used to indicate the transposed x coordinate. (physical Z)
                    if ( IsForwardType(kernel_type) ) {
                        // Always 1 | index into physical y, yet to be expanded | logical x (physical Z) is already expanded (dims_out)
                        L.gridDims = dim3(1, fwd_dims_in.y, fwd_dims_out.w);
                    }
                    else {
                        // FIXME only tested for NONE size change type, dims_in might be correct, not stopping to think about it now.
                        L.gridDims = dim3(1, inv_dims_out.y, inv_dims_out.w);
                    }
                }
                else {
                    if ( IsForwardType(kernel_type) ) {
                        L.gridDims = dim3(1, fwd_dims_out.w, fwd_dims_out.z);
                    }
                    else {
                        L.gridDims = dim3(1, inv_dims_out.w, inv_dims_out.z);
                    }
                }
                break;
            } // 3 dimensional case
            default: {
                MyFFTDebugAssertTrue(false, "Unknown transform_dimension ( " + std::to_string(transform_dimension) + " )");
            }
        }

        // Over ride for partial output coalescing
        if ( kernel_type == c2c_inv_none_XZ ) {
            L.threadsPerBlock.y = XZ_STRIDE;
            L.gridDims.z /= XZ_STRIDE;
        }
        if ( kernel_type == c2c_fwd_none_XYZ || kernel_type == c2c_inv_none_XYZ || kernel_type == c2c_fwd_increase_XYZ ) {
            L.threadsPerBlock.y = XZ_STRIDE;
            L.gridDims.y /= XZ_STRIDE;
        }
    }
    // FIXME
    // Some shared memory over-rides
    if ( kernel_type == c2c_inv_decrease || kernel_type == c2c_inv_increase ) {
        L.mem_offsets.shared_output = inv_dims_out.y;
    }

    // FIXME
    // Some xcorr overrides TODO try the DECREASE approcae
    if ( kernel_type == generic_fwd_increase_op_inv_none ) {
        // FIXME not correct for 3D
        L.threadsPerBlock = dim3(transform_size.N / ept, 1, 1);
    }

    if ( kernel_type == xcorr_fwd_none_inv_decrease ) {
        // FIXME not correct for 3D

        L.threadsPerBlock = dim3(transform_size.N / ept, 1, 1);
        // FIXME
        L.gridDims                      = dim3(1, fwd_dims_out.w, 1);
        L.mem_offsets.physical_x_input  = inv_dims_in.y;
        L.mem_offsets.physical_x_output = inv_dims_out.y;
    }
    return L;
}

void GetCudaDeviceProps(DeviceProps& dp) {
    int major = 0;
    int minor = 0;

    cudaErr(cudaGetDevice(&dp.device_id));
    cudaErr(cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, dp.device_id));
    cudaErr(cudaDeviceGetAttribute(&minor, cudaDevAttrComputeCapabilityMinor, dp.device_id));

    dp.device_arch = major * 100 + minor * 10;

    MyFFTRunTimeAssertTrue(dp.device_arch == 700 || dp.device_arch == 750 || dp.device_arch == 800 || dp.device_arch == 860 || dp.device_arch == 890, "FastFFT currently only supports compute capability [7.0, 7.5, 8.0, 8.6, 8.9].");

    cudaErr(cudaDeviceGetAttribute(&dp.max_shared_memory_per_block, cudaDevAttrMaxSharedMemoryPerBlock, dp.device_id));
    cudaErr(cudaDeviceGetAttribute(&dp.max_shared_memory_per_SM, cudaDevAttrMaxSharedMemoryPerMultiprocessor, dp.device_id));
    cudaErr(cudaDeviceGetAttribute(&dp.max_registers_per_block, cudaDevAttrMaxRegistersPerBlock, dp.device_id));
    cudaErr(cudaDeviceGetAttribute(&dp.max_persisting_L2_cache_size, cudaDevAttrMaxPersistingL2CacheSize, dp.device_id));
}

void CheckSharedMemory(int& memory_requested, DeviceProps& dp) {
    // Depends on GetCudaDeviceProps having been called, which should be happening in the constructor.
    // Throw an error if requesting more than allowed, otherwise, we'll set to requested and let the rest be L1 Cache.
    MyFFTRunTimeAssertFalse(memory_requested > dp.max_shared_memory_per_SM, "The shared memory requested is greater than permitted for this arch.");
    // if (memory_requested > dp.max_shared_memory_per_block) { memory_requested = dp.max_shared_memory_per_block; }
}

void CheckSharedMemory(unsigned int& memory_requested, DeviceProps& dp) {
    // Depends on GetCudaDeviceProps having been called, which should be happening in the constructor.
    // Throw an error if requesting more than allowed, otherwise, we'll set to requested and let the rest be L1 Cache.
    MyFFTRunTimeAssertFalse(memory_requested > dp.max_shared_memory_per_SM, "The shared memory requested is greater than permitted for this arch.");
    // if (memory_requested > dp.max_shared_memory_per_block) { memory_requested = dp.max_shared_memory_per_block; }
}

using namespace FastFFT::KernelFunction;
// my_functor, IKF_t

// 2d explicit instantiations

// TODO: Pass in functor types
// TODO: Take another look at the explicit NOOP vs nullptr and determine if it is really needed
#define INSTANTIATE(COMPUTE_BASE_TYPE, INPUT_TYPE, OTHER_IMAGE_TYPE, RANK)                                                                                                                       \
    template class FourierTransformer<COMPUTE_BASE_TYPE, INPUT_TYPE, OTHER_IMAGE_TYPE, RANK>;                                                                                                    \
                                                                                                                                                                                                 \
    template void FourierTransformer<COMPUTE_BASE_TYPE, INPUT_TYPE, OTHER_IMAGE_TYPE, RANK>::FwdFFT<std::nullptr_t,                                                                              \
                                                                                                    std::nullptr_t>(INPUT_TYPE*, INPUT_TYPE*, std::nullptr_t, std::nullptr_t);                   \
                                                                                                                                                                                                 \
    template void FourierTransformer<COMPUTE_BASE_TYPE, INPUT_TYPE, OTHER_IMAGE_TYPE, RANK>::InvFFT<std::nullptr_t,                                                                              \
                                                                                                    std::nullptr_t>(INPUT_TYPE*, INPUT_TYPE*, std::nullptr_t, std::nullptr_t);                   \
                                                                                                                                                                                                 \
    template void FourierTransformer<COMPUTE_BASE_TYPE, INPUT_TYPE, OTHER_IMAGE_TYPE, RANK>::FwdFFT<my_functor<float, 0, IKF_t::NOOP>,                                                           \
                                                                                                    my_functor<float, 0, IKF_t::NOOP>>(INPUT_TYPE*,                                              \
                                                                                                                                       INPUT_TYPE*,                                              \
                                                                                                                                       my_functor<float, 0, IKF_t::NOOP>,                        \
                                                                                                                                       my_functor<float, 0, IKF_t::NOOP>);                       \
                                                                                                                                                                                                 \
    template void FourierTransformer<COMPUTE_BASE_TYPE, INPUT_TYPE, OTHER_IMAGE_TYPE, RANK>::InvFFT<my_functor<float, 0, IKF_t::NOOP>,                                                           \
                                                                                                    my_functor<float, 0, IKF_t::NOOP>>(INPUT_TYPE*,                                              \
                                                                                                                                       INPUT_TYPE*,                                              \
                                                                                                                                       my_functor<float, 0, IKF_t::NOOP>,                        \
                                                                                                                                       my_functor<float, 0, IKF_t::NOOP>);                       \
                                                                                                                                                                                                 \
    template void FourierTransformer<COMPUTE_BASE_TYPE, INPUT_TYPE, OTHER_IMAGE_TYPE, RANK>::FwdImageInvFFT<my_functor<float, 0, IKF_t::NOOP>,                                                   \
                                                                                                            my_functor<float, 4, IKF_t::CONJ_MUL>,                                               \
                                                                                                            my_functor<float, 0, IKF_t::NOOP>>(INPUT_TYPE*,                                      \
                                                                                                                                               OTHER_IMAGE_TYPE*,                                \
                                                                                                                                               INPUT_TYPE*,                                      \
                                                                                                                                               my_functor<float, 0, IKF_t::NOOP>,                \
                                                                                                                                               my_functor<float, 4, IKF_t::CONJ_MUL>,            \
                                                                                                                                               my_functor<float, 0, IKF_t::NOOP>);               \
    template void FourierTransformer<COMPUTE_BASE_TYPE, INPUT_TYPE, OTHER_IMAGE_TYPE, RANK>::FwdImageInvFFT<my_functor<float, 2, IKF_t::SCALE>,                                                  \
                                                                                                            my_functor<float, 4, IKF_t::CONJ_MUL>,                                               \
                                                                                                            my_functor<float, 0, IKF_t::NOOP>>(INPUT_TYPE*,                                      \
                                                                                                                                               OTHER_IMAGE_TYPE*,                                \
                                                                                                                                               INPUT_TYPE*,                                      \
                                                                                                                                               my_functor<float, 2, IKF_t::SCALE>,               \
                                                                                                                                               my_functor<float, 4, IKF_t::CONJ_MUL>,            \
                                                                                                                                               my_functor<float, 0, IKF_t::NOOP>);               \
    template void FourierTransformer<COMPUTE_BASE_TYPE, INPUT_TYPE, OTHER_IMAGE_TYPE, RANK>::FwdImageInvFFT<my_functor<float, 0, IKF_t::NOOP>,                                                   \
                                                                                                            my_functor<float, 4, IKF_t::CONJ_MUL_THEN_SCALE>,                                    \
                                                                                                            my_functor<float, 0, IKF_t::NOOP>>(INPUT_TYPE*,                                      \
                                                                                                                                               OTHER_IMAGE_TYPE*,                                \
                                                                                                                                               INPUT_TYPE*,                                      \
                                                                                                                                               my_functor<float, 0, IKF_t::NOOP>,                \
                                                                                                                                               my_functor<float, 4, IKF_t::CONJ_MUL_THEN_SCALE>, \
                                                                                                                                               my_functor<float, 0, IKF_t::NOOP>);

INSTANTIATE(float, float, float2, 2);
INSTANTIATE(float, __half, float2, 2);
INSTANTIATE(float, float, __half2, 2);
INSTANTIATE(float, __half, __half2, 2);
#ifdef FastFFT_3d_instantiation
INSTANTIATE(float, float, float2, 3);
INSTANTIATE(float, __half, __half2, 3);
#endif
#undef INSTANTIATE

} // namespace FastFFT
