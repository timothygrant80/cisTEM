#include "gpu_core_headers.h"
#include "gpu_indexing_functions.h"

#include "TemplateMatchingCore.h"

#ifdef cisTEM_USING_FastFFT
#include "../../include/FastFFT/include/FastFFT.cuh"
#endif
// Implementation is in the header as it is only used here for now.
#include "projection_queue.cuh"

using namespace cistem_timer;

void TemplateMatchingCore::Init(MyApp*                    parent_pointer,
                                std::shared_ptr<GpuImage> wanted_template_reconstruction,
                                Image&                    wanted_input_image,
                                Image&                    current_projection,
                                float                     psi_max,
                                float                     psi_start,
                                float                     psi_step,
                                AnglesAndShifts&          angles,
                                EulerSearch&              global_euler_search,
                                const int2                pre_padding,
                                const int2                roi,
                                float                     histogram_sampling,
                                float                     stats_sampling,
                                int                       first_search_position,
                                int                       last_search_position,
                                ProgressBar*              my_progress,
                                long                      total_correlation_positions,
                                bool                      is_running_locally,
                                bool                      use_fast_fft,
                                bool                      use_gpu_prj,
                                int                       number_of_global_search_images_to_save)

{

    MyDebugAssertFalse(object_initialized_, "Init must only be called once!");
    MyDebugAssertFalse(wanted_input_image.is_in_real_space, "Input image must be in Fourier space");
    MyDebugAssertTrue(wanted_input_image.is_in_memory, "Input image must be in memory");
    object_initialized_ = true;

    this->use_gpu_prj   = use_gpu_prj;
    histogram_sampling_ = histogram_sampling;
    stats_sampling_     = stats_sampling;

    this->first_search_position          = first_search_position;
    this->last_search_position           = last_search_position;
    this->angles                         = angles;
    this->global_euler_search            = global_euler_search;
    this->n_global_search_images_to_save = number_of_global_search_images_to_save;

    MyDebugAssertFalse(number_of_global_search_images_to_save > 1, "Only one peak per search position is currently supported");

    this->psi_start = psi_start;
    this->psi_step  = psi_step;
    this->psi_max   = psi_max;

    this->use_fast_fft = use_fast_fft;

    // It seems that I need a copy for these - 1) confirm, 2) if already copying, maybe put straight into pinned mem with cudaHostMalloc
    input_image.CopyFrom(&wanted_input_image);

    this->current_projection.reserve(n_prjs);
    for ( int i = 0; i < n_prjs; i++ ) {
        this->current_projection.emplace_back(current_projection);
        d_current_projection.emplace_back(this->current_projection[i]);
    }
    if ( use_gpu_prj ) {
        template_gpu_shared = wanted_template_reconstruction;
    }

    d_input_image.Init(input_image);
    d_input_image.CopyHostToDevice(input_image);

    d_statistical_buffers_ptrs.push_back(&d_padded_reference);
    d_statistical_buffers_ptrs.push_back(&d_sum1);
    d_statistical_buffers_ptrs.push_back(&d_sumSq1);
    d_statistical_buffers_ptrs.push_back(&d_sum2);
    d_statistical_buffers_ptrs.push_back(&d_sumSq2);
    int n_2d_buffers = 0;
    for ( auto& buffer : d_statistical_buffers_ptrs ) {
        buffer->Allocate(d_input_image.dims.x, d_input_image.dims.y, 1, true);
        n_2d_buffers++;
    }

    d_statistical_buffers_ptrs.push_back(&d_max_intensity_projection);
    d_statistical_buffers_ptrs.push_back(&d_best_psi);
    d_statistical_buffers_ptrs.push_back(&d_best_theta);
    d_statistical_buffers_ptrs.push_back(&d_best_phi);
    for ( int i = n_2d_buffers; i < d_statistical_buffers_ptrs.size( ); i++ ) {
        d_statistical_buffers_ptrs[i]->Allocate(d_input_image.dims.x, d_input_image.dims.y, number_of_global_search_images_to_save, true);
    }

    this->pre_padding = pre_padding;
    this->roi         = roi;

    this->my_progress                 = my_progress;
    this->total_correlation_positions = total_correlation_positions;
    this->is_running_locally          = is_running_locally;

    this->parent_pointer = parent_pointer;

    // For now we are only working on the inner loop, so no need to track best_defocus and best_pixel_size

    // At the outset these are all empty cpu images, so don't xfer, just allocate on gpuDev

    // Transfer the input image_memory_should_not_be_deallocated
};

void TemplateMatchingCore::RunInnerLoop(Image& projection_filter, int threadIDX, long& current_correlation_position) {

    total_number_of_cccs_calculated   = 0;
    total_number_of_histogram_samples = 0;
    total_number_of_stats_samples     = 0;
    bool at_least_100                 = false;

    bool this_is_the_first_run_on_inner_loop = my_dist ? false : true;

#ifdef cisTEM_USING_FastFFT
    // FIXME: Move this to a new method in matche_template.cpp and reference it from a shared pointer.
    if ( use_fast_fft && this_is_the_first_run_on_inner_loop ) {
        // FastFFT pads from the upper left corner, so we need to shift the image so the origins coinicide
        d_input_image.PhaseShift(-(d_input_image.physical_address_of_box_center.x - d_current_projection[0].physical_address_of_box_center.x),
                                 -(d_input_image.physical_address_of_box_center.y - d_current_projection[0].physical_address_of_box_center.y),
                                 0);

        d_input_image.BackwardFFT( );

        FastFFT::FourierTransformer<float, float, float2, 2> FT;

        // TODO: overload that takes and short4's int4's instead of the individual values
        FT.SetForwardFFTPlan(input_image.logical_x_dimension, input_image.logical_y_dimension, d_input_image.logical_z_dimension, d_input_image.dims.x, d_input_image.dims.y, d_input_image.dims.z, true);
        FT.SetInverseFFTPlan(d_input_image.dims.x, d_input_image.dims.y, d_input_image.dims.z, d_input_image.dims.x, d_input_image.dims.y, d_input_image.dims.z, true);

        FT.FwdFFT(d_input_image.real_values);

        // We've done a round trip iFFT/FFT since the input image was normalized to STD 1.0, so re-normalize by 1/n
        d_input_image.is_in_real_space = false;
        // d_input_image.MultiplyByConstant(sqrtf(1.f / d_input_image.number_of_real_space_pixels));
        d_input_image.MultiplyByConstant(1.f / d_input_image.number_of_real_space_pixels / sqrtf(float(d_current_projection[0].number_of_real_space_pixels)));
    }
#endif

    if ( this_is_the_first_run_on_inner_loop ) {
        d_input_image.CopyFP32toFP16buffer(false);
        d_padded_reference.CopyFP32toFP16buffer(false);
        my_dist = std::make_unique<TM_EmpiricalDistribution<__half, __half2>>(d_input_image, pre_padding, roi, histogram_sampling_, stats_sampling_);
    }
    else {
        my_dist->ZeroHistogram( );
    }

    // Make sure we are starting with zeros
    for ( auto& buffer : d_statistical_buffers_ptrs ) {
        buffer->Zeros( );
    }

    //	cudaErr(cudaMemset(sum_sumsq,0,sizeof(Peaks)*d_input_image.real_memory_allocated));

    // Just for reference:
    // cudaStreamSynchronize: Blocks host until ALL work in the stream is completed
    // cudaStreamWaitEvent: Makes all future work in stream wait on an event. Since we are always using cudaStreamPerThread, this is not needed.

    cudaEvent_t mip_is_done_Event;

    cudaErr(cudaEventCreateWithFlags(&mip_is_done_Event, cudaEventBlockingSync));
#ifdef cisTEM_USING_FastFFT
    FastFFT::FourierTransformer<float, __half, __half2, 2> FT;

    // float scale_factor = powf((float)d_current_projection[0].number_of_real_space_pixels, -2.0);
    // float scale_factor = 1.f;
    float scale_factor = sqrtf(1.0f / float(d_input_image.number_of_real_space_pixels));

    FastFFT::KernelFunction::my_functor<float, 4, FastFFT::KernelFunction::CONJ_MUL_THEN_SCALE> conj_mul_then_scale(scale_factor);
    FastFFT::KernelFunction::my_functor<float, 0, FastFFT::KernelFunction::NOOP>                noop;

    if ( use_fast_fft ) {
        // TODO: overload that takes and short4's int4's instead of the individual values
        FT.SetForwardFFTPlan(current_projection[0].logical_x_dimension, current_projection[0].logical_y_dimension, current_projection[0].logical_z_dimension, d_padded_reference.dims.x, d_padded_reference.dims.y, d_padded_reference.dims.z, true);
        FT.SetInverseFFTPlan(d_padded_reference.dims.x, d_padded_reference.dims.y, d_padded_reference.dims.z, d_padded_reference.dims.x, d_padded_reference.dims.y, d_padded_reference.dims.z, true);
    }
#endif
    int   ccc_counter = 0;
    int   current_search_position;
    float average_on_edge;
    float average_of_reals;
    float temp_float;

    int thisDevice;
    cudaGetDevice(&thisDevice);
    wxPrintf("Thread %d is running on device %d\n", threadIDX, thisDevice);

    GpuImage d_projection_filter(projection_filter);
    if ( use_gpu_prj ) {
        d_projection_filter.CopyHostToDevice(projection_filter);
        // FIXME:
        d_projection_filter.CopyFP32toFP16buffer(false);
    }

    int             current_projection_idx = 0;
    int             current_mip_to_process = 0;
    int             total_mip_processed    = 0;
    ProjectionQueue projection_queue(n_prjs);
    // We need to make sure the host blocks on all setup work before we start to make projections,
    // since we are using more than one stream.
    cudaErr(cudaStreamSynchronize(cudaStreamPerThread));
    projection_queue.RecordProjectionReadyBlockingHost(current_projection_idx, cudaStreamPerThread);

    for ( current_search_position = first_search_position; current_search_position <= last_search_position; current_search_position++ ) {

        if ( current_search_position % 10 == 0 ) {
            wxPrintf("Starting position %d/ %d\n", current_search_position, last_search_position);
        }

        for ( float current_psi = psi_start; current_psi <= psi_max; current_psi += psi_step ) {

            constexpr float shifts_in_x_y                               = 0.0f;
            constexpr bool  apply_shifts                                = false;
            constexpr bool  swap_real_space_quadrants_during_projection = true;
            angles.Init(global_euler_search.list_of_search_parameters[current_search_position][0], global_euler_search.list_of_search_parameters[current_search_position][1], current_psi, shifts_in_x_y, shifts_in_x_y);

            current_projection_idx = projection_queue.GetAvailableProjectionIDX( );
            if ( use_gpu_prj ) {

                d_current_projection[current_projection_idx].is_in_real_space = false;
                constexpr float pixel_size                                    = 1.0f;
                constexpr float resolution_limit                              = 1.0f;
                float           real_space_binning_factor                     = 1.0f;

                if ( use_lerp_for_resizing ) {
                    real_space_binning_factor = binning_factor;
                }

                // template_gpu_shared.get( )
                d_current_projection[current_projection_idx].ExtractSliceShiftAndCtf(template_gpu_shared.get( ),
                                                                                     &d_projection_filter,
                                                                                     angles,
                                                                                     pixel_size,
                                                                                     real_space_binning_factor,
                                                                                     resolution_limit,
                                                                                     false,
                                                                                     swap_real_space_quadrants_during_projection,
                                                                                     apply_shifts,
                                                                                     true,
                                                                                     false,
                                                                                     true,
                                                                                     projection_queue.gpu_projection_stream[current_projection_idx]);

                average_of_reals = 0.f;
                average_on_edge  = 0.f;
                projection_queue.RecordGpuProjectionReadyStreamPerThreadWait(current_projection_idx);
                // Default GpuImage methods are in cudaStreamPerThread

                d_current_projection[current_projection_idx].BackwardFFT( );
            }
            else {
                // Make sure the previous copy from host -> device has completed before we start to make another projection.
                // Event is created as non-blocking so this is a busy-wait.
                MyDebugAssertFalse(cpu_template == nullptr, "Template reconstruction is not set with SetCpuTemplate");
                cpu_template->ExtractSlice(current_projection[current_projection_idx], angles, 1.0f, false);

                current_projection[current_projection_idx].SwapRealSpaceQuadrants( );
                current_projection[current_projection_idx].MultiplyPixelWise(projection_filter);
                current_projection[current_projection_idx].BackwardFFT( );
                average_on_edge = current_projection[current_projection_idx].ReturnAverageOfRealValuesOnEdges( );
                // We'll subtract average_on_edge in the normalization prior to scaling
                average_of_reals = current_projection[current_projection_idx].ReturnAverageOfRealValues( ) - average_on_edge;

                // For an intiial test, make projection_queue.cpu_prj_stream[current_projection_idx]
                // a public member.. if it works, make it private and return a reference instead

                d_current_projection[current_projection_idx].CopyHostToDevice(current_projection[current_projection_idx], false, false, projection_queue.gpu_projection_stream[current_projection_idx]);

                // We need to make sure the current cpu projection is not used by the host until the gpu has finished with it, which may be independent of the main work
                // in cudaStreamPerThread, this is a blocking event for the host (if the queue is otherwise full)
                projection_queue.RecordProjectionReadyBlockingHost(current_projection_idx, projection_queue.gpu_projection_stream[current_projection_idx]);
                // We need the main work in cudaStreamPerThread to wait on the transfer in this stream, which if the CPU thread is ahead, should be a non-blocking event
                projection_queue.RecordGpuProjectionReadyStreamPerThreadWait(current_projection_idx);

                // Note: I had deleted this in the dev branch for FastFFT. Review when possible
                // The average in the full padded image will be different;
                // average_of_reals *= ((float)d_current_projection[current_projection_idx].number_of_real_space_pixels / (float)d_padded_reference.number_of_real_space_pixels);
            }

            // For the host to execute the preceding line, it was to wait on the return value from ReturnSumOfSquares. This could be a bit of a performance regression as otherwise it can queue up all the reamining
            // GPU work and get back to calculating the next projection. The commented out method is an attempt around that, but currently the mips come out a little different a bit faster.

            if ( use_fast_fft ) {
#ifdef cisTEM_USING_FastFFT
                // float scale_factor = rsqrtf(d_current_projection[current_projection_idx].ReturnSumOfSquares( ) / (float)d_padded_reference.number_of_real_space_pixels - (average_of_reals * average_of_reals));
                // scale_factor /= powf((float)d_current_projection[current_projection_idx].number_of_real_space_pixels, 1.0);

                float scale_factor = powf(float(d_padded_reference.number_of_real_space_pixels), 1.0);
                scale_factor *= powf((float)d_current_projection[current_projection_idx].number_of_real_space_pixels, 1.5);
                scale_factor = 1.0f;
                // d_current_projection[current_projection_idx].MultiplyByConstant(scale_factor);
                d_current_projection[current_projection_idx].NormalizeRealSpaceStdDeviationAndCastToFp16(scale_factor, average_of_reals, average_on_edge);

                FT.FwdImageInvFFT(d_current_projection[current_projection_idx].real_values_fp16, (__half2*)d_input_image.complex_values_fp16, my_dist->GetCCFArray(current_mip_to_process), noop, conj_mul_then_scale, noop);

                if ( use_gpu_prj ) {
                    // Note the stream change will not affect the padded projection
                    projection_queue.RecordProjectionReadyBlockingHost(current_projection_idx, cudaStreamPerThread);
                }

#endif
            }
            else {
                // The average in the full padded image will be different;
                average_of_reals *= ((float)d_current_projection[current_projection_idx].number_of_real_space_pixels / (float)d_padded_reference.number_of_real_space_pixels);
                d_current_projection[current_projection_idx].NormalizeRealSpaceStdDeviation(float(d_padded_reference.number_of_real_space_pixels), average_of_reals, average_on_edge);
                d_current_projection[current_projection_idx].ClipInto(&d_padded_reference, 0, false, 0, 0, 0, 0);
                // d_current_projection[current_projection_idx].MultiplyByConstant(rsqrtf(d_current_projection[current_projection_idx].ReturnSumOfSquares( ) / (float)d_padded_reference.number_of_real_space_pixels - (average_of_reals * average_of_reals)));

                if ( use_gpu_prj ) {
                    // Note the stream change will not affect the padded projection
                    projection_queue.RecordProjectionReadyBlockingHost(current_projection_idx, cudaStreamPerThread);
                }

                d_padded_reference.ForwardFFT(false);
                //      d_padded_reference.ForwardFFTAndClipInto(d_current_projection,false);
                d_padded_reference.BackwardFFTAfterComplexConjMul(d_input_image.complex_values_fp16, true, my_dist->GetCCFArray(current_mip_to_process));
            }
            // d_padded_reference.MultiplyByConstant(rsqrtf(d_padded_reference.ReturnSumOfSquares( ) / (float)d_padded_reference.number_of_real_space_pixels));

            my_dist->UpdateHostAngleArrays(current_mip_to_process, current_psi, global_euler_search.list_of_search_parameters[current_search_position][1], global_euler_search.list_of_search_parameters[current_search_position][0]);

            current_mip_to_process++;
            if ( current_mip_to_process == my_dist->n_imgs_to_process_at_once( ) ) {
                // Make sure the last stack has been processed before we start the next one
                my_dist->MakeHostWaitOnMipStackIsReadyEvent( );
                // On the first loop this will not do anything, so we can change the active_idx, and move forward to calculate the alternate stack of ccfs while the mip works on this one

                total_mip_processed += current_mip_to_process;
                // current_mip_to_process only matters after the main loop, the TM empirical dist will also update the active_idx_ before returning from Accumulate distribution
                my_dist->AccumulateDistribution(current_mip_to_process, total_number_of_histogram_samples, total_number_of_stats_samples);

                // We've queued up all the work for the current stack, so record the event that will be used to block the host until the stack is ready
                my_dist->RecordMipStackIsReadyBlockingHost( );

                current_mip_to_process = 0;
            }

            ccc_counter++;
            total_number_of_cccs_calculated++;

            if ( ccc_counter % 100 == 0 ) {
                my_dist->MakeHostWaitOnMipStackIsReadyEvent( );
                my_dist->CopySumAndSumSqAndZero(d_sum1, d_sumSq1);
                at_least_100 = true;
            }

            if ( ccc_counter % 10000 == 0 ) {
                // if we are in this block, we must also have been in the % 100 block, so no need to sync again
                d_sum2.AddImage(d_sum1);
                d_sum1.Zeros( );

                d_sumSq2.AddImage(d_sumSq1);
                d_sumSq1.Zeros( );
            }

            current_projection[current_projection_idx].is_in_real_space = false;
            d_padded_reference.is_in_real_space                         = true;

            //			first_loop_complete = true;

            if ( is_running_locally ) {
                if ( ReturnThreadNumberOfCurrentThread( ) == 0 ) {
                    current_correlation_position++;
                    if ( current_correlation_position > total_correlation_positions )
                        current_correlation_position = total_correlation_positions;
                    my_progress->Update(current_correlation_position);
                }
            }
            else {
                temp_float             = current_correlation_position;
                JobResult* temp_result = new JobResult;
                temp_result->SetResult(1, &temp_float);
                parent_pointer->AddJobToResultQueue(temp_result);
            }
        } // loop over psi angles

        // The current goal is to have only one peak per search position.
        if ( n_global_search_images_to_save > 1 )
            UpdateSecondaryPeaks( );

    } // end of outer loop euler sphere position

    projection_queue.PrintTimes( );

    wxPrintf("\t\t\ntotal number %d, total mips %d\n", ccc_counter, total_mip_processed);

    // If we have a total number of cccs that is not a multiple of n_mips_to_process_at_once, we need to process the remaining mips
    // Make sure the last stack has been processed before we start the next one
    my_dist->MakeHostWaitOnMipStackIsReadyEvent( );
    if ( current_mip_to_process > 0 ) {

        // On the first loop this will not do anything, so we can change the active_idx, and move forward to calculate the alternate stack of ccfs while the mip works on this one

        total_mip_processed += current_mip_to_process;
        // current_mip_to_process only matters after the main loop, the TM empirical dist will also update the active_idx_ before returning from Accumulate distribution
        my_dist->AccumulateDistribution(current_mip_to_process, total_number_of_histogram_samples, total_number_of_stats_samples);

        // We've queued up all the work for the current stack, so record the event that will be used to block the host until the stack is ready
        my_dist->RecordMipStackIsReadyBlockingHost( );
        my_dist->MakeHostWaitOnMipStackIsReadyEvent( );

        // This is run in cudaStreamPerThread
        my_dist->CopySumAndSumSqAndZero(d_sum1, d_sumSq1);
    }
    else {
        // if somehow we search less than 100 positions, we never will have run the above code
        if ( ~at_least_100 ) {
            my_dist->CopySumAndSumSqAndZero(d_sum1, d_sumSq1);
        }
    }

    d_sum2.AddImage(d_sum1);
    d_sumSq2.AddImage(d_sumSq1);

    my_dist->MipToImage(d_max_intensity_projection,
                        d_best_psi,
                        d_best_theta,
                        d_best_phi);

    my_dist->FinalAccumulate( );

    if ( n_global_search_images_to_save > 1 ) {
        cudaErr(cudaFreeAsync(secondary_peaks, cudaStreamPerThread));
    }

    cudaErr(cudaStreamSynchronize(cudaStreamPerThread));
}

__global__ void
UpdateSecondaryPeaksKernel(__half*   secondary_peaks,
                           __half2*  mip_psi,
                           __half2*  theta_phi,
                           const int NY,
                           const int NX) {

    //	When returning more than one search result, the peaks are stored in a 3d array,
    // numel * n_peaks * 4 (mip, psi, theta, phi)
    int best_index = 0;
    int offset     = 0;
    for ( int img_index = blockIdx.x * blockDim.x + threadIdx.x; img_index < NX; img_index += blockDim.x * gridDim.x ) {

        best_index = NY;
        for ( int i_peak = 0; i_peak < NY; i_peak++ ) {
            // Check to see if any peak from this search position is in the top n_peaks scores
            if ( __low2half(mip_psi[img_index]) > secondary_peaks[img_index + i_peak * NX] ) {
                best_index = i_peak;
                break;
            }
        }

        // If we didn't find a better peak, this loop will not execute
        // We have a numel * n_peaks * 4 (score, psi, theta, phi) array
        for ( int worst_peak = NY - 1; worst_peak > best_index; worst_peak-- ) {
            offset = img_index + NX * worst_peak;
            // Move the worst peak down one
            secondary_peaks[offset] = secondary_peaks[offset - NX];
            // Psi
            offset += NX * NY;
            secondary_peaks[offset] = secondary_peaks[offset - NX];
            // Theta
            offset += NX * NY;
            secondary_peaks[offset] = secondary_peaks[offset - NX];
            // Phi
            offset += NX * NY;
            secondary_peaks[offset] = secondary_peaks[offset - NX];
        }
        // Now insert the new peak
        if ( best_index < NY ) {
            offset                  = img_index + best_index * NX;
            secondary_peaks[offset] = __low2half(mip_psi[img_index]);
            // Psi
            offset += NX * NY;
            secondary_peaks[offset] = __high2half(mip_psi[img_index]);
            // Theta
            offset += NX * NY;
            secondary_peaks[offset] = __low2half(theta_phi[img_index]);
            // Phi
            offset += NX * NY;
            secondary_peaks[offset] = __high2half(theta_phi[img_index]);
        }
    }
}

void TemplateMatchingCore::UpdateSecondaryPeaks( ) {

    precheck;
    // N
    d_padded_reference.ReturnLaunchParametersLimitSMs(5.f, 1024);

    UpdateSecondaryPeaksKernel<<<d_padded_reference.gridDims, d_padded_reference.threadsPerBlock, 0, cudaStreamPerThread>>>((__half*)secondary_peaks,
                                                                                                                            mip_psi,
                                                                                                                            theta_phi,
                                                                                                                            n_global_search_images_to_save,
                                                                                                                            (int)d_padded_reference.real_memory_allocated);
    postcheck;

    // We need to reset this each outer angle search or we'll never see new maximums
    cudaErr(cudaMemsetAsync(mip_psi, 0, sizeof(__half2) * d_input_image.real_memory_allocated, cudaStreamPerThread));
    cudaErr(cudaMemsetAsync(theta_phi, 0, sizeof(__half2) * d_input_image.real_memory_allocated, cudaStreamPerThread));
}
