#include "gpu_core_headers.h"
#include "gpu_indexing_functions.h"

#include "TemplateMatchingCore.h"

#ifdef ENABLE_FastFFT
#include "../ext/FastFFT/include/FastFFT.cuh"
#endif
// Implementation is in the header as it is only used here for now.
#include "projection_queue.cuh"

using namespace cistem_timer;

constexpr bool use_gpu_prj               = false;
constexpr int  n_mips_to_process_at_once = 10;

static_assert(n_mips_to_process_at_once == 1 || n_mips_to_process_at_once == 10, "n_mips_to_process_at_once must be 1 or 10");

void TemplateMatchingCore::Init(MyApp*           parent_pointer,
                                Image&           wanted_template_reconstruction,
                                Image&           wanted_input_image,
                                Image&           current_projection,
                                float            pixel_size_search_range,
                                float            pixel_size_step,
                                float            pixel_size,
                                float            defocus_search_range,
                                float            defocus_step,
                                float            defocus1,
                                float            defocus2,
                                float            psi_max,
                                float            psi_start,
                                float            psi_step,
                                AnglesAndShifts& angles,
                                EulerSearch&     global_euler_search,
                                float            histogram_min_scaled,
                                float            histogram_step_scaled,
                                int              histogram_number_of_bins,
                                int              max_padding,
                                int              first_search_position,
                                int              last_search_position,
                                ProgressBar*     my_progress,
                                long             total_correlation_positions,
                                bool             is_running_locally,
                                int              number_of_global_search_images_to_save)

{

    MyDebugAssertFalse(object_initialized_, "Init must only be called once!");
    object_initialized_ = true;

    this->first_search_position          = first_search_position;
    this->last_search_position           = last_search_position;
    this->angles                         = angles;
    this->global_euler_search            = global_euler_search;
    this->n_global_search_images_to_save = number_of_global_search_images_to_save;

    this->psi_start = psi_start;
    this->psi_step  = psi_step;
    this->psi_max   = psi_max;

    // It seems that I need a copy for these - 1) confirm, 2) if already copying, maybe put straight into pinned mem with cudaHostMalloc
    template_reconstruction.CopyFrom(&wanted_template_reconstruction);
    input_image.CopyFrom(&wanted_input_image);

    this->current_projection.reserve(n_prjs);
    for ( int i = 0; i < n_prjs; i++ ) {
        this->current_projection.emplace_back(current_projection);
        d_current_projection.emplace_back(this->current_projection[i]);
    }
    if ( use_gpu_prj ) {
        // FIXME: for intial testing, we want to compare GPU and CPU projections, so make a copy
        Image tmp_vol = template_reconstruction;
        if ( ! this->is_gpu_3d_swapped ) {
            tmp_vol.SwapRealSpaceQuadrants( );
            tmp_vol.BackwardFFT( );
            tmp_vol.SwapFourierSpaceQuadrants(true);
            this->is_gpu_3d_swapped = true;
        }
        // TODO: confirm you need the real-values allocated

        this->template_gpu.InitializeBasedOnCpuImage(tmp_vol, false, true);
        this->template_gpu.CopyHostToDeviceTextureComplex3d(tmp_vol);
    }

    d_input_image.Init(input_image);
    d_input_image.CopyHostToDevice(input_image);

    d_statistical_buffers.push_back(&d_padded_reference);
    d_statistical_buffers.push_back(&d_sum1);
    d_statistical_buffers.push_back(&d_sumSq1);
    d_statistical_buffers.push_back(&d_sum2);
    d_statistical_buffers.push_back(&d_sumSq2);
    d_statistical_buffers.push_back(&d_sum3);
    d_statistical_buffers.push_back(&d_sumSq3);
    int n_2d_buffers = 0;
    for ( auto& buffer : d_statistical_buffers ) {
        buffer->Allocate(d_input_image.dims.x, d_input_image.dims.y, 1, true);
        n_2d_buffers++;
    }

    d_statistical_buffers.push_back(&d_max_intensity_projection);
    d_statistical_buffers.push_back(&d_best_psi);
    d_statistical_buffers.push_back(&d_best_theta);
    d_statistical_buffers.push_back(&d_best_phi);
    for ( int i = n_2d_buffers; i < d_statistical_buffers.size( ); i++ ) {
        d_statistical_buffers[i]->Allocate(d_input_image.dims.x, d_input_image.dims.y, number_of_global_search_images_to_save, true);
    }

    this->histogram_max_padding = max_padding;
    this->histogram_min_scaled  = histogram_min_scaled;
    this->histogram_step_scaled = histogram_step_scaled;

    this->my_progress                 = my_progress;
    this->total_correlation_positions = total_correlation_positions;
    this->is_running_locally          = is_running_locally;

    this->parent_pointer = parent_pointer;

    // For now we are only working on the inner loop, so no need to track best_defocus and best_pixel_size

    // At the outset these are all empty cpu images, so don't xfer, just allocate on gpuDev

    // Transfer the input image_memory_should_not_be_deallocated
};

void TemplateMatchingCore::RunInnerLoop(Image& projection_filter, float c_pixel, float c_defocus, int threadIDX, long& current_correlation_position) {

    // This should probably just be a unique pointer and not a vector
    if ( my_dist.empty( ) )
        my_dist.emplace_back(d_input_image, histogram_min_scaled, histogram_step_scaled, histogram_max_padding, n_mips_to_process_at_once, cudaStreamPerThread);
    else
        my_dist.at(0).ZeroHistogram( );

    // Make sure we are starting with zeros
    for ( auto& buffer : d_statistical_buffers ) {
        buffer->Zeros( );
    }

    this->c_defocus                 = c_defocus;
    this->c_pixel                   = c_pixel;
    total_number_of_cccs_calculated = 0;

    // Either do not delete the single precision, or add in a copy here so that each loop over defocus vals
    // have a copy to work with. Otherwise this will not exist on the second loop
    d_input_image.CopyFP32toFP16buffer(false);
    d_padded_reference.CopyFP32toFP16buffer(false);

    __half* psi_array;
    __half* theta_array;
    __half* phi_array;
    __half* d_psi_array;
    __half* d_theta_array;
    __half* d_phi_array;
    __half* ccf_array;

    if constexpr ( n_mips_to_process_at_once > 1 ) {
        psi_array   = new __half[n_mips_to_process_at_once];
        theta_array = new __half[n_mips_to_process_at_once];
        phi_array   = new __half[n_mips_to_process_at_once];

        cudaErr(cudaMallocAsync((void**)&d_psi_array, sizeof(__half) * n_mips_to_process_at_once, cudaStreamPerThread));
        cudaErr(cudaMallocAsync((void**)&d_theta_array, sizeof(__half) * n_mips_to_process_at_once, cudaStreamPerThread));
        cudaErr(cudaMallocAsync((void**)&d_phi_array, sizeof(__half) * n_mips_to_process_at_once, cudaStreamPerThread));
        cudaErr(cudaMallocAsync((void**)&ccf_array, sizeof(__half) * n_mips_to_process_at_once * d_input_image.real_memory_allocated, cudaStreamPerThread));
    }
    cudaErr(cudaMallocAsync((void**)&mip_psi, sizeof(__half2) * d_input_image.real_memory_allocated, cudaStreamPerThread));
    cudaErr(cudaMallocAsync((void**)&theta_phi, sizeof(__half2) * d_input_image.real_memory_allocated, cudaStreamPerThread));
    cudaErr(cudaMallocAsync((void**)&sum_sumsq, sizeof(__half2) * d_input_image.real_memory_allocated, cudaStreamPerThread));
    cudaErr(cudaMemsetAsync(mip_psi, 0, sizeof(__half2) * d_input_image.real_memory_allocated, cudaStreamPerThread));
    cudaErr(cudaMemsetAsync(theta_phi, 0, sizeof(__half2) * d_input_image.real_memory_allocated, cudaStreamPerThread));
    if ( n_global_search_images_to_save > 1 ) {
        cudaErr(cudaMallocAsync((void**)&secondary_peaks, sizeof(__half) * d_input_image.real_memory_allocated * n_global_search_images_to_save * 4, cudaStreamPerThread));
        cudaErr(cudaMemsetAsync(secondary_peaks, 0, sizeof(__half) * d_input_image.real_memory_allocated * n_global_search_images_to_save * 4, cudaStreamPerThread));
    }
    //	cudaErr(cudaMemset(sum_sumsq,0,sizeof(Peaks)*d_input_image.real_memory_allocated));

    // Just for reference:
    // cudaStreamSynchronize: Blocks host until ALL work in the stream is completed
    // cudaStreamWaitEvent: Makes all future work in stream wait on an event. Since we are always using cudaStreamPerThread, this is not needed.

    cudaEvent_t mip_is_done_Event;

    cudaErr(cudaEventCreateWithFlags(&mip_is_done_Event, cudaEventBlockingSync));

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

            angles.Init(global_euler_search.list_of_search_parameters[current_search_position][0], global_euler_search.list_of_search_parameters[current_search_position][1], current_psi, 0.0, 0.0);
            //			current_projection.SetToConstant(0.0f); // This also sets the FFT padding to zero

            current_projection_idx = projection_queue.GetAvailableProjectionIDX( );
            if ( use_gpu_prj ) {

                d_current_projection[current_projection_idx].is_in_real_space = false;
                d_current_projection[current_projection_idx].ExtractSliceShiftAndCtf(&template_gpu, &d_projection_filter, angles, 1.0, 1.0, false, true, true, true, false, true, projection_queue.gpu_projection_stream[current_projection_idx]);
                average_of_reals = 0.f;
                average_on_edge  = 0.f;
                projection_queue.RecordGpuProjectionReadyStreamPerThreadWait(current_projection_idx);
                d_current_projection[current_projection_idx].BackwardFFT( );
            }
            else {

                // Make sure the previous copy from host -> device has completed before we start to make another projection.
                // Event is created as non-blocking so this is a busy-wait.

                template_reconstruction.ExtractSlice(current_projection[current_projection_idx], angles, 1.0f, false);

                current_projection[current_projection_idx].SwapRealSpaceQuadrants( );
                current_projection[current_projection_idx].MultiplyPixelWise(projection_filter);
                current_projection[current_projection_idx].BackwardFFT( );
                average_on_edge = current_projection[current_projection_idx].ReturnAverageOfRealValuesOnEdges( );
                // We'll subtract average_on_edge in the normalization prior to scaling
                average_of_reals = current_projection[current_projection_idx].ReturnAverageOfRealValues( ) - average_on_edge;

                // For an intiial test, make projection_queue.cpu_prj_stream[current_projection_idx]
                // a public member.. if it works, make it private and return a reference instead

                // d_current_projection[current_projection_idx].CopyHostToDevice(current_projection[current_projection_idx], false, false);
                d_current_projection[current_projection_idx].CopyHostToDevice(current_projection[current_projection_idx], false, false, projection_queue.gpu_projection_stream[current_projection_idx]);

                // We need to make sure the current cpu projection is not used by the host until the gpu has finished with it, which may be independent of the main work
                // in cudaStreamPerThread, this is a blocking event for the host (if the queue is otherwise full)
                projection_queue.RecordProjectionReadyBlockingHost(current_projection_idx, projection_queue.gpu_projection_stream[current_projection_idx]);
                // We need the main work in cudaStreamPerThread to wait on the transfer in this stream, which if the CPU thread is ahead, should be a non-blocking event
                projection_queue.RecordGpuProjectionReadyStreamPerThreadWait(current_projection_idx);

                // The average in the full padded image will be different;
                average_of_reals *= ((float)d_current_projection[current_projection_idx].number_of_real_space_pixels / (float)d_padded_reference.number_of_real_space_pixels);
            }

            // d_current_projection[current_projection_idx].MultiplyByConstant(rsqrtf(d_current_projection[current_projection_idx].ReturnSumOfSquares( ) / (float)d_padded_reference.number_of_real_space_pixels - (average_of_reals * average_of_reals)));
            // For the host to execute the preceding line, it was to wait on the return value from ReturnSumOfSquares. This could be a bit of a performance regression as otherwise it can queue up all the reamining
            // GPU work and get back to calculating the next projection. The commented out method is an attempt around that, but currently the mips come out a little different a bit faster.
            d_current_projection[current_projection_idx].NormalizeRealSpaceStdDeviation(float(d_padded_reference.number_of_real_space_pixels), average_of_reals, average_on_edge);
            d_current_projection[current_projection_idx].ClipInto(&d_padded_reference, 0, false, 0, 0, 0, 0);
            // d_current_projection[current_projection_idx].MultiplyByConstant(rsqrtf(d_current_projection[current_projection_idx].ReturnSumOfSquares( ) / (float)d_padded_reference.number_of_real_space_pixels - (average_of_reals * average_of_reals)));

            if ( use_gpu_prj ) {
                // Note the stream change will not affect the padded projection
                projection_queue.RecordProjectionReadyBlockingHost(current_projection_idx, cudaStreamPerThread);
            }

            d_padded_reference.ForwardFFT(false);

            //      d_padded_reference.ForwardFFTAndClipInto(d_current_projection,false);
            if constexpr ( n_mips_to_process_at_once > 1 ) {
                d_padded_reference.BackwardFFTAfterComplexConjMul(d_input_image.complex_values_fp16, true, &ccf_array[current_mip_to_process * d_input_image.real_memory_allocated]);
            }
            else {
                d_padded_reference.BackwardFFTAfterComplexConjMul(d_input_image.complex_values_fp16, true);
            }

            // d_padded_reference.MultiplyByConstant(rsqrtf(d_padded_reference.ReturnSumOfSquares( ) / (float)d_padded_reference.number_of_real_space_pixels));

            // cudaErr(cudaStreamSynchronize(cudaStreamPerThread));

            if constexpr ( n_mips_to_process_at_once > 1 ) {
                psi_array[current_mip_to_process]   = __float2half_rn(current_psi);
                theta_array[current_mip_to_process] = __float2half_rn(global_euler_search.list_of_search_parameters[current_search_position][1]);
                phi_array[current_mip_to_process]   = __float2half_rn(global_euler_search.list_of_search_parameters[current_search_position][0]);
                current_mip_to_process++;
                if ( current_mip_to_process == n_mips_to_process_at_once ) {
                    cudaErr(cudaEventSynchronize(mip_is_done_Event));
                    cudaErr(cudaMemcpyAsync(d_psi_array, psi_array, sizeof(__half) * n_mips_to_process_at_once, cudaMemcpyHostToDevice, cudaStreamPerThread));
                    cudaErr(cudaMemcpyAsync(d_theta_array, theta_array, sizeof(__half) * n_mips_to_process_at_once, cudaMemcpyHostToDevice, cudaStreamPerThread));
                    cudaErr(cudaMemcpyAsync(d_phi_array, phi_array, sizeof(__half) * n_mips_to_process_at_once, cudaMemcpyHostToDevice, cudaStreamPerThread));

                    total_mip_processed += current_mip_to_process;
                    my_dist.at(0).AccumulateDistribution(ccf_array, current_mip_to_process);

                    MipPixelWiseStack(ccf_array, d_psi_array, d_theta_array, d_phi_array, current_mip_to_process);

                    current_mip_to_process = 0;
                }
            }
            else {

                my_dist.at(0).AccumulateDistribution(d_padded_reference.real_values_fp16, 1);

                MipPixelWise(__float2half_rn(current_psi), __float2half_rn(global_euler_search.list_of_search_parameters[current_search_position][1]),
                             __float2half_rn(global_euler_search.list_of_search_parameters[current_search_position][0]));
            }

            ccc_counter++;
            total_number_of_cccs_calculated++;

            if constexpr ( n_mips_to_process_at_once == 1 ) {
                if ( ccc_counter % 10 == 0 ) {
                    AccumulateSums(sum_sumsq, d_sum1, d_sumSq1);
                }
            }

            if ( ccc_counter % 100 == 0 ) {
                d_sum2.AddImage(d_sum1);
                d_sum1.Zeros( );

                d_sumSq2.AddImage(d_sumSq1);
                d_sumSq1.Zeros( );
            }

            if ( ccc_counter % 10000 == 0 ) {
                d_sum3.AddImage(d_sum2);
                d_sum2.Zeros( );

                d_sumSq3.AddImage(d_sumSq2);
                d_sumSq2.Zeros( );
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

    if constexpr ( n_mips_to_process_at_once > 1 ) {
        if ( current_mip_to_process > 0 ) {
            cudaErr(cudaMemcpyAsync(d_psi_array, psi_array, sizeof(__half) * n_mips_to_process_at_once, cudaMemcpyHostToDevice, cudaStreamPerThread));
            cudaErr(cudaMemcpyAsync(d_theta_array, theta_array, sizeof(__half) * n_mips_to_process_at_once, cudaMemcpyHostToDevice, cudaStreamPerThread));
            cudaErr(cudaMemcpyAsync(d_phi_array, phi_array, sizeof(__half) * n_mips_to_process_at_once, cudaMemcpyHostToDevice, cudaStreamPerThread));
            my_dist.at(0).AccumulateDistribution(ccf_array, current_mip_to_process);
            MipPixelWiseStack(ccf_array, d_psi_array, d_theta_array, d_phi_array, current_mip_to_process);
            total_mip_processed += current_mip_to_process;
        }
    }

    else {
        AccumulateSums(sum_sumsq, d_sum1, d_sumSq1);
    }

    wxPrintf("\t\t\ntotal number %d, total mips %d\n", ccc_counter, total_mip_processed);

    d_sum2.AddImage(d_sum1);
    d_sumSq2.AddImage(d_sumSq1);

    d_sum3.AddImage(d_sum2);
    d_sumSq3.AddImage(d_sumSq2);

    MipToImage( );

    my_dist.at(0).FinalAccumulate( );

    cudaErr(cudaFreeAsync(mip_psi, cudaStreamPerThread));
    cudaErr(cudaFreeAsync(sum_sumsq, cudaStreamPerThread));
    cudaErr(cudaFreeAsync(theta_phi, cudaStreamPerThread));

    if constexpr ( n_mips_to_process_at_once > 1 ) {
        cudaErr(cudaFreeAsync(d_psi_array, cudaStreamPerThread));
        cudaErr(cudaFreeAsync(d_theta_array, cudaStreamPerThread));
        cudaErr(cudaFreeAsync(d_phi_array, cudaStreamPerThread));
        cudaErr(cudaFreeAsync(ccf_array, cudaStreamPerThread));
    }

    if ( n_global_search_images_to_save > 1 ) {
        cudaErr(cudaFreeAsync(secondary_peaks, cudaStreamPerThread));
    }

    cudaErr(cudaStreamSynchronize(cudaStreamPerThread));

    if constexpr ( n_mips_to_process_at_once > 1 ) {
        delete[] psi_array;
        delete[] theta_array;
        delete[] phi_array;
    }
}

__global__ void MipPixelWiseKernel(__half* __restrict__ ccf,
                                   __half2* __restrict__ mip_psi,
                                   const int numel,
                                   __half    psi,
                                   __half    theta,
                                   __half    phi,
                                   __half2* __restrict__ sum_sumsq,
                                   __half2* __restrict__ theta_phi) {

    //	Peaks tmp_peak;

    for ( int i = physical_X_1d_grid( ); i < numel; i += GridStride_1dGrid( ) ) {

        const __half  half_val = ccf[i];
        const __half2 input    = __half2half2(half_val * __half(10000.0));
        const __half2 mulVal   = __halves2half2((__half)1.0, half_val);

        sum_sumsq[i] = __hfma2(input, mulVal, sum_sumsq[i]);

        if ( half_val > __low2half(mip_psi[i]) ) {
            //				tmp_peak.mip = half_val;
            mip_psi[i]   = __halves2half2(half_val, psi);
            theta_phi[i] = __halves2half2(theta, phi);
        }
        //
    }
}

void TemplateMatchingCore::MipPixelWise(__half psi, __half theta, __half phi) {

    precheck;
    // N
    d_padded_reference.ReturnLaunchParametersLimitSMs(5.f, 1024);

    MipPixelWiseKernel<<<d_padded_reference.gridDims, d_padded_reference.threadsPerBlock, 0, cudaStreamPerThread>>>((__half*)d_padded_reference.real_values_16f, mip_psi,
                                                                                                                    (int)d_padded_reference.real_memory_allocated,
                                                                                                                    psi, theta, phi, sum_sumsq, theta_phi);
    postcheck;
}

__global__ void MipPixelWiseStackKernel(const __half* __restrict__ ccf,
                                        const __half* __restrict__ psi,
                                        const __half* __restrict__ theta,
                                        const __half* __restrict__ phi,
                                        float* __restrict__ sum,
                                        float* __restrict__ sum_sq,
                                        __half2* __restrict__ mip_psi,
                                        __half2* __restrict__ theta_phi,
                                        int numel,
                                        int n_mips_this_round) {

    int   max_idx;
    float tmp_sum;
    float tmp_sum_sq;
    float max_val;
    float ccf_val;
    // for ( int img_index = blockIdx.x * blockDim.x + threadIdx.x; img_index < NX; img_index += blockDim.x * gridDim.x ) {
    // 2,147,483,647 max_int(32 bit)
    // k3 padded is 5832 4096, so max_int could handle is 89.89 slices

    for ( int i = physical_X_1d_grid( ); i < numel; i += GridStride_1dGrid( ) ) {
        tmp_sum    = 0.f;
        tmp_sum_sq = 0.f;
        max_val    = -10.f;
        for ( int iSlice = 0; iSlice < n_mips_this_round; iSlice++ ) {

            ccf_val = __half2float(ccf[iSlice * numel + i]);
            tmp_sum += ccf_val;
            tmp_sum_sq += (ccf_val * ccf_val);
            if ( ccf_val > max_val ) {
                max_val = ccf_val;
                max_idx = iSlice;
            }
        }

        sum[i] += tmp_sum;
        sum_sq[i] += tmp_sum_sq;

        if ( max_val > -10.f && __float2half_rn(max_val) > __low2half(mip_psi[i]) ) {
            mip_psi[i]   = __halves2half2(__float2half_rn(max_val), psi[max_idx]);
            theta_phi[i] = __halves2half2(theta[max_idx], phi[max_idx]);
        }
    }
}

void TemplateMatchingCore::MipPixelWiseStack(__half* ccf, __half* psi, __half* theta, __half* phi, int n_mips_this_round) {

    precheck;
    // N
    d_padded_reference.ReturnLaunchParametersLimitSMs(5.f, 1024);

    MipPixelWiseStackKernel<<<d_padded_reference.gridDims, d_padded_reference.threadsPerBlock, 0, cudaStreamPerThread>>>(ccf,
                                                                                                                         psi,
                                                                                                                         theta,
                                                                                                                         phi,
                                                                                                                         (float*)d_sum1.real_values,
                                                                                                                         (float*)d_sumSq1.real_values,
                                                                                                                         mip_psi,
                                                                                                                         theta_phi,
                                                                                                                         (int)d_padded_reference.real_memory_allocated,
                                                                                                                         n_mips_this_round);
    postcheck;
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

__global__ void MipToImageKernel(const __half2* __restrict__ mip_psi,
                                 const __half2* __restrict__ theta_phi,
                                 const __half* __restrict__ secondary_peaks,
                                 const int numel,
                                 cufftReal* __restrict__ mip,
                                 cufftReal* __restrict__ psi,
                                 cufftReal* __restrict__ theta,
                                 cufftReal* __restrict__ phi,
                                 const int n_peaks) {

    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    if ( x >= numel )
        return;

    if ( n_peaks == 1 ) {

        mip[x]   = (cufftReal)__low2float(mip_psi[x]);
        psi[x]   = (cufftReal)__high2float(mip_psi[x]);
        theta[x] = (cufftReal)__low2float(theta_phi[x]);
        phi[x]   = (cufftReal)__high2float(theta_phi[x]);
    }
    else {
        int offset;
        for ( int iPeak = 0; iPeak < n_peaks; iPeak++ ) {
            offset = x + numel * iPeak; // out puts are NX * NY * NZ

            mip[offset]   = (cufftReal)secondary_peaks[offset];
            psi[offset]   = (cufftReal)secondary_peaks[offset + numel * n_peaks];
            theta[offset] = (cufftReal)secondary_peaks[offset + numel * n_peaks * 2];
            phi[offset]   = (cufftReal)secondary_peaks[offset + numel * n_peaks * 3];
        }
    }
}

void TemplateMatchingCore::MipToImage( ) {

    precheck;
    dim3 threadsPerBlock = dim3(1024, 1, 1);
    dim3 gridDims        = dim3((d_max_intensity_projection.real_memory_allocated + threadsPerBlock.x - 1) / threadsPerBlock.x, 1, 1);

    MipToImageKernel<<<gridDims, threadsPerBlock, 0, cudaStreamPerThread>>>(mip_psi,
                                                                            theta_phi,
                                                                            secondary_peaks,
                                                                            d_padded_reference.real_memory_allocated,
                                                                            d_max_intensity_projection.real_values,
                                                                            d_best_psi.real_values,
                                                                            d_best_theta.real_values,
                                                                            d_best_phi.real_values,
                                                                            n_global_search_images_to_save);
    postcheck;
}

__global__ void AccumulateSumsKernel(__half2* __restrict__ sum_sumsq, cufftReal* __restrict__ sum, cufftReal* __restrict__ sq_sum, const int numel);

void TemplateMatchingCore::AccumulateSums(__half2* sum_sumsq, GpuImage& sum, GpuImage& sq_sum) {

    precheck;
    dim3 threadsPerBlock = dim3(1024, 1, 1);
    dim3 gridDims        = dim3((sum.real_memory_allocated + threadsPerBlock.x - 1) / threadsPerBlock.x, 1, 1);

    AccumulateSumsKernel<<<gridDims, threadsPerBlock, 0, cudaStreamPerThread>>>(sum_sumsq, sum.real_values, sq_sum.real_values, sum.real_memory_allocated);
    postcheck;
}

__global__ void AccumulateSumsKernel(__half2* __restrict__ sum_sumsq, cufftReal* __restrict__ sum, cufftReal* __restrict__ sq_sum, const int numel) {

    int x = blockIdx.x * blockDim.x + threadIdx.x;
    if ( x < numel ) {

        sum[x]    = __fmaf_rn(0.0001f, __low2float(sum_sumsq[x]), sum[x]);
        sq_sum[x] = __fmaf_rn(0.0001f, __high2float(sum_sumsq[x]), sq_sum[x]);

        sum_sumsq[x] = __halves2half2((__half)0., (__half)0.);
    }
}
