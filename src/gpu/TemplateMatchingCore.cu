/**
 * @file TemplateMatchingCore.cu
 * @brief CUDA implementation of the TemplateMatchingCore class.
 *
 * This file contains the definitions of methods for the TemplateMatchingCore class,
 * which are executed on the GPU using CUDA. It handles the low-level details of
 * GPU memory management, kernel launches, and stream synchronization for template matching.
 *
 * Thread Safety and GPU Execution Model:
 * - Each TemplateMatchingCore instance is designed to be managed by a single CPU host thread.
 * - GPU operations are typically enqueued onto CUDA streams (often `cudaStreamPerThread` or
 *   custom streams managed by `ProjectionQueue`).
 * - `cudaStreamPerThread` provides a separate default stream for each host thread,
 *   allowing for concurrent kernel execution from different host threads if the GPU
 *   hardware supports it.
 * - Synchronization primitives like `cudaEvent_t` and `cudaStreamSynchronize` are used
 *   to manage dependencies between GPU operations and between GPU and CPU.
 * - The `ProjectionQueue` class is used to manage a pool of CUDA streams and events
 *   for asynchronous projection generation and processing, overlapping data transfers
 *   and computations.
 * - Data shared between CPU threads (e.g., input images loaded once) is typically
 *   uploaded to the GPU and then accessed via `std::shared_ptr<GpuImage>` by multiple
 *   `TemplateMatchingCore` instances. The `GpuImage` class itself should handle
 *   any necessary internal GPU-side synchronization if its state can be modified
 *   concurrently (though often these are read-only on the GPU during the core loop).
 * - Result accumulation (e.g., MIPs, best angles) is done on GPU memory local to
 *   the `TemplateMatchingCore` instance. Final aggregation of results from multiple
 *   instances/threads is handled by the calling code (e.g., `MatchTemplateApp`) on the CPU.
 *
 * Key Operations:
 * - `Init`: Sets up GPU resources, allocates memory for projections, statistical buffers,
 *   and initializes search parameters.
 * - `RunInnerLoop`: The main computational kernel. It iterates through assigned search
 *   orientations (Psi angles and Euler search positions).
 *   - Generates 2D projections from a 3D template (either on GPU or CPU then transferred).
 *     - GPU projections use `ExtractSliceShiftAndCtf`.
 *     - CPU projections are made, then copied to GPU.
 *   - Normalizes projections.
 *   - Performs FFTs (potentially using FastFFT library).
 *   - Calculates Cross-Correlation Functions (CCFs) by multiplying in Fourier space
 *     with the FFT of the input image and then inverse FFT.
 *   - Updates an empirical distribution of CCF values using `TM_EmpiricalDistribution`.
 *   - Identifies peaks (Maximum Intensity Projections - MIPs) and corresponding angles.
 * - L2 Cache Management: Methods like `SetL2CachePersisting` and `SetL2AccessPolicy`
 *   are provided to potentially optimize performance on supported GPU architectures
 *   by influencing how data is cached in L2.
 * - `UpdateSecondaryPeaksKernel`: A CUDA global kernel to manage and update a list
 *   of secondary peaks if more than one top peak per search position is required.
 */
#include "gpu_core_headers.h"
#include "gpu_indexing_functions.h"

#include "TemplateMatchingCore.h"

#ifdef cisTEM_USING_FastFFT
#ifdef cisTEM_BUILDING_FastFFT
#include "../../include/FastFFT/include/FastFFT.h"
#include "../../include/FastFFT/include/detail/functors.h"
#else
#include "/opt/FastFFT/include/FastFFT.h"
#include "/opt/FastFFT/include/detail/functors.h"
#endif
#endif
// Implementation is in the header as it is only used here for now.
/**
 * @brief Manages a queue of projections for asynchronous processing.
 *
 * The ProjectionQueue is crucial for overlapping GPU projection generation,
 * data transfers, and the main CCC computation. It uses a set of CUDA streams
 * and events to manage dependencies.
 *
 * For example, one projection can be generated on `projection_queue.gpu_projection_stream[i]`
 * while the main processing of a previous projection occurs on `cudaStreamPerThread`.
 * Events ensure that the main processing waits for the projection to be ready.
 */
#include "projection_queue.cuh"

constexpr bool trouble_shoot_mip = false;

// #define TEST_IES

using namespace cistem_timer;

void TemplateMatchingCore::Init(MyApp*                    parent_pointer,
                                std::shared_ptr<GpuImage> wanted_template_reconstruction,
                                std::shared_ptr<GpuImage> wanted_input_image,
                                Image&                    current_projection,
                                float                     psi_max,
                                float                     psi_start,
                                float                     psi_step,
                                AnglesAndShifts&          angles,
                                EulerSearch&              global_euler_search,
                                const int2                pre_padding,
                                const int2                roi,
                                int                       first_search_position,
                                int                       last_search_position,
                                ProgressBar*              my_progress,
                                long                      total_correlation_positions,
                                bool                      is_running_locally,
                                bool                      use_fast_fft,
                                bool                      use_gpu_prj,
                                int                       number_of_global_search_images_to_save) {

    // --- Thread Safety Note ---
    // This Init method is called once per TemplateMatchingCore instance.
    // It allocates GPU resources that are specific to this instance.
    // `wanted_template_reconstruction` and `wanted_input_image` are shared_ptrs,
    // allowing multiple instances to safely reference the same underlying GPU data
    // (which should be immutable or properly synchronized if modified elsewhere during this time).

    MyDebugAssertFalse(object_initialized_, "Init must only be called once!");
    MyDebugAssertFalse(wanted_input_image->is_in_real_space, "Input image must be in Fourier space");
    MyDebugAssertTrue(wanted_input_image->is_allocated_16f_buffer, "Input image must be in memory");
    object_initialized_ = true;

    this->use_gpu_prj = use_gpu_prj;

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

    this->current_projection.reserve(n_prjs);
    for ( int i = 0; i < n_prjs; i++ ) {
        this->current_projection.emplace_back(current_projection);
        d_current_projection.emplace_back(this->current_projection[i]);
    }
    if ( use_gpu_prj ) {
        template_gpu_shared = wanted_template_reconstruction;
    }

    d_input_image          = wanted_input_image;
    is_set_input_image_ptr = true;

    d_statistical_buffers_ptrs.push_back(&d_padded_reference);
    d_statistical_buffers_ptrs.push_back(&d_sum1);
    d_statistical_buffers_ptrs.push_back(&d_sumSq1);
    d_statistical_buffers_ptrs.push_back(&d_sum2);
    d_statistical_buffers_ptrs.push_back(&d_sumSq2);
    int n_2d_buffers = 0;
    for ( auto& buffer : d_statistical_buffers_ptrs ) {
        buffer->Allocate(d_input_image->dims.x, d_input_image->dims.y, 1, true);
        n_2d_buffers++;
    }

    d_statistical_buffers_ptrs.push_back(&d_max_intensity_projection);
    d_statistical_buffers_ptrs.push_back(&d_best_psi);
    d_statistical_buffers_ptrs.push_back(&d_best_theta);
    d_statistical_buffers_ptrs.push_back(&d_best_phi);
    for ( int i = n_2d_buffers; i < d_statistical_buffers_ptrs.size( ); i++ ) {
        d_statistical_buffers_ptrs[i]->Allocate(d_input_image->dims.x, d_input_image->dims.y, number_of_global_search_images_to_save, true);
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

size_t TemplateMatchingCore::SetL2CachePersisting(const float L2_persistance_fraction) {
    // --- Thread Safety Note ---
    // This method is intended to be called by a single thread (thread 0 assertion)
    // to set a global L2 cache policy. This is a device-wide setting.
    MyDebugAssertTrue(is_set_input_image_ptr, "Input image must be set before calling SetL2CachePersisting");
    MyDebugAssertTrue(ReturnThreadNumberOfCurrentThread( ) == 0, "SetL2CachePersisting must be called from thread 0");

    if ( is_set_L2_cache_persisting || ! L2_persistance_fraction > 0.f )
        return 0;

    // If we aren't set, lets first check to see if we are on a device where this is beneficial
    // FIXME: this probably depends more on the size of the L2 cache than on the device arch. EG 800 may be better than 860 (or smaller images)
    int gpuIDX, major, minor;
    cudaErr(cudaGetDevice(&gpuIDX));
    cudaErr(cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, gpuIDX));
    cudaErr(cudaDeviceGetAttribute(&minor, cudaDevAttrComputeCapabilityMinor, gpuIDX));

    int device_arch = major * 100 + minor * 10;
    if ( device_arch < 800 ) {
        std::cerr << "Device architecture is " << device_arch << " which is less than 800, so we are NOT setting L2 cache persisting" << std::endl;
        return 0;
    }

    // NOTE: we're assuming that we only see one GPU as limited by CUDA_VISIBLE_DEVICES in the run_profile
    // TODO: attributes may be cheaper to query, but this is probably a relatively small cost.
    size_t data_size_bytes = d_input_image->number_of_real_space_pixels * sizeof(__half);
    std::cerr << "Data size in bytes: " << data_size_bytes << std::endl;

    int L2_cache_size, max_persisting_L2_cache_size, accessPolicyMaxWindowSize;
    cudaErr(cudaGetDevice(&gpuIDX));

    cudaErr(cudaDeviceGetAttribute(&max_persisting_L2_cache_size, cudaDevAttrMaxPersistingL2CacheSize, gpuIDX));
    cudaErr(cudaDeviceGetAttribute(&L2_cache_size, cudaDevAttrL2CacheSize, gpuIDX));
    cudaErr(cudaDeviceGetAttribute(&accessPolicyMaxWindowSize, cudaDevAttrMaxAccessPolicyWindowSize, gpuIDX));

    // on 86 and 89 it seeems max_persisting_L2_cache_size < L2_cache_size < accessPolicyMaxWindowSize
    size_t size = std::min(int(L2_cache_size * 0.75), max_persisting_L2_cache_size);
    if ( float(data_size_bytes) / float(size) > L2_persistance_fraction ) {
        std::cerr << "Data size is less than the L2 cache size, so we are NOT setting L2 cache persisting" << std::endl;
        std::cerr << "Data size: " << data_size_bytes << " L2 cache available for persisting size: " << size << std::endl;
        return 0;
    }

    cudaDeviceSetLimit(cudaLimitPersistingL2CacheSize, size); // set-aside 3/4 of L2 cache for persisting accesses or the max allowed

    // In the cuda programming manual, it suggests setting the window size as follows:
    // size_t window_size = std::min(size_t(accessPolicyMaxWindowSize), data_size_bytes);

    // If theh window is > than the allowed cache size, a hit Prop fraction is set in SetL2AccessPolicy
    // This is (afaik) random addresses, which is probably more efficient with a single accessor, however, we are sharing the input data
    // among several threads, so we want to limit the window to the data size or the allowed cache size, and let the hitProp fraction = 1
    size_t window_size = std::min(data_size_bytes, size);

    is_set_L2_cache_persisting = true;

    return window_size;

    // Each thread will set the access policy window for the input image since they have their own stream
};

void TemplateMatchingCore::ClearL2CachePersisting( ) {
    // --- Thread Safety Note ---
    // Similar to SetL2CachePersisting, this likely affects global device state.
    // The assertion "Not implemented" and the TODO suggest caution.
    // Proper synchronization would be needed if multiple instances could call this.
    // TODO: we need to make sure we are the last user of this before clearing.
    // If there is any perf improvement, make this a whole object that we use shared pointers for within template matching core.
    MyDebugAssertTrue(false, "Not implemented");
    cudaCtxResetPersistingL2Cache( ); // Remove any persistent lines in L2
}

void TemplateMatchingCore::SetL2AccessPolicy(const size_t window_size) {
    // --- Thread Safety Note ---
    // This method sets an access policy for the current CUDA stream (`cudaStreamPerThread`).
    // Since each TemplateMatchingCore instance (and thus each host thread using it)
    // has its own `cudaStreamPerThread`, this operation is local to the calling thread's stream
    // and therefore safe in a multi-threaded context where each thread manages its own instance.
    stream_attribute.accessPolicyWindow.base_ptr  = reinterpret_cast<void*>(d_input_image->complex_values_fp16); // Global Memory data pointer
    stream_attribute.accessPolicyWindow.num_bytes = window_size; // Number of bytes for persistence access
    stream_attribute.accessPolicyWindow.hitRatio  = 0.8; // Hint for cache hit ratio
    stream_attribute.accessPolicyWindow.hitProp   = cudaAccessPropertyPersisting; // Persistence Property
    stream_attribute.accessPolicyWindow.missProp  = cudaAccessPropertyStreaming; // Type of access property on cache miss

    cudaStreamSetAttribute(cudaStreamPerThread, cudaStreamAttributeAccessPolicyWindow, &stream_attribute); // Set the attributes to a CUDA Stream
};

void TemplateMatchingCore::ClearL2AccessPolicy( ) {
    // --- Thread Safety Note ---
    // Similar to SetL2AccessPolicy, this clears the policy for the current thread's stream.
    // Safe when each thread manages its own TemplateMatchingCore instance.
    stream_attribute.accessPolicyWindow.num_bytes = 0; // Setting the window size to 0 disable it
    cudaStreamSetAttribute(cudaStreamPerThread, cudaStreamAttributeAccessPolicyWindow, &stream_attribute); // Overwrite the access policy attribute to a CUDA Stream
}

/**
 * @brief Main computational loop for template matching.
 * 
 * --- Thread Safety and GPU Concurrency Overview ---
 * This is the core computational method. Each call to RunInnerLoop is by a CPU thread
 * that owns this TemplateMatchingCore instance.
 *
 * 1. Initialization:
 *    - `my_dist` (TM_EmpiricalDistribution) is initialized or reset. This object manages
 *      its own GPU resources for accumulating CCF statistics, likely on `cudaStreamPerThread`.
 *    - Statistical buffers (`d_max_intensity_projection`, etc.) are zeroed on `cudaStreamPerThread`.
 *
 * 2. ProjectionQueue:
 *    - `ProjectionQueue projection_queue(n_prjs);` creates a helper to manage asynchronous
 *      projection generation. It uses its own set of CUDA streams.
 *    - `projection_queue.RecordProjectionReadyBlockingHost(current_projection_idx, cudaStreamPerThread);`
 *      This seems to be an initial synchronization point.
 *
 * 3. Main Loop (over search positions and psi angles):
 *    - `current_projection_idx = projection_queue.GetAvailableProjectionIDX();`
 *      Gets an index for a projection buffer/stream from the queue. This call might block
 *      if all projection resources are currently busy, ensuring that the host thread
 *      doesn't get too far ahead of the GPU's capacity to process projections.
 *
 *    - GPU Projection Path (`if (use_gpu_prj)`):
 *      - `d_current_projection[idx].ExtractSliceShiftAndCtf(...)` is enqueued on
 *        `projection_queue.gpu_projection_stream[current_projection_idx]`.
 *      - `d_current_projection[idx].BackwardFFT(...)` is also enqueued on the same projection stream.
 *      - This allows projection generation and its initial FFT to happen on a dedicated stream,
 *        potentially in parallel with other operations on `cudaStreamPerThread` or other projection streams.
 *
 *    - CPU Projection Path (`else`):
 *      - CPU performs `ExtractSlice`, `MultiplyPixelWise`, `BackwardFFT`.
 *      - `d_current_projection[idx].CopyHostToDevice(...)` enqueued on `projection_queue.gpu_projection_stream[idx]`.
 *      - `projection_queue.RecordProjectionReadyBlockingHost(...)` ensures host waits if GPU copy falls behind.
 *      - `projection_queue.RecordGpuProjectionReadyStreamPerThreadWait(idx)`: Makes `cudaStreamPerThread`
 *        wait for the H2D copy on `projection_queue.gpu_projection_stream[idx]` to complete before
 *        `cudaStreamPerThread` uses that projection data.
 *
 *    - Normalization and Main FFT/CCF Calculation:
 *      - `d_current_projection[idx].NormalizeRealSpaceStdDeviationAndCastToFp16(...)` (if use_fast_fft) or
 *        `NormalizeRealSpaceStdDeviation` then `ClipInto` (else) are enqueued on
 *        `projection_queue.gpu_projection_stream[idx]` or `cudaStreamPerThread` respectively.
 *      - `projection_queue.RecordGpuProjectionReadyStreamPerThreadWait(idx)`: (If FastFFT) Ensures `cudaStreamPerThread`
 *        waits for normalization on the projection stream.
 *      - `FT.FwdImageInvFFT(...)` (FastFFT path) or `d_padded_reference.ForwardFFT()`, `BackwardFFTAfterComplexConjMul(...)`
 *        (standard path) perform the core CCF calculation. These operations are enqueued on `cudaStreamPerThread`.
 *        The input to these operations is `d_current_projection[idx]` (after normalization) and `d_input_image`.
 *        The output CCF is written to `my_dist->GetCCFArray(current_mip_to_process)`.
 *
 *    - Empirical Distribution Update:
 *      - `my_dist->UpdateHostAngleArrays(...)` (CPU operation).
 *      - `my_dist->AccumulateDistribution(current_mip_to_process)`: This method processes a batch of CCFs.
 *        It likely enqueues kernels on `cudaStreamPerThread` to update histograms and MIPs on the GPU.
 *        It uses its own internal events (`my_dist->MakeHostWaitOnMipStackIsReadyEvent()`,
 *        `my_dist->RecordMipStackIsReadyBlockingHost()`) to manage synchronization for batches of CCFs,
 *        allowing the host to prepare the next batch while the GPU processes the current one.
 *
 * 4. Finalization:
 *    - Remaining CCFs in `my_dist` are processed.
 *    - `my_dist->CopySumAndSumSqAndZero(d_sum1, d_sumSq1)`: Copies statistical sums.
 *    - `my_dist->MipToImage(...)`: Generates final MIP images from accumulated data.
 *    - `my_dist->FinalAccumulate()`: Performs final calculations for the distribution.
 *    - `cudaStreamSynchronize(cudaStreamPerThread)`: Host thread blocks until all work
 *      enqueued on `cudaStreamPerThread` for this `RunInnerLoop` call is complete. This ensures
 *      that all GPU results (MIPs, best angles, statistical sums) are ready before the
 *      function returns and the host thread potentially accesses them or deallocates resources.
 *
 * Overall Concurrency:
 * - Multiple `TemplateMatchingCore` instances (each in its own CPU thread) can run `RunInnerLoop` concurrently.
 * - Within each `RunInnerLoop`, the `ProjectionQueue` allows projection generation/transfer to overlap
 *   with the main CCF calculations using separate CUDA streams.
 * - `TM_EmpiricalDistribution` also uses techniques to batch processing and overlap CPU/GPU work.
 * - Synchronization is handled by CUDA events between streams and `cudaStreamSynchronize` at the end
 *   of major phases or the entire loop.
 */
void TemplateMatchingCore::RunInnerLoop(Image&      projection_filter,
                                        int         threadIDX,
                                        long&       current_correlation_position,
                                        const float min_counter_val,
                                        const float threshold_val) {
    total_number_of_cccs_calculated = 0;
    bool at_least_100               = false;

    bool this_is_the_first_run_on_inner_loop = my_dist ? false : true;

    if ( this_is_the_first_run_on_inner_loop ) {
        d_padded_reference.CopyFP32toFP16buffer(false);
        my_dist = std::make_unique<TM_EmpiricalDistribution<__half, __half2>>(d_input_image.get( ), pre_padding, roi);
    }
    else {
        my_dist->ZeroHistogram( );
    }

    // Note: these shouldn't change after the first run
    my_dist->SetTrimmingAlgoMinCounterVal(min_counter_val);
    my_dist->SetTrimmingAlgoThresholdVal(threshold_val);

    // Make sure we are starting with zeros
    for ( auto& buffer : d_statistical_buffers_ptrs ) {
        buffer->Zeros( );
    }

    // Just for reference:
    // cudaStreamSynchronize: Blocks host until ALL work in the stream is completed
    // cudaStreamWaitEvent: Makes all future work in stream wait on an event. Since we are always using cudaStreamPerThread, this is not needed.

    cudaEvent_t mip_is_done_Event;

    cudaErr(cudaEventCreateWithFlags(&mip_is_done_Event, cudaEventBlockingSync));
#ifdef cisTEM_USING_FastFFT
    FastFFT::FourierTransformer<float, __half, __half2, 2> FT;

    // float scale_factor = powf((float)d_current_projection[0].number_of_real_space_pixels, -2.0);
    // float scale_factor = 1.f;
    float scale_factor = sqrtf(1.0f / float(d_input_image->number_of_real_space_pixels));

    FastFFT::KernelFunction::my_functor<float, 4, FastFFT::KernelFunction::CONJ_MUL_THEN_SCALE> conj_mul_then_scale(scale_factor);
    FastFFT::KernelFunction::my_functor<float, 0, FastFFT::KernelFunction::NOOP>                noop;

    if ( use_fast_fft ) {

        // TODO: overload that takes and short4's int4's instead of the individual values
        FT.SetForwardFFTPlan(current_projection[0].logical_x_dimension, current_projection[0].logical_y_dimension, current_projection[0].logical_z_dimension,
                             d_padded_reference.dims.x, d_padded_reference.dims.y, d_padded_reference.dims.z, true);
        FT.SetInverseFFTPlan(d_padded_reference.dims.x, d_padded_reference.dims.y, d_padded_reference.dims.z,
                             d_padded_reference.dims.x, d_padded_reference.dims.y, d_padded_reference.dims.z, true);
    }

#endif
    int   ccc_counter = 0;
    int   current_search_position;
    float average_on_edge;
    float average_of_reals;
    float temp_float;

    int thisDevice;
    cudaGetDevice(&thisDevice);

    GpuImage d_projection_filter(projection_filter);
    if ( use_gpu_prj ) {
        // d_projection_filter.CopyHostToDevice(projection_filter);
        // d_projection_filter.CopyFP32toFP16buffer(false);
        d_projection_filter.CopyHostToDeviceTextureComplex<2>(projection_filter);
    }

    int             current_projection_idx = 0;
    int             current_mip_to_process = 0;
    int             total_mip_processed    = 0;
    ProjectionQueue projection_queue(n_prjs);
    // We need to make sure the host blocks on all setup work before we start to make projections,
    // since we are using more than one stream.
    // This synchronize ensures that any setup operations on cudaStreamPerThread (like buffer zeroing)
    // are complete before projection_queue starts enqueuing work on its streams that might
    // depend on that setup.
    cudaErr(cudaStreamSynchronize(cudaStreamPerThread));
    projection_queue.RecordProjectionReadyBlockingHost(current_projection_idx, cudaStreamPerThread);

#ifdef TEST_IES
    GpuImage tmp_mask[1];
    tmp_mask->Allocate(d_current_projection[current_projection_idx].dims.x, d_current_projection[current_projection_idx].dims.y, 1, false);
#else
    GpuImage* tmp_mask = nullptr;
#endif
    for ( current_search_position = first_search_position; current_search_position <= last_search_position; current_search_position++ ) {

        if ( current_search_position % 10 == 0 ) {
            wxPrintf("Starting position %d/ %d\n", current_search_position, last_search_position);
        }

        for ( float current_psi = psi_start; current_psi <= psi_max; current_psi += psi_step ) {

            constexpr float shifts_in_x_y                               = 0.0f;
            constexpr bool  apply_shifts                                = false;
            constexpr bool  swap_real_space_quadrants_during_projection = true;
            // --- Per-Orientation Processing ---
            // The ProjectionQueue manages a set of projection "slots".
            // GetAvailableProjectionIDX might block if all slots are busy,
            // providing backpressure to the CPU thread.
            // FIXME, change this to also store psi and to have methods to convert between an index encoded as an int and the actual angles
            angles.Init(global_euler_search.list_of_search_parameters[current_search_position][0], global_euler_search.list_of_search_parameters[current_search_position][1], current_psi, shifts_in_x_y, shifts_in_x_y);
            current_projection_idx = projection_queue.GetAvailableProjectionIDX( );
            if ( use_gpu_prj ) {
                // --- GPU Projection Path ---
                // Projection generation and initial FFT are enqueued on a dedicated
                // stream from the projection_queue. This allows these operations
                // to potentially run concurrently with other work on cudaStreamPerThread
                // or on other projection_queue streams.
                d_current_projection[current_projection_idx].is_in_real_space = false;

#ifdef TEST_IES
                tmp_mask->is_in_real_space = false;
#endif
                constexpr float pixel_size                = 1.0f;
                constexpr float resolution_limit          = 1.0f;
                float           real_space_binning_factor = 1.0f;

                if ( use_lerp_for_resizing ) {
                    real_space_binning_factor = binning_factor;
                }

                // template_gpu_shared.get( )
                constexpr bool apply_ctf       = true;
                constexpr bool use_ctf_texture = true;

                d_current_projection[current_projection_idx].ExtractSliceShiftAndCtf<apply_ctf, use_ctf_texture>(template_gpu_shared.get( ),
                                                                                                                 &d_projection_filter,
                                                                                                                 angles,
                                                                                                                 pixel_size,
                                                                                                                 real_space_binning_factor,
                                                                                                                 resolution_limit,
                                                                                                                 false,
                                                                                                                 swap_real_space_quadrants_during_projection,
                                                                                                                 apply_shifts,
                                                                                                                 true,
                                                                                                                 tmp_mask,
                                                                                                                 projection_queue.gpu_projection_stream[current_projection_idx]);

                average_of_reals = 0.f;
                average_on_edge  = 0.f;

                /*  Keep this comment for future dev to be aware of GOTCHA stream semantics:

                    Default GpuImage methods are in cudaStreamPerThread, now that we can pass a stream to BackwardFFT, we don't need to set this unless we do other
                    ops in cudaStreamPerThread using d_current_projection[current_projection_idx]
                    projection_queue.RecordGpuProjectionReadyStreamPerThreadWait(current_projection_idx);
                */

                d_current_projection[current_projection_idx].BackwardFFT(projection_queue.gpu_projection_stream[current_projection_idx]);
                if constexpr ( trouble_shoot_mip ) {

                    cudaErr(cudaDeviceSynchronize( ));
                    d_current_projection[current_projection_idx].QuickAndDirtyWriteSlice("gpu_prj.mrc", 1);
                    float prj_sum = d_current_projection[current_projection_idx].ReturnSumOfRealValues( );
#ifdef TEST_IES

                    tmp_mask->BackwardFFT(projection_queue.gpu_projection_stream[current_projection_idx]);
                    tmp_mask->QuickAndDirtyWriteSlice("gpu_mask.mrc", 1);
                    float mask_sum = tmp_mask->ReturnSumOfRealValues( );
                    std::cerr << "prj sum: " << prj_sum << std::endl;

                    std::cerr << "Mask sum: " << mask_sum << std::endl;
                    exit(0);
#endif
                }
            }
            else {
                // --- CPU Projection Path ---
                // CPU generates the projection.
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

                // projection_queue.RecordProjectionReadyBlockingHost: Host may block here if the H2D copy
                // on the projection_queue stream hasn't completed, ensuring the CPU doesn't overwrite
                // `current_projection[idx]` while it's still being copied.
                projection_queue.RecordProjectionReadyBlockingHost(current_projection_idx, projection_queue.gpu_projection_stream[current_projection_idx]);
                // projection_queue.RecordGpuProjectionReadyStreamPerThreadWait: cudaStreamPerThread (main work stream)
                // will wait for the H2D copy on projection_queue.gpu_projection_stream[idx] to complete
                // before proceeding with operations that use this projection data.
                projection_queue.RecordGpuProjectionReadyStreamPerThreadWait(current_projection_idx);

                // Note: I had deleted this in the dev branch for FastFFT. Review when possible
                // The average in the full padded image will be different;
                // average_of_reals *= ((float)d_current_projection[current_projection_idx].number_of_real_space_pixels / (float)d_padded_reference.number_of_real_space_pixels);
                if constexpr ( trouble_shoot_mip ) {
                    cudaErr(cudaDeviceSynchronize( ));
                    d_current_projection[current_projection_idx].QuickAndDirtyWriteSlice("gpu_prj.mrc", 1);
                }
            }

            // --- Normalization and CCF Calculation ---
            if ( use_fast_fft ) {
#ifdef cisTEM_USING_FastFFT
                // float scale_factor = rsqrtf(d_current_projection[current_projection_idx].ReturnSumOfSquares( ) / (float)d_padded_reference.number_of_real_space_pixels - (average_of_reals * average_of_reals));
                // scale_factor /= powf((float)d_current_projection[current_projection_idx].number_of_real_space_pixels, 1.0);

                constexpr float scale_factor = 1.0f;
                // FIXME: there is a bunch of wasted math since we are using (1., 0, 0.)
                // TODO: if doing NCC, T3 = N_img * sum(template^2), and we already have access to sum(template^2) when doing this normalization.
                // For the first point, let's just check 1/0/0 and call a different kernel if we know we don't need the other factors (still have to wait on L2Norm called from npp, so we can't do it all on host)
                // For the full NCC, have an overload that passes three more array pointers, one to FFT(image) one to FFT(image^2) and one to conj(FFT(template mask)) // use this for the full NCC given an image mask of all ones
                // For the case the template mask is assumed rotationally invariant, pass a single array pointer that is the full normalization term to be multiplied by T3 then sqrted (this needs to be used AFTER the back FFT)
                d_current_projection[current_projection_idx].NormalizeRealSpaceStdDeviationAndCastToFp16(scale_factor,
                                                                                                         average_of_reals,
                                                                                                         average_on_edge,
                                                                                                         projection_queue.gpu_projection_stream[current_projection_idx]);

                // Make sure the FastFFT, using the cudaStreamPerThread stream waits on  projection_queue.gpu_projection_stream[current_projection_idx] before doing work
                projection_queue.RecordGpuProjectionReadyStreamPerThreadWait(current_projection_idx);

                // Host can be signaled that this projection slot is now free for another CPU projection
                // to be copied into, as the GPU data has been processed up to normalization and cast to fp16.
                // The actual FFT (FwdImageInvFFT) will use the fp16 buffer.
                projection_queue.RecordProjectionReadyBlockingHost(current_projection_idx, projection_queue.gpu_projection_stream[current_projection_idx]);

                // Core CCF calculation (FFT, complex multiply, IFFT) enqueued on cudaStreamPerThread.
                // Input: d_current_projection[idx].real_values_fp16 (from normalization)
                //        d_input_image->complex_values_fp16 (pre-loaded shared input)
                // Output: my_dist->GetCCFArray(current_mip_to_process)
                FT.FwdImageInvFFT(d_current_projection[current_projection_idx].real_values_fp16, (__half2*)d_input_image->complex_values_fp16, my_dist->GetCCFArray(current_mip_to_process), noop, conj_mul_then_scale, noop);

#endif // cisTEM_USING_FastFFT
            }
            else {
                // Standard FFT path (not FastFFT)
                // The average in the full padded image will be different;
                average_of_reals *= ((float)d_current_projection[current_projection_idx].number_of_real_space_pixels / (float)d_padded_reference.number_of_real_space_pixels);
                d_current_projection[current_projection_idx].NormalizeRealSpaceStdDeviation(float(d_padded_reference.number_of_real_space_pixels), average_of_reals, average_on_edge);
                d_current_projection[current_projection_idx].ClipInto(&d_padded_reference, 0, false, 0, 0, 0, 0); // Result in d_padded_reference

                if ( use_gpu_prj ) {
                    // If GPU projection, the original d_current_projection[idx] buffer can be marked ready
                    // for reuse by the host/projection_queue after ClipInto.
                    // The stream here is cudaStreamPerThread as ClipInto was on it.
                    projection_queue.RecordProjectionReadyBlockingHost(current_projection_idx, cudaStreamPerThread);
                }
                // Core CCF calculation (FFT, complex multiply, IFFT) enqueued on cudaStreamPerThread.
                // Input: d_padded_reference (contains normalized projection)
                //        d_input_image->complex_values_fp16
                // Output: my_dist->GetCCFArray(current_mip_to_process)
                d_padded_reference.ForwardFFT(false);
                d_padded_reference.BackwardFFTAfterComplexConjMul(d_input_image->complex_values_fp16, true, my_dist->GetCCFArray(current_mip_to_process));
            }

            if constexpr ( trouble_shoot_mip ) {
                // To trouble shoot
                cudaErr(cudaDeviceSynchronize( ));
                // Just make sure we have the FP16 buffer allocated
                d_padded_reference.CopyFP32toFP16buffer(false);
                cudaErr(cudaMemcpy(d_padded_reference.real_values_fp16, my_dist->GetCCFArray(current_mip_to_process), d_padded_reference.real_memory_allocated * sizeof(__half), cudaMemcpyDeviceToDevice));
                // Move back into the fp32 buffer
                d_padded_reference.CopyFP16buffertoFP32(false);
                // Write out the padded reference
                d_padded_reference.QuickAndDirtyWriteSlice("padded_ref.mrc", 1);
                exit(0);
            }
            // d_padded_reference.MultiplyByConstant(rsqrtf(d_padded_reference.ReturnSumOfSquares( ) / (float)d_padded_reference.number_of_real_space_pixels));

            my_dist->UpdateHostAngleArrays(current_mip_to_process, current_psi, global_euler_search.list_of_search_parameters[current_search_position][1], global_euler_search.list_of_search_parameters[current_search_position][0]);

            current_mip_to_process++;
            if ( current_mip_to_process == my_dist->n_imgs_to_process_at_once( ) - 1 ) {
                // --- Process a Batch of CCFs ---
                // Host waits for the previous batch of MIPs/distribution updates to complete on GPU.
                my_dist->MakeHostWaitOnMipStackIsReadyEvent( );

                total_mip_processed += current_mip_to_process;
                // current_mip_to_process only matters after the main loop, the TM empirical dist will also update the active_idx_ before returning from Accumulate distribution
                my_dist->AccumulateDistribution(current_mip_to_process);

                // Record an event on cudaStreamPerThread after AccumulateDistribution work is enqueued.
                // The host will use this event (via MakeHostWaitOnMipStackIsReadyEvent) before starting
                // the *next* batch, allowing overlap.
                my_dist->RecordMipStackIsReadyBlockingHost( );

                current_mip_to_process = 0; // Reset for the next batch.
            }

            ccc_counter++;
            total_number_of_cccs_calculated++;

            // if ( ccc_counter % 100 == 0 ) {
            //     my_dist->MakeHostWaitOnMipStackIsReadyEvent( );
            //     my_dist->CopySumAndSumSqAndZero(d_sum1, d_sumSq1);
            //     at_least_100 = true;
            // }

            // if ( ccc_counter % 10000 == 0 ) {
            //     // if we are in this block, we must also have been in the % 100 block, so no need to sync again
            //     d_sum2.AddImage(d_sum1);
            //     d_sum1.Zeros( );

            //     d_sumSq2.AddImage(d_sumSq1);
            //     d_sumSq1.Zeros( );
            // }

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
        my_dist->AccumulateDistribution(current_mip_to_process);

        // We've queued up all the work for the current stack, so record the event that will be used to block the host until the stack is ready
        my_dist->RecordMipStackIsReadyBlockingHost( );
        my_dist->MakeHostWaitOnMipStackIsReadyEvent( );
    }

    // This is run in cudaStreamPerThread
    my_dist->CopySumAndSumSqAndZero(d_sum1, d_sumSq1);

    // FIXME: we can get rid of these sum images since we are using Kahan summation now
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

    // --- Final Synchronization ---
    // Host thread blocks until ALL operations enqueued on cudaStreamPerThread
    // for this RunInnerLoop call are complete. This ensures all GPU results are finalized
    // before the function returns.
    cudaErr(cudaStreamSynchronize(cudaStreamPerThread));
}

/**
 * @global
 * @brief CUDA kernel to update secondary peaks. Not currently used but may be useful in future.
 *
 * This kernel is launched if `n_global_search_images_to_save > 1`.
 * It iterates through pixels (img_index) and for each pixel, checks if the current
 * orientation's CCF peak (`mip_psi[img_index]`) is better than any of the
 * already stored secondary peaks for that pixel. If so, it inserts the new peak
 * and shifts existing ones.
 *
 * Threading:
 * - Standard CUDA grid-stride loop. Each thread processes multiple pixels.
 * - Assumes `secondary_peaks` is a 3D-like array storing (score, psi, theta, phi)
 *   for `NY` (n_global_search_images_to_save) peaks for each of `NX` pixels.
 * - `mip_psi` and `theta_phi` contain the current orientation's peak score, psi, theta, phi.
 * - Access to `secondary_peaks` for a given `img_index` is exclusive to the thread(s)
 *   handling that `img_index`. No race conditions between different `img_index`.
 * - Within an `img_index`, the update logic (finding `best_index` and shifting)
 *   is sequential.
 */
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
    cudaErr(cudaMemsetAsync(mip_psi, 0, sizeof(__half2) * d_input_image->real_memory_allocated, cudaStreamPerThread));
    cudaErr(cudaMemsetAsync(theta_phi, 0, sizeof(__half2) * d_input_image->real_memory_allocated, cudaStreamPerThread));
}
