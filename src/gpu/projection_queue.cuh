#ifndef __SRC_GPU_PROJECTION_QUEUE_CUH__
#define __SRC_GPU_PROJECTION_QUEUE_CUH__

/**
 * @brief Defines the number of projection slots in the ProjectionQueue.
 * This determines how many projections can be processed asynchronously.
 */
constexpr int n_prjs = 20;

#include "../core/stopwatch.h"

/**
 * @class ProjectionQueue
 * @brief Manages a queue of CUDA streams and events for asynchronous GPU projection processing.
 *
 * This class is designed to facilitate overlapping GPU computation (e.g., projection generation,
 * FFTs, data transfers) with other operations. It maintains a pool of CUDA streams and associated
 * events to manage the lifecycle of projection data on the GPU.
 *
 * How it works:
 * 1. A fixed number of "projection slots" (`n_prjs_in_queue_`) are created, each with its own
 *    CUDA stream (`gpu_projection_stream`) and two CUDA events:
 *    - `gpu_projection_is_ready_Event`: Signaled on a `gpu_projection_stream` after a GPU projection
 *      (or data transfer to GPU) is complete and ready for further processing by the main
 *      computation stream (e.g., `cudaStreamPerThread`).
 *    - `cpu_projection_is_writeable_Event`: Signaled on a `gpu_projection_stream` (or `cudaStreamPerThread`
 *      depending on the path) after the data in the corresponding CPU projection buffer (if used)
 *      has been copied to the GPU, or after the GPU projection buffer has been consumed by the main
 *      computation stream. This indicates the CPU buffer or GPU projection slot can be reused.
 *
 * 2. `GetAvailableProjectionIDX()`: This is the core method for acquiring a projection slot.
 *    - It first checks `submitted_prj_queue` (projections that are being processed or have finished copying)
 *      to see if any `cpu_projection_is_writeable_Event` has signaled. If so, that slot is moved
 *      back to `available_prj_queue`.
 *    - If `available_prj_queue` is empty, it means all slots are currently in use. The method then
 *      blocks (busy-waits via `cudaEventSynchronize`) on the `cpu_projection_is_writeable_Event`
 *      of the oldest submitted projection, forcing the host to wait until a slot becomes free.
 *    - Once an available slot is found or becomes free, its index is moved from `available_prj_queue`
 *      to `submitted_prj_queue`, and the index is returned to the caller.
 *
 * 3. `RecordProjectionReadyBlockingHost(idx, stream)`:
 *    - Records `cpu_projection_is_writeable_Event[idx]` on the provided `stream`.
 *    - This event is used by `GetAvailableProjectionIDX` to determine when a projection slot (and its
 *      associated CPU buffer, if applicable) can be safely reused by the host for preparing the next projection.
 *      It signals that the GPU has finished with the data that was in that slot for the *previous* iteration.
 *
 * 4. `RecordGpuProjectionReadyStreamPerThreadWait(idx)`:
 *    - Records `gpu_projection_is_ready_Event[idx]` on `gpu_projection_stream[idx]` (the stream where the
 *      projection was generated or H2D copied).
 *    - Then, it makes the main computation stream (`cudaStreamPerThread`) wait for this event.
 *    - This ensures that operations on `cudaStreamPerThread` that depend on the projection data
 *      (e.g., the main CCF calculation) do not start until the projection is actually ready on the GPU.
 *
 * Thread Safety:
 * - This class is intended to be used by a single host CPU thread that manages a `TemplateMatchingCore` instance.
 * - It is NOT thread-safe if multiple CPU threads attempt to call its methods concurrently on the same instance.
 * - All CUDA stream and event operations are managed internally to ensure correct synchronization
 *   for the asynchronous pipeline orchestrated by this single host thread.
 */
class ProjectionQueue {

  private:
    int             n_prjs_in_queue_;
    cudaEvent_t     gpu_projection_is_ready_Event[n_prjs];
    std::queue<int> available_prj_queue;
    std::queue<int> submitted_prj_queue;

    cudaError_t event_status;

  public:
    cudaStream_t gpu_projection_stream[n_prjs]; ///< Dedicated CUDA streams for each projection slot.
    /**
     * @brief Events: CPU-side projection buffer (or GPU slot) is writeable/reusable by the host.
     * Signaled when the GPU is done with the data from the previous use of this slot.
     */
    cudaEvent_t cpu_projection_is_writeable_Event[n_prjs];

    cistem_timer_noop::StopWatch timer; ///< Timer for profiling busy-wait periods.

    /**
     * @brief Constructor for ProjectionQueue.
     * @param wanted_size The number of projection slots to create and manage.
     */
    ProjectionQueue(int wanted_size) : n_prjs_in_queue_(wanted_size) {
        MyDebugAssertFalse(n_prjs_in_queue_ == 0, "ProjectionQueue must be initialized with a size greater than 0");
        // Initialize queues and create CUDA streams and events.
        ResetQueues( );

        int lowest_priority, highest_priority;
        cudaErr(cudaDeviceGetStreamPriorityRange(&lowest_priority, &highest_priority));
        for ( int i = 0; i < n_prjs_in_queue_; i++ ) {
            // Create dedicated streams for projection operations, potentially with a specific priority.
            cudaErr(cudaStreamCreateWithPriority(&gpu_projection_stream[i], cudaStreamNonBlocking, lowest_priority));
            // Events for signaling GPU projection readiness (for main stream to wait on).
            cudaErr(cudaEventCreateWithFlags(&gpu_projection_is_ready_Event[i], cudaEventBlockingSync)); // Or cudaEventDisableTiming for potentially lower overhead
            // Events for signaling CPU buffer/GPU slot reusability (for host to wait on).
            cudaErr(cudaEventCreateWithFlags(&cpu_projection_is_writeable_Event[i], cudaEventBlockingSync)); // Or cudaEventDisableTiming
        }
    }

    /**
     * @brief Destructor for ProjectionQueue.
     * Cleans up all created CUDA streams and events.
     */
    ~ProjectionQueue( ) {
        for ( int i = 0; i < n_prjs_in_queue_; i++ ) {
            cudaErr(cudaStreamDestroy(gpu_projection_stream[i]));
            cudaErr(cudaEventDestroy(gpu_projection_is_ready_Event[i]));
            cudaErr(cudaEventDestroy(cpu_projection_is_writeable_Event[i]));
        }
    }

    /**
     * @brief Resets the available and submitted projection queues.
     * Called during initialization.
     */
    void ResetQueues( ) {
        while ( ! submitted_prj_queue.empty( ) ) {
            submitted_prj_queue.pop( );
        }
        // All projection slots are initially available.
        for ( int i = 0; i < n_prjs_in_queue_; i++ )
            available_prj_queue.push(i);
    }

    /**
     * @brief Gets the index of an available projection slot.
     *
     * This method manages the recycling of projection slots. It checks if any previously
     * submitted projections are now complete (i.e., their `cpu_projection_is_writeable_Event`
     * has been signaled), making their slots available. If no slots are immediately available,
     * it will block and wait for the oldest submitted projection to complete.
     *
     * @return The index of an available projection slot.
     */
    int
    GetAvailableProjectionIDX( ) {

        // Check submitted projections: if the associated cpu_projection_is_writeable_Event has signaled,
        // it means the slot is free. Move it from submitted to available queue.
        while ( ! submitted_prj_queue.empty( ) ) {
            event_status = cudaEventQuery(cpu_projection_is_writeable_Event[submitted_prj_queue.front( )]);
            if ( event_status == cudaErrorNotReady ) {
                // The oldest submitted projection is not yet ready for reuse. Stop checking.
                break;
            }
            else {
                // This slot is ready. Move it to the available queue.
                available_prj_queue.push(submitted_prj_queue.front( ));
                submitted_prj_queue.pop( );
            }
        }

        // If no slots are available after the check, we must wait.
        if ( available_prj_queue.empty( ) ) {
            // This is a critical point for performance. If the host frequently waits here,
            // it means the GPU projection/processing pipeline is a bottleneck or the queue size is too small.
            timer.start("busy wait");
            // Synchronize (block host) on the cpu_projection_is_writeable_Event of the oldest submitted projection.
            // This ensures the host waits until at least one slot becomes free.
            cudaErr(cudaEventSynchronize(cpu_projection_is_writeable_Event[submitted_prj_queue.front( )]));
            timer.lap("busy wait");
            // The slot is now free. Move it to the available queue.
            available_prj_queue.push(submitted_prj_queue.front( ));
            submitted_prj_queue.pop( );
        }

        // Get an available slot, move it to submitted, and return its index.
        submitted_prj_queue.push(available_prj_queue.front( ));
        available_prj_queue.pop( );

        return submitted_prj_queue.back( ); // Return the index of the slot just moved to submitted.
    }

    /**
     * @brief Records an event indicating that the CPU-side buffer for projection `idx` (or the GPU slot itself)
     *        is now free to be overwritten by the host, or the GPU projection in slot `idx` has been consumed.
     *        This event is recorded on the specified `stream`.
     * @param idx The index of the projection slot.
     * @param stream The CUDA stream on which to record the event.
     */
    inline void
    RecordProjectionReadyBlockingHost(int idx, cudaStream_t stream) {
        // This event signals that the resources associated with projection `idx` (for its *previous* use)
        // are no longer needed by the GPU operations enqueued *up to this point on `stream`*.
        // `GetAvailableProjectionIDX` will later query or synchronize on this event.
        cudaErr(cudaEventRecord(cpu_projection_is_writeable_Event[idx], stream));
    }

    /**
     * @brief Makes the main computation stream (`cudaStreamPerThread`) wait for a GPU projection to be ready.
     *
     * This function first records `gpu_projection_is_ready_Event[idx]` on the specific
     * `gpu_projection_stream[idx]` (where the projection was generated or copied). Then, it makes
     * `cudaStreamPerThread` wait for this event. This ensures that any subsequent work on
     * `cudaStreamPerThread` that uses this projection data will only execute after the projection
     * is fully prepared on the GPU.
     * 
     * @param idx The index of the projection slot whose data needs to be waited upon.
     */
    inline void
    RecordGpuProjectionReadyStreamPerThreadWait(int idx) {
        // Record an event on the projection-specific stream (`gpu_projection_stream[idx]`) to mark
        // the point when the projection data in slot `idx` is ready on the GPU.
        cudaErr(cudaEventRecord(gpu_projection_is_ready_Event[idx], gpu_projection_stream[idx]));
        // Make the main processing stream (`cudaStreamPerThread`) wait for the above event to complete.
        // Subsequent kernels on `cudaStreamPerThread` will only launch after projection `idx` is ready.
        cudaErr(cudaStreamWaitEvent(cudaStreamPerThread, gpu_projection_is_ready_Event[idx], cudaEventWaitDefault));
    }

    /**
     * @brief Prints timing information collected by the internal stopwatch.
     * Useful for profiling the duration of busy-waits in `GetAvailableProjectionIDX`.
     */
    void
    PrintTimes( ) {
        timer.print_times( );
    }
};

#endif