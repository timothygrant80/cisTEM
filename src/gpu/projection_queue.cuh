#ifndef __SRC_GPU_PROJECTION_QUEUE_CUH__
#define __SRC_GPU_PROJECTION_QUEUE_CUH__

constexpr int n_prjs = 2;

#include "../core/stopwatch.h"

class ProjectionQueue {

  private:
    int             n_prjs_in_queue_;
    cudaEvent_t     gpu_projection_is_ready_Event[n_prjs];
    std::queue<int> available_prj_queue;
    std::queue<int> submitted_prj_queue;

    cudaError_t event_status;

  public:
    cudaStream_t gpu_projection_stream[n_prjs];
    cudaEvent_t  cpu_projection_is_writeable_Event[n_prjs];

    cistem_timer_noop::StopWatch timer;

    ProjectionQueue(int wanted_size) : n_prjs_in_queue_(wanted_size) {
        MyDebugAssertFalse(n_prjs_in_queue_ == 0, "ProjectionQueue must be initialized with a size greater than 0");
        // We don't need to do anything if the queue has only one member
        ResetQueues( );

        int least_priority, highest_priority;
        cudaErr(cudaDeviceGetStreamPriorityRange(&least_priority, &highest_priority));
        for ( int i = 0; i < n_prjs_in_queue_; i++ ) {
            cudaErr(cudaStreamCreateWithPriority(&gpu_projection_stream[i], cudaStreamNonBlocking, highest_priority - 1));
            cudaErr(cudaEventCreateWithFlags(&gpu_projection_is_ready_Event[i], cudaEventBlockingSync));
            cudaErr(cudaEventCreateWithFlags(&cpu_projection_is_writeable_Event[i], cudaEventBlockingSync));
        }
    }

    ~ProjectionQueue( ) {
        for ( int i = 0; i < n_prjs_in_queue_; i++ ) {
            cudaErr(cudaStreamDestroy(gpu_projection_stream[i]));
            cudaErr(cudaEventDestroy(gpu_projection_is_ready_Event[i]));
            cudaErr(cudaEventDestroy(cpu_projection_is_writeable_Event[i]));
        }
    }

    void ResetQueues( ) {
        while ( ! submitted_prj_queue.empty( ) ) {
            submitted_prj_queue.pop( );
        }
        for ( int i = 0; i < n_prjs_in_queue_; i++ )
            available_prj_queue.push(i);
    }

    int
    GetAvailableProjectionIDX( ) {

        // Remove the oldest projections if they are ready and place them back in the available queue.
        while ( ! submitted_prj_queue.empty( ) ) {
            event_status = cudaEventQuery(cpu_projection_is_writeable_Event[submitted_prj_queue.front( )]);
            if ( event_status == cudaErrorNotReady ) {
                // Okay, we've hit one that isn't ready, so we can stop looking, let's break out
                break;
            }
            else {
                available_prj_queue.push(submitted_prj_queue.front( ));
                submitted_prj_queue.pop( );
            }
        }

        // We only need to spin if there are no more available projections
        if ( available_prj_queue.empty( ) ) {
            timer.start("busy wait");
            cudaErr(cudaEventSynchronize(cpu_projection_is_writeable_Event[submitted_prj_queue.front( )]));
            timer.lap("busy wait");
            available_prj_queue.push(submitted_prj_queue.front( ));
            submitted_prj_queue.pop( );
        }

        submitted_prj_queue.push(available_prj_queue.front( ));
        available_prj_queue.pop( );

        return submitted_prj_queue.back( );
    }

    inline void
    RecordProjectionReadyBlockingHost(int idx, cudaStream_t stream) {
        cudaErr(cudaEventRecord(cpu_projection_is_writeable_Event[idx], stream));
    }

    inline void
    RecordGpuProjectionReadyStreamPerThreadWait(int idx) {
        cudaErr(cudaEventRecord(gpu_projection_is_ready_Event[idx], gpu_projection_stream[idx]));
        cudaErr(cudaStreamWaitEvent(cudaStreamPerThread, gpu_projection_is_ready_Event[idx], cudaEventWaitDefault));
    }

    void
    PrintTimes( ) {
        timer.print_times( );
    }
};

#endif