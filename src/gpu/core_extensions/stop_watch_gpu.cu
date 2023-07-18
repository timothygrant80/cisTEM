
#include <cistem_config.h>

#include "../../gpu/gpu_core_headers.h"
#include "../../gpu/GpuImage.h"

// The noop is defined int he cpu header
namespace cistem_timer {
void StopWatch::lap_sync(std::string name, bool threadsafe) {
    // Typically we only want to track a single thread, which is default. Override this with threadsafe = false;
    if ( threadsafe && ReturnThreadNumberOfCurrentThread( ) != 0 )
        return;

    // Check to see if the event has been encountered. If not, first create it and set elapsed time to zero. Return the events index.
    check_for_name_and_set_current_idx(name);
    if ( is_new ) {
        wxPrintf("a new event name was encountered when calling Stopwatch::lap(%s) at line %d in file %s\n", name, __LINE__, __FILE__);
        exit(-1);
    }

    cudaErr(cudaStreamSynchronize(cudaStreamPerThread));

    record_end( );
    record_measured( );
}
} // namespace cistem_timer