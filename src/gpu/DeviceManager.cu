#include "gpu_core_headers.h"
#include "DeviceManager.h"

DeviceManager::DeviceManager( ){

};

DeviceManager::DeviceManager(int wanted_number_of_gpus) {

    Init(wanted_number_of_gpus);
};

DeviceManager::~DeviceManager( ) {
    if ( is_manager_initialized ) {
        cudaErr(cudaDeviceReset( ));
    }
};

void DeviceManager::Init(int wanted_number_of_gpus) {
    wxPrintf("requesting %d gpus\n", wanted_number_of_gpus);

    int gpu_check = -1;
    cudaErr(cudaGetDeviceCount(&gpu_check));
    wxPrintf("CUDA-capable device count: %d\n", gpu_check);
    if ( wanted_number_of_gpus > MAX_GPU_COUNT ) {
        wxPrintf("There are more gpus available (%d) than the max allowed (%d)\n",
                 wanted_number_of_gpus, MAX_GPU_COUNT);
        wanted_number_of_gpus = MAX_GPU_COUNT;
    }

    size_t free_mem;
    size_t total_mem            = 0;
    size_t max_mem              = 0;
    size_t min_memory_available = 4294967296; // 4 Gb
    size_t min_memory_total     = 8589934592; // 8Gb
    int    selected_GPU         = -1;
    int    cuda_compute_mode;
    float  GPU_score;
    float  GPU_score_max = 0.0f;

    //  memset(&prop, 0, sizeof(cudaDeviceProp));
    //  prop.memoryBusWidth = 8192;
    //  prop.memoryClockRate = 4010000;
    //  prop.multiProcessorCount = 160;
    //  prop.totalGlobalMem = 68174610432; // 64 Gb
    //
    //  cudaErr(cudaChooseDevice(&selected_GPU, &prop));
    //  wxPrintf("Chosen device dumber: %d\n", selected_GPU);

    //  // sleep for a short random bit between 0.5 and 5 seconds. The goal is so mutli-card units have work sent to the next card.
    //  wxMilliSleep((global_random_number_generator.GetUniformRandom() + 1.5) * 2000);
    for ( int iGPU = 0; iGPU < gpu_check; iGPU++ ) {
        cudaDeviceProp prop;
        cudaErr(cudaGetDeviceProperties(&prop, iGPU));
        //	  wxPrintf("Device Number: %d\n", iGPU);
        //	  wxPrintf("  Device name: %s\n", prop.name);
        //	  wxPrintf("  Memory Clock Rate (KHz): %d\n", prop.memoryClockRate);
        //	  wxPrintf("  Memory Bus Width (bits): %d\n", prop.memoryBusWidth);
        //	  wxPrintf("  Peak Memory Bandwidth (GB/s): %f\n", 2.0*prop.memoryClockRate*(prop.memoryBusWidth/8)/1.0e6);
        //	  wxPrintf("  Number of multiprocessors: %d\n", prop.multiProcessorCount);
        //	  wxPrintf("  Threads per multiprocessor: %d\n", prop.maxThreadsPerMultiProcessor);
        //	  wxPrintf("  Memory per multiprocessor: %li\n", long(prop.sharedMemPerMultiprocessor));
        //	  wxPrintf("  Memory on device: %li\n", long(prop.totalGlobalMem));
        //	  wxPrintf("\n");

        cudaErr(cudaDeviceGetAttribute(&cuda_compute_mode, cudaDevAttrComputeMode, iGPU));
        cudaErr(cudaSetDevice(iGPU));

        //	  cudaErr(cudaMemGetInfo(&free_mem,&total_mem));
        //	  wxPrintf("Found %zd free mem out of %zd on iGPU %d\n", free_mem, total_mem, iGPU);
        //	  wxPrintf("cuda_compute_mode, cudaDevAttrComputeMode, cudaComputeModeProhibited = %i %i %i\n", cuda_compute_mode, cudaDevAttrComputeMode, cudaComputeModeProhibited);

        if ( cuda_compute_mode != cudaComputeModeProhibited && prop.totalGlobalMem > min_memory_available ) {
            if ( cuda_compute_mode != cudaComputeModeDefault ) {
                wxPrintf("\n\n\tWarning : Your device id %d is not set to Default compute mode.\n\n", iGPU);
            }
            cudaErr(cudaMemGetInfo(&free_mem, &total_mem));
            //		  wxPrintf("Found %zd free mem out of %zd on iGPU %d\n", free_mem, total_mem, iGPU);
            if ( total_mem < min_memory_total ) {
                // Don't use this device. This might not be the best way to do this.
                cudaErr(cudaDeviceReset( ));
                continue;
            }
            else if ( free_mem > max_mem && free_mem > min_memory_available ) {
                max_mem = free_mem;
            }
            //		  else if (free_mem > max_mem && free_mem > min_memory_available) { max_mem = free_mem; selected_GPU = iGPU; }

            GPU_score = (float)free_mem * (float)prop.memoryClockRate * (float)prop.memoryBusWidth * (float)prop.multiProcessorCount * (float)prop.maxThreadsPerMultiProcessor;
            //		  wxPrintf("  Memory Clock Rate (KHz): %d\n", prop.memoryClockRate);
            //		  wxPrintf("  Memory Bus Width (bits): %d\n", prop.memoryBusWidth);
            //		  wxPrintf("  Number of multiprocessors: %d\n", prop.multiProcessorCount);
            //		  wxPrintf("  Threads per multiprocessor: %d\n", prop.maxThreadsPerMultiProcessor);
            //		  wxPrintf("GPU score = %g\n", GPU_score);
            if ( GPU_score > GPU_score_max ) {
                GPU_score_max = GPU_score;
                selected_GPU  = iGPU;
            }
        }
        cudaErr(cudaDeviceReset( ));
    }

    MyAssertTrue(selected_GPU >= 0, "No suitable GPU found. Terminating...");

    wxPrintf("Found a max mem of %zd bytes on gpuIDX %d\n", max_mem, selected_GPU);
    this->nGPUs  = wanted_number_of_gpus;
    this->gpuIDX = selected_GPU;

    is_manager_initialized = true;
};

void DeviceManager::SetGpu(int cpu_thread_idx) {

    //	// Select the current device
    //	this->gpuIDX = -1;
    //	cudaErr(cudaSetDevice(cpu_thread_idx % this->nGPUs));   // "% num_gpus" allows more CPU threads than GPU devices
    cudaErr(cudaSetDevice(this->gpuIDX));
    //	cudaErr(cudaGetDevice(&gpuIDX));

    //	wxPrintf("For thread %d of nGpus %d assigned %d\n",cpu_thread_idx, nGPUs, gpuIDX);
};

void DeviceManager::ResetGpu( ) {

    cudaErr(cudaDeviceReset( )); // "% num_gpus" allows more CPU threads than GPU devices
};

void DeviceManager::ListDevices( ) {
    // To get GPU load:
    // https://stackoverflow.com/questions/46801136/cuda-get-gpu-load-percent

    int    gpu_check = 0;
    size_t free_mem;
    size_t total_mem;

    cudaErr(cudaGetDeviceCount(&gpu_check));
    wxPrintf("CUDA-capable device count: %d\n\n", gpu_check);

    size_t min_memory_available = 4294967296; // 4 Gb
    int    cuda_compute_mode;

    for ( int iGPU = 0; iGPU < gpu_check; iGPU++ ) {
        cudaErr(cudaDeviceGetAttribute(&cuda_compute_mode, cudaDevAttrComputeMode, iGPU));
        cudaErr(cudaSetDevice(iGPU));
        cudaErr(cudaMemGetInfo(&free_mem, &total_mem));

        cudaDeviceProp prop;
        cudaErr(cudaGetDeviceProperties(&prop, iGPU));
        wxPrintf("Device number: %d\n", iGPU);
        wxPrintf("  Device name: %s\n", prop.name);
        wxPrintf("  Memory clock cate (KHz): %d\n", prop.memoryClockRate);
        wxPrintf("  Memory bus width (bits): %d\n", prop.memoryBusWidth);
        wxPrintf("  Memory bandwidth (GB/s): %f\n", 2.0 * prop.memoryClockRate * (prop.memoryBusWidth / 8) / 1.0e6);
        wxPrintf("  Number of multiprocessors: %d\n", prop.multiProcessorCount);
        wxPrintf("  Threads per multiprocessor: %d\n", prop.maxThreadsPerMultiProcessor);
        wxPrintf("  Maximum 3dArray size: %i, %i, %i\n", prop.maxSurface3D[0], prop.maxSurface3D[1], prop.maxSurface3D[2]); // for texture cache
        // wxPrintf("  Memory per multiprocessor (GB): %f\n", float(prop.sharedMemPerMultiprocessor) / 1024 / 1024 / 1024);
        wxPrintf("  Memory on device (GB): %f\n", float(prop.totalGlobalMem) / 1024 / 1024 / 1024);
        wxPrintf("  Free memory (GB): %f\n", float(free_mem) / 1024 / 1024 / 1024);
        // if (cuda_compute_mode == cudaComputeModeProhibited) wxPrintf("  Device incompatible with CUDA\n");
        if ( prop.totalGlobalMem <= min_memory_available )
            wxPrintf("  *** Not enough memory on device, %f GB needed ***\n", float(min_memory_available) / 1024 / 1024 / 1024);
        wxPrintf("\n");
    }
};
