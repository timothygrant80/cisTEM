#include "gpu_core_headers.h"


DeviceManager::DeviceManager() 
{


};

DeviceManager::DeviceManager(int wanted_number_of_gpus) 
{

  Init(wanted_number_of_gpus);

};

DeviceManager::~DeviceManager() 
{
	if (is_manager_initialized)
	{
		checkCudaErrors(cudaDeviceReset());
	}

};


void DeviceManager::Init(int wanted_number_of_gpus)
{
  wxPrintf("requesting %d gpus\n", wanted_number_of_gpus);

  int gpu_check = -1;
  cudaErr(cudaGetDeviceCount(&gpu_check));
  wxPrintf("CUDA-capable device count: %d\n", gpu_check);
  if (wanted_number_of_gpus > MAX_GPU_COUNT)
  {
    wxPrintf("There are more gpus available (%d) than the max allowed (%d)\n",
              wanted_number_of_gpus, MAX_GPU_COUNT);
	  wanted_number_of_gpus = MAX_GPU_COUNT;
  }

  size_t free_mem;
  size_t total_mem = 0;
  size_t max_mem;
  size_t min_memory_available = 4294967296; // 4 Gb
  size_t min_memory_total = 8589934592; // 8Gb
  int max_mem_idx = 0;
  int cuda_compute_mode;
//  // sleep for a short random bit between 0.5 and 5 seconds. The goal is so mutli-card units have work sent to the next card.
//  wxMilliSleep((global_random_number_generator.GetUniformRandom() + 1.5) * 2000);
  for (int iGPU = 0; iGPU < gpu_check; iGPU++)
  {
	  cudaErr(cudaDeviceGetAttribute(&cuda_compute_mode,cudaDevAttrComputeMode,iGPU));
	  cudaErr(cudaSetDevice(iGPU));
	  if (cuda_compute_mode != cudaComputeModeProhibited)
	  {


		  if (cuda_compute_mode != cudaComputeModeDefault)
		  {
			  wxPrintf("\n\n\tWarning : Your device id %d is not set to Default compute mode.\n\n", iGPU);
		  }
		  cudaErr(cudaMemGetInfo(&free_mem,&total_mem));
//		  wxPrintf("Found %zd free mem out of %zd on iGPU %d\n", free_mem, total_mem, iGPU);
		  if (total_mem < min_memory_total)
		  {
			  // Don't use this device. This migh not be the best way to do this.
			  continue;
		  }
		  else if (free_mem > max_mem && free_mem > min_memory_available) { max_mem = free_mem; max_mem_idx = iGPU; }

	  }
  }

  wxPrintf("Found a max mem of %zd bytes on gpuIDX %d\n", max_mem, max_mem_idx);
  this->nGPUs = wanted_number_of_gpus;
  this->gpuIDX = max_mem_idx;

  is_manager_initialized = true;

};

void DeviceManager::SetGpu(int cpu_thread_idx)
{

  
//	// Select the current device
//	this->gpuIDX = -1;
//	checkCudaErrors(cudaSetDevice(cpu_thread_idx % this->nGPUs));   // "% num_gpus" allows more CPU threads than GPU devices
	cudaErr(cudaSetDevice(this->gpuIDX));
//	checkCudaErrors(cudaGetDevice(&gpuIDX));

//	wxPrintf("For thread %d of nGpus %d assigned %d\n",cpu_thread_idx, nGPUs, gpuIDX);
  


};

void DeviceManager::ReSetGpu()
{

	checkCudaErrors(cudaSetDevice(this->gpuIDX));   // "% num_gpus" allows more CPU threads than GPU devices

};
