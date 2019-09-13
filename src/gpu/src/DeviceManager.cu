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
	// TODO make sure this only destroying the proper context.
	checkCudaErrors(cudaDeviceReset());
};


void DeviceManager::Init(int wanted_number_of_gpus)
{
  wxPrintf("requesting %d gpus\n", wanted_number_of_gpus);

  int gpu_check = -1;
  checkCudaErrors(cudaGetDeviceCount(&gpu_check));
  wxPrintf("CUDA-capable device count: %d\n", gpu_check);
  if (wanted_number_of_gpus > MAX_GPU_COUNT)
  {
    wxPrintf("There are more gpus available (%d) than the max allowed (%d)\n",
              wanted_number_of_gpus, MAX_GPU_COUNT);
	  wanted_number_of_gpus = MAX_GPU_COUNT;
  }

  this->nGPUs = wanted_number_of_gpus;

};

void DeviceManager::SetGpu(int cpu_thread_idx)
{

  
	// Select the current device
	this->gpuIDX = -1;
	checkCudaErrors(cudaSetDevice(cpu_thread_idx % this->nGPUs));   // "% num_gpus" allows more CPU threads than GPU devices
	checkCudaErrors(cudaGetDevice(&gpuIDX));
  wxPrintf("For thread %d of nGpus %d assigned %d\n",cpu_thread_idx, nGPUs, gpuIDX);
  


};

void DeviceManager::ReSetGpu()
{

	checkCudaErrors(cudaSetDevice(this->gpuIDX));   // "% num_gpus" allows more CPU threads than GPU devices

};
