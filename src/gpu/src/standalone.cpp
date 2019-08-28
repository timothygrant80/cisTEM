// Copyright 2016 Howard Hughes Medical Institute.
// All rights reserved.
// Use is subject to The Janelia Research Campus Software License 1.2
// license terms ( http://license.janelia.org/license/janelia_license_1_2.html )
//
//*****************************************************************************
#include "gpu_core_headers.h"



const bool DO_FULL_PROCESS = true;
const bool DO_PROJECTION = true;
const bool CLOBBER_SYNC = true;
const int nThreads = 1;

// #define MAX(a, b) (a > b ? a : b)




__global__ void PadData(const float *ref, float *ref_padded, int3 ref_size, int2 img_size, int2 offset, int padding_jump_value);
__global__ void mipTheData(cufftReal *mip, const cufftReal *correlation_output, const int2 size, const int padding_jump_value);
__global__ void sumAndSquareSum(cufftReal *sum, cufftReal *sum_of_squares, const cufftReal *correlation_output, const int2 size, const int padding_jump_value);


// Complex conj multiplication
__global__ void ComplexPointwiseConjMul(cufftComplex *correlation_output_complex, const Complex *img, const Complex *ref, const int2 img_size, const int padding_jump_value);


void RunSearch(float *host_mip, int number_of_search_positions, float psi_max, float psi_step, int ref_dimension, int img_dimension);
void host_set_random(float *ref_projection, int ref_memory_allocated, int padding_jump_value);
void SetToEulerRotation(float *m, float wanted_euler_phi_in_degrees, float wanted_euler_theta_in_degrees, float wanted_euler_psi_in_degrees);
void SetupTextureMemory(TGPUplan plan[], int3 ref_size, int ref_real_memory_allocated, int padding_jump_value, int iThread );
void UpdateLoopingAndAddressing(TGPUplan plan[], const int3 ref_size, const int iGPU);
inline int ReturnFourierLogicalCoordGivenPhysicalCoord(int physical_index, int physical_address_of_box_center, int logical_dimension );
inline long ReturnFourier1DAddressFromLogicalCoord(TGPUplan plan[], int3 ref_size, int iThread, int wanted_x, int wanted_y, int wanted_z);



void SplitRealAndImag(TGPUplan plan[], int3 ref_size,int iThread, int real_memory_allocated);

float GetUniformRandom();


__global__ void transformKernel_FWD(cudaTextureObject_t thisTexObj_real,
									cudaTextureObject_t thisTexObj_imag,
									cufftComplex *outputData,
									int3 dims,
									float *rotMat,
									cufftComplex extrapVal);

void host_set_random(float *ref_projection, int ref_memory_allocated)
{
	for (int pixel_counter = 0 ; pixel_counter < ref_memory_allocated; pixel_counter++ )
	{
		ref_projection[pixel_counter] = GetUniformRandom();
	}

}



int main()
{
	// Limited to even numbers
	int number_of_search_positions = 1000;//2e6;
    int img_dimension = 1024;
    int ref_dimension = 256;
	float psi_max = 360.0;
	float psi_step = 20.0;

	float *host_mip = NULL;

	RunSearch(host_mip, number_of_search_positions, psi_max, psi_step, ref_dimension, img_dimension);

	return 0;
}

void RunSearch(float *host_mip, int number_of_search_positions, float psi_max, float psi_step, int ref_dimension, int img_dimension)
{


	int iReplicate;
	int nReplicates = number_of_search_positions / nThreads; // The 10 is from the outer loop of Niko's code (ii)


	int number_of_psi_positions;
	int n_psi_angles;

	bool test_mirror = true;

	int3 ref_size;
	int3 complex_dims;
	int2 img_size;
	int2 offset;
	int ref_real_memory_allocated;
	int img_real_memory_allocated;
	int padding_jump_value;



	ref_size.x = ref_dimension; ref_size.y = ref_dimension;
	img_size.x = img_dimension; img_size.y = img_dimension;
	offset.x = img_size.x - ref_dimension; offset.y = img_size.y - ref_dimension;

	if (IsEven(ref_size.x) == true) ref_real_memory_allocated =  ref_size.x / 2 + 1;
	else ref_real_memory_allocated = (ref_size.x - 1) / 2 + 1;
	ref_real_memory_allocated *= ref_size.y;
	ref_real_memory_allocated *= 2;

	if (IsEven(img_size.x) == true) img_real_memory_allocated =  img_size.x / 2 + 1;
	else img_real_memory_allocated = (img_size.x - 1) / 2 + 1;
	img_real_memory_allocated *= img_size.y;
	img_real_memory_allocated *= 2;

	if (IsEven(img_size.x) == true) padding_jump_value = 2;
	else padding_jump_value = 1;

	dim3 dimBlock = dim3(32, 32, 1);
    dim3 ref_dimGrid;


    if (DO_PROJECTION)
    {
    	// if it is 3d, for now assume it to be cubic
    	ref_size.z = ref_size.x;
    	ref_dimGrid  = dim3(ref_size.x / dimBlock.x, ref_size.y / dimBlock.y, ref_size.z);
    	ref_real_memory_allocated *= ref_size.z;
    	complex_dims.x = ref_size.x / 2; complex_dims.y = ref_size.y; complex_dims.z = ref_size.z;
    }
    else
    {

    	ref_size.z = 1;
    	ref_dimGrid  = dim3(ref_size.x / dimBlock.x, ref_size.y / dimBlock.y, 1);
    }


    dim3 img_dimGrid(img_size.x / dimBlock.x, img_size.y / dimBlock.y, 1);



	//Solver config
	TGPUplan      plan[MAX_GPU_COUNT]; // TODO change this to a separate GPU thread class
	int nGPUs = 2;


	printf("Starting simpleMultiGPU\n");
	checkCudaErrors(cudaGetDeviceCount(&nGPUs));

	if (nGPUs > MAX_GPU_COUNT)
	{
		nGPUs = MAX_GPU_COUNT;
	}

	printf("CUDA-capable device count: %i\n", nGPUs);

	printf("Generating input data...\n\n");

	printf("Searching %d positions (%3.3e)\n", number_of_search_positions, (float)number_of_search_positions);


	//Create streams for issuing GPU command asynchronously and allocate memory (GPU and System page-locked)
  // Add dynamic allocation that could be large than nGPUs and is not smaller
//	omp_set_nested(1);
    omp_set_num_threads(nThreads);  // create as many CPU threads as there are CUDA devices
	#pragma omp parallel //for num_threads(nThreads)
//    for ( int iLoop = 1; iLoop <= nThreads; iLoop++)
	{

    	unsigned int cpu_thread_id = omp_get_thread_num();
		unsigned int num_cpu_threads = omp_get_num_threads();

		// Select the current device
	    int gpu_idx = -1;
	    unsigned int iGPU; // FIXME change this to iThread
		checkCudaErrors(cudaSetDevice(cpu_thread_id % nGPUs));   // "% num_gpus" allows more CPU threads than GPU devices
		checkCudaErrors(cudaGetDevice(&gpu_idx));
		iGPU = cpu_thread_id;
		printf("CPU thread %d (of %d) uses CUDA device %d %d\n", cpu_thread_id, num_cpu_threads, gpu_idx, iGPU);

		plan[iGPU].first_loop = true;

		if (DO_PROJECTION) {UpdateLoopingAndAddressing(plan, ref_size,iGPU);}




	    // create cuda event handles
	    cudaEvent_t start, stop;
	    checkCudaErrors(cudaEventCreate(&start));
	    checkCudaErrors(cudaEventCreate(&stop));



		checkCudaErrors(cudaStreamCreate(&plan[iGPU].stream));
		printf("stream pointer %p at line %d\n",plan[iGPU].stream,__LINE__);

		//Allocate pinned host memory
		checkCudaErrors(cudaMallocHost((void **)&plan[iGPU].ref,      sizeof(float)*ref_real_memory_allocated));
		checkCudaErrors(cudaMallocHost((void **)&plan[iGPU].ref_real, sizeof(float)*ref_real_memory_allocated/2));
		checkCudaErrors(cudaMallocHost((void **)&plan[iGPU].ref_imag, sizeof(float)*ref_real_memory_allocated/2));
		checkCudaErrors(cudaMallocHost((void **)&plan[iGPU].img, sizeof(float)*img_real_memory_allocated));

		//Allocate device memory

		// The reference comes in as either a projection in Fourier space or the 3d in Fourier space. Overkill on allocations, but less confusing for demonstration. ONly the 3d will survive testing.
		checkCudaErrors(cudaMalloc((void **)&plan[iGPU].d_ref3d_real, sizeof(float)*ref_real_memory_allocated/2));
		checkCudaErrors(cudaMalloc((void **)&plan[iGPU].d_ref3d_imag, sizeof(float)*ref_real_memory_allocated/2));
		checkCudaErrors(cudaMalloc((void **)&plan[iGPU].d_ref, sizeof(float)*ref_real_memory_allocated));
		plan[iGPU].d_ref_complex = (cufftComplex *)plan[iGPU].d_ref;

		printf("stream pointer %p at line %d\n",plan[iGPU].stream,__LINE__);

		//
		checkCudaErrors(cudaMalloc((void **)&plan[iGPU].d_ref_padded, sizeof(float)*img_real_memory_allocated));
		plan[iGPU].d_ref_padded_complex = (cufftComplex *)plan[iGPU].d_ref_padded;

		// The image comes in from the host pre-processed and FFT'd
		checkCudaErrors(cudaMalloc((void **)&plan[iGPU].d_img_complex, sizeof(float)*img_real_memory_allocated)); \

		// B/c the same CTF's will be used repeatedly, it probably makes sense to pass CTF images not objects in from the host
		// TODO

		printf("stream pointer %p at line %d\n",plan[iGPU].stream,__LINE__);

		checkCudaErrors(cudaMalloc((void **)&plan[iGPU].d_correlation_output, sizeof(float)*img_real_memory_allocated));
		plan[iGPU].d_correlation_output_complex = (cufftComplex *)plan[iGPU].d_correlation_output_complex;

		checkCudaErrors(cudaMalloc((void **)&plan[iGPU].d_mip, sizeof(float)*img_real_memory_allocated));
		checkCudaErrors(cudaMalloc((void **)&plan[iGPU].d_sum, sizeof(float)*img_real_memory_allocated));
		checkCudaErrors(cudaMalloc((void **)&plan[iGPU].d_sum_of_squares, sizeof(float)*img_real_memory_allocated));

		printf("stream pointer %p at line %d\n",plan[iGPU].stream,__LINE__);

		// Memory for the projections we will extract from the 3d ref
		checkCudaErrors(cudaMalloc((void **)&plan[iGPU].d_prj, sizeof(float)*ref_real_memory_allocated/ref_size.z));
		plan[iGPU].d_prj_complex = (cufftComplex *)plan[iGPU].d_prj_complex;

		printf("stream pointer %p at line %d\n",plan[iGPU].stream,__LINE__);

		if (DO_PROJECTION)
		{


			SetupTextureMemory(plan,  ref_size,  ref_real_memory_allocated,  padding_jump_value,  iGPU );
			printf("stream pointer %p at line %d\n",plan[iGPU].stream,__LINE__);
			SplitRealAndImag(plan,  ref_size,  iGPU, ref_real_memory_allocated);
			printf("stream pointer %p at line %d\n",plan[iGPU].stream,__LINE__);

			checkCudaErrors(cudaMemcpyAsync((void*) plan[iGPU].d_ref3d_real,(const void*) plan[iGPU].ref_real, sizeof(float)*ref_real_memory_allocated/2, cudaMemcpyHostToDevice,plan[iGPU].stream));
			checkCudaErrors(cudaMemcpyAsync((void*) plan[iGPU].d_ref3d_imag,(const void*) plan[iGPU].ref_imag, sizeof(float)*ref_real_memory_allocated/2, cudaMemcpyHostToDevice,plan[iGPU].stream));

		}


		printf("stream pointer %p at line %d\n",plan[iGPU].stream,__LINE__);




		printf("stream pointer %p at line %d\n",plan[iGPU].stream,__LINE__);

		// Not sure that a plan is needed for each? But I think having it on that GPUs memory makes sense
		int fftDims[2];
		// backward plan ref
		fftDims[0] = ref_size.x; fftDims[1] = ref_size.y;
		cufftPlanMany(&plan[iGPU].cuda_plan_inverse_ref, 2, fftDims, NULL, NULL, NULL,NULL, NULL, NULL, CUFFT_C2R, 1);
		cufftSetStream(plan[iGPU].cuda_plan_inverse_ref, plan[iGPU].stream);

		// backward plan img
		fftDims[0] = img_size.x; fftDims[1] = img_size.y;
		cufftPlanMany(&plan[iGPU].cuda_plan_inverse_img, 2, fftDims, NULL, NULL, NULL,NULL, NULL, NULL, CUFFT_C2R, 1);
		cufftSetStream(plan[iGPU].cuda_plan_inverse_img, plan[iGPU].stream);

		// forward plan
		cufftPlanMany(&plan[iGPU].cuda_plan_forward, 2, fftDims, NULL, NULL, NULL,NULL, NULL, NULL, CUFFT_R2C, 1);
		cufftSetStream(plan[iGPU].cuda_plan_forward, plan[iGPU].stream);

		// Fill the host image with random numbers - these are a standin for the pre-whitened etc. image to search
		host_set_random(plan[iGPU].img, img_real_memory_allocated);
		checkCudaErrors(cudaMemcpyAsync((void*) plan[iGPU].d_img_complex,(const void*) plan[iGPU].img, sizeof(float)*img_real_memory_allocated, cudaMemcpyHostToDevice,plan[iGPU].stream));

		host_set_random(plan[iGPU].ref, ref_real_memory_allocated);

		printf("stream pointer %p at line %d\n",plan[iGPU].stream,__LINE__);


		// Make sure all of the devices are ready to start. Mayby this could be a stream sync if the parallel block is extended
		checkCudaErrors(cudaStreamSynchronize( plan[iGPU].stream ));

		// Get rid of the host image freeing up the pinned memory
		checkCudaErrors(cudaFreeHost(plan[iGPU].img));


		/// Now we move on to the actual computation


		for (iReplicate = 0; iReplicate < nReplicates; iReplicate++)
		{


			printf("Doing replicate %d on thread %d\n",iReplicate, iGPU);
			if ( ! plan[iGPU].first_loop )
			{
				// Once the new data is copied in, make sure all of the computes from the last loop have finished
				checkCudaErrors(cudaStreamSynchronize( plan[iGPU].stream ));
			}

			float set_ref_to_value = 0.0f;
			// The padded reference needs to be rezeroed (or possible set to the mean)
			checkCudaErrors(cudaMemsetAsync(plan[iGPU].d_ref_padded, set_ref_to_value,img_real_memory_allocated,plan[iGPU].stream));
			if (CLOBBER_SYNC) {cudaDeviceSynchronize();}



			// Get to a full padded reference that is in Fourier Space
			if (DO_FULL_PROCESS)
			{


				if (DO_PROJECTION)
				{
					// Get the reference projection from the 3d, using for now some random angles
					float rotMat[9] = {0.0f,0.0f,0.0f,0.0f,0.0f,0.0f,0.0f,0.0f,0.0f};
					cufftComplex EXTRAPVAL;
					EXTRAPVAL.x = 0.0f;
					EXTRAPVAL.y = 0.0f;
					SetToEulerRotation(rotMat, GetUniformRandom()*2*PIf,GetUniformRandom()*2*PIf,GetUniformRandom()*2*PIf);
					if (CLOBBER_SYNC) {cudaDeviceSynchronize();}

					  transformKernel_FWD<<< ref_dimGrid, dimBlock, 0, plan[iGPU].stream >>>(plan[iGPU].thisTexObj_real,
							  	  	  	  	  	  	  	  	  	  	  	  	  	  	  	  	 plan[iGPU].thisTexObj_imag,
																							 plan[iGPU].d_prj_complex,
																							 complex_dims, rotMat,EXTRAPVAL);
						if (CLOBBER_SYNC) {cudaDeviceSynchronize();}

					checkCudaErrors(cudaMalloc((void **)&plan[iGPU].d_prj, sizeof(float)*ref_real_memory_allocated/ref_size.z));
					checkCudaErrors(cudaMalloc((void **)&plan[iGPU].d_prj_complex, sizeof(float)*ref_real_memory_allocated/ref_size.z));
				}
				else
				{
					// Get the reference projection, from the host
					checkCudaErrors(cudaMemcpyAsync((void*)plan[iGPU].d_ref_complex,(void*)plan[iGPU].ref, sizeof(float)*ref_real_memory_allocated, cudaMemcpyHostToDevice,plan[iGPU].stream));
					if (CLOBBER_SYNC) {cudaDeviceSynchronize();}
				}



				if (ref_size.x != img_size.x || ref_size.y != img_size.y)
				{
					// Queue up the padding kernel
					PadData<<< ref_dimGrid, dimBlock, 0, plan[iGPU].stream >>>( plan[iGPU].d_ref, plan[iGPU].d_ref_padded, ref_size, img_size, offset, padding_jump_value);
					if (CLOBBER_SYNC) {cudaDeviceSynchronize();}
				}

				// We are actually passing the fft of the projection, so we need to take ifft prior to padding
				checkCudaErrors(cufftExecC2R(plan[iGPU].cuda_plan_inverse_ref, plan[iGPU].d_ref_complex, plan[iGPU].d_ref));
				if (CLOBBER_SYNC) {cudaDeviceSynchronize();}

				// Now forward FFT the padded image
				checkCudaErrors(cufftExecR2C(plan[iGPU].cuda_plan_forward, plan[iGPU].d_ref_padded, plan[iGPU].d_ref_padded_complex));
				if (CLOBBER_SYNC) {cudaDeviceSynchronize();}


			}
			else
			{
				// Once the host has made a projection, queue it for async xfer
				// I could probably combine the memcopy and the padding into a memcpy2DToArray
				checkCudaErrors(cudaMemcpyAsync((void*)plan[iGPU].d_ref,(void*)plan[iGPU].ref, sizeof(float)*ref_real_memory_allocated, cudaMemcpyHostToDevice,plan[iGPU].stream));
				if (CLOBBER_SYNC) {checkCudaErrors(cudaStreamSynchronize(plan[iGPU].stream));}


				if (ref_size.x != img_size.x || ref_size.y != img_size.y)
				{
					// Queue up the padding kernel
					PadData<<< ref_dimGrid, dimBlock, 0, plan[iGPU].stream >>>( plan[iGPU].d_ref, plan[iGPU].d_ref_padded, ref_size, img_size, offset, padding_jump_value);
					if (CLOBBER_SYNC) {cudaDeviceSynchronize();}
				}


//				 Here to match the original test we assume a magically xformed reference
				plan[iGPU].d_ref_complex =  reinterpret_cast<cufftComplex*>(plan[iGPU].d_ref_padded);
				if (CLOBBER_SYNC) {cudaStreamSynchronize(plan[iGPU].stream);}



			}



			// Now we want to do the complex conjugate multiply
			ComplexPointwiseConjMul<<< img_dimGrid, dimBlock, 0, plan[iGPU].stream >>>(plan[iGPU].d_correlation_output_complex,(const Complex*)plan[iGPU].d_img_complex,(const Complex*)plan[iGPU].d_ref_padded_complex,img_size, padding_jump_value);
			if (CLOBBER_SYNC) {cudaDeviceSynchronize();}

			// Now get the mip from the ifft of the correlation output
			checkCudaErrors(cufftExecC2R(plan[iGPU].cuda_plan_inverse_img, plan[iGPU].d_correlation_output_complex, plan[iGPU].d_correlation_output));
			if (CLOBBER_SYNC) {cudaDeviceSynchronize();}


			// Now we need to see if accumulate into the mip
			if (plan[iGPU].first_loop)
			{

//				checkCudaErrors(cudaStreamSynchronize(plan[iGPU].stream));
				checkCudaErrors(cudaMemcpyAsync((void*) plan[iGPU].d_mip, (const void*)plan[iGPU].d_correlation_output, sizeof(cufftReal)*img_real_memory_allocated, cudaMemcpyDeviceToDevice, plan[iGPU].stream));
				if (CLOBBER_SYNC) {checkCudaErrors(cudaStreamSynchronize(plan[iGPU].stream));}

				// Get to a full padded reference that is in Fourier Space
				if (DO_FULL_PROCESS)
				{
					// The sum and sum of squares need to be initialized in the first loop
					checkCudaErrors(cudaMemsetAsync(plan[iGPU].d_sum, 0.0f, img_real_memory_allocated,plan[iGPU].stream));
					if (CLOBBER_SYNC) {cudaDeviceSynchronize();}
					checkCudaErrors(cudaMemsetAsync(plan[iGPU].d_sum_of_squares, 0.0f, img_real_memory_allocated,plan[iGPU].stream));
					// This needs to complete before we get to the next kernel - it should, but throw a synchronize explicitly
					checkCudaErrors(cudaStreamSynchronize(plan[iGPU].stream));

					sumAndSquareSum<<< img_dimGrid, dimBlock, 0, plan[iGPU].stream >>>( plan[iGPU].d_sum, plan[iGPU].d_sum_of_squares, plan[iGPU].d_correlation_output, img_size, padding_jump_value);
					if (CLOBBER_SYNC) {checkCudaErrors(cudaStreamSynchronize(plan[iGPU].stream));}
				}



				plan[iGPU].first_loop = false;
			}
			else
			{

				 mipTheData<<< img_dimGrid, dimBlock, 0, plan[iGPU].stream >>>(plan[iGPU].d_mip,  plan[iGPU].d_correlation_output, img_size,  padding_jump_value);
					// Get to a full padded reference that is in Fourier Space
				if (DO_FULL_PROCESS)
				{
				 if (CLOBBER_SYNC) {checkCudaErrors(cudaStreamSynchronize(plan[iGPU].stream));}
				 sumAndSquareSum<<< img_dimGrid, dimBlock, 0, plan[iGPU].stream >>>( plan[iGPU].d_sum, plan[iGPU].d_sum_of_squares, plan[iGPU].d_correlation_output, img_size, padding_jump_value);
				}


			}

			if (CLOBBER_SYNC) {checkCudaErrors(cudaStreamSynchronize(plan[iGPU].stream));}




		} // end loop over replicates



	}// end of parallel block for allocation



    // Now we want to sync everything and accumulate the final mips
    cudaDeviceSynchronize();

	if (DO_FULL_PROCESS)
	{

		for (int iThread = 1; iThread < nThreads; iThread++)
		{
			mipTheData<<< img_dimGrid, dimBlock, 0, plan[0].stream >>>(plan[0].d_mip,  plan[iThread].d_mip, img_size,  padding_jump_value);
		}

	    cudaDeviceSynchronize();

	    // FIXME FIXME
		// Send the results back to the host - I think this pointer should have been passed in from the host originally! FIXME
//		checkCudaErrors(cudaMallocHost((void **)host_mip, sizeof(float)*img_real_memory_allocated));

//		checkCudaErrors(cudaMemcpy((void*)host_mip,(const void*)plan[0].d_mip, sizeof(float)*img_real_memory_allocated, cudaMemcpyDeviceToHost));

	}

	// Now clean everything up
    for (int iThread = 0; iThread < nThreads; iThread++)
    {
    	// Clean up and free memory
		checkCudaErrors(cudaFree(plan[iThread].d_ref));
		checkCudaErrors(cudaFree(plan[iThread].d_ref_padded));
		checkCudaErrors(cudaFree(plan[iThread].d_ref_padded_complex));
		checkCudaErrors(cudaFree(plan[iThread].d_img_complex)); // The normalized complex image is passed in
		checkCudaErrors(cudaFree(plan[iThread].d_mip));
		checkCudaErrors(cudaFree(plan[iThread].d_correlation_output));
		checkCudaErrors(cudaFree(plan[iThread].d_correlation_output_complex));



		if (DO_FULL_PROCESS)
		{
			checkCudaErrors(cudaFree(plan[iThread].d_ref_complex));

		}

		//De-Allocate pinned host memory
		checkCudaErrors(cudaFreeHost(plan[iThread].ref));

    }





}


float GetUniformRandom() {
	float rnd1 = (float)rand();
	float hmax = ((float) RAND_MAX) / 2.0;
	float rnd2 = (rnd1 - hmax) / hmax;
	return rnd2;
}

__global__ void mipTheData(cufftReal *mip, const cufftReal *correlation_output, const int2 size, const int padding_jump_value)
{
	  const unsigned int x = blockIdx.x*blockDim.x + threadIdx.x;
	  if (x >= size.x - padding_jump_value) { return ; }
	  const unsigned int y = blockIdx.y*blockDim.y + threadIdx.y;
	  if (y >= size.y) { return ; }


	  int address = y*size.x + x;

	  mip[address] = MAX(mip[address], correlation_output[address]);
}

// It might be faster to split these two kernels so they can run concurrent? I can see the memory access also being a bummer
__global__ void sumAndSquareSum(cufftReal *sum, cufftReal *sum_of_squares, const cufftReal *correlation_output, const int2 size, const int padding_jump_value)
{
	  const unsigned int x = blockIdx.x*blockDim.x + threadIdx.x;
	  if (x >= size.x - padding_jump_value) { return ; }
	  const unsigned int y = blockIdx.y*blockDim.y + threadIdx.y;
	  if (y >= size.y) { return ; }

	  int address = y*size.x + x;

	  sum[address] += correlation_output[address];
	  sum_of_squares[address] += correlation_output[address]*correlation_output[address];
}

////////////////////////////////////////////////////////////////////////////////
// Pad data////////////////////////////////////////////////////////////////////////////////

__global__ void PadData(const float *ref, float *ref_padded, int3 ref_size, int2 img_size, int2 offset, int padding_jump_value)
{

	/*
	 * It is assumed that the kernel is << half the image size so that simply padding
	 * to the image size is sufficient to prevent wraparound.
	 */
	  const unsigned int x = blockIdx.x*blockDim.x + threadIdx.x;
	  if (x >= ref_size.x - padding_jump_value) { return ; }
	  const unsigned int y = blockIdx.y*blockDim.y + threadIdx.y;
	  if (y >= ref_size.y) { return ; }


	  ref_padded[((y+offset.y)*(img_size.x)) + (x + offset.x)] = ref[y*ref_size.x + x];

}

////////////////////////////////////////////////////////////////////////////////
// Complex operations
////////////////////////////////////////////////////////////////////////////////

// Complex pointwise multiplication
__global__ void ComplexPointwiseConjMul(cufftComplex *correlation_output_complex, const Complex *img, const Complex *ref, int2 img_size, int padding_jump_value)
{
	  const int x = blockIdx.x*blockDim.x + threadIdx.x;
	  if (x >= img_size.x / 2 ) { return ; }
	  const int y = blockIdx.y*blockDim.y + threadIdx.y;
	  if (y >= img_size.y) { return ; }

	  long address = y*(img_size.x/2) + x/2;
	  // Setting all these values seems like overkill FIXME
	  Complex c; c.y = 0.0f;
	  c.x = (img[address].x*ref[address].x+img[address].y*ref[address].y);

	  correlation_output_complex[address] = c;


}

__global__ void transformKernel_FWD(cudaTextureObject_t thisTexObj_real,
									cudaTextureObject_t thisTexObj_imag,
									cufftComplex *outputData,
									int3 dims,
									float *R,
									cufftComplex extrapVal)

{


  // calculate normalized texture coordinates
  unsigned int x = blockIdx.x*blockDim.x + threadIdx.x;
  if (x >= dims.x) { return ; }
  unsigned int y = blockIdx.y*blockDim.y + threadIdx.y;
  if (y >= dims.y) { return ; }
  unsigned int z = blockIdx.z;
  if (z >= dims.z) { return ; }

  float u,v,w,tu,tv,tw;


	u = (float)x - (float)dims.x/2;
	v = (float)y - (float)dims.y/2;
	w = (float)z - (float)dims.z/2;
	// TODO make sure this isn't the transpose of the actual transform
	tu = u*R[0] + v*R[1] + w*R[2];
	tv = u*R[3] + v*R[4] + w*R[5];
	tw = u*R[6] + v*R[7] + w*R[8];



  tu /= (float)dims.x;
  tv /= (float)dims.y;
  tw /= (float)dims.z;
  tu += 0.5f;
  tv += 0.5f;
  tw += 0.5f;

  if (tu < 0 | tv < 0 | tw < 0 | tu > 1 | tv > 1 | tw > 1)
  {
    outputData[ (z*dims.y + y) * dims.x + x ] = extrapVal;
  }
  else
  {
    outputData[ (dims.y + y) * dims.x + x ].x = tex3D<float>(thisTexObj_real, tu, tv, tw);
    outputData[ (dims.y + y) * dims.x + x ].y = tex3D<float>(thisTexObj_imag, tu, tv, tw);
  }

}



void SetToEulerRotation(float *m, float wanted_euler_phi_in_degrees, float wanted_euler_theta_in_degrees, float wanted_euler_psi_in_degrees)
{

	float			cos_phi;
	float			sin_phi;
	float			cos_theta;
	float			sin_theta;
	float			cos_psi;
	float			sin_psi;

//	cos_phi = cosf(deg_2_rad(wanted_euler_phi_in_degrees));
//	sin_phi = sinf(deg_2_rad(wanted_euler_phi_in_degrees));
//	cos_theta = cosf(deg_2_rad(wanted_euler_theta_in_degrees));
//	sin_theta = sinf(deg_2_rad(wanted_euler_theta_in_degrees));
//	cos_psi = cosf(deg_2_rad(wanted_euler_psi_in_degrees));
//	sin_psi = sinf(deg_2_rad(wanted_euler_psi_in_degrees));

	sincosf(deg_2_rad(wanted_euler_phi_in_degrees),&sin_phi,&cos_phi);
	sincosf(deg_2_rad(wanted_euler_theta_in_degrees),&sin_theta,&cos_theta);
	sincosf(deg_2_rad(wanted_euler_psi_in_degrees),&sin_psi,&cos_psi);


    // Equivalent to BH_defineMatrix SPIDER fwd (invVector)
  m[0] = cos_phi * cos_theta * cos_psi - sin_phi * sin_psi;
  m[1] = sin_phi * cos_theta * cos_psi + cos_phi * sin_psi;
  m[2] = -sin_theta * cos_psi;
  m[3] = -cos_phi * cos_theta * sin_psi - sin_phi * cos_psi;
  m[4] = -sin_phi * cos_theta * sin_psi + cos_phi * cos_psi;
  m[5] = sin_theta * sin_psi;
  m[6] = sin_theta * cos_phi;
  m[7] = sin_theta * sin_phi;
  m[8] = cos_theta;


}



void SplitRealAndImag(TGPUplan plan[], int3 ref_size, int iThread, int real_memory_allocated)
{

	// Split off the real and imaginary parts of the reference volume, padding the x-dimension to include the first row of negative frequency
	int i,j,k;
	int physical_y;
	int physical_z;
	long pixel_counter = 0;
	long hermitian_address = 0;

	printf("%d %f\n",iThread, plan[iThread].ref[0]);

	for (k = 0; k <= plan[iThread].physical_upper_bound_complex.z; k++)
		{
			physical_z = ReturnFourierLogicalCoordGivenPhysicalCoord(k, plan[iThread].physical_address_of_box_center.z, ref_size.z);

			for (j = 0; j <= plan[iThread].physical_upper_bound_complex.y; j++)
			{

				physical_y = ReturnFourierLogicalCoordGivenPhysicalCoord(j, plan[iThread].physical_address_of_box_center.y, ref_size.y);

				for (i = 0; i <= plan[iThread].physical_upper_bound_complex.x; i++)
				{

					if ( i == 0 )
					{

//						// First get the frequency from -1 and then increment the pointer
//						hermitian_address = ReturnFourier1DAddressFromLogicalCoord(plan, ref_size, iThread, -1, -physical_y, -physical_z);
//
//
//						plan[iThread].ref_real[pixel_counter] = (float)plan[iThread].ref[hermitian_address];
//						plan[iThread].ref_imag[pixel_counter] = (float)plan[iThread].ref[hermitian_address+1];

//						pixel_counter++;
					}

					plan[iThread].ref_real[pixel_counter] = (float)plan[iThread].ref[pixel_counter];
					plan[iThread].ref_imag[pixel_counter] = (float)plan[iThread].ref[pixel_counter+1];
					pixel_counter++;
				}
			}
		}
//	for (long iPixel = 0 ; iPixel < ref_real_memory_allocated / 2; iPixel++)
//	{
//		plan[iThread].d_ref3d_real[iPixel] = (float)d_ref_complex[iPixel].x;
//		plan[iThread].d_ref3d_imag[iPixel] = (float)d_ref_complex[iPixel].y;
//	}

}


void SetupTextureMemory(TGPUplan plan[], int3 ref_size, int ref_real_memory_allocated, int padding_jump_value, int iThread )
{

	 // TODO make this into a single structure that is less confusing. I.e. a complex texture object
	  plan[iThread].channelDesc = cudaCreateChannelDesc<float>();

	  checkCudaErrors(cudaMalloc3DArray(&plan[iThread].cuArray_real,
	                                    &plan[iThread].channelDesc,
	                                    make_cudaExtent(ref_size.x / 2,ref_size.y,ref_size.z)));

	  checkCudaErrors(cudaMalloc3DArray(&plan[iThread].cuArray_imag,
	                                    &plan[iThread].channelDesc,
	                                    make_cudaExtent(ref_size.x / 2,ref_size.y,ref_size.z)));

//	  plan[iThread].params3d_real =  {0};
	  plan[iThread].params3d_real.extent = make_cudaExtent(ref_size.x / 2,ref_size.y,ref_size.z);
	  plan[iThread].params3d_real.srcPtr = make_cudaPitchedPtr(plan[iThread].d_ref3d_real, ref_size.x / 2 *sizeof(cufftReal),ref_size.x / 2,ref_size.y);
	  plan[iThread].params3d_real.dstArray = plan[iThread].cuArray_real;
	  plan[iThread].params3d_real.kind = cudaMemcpyDeviceToDevice;

	  plan[iThread].params3d_imag.extent = make_cudaExtent(ref_size.x / 2,ref_size.y,ref_size.z);
	  plan[iThread].params3d_imag.srcPtr = make_cudaPitchedPtr(plan[iThread].d_ref3d_imag, ref_size.x / 2 *sizeof(cufftReal),ref_size.x / 2,ref_size.y);
	  plan[iThread].params3d_imag.dstArray = plan[iThread].cuArray_imag;
	  plan[iThread].params3d_imag.kind = cudaMemcpyDeviceToDevice;
//
	  cudaMemcpy3D(&plan[iThread].params3d_real);
	  cudaMemcpy3D(&plan[iThread].params3d_imag);


	  memset(&plan[iThread].resDesc_real, 0, sizeof(cudaResourceDesc));
	  plan[iThread].resDesc_real.resType = cudaResourceTypeArray;
	  plan[iThread].resDesc_real.res.array.array = plan[iThread].cuArray_real;

	  memset(&plan[iThread].resDesc_imag, 0, sizeof(cudaResourceDesc));
	  plan[iThread].resDesc_imag.resType = cudaResourceTypeArray;
	  plan[iThread].resDesc_imag.res.array.array = plan[iThread].cuArray_imag;

	  memset(&plan[iThread].texDesc_real,0,sizeof(cudaTextureDesc));
	  plan[iThread].texDesc_real.filterMode = cudaFilterModeLinear;
	  plan[iThread].texDesc_real.readMode = cudaReadModeElementType;
	  plan[iThread].texDesc_real.normalizedCoords = true;
	  plan[iThread].texDesc_real.addressMode[0] = cudaAddressModeWrap;
	  plan[iThread].texDesc_real.addressMode[1] = cudaAddressModeWrap;
	  plan[iThread].texDesc_real.addressMode[2] = cudaAddressModeWrap;

	  memset(&plan[iThread].texDesc_imag,0,sizeof(cudaTextureDesc));
	  plan[iThread].texDesc_imag.filterMode = cudaFilterModeLinear;
	  plan[iThread].texDesc_imag.readMode = cudaReadModeElementType;
	  plan[iThread].texDesc_imag.normalizedCoords = true;
	  plan[iThread].texDesc_imag.addressMode[0] = cudaAddressModeWrap;
	  plan[iThread].texDesc_imag.addressMode[1] = cudaAddressModeWrap;
	  plan[iThread].texDesc_imag.addressMode[2] = cudaAddressModeWrap;
//
	  checkCudaErrors(cudaCreateTextureObject(&plan[iThread].thisTexObj_real,&plan[iThread].resDesc_real,&plan[iThread].texDesc_real,NULL));
	  checkCudaErrors(cudaCreateTextureObject(&plan[iThread].thisTexObj_imag,&plan[iThread].resDesc_imag,&plan[iThread].texDesc_imag,NULL));

}


void UpdateLoopingAndAddressing(TGPUplan plan[], const int3 ref_size, const int iGPU)
{

	// Taken directly from the cpu Image class

	// FIXME these should be member variables in a gpuImage class
	plan[iGPU].physical_upper_bound_complex = make_int3(ref_size.x / 2, ref_size.y - 1 , ref_size.z - 1);

	plan[iGPU].physical_index_of_first_negative_frequency = make_int3(0,0,0);
	plan[iGPU].logical_upper_bound_complex  = make_int3(0,0,0);
	plan[iGPU].logical_lower_bound_complex  = make_int3(0,0,0);
	plan[iGPU].logical_upper_bound_real = make_int3(0,0,0);
	plan[iGPU].logical_lower_bound_real  = make_int3(0,0,0);
	plan[iGPU].physical_address_of_box_center = make_int3(ref_size.x/2, ref_size.y/2, ref_size.z/2);

	//physical_index_of_first_negative_frequency_x = logical_x_dimension / 2 + 1;
	if (IsEven(ref_size.y) == true)
	{
		plan[iGPU].physical_index_of_first_negative_frequency.y = ref_size.y / 2;
	}
	else
	{
		plan[iGPU].physical_index_of_first_negative_frequency.y = ref_size.y / 2 + 1;
	}

	if (IsEven(ref_size.z) == true)
	{
		plan[iGPU].physical_index_of_first_negative_frequency.z = ref_size.z / 2;
	}
	else
	{
		plan[iGPU].physical_index_of_first_negative_frequency.z = ref_size.z / 2 + 1;
	}


    // Update the Fourier voxel size
	plan[iGPU].fourier_voxel_size = make_float3(1.0f / float(ref_size.x),1.0f / float(ref_size.y),1.0f / float(ref_size.z));

	// Logical bounds
	if (IsEven(ref_size.x) == true)
	{
		plan[iGPU].logical_lower_bound_complex.x = -ref_size.x / 2;
		plan[iGPU].logical_upper_bound_complex.x =  ref_size.x / 2;
		plan[iGPU].logical_lower_bound_real.x    = -ref_size.x / 2;
		plan[iGPU].logical_upper_bound_real.x    =  ref_size.x / 2 - 1;
	}
	else
	{
		plan[iGPU].logical_lower_bound_complex.x = -(ref_size.x-1) / 2;
		plan[iGPU].logical_upper_bound_complex.x =  (ref_size.x-1) / 2;
		plan[iGPU].logical_lower_bound_real.x    = -(ref_size.x-1) / 2;
		plan[iGPU].logical_upper_bound_real.x    =  (ref_size.x-1) / 2;
	}


	if (IsEven(ref_size.y) == true)
	{
		plan[iGPU].logical_lower_bound_complex.y = -ref_size.y / 2;
		plan[iGPU].logical_upper_bound_complex.y =  ref_size.y / 2 - 1;
	    plan[iGPU].logical_lower_bound_real.y    = -ref_size.y / 2;
	    plan[iGPU].logical_upper_bound_real.y    =  ref_size.y / 2 - 1;
	}
	else
	{
		plan[iGPU].logical_lower_bound_complex.y = -(ref_size.y-1) / 2;
		plan[iGPU].logical_upper_bound_complex.y =  (ref_size.y-1) / 2;
	    plan[iGPU].logical_lower_bound_real.y    = -(ref_size.y-1) / 2;
	    plan[iGPU].logical_upper_bound_real.y   =  (ref_size.y-1) / 2;
	}

	if (IsEven(ref_size.z) == true)
	{
		plan[iGPU].logical_lower_bound_complex.z = -ref_size.z / 2;
		plan[iGPU].logical_upper_bound_complex.z =  ref_size.z / 2 - 1;
		plan[iGPU].logical_lower_bound_real.z    = -ref_size.z / 2;
		plan[iGPU].logical_upper_bound_real.z    =  ref_size.z / 2 - 1;

	}
	else
	{
		plan[iGPU].logical_lower_bound_complex.z = -(ref_size.z - 1) / 2;
		plan[iGPU].logical_upper_bound_complex.z =  (ref_size.z - 1) / 2;
		plan[iGPU].logical_lower_bound_real.z    = -(ref_size.z - 1) / 2;
		plan[iGPU].logical_upper_bound_real.z    =  (ref_size.z - 1) / 2;
	}
}

inline int ReturnFourierLogicalCoordGivenPhysicalCoord(int physical_index, int physical_address_of_box_center, int logical_dimension )
{

    if (physical_index > physical_address_of_box_center)
    {
    	 return physical_index - logical_dimension;
    }
    else return physical_index;
}


inline long ReturnFourier1DAddressFromLogicalCoord(TGPUplan plan[], int3 ref_size, int iThread, int wanted_x, int wanted_y, int wanted_z)
{

	int physical_x_address;
	int physical_y_address;
	int physical_z_address;

	if (wanted_x >= 0)
	{
		physical_x_address = wanted_x;

		if (wanted_y >= 0)
		{
			physical_y_address = wanted_y;
		}
		else
		{
			physical_y_address = ref_size.y + wanted_y;
		}

		if (wanted_z >= 0)
		{
			physical_z_address = wanted_z;
		}
		else
		{
			physical_z_address = ref_size.z + wanted_z;
		}
	}
	else
	{
		physical_x_address = -wanted_x;

		if (wanted_y > 0)
		{
			physical_y_address = ref_size.y - wanted_y;
		}
		else
		{
			physical_y_address = -wanted_y;
		}

		if (wanted_z > 0)
		{
			physical_z_address = ref_size.z - wanted_z;
		}
		else
		{
			physical_z_address = -wanted_z;
		}
	}

	return ((long(plan[iThread].physical_upper_bound_complex.x + 1) * long(plan[iThread].physical_upper_bound_complex.y+ 1)) * physical_z_address) + (long(plan[iThread].physical_upper_bound_complex.x + 1) * physical_y_address) + physical_x_address;

//	return ReturnFourier1DAddressFromPhysicalCoord(physical_x_address, physical_y_address, physical_z_address);
};


