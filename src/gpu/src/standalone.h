/*
 * standalone.h
 *
 *  Created on: Jul 24, 2019
 *      Author: himesb
 */

////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/*
 *  This first bit is from the sample file simpleMultiGPU.h
 */
//#ifndef SIMPLEMULTIGPU_H
//#define SIMPLEMULTIGPU_H


typedef struct
{


    //Stream for asynchronous command execution
    cudaStream_t stream;

    cufftHandle cuda_plan_forward;
	cufftHandle cuda_plan_inverse_ref;
	cufftHandle cuda_plan_inverse_img;

	int3 physical_upper_bound_complex;
	int3 logical_upper_bound_complex;
	int3 logical_lower_bound_complex;
	int3 logical_lower_bound_real;
	int3 logical_upper_bound_real;
	int3 physical_address_of_box_center;

	int3 physical_index_of_first_negative_frequency;
	float3 fourier_voxel_size;

	float *ref;
	float *ref_real; float *ref_imag;
	float *ref_3d;
	float *img;
	float *correlation_output;
	float *mip;

	float m[3][3];

	cufftReal *d_ref; 			cufftComplex *d_ref_complex;
	cufftReal *d_ref3d_imag;
	cufftReal *d_ref3d_real;


	cufftReal *d_prj; 					cufftComplex *d_prj_complex;
	cufftReal *d_ref_padded; 			cufftComplex *d_ref_padded_complex;
	cufftReal *d_img; 					cufftComplex *d_img_complex;
	cufftReal *d_correlation_output;	cufftComplex *d_correlation_output_complex;

	cufftReal *d_mip;
	cufftReal *d_sum;
	cufftReal *d_sum_of_squares;


	bool first_loop;

	int gpu_id;


/*	texture<float, 3, cudaReadModeElementType> tex;*/
	cudaTextureObject_t thisTexObj_real;
	cudaTextureObject_t thisTexObj_imag;

	cudaArray *cuArray_real;
	cudaArray *cuArray_imag;

	cudaChannelFormatDesc channelDesc;

	cudaMemcpy3DParms params3d_real;
	cudaMemcpy3DParms params3d_imag;


	struct cudaResourceDesc resDesc_real;
	struct cudaResourceDesc resDesc_imag;

	struct cudaTextureDesc texDesc_real;
	struct cudaTextureDesc texDesc_imag;




} TGPUplan;

typedef struct  {

	int ref_x;
	int ref_y;
	int img_x;
	int img_y;
	int ref_real_memory_allocated;
	int img_real_memory_allocated;

} pDIMS;

//extern "C"
//void launch_reduceKernel(float *d_Result, float *d_Input, int N, int BLOCK_N, int THREAD_N, cudaStream_t &s);

//#endif

////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////


