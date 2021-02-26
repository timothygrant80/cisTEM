/*
 * Histogram.h
 *
 *  Created on: Aug 29, 2019
 *      Author: himesb
 */

#ifndef HISTOGRAM_H_
#define HISTOGRAM_H_

class Histogram {

public:


	Histogram();
	Histogram(int histogram_n_bins, float histogram_min, float histogram_step);
	virtual ~Histogram();

	dim3 threadsPerBlock_img;
	dim3 gridDims_img;

	dim3 threadsPerBlock_accum_array;
	dim3 gridDims_accum_array;

//	float* histogram;		bool is_allocated_histogram; // histogram_n_bins in size;
	float* histogram;		bool is_allocated_histogram; // histogram_n_bins in size;

	size_t size_of_temp_hist;
	float* cummulative_histogram;

	int histogram_n_bins; //
//	float histogram_min;
//	float histogram_max;
//	float histogram_step;
	__half histogram_min;
	__half histogram_max;
	__half histogram_step;

	int max_padding;


	void SetInitialValues();
	void Init(int histogram_n_bins, float histogram_min, float histogram_step);
	void BufferInit(NppiSize npp_ROI);
	void AddToHistogram(GpuImage &input_image);
	void Accumulate(GpuImage &input_image);

	void CopyToHostAndAdd(long* array_to_add_to);

private:





};

#endif /* HISTOGRAM_H_ */
