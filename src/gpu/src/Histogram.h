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

	Npp8u* histogram_buffer;		bool is_allocated_histogram_buffer;
	Npp32s* histogram;				bool is_allocated_histogram; // histogram_n_bins in size;
	Npp32f* histogram_bin_values;	bool is_allocated_histogram_bin_values; // histogram_n_bins + 1 in size
	NppiSize vector_ROI;

	int histogram_n_bins; //
	float histogram_min;
	float histogram_max;
	float histogram_step;

	void SetInitialValues();
	void Init(int histogram_n_bins, float histogram_min, float histogram_step);
	void BufferInit(NppiSize npp_ROI);
	void AddToHistogram(GpuImage &input_image);
	void CopyToHostAndAdd(long* array_to_add_to);

private:

	Npp32s* cummulative_histogram;




};

#endif /* HISTOGRAM_H_ */
