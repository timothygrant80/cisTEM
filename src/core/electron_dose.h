class ElectronDose {
private:

public:

	float acceleration_voltage;

	float critical_dose_a;
	float critical_dose_b;
	float critical_dose_c;

	float voltage_scaling_factor;

	float pixel_size;

	ElectronDose();
	ElectronDose(float wanted_acceleration_voltage, float wanted_pixel_size);


	void Init(float wanted_acceleration_voltage, float wanted_pixel_size);
	float ReturnCriticalDose(float spatial_frequency);
	float ReturnDoseFilter(float dose_at_end_of_frame, float critical_dose);
	float ReturnCummulativeDoseFilter(float dose_at_end_of_exosure, float critical_dose);
	void CalculateCummulativeDoseFilterAs1DArray(Image *ref_image, float *filter_array, float dose_finish);


	void CalculateDoseFilterAs1DArray(Image *ref_image, float *filter_array, float dose_start, float dose_finish);
};

inline float ElectronDose::ReturnCriticalDose(float spatial_frequency)
{
	return (critical_dose_a * powf(spatial_frequency, critical_dose_b) + critical_dose_c) * voltage_scaling_factor;
}


inline float ElectronDose::ReturnDoseFilter(float dose_at_end_of_frame, float critical_dose)
{
	return expf((-0.5 * dose_at_end_of_frame) / critical_dose);
}

inline float ElectronDose::ReturnCummulativeDoseFilter(float dose_at_end_of_exosure, float critical_dose)
{
	// The integrated exposure. Included in particular for the matched filter.
	// Calculated on Wolfram Alpha = integrate exp[ -0.5 * (x/a) ] from x=0 to x=t
	return critical_dose * (2.0f - 2.0f * exp((-0.5 * dose_at_end_of_exosure) / critical_dose)) / dose_at_end_of_exosure;
}

