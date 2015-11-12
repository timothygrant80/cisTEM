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
	void CalculateDoseFilterAs1DArray(Image *ref_image, float *filter_array, float dose_start, float dose_finish);
};

inline float ElectronDose::ReturnCriticalDose(float spatial_frequency)
{
	return (critical_dose_a * pow(spatial_frequency, critical_dose_b) + critical_dose_c) * voltage_scaling_factor;
}


inline float ElectronDose::ReturnDoseFilter(float dose_at_end_of_frame, float critical_dose)
{
	return exp((-0.5 * dose_at_end_of_frame) / critical_dose);
}

