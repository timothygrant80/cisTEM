class CTF {

private:
	float spherical_aberration;
	float wavelength;
	float amplitude_contrast;
	float defocus_1;
	float defocus_2;
	float defocus_half_range;
	float astigmatism_azimuth;
	float additional_phase_shift;
	// Fitting parameters
	float lowest_frequency_for_fitting;
	float highest_frequency_for_fitting;
	float astigmatism_tolerance;
	// Precomputed terms to make evaluations faster
	float precomputed_amplitude_contrast_term;
	float squared_wavelength;

public:

	// Constructors, destructors
	CTF();
	CTF(		float acceleration_voltage,
				float spherical_aberration,
				float amplitude_contrast,
				float defocus_1,
				float defocus_2,
				float astigmatism_azimuth,
				float lowest_frequency_for_fitting,
				float highest_frequency_for_fitting,
				float astigmatism_tolerance,
				float pixel_size,
				float additional_phase_shift);
	~CTF();

	void Init(	float acceleration_voltage,
				float spherical_aberration,
				float amplitude_contrast,
				float defocus_1,
				float defocus_2,
				float astigmatism_azimuth,
				float lowest_frequency_for_fitting,
				float highest_frequency_for_fitting,
				float astigmatism_tolerance,
				float pixel_size,
				float additional_phase_shift);

	//
	float Evaluate(float squared_spatial_frequency, float azimuth);
	float PhaseShiftGivenSquaredSpatialFrequencyAndAzimuth(float squared_spatial_frequency, float azimuth);
	float DefocusGivenAzimuth(float azimuth);
	float WavelengthGivenAccelerationVoltage(float acceleration_voltage);
};
