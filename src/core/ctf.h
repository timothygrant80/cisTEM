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
	float cubed_wavelength;

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

	CTF(		float wanted_acceleration_voltage, // keV
				float wanted_spherical_aberration, // mm
				float wanted_amplitude_contrast,
				float wanted_defocus_1_in_angstroms, // A
				float wanted_defocus_2_in_angstroms, //A
				float wanted_astigmatism_azimuth, // degrees
				float pixel_size, // A
				float wanted_additional_phase_shift_in_radians );// rad

	~CTF();

	void Init(	float wanted_acceleration_voltage_in_kV, // keV
				float wanted_spherical_aberration_in_mm, // mm
				float wanted_amplitude_contrast,
				float wanted_defocus_1_in_angstroms, // A
				float wanted_defocus_2_in_angstroms, //A
				float wanted_astigmatism_azimuth_in_degrees, // degrees
				float wanted_lowest_frequency_for_fitting_in_reciprocal_angstroms, // 1/A
				float wanted_highest_frequency_for_fitting_in_reciprocal_angstroms, // 1/A
				float wanted_astigmatism_tolerance_in_angstroms, // A. Set to negative to indicate no restraint on astigmatism.
				float pixel_size_in_angstroms, // A
				float wanted_additional_phase_shift_in_radians); //rad

	void Init(	float wanted_acceleration_voltage_in_kV, // keV
				float wanted_spherical_aberration_in_mm, // mm
				float wanted_amplitude_contrast,
				float wanted_defocus_1_in_angstroms, // A
				float wanted_defocus_2_in_angstroms, //A
				float wanted_astigmatism_azimuth_in_degrees, // degrees
				float pixel_size_in_angstroms, // A
				float wanted_additional_phase_shift_in_radians); //rad



	void SetDefocus(float wanted_defocus_1_pixels, float wanted_defocus_2_pixels, float wanted_astigmatism_angle_radians);
	void SetAdditionalPhaseShift(float wanted_additional_phase_shift_radians);
	//
	std::complex<float> EvaluateComplex(float squared_spatial_frequency, float azimuth);
	float Evaluate(float squared_spatial_frequency, float azimuth);
	float PhaseShiftGivenSquaredSpatialFrequencyAndAzimuth(float squared_spatial_frequency, float azimuth);
	float DefocusGivenAzimuth(float azimuth);
	float WavelengthGivenAccelerationVoltage(float acceleration_voltage);
	inline float GetLowestFrequencyForFitting() { return lowest_frequency_for_fitting; };
	inline float GetHighestFrequencyForFitting() { return highest_frequency_for_fitting; };
	inline float GetAstigmatismTolerance() { return astigmatism_tolerance; };
	inline float GetAstigmatism(){ return defocus_1 - defocus_2; };
	bool IsAlmostEqualTo(CTF *wanted_ctf, float delta_defocus = 100.0);
	void EnforceConvention();
	inline float GetDefocus1() { return defocus_1; };
	inline float GetDefocus2() { return defocus_2; };
	inline float GetAstigmatismAzimuth() { return astigmatism_azimuth; };
	inline float GetAdditionalPhaseShift() { return additional_phase_shift; };
	inline float GetWavelength() { return wavelength; };
	int ReturnNumberOfExtremaBeforeSquaredSpatialFrequency(float squared_spatial_frequency, float azimuth);
	float ReturnSquaredSpatialFrequencyGivenPhaseShiftAndAzimuth(float phase_shift, float azimuth);
	float ReturnSquaredSpatialFrequencyOfAZero(int which_zero, float azimuth);
	float ReturnSquaredSpatialFrequencyOfPhaseShiftExtremum(float azimuth);
};
