class ElectronDose {
  private:

  public:
    float acceleration_voltage;

    float critical_dose_a;
    float critical_dose_b;
    float critical_dose_c;
    float reduced_critical_dose_b;

    float voltage_scaling_factor;

    float pixel_size;

    ElectronDose( );
    ElectronDose(float wanted_acceleration_voltage, float wanted_pixel_size);

    void  Init(float wanted_acceleration_voltage, float wanted_pixel_size);
    float ReturnCriticalDose(float spatial_frequency);
    float ReturnDoseFilter(float dose_at_end_of_frame, float critical_dose);
    float ReturnCummulativeDoseFilter(float dose_at_start_of_exposure, float dose_at_end_of_exposure, float critical_dose);
    void  CalculateCummulativeDoseFilterAs1DArray(Image* ref_image, float* filter_array, float dose_start, float dose_finish);

    void CalculateDoseFilterAs1DArray(Image* ref_image, float* filter_array, float dose_start, float dose_finish);
};

inline float ElectronDose::ReturnCriticalDose(float spatial_frequency) {
    return (critical_dose_a * powf(spatial_frequency, reduced_critical_dose_b) + critical_dose_c) * voltage_scaling_factor;
}

inline float ElectronDose::ReturnDoseFilter(float dose_at_end_of_frame, float critical_dose) {
    return expf((-0.5f * dose_at_end_of_frame) / critical_dose);
}

inline float ElectronDose::ReturnCummulativeDoseFilter(float dose_at_start_of_exposure, float dose_at_end_of_exposure, float critical_dose) {
    // The integrated exposure. Included in particular for the matched filter.
    // Calculated on Wolfram Alpha = integrate exp[ -0.5 * (x/a) ] from x=t1 to x=t2
    // We need to add the division by the total dose to normalize the filter
    return 2.0f * critical_dose * (expf(-0.5f * dose_at_start_of_exposure / critical_dose) - expf(-0.5f * dose_at_end_of_exposure / critical_dose)) / dose_at_end_of_exposure;
}
