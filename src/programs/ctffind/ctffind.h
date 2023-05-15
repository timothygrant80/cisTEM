#ifndef PROGRAMS_CTFFIND_CTFFIND_H_
#define PROGRAMS_CTFFIND_CTFFIND_H_

#define use_epa_rather_than_zero_counting

class CTFTilt {
    int   refine_mode;
    int   micrograph_subregion_dimension;
    int   micrograph_binned_dimension_for_ctf;
    int   micrograph_square_dimension;
    int   sub_section_dimension_x;
    int   sub_section_dimension_y;
    int   box_size;
    int   n_sections;
    int   n_steps;
    int   box_convolution;
    int   input_image_x_dimension;
    int   input_image_y_dimension;
    int   image_buffer_counter;
    float tilt_binning_factor;
    float original_pixel_size;
    float ctf_fit_pixel_size;
    float tilt_fit_pixel_size;
    float low_res_limit;
    float acceleration_voltage_in_kV;
    float spherical_aberration_in_mm;
    float amplitude_contrast;
    float additional_phase_shift_in_radians;
    float high_res_limit_ctf_fit;
    float high_res_limit_tilt_fit;
    float minimum_defocus;
    float maximum_defocus;

    Image  input_image;
    Image  input_image_binned;
    Image  average_spectrum;
    Image  power_spectrum_binned_image;
    Image  power_spectrum_sub_section;
    Image  resampled_power_spectrum;
    Image  resampled_power_spectrum_binned_image;
    Image  average_power_spectrum;
    Image  ctf_transform;
    Image  ctf_image;
    Image  sub_section;
    Image* resampled_power_spectra;
    Image* input_image_buffer;

    bool rough_defocus_determined;
    bool defocus_astigmatism_determined;
    bool power_spectra_calculated;

  public:
    CTFTilt(ImageFile& wanted_input_file, float wanted_high_res_limit_ctf_fit, float wanted_high_res_limit_tilt_fit, float wanted_minimum_defocus, float wanted_maximum_defocus,
            float wanted_pixel_size, float wanted_acceleration_voltage_in_kV, float wanted_spherical_aberration_in_mm, float wanted_amplitude_contrast, float wanted_additional_phase_shift_in_radians);
    ~CTFTilt( );
    void   CalculatePowerSpectra(bool subtract_average = false);
    void   UpdateInputImage(Image* wanted_input_image);
    float  FindRoughDefocus( );
    float  FindDefocusAstigmatism( );
    float  SearchTiltAxisAndAngle( );
    float  RefineTiltAxisAndAngle( );
    float  CalculateTiltCorrectedSpectra(bool resample_if_pixel_too_small, float pixel_size_of_input_image, float target_pixel_size_after_resampling,
                                         int box_size, Image* resampled_spectrum);
    double ScoreValues(double[]);

    float defocus_1;
    float defocus_2;
    float astigmatic_angle;
    float best_tilt_axis;
    float best_tilt_angle;
};

double SampleTiltScoreFunctionForSimplex(void* pt2Object, double values[]);

class ImageCTFComparison {
  public:
    ImageCTFComparison(int wanted_number_of_images, CTF wanted_ctf, float wanted_pixel_size, bool should_find_phase_shift, bool wanted_astigmatism_is_known, float wanted_known_astigmatism, float wanted_known_astigmatism_angle, bool should_fit_defocus_sweep);
    ~ImageCTFComparison( );
    void  SetImage(int wanted_image_number, Image* new_image);
    void  SetCTF(CTF new_ctf);
    CTF   ReturnCTF( );
    bool  AstigmatismIsKnown( );
    float ReturnKnownAstigmatism( );
    float ReturnKnownAstigmatismAngle( );
    bool  FindPhaseShift( );
    void  SetupQuickCorrelation( );

    int    number_of_images;
    Image* img; // Usually an amplitude spectrum, or an array of amplitude spectra
    int    number_to_correlate;
    double norm_image;
    double image_mean;
    float* azimuths;
    float* spatial_frequency_squared;
    int*   addresses;

  private:
    CTF   ctf;
    float pixel_size;
    bool  find_phase_shift;
    bool  astigmatism_is_known;
    float known_astigmatism;
    float known_astigmatism_angle;
    bool  fit_defocus_sweep;
};

class CurveCTFComparison {
  public:
    float* curve; // Usually the 1D rotational average of the amplitude spectrum of an image
    int    number_of_bins;
    float  reciprocal_pixel_size; // In reciprocal pixels
    CTF    ctf;
    bool   find_phase_shift;
};

float CtffindObjectiveFunction(void* scoring_parameters, float array_of_values[]);
float CtffindCurveObjectiveFunction(void* scoring_parameters, float array_of_values[]);

#endif /* PROGRAMS_CTFFIND_CTFFIND_H_ */
