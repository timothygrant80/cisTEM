#ifndef SRC_PROGRAMS_CTFFIND_CTFFIND_H_
#define SRC_PROGRAMS_CTFFIND_CTFFIND_H_

#define use_epa_rather_than_zero_counting

double SampleTiltScoreFunctionForSimplex(void* pt2Object, double values[]);
double SampleTiltScoreFunctionForSimplexTiltAxis(void* pt2Object, double values[]);
//TODO: gives this function a good name that describes what it actually does (it actually rescales the power spectrum!)
float PixelSizeForFitting(bool resample_if_pixel_too_small, float pixel_size_of_input_image, float target_pixel_size_after_resampling,
                          int box_size, Image* current_power_spectrum, Image* resampled_power_spectrum, bool do_resampling = true, float stretch_factor = 1.0f);

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

    bool        debug;
    wxJSONValue debug_json_output;
    std::string debug_json_output_filename;

  public:
    CTFTilt(ImageFile& wanted_input_file, float wanted_high_res_limit_ctf_fit, float wanted_high_res_limit_tilt_fit, float wanted_minimum_defocus, float wanted_maximum_defocus,
            float wanted_pixel_size, float wanted_acceleration_voltage_in_kV, float wanted_spherical_aberration_in_mm, float wanted_amplitude_contrast, float wanted_additional_phase_shift_in_radians,
            bool wanted_debug = false, std::string wanted_debug_json_output_filename = "");
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
    double ScoreValuesFixedDefocus(double[]);

    float defocus_1;
    float defocus_2;
    float astigmatic_angle;
    float best_tilt_axis;
    float best_tilt_angle;
};

class ImageCTFComparison {
  public:
    ImageCTFComparison(int wanted_number_of_images, CTF wanted_ctf, float wanted_pixel_size, bool should_find_phase_shift, bool wanted_astigmatism_is_known, float wanted_known_astigmatism, float wanted_known_astigmatism_angle, bool should_fit_defocus_sweep);
    ~ImageCTFComparison( );
    void  SetImage(int wanted_image_number, Image* new_image);
    void  SetCTF(CTF new_ctf);
    void  SetFitWithThicknessNodes(bool wanted_fit_with_thickness_nodes);
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
    bool   fit_with_thickness_nodes;
    bool   fit_nodes_downweight_nodes = false;
    bool   fit_nodes_rounded_square   = false;

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
    bool   find_thickness_nodes     = false;
    bool   fit_nodes_rounded_square = false;
};

float FindRotationalAlignmentBetweenTwoStacksOfImages(Image* self, Image* other_image, int number_of_images, float search_half_range, float search_step_size, float minimum_radius, float maximum_radius);
void  ComputeImagesWithNumberOfExtremaAndCTFValues(CTF* ctf, Image* number_of_extrema, Image* ctf_values);
int   ReturnSpectrumBinNumber(int number_of_bins, float number_of_extrema_profile[], Image* number_of_extrema, long address, Image* ctf_values, float ctf_values_profile[]);
void  ComputeRotationalAverageOfPowerSpectrum(Image* spectrum, CTF* ctf, Image* number_of_extrema, Image* ctf_values, int number_of_bins, double spatial_frequency[], double average[], double average_fit[], double average_renormalized[], float number_of_extrema_profile[], float ctf_values_profile[]);
void  ComputeEquiPhaseAverageOfPowerSpectrum(Image* spectrum, CTF* ctf, Curve* epa_pre_max, Curve* epa_post_max);
void  OverlayCTF(Image* spectrum, CTF* ctf, Image* number_of_extrema, Image* ctf_values, int number_of_bins_in_1d_spectra, double spatial_frequency[], double rotational_average_astig[], float number_of_extrema_profile[], float ctf_values_profile[], Curve* equiphase_average_pre_max, Curve* equiphase_average_post_max, bool fit_nodes = false);
void  ComputeFRCBetween1DSpectrumAndFit(int number_of_bins, double average[], double fit[], float number_of_extrema_profile[], double frc[], double frc_sigma[], int first_fit_bin);
void  RescaleSpectrumAndRotationalAverage(Image* spectrum, Image* number_of_extrema, Image* ctf_values, int number_of_bins, double spatial_frequency[], double average[], double average_fit[], float number_of_extrema_profile[], float ctf_values_profile[], int last_bin_without_aliasing, int last_bin_with_good_fit);
void  Renormalize1DSpectrumForFRC(int number_of_bins, double average[], double fit[], float number_of_extrema_profile[]);
float ReturnAzimuthToUseFor1DPlots(CTF* ctf);
float CtffindObjectiveFunction(void* scoring_parameters, float array_of_values[]);
float CtffindCurveObjectiveFunction(void* scoring_parameters, float array_of_values[]);

struct CTFNodeFitInput {
    CTF*                current_ctf;
    int                 last_bin_with_good_fit;
    int                 number_of_bins_in_1d_spectra;
    float               pixel_size_for_fitting;
    double*             spatial_frequency;
    double*             rotational_average_astig;
    double*             rotational_average_astig_fit;
    Curve&              equiphase_average_pre_max;
    Curve&              equiphase_average_post_max;
    CurveCTFComparison* comparison_object_1D;
    ImageCTFComparison* comparison_object_2D;
    bool                bruteforce_1D;
    bool                refine_2D;
    float               low_resolution_limit;
    float               high_resolution_limit;
    double*             fit_frc;
    double*             fit_frc_sigma;
    Image*              average_spectrum;
    bool                debug;
    std::string         debug_filename;
    bool                use_rounded_square;
    bool                downweight_nodes;
};

struct CTFNodeFitOuput {
    int last_bin_with_good_fit;
};

CTFNodeFitOuput fit_thickness_nodes(CTFNodeFitInput* input);

#endif
