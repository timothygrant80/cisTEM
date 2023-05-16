/*  \brief  Spectrum Image class (derived from the Image class)

	for processing images that contain power spectra

*/

class SpectrumImage : public Image {
  public:
    float FindRotationalAlignmentBetweenTwoStacksOfImages(Image* other_image, int number_of_images, float search_half_range, float search_step_size, float minimum_radius, float maximum_radius);
    void  GeneratePowerspectrum(CTF ctf_to_apply);
    void  OverlayCTF(CTF* ctf, Image* number_of_extrema, Image* ctf_values, int number_of_bins_in_1d_spectra, double spatial_frequency[], double rotational_average_astig[], float number_of_extrema_profile[], float ctf_values_profile[], Curve* equiphase_average_pre_max, Curve* equiphase_average_post_max, bool fit_nodes = false);
    void  ComputeRotationalAverageOfPowerSpectrum(CTF* ctf, Image* number_of_extrema, Image* ctf_values, int number_of_bins, double spatial_frequency[], double average[], double average_fit[], double average_renormalized[], float number_of_extrema_profile[], float ctf_values_profile[]);
    void  ComputeEquiPhaseAverageOfPowerSpectrum(CTF* ctf, Curve* epa_pre_max, Curve* epa_post_max);
    void  RescaleSpectrumAndRotationalAverage(Image* number_of_extrema, Image* ctf_values, int number_of_bins, double spatial_frequency[], double average[], double average_fit[], float number_of_extrema_profile[], float ctf_values_profile[], int last_bin_without_aliasing, int last_bin_with_good_fit);
    float DilatePowerspectrumToNewPixelSize(bool resample_if_pixel_too_small, float pixel_size_of_input_image, float target_pixel_size_after_resampling,
                                            int box_size, Image* resampled_power_spectrum, bool do_resampling = true, float stretch_factor = 1.0f);
};