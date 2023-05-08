/*  \brief  Spectrum Image class (derived from the Image class)

	for processing images that contain power spectra

*/

class SpectrumImage : public Image {
  public:
    float FindRotationalAlignmentBetweenTwoStacksOfImages(Image* other_image, int number_of_images, float search_half_range, float search_step_size, float minimum_radius, float maximum_radius);
    void  GeneratePowerspectrum(CTF ctf_to_apply);
    void  OverlayCTF(CTF* ctf, Image* number_of_extrema, Image* ctf_values, int number_of_bins_in_1d_spectra, double spatial_frequency[], double rotational_average_astig[], float number_of_extrema_profile[], float ctf_values_profile[], Curve* equiphase_average_pre_max, Curve* equiphase_average_post_max);
    void  ComputeRotationalAverageOfPowerSpectrum(CTF* ctf, Image* number_of_extrema, Image* ctf_values, int number_of_bins, double spatial_frequency[], double average[], double average_fit[], double average_renormalized[], float number_of_extrema_profile[], float ctf_values_profile[]);
};