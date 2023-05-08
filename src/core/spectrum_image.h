/*  \brief  Spectrum Image class (derived from the Image class)

	for processing images that contain power spectra

*/

class SpectrumImage : public Image {
  public:
    float FindRotationalAlignmentBetweenTwoStacksOfImages(Image* other_image, int number_of_images, float search_half_range, float search_step_size, float minimum_radius, float maximum_radius);
    void  GeneratePowerspectrum(CTF ctf_to_apply);
};