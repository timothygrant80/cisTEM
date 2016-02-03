/*  \brief  ResolutionStatistics class */

class ResolutionStatistics {

public:

	Curve		FSC;
	Curve		part_FSC;
	Curve		part_SSNR;
	Curve		rec_SSNR;

	float		pixel_size;
	int			number_of_bins;
	int			number_of_bins_extended;	// Extend table to include corners in 3D Fourier space

	ResolutionStatistics();
	ResolutionStatistics(float wanted_pixel_size, int wanted_number_of_bins = 0);
//	~ResolutionStatistics();

	void Init(float wanted_pixel_size, int wanted_number_of_bins = 0);
	void CalculateFSC(Image &reconstructed_volume_1, Image &reconstructed_volume_2);
	void CalculateParticleFSCandSSNR(float mask_volume_in_voxels, float molecular_mass_in_kDa, float pixel_size);
	void CalculateParticleSSNR(Image &image_reconstruction, float *ctf_reconstruction);
};
