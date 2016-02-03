/*  \brief  ReconstructedVolume class */

class ReconstructedVolume {

public:

	float						pixel_size;
	float						mask_volume_in_voxels;
	float						molecular_mass_in_kDa;
//	SymmetryOperator			symmetry;
	Image						density_map;
	ResolutionStatistics		statistics;

	bool						has_been_initialized;
	bool						has_masked_applied;
	bool						was_corrected;
	bool						has_statistics;
	bool						has_been_filtered;

	ReconstructedVolume();
//	~ReconstructedVolume();

	void Init(Reconstruct3d &image_reconstruction, float wanted_pixel_size);
	void Calculate3DSimple(Reconstruct3d &reconstruction);
	void Calculate3DOptimal(Reconstruct3d &reconstruction, float pssnr_correction_factor = 1.0);
	float Correct3D(float mask_radius = 0.0);
	void CosineMask(float wanted_mask_radius, float wanted_mask_edge);
	void OptimalFilter();
	void PrintStatistics();
};
