/*  \brief  ReconstructedVolume class */

class Reconstruct3D;

class ReconstructedVolume {

public:

	float						pixel_size;
	float						mask_volume_in_voxels;
	float						molecular_mass_in_kDa;
	SymmetryMatrix				symmetry_matrices;
	Image						density_map;
	ResolutionStatistics		statistics;

	bool						has_been_initialized;
	bool						has_masked_applied;
	bool						was_corrected;
	bool						has_statistics;
	bool						has_been_filtered;

	ReconstructedVolume();
//	~ReconstructedVolume();

	void InitWithReconstruct3D(Reconstruct3D &image_reconstruction, float wanted_pixel_size);
	void InitWithDimensions(int wanted_logical_x_dimension, int wanted_logical_y_dimension, int wanted_logical_z_dimension, float wanted_pixel_size, wxString = "C1");
	void CalculateProjection(Image &projection, Image &CTF, AnglesAndShifts &angles_and_shifts_of_projection, float mask_radius = 0.0, float mask_falloff = 0.0, float resolution_limit = 1.0, bool swap_quadrants = true);
	void Calculate3DSimple(Reconstruct3D &reconstruction);
	void Calculate3DOptimal(Reconstruct3D &reconstruction, float pssnr_correction_factor = 1.0);
	float Correct3D(float mask_radius = 0.0);
	void CosineRingMask(float wanted_inner_mask_radius, float wanted_outer_mask_radius, float wanted_mask_edge);
	void CosineMask(float wanted_mask_radius, float wanted_mask_edge);
	void OptimalFilter();
	void PrintStatistics();
	void WriteStatisticsToFile(wxString output_file);
	void ReadStatisticsFromFile(wxString input_file);
};
