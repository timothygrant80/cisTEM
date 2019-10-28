/*  \brief  ReconstructedVolume class */

class Reconstruct3D;
class NumericTextFile;

class ReconstructedVolume {

public:

	float						pixel_size;
	float						mask_volume_in_voxels;
	float						molecular_mass_in_kDa;
	float						mask_radius;
	wxString 					symmetry_symbol;
	SymmetryMatrix				symmetry_matrices;
	Image						*density_map;
	Image						current_projection;
//	ResolutionStatistics		statistics;
	float						current_resolution_limit;
	float						current_ctf;
	float						current_phi;
	float						current_theta;
	float						current_psi;
	float						current_shift_x;
	float						current_shift_y;
	float						current_mask_radius;
	float						current_mask_falloff;
	bool						current_whitening;
	bool						current_swap_quadrants;

	bool						volume_initialized;
	bool						projection_initialized;
	bool						has_masked_applied;
	bool						was_corrected;
//	bool						has_statistics;
	bool						has_been_filtered;
	bool						whitened_projection;

	ReconstructedVolume(float wanted_molecular_mass_in_kDa = 0.0);
	~ReconstructedVolume();

	ReconstructedVolume & operator = (const ReconstructedVolume &t);
	ReconstructedVolume & operator = (const ReconstructedVolume *t);

	void CopyAllButVolume(const ReconstructedVolume *other_volume);
	void Deallocate();
	void InitWithReconstruct3D(Reconstruct3D &image_reconstruction, float wanted_pixel_size);
	void InitWithDimensions(int wanted_logical_x_dimension, int wanted_logical_y_dimension, int wanted_logical_z_dimension, float wanted_pixel_size, wxString = "C1");
	void PrepareForProjections(float low_resolution_limit, float high_resolution_limit, bool approximate_binning = false, bool apply_binning = true);
//	void PrepareForProjections(float resolution_limit, bool approximate_binning = false, bool apply_binning = true);
	void CalculateProjection(Image &projection, Image &CTF, AnglesAndShifts &angles_and_shifts_of_projection, float mask_radius = 0.0, float mask_falloff = 0.0,
		float resolution_limit = 1.0, bool swap_quadrants = false, bool apply_shifts = false, bool whiten = false, bool apply_ctf = false, bool abolute_ctf = false, bool calculate_projection = true);
	void Calculate3DSimple(Reconstruct3D &reconstruction);
	void Calculate3DOptimal(Reconstruct3D &reconstruction, ResolutionStatistics &statistics, float weiner_filter_nominater = 1.0f);
	float Correct3D(float mask_radius = 0.0);
	void CosineRingMask(float wanted_inner_mask_radius, float wanted_outer_mask_radius, float wanted_mask_edge);
	void CosineMask(float wanted_mask_radius, float wanted_mask_edge);
	void OptimalFilter(ResolutionStatistics &statistics);
	void FinalizeSimple(Reconstruct3D &reconstruction, int &original_box_size, float &original_pixel_size, float &pixel_size,
			float &inner_mask_radius, float &outer_mask_radius, float &mask_falloff, wxString &output_volume);
	void FinalizeOptimal(Reconstruct3D &reconstruction, Image *density_map_1, Image *density_map_2,
			float &original_pixel_size, float &pixel_size, float &inner_mask_radius, float &outer_mask_radius, float &mask_falloff,
			bool center_mass, wxString &output_volume, NumericTextFile &output_statistics, ResolutionStatistics *copy_of_statistics = NULL, float weiner_filter_nominator = 1.0f);
	void FinalizeML(Reconstruct3D &reconstruction, Image *density_map_1, Image *density_map_2,
			float &original_pixel_size, float &pixel_size, float &inner_mask_radius, float &outer_mask_radius, float &mask_falloff,
			wxString &output_volume, NumericTextFile &output_statistics, ResolutionStatistics *copy_of_statistics = NULL);
	void Calculate3DML(Reconstruct3D &reconstruction);
	float ComputeOrientationDistributionEfficiency(Reconstruct3D &reconstruction);
};
