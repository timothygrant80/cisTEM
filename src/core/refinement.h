class RefinementResult {

public:

	RefinementResult();
	~RefinementResult();
	long position_in_stack;
	float psi;
	float theta;
	float phi;
	float xshift;
	float yshift;
	float defocus1;
	float defocus2;
	float defocus_angle;
	float phase_shift;
	float occupancy;
	float logp;
	float sigma;
	float score;
	int image_is_active;
};

WX_DECLARE_OBJARRAY(RefinementResult, ArrayofRefinementResults);


class ClassRefinementResults {
public :

	ResolutionStatistics class_resolution_statistics;
	ArrayofRefinementResults particle_refinement_results;
};

WX_DECLARE_OBJARRAY(ClassRefinementResults, ArrayofClassRefinementResults);

class Refinement {

public :
	Refinement();
	~Refinement();

	long refinement_id;
	long refinement_package_asset_id;
	wxString name;
	bool refinement_was_imported_or_generated;
	wxDateTime datetime_of_run;
	long starting_refinement_id;
	long number_of_particles;
	int number_of_classes;
	float low_resolution_limit;
	float high_resolution_limit;
	float mask_radius;
	float signed_cc_resolution_limit;
	float global_resolution_limit;
	float global_mask_radius;
	int number_results_to_refine;
	float angular_search_step;
	float search_range_x;
	float search_range_y;
	float classification_resolution_limit;
	bool should_focus_classify;
	float sphere_x_coord;
	float sphere_y_coord;
	float sphere_z_coord;
	float sphere_radius;
	bool should_refine_ctf;
	float defocus_search_range;
	float defocus_search_step;
	int resolution_statistics_box_size;
	float resolution_statistics_pixel_size;
	float percent_used;
	//wxArrayDouble average_occupancy;

	void SizeAndFillWithEmpty(long number_of_particles, int number_of_classes);
	void UpdateOccupancies(bool use_old_occupancies = true);
	wxArrayDouble UpdatePSSNR();

	wxArrayLong reference_volume_ids;
	ArrayofClassRefinementResults class_refinement_results;

	wxArrayString WriteFrealignParameterFiles(wxString base_filename, float percent_used_overide = 1.0);
	wxArrayString WriteResolutionStatistics(wxString base_filename);


};

WX_DECLARE_OBJARRAY(Refinement, ArrayofRefinements);

class ShortRefinementInfo {

public :
	ShortRefinementInfo();

	long refinement_id;
	long refinement_package_asset_id;
	wxString name;
	long number_of_particles;
	int number_of_classes;

	ShortRefinementInfo & operator = (const Refinement &other_refinement);
	ShortRefinementInfo & operator = (const Refinement *other_other_refinement);
};

WX_DECLARE_OBJARRAY(ShortRefinementInfo, ArrayofShortRefinementInfos);



