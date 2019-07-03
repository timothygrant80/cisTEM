class ClassificationResult {

public:

	ClassificationResult();
	~ClassificationResult();
	long position_in_stack;
	float psi;
	float xshift;
	float yshift;
	int best_class;
	float sigma;
	float logp;
	float amplitude_contrast;
	float pixel_size;
	float microscope_voltage_kv;
	float microscope_spherical_aberration_mm;
	float beam_tilt_x;
	float beam_tilt_y;
	float image_shift_x;
	float image_shift_y;
	float defocus_1;
	float defocus_2;
	float defocus_angle;
	float phase_shift;
};

WX_DECLARE_OBJARRAY(ClassificationResult, ArrayofClassificationResults);

class Classification {

public :
	Classification();
	~Classification();

	long classification_id;
	long refinement_package_asset_id;
	wxString name;
	wxString class_average_file;
	bool classification_was_imported_or_generated;
	wxDateTime datetime_of_run;
	long starting_classification_id;
	long number_of_particles;
	int number_of_classes;
	float low_resolution_limit;
	float high_resolution_limit;
	float mask_radius;
	float angular_search_step;
	float search_range_x;
	float search_range_y;
	float smoothing_factor;
	bool exclude_blank_edges;
	bool auto_percent_used;
	float percent_used;

	float ReturnXShiftByPositionInStack(long wanted_position_in_stack);
	float ReturnYShiftByPositionInStack(long wanted_position_in_stack);


	void SizeAndFillWithEmpty(long number_of_particles);

	ArrayofClassificationResults classification_results;
	wxString WriteFrealignParameterFiles(wxString base_filename, RefinementPackage *parent_refinement_package);
	wxString WritecisTEMStarFile(wxString base_filename, RefinementPackage *parent_refinement_package);
};

WX_DECLARE_OBJARRAY(Classification, ArrayofClassifications);

class ShortClassificationInfo {

public :
	ShortClassificationInfo();

	long classification_id;
	long refinement_package_asset_id;
	wxString name;
	wxString class_average_file;
	long number_of_particles;
	int number_of_classes;
	float high_resolution_limit;

	ShortClassificationInfo & operator = (const Classification &other_classification);
	ShortClassificationInfo & operator = (const Classification *other_classification);
};

WX_DECLARE_OBJARRAY(ShortClassificationInfo, ArrayofShortClassificationInfos);



