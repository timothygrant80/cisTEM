class RefinementResult {

  public:
    RefinementResult( );
    ~RefinementResult( );
    long  position_in_stack;
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
    int   image_is_active;
    float pixel_size;
    float microscope_voltage_kv;
    float microscope_spherical_aberration_mm;
    float amplitude_contrast;
    float beam_tilt_x;
    float beam_tilt_y;
    float image_shift_x;
    float image_shift_y;
    int   beam_tilt_group;
    int   particle_group;
    int   assigned_subset;
    float pre_exposure;
    float total_exposure;
};

typedef std::vector<RefinementResult> ArrayofRefinementResults;

class ClassRefinementResults {
  public:
    ClassRefinementResults( );
    ~ClassRefinementResults( );

    ResolutionStatistics class_resolution_statistics;

    float global_resolution_limit;
    float high_resolution_limit;
    float classification_resolution_limit;
    float low_resolution_limit;
    float mask_radius;
    float signed_cc_resolution_limit;
    float global_mask_radius;
    int   number_results_to_refine;
    float angular_search_step;
    float search_range_x;
    float search_range_y;
    bool  should_focus_classify;
    float sphere_x_coord;
    float sphere_y_coord;
    float sphere_z_coord;
    float sphere_radius;
    bool  should_refine_ctf;
    float defocus_search_range;
    float defocus_search_step;

    long reconstructed_volume_asset_id;
    long reconstruction_id;

    float average_occupancy;
    float estimated_resolution;

    bool  should_auto_mask;
    bool  should_refine_input_params;
    bool  should_use_supplied_mask;
    long  mask_asset_id;
    float mask_edge_width;
    float outside_mask_weight;
    bool  should_low_pass_filter_mask;
    float filter_resolution;

    ArrayofRefinementResults particle_refinement_results;
};

typedef std::vector<ClassRefinementResults> ArrayofClassRefinementResults;

class Refinement {

  public:
    Refinement( );
    ~Refinement( );

    long       refinement_id;
    long       refinement_package_asset_id;
    std::string   name;
    bool       resolution_statistics_are_generated;
    DateTime datetime_of_run;
    long       starting_refinement_id;
    long       number_of_particles;
    int        number_of_classes;
    float      percent_used;

    int   resolution_statistics_box_size;
    float resolution_statistics_pixel_size;

    void         SizeAndFillWithEmpty(long number_of_particles, int number_of_classes);
    void         UpdateOccupancies(bool use_old_occupancies = true);
    void         UpdateAverageOccupancy( );
    std::vector<float> UpdatePSSNR( );

    std::vector<long> reference_volume_ids;

    ArrayofClassRefinementResults class_refinement_results;

    float ReturnChangeInAverageOccupancy(Refinement& other_refinement);

    RefinementResult ReturnRefinementResultByClassAndPositionInStack(int wanted_class, long wanted_position_in_stack);

    void WriteSingleClassFrealignParameterFile(std::string filename, int wanted_class, float percent_used_overide = 1.0f, float sigma_override = 0.0f);
    void WriteSingleClasscisTEMStarFile(std::string filename, int wanted_class, float percent_used_overide = 1.0f, float sigma_override = 0.0f, bool write_binary_file = false);

    std::vector<std::string> WriteFrealignParameterFiles(std::string base_filename, float percent_used_overide = 1.0f, float sigma_override = 0.0f);
    std::vector<std::string> WritecisTEMStarFiles(std::string base_filename, float percent_used_overide = 1.0f, float sigma_override = 0.0f, bool write_binary_files = false);
    std::vector<std::string> WriteResolutionStatistics(std::string base_filename, float pssnr_division_factor = 1.0f);

    long ReturnNumberOfActiveParticlesInFirstClass( );

    int                                  ReturnClassWithHighestOccupanyForGivenParticle(long wanted_particle);
    ArrayofAngularDistributionHistograms ReturnAngularDistributions(std::string desired_symmetry);
    void                                 FillAngularDistributionHistogram(std::string wanted_symmetry, int wanted_class, int number_of_theta_bins, int number_of_phi_bins, AngularDistributionHistogram& histogram_to_fill);

    void SetAllPixelSizes(float wanted_pixel_size);
    void SetAllVoltages(float wanted_voltage_in_kV);
    void SetAllCs(float wanted_Cs_in_mm);
    void SetAllAmplitudeContrast(float wanted_amplitude_contrast);
    void SetAssignedSubsetToEvenOdd( );
};

typedef std::vector<Refinement> ArrayofRefinements;

class ShortRefinementInfo {

  public:
    ShortRefinementInfo( );

    long     refinement_id;
    long     refinement_package_asset_id;
    std::string name;
    long     number_of_particles;
    int      number_of_classes;

    std::vector<float> average_occupancy;
    std::vector<float> estimated_resolution;
    std::vector<long>  reconstructed_volume_asset_ids;

    ShortRefinementInfo& operator=(const Refinement& other_refinement);
    ShortRefinementInfo& operator=(const Refinement* other_other_refinement);
};

typedef std::vector<ShortRefinementInfo> ArrayofShortRefinementInfos;
