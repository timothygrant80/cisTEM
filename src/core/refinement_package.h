class RefinementPackageParticleInfo {

  public:
    RefinementPackageParticleInfo( );
    ~RefinementPackageParticleInfo( );

    long  parent_image_id;
    long  position_in_stack;
    long  original_particle_position_asset_id;
    float x_pos;
    float y_pos;
    float pixel_size;
    float defocus_1;
    float defocus_2;
    float defocus_angle;
    float phase_shift;
    float spherical_aberration;
    float amplitude_contrast;
    float microscope_voltage;
    int   assigned_subset;
};

typedef std::vector<RefinementPackageParticleInfo> ArrayOfRefinmentPackageParticleInfos;

class RefinementPackage {

  public:
    RefinementPackage( );
    ~RefinementPackage( );

    long     asset_id;
    std::string stack_filename;
    std::string name;
    int      stack_box_size;
    float    output_pixel_size;

    int number_of_classes;

    std::string symmetry;
    double   estimated_particle_size_in_angstroms;
    double   estimated_particle_weight_in_kda;
    double   lowest_resolution_of_intial_parameter_generated_3ds;

    bool stack_has_white_protein;

    int  number_of_run_refinments;
    long last_refinment_id;

    std::vector<long> references_for_next_refinement;
    std::vector<long> refinement_ids;
    std::vector<long> classification_ids;

    ArrayOfRefinmentPackageParticleInfos contained_particles;

    RefinementPackageParticleInfo ReturnParticleInfoByPositionInStack(long wanted_position_in_stack);

    long ReturnLastRefinementID( );
};

typedef std::vector<RefinementPackage> ArrayOfRefinementPackages;
