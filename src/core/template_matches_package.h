/* class RefinementPackageParticleInfo {

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

typedef std::vector<RefinementPackageParticleInfo> ArrayOfRefinmentPackageParticleInfos; */

class TemplateMatchesPackage {

  public:
    TemplateMatchesPackage( );
    ~TemplateMatchesPackage( );

    long     asset_id;
    std::string starfile_filename;
    std::string name;
    long     contained_match_count;

    std::vector<long> match_template_result_ids;
};

typedef std::vector<TemplateMatchesPackage> ArrayOfTemplateMatchesPackages;
