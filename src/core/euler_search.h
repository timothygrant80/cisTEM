class EulerSearch {
    // Brute-force search to find matching projections

  public:
    int     number_of_search_dimensions;
    int     refine_top_N;
    int     number_of_search_positions;
    int     best_parameters_to_keep;
    float   angular_step_size;
    float   max_search_x;
    float   max_search_y;
    float   phi_max;
    float   phi_start;
    float   theta_max;
    float   theta_start;
    float   psi_max;
    float   psi_step;
    float   psi_start;
    float** list_of_search_parameters;
    float** list_of_best_parameters;
    //	Kernel2D	 **kernel_index;
    //	float		 *best_values;
    //	float		 best_score;
    float        resolution_limit;
    ParameterMap parameter_map;
    bool         test_mirror;
    std::string     symmetry_symbol;

    // Constructors & destructors
    EulerSearch( );
    ~EulerSearch( );

    EulerSearch(const EulerSearch& other_search);

    EulerSearch& operator=(const EulerSearch& t);
    EulerSearch& operator=(const EulerSearch* t);

    // Methods
    void Init(float wanted_resolution_limit, ParameterMap& wanted_parameter_map, int wanted_parameters_to_keep);
    void InitGrid(std::string wanted_symmetry_symbol, float angular_step_size, float wanted_phi_start, float wanted_theta_start, float wanted_psi_max, float wanted_psi_step, float wanted_psi_start, float wanted_resolution_limit, ParameterMap& parameter_map, int wanted_parameters_to_keep);
    void InitRandom(std::string wanted_symmetry_symbol, float wanted_psi_step, int wanted_number_of_search_positions, float wanted_resolution_limit, ParameterMap& wanted_parameter_map, int wanted_parameters_to_keep);
    void Run(Particle& particle, Image& input_3d, Image* projections);
    void CalculateGridSearchPositions(bool random_start_angle = true);
    void CalculateRandomSearchPositions( );
    void SetSymmetryLimits( );
    //	void RotateFourier2DFromIndex(Image &image_to_rotate, Image &rotated_image, Kernel2D &kernel_index);
    //	void RotateFourier2DIndex(Image &image_to_rotate, Kernel2D &kernel_index, AnglesAndShifts &rotation_angle, float resolution_limit = 1.0, float padding_factor = 1.0);
    Kernel2D ReturnLinearInterpolatedFourierKernel2D(Image& image_to_rotate, float& x, float& y);

    // Scratch kept between calls to Run(): one EulerSearch per thread, so these are per thread. Allocating and
    // FFT-planning them for every particle serialised the threads on the FFTW planning mutex (see Image::Allocate).
    Image* run_flipped_image;
    Image* run_padded_image;
    Image* run_projection_image;
    Image* run_rotation_cache;
    int    run_rotation_cache_size;
    // The correlation maps of one orientation for every in-plane angle, transformed together by one many-transform
    // plan (run_batch_complex: the products, one half-plane per angle in the Image layout; run_batch_real: the maps,
    // unpadded), made once per thread for a size.
    float*     run_batch_complex;
    float*     run_batch_real;
    fftwf_plan run_batch_plan;      // transforms run_batch_chunk maps
    fftwf_plan run_batch_plan_rest; // the remainder when the angle count is not a multiple of the chunk, else NULL
    int        run_batch_count;
    int        run_batch_chunk;
    int        run_batch_nx;
    int        run_batch_ny;
    void       EnsureBatchPlan(int count, int nx, int ny);
    void       FreeBatchPlan( );
    void       FreeRunScratch( );
};
