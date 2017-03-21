class EulerSearch
{
	// Brute-force search to find matching projections
public:
	int			number_of_search_dimensions;
	int			refine_top_N;
	int			number_of_search_positions;
	int			best_parameters_to_keep;
	float 		angular_step_size;
	float		max_search_x;
	float		max_search_y;
	float		phi_max;
	float		phi_start;
	float		theta_max;
	float		theta_start;
	float		psi_max;
	float		psi_step;
	float		psi_start;
	float		**list_of_search_parameters;
	float		**list_of_best_parameters;
//	Kernel2D	**kernel_index;
//	float		*best_values;
//	float		best_score;
	float		resolution_limit;
	bool		*parameter_map;
	bool		test_mirror;
	wxString	symmetry_symbol;

	// Constructors & destructors
	EulerSearch();
	~EulerSearch();

	// Methods
	void Init(float wanted_resolution_limit, bool *wanted_parameter_map, int wanted_parameters_to_keep);
	void InitGrid(wxString wanted_symmetry_symbol, float angular_step_size, float wanted_phi_start, float wanted_theta_start, float wanted_psi_max, float wanted_psi_step, float wanted_psi_start, float wanted_resolution_limit, bool *parameter_map, int wanted_parameters_to_keep);
	void InitRandom(wxString wanted_symmetry_symbol, float wanted_psi_step, int wanted_number_of_search_positions, float wanted_resolution_limit, bool *wanted_parameter_map, int wanted_parameters_to_keep);
	void Run(Particle &particle, Image &input_3d, float *starting_values, Image *projections);
	void CalculateGridSearchPositions();
	void CalculateRandomSearchPositions();
	void SetSymmetryLimits();
//	void RotateFourier2DFromIndex(Image &image_to_rotate, Image &rotated_image, Kernel2D &kernel_index);
//	void RotateFourier2DIndex(Image &image_to_rotate, Kernel2D &kernel_index, AnglesAndShifts &rotation_angle, float resolution_limit = 1.0, float padding_factor = 1.0);
	Kernel2D ReturnLinearInterpolatedFourierKernel2D(Image &image_to_rotate, float &x, float &y);
};
