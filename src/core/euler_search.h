class EulerSearch
{
	// Brute-force search to find matching projections
public:
	int		number_of_search_dimensions;
	int		refine_top_N;
	int		number_of_search_positions;
	int		best_parameters_to_keep;
	float 	angular_step_size;
	float	psi_max;
	float	psi_step;
	float	*starting_values;
	float	**list_of_search_parameters;
	float	**list_of_best_parameters;
//	Kernel2D **kernel_index;
//	float	*best_values;
//	float	best_score;
	float	resolution_limit;
	bool	*parameter_map;
	bool	test_mirror;

	// Constructors & destructors
	EulerSearch();
	~EulerSearch();

	// Methods
	void Init(float wanted_resolution_limit, bool *wanted_parameter_map, int wanted_parameters_to_keep);
	void InitGrid(float angular_step_size, float wanted_psi_step, float wanted_resolution_limit, bool *parameter_map, int wanted_parameters_to_keep);
	void InitRandom(float wanted_psi_step, float *starting_values, int wanted_number_of_search_positions, float wanted_resolution_limit, bool *wanted_parameter_map, int wanted_parameters_to_keep);
	void Run(Particle &particle, Image &input_3d, Image *projections, Kernel2D **kernel_index);
	void CalculateGridSearchPositions();
	void CalculateRandomSearchPositions();
	void RotateFourier2DFromIndex(Image &image_to_rotate, Image &rotated_image, Kernel2D &kernel_index);
	void RotateFourier2DIndex(Image &image_to_rotate, Kernel2D &kernel_index, AnglesAndShifts &rotation_angle, float resolution_limit = 1.0, float padding_factor = 1.0);
	Kernel2D ReturnLinearInterpolatedFourierKernel2D(Image &image_to_rotate, float &x, float &y);
};
