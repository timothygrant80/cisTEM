class BruteForceSearch
{
	//
private:
	int	number_of_dimensions;
	bool is_in_memory;
	float (*target_function)(void* parameters, float []);
	void *parameters;
	float *starting_value;
	float *best_value;
	float *half_range;
	float *step_size;
	bool *dimension_at_max;
	float best_score;
	int num_iterations;
	bool minimise_at_every_step;
	bool print_progress_bar;
	int desired_num_threads;

public:
	// Constructors & destructors
	BruteForceSearch();
	~BruteForceSearch();

	// Methods
	void Init(float (*function_to_minimize)(void* parameters, float []), void *parameters, int num_dim, float starting_value[], float half_range[], float step_size[], bool minimise_at_every_step, bool print_progress_bar, int wanted_desired_num_threads = 12 );
	void Run();
	void IncrementCurrentValues(float *current_values, bool &search_is_now_completed);
	float GetBestValue(int index);
	inline float GetBestScore() {return best_score;};

};
