class ConjugateGradient
{
	//
private:
	bool 	is_in_memory;
	int		n;						//	Number of parameters to refine
	int		num_function_calls;		//	Number of calls to the scoring function
	float	*best_values;			//	Final best parameters determined by conjugate gradient minimization
	float 	*e;						//	Recalculated accuracy for subroutine va04
	float 	escale;					//	Anticipated change in the parameters during conjugate gradient minimization
	float	best_score;				//	Final best score
	//
	float (*target_function)(void* parameters, float []);
	void *parameters;

public:
	// Constructors & destructors
	ConjugateGradient();
	~ConjugateGradient();

	// Methods
	void Init(float (*function_to_minimize)(void* parameters, float []), void *parameters, int num_dim, float starting_value[], float accuracy[] );
	void Run();
	inline float GetBestValue(int index) { return best_values[index]; };
	inline float GetBestScore() { return best_score; };

};

extern "C" {
int va04a_(int *n, float *e, float *escale, int *self_num_function_calls__,
		float (*self_target_function) (void* parameters, float[]),
		void *self_parameter_ptr__, float *self_best_score__, int *iprint,
		int *icon, int *maxit, float *x);
}
