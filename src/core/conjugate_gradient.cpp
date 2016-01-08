#include "core_headers.h"

ConjugateGradient::ConjugateGradient()
{
	is_in_memory = false;
	n = 0;
	num_function_calls = 0;
	best_values = NULL;
	e = NULL;
	escale = 0;
	best_score = std::numeric_limits<float>::max();
	target_function = NULL;
	parameters = NULL;
}

ConjugateGradient::~ConjugateGradient()
{
	if (is_in_memory)
	{
		delete best_values;
		delete e;
		is_in_memory = false;
	}
}

void ConjugateGradient::Init(float (*function_to_minimize)(void* parameters, float []), void *parameters_to_pass, int num_dim, float starting_value[], float accuracy[] )
{

	// Copy pointers to the target function and the needed parameters
	target_function = function_to_minimize;
	parameters = parameters_to_pass;


	// Allocate memory
	n 					= 	num_dim;
	best_values 		=	new float[n];
	e					=	new float[n];

	// Initialise values
	escale = 100.0;
	num_function_calls = 0;


	for (int dim_counter=0; dim_counter < n; dim_counter++)
	{
		best_values[dim_counter]	=	starting_value[dim_counter];
		e[dim_counter]				=	accuracy[dim_counter];
	}

	// Call the target function to find out our starting score
	best_score = target_function(parameters,starting_value);
	MyDebugPrint("Starting score = %f\n",best_score);
}

void ConjugateGradient::GetBestValues(float *best_values_returned)
{
	for (int dim_counter=0; dim_counter < n; dim_counter++)
	{
		best_values_returned[dim_counter] = best_values[dim_counter];
	}
}

void ConjugateGradient::GetBestScore(float best_score_returned)
{
	best_score_returned = best_score;
}

void ConjugateGradient::Run()
{
	int iprint = 0;
	int icon = 1;
	int maxit = 50;

	// call va04a(self,iprint,icon,maxit)
}
