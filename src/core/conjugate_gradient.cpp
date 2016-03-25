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
		delete [] best_values;
		delete [] e;
		is_in_memory = false;
	}
}

float ConjugateGradient::Init(float (*function_to_minimize)(void* parameters, float []), void *parameters_to_pass, int num_dim, float starting_value[], float accuracy[] )
{

	MyDebugAssertTrue(num_dim > 0,"Initializing conjugate gradient with zero dimensions");

// Copy pointers to the target function and the needed parameters
	target_function = function_to_minimize;
	parameters = parameters_to_pass;


	// Allocate memory
	n 					= 	num_dim;
	best_values 		=	new float[n];
	e					=	new float[n];
	is_in_memory		=	true;

	// Initialise values
	escale = 1000.0;
	num_function_calls = 0;


	for (int dim_counter=0; dim_counter < n; dim_counter++)
	{
		best_values[dim_counter]	=	starting_value[dim_counter];
		e[dim_counter]				=	accuracy[dim_counter];
	}

	// Call the target function to find out our starting score
	best_score = target_function(parameters,starting_value);

//	MyDebugPrint("Starting score = %f\n",best_score);
	return best_score;

}

float ConjugateGradient::Run()
{
	int iprint = 0;
	int icon = 1;
	int maxit = 50;
	int va04_success = 0;

	va04_success = va04a_(&n,e,&escale,&num_function_calls,target_function,parameters,&best_score,&iprint,&icon,&maxit,best_values);

	return best_score;
}
