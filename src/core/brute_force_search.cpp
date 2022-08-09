#include "core_headers.h"

BruteForceSearch::BruteForceSearch( ) {
    // Nothing to do until Init is called
    number_of_dimensions   = 0;
    is_in_memory           = false;
    target_function        = NULL;
    parameters             = NULL;
    starting_value         = NULL;
    best_value             = NULL;
    half_range             = NULL;
    step_size              = NULL;
    dimension_at_max       = NULL;
    best_score             = std::numeric_limits<float>::max( );
    num_iterations         = 0;
    minimise_at_every_step = false;
    print_progress_bar     = false;
}

BruteForceSearch::~BruteForceSearch( ) {
    //
    if ( is_in_memory ) {
        delete[] starting_value;
        delete[] best_value;
        delete[] half_range;
        delete[] step_size;
        delete[] dimension_at_max;
        is_in_memory = false;
    }
}

void BruteForceSearch::Init(float (*function_to_minimize)(void* parameters, float[]), void* wanted_parameters, int num_dim, float wanted_starting_value[], float wanted_half_range[], float wanted_step_size[], bool should_minimise_at_every_step, bool should_print_progress_bar, int wanted_desired_num_threads) {
    MyDebugAssertFalse(is_in_memory, "Brute force search object is already setup");
    MyDebugAssertTrue(num_dim > 0, "Bad number of dimensions: %i", num_dim);
    MyDebugAssertTrue(num_dim <= 16, "Recent versions of BruteForceSearch (with threading) may not work with more than 16 dimensions");

    // Local variables
    float* current_values = new float[num_dim];
    bool   search_done;

    // Allocate memory
    number_of_dimensions = num_dim;
    starting_value       = new float[number_of_dimensions];
    best_value           = new float[number_of_dimensions];
    half_range           = new float[number_of_dimensions];
    step_size            = new float[number_of_dimensions];
    dimension_at_max     = new bool[number_of_dimensions];
    is_in_memory         = true;

    // Copy starting values, half range and step size over; initialise
    for ( int dim_counter = 0; dim_counter < number_of_dimensions; dim_counter++ ) {
        starting_value[dim_counter]   = wanted_starting_value[dim_counter];
        half_range[dim_counter]       = wanted_half_range[dim_counter];
        step_size[dim_counter]        = wanted_step_size[dim_counter];
        dimension_at_max[dim_counter] = false;
        current_values[dim_counter]   = -std::numeric_limits<float>::max( );
    }

    //
    minimise_at_every_step = should_minimise_at_every_step;

    // Work out how many iterations the exhaustive search will take
    // we will start at lowest value for each dimension, then go in step sizes, until we get to upper limit.
    // we will also do extra iterations for the starting value and the upper limit
    num_iterations = 0;
    while ( true ) {
        IncrementCurrentValues(current_values, search_done);
        if ( search_done ) {
            break;
        }
        num_iterations++;
    }

    //
    target_function     = function_to_minimize;
    parameters          = wanted_parameters;
    print_progress_bar  = should_print_progress_bar;
    desired_num_threads = wanted_desired_num_threads;
    delete[] current_values;
}

float BruteForceSearch::GetBestValue(int index) {
    MyDebugAssertTrue(index < number_of_dimensions, "Index %i does not exist (number of dimensions = %i", index, number_of_dimensions);
    return best_value[index];
}

// In the BF loop, this should be called before the scoring function. Before the BF loop, set all values in current_values to -huge
void BruteForceSearch::IncrementCurrentValues(float* current_values, bool& search_is_now_completed) {
    MyDebugAssertTrue(is_in_memory, "Brute force search object not allocated");

    int i;
    int j;

    // do the increment
    search_is_now_completed = true;
    for ( i = 0; i < number_of_dimensions; i++ ) {
        // if we haven't reached the max for this dimension, increment it, and reset all previous dimensions to their starting point
        if ( ! dimension_at_max[i] ) {
            // if we got here, it means our search is not over yet
            search_is_now_completed = false;
            // increment the ith dimension
            // it could be that the ith dimension has a very large negative value, perhaps because the search hasn't even started yet
            if ( current_values[i] < starting_value[i] - half_range[i] ) {
                current_values[i] = starting_value[i] - half_range[i];
            }
            else {
                current_values[i] += step_size[i];
            }
            // Reset all previous dimensions to their starting values and reset their dimension_at_max flags to false
            if ( i > 0 ) {
                for ( j = 0; j < i; j++ ) {
                    current_values[j] = starting_value[j] - half_range[j];
                    MyDebugAssertTrue(dimension_at_max[j], "failed sanity check");
                    dimension_at_max[j] = false;
                }
            }
            // if the ith dimension has reached or gone over its max, set it at the max, and set its logical flag dimension_at_max to .true.
            if ( current_values[i] >= starting_value[i] + half_range[i] ) {
                current_values[i]   = starting_value[i] + half_range[i];
                dimension_at_max[i] = true;
                if ( i == number_of_dimensions - 1 ) {
                    search_is_now_completed = true;
                }
            }
            break;
        }
    } // end of loop over dimensions

    for ( i = 0; i < number_of_dimensions; i++ ) {
        if ( current_values[i] < starting_value[i] - half_range[i] ) {
            current_values[i] = starting_value[i] - half_range[i];
        }

        MyDebugAssertFalse(std::isnan(current_values[i]), "Oops. NaN");
    }
}

// Run the brute force minimiser's exhaustive search
// DNM: parallelized by storing the step values
void BruteForceSearch::Run( ) {
    MyDebugAssertTrue(is_in_memory, "BruteForceSearch object not allocated");

    // Private variables
    // DNM: Switch to a fixed dimensioned array so that OpenMP will make copies automatically
    float  current_values[16];
    float* accuracy_for_local_minimization       = new float[number_of_dimensions];
    float* current_values_for_local_minimization = new float[number_of_dimensions];
    // DNM: New arrays for holding the values at each step, the scores from each step,
    // and the best values from the CG search at each step
    float*       all_values            = new float[number_of_dimensions * num_iterations];
    float*       all_scores            = new float[num_iterations];
    float*       all_local_best_values = new float[number_of_dimensions * num_iterations];
    int          i;
    int          num_iterations_completed;
    int          current_iteration;
    bool         search_completed;
    float        current_score;
    int          numThreads, maxThreads = 12;
    ProgressBar* my_progress_bar;

    // The starting values and the corresponding score
    best_score = target_function(parameters, starting_value);
    for ( i = 0; i < number_of_dimensions; i++ ) {
        best_value[i] = starting_value[i];
    }

    // The starting point for the brute-force search
    for ( i = 0; i < number_of_dimensions; i++ ) {
        current_values[i]   = -std::numeric_limits<float>::max( );
        dimension_at_max[i] = false;
    }

    // DNM: Go through the search steps and save the values to be set at each step
    for ( current_iteration = 0; current_iteration < num_iterations; current_iteration++ ) {
        IncrementCurrentValues(current_values, search_completed);
        MyDebugAssertFalse(search_completed, "Failed sanity check");
        for ( i = 0; i < number_of_dimensions; i++ ) {
            all_values[number_of_dimensions * current_iteration + i] = current_values[i];
        }
    }

    // The accuracy for the local minimization
    if ( minimise_at_every_step )
        for ( i = 0; i < number_of_dimensions; i++ ) {
            accuracy_for_local_minimization[i] = step_size[i] * 0.5;
        }

    // How many iterations have we completed?
    num_iterations_completed = 0;

    //numThreads = ReturnAppropriateNumberOfThreads(desired_num_threads);
    numThreads = CheckNumberOfThreads(desired_num_threads);

    if ( numThreads > 1 && ! minimise_at_every_step ) {
        wxPrintf("\nRunning brute-force search with %i OpenMP threads\n", numThreads);
    }

    if ( print_progress_bar ) {
        my_progress_bar = new ProgressBar(num_iterations);
    }

    // start the brute-force search iterations
    // DNM the minimizer did not work right with threads, so just run it normally
    if ( minimise_at_every_step ) {
        for ( current_iteration = 0; current_iteration < num_iterations; current_iteration++ ) {
            // what values for the parameters should we try now?
            // DNM: switch to getting from the list
            //IncrementCurrentValues(current_values,search_completed);
            for ( i = 0; i < number_of_dimensions; i++ )
                current_values[i] = all_values[number_of_dimensions * current_iteration + i];

            ConjugateGradient local_minimizer;
            for ( i = 0; i < number_of_dimensions; i++ ) {
                current_values_for_local_minimization[i] = current_values[i];
            }
            local_minimizer.Init(target_function, parameters, number_of_dimensions, current_values_for_local_minimization, accuracy_for_local_minimization);
            local_minimizer.Run( );
            // DNM: store the results
            all_scores[current_iteration] = local_minimizer.GetBestScore( );
            for ( i = 0; i < number_of_dimensions; i++ )
                all_local_best_values[number_of_dimensions * current_iteration + i] = local_minimizer.GetBestValue(i);

            // Progress
            if ( print_progress_bar ) {
                num_iterations_completed++;
                my_progress_bar->Update(num_iterations_completed);
            }
        }
    }
    else {

#pragma omp parallel for default(none) num_threads(numThreads) shared(all_values, all_local_best_values, all_scores, numThreads, num_iterations_completed, my_progress_bar) private(current_iteration, current_values, current_values_for_local_minimization, i, accuracy_for_local_minimization)
        for ( current_iteration = 0; current_iteration < num_iterations; current_iteration++ ) {
            // Grab the next values to be tried
            for ( i = 0; i < number_of_dimensions; i++ ) {
                current_values[i] = all_values[number_of_dimensions * current_iteration + i];
            }

            // Try the current parameters by calling the scoring function
            all_scores[current_iteration] = target_function(parameters, current_values);

            // Progress
            if ( print_progress_bar ) {
#pragma omp atomic
                num_iterations_completed++;
                my_progress_bar->Update(num_iterations_completed);
            }
        }
    }

    if ( print_progress_bar ) {
        delete my_progress_bar;
    }

    // DNM: Loop through the iterations again and look for the best result
    for ( current_iteration = 0; current_iteration < num_iterations; current_iteration++ ) {
        for ( i = 0; i < number_of_dimensions; i++ ) {
            current_values[i] = all_values[number_of_dimensions * current_iteration + i];
        }
        current_score = all_scores[current_iteration];
        // Print debug
        /*wxPrintf("(BF dbg, it %04i) Values: ",current_iteration);
		for (i=0;i<number_of_dimensions;i++)
		{
			wxPrintf("%g ",current_values[i]);
		}
		wxPrintf(" Score: %g\n",current_score);*/

        // if the score is the best we've seen so far, remember the values and the score
        if ( current_score < best_score ) {
            best_score = current_score;
            //wxPrintf("new best values: ");
            if ( minimise_at_every_step ) {
                for ( i = 0; i < number_of_dimensions; i++ ) {
                    best_value[i] = all_local_best_values[number_of_dimensions * current_iteration + i];
                    //wxPrintf("%g ",best_value[i]);
                }
            }
            else {
                for ( i = 0; i < number_of_dimensions; i++ ) {
                    best_value[i] = current_values[i];
                    //wxPrintf("%g ",best_value[i]);
                }
            }
            //wxPrintf("\n");
        }

    } // End of loop over exhaustive search operation

    delete[] accuracy_for_local_minimization;
    delete[] current_values_for_local_minimization;
    delete[] all_values;
    delete[] all_scores;
    delete[] all_local_best_values;
}
