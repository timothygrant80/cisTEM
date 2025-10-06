#include "../../core/core_headers.h"
#include <iostream>
#include <vector>

using namespace std;
using namespace cistem;

class unblur_refine_alignment_object {

  public:
    // long image_counter;
    Image* input_stack;
    int    number_of_images;
    int    max_iterations;
    float  unitless_bfactor;
    bool   mask_central_cross;
    int    width_of_vertical_line;
    int    width_of_horizontal_line;
    float  inner_radius_for_peak_search;
    float  outer_radius_for_peak_search;
    float  max_shift_convergence_threshold;
    float  pixel_size;
    int    number_of_frames_for_running_average;
    int    savitzy_golay_window_size;
    int    max_threads;
    float* x_shifts;
    float* y_shifts;
    bool   reverse_shift;
    int    running_average_counter;

    long  iteration_counter;
    Image sum_of_images;

    Image* running_average_stack;

    Image* stack_for_alignment; // pointer that can be switched between running average stack and image stack if necessary

    Curve x_shifts_curve;
    Curve y_shifts_curve;

    float middle_image_x_shift;
    float middle_image_y_shift;

    float* current_x_shifts;
    float* current_y_shifts;
    float* smooth_x_shifts;
    float* smooth_y_shifts;

    float max_shift;
    float total_shift;
    int   number_of_middle_image; // = number_of_images / 2;
    int   running_average_half_size; // = (number_of_frames_for_running_average - 1) / 2;

    void Initialize(Image* input_stack, int number_of_images, int max_iterations, float unitless_bfactor, bool mask_central_cross, int width_of_vertical_line, int width_of_horizontal_line, float inner_radius_for_peak_search, float outer_radius_for_peak_search, float max_shift_convergence_threshold, float pixel_size, int number_of_frames_for_running_average, int savitzy_golay_window_size, int max_threads, float* x_shifts, float* y_shifts, bool reverse_shift);
    //  {
    //     this->input_stack                          = input_stack;
    //     this->number_of_images                     = number_of_images;
    //     this->max_iterations                       = max_iterations;
    //     this->unitless_bfactor                     = unitless_bfactor;
    //     this->mask_central_cross                   = mask_central_cross;
    //     this->width_of_vertical_line               = width_of_vertical_line;
    //     this->width_of_horizontal_line             = width_of_horizontal_line;
    //     this->inner_radius_for_peak_search         = inner_radius_for_peak_search;
    //     this->outer_radius_for_peak_search         = outer_radius_for_peak_search;
    //     this->max_shift_convergence_threshold      = max_shift_convergence_threshold;
    //     this->pixel_size                           = pixel_size;
    //     this->number_of_frames_for_running_average = number_of_frames_for_running_average;
    //     this->max_threads                          = max_threads;
    //     this->reverse_shift                        = reverse_shift;
    //     this->savitzy_golay_window_size            = savitzy_golay_window_size;
    //     this->current_x_shifts                     = new float[number_of_images];
    //     this->current_y_shifts                     = new float[number_of_images];
    //     this->smooth_x_shifts                      = new float[number_of_images];
    //     this->smooth_y_shifts                      = new float[number_of_images];
    //     this->x_shifts                             = x_shifts;
    //     this->y_shifts                             = y_shifts;

    //     this->number_of_middle_image = this->number_of_images / 2;

    //     this->running_average_half_size = (this->number_of_frames_for_running_average - 1) / 2;
    //     if ( this->running_average_half_size < 1 )
    //         this->running_average_half_size = 1;

    //     if ( IsOdd(savitzy_golay_window_size) == false )
    //         this->savitzy_golay_window_size++;
    //     if ( savitzy_golay_window_size < 3 )
    //         this->savitzy_golay_window_size = 3;

    //     this->sum_of_images.Allocate(input_stack[0].logical_x_dimension, input_stack[0].logical_y_dimension, false);
    //     this->sum_of_images.SetToConstant(0.0);

    //     if ( number_of_frames_for_running_average > 1 ) {
    //         this->running_average_stack = new Image[number_of_images];

    //         for ( int image_counter = 0; image_counter < number_of_images; image_counter++ ) {
    //             this->running_average_stack[image_counter].Allocate(this->input_stack[image_counter].logical_x_dimension, input_stack[image_counter].logical_y_dimension, 1, false);
    //         }

    //         this->stack_for_alignment = this->running_average_stack;
    //     }
    //     else
    //         this->stack_for_alignment = this->input_stack;

    //     // prepare the initial sum
    //     for ( int image_counter = 0; image_counter < number_of_images; image_counter++ ) {
    //         this->sum_of_images.AddImage(&input_stack[image_counter]);
    //         this->current_x_shifts[image_counter] = 0;
    //         this->current_y_shifts[image_counter] = 0;
    //     }
    // }

    // perform the main alignment loop until we reach a max shift less than wanted, or max iterations
    void Update_Running_Average( );
    //      {
    //         int start_frame_for_average;
    //         int end_frame_for_average;

    // #pragma omp parallel for default(shared) num_threads(this->max_threads) private(start_frame_for_average, end_frame_for_average, running_average_counter)
    //         for ( int image_counter = 0; image_counter < number_of_images; image_counter++ ) {
    //             start_frame_for_average = image_counter - this->running_average_half_size;
    //             end_frame_for_average   = image_counter + this->running_average_half_size;

    //             if ( start_frame_for_average < 0 ) {
    //                 end_frame_for_average -= start_frame_for_average; // add it to the right
    //                 start_frame_for_average = 0;
    //             }

    //             if ( end_frame_for_average >= number_of_images ) {
    //                 start_frame_for_average -= (end_frame_for_average - (number_of_images - 1));
    //                 end_frame_for_average = number_of_images - 1;
    //             }

    //             if ( start_frame_for_average < 0 )
    //                 start_frame_for_average = 0;
    //             if ( end_frame_for_average >= number_of_images )
    //                 end_frame_for_average = number_of_images - 1;
    //             this->running_average_stack[image_counter].SetToConstant(0.0f);

    //             for ( int running_average_counter = start_frame_for_average; running_average_counter <= end_frame_for_average; running_average_counter++ ) {
    //                 this->running_average_stack[image_counter].AddImage(&this->input_stack[running_average_counter]);
    //             }
    //         }
    //     }

    void Calculate_Shifts(std::vector<int> target_index, float target_unitless_bfactor);
    //     {
    //         Peak  my_peak;
    //         Image sum_of_images_minus_current;
    // #pragma omp parallel default(shared) num_threads(this->max_threads) private(sum_of_images_minus_current, my_peak)
    //         { // for omp
    //             sum_of_images_minus_current.Allocate(input_stack[0].logical_x_dimension, input_stack[0].logical_y_dimension, false);

    // #pragma omp for
    //             for ( int i = 0; i < target_index.size( ); ++i ) {
    //                 int image_counter = target_index[i];
    //                 sum_of_images_minus_current.CopyFrom(&sum_of_images);
    //                 sum_of_images_minus_current.SubtractImage(&this->stack_for_alignment[image_counter]);
    //                 sum_of_images_minus_current.ApplyBFactor(target_unitless_bfactor);

    //                 if ( mask_central_cross == true ) {
    //                     sum_of_images_minus_current.MaskCentralCross(this->width_of_vertical_line, this->width_of_horizontal_line);
    //                 }

    //                 // compute the cross correlation function and find the peak
    //                 sum_of_images_minus_current.CalculateCrossCorrelationImageWith(&this->stack_for_alignment[image_counter]);
    //                 my_peak = sum_of_images_minus_current.FindPeakWithParabolaFit(this->inner_radius_for_peak_search, this->outer_radius_for_peak_search);

    //                 this->current_x_shifts[image_counter] = my_peak.x;
    //                 this->current_y_shifts[image_counter] = my_peak.y;
    //                 // wxPrintf("after update %i %f %f \n", image_counter, this->current_x_shifts[image_counter], this->current_y_shifts[image_counter]);
    //             }

    //             sum_of_images_minus_current.Deallocate( );

    //         } // end omp
    //     }

    void Smooth_Shifts( );
    //  {
    //     this->x_shifts_curve.ClearData( );
    //     this->y_shifts_curve.ClearData( );

    //     for ( int image_counter = 0; image_counter < number_of_images; image_counter++ ) {
    //         this->x_shifts_curve.AddPoint(image_counter, this->x_shifts[image_counter] + this->current_x_shifts[image_counter]);
    //         this->y_shifts_curve.AddPoint(image_counter, this->y_shifts[image_counter] + this->current_y_shifts[image_counter]);
    //     }
    //     if ( inner_radius_for_peak_search != 0 ) {
    //         if ( this->x_shifts_curve.number_of_points > 2 ) {
    //             this->x_shifts_curve.FitPolynomialToData(4);
    //             this->y_shifts_curve.FitPolynomialToData(4);
    //         }
    //         for ( int image_counter = 0; image_counter < number_of_images; image_counter++ ) {
    //             this->smooth_x_shifts[image_counter] = this->x_shifts_curve.polynomial_fit[image_counter] - this->x_shifts[image_counter];
    //             this->smooth_y_shifts[image_counter] = this->y_shifts_curve.polynomial_fit[image_counter] - this->y_shifts[image_counter];
    //         }
    //         //copy back
    //         //                     for ( int image_counter = 0; image_counter < number_of_images; image_counter++ ) {
    //         //                         this->current_x_shifts[image_counter] = this->x_shifts_curve.polynomial_fit[image_counter] - this->x_shifts[image_counter];
    //         //                         this->current_y_shifts[image_counter] = this->y_shifts_curve.polynomial_fit[image_counter] - this->y_shifts[image_counter];

    //         // #ifdef PRINT_VERBOSE
    //         //                         wxPrintf("After SG = %li : %f, %f\n", image_counter, x_shifts_curve.savitzky_golay_fit[image_counter], y_shifts_curve.savitzky_golay_fit[image_counter]);
    //         // #endif
    //         //                     }
    //     }
    //     else {
    //         if ( this->savitzy_golay_window_size < this->x_shifts_curve.number_of_points ) // when the input movie is dodgy (very few frames), the fitting won't work
    //         {
    //             this->x_shifts_curve.FitSavitzkyGolayToData(this->savitzy_golay_window_size, 1);
    //             this->y_shifts_curve.FitSavitzkyGolayToData(this->savitzy_golay_window_size, 1);
    //         }
    //         for ( int image_counter = 0; image_counter < number_of_images; image_counter++ ) {
    //             this->smooth_x_shifts[image_counter] = this->x_shifts_curve.savitzky_golay_fit[image_counter] - this->x_shifts[image_counter];
    //             this->smooth_y_shifts[image_counter] = this->y_shifts_curve.savitzky_golay_fit[image_counter] - this->y_shifts[image_counter];
    //         }
    //     }
    // };

    void Update_SumImage( );
    //  {
    //     this->sum_of_images.SetToConstant(0.0);
    //     for ( int image_counter = 0; image_counter < this->number_of_images; image_counter++ ) {
    //         this->sum_of_images.AddImage(&this->input_stack[image_counter]);
    //     }
    // }

    std::vector<int> SearchOutlierIndices( );
    //  {
    //     std::vector<double> diffx_vec(number_of_images);
    //     std::vector<double> diffy_vec(number_of_images);
    //     std::vector<double> diff_vec(number_of_images);

    //     for ( int image_counter = 0; image_counter < number_of_images; image_counter++ ) {
    //         diffx_vec[image_counter] = this->x_shifts[image_counter] + this->current_x_shifts[image_counter] - this->smooth_x_shifts[image_counter];
    //         diffy_vec[image_counter] = this->y_shifts[image_counter] + this->current_y_shifts[image_counter] - this->smooth_y_shifts[image_counter];
    //         diff_vec[image_counter]  = sqrtf(powf(diffx_vec[image_counter], 2) + powf(diffy_vec[image_counter], 2));
    //     }

    //     double stdx = calculateStdDev(diffx_vec);
    //     double stdy = calculateStdDev(diffy_vec);
    //     double std  = calculateStdDev(diff_vec);

    //     std::vector<int> outlier_ind = findOutlierUpperBoundIndices(diff_vec);

    //     return outlier_ind;
    // }

    void alignment_refine(bool use_smoothed_shifts = true);
    //      {

    //         std::vector<int> full_stack_index;
    //         for ( int i = 0; i < this->number_of_images; i++ ) {
    //             full_stack_index.push_back(i);
    //         }

    //         for ( iteration_counter = 1; iteration_counter <= this->max_iterations; iteration_counter++ ) {
    //             //	wxPrintf("Starting iteration number %li\n\n", iteration_counter);
    //             max_shift = -FLT_MAX;

    //             // make the current running average if necessary

    //             if ( number_of_frames_for_running_average > 1 ) {
    //                 Update_Running_Average( );
    //             }

    //             Calculate_Shifts(full_stack_index, this->unitless_bfactor);
    //             // smooth the shifts
    //             Smooth_Shifts( );

    //             if ( use_smoothed_shifts ) {
    //                 for ( int image_counter = 0; image_counter < number_of_images; image_counter++ ) {
    //                     this->current_x_shifts[image_counter] = this->smooth_x_shifts[image_counter];
    //                     this->current_y_shifts[image_counter] = this->smooth_y_shifts[image_counter];
    //                 }
    //             }
    //             else {
    //                 std::vector<int> outlier_ind;
    //                 outlier_ind = SearchOutlierIndices( );

    //                 Calculate_Shifts(outlier_ind, this->unitless_bfactor * 2);
    //                 Smooth_Shifts( ); //update the smooth shifts

    //                 //check back the outliers and std
    //                 std::vector<int> new_outlier_ind = SearchOutlierIndices( );
    //                 // subtract shift of the middle image from all images to keep things centred around it
    //                 // for ( int i = 0; i < outlier_ind.size( ); i++ ) {
    //                 //     wxPrintf("outlier 2 %d %f %f\n", outlier_ind[i], this->current_x_shifts[outlier_ind[i]], this->current_y_shifts[outlier_ind[i]]);
    //                 // }

    //                 std::vector<int> common_elements;
    //                 common_elements = FindCommonElements(outlier_ind, new_outlier_ind);
    //                 // we replace the commen elements with the old bfactor
    //                 Calculate_Shifts(common_elements, this->unitless_bfactor);
    //             }
    //             middle_image_x_shift = this->current_x_shifts[this->number_of_middle_image];
    //             middle_image_y_shift = this->current_y_shifts[this->number_of_middle_image];

    //             for ( int image_counter = 0; image_counter < number_of_images; image_counter++ ) {
    //                 this->current_x_shifts[image_counter] -= middle_image_x_shift;
    //                 this->current_y_shifts[image_counter] -= middle_image_y_shift;

    //                 total_shift = sqrt(pow(this->current_x_shifts[image_counter], 2) + pow(this->current_y_shifts[image_counter], 2));
    //                 if ( total_shift > max_shift )
    //                     max_shift = total_shift;
    //             }

    //             // actually shift the images, also add the subtracted shifts to the overall shifts
    // #pragma omp parallel for default(shared) num_threads(this->max_threads)
    //             for ( int image_counter = 0; image_counter < number_of_images; image_counter++ ) {
    //                 this->input_stack[image_counter].PhaseShift(this->current_x_shifts[image_counter], this->current_y_shifts[image_counter], 0.0);
    //                 this->x_shifts[image_counter] += this->current_x_shifts[image_counter];
    //                 this->y_shifts[image_counter] += this->current_y_shifts[image_counter];
    //             }

    //             // check to see if the convergence criteria have been reached and return if so
    //             // wxPrintf(" max shift convergence %f\n", this->max_shift_convergence_threshold);

    //             if ( iteration_counter >= max_iterations || max_shift <= this->max_shift_convergence_threshold ) {
    //                 // wxPrintf("returning, iteration = %li, max_shift = %f\n", iteration_counter, max_shift);
    //                 // wxPrintf(" max shift convergence %f\n", this->max_shift_convergence_threshold);
    //                 delete[] this->current_x_shifts;
    //                 delete[] this->current_y_shifts;
    //                 delete[] this->smooth_x_shifts;
    //                 delete[] this->smooth_y_shifts;
    //                 if ( number_of_frames_for_running_average > 1 ) {
    //                     delete[] this->running_average_stack;
    //                 }
    //                 // return;
    //                 break;
    //             }
    //             else {
    //                 // wxPrintf("Not. returning, iteration = %li, max_shift = %f\n", iteration_counter, max_shift);
    //             }

    //             // going to be doing another round so we need to make the new sum..
    //             Update_SumImage( );
    //         }

    //         if ( reverse_shift ) {
    // #pragma omp parallel for default(shared) num_threads(this->max_threads)
    //             for ( int image_counter = 0; image_counter < number_of_images; image_counter++ ) {
    //                 input_stack[image_counter].PhaseShift(-this->x_shifts[image_counter], -this->y_shifts[image_counter], 0.0);
    //             }
    //         }
    //     };
};
