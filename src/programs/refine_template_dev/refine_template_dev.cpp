#include "../../core/core_headers.h"
#include "./refine_template_dev.h"

class
        RefineTemplateDevApp : public MyApp {
  public:
    bool DoCalculation( );
    void DoInteractiveUserInput( );

  private:
};

class TemplateComparisonObject {
  public:
    Image *          input_reconstruction, *windowed_particle, *projection_filter;
    CTF*             orig_ctf;
    CTF*             optim_ctf;
    AnglesAndShifts* angles;
    Curve*           whitening_filter;
    Peak             result_peak;
};

// This is the function which will be minimized
Peak TemplateScore(void* scoring_parameters) {
    TemplateComparisonObject* comparison_object = reinterpret_cast<TemplateComparisonObject*>(scoring_parameters);
    Image                     current_projection;
    //	Peak box_peak;

    current_projection.Allocate(comparison_object->projection_filter->logical_x_dimension, comparison_object->projection_filter->logical_x_dimension, false);

    comparison_object->input_reconstruction->ExtractSlice(current_projection, *comparison_object->angles, 1.0f, false);
    current_projection.SwapRealSpaceQuadrants( );

    current_projection.MultiplyPixelWise(*comparison_object->projection_filter);
    current_projection.ZeroCentralPixel( );
    current_projection.DivideByConstant(sqrtf(current_projection.ReturnSumOfSquares( )));
#ifdef MKL
    // Use the MKL
    vmcMulByConj(current_projection.real_memory_allocated / 2, reinterpret_cast<MKL_Complex8*>(comparison_object->windowed_particle->complex_values), reinterpret_cast<MKL_Complex8*>(current_projection.complex_values), reinterpret_cast<MKL_Complex8*>(current_projection.complex_values), VML_EP | VML_FTZDAZ_ON | VML_ERRMODE_IGNORE);
#else
    for ( long pixel_counter = 0; pixel_counter < current_projection.real_memory_allocated / 2; pixel_counter++ ) {
        current_projection.complex_values[pixel_counter] = std::conj(current_projection.complex_values[pixel_counter]) * comparison_object->windowed_particle->complex_values[pixel_counter];
    }
#endif
    current_projection.BackwardFFT( );

    return current_projection.FindPeakWithIntegerCoordinates(0.0F, 5.0F);
}

float PerMatchObjectiveFunction(void* scoring_parameters, float* parameters_to_optimize) {
    TemplateComparisonObject* comparison_object = reinterpret_cast<TemplateComparisonObject*>(scoring_parameters);
    Image                     current_projection;
    //	Peak box_peak;
    comparison_object->angles->Init(parameters_to_optimize[0], parameters_to_optimize[1], parameters_to_optimize[2], 0.0f, 0.0f);
    comparison_object->optim_ctf->CopyFrom(*comparison_object->orig_ctf);
    comparison_object->optim_ctf->SetDefocus(comparison_object->orig_ctf->GetDefocus1( ) + parameters_to_optimize[3], comparison_object->orig_ctf->GetDefocus2( ) + parameters_to_optimize[3], comparison_object->orig_ctf->GetAstigmatismAzimuth( ));
    comparison_object->projection_filter->CalculateCTFImage(*(comparison_object->optim_ctf));
    comparison_object->projection_filter->ApplyCurveFilter(comparison_object->whitening_filter);
    comparison_object->result_peak = TemplateScore(comparison_object);
    return -(comparison_object->result_peak.value);
}

IMPLEMENT_APP(RefineTemplateDevApp)

// override the DoInteractiveUserInput

void RefineTemplateDevApp::DoInteractiveUserInput( ) {
    RefineTemplateArguments arguments;
    arguments.userinput( );
    arguments.setargument(my_current_job);
}

// override the do calculation method which will be what is actually run..

bool RefineTemplateDevApp::DoCalculation( ) {
    wxDateTime               start_time = wxDateTime::Now( );
    cisTEMParameters         input_matches;
    std::string              current_image_filename;
    ImageFile                current_image_file;
    Image                    current_image;
    ImageFile                input_template_file;
    Image                    input_template;
    CTF                      ctf;
    CTF                      optim_ctf;
    Curve                    whitening_filter;
    Curve                    number_of_terms;
    Image                    windowed_particle;
    Image                    projection_filter;
    Image                    padded_reference;
    TemplateComparisonObject comparison_object;
    AnglesAndShifts          angles;
    Peak                     template_peak;
    double                   starting_score;
    ConjugateGradient        conjugate_gradient_minizer;
    float                    cg_match_starting_values[4];
    float                    cg_match_accuracy[4];

    RefineTemplateArguments arguments;
    float                   padding = 1.0f;
    arguments.recieve(my_current_job.arguments);
    input_matches.ReadFromcisTEMStarFile(arguments.input_starfile);
    input_matches.parameters_to_write.SetActiveParameters(PSI | THETA | PHI | X_SHIFT | Y_SHIFT | DEFOCUS_1 | DEFOCUS_2 | DEFOCUS_ANGLE | SCORE | MICROSCOPE_VOLTAGE | MICROSCOPE_CS | AMPLITUDE_CONTRAST | BEAM_TILT_X | BEAM_TILT_Y | IMAGE_SHIFT_X | IMAGE_SHIFT_Y | ORIGINAL_IMAGE_FILENAME | SCORE_CHANGE | PIXEL_SIZE);

    // Read template
    input_template_file.OpenFile(arguments.input_template, false);
    input_template.ReadSlices(&input_template_file, 1, input_template_file.ReturnNumberOfSlices( ));
    if ( padding != 1.0f ) {
        input_template.Resize(input_template.logical_x_dimension * padding, input_template.logical_y_dimension * padding, input_template.logical_z_dimension * padding, input_template.ReturnAverageOfRealValuesOnEdges( ));
    }
    input_template.ForwardFFT( );
    input_template.ZeroCentralPixel( );
    input_template.SwapRealSpaceQuadrants( );

    if ( arguments.start_position < 1 )
        arguments.start_position = 1;
    if ( arguments.end_position > input_matches.all_parameters.GetCount( ) or arguments.end_position < 1 )
        arguments.end_position = input_matches.all_parameters.GetCount( );
    if ( arguments.num_threads > arguments.end_position - arguments.start_position + 1 )
        arguments.num_threads = arguments.end_position - arguments.start_position + 1;

#pragma omp parallel num_threads(arguments.num_threads) default(none) shared(arguments, input_matches, input_template_file, input_template) private(current_image_filename, current_image_file, current_image, whitening_filter, number_of_terms, ctf, optim_ctf, windowed_particle, projection_filter, padded_reference, angles, template_peak, starting_score, conjugate_gradient_minizer, cg_match_starting_values, cg_match_accuracy, comparison_object)
    // Iterate over all the particles in the input starfile
    {
        comparison_object.input_reconstruction = &input_template;
        comparison_object.windowed_particle    = &windowed_particle;
        comparison_object.projection_filter    = &projection_filter;
        comparison_object.angles               = &angles;
        comparison_object.optim_ctf            = &optim_ctf;
        comparison_object.orig_ctf             = &ctf;
        comparison_object.whitening_filter     = &whitening_filter;
#pragma omp for schedule(dynamic, 100)
        for ( long match_id = arguments.start_position - 1; match_id < arguments.end_position; match_id++ ) {
            // If the current image is different from the previous image, then load the image
            if ( current_image_filename.compare(input_matches.all_parameters[match_id].original_image_filename.ToStdString( )) != 0 ) {
                current_image_file.OpenFile(input_matches.all_parameters[match_id].original_image_filename.ToStdString( ), false);
                current_image_filename = input_matches.all_parameters[match_id].original_image_filename.ToStdString( );
                MyDebugPrint("Reading image %s", input_matches.all_parameters[match_id].original_image_filename.ToStdString( ));
                current_image.ReadSlice(&current_image_file, 1);
                MyDebugPrint("Image size is %i x %i", current_image.logical_x_dimension, current_image.logical_y_dimension);
                whitening_filter.SetupXAxis(0.0, 0.5 * sqrtf(2.0), int((current_image.logical_x_dimension / 2.0 + 1.0) * sqrtf(2.0) + 1.0));
                number_of_terms.SetupXAxis(0.0, 0.5 * sqrtf(2.0), int((current_image.logical_x_dimension / 2.0 + 1.0) * sqrtf(2.0) + 1.0));

                // remove outliers

                current_image.ReplaceOutliersWithMean(5.0f);
                current_image.ForwardFFT( );

                current_image.ZeroCentralPixel( );
                current_image.Compute1DPowerSpectrumCurve(&whitening_filter, &number_of_terms);
                whitening_filter.SquareRoot( );
                whitening_filter.Reciprocal( );
                whitening_filter.MultiplyByConstant(1.0f / whitening_filter.ReturnMaximumValue( ));

                current_image.ApplyCurveFilter(&whitening_filter);
                current_image.ZeroCentralPixel( );
                current_image.DivideByConstant(sqrt(current_image.ReturnSumOfSquares( )));
                current_image.BackwardFFT( );
            }

            // Setup the ctf
            ctf.Init(input_matches.all_parameters[match_id].microscope_voltage_kv,
                     input_matches.all_parameters[match_id].microscope_spherical_aberration_mm,
                     input_matches.all_parameters[match_id].amplitude_contrast,
                     input_matches.all_parameters[match_id].defocus_1,
                     input_matches.all_parameters[match_id].defocus_2,
                     input_matches.all_parameters[match_id].defocus_angle,
                     0.0,
                     0.0,
                     0.0,
                     input_matches.all_parameters[match_id].pixel_size,
                     deg_2_rad(input_matches.all_parameters[match_id].phase_shift));
            optim_ctf.CopyFrom(ctf);

            windowed_particle.Allocate(input_template_file.ReturnXSize( ), input_template_file.ReturnXSize( ), true);
            projection_filter.Allocate(input_template_file.ReturnXSize( ), input_template_file.ReturnXSize( ), false);
            padded_reference.CopyFrom(&current_image);
            //padded_reference.RealSpaceIntegerShift(input_matches.all_parameters[match_id].x_shift / input_matches.all_parameters[match_id].pixel_size, input_matches.all_parameters[match_id].y_shift / input_matches.all_parameters[match_id].pixel_size);
            padded_reference.ClipInto(&windowed_particle, 0.0f, false, 1.0f, input_matches.all_parameters[match_id].x_shift / input_matches.all_parameters[match_id].pixel_size - current_image.physical_address_of_box_center_x, input_matches.all_parameters[match_id].y_shift / input_matches.all_parameters[match_id].pixel_size - current_image.physical_address_of_box_center_y);

            windowed_particle.ForwardFFT( );
            windowed_particle.SwapRealSpaceQuadrants( );

            angles.Init(input_matches.all_parameters[match_id].phi, input_matches.all_parameters[match_id].theta, input_matches.all_parameters[match_id].psi, 0.0, 0.0);
            projection_filter.CalculateCTFImage(ctf);
            projection_filter.ApplyCurveFilter(&whitening_filter);
            template_peak  = TemplateScore(&comparison_object);
            starting_score = template_peak.value * sqrtf(projection_filter.logical_x_dimension * projection_filter.logical_y_dimension);
            //input_matches.all_parameters[match_id].score             = starting_score;
            input_matches.all_parameters[match_id].position_in_stack = 1;
            cg_match_starting_values[0]                              = input_matches.all_parameters[match_id].phi;
            cg_match_starting_values[1]                              = input_matches.all_parameters[match_id].theta;
            cg_match_starting_values[2]                              = input_matches.all_parameters[match_id].psi;
            cg_match_starting_values[3]                              = 0.0f;
            cg_match_accuracy[0]                                     = 1.0f;
            cg_match_accuracy[1]                                     = 1.0f;
            cg_match_accuracy[2]                                     = 1.0f;
            cg_match_accuracy[3]                                     = 0.1f;
            float  x                                                 = conjugate_gradient_minizer.Init(&PerMatchObjectiveFunction, &comparison_object, 4, cg_match_starting_values, cg_match_accuracy);
            float  y                                                 = conjugate_gradient_minizer.Run( );
            float* result_array                                      = conjugate_gradient_minizer.GetPointerToBestValues( );
            MyDebugPrint("Starting score = %f, ending score = %f", starting_score, -y * sqrtf(projection_filter.logical_x_dimension * projection_filter.logical_y_dimension));
            MyDebugPrint("Phi: %f -> %f, Theta %f -> %f, Psi %f -> %f", input_matches.all_parameters[match_id].phi, conjugate_gradient_minizer.GetBestValue(0), input_matches.all_parameters[match_id].theta, conjugate_gradient_minizer.GetBestValue(1), input_matches.all_parameters[match_id].psi, conjugate_gradient_minizer.GetBestValue(2));
            MyDebugPrint("DefocusChange: %f", conjugate_gradient_minizer.GetBestValue(3));
            PerMatchObjectiveFunction(&comparison_object, result_array);
            MyDebugPrint("X shift: %f, Y shift: %f", comparison_object.result_peak.x, comparison_object.result_peak.y);
            input_matches.all_parameters[match_id].phi          = result_array[0];
            input_matches.all_parameters[match_id].theta        = result_array[1];
            input_matches.all_parameters[match_id].psi          = result_array[2];
            input_matches.all_parameters[match_id].score        = -y * sqrtf(projection_filter.logical_x_dimension * projection_filter.logical_y_dimension);
            input_matches.all_parameters[match_id].score_change = input_matches.all_parameters[match_id].score - starting_score;
            input_matches.all_parameters[match_id].x_shift += comparison_object.result_peak.x * input_matches.all_parameters[match_id].pixel_size;
            input_matches.all_parameters[match_id].y_shift += comparison_object.result_peak.y * input_matches.all_parameters[match_id].pixel_size;
            input_matches.all_parameters[match_id].defocus_1 += result_array[3];
            input_matches.all_parameters[match_id].defocus_2 += result_array[3];
        }
    }
    input_matches.WriteTocisTEMStarFile(arguments.output_starfile, -1, -1, -1, -1);

    return true;
}
