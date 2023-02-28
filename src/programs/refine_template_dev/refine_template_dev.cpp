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
    AnglesAndShifts* angles;
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

    return current_projection.FindPeakWithIntegerCoordinates( );
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
    cisTEMParameters         output_matches;
    std::string              current_image_filename;
    ImageFile                current_image_file;
    Image                    current_image;
    ImageFile                input_template_file;
    Image                    input_template;
    CTF                      ctf;
    Curve                    whitening_filter;
    Curve                    number_of_terms;
    Image                    windowed_particle;
    Image                    projection_filter;
    Image                    padded_reference;
    TemplateComparisonObject comparison_object;
    AnglesAndShifts          angles;
    Peak                     template_peak;

    RefineTemplateArguments arguments;
    float                   padding = 1.0f;
    arguments.recieve(my_current_job.arguments);

    input_matches.ReadFromcisTEMStarFile(arguments.input_starfile);
    output_matches.parameters_to_write.SetActiveParameters(PSI | THETA | PHI | X_SHIFT | Y_SHIFT | DEFOCUS_1 | DEFOCUS_2 | DEFOCUS_ANGLE | SCORE | MICROSCOPE_VOLTAGE | MICROSCOPE_CS | AMPLITUDE_CONTRAST | BEAM_TILT_X | BEAM_TILT_Y | IMAGE_SHIFT_X | IMAGE_SHIFT_Y | ORIGINAL_IMAGE_FILENAME);
    output_matches.PreallocateMemoryAndBlank(input_matches.all_parameters.GetCount( ));

    // Read template
    input_template_file.OpenFile(arguments.input_template, false);
    input_template.ReadSlices(&input_template_file, 1, input_template_file.ReturnNumberOfSlices( ));
    if ( padding != 1.0f ) {
        input_template.Resize(input_template.logical_x_dimension * padding, input_template.logical_y_dimension * padding, input_template.logical_z_dimension * padding, input_template.ReturnAverageOfRealValuesOnEdges( ));
    }
    input_template.ForwardFFT( );
    input_template.ZeroCentralPixel( );
    input_template.SwapRealSpaceQuadrants( );

    comparison_object.input_reconstruction = &input_template;
    comparison_object.windowed_particle    = &windowed_particle;
    comparison_object.projection_filter    = &projection_filter;
    comparison_object.angles               = &angles;

    // Iterate over all the particles in the input starfile
    for ( long match_id = 0; match_id < input_matches.all_parameters.GetCount( ); match_id++ ) {
        // If the current image is different from the previous image, then load the image
        if ( current_image_filename.compare(input_matches.all_parameters[match_id].original_image_filename.ToStdString( )) != 0 ) {
            current_image_file.OpenFile(input_matches.all_parameters[match_id].original_image_filename.ToStdString( ), false);
            current_image_filename = input_matches.all_parameters[match_id].original_image_filename.ToStdString( );
            MyDebugPrint("Reading image %s", input_matches.all_parameters[match_id].original_image_filename.ToStdString( ));
            current_image.ReadSlice(&current_image_file, 1);
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
        template_peak         = TemplateScore(&comparison_object);
        double starting_score = template_peak.value * sqrtf(projection_filter.logical_x_dimension * projection_filter.logical_y_dimension);
        double delta          = starting_score - input_matches.all_parameters[match_id].score;
        if ( delta < -2.0f ) {
            MyDebugPrint("Template peak of match %li delta is %f at %i, %i", match_id, delta, static_cast<int>(input_matches.all_parameters[match_id].x_shift / input_matches.all_parameters[match_id].pixel_size), static_cast<int>(input_matches.all_parameters[match_id].y_shift / input_matches.all_parameters[match_id].pixel_size));
        }
    }

    return true;
}
