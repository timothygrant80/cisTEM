#include "../../core/core_headers.h"

class
        Project3DApp : public MyApp {

  public:
    bool DoCalculation( );
    void DoInteractiveUserInput( );

  private:
};

IMPLEMENT_APP(Project3DApp)

// override the DoInteractiveUserInput

void Project3DApp::DoInteractiveUserInput( ) {
    wxString input_star_filename;
    wxString input_reconstruction;
    wxString ouput_projection_stack;
    int      first_particle = 1;
    int      last_particle  = 0;
    float    pixel_size     = 1;
    //	float		voltage_kV = 300.0;
    //	float		spherical_aberration_mm = 2.7;
    //	float		amplitude_contrast = 0.07;
    //	float		beam_tilt_x;
    //	float		beam_tilt_y;
    //	float		particle_shift_x;
    //	float		particle_shift_y;
    float    mask_radius = 100.0;
    float    padding     = 1.0;
    float    wanted_SNR  = 1.0;
    wxString my_symmetry = "C1";
    bool     apply_CTF;
    bool     apply_shifts;
    bool     apply_mask;
    bool     add_noise;
    bool     project_based_on_star;
    float    angular_step;

    int max_threads;

    UserInput* my_input = new UserInput("Project3D", 1.0);

    input_reconstruction  = my_input->GetFilenameFromUser("Input reconstruction", "The 3D reconstruction from which projections are calculated", "my_reconstruction.mrc", true);
    project_based_on_star = my_input->GetYesNoFromUser("Angles from input .star file?", "If yes, angles and shifts will be obtained from an input star file, if not they will be made on a grid", "Yes");

    if ( project_based_on_star == true ) {
        input_star_filename = my_input->GetFilenameFromUser("Input cisTEM star filename", "The input parameter file, containing your particle alignment parameters", "my_parameters.star", true);
        first_particle      = my_input->GetIntFromUser("First particle to project (0 = first in list)", "The first particle in the stack for which a projection should be calculated", "1", 0);
        last_particle       = my_input->GetIntFromUser("Last particle to project (0 = last in list)", "The last particle in the stack for which a projection should be calculated", "0", 0);
        apply_CTF           = my_input->GetYesNoFromUser("Apply CTF", "Should the CTF be applied to the output projections?", "No");
        apply_shifts        = my_input->GetYesNoFromUser("Apply shifts", "Should the particle translations be applied to the output projections?", "No");
    }
    else {
        angular_step = my_input->GetFloatFromUser("Angular step for projection", "Angular step size for grid projection", "0.0", 0.1);
        apply_CTF    = false;
        apply_shifts = false;
    }

    ouput_projection_stack = my_input->GetFilenameFromUser("Output projection stack", "The output image stack, containing the 2D projections", "my_projection_stack.mrc", false);
    pixel_size             = my_input->GetFloatFromUser("Pixel size of reconstruction (A)", "Pixel size of input images in Angstroms", "1.0", 0.0);

    //	voltage_kV = my_input->GetFloatFromUser("Beam energy (keV)", "The energy of the electron beam used to image the sample in kilo electron volts", "300.0", 0.0);
    //	spherical_aberration_mm = my_input->GetFloatFromUser("Spherical aberration (mm)", "Spherical aberration of the objective lens in millimeters", "2.7", 0.0);
    //	amplitude_contrast = my_input->GetFloatFromUser("Amplitude contrast", "Assumed amplitude contrast", "0.07", 0.0, 1.0);
    //	beam_tilt_x = my_input->GetFloatFromUser("Beam tilt along x [mrad]", "Beam tilt to be applied along the x axis in mrad", "0.0", -100.0, 100.0);
    //	beam_tilt_y = my_input->GetFloatFromUser("Beam tilt along y [mrad]", "Beam tilt to be applied along the y axis in mrad", "0.0", -100.0, 100.0);
    //	particle_shift_x = my_input->GetFloatFromUser("Particle shift along x (A)", "Average particle shift along the x axis as a result of beam tilt in A", "0.0", -1.0, 1.0);
    //	particle_shift_y = my_input->GetFloatFromUser("Particle shift along y (A)", "Average particle shift along the y axis as a result of beam tilt in A", "0.0", -1.0, 1.0);
    mask_radius = my_input->GetFloatFromUser("Mask radius (A)", "Radius of a circular mask to be applied to the final reconstruction in Angstroms", "100.0", 0.0);
    wanted_SNR  = my_input->GetFloatFromUser("Wanted SNR", "The ratio of signal to noise variance after adding Gaussian noise and before masking", "1.0", 0.0);
    padding     = my_input->GetFloatFromUser("Padding factor", "Factor determining how much the input volume is padded to improve projections", "1.0", 1.0);
    my_symmetry = my_input->GetSymmetryFromUser("Particle symmetry", "The assumed symmetry of the particle to be reconstructed", "C1");
    apply_mask  = my_input->GetYesNoFromUser("Apply mask", "Should the particles be masked with the circular mask?", "No");
    add_noise   = my_input->GetYesNoFromUser("Add noise", "Should the Gaussian noise be added?", "No");

#ifdef _OPENMP
    max_threads = my_input->GetIntFromUser("Max. threads to use for calculation", "When threading, what is the max threads to run", "1", 1);
#else
    max_threads = 1;
#endif

    delete my_input;

    //	my_current_job.Reset(14);
    my_current_job.ManualSetArguments("tttiifffftbbbbbfi", input_star_filename.ToUTF8( ).data( ),
                                      input_reconstruction.ToUTF8( ).data( ),
                                      ouput_projection_stack.ToUTF8( ).data( ),
                                      first_particle, last_particle,
                                      pixel_size, mask_radius, wanted_SNR, padding,
                                      my_symmetry.ToUTF8( ).data( ),
                                      apply_CTF, apply_shifts, apply_mask, add_noise, project_based_on_star, angular_step, max_threads);
}

// override the do calculation method which will be what is actually run..

bool Project3DApp::DoCalculation( ) {
    wxString input_star_filename     = my_current_job.arguments[0].ReturnStringArgument( );
    wxString input_reconstruction    = my_current_job.arguments[1].ReturnStringArgument( );
    wxString output_projection_stack = my_current_job.arguments[2].ReturnStringArgument( );
    int      first_particle          = my_current_job.arguments[3].ReturnIntegerArgument( );
    int      last_particle           = my_current_job.arguments[4].ReturnIntegerArgument( );
    float    pixel_size              = my_current_job.arguments[5].ReturnFloatArgument( );
    //	float    voltage_kV							= my_current_job.arguments[6].ReturnFloatArgument();
    //	float 	 spherical_aberration_mm			= my_current_job.arguments[7].ReturnFloatArgument();
    //	float    amplitude_contrast					= my_current_job.arguments[8].ReturnFloatArgument();
    //	float    beam_tilt_x						= my_current_job.arguments[9].ReturnFloatArgument();
    //	float    beam_tilt_y						= my_current_job.arguments[10].ReturnFloatArgument();
    //	float    particle_shift_x					= my_current_job.arguments[11].ReturnFloatArgument();
    //	float    particle_shift_y					= my_current_job.arguments[12].ReturnFloatArgument();
    float    mask_radius           = my_current_job.arguments[6].ReturnFloatArgument( );
    float    wanted_SNR            = my_current_job.arguments[7].ReturnFloatArgument( );
    float    padding               = my_current_job.arguments[8].ReturnFloatArgument( );
    wxString my_symmetry           = my_current_job.arguments[9].ReturnStringArgument( );
    bool     apply_CTF             = my_current_job.arguments[10].ReturnBoolArgument( );
    bool     apply_shifts          = my_current_job.arguments[11].ReturnBoolArgument( );
    bool     apply_mask            = my_current_job.arguments[12].ReturnBoolArgument( );
    bool     add_noise             = my_current_job.arguments[13].ReturnBoolArgument( );
    bool     project_based_on_star = my_current_job.arguments[14].ReturnBoolArgument( );
    float    angular_step          = my_current_job.arguments[15].ReturnFloatArgument( );
    int      max_threads           = my_current_job.arguments[16].ReturnIntegerArgument( );

    Image               projection_image;
    Image               final_image;
    ReconstructedVolume input_3d;
    Image               projection_3d;

    int image_counter = 0;
    int number_of_projections_to_calculate;

    int        current_image;
    float      average_score = 0.0;
    float      average_sigma = 0.0;
    float      variance;
    float      mask_falloff = 10.0;
    wxArrayInt lines_to_process;

    cisTEMParameterLine input_parameters;

    cisTEMParameters input_star_file;
    if ( project_based_on_star == true )
        input_star_file.ReadFromcisTEMStarFile(input_star_filename);

    EulerSearch  global_euler_search;
    ParameterMap parameter_map; // needed for euler search init
    parameter_map.SetAllTrue( );

    if ( project_based_on_star == false ) {
        // this line sets up a grid, however it was designed for refinement and uses a "test" mirror variable.  If set to true,
        // then the max theta is 90 and the mirror of each projection is checked in the refinement.  We want to overide this so that
        // we get the theta > 90 projections.  So first we do it normally :-

        global_euler_search.InitGrid(my_symmetry, angular_step, 0.0f, 0.0f, 360, 180, 180, pixel_size / 8, parameter_map, 20);

        // Then, if the symmetry is C, we set the theta max to 180 and recalculate the grid search positions.

        if ( my_symmetry.StartsWith("C") ) {
            if ( global_euler_search.test_mirror == true ) // otherwise the theta max is set to 90.0 and test_mirror is set to true.  However, I don't want to have to test the mirrors.
            {
                global_euler_search.theta_max = 180.0f;
            }
        }

        global_euler_search.CalculateGridSearchPositions(false);
    }

    MRCFile         input_file(input_reconstruction.ToStdString( ), false);
    MRCFile         output_file(output_projection_stack.ToStdString( ), true);
    AnglesAndShifts my_parameters;
    CTF             my_ctf;

    if ( (input_file.ReturnXSize( ) != input_file.ReturnYSize( )) || (input_file.ReturnXSize( ) != input_file.ReturnZSize( )) ) {
        MyPrintWithDetails("Error: Input reconstruction is not cubic\n");
        DEBUG_ABORT;
    }
    if ( last_particle < first_particle && last_particle != 0 ) {
        MyPrintWithDetails("Error: Number of last particle to refine smaller than number of first particle to refine\n");
        DEBUG_ABORT;
    }

    //	beam_tilt_x /= 1000.0f;
    //	beam_tilt_y /= 1000.0f;

    input_3d.InitWithDimensions(input_file.ReturnXSize( ), input_file.ReturnYSize( ), input_file.ReturnZSize( ), pixel_size, my_symmetry);
    input_3d.density_map->ReadSlices(&input_file, 1, input_3d.density_map->logical_z_dimension);
    //	input_3d.density_map->AddConstant(- input_3d.density_map->ReturnAverageOfRealValuesOnEdges());
    if ( padding != 1.0 ) {
        input_3d.density_map->Resize(input_3d.density_map->logical_x_dimension * padding, input_3d.density_map->logical_y_dimension * padding, input_3d.density_map->logical_z_dimension * padding, input_3d.density_map->ReturnAverageOfRealValuesOnEdges( ));
    }
    input_3d.mask_radius = mask_radius;
    input_3d.density_map->CorrectSinc(mask_radius / pixel_size);
    input_3d.PrepareForProjections(0.0, 2.0 * pixel_size);
    //input_3d.density_map->ForwardFFT();
    //input_3d.density_map->SwapRealSpaceQuadrants();

    // write first image to output file, so threads can write out of sequence..

    //	if (max_threads > 1)
    //	{
    //		projection_image.Allocate(input_3d.density_map->logical_x_dimension, input_3d.density_map->logical_y_dimension, false);
    //		projection_image.WriteSlice(&output_file, 1);
    //	}

    // Read whole parameter file to work out average image_sigma_noise and average score

    if ( project_based_on_star == true ) {
        average_sigma = input_star_file.ReturnAverageSigma( );
        average_score = input_star_file.ReturnAverageScore( );
        if ( first_particle == 0 )
            first_particle = 1;
        if ( last_particle == 0 )
            last_particle = input_star_file.ReturnMaxPositionInStack( );

        for ( current_image = 0; current_image < input_star_file.ReturnNumberofLines( ); current_image++ ) {
            if ( input_star_file.ReturnPositionInStack(current_image) >= first_particle && input_star_file.ReturnPositionInStack(current_image) <= last_particle ) {
                lines_to_process.Add(current_image);
            }
        }

        wxPrintf("\nAverage sigma noise = %f, average score = %f\nNumber of projections to calculate = %li\n\n", average_sigma, average_score, lines_to_process.GetCount( ));
    }

    if ( project_based_on_star == true )
        number_of_projections_to_calculate = lines_to_process.GetCount( );
    else
        number_of_projections_to_calculate = global_euler_search.number_of_search_positions;

    ProgressBar* my_progress = new ProgressBar(number_of_projections_to_calculate);
    projection_3d.CopyFrom(input_3d.density_map);

#pragma omp parallel num_threads(max_threads) default(none) shared(global_random_number_generator, input_star_file, first_particle, last_particle, apply_CTF, apply_shifts, \
                                                                   pixel_size, output_file, add_noise, wanted_SNR, apply_mask, mask_radius, my_progress, lines_to_process, image_counter, projection_3d, input_file, global_euler_search, number_of_projections_to_calculate, project_based_on_star) private(current_image, input_parameters, my_parameters, my_ctf, projection_image, final_image, variance)
    {

        projection_image.Allocate(input_file.ReturnXSize( ), input_file.ReturnYSize( ), false);
        final_image.Allocate(input_file.ReturnXSize( ), input_file.ReturnYSize( ), true);
        RandomNumberGenerator local_random_generator(int(fabsf(global_random_number_generator.GetUniformRandom( ) * 50000)), true);

#pragma omp for ordered schedule(static, 1)
        for ( current_image = 0; current_image < number_of_projections_to_calculate; current_image++ ) {
            if ( project_based_on_star == true ) {
                input_parameters = input_star_file.ReturnLine(lines_to_process[current_image]);
                my_parameters.Init(input_parameters.phi, input_parameters.theta, input_parameters.psi, input_parameters.x_shift, input_parameters.y_shift);
                my_ctf.Init(input_parameters.microscope_voltage_kv, input_parameters.microscope_spherical_aberration_mm, input_parameters.amplitude_contrast, input_parameters.defocus_1, input_parameters.defocus_2, input_parameters.defocus_angle, 0.0, 0.0, 0.0, pixel_size, input_parameters.phase_shift, -input_parameters.beam_tilt_x / 1000.0f, -input_parameters.beam_tilt_y / 1000.0f, -input_parameters.image_shift_x, -input_parameters.image_shift_y);
            }
            else {
                my_parameters.Init(global_euler_search.list_of_search_parameters[current_image][0], global_euler_search.list_of_search_parameters[current_image][1], 0.0, 0.0, 0.0f);
            }

            projection_3d.ExtractSlice(projection_image, my_parameters);
            projection_image.complex_values[0] = projection_3d.complex_values[0];

            if ( apply_CTF )
                projection_image.ApplyCTF(my_ctf, false, true);
            if ( apply_shifts )
                projection_image.PhaseShift(input_parameters.x_shift / pixel_size, input_parameters.y_shift / pixel_size);

            projection_image.SwapRealSpaceQuadrants( );
            projection_image.BackwardFFT( );

            if ( project_based_on_star == true )
                projection_image.ChangePixelSize(&final_image, input_parameters.pixel_size / pixel_size, 0.001f);
            else
                final_image.CopyFrom(&projection_image);

            if ( add_noise && wanted_SNR != 0.0 ) {
                variance = final_image.ReturnVarianceOfRealValues( );
                final_image.AddGaussianNoise(sqrtf(variance / wanted_SNR), &local_random_generator);
            }

            if ( apply_mask )
                final_image.CosineMask(mask_radius / input_parameters.pixel_size, 6.0);

#pragma omp ordered
            final_image.WriteSlice(&output_file, current_image + 1);

#pragma omp atomic
            image_counter++;

            if ( is_running_locally == true && ReturnThreadNumberOfCurrentThread( ) == 0 )
                my_progress->Update(image_counter);
        }
    }
    // end omp
    delete my_progress;

    wxPrintf("\n\nProject3D: Normal termination\n\n");

    return true;
}
