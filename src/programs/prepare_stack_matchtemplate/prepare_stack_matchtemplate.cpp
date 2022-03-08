#include "../../core/core_headers.h"


class
MakeParticleStack : public MyApp
{
    public:

    bool DoCalculation();
    void DoInteractiveUserInput();

    private:
};

IMPLEMENT_APP(MakeParticleStack)


// override the DoInteractiveUserInput

void MakeParticleStack::DoInteractiveUserInput()
{

    wxString    input_mip_filename;
    wxString    input_image_filename;
    wxString    input_best_psi_filename;
    wxString    input_best_theta_filename;
    wxString    input_best_phi_filename;
    wxString    input_best_defocus_filename;
    wxString    output_star_filename;
    wxString    output_particle_stack_filename;
    wxString    xyz_coords_filename;

    float wanted_threshold;
    float min_peak_radius;
    float pixel_size = 1;
    float voltage_kV = 300.0;
    float spherical_aberration_mm = 2.7;
    float amplitude_contrast = 0.07;
    float average_defocus_1 = 5000.0;
    float average_defocus_2 = 5000.0;
    float average_defocus_angle = 0.0;
    int box_size = 256;
    int    result_number = 1;
    int mip_x_dimension = 0;
    int mip_y_dimension = 0;
    bool read_coordinates;

    UserInput *my_input = new UserInput("MakeParticleStack", 1.00);

    read_coordinates = my_input->GetYesNoFromUser("Read coordinates from file?", "Should the target coordinates be read from a file instead of search results?", "No");
    if (! read_coordinates)
    {
        input_mip_filename = my_input->GetFilenameFromUser("Input MIP file", "The file for saving the maximum intensity projection image", "mip.mrc", false);
        input_best_psi_filename = my_input->GetFilenameFromUser("Input psi file", "The file containing the best psi image", "psi.mrc", false);
        input_best_theta_filename = my_input->GetFilenameFromUser("Input theta file", "The file containing the best psi image", "theta.mrc", false);
        input_best_phi_filename = my_input->GetFilenameFromUser("Input phi file", "The file containing the best phi image", "phi.mrc", false);
        input_best_defocus_filename = my_input->GetFilenameFromUser("Input defocus file", "The file with the best defocus image", "defocus.mrc", true);
        xyz_coords_filename = my_input->GetFilenameFromUser("Output x,y,z coordinate file", "The file for saving the x,y,z coordinates of the found targets", "coordinates.txt", false);
        wanted_threshold = my_input->GetFloatFromUser("Peak threshold", "Peaks over this size will be taken", "7.5", 0.0);
        min_peak_radius = my_input->GetFloatFromUser("Min Peak Radius (px.)", "Essentially the minimum closeness for peaks", "10.0", 1.0);
        result_number = my_input->GetIntFromUser("Result number to process", "If input files contain results from several searches, which one should be used?", "1", 1);
    }
    else
    {
        mip_x_dimension = my_input->GetIntFromUser("X-dimension of original MIP", "The x-dimension of the MIP that contained the peaks listed in the input coordinate file", "5760", 100);
        mip_y_dimension = my_input->GetIntFromUser("Y-dimension of original MIP", "The y-dimension of the MIP that contained the peaks listed in the input coordinate file", "4092", 100);
        xyz_coords_filename = my_input->GetFilenameFromUser("Input x,y,z coordinate file", "The file containing the x,y,z coordinates of the found targets", "coordinates.txt", false);
    }
    input_image_filename = my_input->GetFilenameFromUser("Input image file", "The image that was searched", "image.mrc", false);
    output_star_filename = my_input->GetFilenameFromUser("Output star file", "The star file containing the particle alignment parameters", "particle_stack.star", false);
    output_particle_stack_filename = my_input->GetFilenameFromUser("Output particle stack", "The output image stack, containing the picked particles", "particle_stack.mrc", false);
    box_size = my_input->GetIntFromUser("Box size for particles (px.)", "The pixel dimensions of the box used to cut out the particles", "256", 10);
    pixel_size = my_input->GetFloatFromUser("Pixel size of image (A)", "Pixel size of input image in Angstroms", "1.0", 0.0);
    average_defocus_1 = my_input->GetFloatFromUser("Average defocus 1 (A)", "The average defocus estimated for the image in direction 1", "5000.0");
    average_defocus_2 = my_input->GetFloatFromUser("Average defocus 2 (A)", "The average defocus estimated for the image in direction 2", "5000.0");
    average_defocus_angle = my_input->GetFloatFromUser("Average defocus angle (deg)", "The average defocus angle estimated for the image", "0.0");
    voltage_kV = my_input->GetFloatFromUser("Beam energy (keV)", "The energy of the electron beam used to image the sample in kilo electron volts", "300.0", 0.0);
    spherical_aberration_mm = my_input->GetFloatFromUser("Spherical aberration (mm)", "Spherical aberration of the objective lens in millimeters", "2.7", 0.0);
    amplitude_contrast = my_input->GetFloatFromUser("Amplitude contrast", "Assumed amplitude contrast", "0.07", 0.0, 1.0);

    delete my_input;

//    my_current_job.Reset(14);
    my_current_job.ManualSetArguments("tttttttttifffffffffbiii",
                                                    input_mip_filename.ToUTF8().data(),
                                                    input_best_psi_filename.ToUTF8().data(),
                                                    input_best_theta_filename.ToUTF8().data(),
                                                    input_best_phi_filename.ToUTF8().data(),
                                                    input_best_defocus_filename.ToUTF8().data(),
                                                    xyz_coords_filename.ToUTF8().data(),
                                                    input_image_filename.ToUTF8().data(),
                                                    output_star_filename.ToUTF8().data(),
                                                    output_particle_stack_filename.ToUTF8().data(),
                                                    box_size,
                                                    pixel_size,
                                                    average_defocus_1,
                                                    average_defocus_2,
                                                    average_defocus_angle,
                                                    voltage_kV,
                                                    spherical_aberration_mm,
                                                    amplitude_contrast,
                                                    wanted_threshold,
                                                    min_peak_radius,
                                                    read_coordinates,
                                                    mip_x_dimension, mip_y_dimension,
                                                    result_number);
}

// override the do calculation method which will be what is actually run..

bool MakeParticleStack::DoCalculation()
{

    wxDateTime start_time = wxDateTime::Now();

    wxString    input_mip_filename = my_current_job.arguments[0].ReturnStringArgument();
    wxString    input_best_psi_filename = my_current_job.arguments[1].ReturnStringArgument();
    wxString    input_best_theta_filename = my_current_job.arguments[2].ReturnStringArgument();
    wxString    input_best_phi_filename = my_current_job.arguments[3].ReturnStringArgument();
    wxString    input_best_defocus_filename = my_current_job.arguments[4].ReturnStringArgument();
    wxString    xyz_coords_filename = my_current_job.arguments[5].ReturnStringArgument();
    wxString    input_image_filename = my_current_job.arguments[6].ReturnStringArgument();
    wxString    output_star_filename = my_current_job.arguments[7].ReturnStringArgument();
    wxString    output_particle_stack_filename = my_current_job.arguments[8].ReturnStringArgument();
    int         box_size = my_current_job.arguments[9].ReturnIntegerArgument();
    float        pixel_size = my_current_job.arguments[10].ReturnFloatArgument();
    float        average_defocus_1 = my_current_job.arguments[11].ReturnFloatArgument();
    float        average_defocus_2 = my_current_job.arguments[12].ReturnFloatArgument();
    float        average_defocus_angle = my_current_job.arguments[13].ReturnFloatArgument();
    float        voltage_kV = my_current_job.arguments[14].ReturnFloatArgument();
    float        spherical_aberration_mm = my_current_job.arguments[15].ReturnFloatArgument();
    float        amplitude_contrast = my_current_job.arguments[16].ReturnFloatArgument();
    float        wanted_threshold = my_current_job.arguments[17].ReturnFloatArgument();
    float        min_peak_radius = my_current_job.arguments[18].ReturnFloatArgument();
    bool         read_coordinates = my_current_job.arguments[19].ReturnBoolArgument();
    int         mip_x_dimension = my_current_job.arguments[20].ReturnIntegerArgument();
    int         mip_y_dimension = my_current_job.arguments[21].ReturnIntegerArgument();
    int         result_number = my_current_job.arguments[22].ReturnIntegerArgument();

    Image mip_image;
    Image psi_image;
    Image theta_image;
    Image phi_image;
    Image defocus_image;
    Image current_particle;
    Image micrograph;

    Peak current_peak;

    float current_phi;
    float current_theta;
    float current_psi;
    float current_defocus;
    float current_pixel_size = 1.0f;

    int number_of_peaks_found = 0;
    float sq_dist_x, sq_dist_y;
    float micrograph_mean;
    float variance;
    long address;
    long text_file_access_type;
    int i,j;

    float coordinates[8];
    if (read_coordinates) text_file_access_type = OPEN_TO_READ;
    else text_file_access_type = OPEN_TO_WRITE;
    NumericTextFile coordinate_file(xyz_coords_filename, text_file_access_type, 8);
    cisTEMParameterLine output_parameters;
    cisTEMParameters output_star_file;
    cisTEMParameters output_micrograph_star_file;

    // Preallocate space: number of peaks not known, so assume large enough number
    // FIXME all searches max out with hardcode 10K these all need to be set somehwere.
    output_star_file.PreallocateMemoryAndBlank(10000);
    output_micrograph_star_file.PreallocateMemoryAndBlank(10000);

    if (! read_coordinates) {
        coordinate_file.WriteCommentLine("         Psi          Theta            Phi              X              Y              Z      PixelSize           Peak");

        mip_image.QuickAndDirtyReadSlice(input_mip_filename.ToStdString(), result_number);
        psi_image.QuickAndDirtyReadSlice(input_best_psi_filename.ToStdString(), result_number);
        theta_image.QuickAndDirtyReadSlice(input_best_theta_filename.ToStdString(), result_number);
        phi_image.QuickAndDirtyReadSlice(input_best_phi_filename.ToStdString(), result_number);
        defocus_image.QuickAndDirtyReadSlice(input_best_defocus_filename.ToStdString(), result_number);
        mip_x_dimension = mip_image.logical_x_dimension;
        mip_y_dimension = mip_image.logical_y_dimension;

        min_peak_radius = powf(min_peak_radius, 2);
    }

    micrograph.QuickAndDirtyReadSlice(input_image_filename.ToStdString(), 1);
    micrograph_mean = micrograph.ReturnAverageOfRealValues();
//    address = 0;
//    for (j = 0; j < micrograph.logical_y_dimension; j++)
//    {
//        for (i = 0; i < micrograph.logical_x_dimension; i++)
//        {
//            address++;
//            micrograph.real_values[address] = i + 10000.0f * j;
//        }
//        address += micrograph.padding_jump_value;
//    }

    // assume square

    current_particle.Allocate(box_size, box_size, true);

    // loop until the found peak is below the threshold

    wxPrintf("\n");
    float centered_peak_x(0.f), centered_peak_y(0.f);
    while (1 == 1) {
        if (! read_coordinates) {
            // look for a peak..
            // match_template box/4+1 (unless override), refine_template 50, here box/2 : FIXME this is ridiculous
            current_peak = mip_image.FindPeakWithIntegerCoordinates(0.0, FLT_MAX, 50); ///box_size / 2 + 1);
            if (current_peak.value < wanted_threshold) break;

            // ok we have peak..

            number_of_peaks_found++;

            // get angles and mask out the local area so it won't be picked again..

            address = 0;

            centered_peak_x = current_peak.x * pixel_size;
            centered_peak_y = current_peak.y * pixel_size;
            current_peak.x = current_peak.x + mip_image.physical_address_of_box_center_x;
            current_peak.y = current_peak.y + mip_image.physical_address_of_box_center_y;

//            wxPrintf("Peak = %f, %f, %f : %f\n", current_peak.x, current_peak.y, current_peak.value);

            for ( j = 0; j < mip_y_dimension; j ++ ) {
                sq_dist_y = float(pow(j-current_peak.y, 2));
                for ( i = 0; i < mip_x_dimension; i ++ ) {
                    sq_dist_x = float(pow(i-current_peak.x,2));

                    // The square centered at the pixel
                    if ( sq_dist_x + sq_dist_y <= min_peak_radius ) {
                        mip_image.real_values[address] = -FLT_MAX;
                    }

                    if (sq_dist_x == 0 && sq_dist_y == 0) {
                        current_phi = phi_image.real_values[address];
                        current_theta = theta_image.real_values[address];
                        current_psi = psi_image.real_values[address];
                        current_defocus = defocus_image.real_values[address];
                    }

                    address++;
                }
                address += mip_image.padding_jump_value;
            }
            coordinates[0] = current_psi;
            coordinates[1] = current_theta;
            coordinates[2] = current_phi;
            coordinates[3] = current_peak.x * pixel_size;
            coordinates[4] = current_peak.y * pixel_size;
            coordinates[5] = current_defocus;
            coordinates[6] = current_pixel_size;
            coordinates[7] = current_peak.value;
            coordinate_file.WriteLine(coordinates);
        }
        else {
            coordinate_file.ReadLine(coordinates);
            number_of_peaks_found++;
            current_psi = coordinates[0];
            current_theta = coordinates[1];
            current_phi = coordinates[2];
            current_peak.x = coordinates[3] / pixel_size;
            current_peak.y = coordinates[4] / pixel_size;
            current_defocus = coordinates[5];
            current_pixel_size = coordinates[6];
            current_peak.value = coordinates[7];
        }

        output_parameters.SetAllToZero();
        output_parameters.position_in_stack = number_of_peaks_found;
        output_parameters.psi = current_psi;
        output_parameters.theta = current_theta;
        output_parameters.phi = current_phi;
        output_parameters.defocus_1 = average_defocus_1 + current_defocus;
        output_parameters.defocus_2 = average_defocus_2 + current_defocus;
        output_parameters.defocus_angle = average_defocus_angle;
        output_parameters.pixel_size = pixel_size;
        output_parameters.microscope_voltage_kv = voltage_kV;
        output_parameters.microscope_spherical_aberration_mm = spherical_aberration_mm;
        output_parameters.amplitude_contrast = amplitude_contrast;
        output_parameters.occupancy = 1.0f;
        output_parameters.sigma = 10.0f;
        output_parameters.logp = 5000.0f;
        output_parameters.score = 50.0f;
        output_parameters.image_is_active = 1;
        output_parameters.stack_filename = output_particle_stack_filename;
        output_parameters.original_image_filename = input_image_filename;

        output_star_file.all_parameters[number_of_peaks_found] = output_parameters;
        // Set the X/Y to  micograph positions, value is zero if coordinates were read in from a file.
        output_parameters.x_shift = centered_peak_x;
        output_parameters.y_shift = centered_peak_y;
        output_micrograph_star_file.all_parameters[number_of_peaks_found] = output_parameters;
        // Reset the values to zero
        output_parameters.x_shift = 0.f;
        output_parameters.y_shift = 0.f;

        wxPrintf("Peak %4i at x, y, psi, theta, phi, defocus, pixel size = %12.6f, %12.6f, %12.6f, %12.6f, %12.6f, %12.6f, %12.6f : %10.6f\n", number_of_peaks_found, current_peak.x * pixel_size, current_peak.y * pixel_size, current_psi, current_theta, current_phi, current_defocus, current_pixel_size, current_peak.value);

        micrograph.ClipInto(&current_particle, micrograph_mean, false, 1.0,
            int(current_peak.x - micrograph.physical_address_of_box_center_x),
            int(current_peak.y - micrograph.physical_address_of_box_center_y), 0);
//        micrograph.ClipInto(&current_particle, micrograph_mean, false, 1.0, int(current_peak.x * pixel_size), int(current_peak.y * pixel_size), 0);
//        micrograph.ClipInto(&current_particle, micrograph_mean, false, 1.0, int(- current_peak.x * pixel_size + current_particle.physical_address_of_box_center_x), int(- current_peak.y * pixel_size + current_particle.physical_address_of_box_center_y), 0);
        variance = current_particle.ReturnVarianceOfRealValues();
        if (variance == 0.0f) variance = 1.0f;
        current_particle.AddMultiplyConstant(-current_particle.ReturnAverageOfRealValuesOnEdges(), 1.0f / sqrtf(variance));
        if (number_of_peaks_found == 1) current_particle.QuickAndDirtyWriteSlice(output_particle_stack_filename.ToStdString(), number_of_peaks_found, true, pixel_size);
        else current_particle.QuickAndDirtyWriteSlice(output_particle_stack_filename.ToStdString(), number_of_peaks_found);

        if (read_coordinates && coordinate_file.number_of_lines == number_of_peaks_found) break;
    }

    output_star_file.WriteTocisTEMStarFile(output_star_filename, -1, -1, 1, number_of_peaks_found);
    output_micrograph_star_file.WriteTocisTEMStarFile(output_star_filename + "_micrograph.star", -1, -1, 1, number_of_peaks_found);

    if (is_running_locally == true) {

        wxPrintf("\nFound %i peaks.\n\n", number_of_peaks_found);
        wxPrintf("\nMake Particle Stack: Normal termination\n");
        wxDateTime finish_time = wxDateTime::Now();
        wxPrintf("Total Run Time : %s\n\n", finish_time.Subtract(start_time).Format("%Hh:%Mm:%Ss"));
    }

    return true;
}
