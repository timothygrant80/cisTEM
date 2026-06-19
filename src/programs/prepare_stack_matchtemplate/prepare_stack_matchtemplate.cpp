#include "../../core/core_headers.h"

#include "../match_template/template_matching_peak_extractor.h"

class
        MakeParticleStack : public MyApp {
  public:
    bool DoCalculation( );
    void DoInteractiveUserInput( );
    void AddCommandLineOptions( ) override;

  private:
};

IMPLEMENT_APP(MakeParticleStack)

// override the DoInteractiveUserInput

void MakeParticleStack::DoInteractiveUserInput( ) {

    wxString input_mip_filename;
    wxString input_image_filename;
    wxString input_best_psi_filename;
    wxString input_best_theta_filename;
    wxString input_best_phi_filename;
    wxString input_best_defocus_filename;
    wxString output_star_filename;
    wxString output_particle_stack_filename;
    wxString xyz_coords_filename;

    float wanted_threshold;
    float min_peak_radius;
    float pixel_size              = 1;
    float voltage_kV              = 300.0;
    float spherical_aberration_mm = 2.7;
    float amplitude_contrast      = 0.07;
    float average_defocus_1       = 5000.0;
    float average_defocus_2       = 5000.0;
    float average_defocus_angle   = 0.0;
    int   box_size                = 256;
    int   result_number           = 1;
    int   mip_x_dimension         = 0;
    int   mip_y_dimension         = 0;
    bool  read_coordinates;

    UserInput* my_input = new UserInput("MakeParticleStack", 1.00);

    read_coordinates = my_input->GetYesNoFromUser("Read coordinates from file?", "Should the target coordinates be read from a file instead of search results?", "No");
    if ( ! read_coordinates ) {
        input_mip_filename          = my_input->GetFilenameFromUser("Input MIP file", "The file for saving the maximum intensity projection image", "mip.mrc", false);
        input_best_psi_filename     = my_input->GetFilenameFromUser("Input psi file", "The file containing the best psi image", "psi.mrc", false);
        input_best_theta_filename   = my_input->GetFilenameFromUser("Input theta file", "The file containing the best psi image", "theta.mrc", false);
        input_best_phi_filename     = my_input->GetFilenameFromUser("Input phi file", "The file containing the best phi image", "phi.mrc", false);
        input_best_defocus_filename = my_input->GetFilenameFromUser("Input defocus file", "The file with the best defocus image", "defocus.mrc", true);
        xyz_coords_filename         = my_input->GetFilenameFromUser("Output x,y,z coordinate file", "The file for saving the x,y,z coordinates of the found targets", "coordinates.txt", false);
        wanted_threshold            = my_input->GetFloatFromUser("Peak threshold", "Peaks over this size will be taken", "7.5", 0.0);
        min_peak_radius             = my_input->GetFloatFromUser("Min Peak Radius (px.)", "Essentially the minimum closeness for peaks", "10.0", 1.0);
        result_number               = my_input->GetIntFromUser("Result number to process", "If input files contain results from several searches, which one should be used?", "1", 1);
    }
    else {
        mip_x_dimension     = my_input->GetIntFromUser("X-dimension of original MIP", "The x-dimension of the MIP that contained the peaks listed in the input coordinate file", "5760", 100);
        mip_y_dimension     = my_input->GetIntFromUser("Y-dimension of original MIP", "The y-dimension of the MIP that contained the peaks listed in the input coordinate file", "4092", 100);
        xyz_coords_filename = my_input->GetFilenameFromUser("Input x,y,z coordinate file", "The file containing the x,y,z coordinates of the found targets", "coordinates.txt", false);
    }
    input_image_filename              = my_input->GetFilenameFromUser("Input image file", "The image that was searched", "image.mrc", false);
    output_star_filename              = my_input->GetFilenameFromUser("Output star file", "The star file containing the particle alignment parameters", "particle_stack.star", false);
    output_particle_stack_filename    = my_input->GetFilenameFromUser("Output particle stack", "The output image stack, containing the picked particles", "particle_stack.mrc", false);
    box_size                          = my_input->GetIntFromUser("Box size for particles (px.)", "The pixel dimensions of the box used to cut out the particles", "256", 10);
    pixel_size                        = my_input->GetFloatFromUser("Pixel size of image (A)", "Pixel size of input image in Angstroms", "1.0", 0.0);
    average_defocus_1                 = my_input->GetFloatFromUser("Average defocus 1 (A)", "The average defocus estimated for the image in direction 1", "5000.0");
    average_defocus_2                 = my_input->GetFloatFromUser("Average defocus 2 (A)", "The average defocus estimated for the image in direction 2", "5000.0");
    average_defocus_angle             = my_input->GetFloatFromUser("Average defocus angle (deg)", "The average defocus angle estimated for the image", "0.0");
    voltage_kV                        = my_input->GetFloatFromUser("Beam energy (keV)", "The energy of the electron beam used to image the sample in kilo electron volts", "300.0", 0.0);
    spherical_aberration_mm           = my_input->GetFloatFromUser("Spherical aberration (mm)", "Spherical aberration of the objective lens in millimeters", "2.7", 0.0);
    amplitude_contrast                = my_input->GetFloatFromUser("Amplitude contrast", "Assumed amplitude contrast", "0.07", 0.0, 1.0);
    bool use_peak_sampling_correction = my_input->GetYesNoFromUser("Use peak sampling correction", "Apply peak height sampling correction", "Yes");

    delete my_input;

    //	my_current_job.Reset(14);
    my_current_job.ManualSetArguments("tttttttttifffffffffbiiib",
                                      input_mip_filename.ToUTF8( ).data( ),
                                      input_best_psi_filename.ToUTF8( ).data( ),
                                      input_best_theta_filename.ToUTF8( ).data( ),
                                      input_best_phi_filename.ToUTF8( ).data( ),
                                      input_best_defocus_filename.ToUTF8( ).data( ),
                                      xyz_coords_filename.ToUTF8( ).data( ),
                                      input_image_filename.ToUTF8( ).data( ),
                                      output_star_filename.ToUTF8( ).data( ),
                                      output_particle_stack_filename.ToUTF8( ).data( ),
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
                                      result_number,
                                      use_peak_sampling_correction);
}

void MakeParticleStack::AddCommandLineOptions( ) {
}

// override the do calculation method which will be what is actually run..

bool MakeParticleStack::DoCalculation( ) {

    wxDateTime start_time = wxDateTime::Now( );

    wxString input_mip_filename             = my_current_job.arguments[0].ReturnStringArgument( );
    wxString input_best_psi_filename        = my_current_job.arguments[1].ReturnStringArgument( );
    wxString input_best_theta_filename      = my_current_job.arguments[2].ReturnStringArgument( );
    wxString input_best_phi_filename        = my_current_job.arguments[3].ReturnStringArgument( );
    wxString input_best_defocus_filename    = my_current_job.arguments[4].ReturnStringArgument( );
    wxString xyz_coords_filename            = my_current_job.arguments[5].ReturnStringArgument( );
    wxString input_image_filename           = my_current_job.arguments[6].ReturnStringArgument( );
    wxString output_star_filename           = my_current_job.arguments[7].ReturnStringArgument( );
    wxString output_particle_stack_filename = my_current_job.arguments[8].ReturnStringArgument( );
    int      box_size                       = my_current_job.arguments[9].ReturnIntegerArgument( );
    float    pixel_size                     = my_current_job.arguments[10].ReturnFloatArgument( );
    float    average_defocus_1              = my_current_job.arguments[11].ReturnFloatArgument( );
    float    average_defocus_2              = my_current_job.arguments[12].ReturnFloatArgument( );
    float    average_defocus_angle          = my_current_job.arguments[13].ReturnFloatArgument( );
    float    voltage_kV                     = my_current_job.arguments[14].ReturnFloatArgument( );
    float    spherical_aberration_mm        = my_current_job.arguments[15].ReturnFloatArgument( );
    float    amplitude_contrast             = my_current_job.arguments[16].ReturnFloatArgument( );
    float    wanted_threshold               = my_current_job.arguments[17].ReturnFloatArgument( );
    float    min_peak_radius                = my_current_job.arguments[18].ReturnFloatArgument( );
    bool     read_coordinates               = my_current_job.arguments[19].ReturnBoolArgument( );
    int      mip_x_dimension                = my_current_job.arguments[20].ReturnIntegerArgument( );
    int      mip_y_dimension                = my_current_job.arguments[21].ReturnIntegerArgument( );
    int      result_number                  = my_current_job.arguments[22].ReturnIntegerArgument( );
    bool     use_peak_sampling_correction   = my_current_job.arguments[23].ReturnBoolArgument( );

    Image mip_image;
    Image psi_image;
    Image theta_image;
    Image phi_image;
    Image defocus_image;
    Image micrograph;

    int  number_of_peaks_found = 0;
    long text_file_access_type;

    float coordinates[8];
    if ( read_coordinates )
        text_file_access_type = OPEN_TO_READ;
    else
        text_file_access_type = OPEN_TO_WRITE;
    NumericTextFile     coordinate_file(xyz_coords_filename, text_file_access_type, 8);
    cisTEMParameterLine output_parameters;
    cisTEMParameters    output_star_file;

    // Preallocate space: number of peaks not known, so assume large enough number
    output_star_file.PreallocateMemoryAndBlank(cistem::match_template::MAX_ALLOWED_NUMBER_OF_PEAKS);

    float search_pixel_size = pixel_size; // default to input pixel size

    if ( ! read_coordinates ) {
        // Read search pixel size from MIP header and load all result images
        ImageFile mip_file(input_mip_filename.ToStdString( ), false);
        search_pixel_size = mip_file.ReturnPixelSize( );

        mip_image.QuickAndDirtyReadSlice(input_mip_filename.ToStdString( ), result_number);
        psi_image.QuickAndDirtyReadSlice(input_best_psi_filename.ToStdString( ), result_number);
        theta_image.QuickAndDirtyReadSlice(input_best_theta_filename.ToStdString( ), result_number);
        phi_image.QuickAndDirtyReadSlice(input_best_phi_filename.ToStdString( ), result_number);
        defocus_image.QuickAndDirtyReadSlice(input_best_defocus_filename.ToStdString( ), result_number);
        mip_x_dimension = mip_image.logical_x_dimension;
        mip_y_dimension = mip_image.logical_y_dimension;

        coordinate_file.WriteCommentLine("SEARCH_PIXEL_SIZE %f", search_pixel_size);
        coordinate_file.WriteCommentLine("         Psi          Theta            Phi              X              Y              Z      PixelSize           Peak");
        min_peak_radius = powf(min_peak_radius, 2);
    }
    else {
        // No MIP file in coordinate mode. Allocate mip_image with user-provided dimensions
        // for bounds checking. Read search_pixel_size from coordinate file comment.
        mip_image.Allocate(mip_x_dimension, mip_y_dimension, true);

        if ( ! coordinate_file.ReadCommentValueAsFloat("SEARCH_PIXEL_SIZE", search_pixel_size) ) {
            wxPrintf("WARNING: No SEARCH_PIXEL_SIZE found in coordinate file (possibly older format). "
                     "Assuming search pixel size equals input pixel size (%f A).\n",
                     pixel_size);
        }
    }
    float mip_to_micrograph_scale = search_pixel_size / pixel_size;

    micrograph.QuickAndDirtyReadSlice(input_image_filename.ToStdString( ), 1);

    std::vector<Peak>                  peak_list;
    std::vector<Peak>                  upsampled_peak_list;
    ArrayOfTemplateMatchFoundPeakInfos all_peak_infos;

    // Create extractor - no pixel_size_image in prepare_stack_matchtemplate (pixel size is per-micrograph)
    TemplateMatchingPeakExtractor extractor(
            mip_image, phi_image, theta_image, psi_image,
            defocus_image, nullptr,
            pixel_size, search_pixel_size);

    wxPrintf("\n");
    if ( ! read_coordinates ) {
        // Search mode: find peaks in MIP

        mip_image.FindPeakWithIntegerCoordinatesForManyPeaks(
                peak_list,
                upsampled_peak_list,
                wanted_threshold,
                use_peak_sampling_correction ? cistem::match_template::PEAK_THRESHOLD_SCALE : 1.0f,
                sqrtf(min_peak_radius), 0);

        extractor.TransferAndSortPeakInfo(peak_list, upsampled_peak_list, use_peak_sampling_correction, all_peak_infos);
        number_of_peaks_found = all_peak_infos.GetCount( );

        // Write coordinate file
        for ( int i = 0; i < all_peak_infos.GetCount( ); i++ ) {
            coordinates[0] = all_peak_infos[i].psi;
            coordinates[1] = all_peak_infos[i].theta;
            coordinates[2] = all_peak_infos[i].phi;
            coordinates[3] = all_peak_infos[i].x_pos;
            coordinates[4] = all_peak_infos[i].y_pos;
            coordinates[5] = all_peak_infos[i].defocus;
            coordinates[6] = all_peak_infos[i].pixel_size;
            coordinates[7] = all_peak_infos[i].peak_height;
            coordinate_file.WriteLine(coordinates);
        }
    }
    else {
        // Read mode: load peaks from coordinate file
        extractor.ReadPeaksFromCoordinateFile(coordinate_file, peak_list, all_peak_infos);
        number_of_peaks_found = all_peak_infos.GetCount( );
    }

    for ( int i = 0; i < all_peak_infos.GetCount( ); i++ ) {
        wxPrintf("Peak %4i at x, y, psi, theta, phi, defocus, pixel size = %12.6f, %12.6f, %12.6f, %12.6f, %12.6f, %12.6f, %12.6f : %10.6f\n",
                 i + 1, all_peak_infos[i].x_pos, all_peak_infos[i].y_pos, all_peak_infos[i].psi,
                 all_peak_infos[i].theta, all_peak_infos[i].phi, all_peak_infos[i].defocus,
                 all_peak_infos[i].pixel_size, all_peak_infos[i].peak_height);
    }

    extractor.CreateParticleStack(
            peak_list, all_peak_infos, micrograph,
            output_particle_stack_filename, output_star_filename,
            box_size, mip_to_micrograph_scale,
            voltage_kV, spherical_aberration_mm, amplitude_contrast,
            average_defocus_1, average_defocus_2, average_defocus_angle,
            input_image_filename);

    if ( is_running_locally == true ) {
        wxPrintf("\nFound %i peaks.\n\n", number_of_peaks_found);
        wxPrintf("\nMake Particle Stack: Normal termination\n");
        wxDateTime finish_time = wxDateTime::Now( );
        wxPrintf("Total Run Time : %s\n\n", finish_time.Subtract(start_time).Format("%Hh:%Mm:%Ss"));
    }

    return true;
}
