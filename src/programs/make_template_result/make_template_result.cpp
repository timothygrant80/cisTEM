#include "../../core/core_headers.h"

#include "../../constants/constants.h"

#include "../match_template/template_matching_peak_extractor.h"

class
        MakeTemplateResult : public MyApp {
  public:
    bool DoCalculation( );
    void DoInteractiveUserInput( );

  private:
};

IMPLEMENT_APP(MakeTemplateResult)

// override the DoInteractiveUserInput

void MakeTemplateResult::DoInteractiveUserInput( ) {

    wxString input_reconstruction_filename;
    wxString input_mip_filename;
    wxString input_best_psi_filename;
    wxString input_best_theta_filename;
    wxString input_best_phi_filename;
    wxString input_best_defocus_filename;
    wxString input_best_pixel_size_filename;
    wxString output_result_image_filename;
    wxString output_slab_filename;
    wxString xyz_coords_filename;

    float wanted_threshold;
    float min_peak_radius;
    float slab_thickness;
    float pixel_size;
    float binning_factor;
    int   result_number;
    int   mip_x_dimension = 0;
    int   mip_y_dimension = 0;
    bool  read_coordinates;

    UserInput* my_input = new UserInput("MakeTemplateResult", 1.00);

    read_coordinates = my_input->GetYesNoFromUser("Read coordinates from file?", "Should the target coordinates be read from a file instead of search results?", "No");
    if ( ! read_coordinates ) {
        input_mip_filename             = my_input->GetFilenameFromUser("Input MIP file", "The file for saving the maximum intensity projection image", "mip.mrc", true);
        input_best_psi_filename        = my_input->GetFilenameFromUser("Input psi file", "The file containing the best psi image", "psi.mrc", true);
        input_best_theta_filename      = my_input->GetFilenameFromUser("Input theta file", "The file containing the best psi image", "theta.mrc", true);
        input_best_phi_filename        = my_input->GetFilenameFromUser("Input phi file", "The file containing the best psi image", "phi.mrc", true);
        input_best_defocus_filename    = my_input->GetFilenameFromUser("Input defocus file", "The file with the best defocus image", "defocus.mrc", true);
        input_best_pixel_size_filename = my_input->GetFilenameFromUser("Input pixel size file", "The file with the best pixel size image", "pixel_size.mrc", true);
        xyz_coords_filename            = my_input->GetFilenameFromUser("Output x,y,z coordinate file", "The file for saving the x,y,z coordinates of the found targets", "coordinates.txt", false);
        wanted_threshold               = my_input->GetFloatFromUser("Peak threshold", "Peaks over this size will be taken", "7.5", 0.0);
        min_peak_radius                = my_input->GetFloatFromUser("Min Peak Radius (px.)", "Essentially the minimum closeness for peaks", "10.0", 1.0);
        result_number                  = my_input->GetIntFromUser("Result number to process", "If input files contain results from several searches, which one should be used?", "1", 1);
    }
    else {
        mip_x_dimension     = my_input->GetIntFromUser("X-dimension of original MIP", "The x-dimension of the MIP that contained the peaks listed in the input coordinate file", "5760", 100);
        mip_y_dimension     = my_input->GetIntFromUser("Y-dimension of original MIP", "The y-dimension of the MIP that contained the peaks listed in the input coordinate file", "4092", 100);
        xyz_coords_filename = my_input->GetFilenameFromUser("Input x,y,z coordinate file", "The file containing the x,y,z coordinates of the found targets", "coordinates.txt", false);
    }
    input_reconstruction_filename     = my_input->GetFilenameFromUser("Input template reconstruction", "The 3D reconstruction from which projections are calculated", "reconstruction.mrc", true);
    output_result_image_filename      = my_input->GetFilenameFromUser("Output 2D projection montage", "The file for saving the found result", "result.mrc", false);
    output_slab_filename              = my_input->GetFilenameFromUser("Output slab volume montage", "The file for saving the slab with the found targets", "slab.mrc", false);
    slab_thickness                    = my_input->GetFloatFromUser("Sample thickness (A)", "The thickness of the sample that was searched", "2000.0", 100.0);
    pixel_size                        = my_input->GetFloatFromUser("Pixel size of images (A)", "Pixel size of input images in Angstroms", "1.0", 0.0);
    binning_factor                    = my_input->GetFloatFromUser("Binning factor for slab", "Factor to reduce size of output slab", "4.0", 0.0);
    bool use_peak_sampling_correction = my_input->GetYesNoFromUser("Use peak sampling correction", "Apply peak height sampling correction", "Yes");

    delete my_input;

    //	my_current_job.Reset(14);
    my_current_job.ManualSetArguments("ttttttttttfffffbiiib", input_reconstruction_filename.ToUTF8( ).data( ),
                                      input_mip_filename.ToUTF8( ).data( ),
                                      input_best_psi_filename.ToUTF8( ).data( ),
                                      input_best_theta_filename.ToUTF8( ).data( ),
                                      input_best_phi_filename.ToUTF8( ).data( ),
                                      input_best_defocus_filename.ToUTF8( ).data( ),
                                      input_best_pixel_size_filename.ToUTF8( ).data( ),
                                      output_result_image_filename.ToUTF8( ).data( ),
                                      output_slab_filename.ToUTF8( ).data( ),
                                      xyz_coords_filename.ToUTF8( ).data( ),
                                      wanted_threshold,
                                      min_peak_radius,
                                      slab_thickness,
                                      pixel_size, binning_factor,
                                      read_coordinates,
                                      mip_x_dimension, mip_y_dimension,
                                      result_number,
                                      use_peak_sampling_correction);
}

// override the do calculation method which will be what is actually run..

bool MakeTemplateResult::DoCalculation( ) {

    wxDateTime start_time = wxDateTime::Now( );

    wxString input_reconstruction_filename  = my_current_job.arguments[0].ReturnStringArgument( );
    wxString input_mip_filename             = my_current_job.arguments[1].ReturnStringArgument( );
    wxString input_best_psi_filename        = my_current_job.arguments[2].ReturnStringArgument( );
    wxString input_best_theta_filename      = my_current_job.arguments[3].ReturnStringArgument( );
    wxString input_best_phi_filename        = my_current_job.arguments[4].ReturnStringArgument( );
    wxString input_best_defocus_filename    = my_current_job.arguments[5].ReturnStringArgument( );
    wxString input_best_pixel_size_filename = my_current_job.arguments[6].ReturnStringArgument( );
    wxString output_result_image_filename   = my_current_job.arguments[7].ReturnStringArgument( );
    wxString output_slab_filename           = my_current_job.arguments[8].ReturnStringArgument( );
    wxString xyz_coords_filename            = my_current_job.arguments[9].ReturnStringArgument( );
    float    wanted_threshold               = my_current_job.arguments[10].ReturnFloatArgument( );
    float    min_peak_radius                = my_current_job.arguments[11].ReturnFloatArgument( );
    float    slab_thickness                 = my_current_job.arguments[12].ReturnFloatArgument( );
    float    pixel_size                     = my_current_job.arguments[13].ReturnFloatArgument( );
    float    binning_factor                 = my_current_job.arguments[14].ReturnFloatArgument( );
    bool     read_coordinates               = my_current_job.arguments[15].ReturnBoolArgument( );
    int      mip_x_dimension                = my_current_job.arguments[16].ReturnIntegerArgument( );
    int      mip_y_dimension                = my_current_job.arguments[17].ReturnIntegerArgument( );
    int      result_number                  = my_current_job.arguments[18].ReturnIntegerArgument( );
    bool     use_peak_sampling_correction   = my_current_job.arguments[19].ReturnBoolArgument( );

    float padding = 2.0f;

    ImageFile input_reconstruction_file;

    input_reconstruction_file.OpenFile(input_reconstruction_filename.ToStdString( ), false);

    Image output_image;
    Image mip_image;
    Image psi_image;
    Image theta_image;
    Image phi_image;
    Image defocus_image;
    Image pixel_size_image;
    Image input_reconstruction;
    Image binned_3d_reconstruction;
    Image current_projection;
    Image padded_projection;
    Image slab;

    int   number_of_peaks_found = 0;
    int   slab_thickness_in_pixels;
    int   binned_3d_dimension;
    float binned_3d_pixel_size;
    float max_density;
    long  text_file_access_type;

    float coordinates[8];
    if ( read_coordinates )
        text_file_access_type = OPEN_TO_READ;
    else
        text_file_access_type = OPEN_TO_WRITE;
    NumericTextFile coordinate_file(xyz_coords_filename, text_file_access_type, 8);
    float           search_pixel_size = pixel_size; // default to input pixel size
    if ( ! read_coordinates ) {
        ImageFile mip_file(input_mip_filename.ToStdString( ), false);
        search_pixel_size = mip_file.ReturnPixelSize( );

        coordinate_file.WriteCommentLine("SEARCH_PIXEL_SIZE %f", search_pixel_size);
        coordinate_file.WriteCommentLine("         Psi          Theta            Phi              X              Y              Z      PixelSize           Peak");

        mip_image.QuickAndDirtyReadSlice(input_mip_filename.ToStdString( ), result_number);
        psi_image.QuickAndDirtyReadSlice(input_best_psi_filename.ToStdString( ), result_number);
        theta_image.QuickAndDirtyReadSlice(input_best_theta_filename.ToStdString( ), result_number);
        phi_image.QuickAndDirtyReadSlice(input_best_phi_filename.ToStdString( ), result_number);
        defocus_image.QuickAndDirtyReadSlice(input_best_defocus_filename.ToStdString( ), result_number);
        pixel_size_image.QuickAndDirtyReadSlice(input_best_pixel_size_filename.ToStdString( ), result_number);
        mip_x_dimension = mip_image.logical_x_dimension;
        mip_y_dimension = mip_image.logical_y_dimension;

        min_peak_radius = powf(min_peak_radius, 2);
    }
    else {
        // Read search_pixel_size from coordinate file comment. Fall back to input pixel size
        // for older coordinate files that don't have this metadata.
        if ( ! coordinate_file.ReadCommentValueAsFloat("SEARCH_PIXEL_SIZE", search_pixel_size) ) {
            wxPrintf("WARNING: No SEARCH_PIXEL_SIZE found in coordinate file (possibly older format). "
                     "Assuming search pixel size equals input pixel size (%f A).\n",
                     pixel_size);
        }
    }

    output_image.Allocate(mip_x_dimension, mip_y_dimension, 1);
    output_image.SetToConstant(0.0f);

    // Read reconstruction - will be resized in peak extractor constructor if needed
    input_reconstruction.ReadSlices(&input_reconstruction_file, 1, input_reconstruction_file.ReturnNumberOfSlices( ));

    // Setup binned reconstruction for slab
    binned_3d_reconstruction.CopyFrom(&input_reconstruction);
    binned_3d_dimension = myroundint(float(input_reconstruction.logical_x_dimension) / binning_factor);
    if ( IsOdd(binned_3d_dimension) )
        binned_3d_dimension++;
    binning_factor           = float(input_reconstruction.logical_x_dimension) / float(binned_3d_dimension);
    binned_3d_pixel_size     = pixel_size * binning_factor;
    slab_thickness_in_pixels = myroundint(slab_thickness / binned_3d_pixel_size);
    wxPrintf("\nSlab dimensions = %i %i %i\n", myroundint(mip_x_dimension / binning_factor), myroundint(mip_y_dimension / binning_factor), slab_thickness_in_pixels);

    slab.Allocate(myroundint(mip_x_dimension / binning_factor), myroundint(mip_y_dimension / binning_factor), slab_thickness_in_pixels);
    slab.SetToConstant(0.0f);

    if ( binned_3d_dimension != input_reconstruction.logical_x_dimension ) {
        binned_3d_reconstruction.ForwardFFT( );
        binned_3d_reconstruction.Resize(binned_3d_dimension, binned_3d_dimension, binned_3d_dimension);
        binned_3d_reconstruction.BackwardFFT( );
    }
    max_density = binned_3d_reconstruction.ReturnAverageOfMaxN( );
    binned_3d_reconstruction.DivideByConstant(max_density);

    // Apply padding to reconstruction if needed - must happen before CreateResultImages
    if ( padding != 1.0f ) {
        input_reconstruction.Resize(input_reconstruction.logical_x_dimension * padding, input_reconstruction.logical_y_dimension * padding, input_reconstruction.logical_z_dimension * padding, input_reconstruction.ReturnAverageOfRealValuesOnEdges( ));
    }

    // assume cube
    current_projection.Allocate(input_reconstruction.logical_x_dimension, input_reconstruction.logical_x_dimension, false);
    if ( padding != 1.0f )
        padded_projection.Allocate(input_reconstruction_file.ReturnXSize( ) * padding, input_reconstruction_file.ReturnXSize( ) * padding, false);

    std::vector<Peak>                  peak_list;
    std::vector<Peak>                  upsampled_peak_list;
    ArrayOfTemplateMatchFoundPeakInfos all_peak_infos;

    if ( ! read_coordinates ) {
        // Search mode: find peaks in MIP
        Image masked_mip;
        masked_mip = mip_image;
        masked_mip.FindPeakWithIntegerCoordinatesForManyPeaks(
                peak_list,
                upsampled_peak_list,
                wanted_threshold,
                use_peak_sampling_correction ? cistem::match_template::PEAK_THRESHOLD_SCALE : 1.0f,
                sqrtf(min_peak_radius),
                0);

        TemplateMatchingPeakExtractor extractor(
                mip_image, phi_image, theta_image, psi_image,
                defocus_image, &pixel_size_image,
                pixel_size, search_pixel_size);

        extractor.TransferAndSortPeakInfo(peak_list, upsampled_peak_list, use_peak_sampling_correction, all_peak_infos);

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

        number_of_peaks_found = all_peak_infos.GetCount( );

        wxPrintf("\n");
        for ( int i = 0; i < all_peak_infos.GetCount( ); i++ ) {
            wxPrintf("Peak %4i at x, y, psi, theta, phi, defocus, pixel size = %12.6f, %12.6f, %12.6f, %12.6f, %12.6f, %12.6f, %12.6f : %10.6f\n",
                     i + 1, all_peak_infos[i].x_pos, all_peak_infos[i].y_pos, all_peak_infos[i].psi,
                     all_peak_infos[i].theta, all_peak_infos[i].phi, all_peak_infos[i].defocus,
                     all_peak_infos[i].pixel_size, all_peak_infos[i].peak_height);
        }

        extractor.CreateResultImages(
                peak_list, all_peak_infos,
                input_reconstruction, current_projection, output_image,
                true,
                (padding != 1.0f) ? &padded_projection : nullptr,
                &slab, &binned_3d_reconstruction, binned_3d_pixel_size);
    }
    else {
        // Read mode: load peaks from coordinate file
        TemplateMatchingPeakExtractor extractor(
                output_image, phi_image, theta_image, psi_image,
                defocus_image, nullptr,
                pixel_size, search_pixel_size);

        extractor.ReadPeaksFromCoordinateFile(coordinate_file, peak_list, all_peak_infos);
        number_of_peaks_found = all_peak_infos.GetCount( );

        wxPrintf("\n");
        for ( int i = 0; i < all_peak_infos.GetCount( ); i++ ) {
            wxPrintf("Peak %4i at x, y, psi, theta, phi, defocus, pixel size = %12.6f, %12.6f, %12.6f, %12.6f, %12.6f, %12.6f, %12.6f : %10.6f\n",
                     i + 1, all_peak_infos[i].x_pos, all_peak_infos[i].y_pos, all_peak_infos[i].psi,
                     all_peak_infos[i].theta, all_peak_infos[i].phi, all_peak_infos[i].defocus,
                     all_peak_infos[i].pixel_size, all_peak_infos[i].peak_height);
        }

        extractor.CreateResultImages(
                peak_list, all_peak_infos,
                input_reconstruction, current_projection, output_image,
                true,
                (padding != 1.0f) ? &padded_projection : nullptr,
                &slab, &binned_3d_reconstruction, binned_3d_pixel_size);
    }

    // save the output image
    output_image.QuickAndDirtyWriteSlice(output_result_image_filename.ToStdString( ), 1, true, search_pixel_size);
    slab.QuickAndDirtyWriteSlices(output_slab_filename.ToStdString( ), 1, slab_thickness_in_pixels, true, binned_3d_pixel_size);

    if ( is_running_locally == true ) {
        wxPrintf("\nFound %i peaks.\n\n", number_of_peaks_found);
        wxPrintf("\nMake Template Results: Normal termination\n");
        wxDateTime finish_time = wxDateTime::Now( );
        wxPrintf("Total Run Time : %s\n\n", finish_time.Subtract(start_time).Format("%Hh:%Mm:%Ss"));
    }

    return true;
}
