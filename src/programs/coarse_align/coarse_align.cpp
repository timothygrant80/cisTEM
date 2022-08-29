#include "../../core/core_headers.h"

class
        NikoTestApp : public MyApp {

  public:
    bool DoCalculation( );
    void DoInteractiveUserInput( );

  private:
};

IMPLEMENT_APP(NikoTestApp)

// override the DoInteractiveUserInput

void NikoTestApp::DoInteractiveUserInput( ) {
    UserInput* my_input = new UserInput("Coarse Align", 1.0);

    wxString input_imgstack = my_input->GetFilenameFromUser("Input image file name", "Name of input image file *.mrc", "input.mrc", true);
    wxString angle_filename = my_input->GetFilenameFromUser("Tilt Angle filename", "The tilts, *.tlt", "ang.tlt", true);
    // wxString coordinates_filename  = my_input->GetFilenameFromUser("Coordinates (PLT) filename", "The input particle coordinates, in Imagic-style PLT forlmat", "coos.plt", true);
    // int output_stack_box_size = my_input->GetIntFromUser("Box size for output candidate particle images (pixels)", "In pixels. Give 0 to skip writing particle images to disk.", "256", 0);

    delete my_input;

    my_current_job.Reset(4);
    my_current_job.ManualSetArguments("tt", input_imgstack.ToUTF8( ).data( ), angle_filename.ToUTF8( ).data( ));
}

bool NikoTestApp::DoCalculation( ) {
    wxPrintf("Hello world4\n");
    // ===========================image stack parameters initialization======================================
    // user passed parameters-------------
    wxString input_imgstack = my_current_job.arguments[0].ReturnStringArgument( );
    wxString angle_filename = my_current_job.arguments[1].ReturnStringArgument( );
    wxString weightname     = "weighttest----------";
    // wxString    outputpath     = "/groups/lingli/Documents/cisTEM/build/tomoalign_Intel-gpu-debug-static/src/Output/";
    // std::string outputpathstd  = "/groups/lingli/Documents/cisTEM/build/tomoalign_Intel-gpu-debug-static/src/Output/";
    wxString    outputpath    = "/groups/lingli/Documents/cisTEM/build/tomoalign_Intel-gpu-debug-static/src/Output/";
    std::string outputpathstd = "/groups/lingli/Documents/cisTEM/build/tomoalign_Intel-gpu-debug-static/src/Output/";

    // local parameters-------------------
    MRCFile          input_stack(input_imgstack.ToStdString( ), false), output_stack, stretched_stack;
    NumericTextFile *input_coos_file, *tilt_angle_file, *peak_points, *shift_file, *peak_points_raw;
    tilt_angle_file = new NumericTextFile(angle_filename, OPEN_TO_READ, 1);
    peak_points     = new NumericTextFile(outputpath + "peakpoints_newcurved.txt", OPEN_TO_WRITE, 3);
    peak_points_raw = new NumericTextFile(outputpath + "peakpoints_pk_img.txt", OPEN_TO_WRITE, 3);
    shift_file      = new NumericTextFile(outputpath + "shifts_newcurved.txt", OPEN_TO_WRITE, 3);

    int      image_no = input_stack.ReturnNumberOfSlices( );
    MRCFile* peakfile = new MRCFile[image_no];
    int      center_index;
    float    tilts[image_no], start_angle, tilt_step, phi; // angle related
    float    shifts[image_no][2];
    float    peaks[image_no][2];

    ProgressBar* my_progress = new ProgressBar(input_stack.ReturnNumberOfSlices( ));
    Image        current_image, ref_image;
    Image        weightingmap_image;

    //---------------------------------- loading the tilts into an array -------------------------
    for ( int i = 0; i < image_no; i++ ) {
        tilt_angle_file->ReadLine(&tilts[i]);
        // wxPrintf("angle %i ; % g\n", i, tilts[i]);
    }

    start_angle  = tilts[0];
    tilt_step    = tilts[1] - tilts[0];
    center_index = int(-start_angle / tilt_step) + 1;

    for ( int i = 0; i < image_no; i++ )
        tilts[i] = tilts[i] / 180.0 * PI;
    // center_index  = int(image_no / 2);

    wxPrintf("image number in the stack: %i\n", image_no);
    wxPrintf("tomo angles: \nstart angle is %g, tilt steps %g \n", start_angle, tilt_step);
    wxPrintf("center index is: %i\n", center_index);

    //=========================================image processing ========================================

    // filtering options ----------------------------------
    int   bin      = 4;
    float sigma_l  = 0.05; // sigma regarding gaussian distribution. at 3 sigma, the probability down to 0.00135
    float sigma_h  = 0.01;
    float radius_l = 0.05;
    float radius_h = 0.0;

    //some testing operations ----------------------------------
    // load a sample image:
    ref_image.ReadSlice(&input_stack, 4);
    ref_image.WriteSlicesAndFillHeader(outputpathstd + "img_original.mrc", 1);
    int image_dim_x = ref_image.logical_x_dimension;
    int image_dim_y = ref_image.logical_y_dimension;
    wxPrintf("image dimension: %i, %i\n", image_dim_x, image_dim_y);

    // test bin:
    ref_image.RealSpaceBinning(bin, bin, 1);
    image_dim_x = ref_image.logical_x_dimension;
    image_dim_y = ref_image.logical_y_dimension;
    wxPrintf("image dimension after binning: %i, %i\n", image_dim_x, image_dim_y);
    int scale_binned = image_dim_x * image_dim_y;
    ref_image.WriteSlicesAndFillHeader(outputpathstd + "img_binned.mrc", 1);

    // // test padding
    // int   pad_dim = 1120;
    // Image padded_image;
    // ref_image.ForwardFFT(false);
    // padded_image.Allocate(pad_dim, pad_dim, true);
    // padded_image.SetToConstant(0.0);
    // ref_image.ClipInto(&padded_image, 0, false);
    // padded_image.BackwardFFT( );
    // ref_image.CopyFrom(&padded_image);
    // ref_image.WriteSlicesAndFillHeader(outputpathstd + "img_binned_padded.mrc", 1);
    // image_dim_x = ref_image.logical_x_dimension;
    // image_dim_y = ref_image.logical_y_dimension;
    // cos_stretched_image.DivideByConstant(scale);
    //to end-----------------------
    ref_image.ForwardFFT( );
    // ref_image.GaussianLowPassFilter(sigma_l);
    // ref_image.GaussianHighPassFilter(sigma_h);
    ref_image.GaussianLowPassRadiusFilter(radius_l, sigma_l);
    ref_image.GaussianHighPassRadiusFilter(radius_h, sigma_h);
    ref_image.BackwardFFT( );
    ref_image.WriteSlicesAndFillHeader(outputpathstd + "filteredimg.mrc", 1);

    // mask parameters ---------------------------------------
    //the mask parameter didn't really influence the result
    float mask_radius_x = image_dim_x / 2.0 - image_dim_x / 10.0;
    float mask_radius_y = image_dim_x / 2.0 - image_dim_x / 10.0;
    float mask_radius_z = 1;
    float mask_edge     = image_dim_x / 4.0;
    // float mask_radius_x = image_dim_x / 2.0 - 96;
    // float mask_radius_y = image_dim_x / 2.0 - 96;
    // float mask_radius_z = 1;
    // float mask_edge     = 192;

    // test square mask:
    ref_image.CosineRectangularMask(mask_radius_x, mask_radius_y, mask_radius_z, mask_edge);
    // ref_image.SquareMaskWithCosineEdge(image_dim_x / 2.0, image_dim_x / 4.0 - 10, 1, 0, 0, 0);
    ref_image.WriteSlicesAndFillHeader(outputpathstd + "squaremsked.mrc", 1);

    Image squaremask;
    squaremask.Allocate(image_dim_x, image_dim_y, true);
    squaremask.SetToConstant(1.0);
    // squaremask.SquareMaskWithValue(int(image_dim_x / 3), 0.0, false, 0, 0, 0);
    // squaremask.ForwardFFT( );
    // squaremask.GaussianLowPassFilter(0.001);
    // squaremask.BackwardFFT( );
    squaremask.SquareMaskWithCosineEdge(image_dim_x / 2.0, image_dim_x / 4.0 - 10, 0, 0, 0, 0);
    squaremask.WriteSlicesAndFillHeader(outputpathstd + "square_mask.mrc", 1);

    //--------------------------------initialize the weighting curve------------------------------------
    Peak  peak;
    float tmppeak[3];

    Curve weighting_curve;
    float theta = 10;

    weighting_curve.SetupXAxis(0.0, 0.5 * sqrtf(2.0), int((ref_image.ReturnLargestLogicalDimension( ) / 2.0 + 1.0) * sqrtf(2.0) + 1.0));
    for ( int i = 0; i < weighting_curve.number_of_points; i++ ) {
        float x = weighting_curve.data_x[i];
        // weighting_curve.AddValueAtXUsingLinearInterpolation(x, exp(-pow(x, 2) / 0.2), 1); //curve
        // weighting_curve.AddValueAtXUsingLinearInterpolation(x, exp(-pow(x / theta, 2) / 2.0) / theta / sqrtf(2 * PI), 1); //newcurve
        // wxPrintf("bin=%i, x= %g, y=%g\n", i, weighting_curve.data_x[i], weighting_curve.data_y[i]);
        weighting_curve.AddValueAtXUsingLinearInterpolation(x, exp(-pow(x / 0.1, 2)), 1); //curve
    }
    weighting_curve.WriteToFile(outputpath + "weighting_newcurve.txt");
    //write out the weighting map for peak
    float scale = image_dim_x * image_dim_y;
    weightingmap_image.Allocate(ref_image.logical_x_dimension, ref_image.logical_y_dimension, true);
    weightingmap_image.SetToConstant(1.0);
    weightingmap_image.MultiplyByWeightsCurveReal(weighting_curve, 1.0);
    // weightingmap_image.WriteSlicesAndFillHeader(outputpath + "newweightingmap.mrc", 1);
    weightingmap_image.WriteSlicesAndFillHeader(outputpathstd + "newweightingmap.mrc", 1);

    // wxPrintf("physical address of box center %i, %i \n", ref_image.physical_address_of_box_center_x, ref_image.physical_address_of_box_center_y);
    // wxPrintf("image dimensin: %i\n", ref_image.ReturnLargestLogicalDimension( ));
    // wxPrintf("half image: %i \n ", int(ref_image.ReturnLargestLogicalDimension( ) / 2.0 + 1.0));
    // wxPrintf("no. of weighting points: %i \n", int((ref_image.ReturnLargestLogicalDimension( ) / 2.0 + 1.0) * sqrtf(2.0) + 1.0));

    //============================do cross correlation to calculate peak image================================================
    Image cos_clipped_image, cos_stretched_image;
    int   stretched_dimension;
    int   stretched_dim_X, stretched_dim_Y;
    int   current_index, ref_index;

    for ( int i = 1; i <= image_no; i++ ) {
        // for ( int i = 1; i <= 3; i++ ) {
        // for ( int i = 60; i <= image_no; i++ ) {
        //-----------------------------
        // sigma_h = 0.02;
        // for ( int j = 0; j <= 6; j++ ) {
        //     int i   = 12;
        //     sigma_h = sigma_h - 0.001;
        //-----------------------------

        if ( i < center_index ) {
            current_index = i;
            ref_index     = i + 1;
        }
        else if ( i == center_index ) {
            current_index = i;
            ref_index     = i;
        }
        else {
            current_index = i;
            ref_index     = i - 1;
        }

        current_image.ReadSlice(&input_stack, current_index);
        ref_image.ReadSlice(&input_stack, ref_index);
        // Remove Outliers-------------
        current_image.ReplaceOutliersWithMean(3.0f);
        ref_image.ReplaceOutliersWithMean(3.0f);
        // image bin------------------
        current_image.RealSpaceBinning(bin, bin, 1);
        ref_image.RealSpaceBinning(bin, bin, 1);
        // // image padding-------------
        // current_image.ForwardFFT(false);
        // padded_image.SetToConstant(0.0);
        // current_image.ClipInto(&padded_image, 0, false);
        // padded_image.BackwardFFT( );
        // padded_image.DivideByConstant(scale_binned);
        // current_image.CopyFrom(&padded_image);

        // ref_image.ForwardFFT(false);
        // padded_image.SetToConstant(0.0);
        // ref_image.ClipInto(&padded_image, 0, false);
        // padded_image.BackwardFFT( );
        // padded_image.DivideByConstant(scale_binned);
        // ref_image.CopyFrom(&padded_image);

        // add filter-----------------
        current_image.ForwardFFT( );
        // current_image.GaussianLowPassRadiusFilter(radius_l, sigma_l);
        current_image.GaussianLowPassFilter(sigma_l);
        current_image.GaussianHighPassRadiusFilter(radius_h, sigma_h);
        current_image.BackwardFFT( );
        ref_image.ForwardFFT( );
        // ref_image.GaussianLowPassRadiusFilter(radius_l, sigma_l);
        ref_image.GaussianLowPassFilter(sigma_l);
        ref_image.GaussianHighPassRadiusFilter(radius_h, sigma_h);
        ref_image.BackwardFFT( );

        current_image.WriteSlicesAndFillHeader(wxString::Format(outputpath + "input_bf_stretch%02i_%.3f.mrc", current_index, sigma_h).ToStdString( ), true);
        // ref_image.WriteSlicesAndFillHeader(wxString::Format(outputpath + "ref%02i_%.3f.mrc", ref_index, sigma_h).ToStdString( ), true);

        // current image stretch------------------------------------------------------------------------
        // stretched_dimension = myroundint(image_dim_x * cosf(tilts[current_index - 1] - tilts[ref_index - 1]));
        // cos_clipped_image.Allocate(stretched_dimension, image_dim_y, 1, true);

        // current_image.ClipInto(&cos_clipped_image, 0.0, false, 0.0, 0, 0, 0);
        // wxPrintf("current image: %i, %i     stretched image: %i, %i \n", current_image.logical_x_dimension, current_image.logical_y_dimension,
        //          cos_clipped_image.logical_x_dimension, cos_clipped_image.logical_y_dimension);
        // current_image.WriteSlicesAndFillHeader(wxString::Format(outputpath + "input_bfstretch%02i_%.3f.mrc", current_index, sigma_h).ToStdString( ), true);
        // ref_image.WriteSlicesAndFillHeader(wxString::Format(outputpath + "ref_bfstretch%02i_%.3f.mrc", ref_index, sigma_h).ToStdString( ), true);

        // stretch only along X axis, assuming phi is 0. --------------------
        // stretched_dimension = myroundint(image_dim_x * (1 - cosf(tilts[ref_index - 1]) + cosf(tilts[current_index - 1] * 1.01)));
        stretched_dimension = myroundint(image_dim_x * (1 - cosf(tilts[ref_index - 1]) + cosf(tilts[current_index - 1])));

        cos_clipped_image.Allocate(stretched_dimension, image_dim_y, 1, true);

        // //stretch along X and Y, i.e., phi is considered
        // stretched_dim_X = myroundint(image_dim_x * cosf(phi) * (1 - cosf(tilts[ref_index - 1]) + cosf(tilts[current_index - 1])));
        // stretched_dim_Y = myroundint(image_dim_y * sinf(phi) * (1 - cosf(tilts[ref_index - 1]) + cosf(tilts[current_index - 1])));
        cos_clipped_image.Allocate(stretched_dimension, image_dim_y, 1, true);

        current_image.ClipInto(&cos_clipped_image, 0.0, false, 0.0, 0, 0, 0);
        wxPrintf("current image: %i, %i     stretched image: %i, %i \n", current_image.logical_x_dimension, current_image.logical_y_dimension,
                 cos_clipped_image.logical_x_dimension, cos_clipped_image.logical_y_dimension);

        // cos_clipped_image.WriteSlicesAndFillHeader("clipped_to_0deg_03.mrc", 1);

        float meanvalue = cos_clipped_image.ReturnAverageOfRealValues( );
        cos_clipped_image.ForwardFFT(false);
        cos_stretched_image.Allocate(image_dim_x, image_dim_y, 1, true);
        cos_stretched_image.SetToConstant(0.0); // To avoid valgrind uninitialised errors, but maybe this is a waste?
        cos_clipped_image.ClipInto(&cos_stretched_image, meanvalue, false);
        cos_stretched_image.BackwardFFT( );
        cos_stretched_image.DivideByConstant(scale);

        //add filter position two
        // // add filter-----------------
        // cos_stretched_image.ForwardFFT( );
        // // cos_stretched_image.GaussianLowPassRadiusFilter(radius_l, sigma_l);
        // cos_stretched_image.GaussianLowPassFilter(sigma_l);
        // cos_stretched_image.GaussianHighPassRadiusFilter(radius_h, sigma_h);
        // cos_stretched_image.BackwardFFT( );
        // ref_image.ForwardFFT( );
        // // ref_image.GaussianLowPassRadiusFilter(radius_l, sigma_l);
        // ref_image.GaussianLowPassFilter(sigma_l);
        // ref_image.GaussianHighPassRadiusFilter(radius_h, sigma_h);
        // ref_image.BackwardFFT( );
        //
        // cos_stretched_image.CopyFrom(&current_image);
        // add mask
        // cos_stretched_image.SquareMaskWithValue(int(image_dim_x * 4 / 5), meanvalue, 0, 15, 0);
        // ref_image.SquareMaskWithValue(int(image_dim_x * 4 / 5), ref_image.ReturnAverageOfRealValues( ), 0, 0, 0);
        //------apply square mask
        // cos_stretched_image.MultiplyPixelWise(squaremask);
        // ref_image.MultiplyPixelWise(squaremask);
        // cos_stretched_image.CosineRectangularMask(image_dim_x / 4.0, image_dim_y / 4.0, 1, image_dim_x / 2.0 - 20);
        // ref_image.CosineRectangularMask(image_dim_x / 4.0 - 20, image_dim_y / 4.0 - 20, 1, image_dim_x / 2.0 - 20);
        cos_stretched_image.CosineRectangularMask(mask_radius_x, mask_radius_y, mask_radius_z, mask_edge);
        ref_image.CosineRectangularMask(mask_radius_x, mask_radius_y, mask_radius_z, mask_edge);

        // cos_stretched_image.WriteSlicesAndFillHeader("stretched_to_0deg_03.mrc", 1);
        cos_stretched_image.WriteSlicesAndFillHeader(wxString::Format(outputpath + "input%02i_%.3f.mrc", current_index, sigma_h).ToStdString( ), true);
        ref_image.WriteSlicesAndFillHeader(wxString::Format(outputpath + "ref%02i_%.3f.mrc", ref_index, sigma_h).ToStdString( ), true);

        // calculate the peak image -----------------------------------------------------------------
        cos_stretched_image.CalculateCrossCorrelationImageWith(&ref_image);
        // cos_stretched_image.Normalize( );
        cos_stretched_image.ZeroFloatAndNormalize( );
        // cos_stretched_image.MultiplyByWeightsCurveReal(weighting_curve, 1.0);
        // cos_stretched_image.CalculateCrossCorrelationImageWith(&weightingmap_image);
        peak = cos_stretched_image.FindPeakWithIntegerCoordinates(0, 400, 10);

        //peak fixing ------------------------------
        peaks[current_index - 1][0] = peak.x;
        peaks[current_index - 1][1] = peak.y;
        tmppeak[0] = current_index, tmppeak[1] = peak.x, tmppeak[2] = peak.y;
        peak_points_raw->WriteLine(tmppeak);
        peaks[current_index - 1][0] = peak.x / image_dim_x * stretched_dimension;
        peaks[current_index - 1][1] = peak.y;
        tmppeak[0] = current_index, tmppeak[1] = peaks[current_index - 1][0], tmppeak[2] = peaks[current_index - 1][1];
        peak_points->WriteLine(tmppeak);

        wxPrintf("image: %i, peak position: x = %g, y = %g ,value = %g\n\n", current_index, peak.x, peak.y, peak.value);
        // peakfile[current_index - 1].OpenFile(wxString::Format(outputpath + "stretch_newcurvepeak%02i.mrc", current_index).ToStdString( ), true);
        // cos_stretched_image.WriteSlice(&peakfile[current_index - 1], 1);
        cos_stretched_image.WriteSlicesAndFillHeader(wxString::Format(outputpath + "peak%02i_%.3f_%.3f.mrc", current_index, sigma_l, sigma_h).ToStdString( ), true);
    }

    // shifting matrix creating ==============================================================
    int bound = wxMax(image_no - center_index, center_index - 1);
    wxPrintf("image_no, center_index: %i %i \n", image_no, center_index);
    wxPrintf("left right bound: %i %i %i \n", image_no - center_index, center_index - 1, bound);
    int arr_center_index        = center_index - 1;
    shifts[arr_center_index][0] = -peaks[arr_center_index][0] * bin;
    shifts[arr_center_index][1] = -peaks[arr_center_index][0] * bin;
    // tmppeak[0] = center_index, tmppeak[1] = shifts[center_index - 1][0], tmppeak[2] = shifts[center_index - 1][1];
    // shift_file->WriteLine(tmppeak);
    for ( int i = 1; i <= bound; i++ ) {
        if ( arr_center_index + i < image_no ) {
            shifts[arr_center_index + i][0] = -peaks[arr_center_index + i][0] * bin + shifts[arr_center_index + i - 1][0];
            shifts[arr_center_index + i][1] = -peaks[arr_center_index + i][1] * bin + shifts[arr_center_index + i - 1][1];
            // tmppeak[0] = arr_center_index + i + 1, tmppeak[1] = shifts[arr_center_index + i][0], tmppeak[2] = shifts[arr_center_index + i][1];
            // shift_file->WriteLine(tmppeak);
        }
        if ( arr_center_index - i >= 0 ) {
            shifts[arr_center_index - i][0] = -peaks[arr_center_index - i][0] * bin + shifts[arr_center_index - i + 1][0];
            shifts[arr_center_index - i][1] = -peaks[arr_center_index - i][1] * bin + shifts[arr_center_index - i + 1][1];
            // tmppeak[0] = arr_center_index - i + 1, tmppeak[1] = shifts[arr_center_index - i][0], tmppeak[2] = shifts[arr_center_index - i][1];
            // shift_file->WriteLine(tmppeak);
        }
        // wxPrintf
    }

    //image stack shift ========================================================================
    output_stack.OpenFile(outputpathstd + "outputstack.mrc", true);
    for ( int i = 0; i < image_no; i++ ) {
        current_image.ReadSlice(&input_stack, i + 1);
        current_image.PhaseShift(shifts[i][0], shifts[i][1], 0);
        current_image.WriteSlice(&output_stack, i + 1);
        tmppeak[0] = i, tmppeak[1] = shifts[i][0], tmppeak[2] = shifts[i][1];
        shift_file->WriteLine(tmppeak);
    }

    delete my_progress;
    // delete input_coos_file;
    delete[] peakfile;
    // delete[] shift_file;
    // delete[] tmpang;
    // delete[] tilt_angle_file;

    return true;
}

// //image stretching
// // The regular way (the input pixel size was large enough)
// resampling_is_necessary = current_power_spectrum->logical_x_dimension != box_size || current_power_spectrum->logical_y_dimension != box_size;
// if ( do_resampling ) {
//     if ( resampling_is_necessary || stretch_factor != 1.0f ) {
//         stretched_dimension = myroundint(box_size * stretch_factor);
//         if ( IsOdd(stretched_dimension) )
//             stretched_dimension++;
//         if ( fabsf(stretched_dimension - box_size * stretch_factor) > fabsf(stretched_dimension - 2 - box_size * stretch_factor) )
//             stretched_dimension -= 2;

//         current_power_spectrum->ForwardFFT(false);new
//         resampled_power_spectrum->Allocate(stretched_dimension, stretched_dimension, 1, false);
//         current_power_spectrum->ClipInto(resampled_power_spectrum);
//         resampled_power_spectrum->BackwardFFT( );
//         temp_image.Allocate(box_size, box_size, 1, true);
//         temp_image.SetToConstant(0.0); // To avoid valgrind uninitialised errors, but maybe this is a waste?
//         resampled_power_spectrum->ClipInto(&temp_image);
//         resampled_power_spectrum->Consume(&temp_image);
//     }
//     else {
//         resampled_power_spectrum->CopyFrom(current_power_spectrum);
//     }