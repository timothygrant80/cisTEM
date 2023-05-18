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

// double sliceEdgeMean(float* array, int nxdim, int ixlo, int ixhi, int iylo,
//                      int iyhi) {
//     double dmean, sum = 0.;
//     int    ix, iy;
//     for ( ix = ixlo; ix <= ixhi; ix++ )
//         sum += array[ix + iylo * nxdim] + array[ix + iyhi * nxdim];

//     for ( iy = iylo + 1; iy < iyhi; iy++ )
//         sum += array[ixlo + iy * nxdim] + array[ixhi + iy * nxdim];

//     dmean = sum / (2 * (ixhi - ixlo + iyhi - iylo));
//     return dmean;
// }

///--------------------------------IMOD module start---------------------------
void rotmagstrToAmat(float theta, float smag, float str, float phi, float* a11,
                     float* a12, float* a21, float* a22) {
    double ator = 0.0174532925;
    float  sinth, costh, sinphi, cosphi, sinphisq, cosphisq, f1, f2, f3;

    costh    = (float)cos(ator * theta);
    sinth    = (float)sin(ator * theta);
    cosphi   = (float)cos(ator * phi);
    sinphi   = (float)sin(ator * phi);
    cosphisq = cosphi * cosphi;
    sinphisq = sinphi * sinphi;
    f1       = smag * (str * cosphisq + sinphisq);
    f2       = smag * (str - 1.) * cosphi * sinphi;
    f3       = smag * (str * sinphisq + cosphisq);
    *a11     = f1 * costh - f2 * sinth;
    *a12     = f2 * costh - f3 * sinth;
    *a21     = f1 * sinth + f2 * costh;
    *a22     = f2 * sinth + f3 * costh;
}

/*!
 * Takes the inverse of transform [f] and returns the result in [finv], which can be the 
 * same as [f].
 */
void xfInvert(float* f, float* finv, int rows) {
    float tmp[9];
    float denom   = f[0] * f[rows + 1] - f[rows] * f[1];
    int   idx     = 2 * rows;
    int   idy     = 2 * rows + 1;
    tmp[0]        = f[rows + 1] / denom;
    tmp[rows]     = -f[rows] / denom;
    tmp[1]        = -f[1] / denom;
    tmp[rows + 1] = f[0] / denom;
    tmp[idx]      = -(tmp[0] * f[idx] + tmp[rows] * f[idy]);
    tmp[idy]      = -(tmp[1] * f[idx] + tmp[rows + 1] * f[idy]);
    if ( rows > 2 ) {
        tmp[2] = tmp[5] = 0.;
        tmp[8]          = 1.;
    }
    for ( idx = 0; idx < 3 * rows; idx++ )
        finv[idx] = tmp[idx];
}

/*!
 * Applies transform [f] to the point [x], [y], with the center of transformation at
 * [xcen], [ycen], and returns the result in [xp], [yp], which can be the same as
 * [x], [y].
 */
void xfApply(float* f, float xcen, float ycen, float x, float y, float* xp, float* yp,
             int rows) {
    float xadj = x - xcen;
    float yadj = y - ycen;
    *xp        = f[0] * xadj + f[rows] * yadj + f[2 * rows] + xcen;
    *yp        = f[1] * xadj + f[rows + 1] * yadj + f[2 * rows + 1] + ycen;
}

///--------------------------------IMOD module end---------------------------

// Image ImageProc(Image input_image, int bin) {
//     input_image.ReplaceOutliersWithMean(3.0f);

//     input_image.Normalize(10);

//     input_image.RealSpaceBinning(bin, bin, 1);
//     Image padded_image;
//     // int   pad_dim_X = 1120, pad_dim_Y = 1120;
//     int pad_dim_X = 1728, pad_dim_Y = 1232;
//     // int pad_dim_X = input_image.logical_x_dimension * 1.2, pad_dim_Y = input_image.logical_y_dimension * 1.2;
//     // int pad_dim_X = int(input_image.logical_x_dimension( ) / 4), pad_dim_Y = int(input_image.logical_y_dimension( ) / 4);
//     padded_image.Allocate(pad_dim_X, pad_dim_Y, true);
//     // sample_image.ForwardFFT( );
//     float imagemean = input_image.ReturnAverageOfRealValues( );
//     // float edgemean  = sliceEdgeMean(input_image.real_values, input_image.logical_x_dimension, 0, input_image.logical_x_dimension - 1, 0, input_image.logical_y_dimension - 1);
//     // wxPrintf("edge mean is : %g\n", edgemean);
//     wxPrintf("image mean is : %g\n", imagemean);
//     wxPrintf("cistem edge mean is : %g\n", input_image.ReturnAverageOfRealValuesOnEdges( ));
//     input_image.ClipInto(&padded_image, input_image.ReturnAverageOfRealValuesOnEdges( ));
//     // wxPrintf("image bin\n\n");
//     // return input_image;
//     return padded_image;
// }

Image ImageFilter(Image input_image, float radius_lp, float sigma_lp, float radius_hp, float sigma_hp) {
    input_image.ForwardFFT( );
    input_image.GaussianLowPassRadiusFilter(radius_lp, sigma_lp);
    // input_image.GaussianLowPassFilter(sigma_lp);
    input_image.GaussianHighPassRadiusFilter(radius_hp, sigma_hp);
    input_image.BackwardFFT( );
    return input_image;
}

Image ImageStretch(Image input_image, int Stretch_Dim_X, int Stretch_Dim_Y, float padding_value) {
    Image stretched_image;
    int   scale;

    scale = input_image.logical_x_dimension * input_image.logical_y_dimension;

    stretched_image.Allocate(Stretch_Dim_X, Stretch_Dim_Y, true);
    stretched_image.SetToConstant(0.0); // To avoid valgrind uninitialised errors, but maybe this is a waste?
    input_image.ForwardFFT(false);
    input_image.ClipInto(&stretched_image, padding_value, false);
    stretched_image.BackwardFFT( );
    stretched_image.DivideByConstant(scale);
    return stretched_image;
}

Image ImageTrim(Image input_image, int Trim_Dim_X, int Trim_Dim_Y) {
    Image trimed_image;
    trimed_image.Allocate(Trim_Dim_X, Trim_Dim_Y, 1, true);
    input_image.ClipInto(&trimed_image, 0.0, false, 0.0, 0, 0, 0);
    return trimed_image;
}

Image WeghtingMapGenerate(int dim_x, int dim_y, float sigma) {
    Curve weighting_curve;
    Image weighting_image;
    int   curve_points;
    curve_points = int((wxMax(dim_x, dim_y) / 2.0 + 1.0) * sqrtf(2.0) + 1.0);
    weighting_curve.SetupXAxis(0.0, 0.5 * sqrtf(2.0), curve_points);
    for ( int i = 0; i < weighting_curve.number_of_points; i++ ) {
        float x = weighting_curve.data_x[i];
        // weighting_curve.AddValueAtXUsingLinearInterpolation(x, exp(-pow(x, 2) / 0.2), 1); //curve
        // weighting_curve.AddValueAtXUsingLinearInterpolation(x, exp(-pow(x / theta, 2) / 2.0) / theta / sqrtf(2 * PI), 1); //newcurve
        // wxPrintf("bin=%i, x= %g, y=%g\n", i, weighting_curve.data_x[i], weighting_curve.data_y[i]);
        weighting_curve.AddValueAtXUsingLinearInterpolation(x, exp(-pow(x / sigma, 2) / 2.0), 1); //curve
    }
    // weighting_curve.WriteToFile(outputpath + "weighting_newcurve.txt");
    //write out the weighting map for peak

    // wxPrintf("physical address of box center %i, %i \n", ref_image.physical_address_of_box_center_x, ref_image.physical_address_of_box_center_y);
    // wxPrintf("image dimensin: %i\n", ref_image.ReturnLargestLogicalDimension( ));
    // wxPrintf("half image: %i \n ", int(ref_image.ReturnLargestLogicalDimension( ) / 2.0 + 1.0));
    // wxPrintf("no. of weighting points: %i \n", int((ref_image.ReturnLargestLogicalDimension( ) / 2.0 + 1.0) * sqrtf(2.0) + 1.0));

    weighting_image.Allocate(dim_x, dim_y, true);
    weighting_image.SetToConstant(1.0);
    weighting_image.MultiplyByWeightsCurveReal(weighting_curve, 1.0);
    // weightingmap_image.WriteSlicesAndFillHeader(outputpathstd + "newweightingmap.mrc", 1);
    return weighting_image;
}

Image CosStretch(Image input_image, float phi, float trim_ratio) {
    Image stretched_img;
    Image cos_clipped_image;
    int   trim_dim_X, trim_dim_Y;
    // trim_ratio = cosf(tilts[ref_index - 1]) - 1.02 * cosf(tilts[current_index - 1]);
    trim_dim_X = myroundint(input_image.logical_x_dimension * (1 - fabs(cosf(phi)) * (trim_ratio)));
    trim_dim_Y = myroundint(input_image.logical_y_dimension * (1 - fabs(sinf(phi)) * (trim_ratio)));

    cos_clipped_image = ImageTrim(input_image, trim_dim_X, trim_dim_Y);
    stretched_img     = ImageStretch(cos_clipped_image, input_image.logical_x_dimension, input_image.logical_y_dimension, cos_clipped_image.ReturnAverageOfRealValues( ));
    // wxPrintf("current image: %i, %i     stretched image: %i, %i \n", input_image.logical_x_dimension, input_image.logical_y_dimension,
    //  cos_clipped_image.logical_x_dimension, cos_clipped_image.logical_y_dimension);
    return stretched_img;
}

float** CoarseAlign(MRCFile* input_stack, float* tilts, float phi_ang, wxString outputpath, float sigma_l, float sigma_h, float radius_l, float radius_h) {

    int      image_no         = input_stack->ReturnNumberOfSlices( );
    MRCFile* stretch_check_st = new MRCFile[image_no];
    // wxPrintf("images: %d", img_no);
    // float** a = NULL;
    // Allocate2DFloatArray(a, img_no, 2);
    // for ( int i = 0; i < img_no; i++ ) {
    //     a[i][0] = i;
    //     a[i][1] = i * i;
    // }
    // return a;
    NumericTextFile *shift_file, *peak_points, *peak_points_raw;
    shift_file      = new NumericTextFile(outputpath + "shifts_newcurved.txt", OPEN_TO_WRITE, 3);
    peak_points     = new NumericTextFile(outputpath + "peakpoints_newcurved.txt", OPEN_TO_WRITE, 4);
    peak_points_raw = new NumericTextFile(outputpath + "peakpoints_pk_img.txt", OPEN_TO_WRITE, 4);
    Image weightingmap_image;
    Image current_image, ref_image;
    Image tmp_image;
    // dimension parameters ----------------------------------
    int raw_image_dim_x, raw_image_dim_y;
    int B_image_dim_x, B_image_dim_y;
    int P_image_dim_x, P_image_dim_y;
    int T_edge_dim_x, T_edge_dim_y;
    int image_dim_x, image_dim_y;
    int trim_dim_X, trim_dim_Y;
    int pad_dim;
    int bin;
    // float rone;
    float phi = deg_2_rad(phi_ang);
    // float phi     = 0.0 / 180 * PI;
    // float phi_ang = 86.3;
    float rone      = 0.0 / 180 * PI;
    float cosphi    = cosf(phi);
    float sinphi    = sinf(phi);
    raw_image_dim_x = input_stack->ReturnXSize( );
    raw_image_dim_y = input_stack->ReturnYSize( );
    float stretch[image_no];
    // float shifts[image_no][2];
    float   peaks[image_no][2];
    float** shifts = NULL;
    Allocate2DFloatArray(shifts, image_no, 2);

    if ( wxMin(raw_image_dim_x, raw_image_dim_y) <= 1024 ) {
        bin = 1;
    }
    else {
        // wxPrintf("bin1, bin2: %g, %g\n", float(raw_image_dim_x) / 1024., float(raw_image_dim_y) / 1024.);
        wxPrintf("bin1, bin2: %i, %i\n", myroundint(float(raw_image_dim_x) / 1024.), myroundint(float(raw_image_dim_y) / 1024.));
        // bin = wxMax(int(round(raw_image_dim_x / 1024)), int(round(raw_image_dim_y / 1024)));
        bin = wxMax(myroundint(float(raw_image_dim_x) / 1024.), myroundint(float(raw_image_dim_y) / 1024.));
    }
    if ( bin > 4 ) {
        bin = 4;
    }

    // bin = 4;
    wxPrintf("raw dimmension: %i, %i\n", raw_image_dim_x, raw_image_dim_y);
    B_image_dim_x = ceilf(float(raw_image_dim_x) / bin);
    B_image_dim_y = ceilf(float(raw_image_dim_y) / bin);
    // B_image_dim_x = 950;
    // B_image_dim_y = 950;
    wxPrintf("binning :%i\tdimension after binning x,y: %i, %i \n", bin, B_image_dim_x, B_image_dim_y);
    P_image_dim_x = myroundint(1.2 * B_image_dim_x);
    P_image_dim_y = myroundint(1.2 * B_image_dim_y);
    // P_image_dim_x = 1140;
    // P_image_dim_y = 1140;
    wxPrintf("binning :%i\tdimension after padding x,y: %i, %i \n", bin, P_image_dim_x, P_image_dim_y);
    // T_edge_dim_x = myroundint(0.1 * B_image_dim_x);
    // T_edge_dim_y = myroundint(0.1 * B_image_dim_y);
    T_edge_dim_x = (P_image_dim_x - B_image_dim_x) / 2.0;
    T_edge_dim_y = (P_image_dim_y - B_image_dim_y) / 2.0;
    int cos_edge = wxMin(T_edge_dim_x, T_edge_dim_y);
    wxPrintf("Taper edge width x,y: %i, %i \n", T_edge_dim_x, T_edge_dim_y);
    image_dim_x = P_image_dim_x;
    image_dim_y = P_image_dim_y;

    // int new_z_dimension = ceilf(float(logical_z_dimension) / bin_z);
    // filtering parameters ----------------------------------
    // good set
    // float sigma_l  = 0.05; // sigma regarding gaussian distribution. at 3 sigma, the probability down to 0.00135
    // float sigma_h  = 0.01;
    // float radius_l = 0.0;
    // float radius_h = 0.0;

    // // test set
    // float sigma_l  = 0.05; // sigma regarding gaussian distribution. at 3 sigma, the probability down to 0.00135
    // float sigma_h  = 0.03;
    // float radius_l = 0.25 / 1.414;
    // float radius_h = 0.0;

    // //---------------------------------- loading the tilts into an array -------------------------
    // for ( int i = 0; i < image_no; i++ ) {
    //     tilt_angle_file->ReadLine(&tilts[i]);
    //     // wxPrintf("angle %i ; % g\n", i, tilts[i]);
    // }
    // for ( int i = 0; i < image_no; i++ ) {
    //     imod_peak_file->ReadLine(imodpeaks[i]);
    // }
    float start_angle  = tilts[0];
    float tilt_step    = tilts[1] - tilts[0];
    int   center_index = int(-start_angle / tilt_step) + 1;

    for ( int i = 0; i < image_no; i++ )
        tilts[i] = (tilts[i]) / 180.0 * PI;
    // center_index = int(image_no / 2);

    wxPrintf("image number in the stack: %i\n", image_no);
    wxPrintf("tomo angles: \nstart angle is %g, tilt steps %g \n", start_angle, tilt_step);
    wxPrintf("center index is: %i\n", center_index);

    //=========================================image processing ========================================

    // basic parameters ---------------------------------------

    Peak  peak;
    float tmppeak[4];

    // whole block
    //============================do cross correlation to calculate peak image================================================
    //initialize the weighting map for peak------------------------------------
    weightingmap_image = WeghtingMapGenerate(image_dim_x, image_dim_y, 0.1);
    Image cos_clipped_image, cos_stretched_image;
    Image peak_image;

    peak_image.Allocate(image_dim_x, image_dim_y, true);
    float trim_ratio;
    int   current_index, ref_index;
    float xpeak[10];
    float ypeak[10];
    float valpeak[10];
    Peak  peak1, peak2;
    float width;
    float widthMin;
    int   maxPeaks    = 10;
    float minStrength = 0.05;
    auto  findpeak    = [&]( ) {
        // calculate the peak image -----------------------------------------------------------------
        peak_image.CopyFrom(&cos_stretched_image);
        peak_image.CalculateCrossCorrelationImageWith(&ref_image); // cos_stretched_image.Normalize( );
        // peak_image.ZeroFloatAndNormalize( );
        // cos_stretched_image.MultiplyByWeightsCurveReal(weighting_curve, 1.0);
        // cos_stretched_image.CalculateCrossCorrelationImageWith(&weightingmap_image);
        // peak1 = peak_image.FindPeakWithIntegerCoordinates(0, 1000, 10);

        // // find peak list
        // peak_image.XCorrPeakFindWidth(image_dim_x + 2, image_dim_y, xpeak, ypeak,
        //                                   valpeak, &width, &widthMin, maxPeaks, minStrength);
        // wxPrintf("peaks: %g, %g, %g\n", xpeak[0] - image_dim_x / 2, ypeak[0] - image_dim_y / 2, valpeak[0]);
        // wxPrintf("peaks: %g, %g, %g\n", xpeak[1] - image_dim_x / 2, ypeak[1] - image_dim_y / 2, valpeak[1]);
        // wxPrintf("peaks: %g, %g, %g\n", xpeak[2] - image_dim_x / 2, ypeak[2] - image_dim_y / 2, valpeak[2]);

        peak1 = peak_image.FindPeakWithIntegerCoordinates( );
        // peak1 = peak_image.FindPeakWithParabolaFit( );
        // peak1 = peak_image.FindPeakAtOriginFast2D(1000, 1000);

        wxPrintf("cistem peak1: x y value %g, %g, %g\n", peak1.x, peak1.y, peak1.value);
        peak_image.WriteSlicesAndFillHeader(wxString::Format(outputpath + "peak1%02i_%.3f_%.3f.mrc", current_index, sigma_l, sigma_h).ToStdString( ), true);
        tmp_image.CopyFrom(&ref_image);
        tmp_image.PhaseShift(peak1.x, peak1.y, 0);
        // tmp_image.WriteSlice(&stretch_check_st[current_index - 1], 3);

        // // // peak refine
        // cos_stretched_image.PhaseShift(-peak1.x, -peak1.y, 0);
        // cos_stretched_image = ImageFilter(cos_stretched_image, 0.2, 0.05, 0, 0.03);
        // ref_image           = ImageFilter(ref_image, 0.2, 0.05, 0, 0.03);
        // cos_stretched_image.CalculateCrossCorrelationImageWith(&ref_image);
        // // cos_stretched_image.MultiplyPixelWise(weightingmap_image);
        // // cos_stretched_image.WriteSlicesAndFillHeader(wxString::Format(outputpath + "peak2%02i_%.3f_%.3f.mrc", current_index, sigma_l, sigma_h).ToStdString( ), true);

        // // // peak_image.MultiplyPixelWise(weightingmap_image);
        // peak2 = cos_stretched_image.FindPeakWithIntegerCoordinates(0, 400, 10);
        // wxPrintf("weighted peak: x y %g, %g\n", peak2.x, peak2.y);
        // just test, currently only used peak1
        // peak2 = peak_image.MultiplyByWeightsCurveReal(weighting_curve, 1.0);
        peak = peak1;
    };
    Image padded_current;
    Image padded_ref;
    padded_current.Allocate(image_dim_x, image_dim_y, true);
    padded_ref.Allocate(image_dim_x, image_dim_y, true);

    Peak  max_peak;
    int   best_trim_X, best_trim_Y;
    float best_ratio;
    float best_cur_tilt, best_ref_tilt, best_stretch;
    float forward_xf[6];
    float bacward_xf[6];
    Image binned_current;
    Image binned_ref;
    binned_current.Allocate(B_image_dim_x, B_image_dim_y, true);
    binned_ref.Allocate(B_image_dim_x, B_image_dim_y, true);

    for ( int i = 1; i <= image_no; i++ ) {

        // for ( int i = 1; i <= 1; i++ ) {
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
        wxPrintf("--------image index-----------%i\n", current_index);
        current_image.ReadSlice(input_stack, current_index);
        ref_image.ReadSlice(input_stack, ref_index);

        // preprocessing: remove outlier and binning ------------------------------------------------------------------------
        // current_image = ImageProc(current_image, bin);
        // ref_image     = ImageProc(ref_image, bin);

        // current_image.ReplaceOutliersWithMean(3.0f);
        // float meancur = current_image.ReturnAverageOfRealValues( );
        // current_image.ForwardFFT( );
        // current_image.ClipInto(&binned_current, meancur, true, 1.0);
        // binned_current.BackwardFFT( );
        // current_image.CopyFrom(&binned_current);

        // ref_image.ReplaceOutliersWithMean(3.0f);
        // float meanref = ref_image.ReturnAverageOfRealValues( );
        // ref_image.ForwardFFT( );
        // ref_image.ClipInto(&binned_ref, meanref, true, 1.0);
        // binned_ref.BackwardFFT( );
        // ref_image.CopyFrom(&binned_ref);
        current_image.RealSpaceBinning(bin, bin, 1);
        ref_image.RealSpaceBinning(bin, bin, 1);

        // image writing
        current_image.WriteSlicesAndFillHeader(wxString::Format(outputpath + "input_bf_stretch%02i_%.3f.mrc", current_index, sigma_h).ToStdString( ), true);
        // ref_image.WriteSlicesAndFillHeader(wxString::Format(outputpath + "ref%02i_%.3f.mrc", ref_index, sigma_h).ToStdString( ), true);

        // current image stretch------------------------------------------------------------------------------
        //stretch along X and Y, i.e., phi is considered, if phi > 90, then cosf(phi) < 0. so we use abs
        // trim_ratio = cosf(tilts[ref_index - 1]) - cosf(tilts[current_index - 1]);

        // stretch_check_st[current_index - 1].OpenFile(wxString::Format(outputpath + "stretch_check_%02i.mrc", current_index).ToStdString( ), true);
        // current_image.WriteSlice(&stretch_check_st[current_index - 1], 1);

        max_peak.x     = 0;
        max_peak.y     = 0;
        max_peak.value = 0;
        // for ( int j = -5; j < 6; j++ ) {
        // for ( int j = -15; j < 11; j++ ) {
        for ( int j = 1; j < 2; j++ ) {
            // float ratio         = 1 + float(j) / 100;
            // trim_ratio  = cosf(tilts[ref_index - 1]) - ratio * cosf(tilts[current_index - 1]);
            float ratio = rone * j;
            trim_ratio  = cosf(tilts[ref_index - 1] + rone * j) - cosf(tilts[current_index - 1] + rone * j);
            // float stretch;
            stretch[current_index - 1] = cosf(tilts[ref_index - 1] + rone * j) / cosf(tilts[current_index - 1] + rone * j);
            // trim_ratio          = cosf(tilts[ref_index - 1]) + rone * j - cosf(tilts[current_index - 1] + rone * j);
            trim_dim_X = myroundint(B_image_dim_x * (1 - fabs(cosf(phi)) * (trim_ratio)));
            trim_dim_Y = myroundint(B_image_dim_y * (1 - fabs(sinf(phi)) * (trim_ratio)));
            wxPrintf("trim dim: %i, %i\n", trim_dim_X, trim_dim_Y);
            cos_stretched_image = CosStretch(current_image, phi, trim_ratio);
            cos_stretched_image.WriteSlicesAndFillHeader(wxString::Format(outputpath + "input_stretch%02i_%.3f_R%.3f.mrc", current_index, sigma_h, ratio).ToStdString( ), true);

            // image writing -----
            // cos_stretched_image.WriteSlice(&stretch_check_st[current_index - 1], 2);
            // ref_image.WriteSlice(&stretch_check_st[current_index-1], 3);
            // stretch_check_st[current_index-1].CloseFile( );
            // // 2---------set mean 0--------------------
            wxPrintf("shift mean value\n");
            // cos_stretched_image.AddConstant(-cos_stretched_image.ReturnAverageOfRealValues( ));
            // ref_image.AddConstant(-ref_image.ReturnAverageOfRealValues( ));
            // cos_stretched_image.ZeroFloatAndNormalize(10.0);
            // ref_image.ZeroFloatAndNormalize(10.0);
            // cos_stretched_image.Normalize( );
            // ref_image.Normalize( );
            // cos_stretched_image.WriteSlicesAndFillHeader(wxString::Format(outputpath + "input_norm%02i_%.3f_R%.3f.mrc", current_index, sigma_h, ratio).ToStdString( ), true);

            //------------tapering ------------------------------
            // taper method 1
            cos_stretched_image.TaperEdges( );
            ref_image.TaperEdges( );
            // cos_stretched_image.WriteSlicesAndFillHeader(wxString::Format(outputpath + "input_tapedge1%02i_%.3f_R%.3f.mrc", current_index, sigma_h, ratio).ToStdString( ), true);

            // taper method 2
            cos_stretched_image.CosineRectangularMask(B_image_dim_x / 2.0 - float(cos_edge / 2.0), B_image_dim_y / 2.0 - float(cos_edge / 2.0), 1, float(cos_edge));
            ref_image.CosineRectangularMask(B_image_dim_x / 2.0 - float(cos_edge / 2.0), B_image_dim_y / 2.0 - float(cos_edge / 2.0), 1, float(cos_edge));

            // taper method 3
            // cos_stretched_image.SetToConstant(1.0);
            // cos_stretched_image.TaperLinear(T_edge_dim_x, T_edge_dim_y, 1, B_image_dim_x / 2.0, B_image_dim_y / 2.0);
            // ref_image.TaperLinear(T_edge_dim_x, T_edge_dim_y, 1, B_image_dim_x / 2.0, B_image_dim_y / 2.0);
            cos_stretched_image.WriteSlicesAndFillHeader(wxString::Format(outputpath + "input_taper%02i_%.3f_R%.3f.mrc", current_index, sigma_h, ratio).ToStdString( ), true);

            // cos_stretched_image.WriteSlicesAndFillHeader(wxString::Format(outputpath + "input_tapnofilt%02i_%.3f_R%.3f.mrc", current_index, sigma_h, ratio).ToStdString( ), true);

            //-----------------------padding------------------------------------
            cos_stretched_image.ClipInto(&padded_current, cos_stretched_image.ReturnAverageOfRealValuesOnEdges( ));
            cos_stretched_image.CopyFrom(&padded_current);
            ref_image.ClipInto(&padded_ref, ref_image.ReturnAverageOfRealValuesOnEdges( ));
            ref_image.CopyFrom(&padded_ref);
            cos_stretched_image.WriteSlicesAndFillHeader(wxString::Format(outputpath + "input_taperpad%02i_%.3f_R%.3f.mrc", current_index, sigma_h, ratio).ToStdString( ), true);

            cos_stretched_image.ZeroFloatAndNormalize(10.0);
            ref_image.ZeroFloatAndNormalize(10.0);
            // wxPrintf("mean and std: %g %g\n", cos_stretched_image.ReturnAverageOfRealValues( ), cos_stretched_image.ReturnSigmaNoise( ));
            // cos_stretched_image.WriteSlicesAndFillHeader(wxString::Format(outputpath + "test%02i_%.3f_R%.3f.mrc", current_index, sigma_h, ratio).ToStdString( ), true);
            cos_stretched_image.WriteSlicesAndFillHeader(wxString::Format(outputpath + "input_nofilt%02i_%.3f_R%.3f.mrc", current_index, sigma_h, ratio).ToStdString( ), true);

            // image filtering ------------------------------------------------------------------------------
            wxPrintf("filter\n");
            cos_stretched_image = ImageFilter(cos_stretched_image, radius_l, sigma_l, radius_h, sigma_h);
            ref_image           = ImageFilter(ref_image, radius_l, sigma_l, radius_h, sigma_h);

            wxPrintf("writing\n");
            cos_stretched_image.WriteSlicesAndFillHeader(wxString::Format(outputpath + "input%02i_%.3f_R%.3f.mrc", current_index, sigma_h, ratio).ToStdString( ), true);
            ref_image.WriteSlicesAndFillHeader(wxString::Format(outputpath + "ref%02i_%.3f.mrc", ref_index, sigma_h).ToStdString( ), true);

            // // calculate the peak image -----------------------------------------------------------------
            findpeak( );
            // wxPrintf("ratio: %g, peak: %g %g %g\n", ratio, peak.x, peak.y, peak.value);
            // if ( j == 0 ) {
            //     max_peak    = peak;
            //     best_trim_X = trim_dim_X;
            //     best_trim_Y = trim_dim_Y;
            // }
            // if ( peak.value > max_peak.value ) {
            max_peak = peak;
            // best_trim_X   = trim_dim_X;
            // best_trim_Y   = trim_dim_Y;
            // best_ratio    = ratio;
            best_ref_tilt = tilts[ref_index - 1] + rone * j;
            best_cur_tilt = tilts[current_index - 1] + rone * j;
            best_stretch  = abs(stretch[current_index - 1]);
            // ang_tilt_cur =
            wxPrintf("stretch, tilt_cur, tilt_ref %i are %g, %g, %g\n", current_index, stretch[current_index - 1], tilts[current_index - 1] / PI * 180.0, tilts[ref_index - 1] / PI * 180.0);
            wxPrintf("update best ratio\n");
            // }
            // peak = max_peak;
        }
        peak = max_peak;
        // wxPrintf("best ratio for %i is %g\n", current_index, best_ratio);
        // stretch_check_st[current_index - 1].CloseFile( );
        //peak fixing ------------------------------
        peaks[current_index - 1][0] = peak.x;
        peaks[current_index - 1][1] = peak.y;
        tmppeak[0] = current_index, tmppeak[1] = peak.x, tmppeak[2] = peak.y, tmppeak[3] = peak.value;
        peak_points_raw->WriteLine(tmppeak);
        // peaks[current_index - 1][0] = peak.x / image_dim_x * trim_dim_X;
        // peaks[current_index - 1][1] = peak.y / image_dim_y * trim_dim_Y;
        // rotation matrix application:
        // rotmagstrToAmat(0, 1.0, best_stretch, phi * PI / 180, &forward_xf[0], &forward_xf[2], &forward_xf[1], &forward_xf[3]);
        rotmagstrToAmat(0, 1.0, best_stretch, phi_ang, &forward_xf[0], &forward_xf[2], &forward_xf[1], &forward_xf[3]);
        forward_xf[4] = 0, forward_xf[5] = 0;
        xfInvert(forward_xf, bacward_xf, 2);
        // wxPrintf("fs              %g, %g, %g, %g, %g, %g \n", forward_xf[0], forward_xf[1], forward_xf[2], forward_xf[3], forward_xf[4], forward_xf[5]);
        // wxPrintf("finv            %g, %g, %g, %g, %g, %g \n", bacward_xf[0], bacward_xf[1], bacward_xf[2], bacward_xf[3], bacward_xf[4], bacward_xf[5]);

        xfApply(bacward_xf, 0.0, 0.0, peak.x, peak.y, &peaks[current_index - 1][0], &peaks[current_index - 1][1], 2);
        // xfApply(bacward_xf, 0.0, 0.0, imodpeaks[current_index - 1][0], imodpeaks[current_index - 1][1], &imodpeaks_cor[current_index - 1][0], &imodpeaks_cor[current_index - 1][1], 2);
        // xfApply(bacward_xf, 0.0, 0.0, 1.0, 1.0, &peaks[current_index - 1][0], &peaks[current_index - 1][1], 2);
        // peaks[current_index - 1][0] = peak.x / image_dim_x * best_trim_X;
        // peaks[current_index - 1][1] = peak.y / image_dim_y * best_trim_Y;
        tmppeak[0] = current_index, tmppeak[1] = peaks[current_index - 1][0], tmppeak[2] = peaks[current_index - 1][1], tmppeak[3] = peak.value;
        peak_points->WriteLine(tmppeak);

        // tmppeak[0] = current_index, tmppeak[1] = imodpeaks[current_index - 1][0], tmppeak[2] = imodpeaks[current_index - 1][1], tmppeak[3] = imodpeaks[current_index - 1][2];
        // imod_peak_points_raw->WriteLine(tmppeak);
        // tmppeak[0] = current_index, tmppeak[1] = imodpeaks_cor[current_index - 1][0], tmppeak[2] = imodpeaks_cor[current_index - 1][1], tmppeak[3] = imodpeaks[current_index - 1][2];
        // imod_peak_points->WriteLine(tmppeak);

        wxPrintf("image: %i, peak position: x = %g, y = %g ,value = %g\n\n", current_index, peak.x, peak.y, peak.value);
        // wxPrintf("image: %i, imod peak pos: x = %g, y = %g ,value = %g\n\n", current_index, imodpeaks[current_index - 1][0], imodpeaks[current_index - 1][1], imodpeaks[current_index - 1][2]);

        wxPrintf("image: %i, correctedpeak: x = %g, y = %g ,value = %g\n\n", current_index, peaks[current_index - 1][0], peaks[current_index - 1][1], peak.value);
        // wxPrintf("image: %i, correctedImod: x = %g, y = %g ,value = %g\n\n", current_index, imodpeaks_cor[current_index - 1][0], imodpeaks_cor[current_index - 1][1], imodpeaks[current_index - 1][2]);

        wxPrintf("image: %i, unbinned correctedpeak: x = %g, y = %g ,value = %g\n\n", current_index, bin * peaks[current_index - 1][0], bin * peaks[current_index - 1][1], peak.value);
        // wxPrintf("image: %i, unbinned imod_cor: x = %g, y = %g ,value = %g\n\n", current_index, bin * imodpeaks_cor[current_index - 1][0], bin * imodpeaks_cor[current_index - 1][1], imodpeaks[current_index - 1][2]);

        // peakfile[current_index - 1].OpenFile(wxString::Format(outputpath + "stretch_newcurvepeak%02i.mrc", current_index).ToStdString( ), true);
        // cos_stretched_image.WriteSlice(&peakfile[current_index - 1], 1);
        // cos_stretched_image.WriteSlicesAndFillHeader(wxString::Format(outputpath + "peak%02i_%.3f_%.3f.mrc", current_index, sigma_l, sigma_h).ToStdString( ), true);

        // wxPrintf("image: %i, correctedpeak: x = %g, y = %g ,value = %g\n\n", current_index, bin * peaks[current_index - 1][0], bin * peaks[current_index - 1][1], peak.value);
    }

    // shifting matrix creating ==============================================================
    //for imod
    // for ( int i = 0; i < image_no; i++ ) {
    //     peaks[i][0] = -imodpeaks_cor[i][0];
    //     peaks[i][1] = -imodpeaks_cor[i][1];
    // }
    int bound = wxMax(image_no - center_index, center_index - 1);
    wxPrintf("image_no, center_index: %i %i \n", image_no, center_index);
    wxPrintf("left right bound: %i %i %i \n", image_no - center_index, center_index - 1, bound);
    int   arr_center_index = center_index - 1;
    float scale_ratio_X, scale_ratio_Y;
    wxPrintf("raw: %i, binned %i\n", raw_image_dim_x, image_dim_x);
    // scale_ratio_X = float(raw_image_dim_x) / float(image_dim_x);
    // scale_ratio_Y = float(raw_image_dim_y) / float(image_dim_y);
    // scale_ratio_X = bin;
    // scale_ratio_Y = bin;
    scale_ratio_X = float(raw_image_dim_x) / float(B_image_dim_x);
    scale_ratio_Y = float(raw_image_dim_y) / float(B_image_dim_y);
    wxPrintf("scale ratio %g %g \n", scale_ratio_X, scale_ratio_Y);
    shifts[arr_center_index][0] = -peaks[arr_center_index][0] * scale_ratio_X;
    shifts[arr_center_index][1] = -peaks[arr_center_index][0] * scale_ratio_Y;

    tmppeak[0] = center_index, tmppeak[1] = shifts[center_index - 1][0], tmppeak[2] = shifts[center_index - 1][1];
    shift_file->WriteLine(tmppeak);
    // imod_shift_file->WriteLine(tmppeak);

    float cos_ratio_tmp;
    float axis_adjust_veclen, axis_adjust;

    for ( int i = 1; i <= bound; i++ ) {
        if ( arr_center_index + i < image_no ) {
            wxPrintf("centers: %g, %g\n", shifts[arr_center_index + i - 1][0], shifts[arr_center_index + i - 1][1]);
            axis_adjust_veclen = shifts[arr_center_index + i - 1][0] * cosphi + shifts[arr_center_index + i - 1][1] * sinphi;
            axis_adjust        = axis_adjust_veclen * (1.0 / stretch[arr_center_index + i] - 1.);
            wxPrintf("image %i, stretch ratio: %f, axis_adjust: %f \n ", arr_center_index + i, stretch[arr_center_index + i], axis_adjust);
            shifts[arr_center_index + i][0] = -peaks[arr_center_index + i][0] * scale_ratio_X + shifts[arr_center_index + i - 1][0] + axis_adjust * cosphi;
            shifts[arr_center_index + i][1] = -peaks[arr_center_index + i][1] * scale_ratio_Y + shifts[arr_center_index + i - 1][1] + axis_adjust * sinphi;
            tmppeak[0] = arr_center_index + i + 1, tmppeak[1] = shifts[arr_center_index + i][0], tmppeak[2] = shifts[arr_center_index + i][1];
            shift_file->WriteLine(tmppeak);
            // imod_shift_file->WriteLine(tmppeak);
            // #cumXrot = xFromCen * cosPhi + yFromCen * sinPhi
            // #xAdjust = cumXrot * (cosRatio - 1.)
            // #xshift  = xshift + xAdjust * cosPhi
            // #yshift  = yshift + xAdjust * sinPhi
        }
        if ( arr_center_index - i >= 0 ) {
            wxPrintf("centers: %g, %g\n", shifts[arr_center_index - i + 1][0], shifts[arr_center_index - i + 1][1]);
            axis_adjust_veclen = shifts[arr_center_index - i + 1][0] * cosphi + shifts[arr_center_index - i + 1][1] * sinphi;
            axis_adjust        = axis_adjust_veclen * (1.0 / stretch[arr_center_index - i] - 1.);

            shifts[arr_center_index - i][0] = -peaks[arr_center_index - i][0] * scale_ratio_X + shifts[arr_center_index - i + 1][0] + axis_adjust * cosphi;
            shifts[arr_center_index - i][1] = -peaks[arr_center_index - i][1] * scale_ratio_Y + shifts[arr_center_index - i + 1][1] + axis_adjust * sinphi;
            wxPrintf("image %i, stretch ratio: %f, axis_adjust: %f \n ", arr_center_index - i, stretch[arr_center_index - i], axis_adjust);
            tmppeak[0] = arr_center_index - i + 1, tmppeak[1] = shifts[arr_center_index - i][0], tmppeak[2] = shifts[arr_center_index - i][1];
            shift_file->WriteLine(tmppeak);
            // imod_shift_file->WriteLine(tmppeak);
        }
        // wxPrintf
    }
    return shifts;
}

void NikoTestApp::DoInteractiveUserInput( ) {
    UserInput* my_input = new UserInput("Coarse Align", 1.0);

    wxString input_imgstack      = my_input->GetFilenameFromUser("Input image file name", "Name of input image file *.mrc", "input.mrc", true);
    wxString angle_filename      = my_input->GetFilenameFromUser("Tilt Angle filename", "The tilts, *.tlt", "ang.tlt", true);
    float    rotation_angle      = my_input->GetFloatFromUser("rotation angle of the tomography", "phi in degrees, 0.0 for none rotation", "0.0");
    float    sigma_lowpass       = my_input->GetFloatFromUser("sigma of low pass filter", "sigma_lowpass, 0.0 for no gradient at low pass cutoff radius", "0.0");
    float    sigma_highpass      = my_input->GetFloatFromUser("sigma of high pass filter", "sigma_highpass, 0.0 for no gradient at high pass cutoff radius", "0.0");
    float    radius_lowpass      = my_input->GetFloatFromUser("cutoff radius for low pass filter", "lowpass filter radius, 0.5*sqrt(2) for no lowpass", "0.5", 0.0, 0.5 * 0.707);
    float    radius_highpass     = my_input->GetFloatFromUser("cutoff radius for high pass filter", "highpass filter radius, 0.5*sqrt(2) for no highpass", "0.0", 0.0, 0.5 * 0.707);
    bool     write_aligned_stack = my_input->GetYesNoFromUser("Do you want to write the aligned stack?", "Answer yes if you do", "No");

    // float sigma_l  = 0.05; // sigma regarding gaussian distribution. at 3 sigma, the probability down to 0.00135
    // float sigma_h  = 0.03;
    // float radius_l = 0.25 / 1.414;
    // float radius_h = 0.0;

    // wxString coordinates_filename  = my_input->GetFilenameFromUser("Coordinates (PLT) filename", "The input particle coordinates, in Imagic-style PLT forlmat", "coos.plt", true);
    // int output_stack_box_size = my_input->GetIntFromUser("Box size for output candidate particle images (pixels)", "In pixels. Give 0 to skip writing particle images to disk.", "256", 0);
    // wxString input_parameters = my_input->GetFilenameFromUser("Input Paramter filename", "input parameters, *.par", "input_para.txt", true);
    delete my_input;

    my_current_job.Reset(8);
    my_current_job.ManualSetArguments("ttfffffb", input_imgstack.ToUTF8( ).data( ), angle_filename.ToUTF8( ).data( ), rotation_angle, sigma_lowpass, sigma_highpass, radius_lowpass, radius_highpass, write_aligned_stack);
}

bool NikoTestApp::DoCalculation( ) {
    wxPrintf("Hello world4\n");
    // ===========================image stack parameters initialization======================================
    // user passed parameters-------------
    wxString input_imgstack      = my_current_job.arguments[0].ReturnStringArgument( );
    wxString angle_filename      = my_current_job.arguments[1].ReturnStringArgument( );
    float    phi_ang             = my_current_job.arguments[2].ReturnFloatArgument( );
    float    sigma_l             = my_current_job.arguments[3].ReturnFloatArgument( ); // sigma regarding gaussian distribution. at 3 sigma, the probability down to 0.00135
    float    sigma_h             = my_current_job.arguments[4].ReturnFloatArgument( );
    float    radius_l            = my_current_job.arguments[5].ReturnFloatArgument( );
    float    radius_h            = my_current_job.arguments[6].ReturnFloatArgument( );
    bool     write_aligned_stack = my_current_job.arguments[7].ReturnBoolArgument( );

    // wxString outputpath = "/groups/lingli/Documents/cisTEM/build/tomoalign_Intel-gpu-debug-static/src/Test/";
    wxString outputpath = "/data/lingli/Lingli_20221028/grid2_process/MotCor202301/AlignTest1";

    float phi    = deg_2_rad(phi_ang);
    float rone   = 0.0 / 180 * PI;
    float cosphi = cosf(phi);
    float sinphi = sinf(phi);

    // local parameters-------------------
    MRCFile          input_stack(input_imgstack.ToStdString( ), false), stretched_stack;
    NumericTextFile *input_coos_file, *tilt_angle_file, *peak_points, *shift_file, *peak_points_raw;
    NumericTextFile *imod_peak_points, *imod_peak_points_raw, *imod_peak_file, *imod_shift_file;
    NumericTextFile* input_par_file;
    tilt_angle_file = new NumericTextFile(angle_filename, OPEN_TO_READ, 1);
    // imod_peak_file  = new NumericTextFile("/groups/lingli/Documents/cisTEM/build/tomoalign_Intel-gpu-debug-static/src/cisTEM_peaks/peaks_in_order.txt", OPEN_TO_READ, 3);
    imod_peak_file = new NumericTextFile("/groups/lingli/Documents/cisTEM/build/tomoalign_Intel-gpu-debug-static/src/cisTEM_peaks/imodrawpeaks_in_order.txt", OPEN_TO_READ, 3);
    // input_par_file  = new NumericTextFile(input_parameters, OPEN_TO_READ, 1);
    peak_points     = new NumericTextFile(outputpath + "peakpoints_newcurved.txt", OPEN_TO_WRITE, 4);
    peak_points_raw = new NumericTextFile(outputpath + "peakpoints_pk_img.txt", OPEN_TO_WRITE, 4);
    // imod_peak_points     = new NumericTextFile(outputpath + "imod_peakpoints_newcurved.txt", OPEN_TO_WRITE, 4);
    // imod_peak_points_raw = new NumericTextFile(outputpath + "imod_peakpoints_pk_img.txt", OPEN_TO_WRITE, 4);
    shift_file = new NumericTextFile(outputpath + "shifts_newcurved.txt", OPEN_TO_WRITE, 3);
    // imod_shift_file      = new NumericTextFile(outputpath + "imod_shift_file.txt", OPEN_TO_WRITE, 3);

    int      image_no = input_stack.ReturnNumberOfSlices( );
    MRCFile* peakfile = new MRCFile[image_no];

    int   center_index;
    float tilts[image_no], start_angle, tilt_step; // angle related
    float stretch[image_no];
    // float shifts[image_no][2];
    float shifts_imod[image_no][2];
    float peaks[image_no][2];
    float imodpeaks[image_no][3];
    float imodpeaks_cor[image_no][2];
    // float imodpeak_tmp[3];

    ProgressBar* my_progress = new ProgressBar(input_stack.ReturnNumberOfSlices( ));
    Image        current_image, ref_image;

    Image weightingmap_image;
    Image tmp_image;

    //---------------------------------- loading the tilts into an array -------------------------
    for ( int i = 0; i < image_no; i++ ) {
        tilt_angle_file->ReadLine(&tilts[i]);
        // wxPrintf("angle %i ; % g\n", i, tilts[i]);
    }

    // cross correlation alignment========================================================================
    float** shifts = NULL;
    Allocate2DFloatArray(shifts, image_no, 2);
    shifts = CoarseAlign(&input_stack, tilts, phi_ang, outputpath, sigma_l, sigma_h, radius_l, radius_h);

    //image stack shift ========================================================================
    if ( write_aligned_stack ) {
        MRCFile output_stack;
        // output_stack.OpenFile(outputpathstd + "outputstack.mrc", true);
        output_stack.OpenFile(outputpath.ToStdString( ) + "outputstack.mrc", true);
        MRCFile rotated_stack;

        // rotated_stack.OpenFile(outputpathstd + "rotatedstack.mrc", true);
        rotated_stack.OpenFile(outputpath.ToStdString( ) + "rotatedstack.mrc", true);
        for ( int i = 0; i < image_no; i++ ) {
            current_image.ReadSlice(&input_stack, i + 1);
            current_image.TaperEdges( );
            current_image.ReplaceOutliersWithMean(3.0f);
            current_image.ZeroFloatAndNormalize(10.0);
            // current_image.PhaseShift(shifts[i][0], shifts[i][1], 0);
            current_image.RealSpaceShift(myroundint(shifts[i][0]), myroundint(shifts[i][1]), 0);
            current_image.WriteSlice(&output_stack, i + 1);
            current_image.Rotate2DInPlace(phi_ang);
            current_image.WriteSlice(&rotated_stack, i + 1);
            // wxPrintf("current slice: %d\n", i);
            // tmppeak[0] = i, tmppeak[1] = shifts[i][0], tmppeak[2] = shifts[i][1];
            // shift_file->WriteLine(tmppeak);
            // current_image.Rotate2D
        }
        output_stack.CloseFile( );
        rotated_stack.CloseFile( );
    }

    // whole block

    // //some image operations testing===============================================
    // Image sample_image;
    // // load a sample image:
    // sample_image.ReadSlice(&input_stack, 4);
    // sample_image.WriteSlicesAndFillHeader(outputpath.ToStdString( ) + "img_original.mrc", 1);
    // sample_image.RealSpaceShift(500, -40, 0);
    // sample_image.WriteSlicesAndFillHeader(outputpath.ToStdString( ) + "img_shifted.mrc", 1);

    // // test bin:
    // // ref_image.RealSpaceBinning(bin, bin, 1);
    // sample_image           = ImageProc(sample_image, bin);
    // int binned_image_dim_x = sample_image.logical_x_dimension;
    // int binned_image_dim_y = sample_image.logical_y_dimension;
    // wxPrintf("image dimension after binning: %i, %i\n", binned_image_dim_x, binned_image_dim_y);
    // sample_image.WriteSlicesAndFillHeader(outputpathstd + "img_binned.mrc", 1);

    // // // // test padding/stretching
    // Image padded_image;
    // pad_dim = 1120;
    // padded_image.Allocate(pad_dim, pad_dim, true);
    // // sample_image.ForwardFFT( );
    // sample_image.ClipInto(&padded_image, sample_image.ReturnAverageOfRealValues( ));
    // // padded_image.BackwardFFT( );

    // // pad_dim      = 1120;
    // // sample_image = ImageStretch(sample_image, pad_dim, pad_dim, 0);
    // padded_image.WriteSlicesAndFillHeader(outputpathstd + "img_binned_padded.mrc", 1);

    // //to end-----------------------
    // // // test filter
    // // sample_image = ImageFilter(sample_image, radius_l, sigma_l, radius_h, sigma_h);
    // // sample_image.WriteSlicesAndFillHeader(outputpathstd + "filteredimg.mrc", 1);

    // // test square mask:
    // sample_image.CosineRectangularMask(mask_radius_x, mask_radius_y, mask_radius_z, mask_edge);
    // sample_image.WriteSlicesAndFillHeader(outputpathstd + "squaremsked.mrc", 1);

    // //to end-----------------------
    // // // test filter
    // sample_image = ImageFilter(sample_image, radius_l, sigma_l, radius_h, sigma_h);
    // sample_image.WriteSlicesAndFillHeader(outputpathstd + "filteredimg.mrc", 1);

    // // //testing end ===============================================
    // test input

    delete my_progress;
    // delete input_coos_file;
    delete[] peakfile;
    // delete[] stretch_check_st;

    // delete[] shift_file;
    // delete[] tmpang;
    // delete[] tilt_angle_file;

    return true;
}