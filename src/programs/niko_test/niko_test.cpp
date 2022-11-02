#include "../../core/core_headers.h"
#include <iostream>
#include <string>
#include <fstream>

class
        NikoTestApp : public MyApp {

  public:
    bool DoCalculation( );
    void DoInteractiveUserInput( );

  private:
};

IMPLEMENT_APP(NikoTestApp)

// override the DoInteractiveUserInput

// void NikoTestApp::DoInteractiveUserInput( ) {
//     UserInput* my_input = new UserInput("TrimStack", 1.0);

//     wxString input_imgstack = my_input->GetFilenameFromUser("Input image file name", "Name of input image file *.mrc", "input.mrc", true);
//     // // wxString output_stack_filename = my_input->GetFilenameFromUser("Filename for output stack of particles.", "A stack of particles will be written to disk", "particles.mrc", false);
//     wxString angle_filename = my_input->GetFilenameFromUser("Tilt Angle filename", "The tilts, *.tlt", "ang.tlt", true);
//     // wxString coordinates_filename  = my_input->GetFilenameFromUser("Coordinates (PLT) filename", "The input particle coordinates, in Imagic-style PLT forlmat", "coos.plt", true);
//     int img_index = my_input->GetIntFromUser("Box size for output candidate particle images (pixels)", "In pixels. Give 0 to skip writing particle images to disk.", "256", 0);

//     delete my_input;

//     my_current_job.Reset(3);
//     my_current_job.ManualSetArguments("tti", input_imgstack.ToUTF8( ).data( ), angle_filename.ToUTF8( ).data( ), img_index);
// }

// override the do calculation method which will be what is actually run..

/*!
 * Finds the coordinates of up to the [maxPeaks] highest peaks in [array], which is 
 * dimensioned to [nxdim] by [ny], and returns the positions in [xpeak], [ypeak], and the 
 * peak values in [peak].  If [minStrength] is greater than 0, then only those peaks that
 * are greater than that fraction of the highest peak will be returned.  In addition, if 
 * [width] and [widthMin] are not NULL, the distance from the
 * peak to the position at half of the peak height is measured in 8 directions, the 
 * overall mean width of the peak is returned in [width], and the minimum width along one
 * of the four axes is returned in [widthMin].  The X size of the 
 * image is assumed to be [nxdim] - 2.  The sub-pixel position is determined by fitting
 * a parabola separately in X and Y to the peak and 2 adjacent points.  Positions
 * are numbered from zero and coordinates bigger than half the image size are
 * shifted to be negative.  The positions are thus the amount to shift a second
 * image in a correlation to align it to the first.  If fewer than [maxPeaks]
 * peaks are found, then the remaining values in [peaks] will be -1.e30.
 */
#define B3DMIN(a, b) ((a) < (b) ? (a) : (b))
#define B3DMAX(a, b) ((a) > (b) ? (a) : (b))
#define B3DCLAMP(a, b, c) a = B3DMAX((b), B3DMIN((c), (a)))

void XCorrPeakFindWidth(float* array, int nxdim, int ny, float* xpeak, float* ypeak,
                        float* peak, float* width, float* widthMin, int maxPeaks,
                        float minStrength) {
    float cx, cy, y1, y2, y3, local, val;
    float widthTemp[4];
    int   ixpeak, iypeak, ix, iy, ixBlock, iyBlock, ixStart, ixEnd, iyStart, iyEnd;
    int   i, j, ixm, ixp, iyb, iybm, iybp, idx, idy;
    int   nx        = nxdim - 2;
    int   blockSize = 5;
    int   nBlocksX  = (nx + blockSize - 1) / blockSize;
    int   nBlocksY  = (ny + blockSize - 1) / blockSize;
    int   ixTopPeak = 10 * nx;
    int   iyTopPeak = 10 * ny;
    float xLimCen, yLimCen, xLimRadSq, yLimRadSq, threshold = 0.;

    float sApplyLimits = -1;
    float sLimitXlo    = -3000;
    float sLimitYlo    = -3000;
    float sLimitXhi    = 3000;
    float sLimitYhi    = 3000;

    /* If using elliptical limits, compute center and squares of radii */
    if ( sApplyLimits < 0 ) {
        xLimCen   = 0.5 * (sLimitXlo + sLimitXhi);
        cx        = B3DMAX(1., (sLimitXhi - sLimitXlo) / 2.);
        xLimRadSq = cx * cx;
        yLimCen   = 0.5 * (sLimitYlo + sLimitYhi);
        cy        = B3DMAX(1., (sLimitYhi - sLimitYlo) / 2.);
        yLimRadSq = cy * cy;
    }

    /* find peaks */
    for ( i = 0; i < maxPeaks; i++ ) {
        peak[i]  = -1.e30f;
        xpeak[i] = 0.;
        ypeak[i] = 0.;
    }

    /* Look for highest peak if looking for one peak or if there is a minimum strength */
    if ( maxPeaks < 2 || minStrength > 0. ) {

        /* Find one peak within the limits */
        if ( sApplyLimits ) {
            for ( iy = 0; iy < ny; iy++ ) {
                idy = (iy > ny / 2) ? iy - ny : iy;
                if ( idy < sLimitYlo || idy > sLimitYhi )
                    continue;
                for ( ix = 0; ix < nx; ix++ ) {
                    idx = (ix > nx / 2) ? ix - nx : ix;
                    if ( idx >= sLimitXlo && idx <= sLimitXhi && array[ix + iy * nxdim] > *peak ) {
                        if ( sApplyLimits < 0 ) {
                            cx = idx - xLimCen;
                            cy = idy - yLimCen;
                            if ( cx * cx / xLimRadSq + cy * cy / yLimRadSq > 1. )
                                continue;
                        }
                        *peak  = array[ix + iy * nxdim];
                        ixpeak = ix;
                        iypeak = iy;
                    }
                }
            }
            if ( *peak > -0.9e30 ) {
                *xpeak = (float)ixpeak;
                *ypeak = (float)iypeak;
            }
        }
        else {

            /* Or just find the one peak in the whole area */
            for ( iy = 0; iy < ny; iy++ ) {
                for ( ix = iy * nxdim; ix < nx + iy * nxdim; ix++ ) {
                    if ( array[ix] > *peak ) {
                        *peak  = array[ix];
                        ixpeak = ix - iy * nxdim;
                        iypeak = iy;
                    }
                }
            }
            *xpeak = (float)ixpeak;
            *ypeak = (float)iypeak;
        }
        threshold = minStrength * *peak;
        ixTopPeak = ixpeak;
        iyTopPeak = iypeak;
    }

    /* Now find all requested peaks */
    if ( maxPeaks > 1 ) {

        // Check for local peaks by looking at the highest point in each local
        // block
        for ( iyBlock = 0; iyBlock < nBlocksY; iyBlock++ ) {

            // Block start and end in Y
            iyStart = iyBlock * blockSize;
            iyEnd   = iyStart + blockSize;
            if ( iyEnd > ny )
                iyEnd = ny;

            // Test if entire block is outside limits
            if ( sApplyLimits && (iyStart > ny / 2 || iyEnd <= ny / 2) ) {
                idy = (iyStart > ny / 2) ? iyStart - ny : iyStart;
                if ( idy > sLimitYhi )
                    continue;
                idy = (iyEnd > ny / 2) ? iyEnd - ny : iyEnd;
                if ( idy < sLimitYlo )
                    continue;
            }

            // Loop on X blocks, get start and end in Y
            for ( ixBlock = 0; ixBlock < nBlocksX; ixBlock++ ) {
                ixStart = ixBlock * blockSize;
                ixEnd   = ixStart + blockSize;
                if ( ixEnd > nx )
                    ixEnd = nx;

                // Test if entire block is outside limits
                if ( sApplyLimits && (ixStart > nx / 2 || ixEnd <= nx / 2) ) {
                    idx = (ixStart > nx / 2) ? ixStart - nx : ixStart;
                    if ( idx > sLimitXhi )
                        continue;
                    idx = (ixEnd > nx / 2) ? ixEnd - nx : ixEnd;
                    if ( idx < sLimitXlo )
                        continue;
                }

                // Loop on every pixel in the block; have to test each pixel
                local = -1.e30f;
                for ( iy = iyStart; iy < iyEnd; iy++ ) {
                    if ( sApplyLimits ) {
                        idy = (iy > ny / 2) ? iy - ny : iy;
                        if ( idy < sLimitYlo || idy > sLimitYhi )
                            continue;
                    }
                    for ( ix = ixStart; ix < ixEnd; ix++ ) {
                        if ( sApplyLimits ) {
                            idx = (ix > nx / 2) ? ix - nx : ix;
                            if ( idx < sLimitXlo || idx > sLimitXhi )
                                continue;

                            // Apply elliptical test
                            if ( sApplyLimits < 0 ) {
                                cx = idx - xLimCen;
                                cy = idy - yLimCen;
                                if ( cx * cx / xLimRadSq + cy * cy / yLimRadSq > 1. )
                                    continue;
                            }
                        }
                        val = array[ix + iy * nxdim];
                        if ( val > local && val > peak[maxPeaks - 1] && val > threshold ) {
                            local  = val;
                            ixpeak = ix;
                            iypeak = iy;
                        }
                    }
                }

                // evaluate local peak for truly being local.
                // Allow equality on one side, otherwise identical adjacent values are lost
                if ( local > -0.9e30 ) {
                    ixm  = (ixpeak + nx - 1) % nx;
                    ixp  = (ixpeak + 1) % nx;
                    iyb  = iypeak * nxdim;
                    iybp = ((iypeak + 1) % ny) * nxdim;
                    iybm = ((iypeak + ny - 1) % ny) * nxdim;

                    if ( local > array[ixpeak + iybm] && local >= array[ixpeak + iybp] &&
                         local > array[ixm + iyb] && local >= array[ixp + iyb] &&
                         local > array[ixm + iybp] && local >= array[ixp + iybm] &&
                         local > array[ixp + iybp] && local >= array[ixm + iybm] &&
                         (ixpeak != ixTopPeak || iypeak != iyTopPeak) ) {

                        // Insert peak into the list
                        for ( i = 0; i < maxPeaks; i++ ) {
                            if ( peak[i] < local ) {
                                for ( j = maxPeaks - 1; j > i; j-- ) {
                                    peak[j]  = peak[j - 1];
                                    xpeak[j] = xpeak[j - 1];
                                    ypeak[j] = ypeak[j - 1];
                                }
                                peak[i]  = local;
                                xpeak[i] = (float)ixpeak;
                                ypeak[i] = (float)iypeak;
                                break;
                            }
                        }
                    }
                }
            }
        }
    }

    // for ( i = 0; i < maxPeaks; i++ ) {
    //     if ( peak[i] < -0.9e30 )
    //         continue;

    // // Add 0.2 just in case float was less than int assigned to it
    // ixpeak = (int)(xpeak[i] + 0.2);
    // iypeak = (int)(ypeak[i] + 0.2);

    // /* simply fit a parabola to the two adjacent points in X or Y */

    // y1 = array[(ixpeak + nx - 1) % nx + iypeak * nxdim];
    // y2 = peak[i];
    // y3 = array[(ixpeak + 1) % nx + iypeak * nxdim];
    // cx = (float)parabolicFitPosition(y1, y2, y3);

    // y1 = array[ixpeak + ((iypeak + ny - 1) % ny) * nxdim];
    // y3 = array[ixpeak + ((iypeak + 1) % ny) * nxdim];
    // cy = (float)parabolicFitPosition(y1, y2, y3);

    // /*    return adjusted pixel coordinate */
    // xpeak[i] = ixpeak + cx;
    // ypeak[i] = iypeak + cy;
    // if ( xpeak[i] > nx / 2 )
    //     xpeak[i] = xpeak[i] - nx;
    // if ( ypeak[i] > ny / 2 )
    //     ypeak[i] = ypeak[i] - ny;

    // /* Return width if non-NULL */
    // if ( width && widthMin ) {
    //     widthTemp[0] = peakHalfWidth(array, ixpeak, iypeak, nx, ny, 1, 0) +
    //                    peakHalfWidth(array, ixpeak, iypeak, nx, ny, -1, 0);
    //     widthTemp[1] = peakHalfWidth(array, ixpeak, iypeak, nx, ny, 0, 1) +
    //                    peakHalfWidth(array, ixpeak, iypeak, nx, ny, 0, -1);
    //     widthTemp[2] = peakHalfWidth(array, ixpeak, iypeak, nx, ny, 1, 1) +
    //                    peakHalfWidth(array, ixpeak, iypeak, nx, ny, -1, -1);
    //     widthTemp[3] = peakHalfWidth(array, ixpeak, iypeak, nx, ny, 1, -1) +
    //                    peakHalfWidth(array, ixpeak, iypeak, nx, ny, -1, 1);
    //     avgSD(widthTemp, 4, &width[i], &cx, &cy);
    //     widthMin[i] = B3DMIN(widthTemp[0], widthTemp[1]);
    //     widthMin[i] = B3DMIN(widthMin[i], widthTemp[2]);
    //     widthMin[i] = B3DMIN(widthMin[i], widthTemp[3]);
    // }
    // }
    sApplyLimits = 0;
}

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

double sliceEdgeMean(float* array, int nxdim, int ixlo, int ixhi, int iylo,
                     int iyhi) {
    double dmean, sum = 0.;
    int    ix, iy;
    for ( ix = ixlo; ix <= ixhi; ix++ )
        sum += array[ix + iylo * nxdim] + array[ix + iyhi * nxdim];

    for ( iy = iylo + 1; iy < iyhi; iy++ )
        sum += array[ixlo + iy * nxdim] + array[ixhi + iy * nxdim];

    dmean = sum / (2 * (ixhi - ixlo + iyhi - iylo));
    return dmean;
}

// /* Find the point in one direction away from the peak pixel where it falls by half */
// static float peakHalfWidth(float* array, int ixPeak, int iyPeak, int nx, int ny, int delx,
//                            int dely) {
//     int   nxdim = nx + 2;
//     float peak  = array[ixPeak + iyPeak * nxdim];
//     int   dist, ix, iy;
//     float scale   = (float)sqrt((double)delx * delx + dely * dely);
//     float lastVal = peak, val;

//     for ( dist = 1; dist < B3DMIN(nx, ny) / 4; dist++ ) {
//         ix  = (ixPeak + dist * delx + nx) % nx;
//         iy  = (iyPeak + dist * dely + ny) % ny;
//         val = array[ix + iy * nxdim];
//         if ( val < peak / 2. )
//             return scale * (float)(dist + (lastVal - peak / 2.) / (lastVal - val) - 1.);
//         lastVal = val;
//     }
//     return scale * (float)dist;
// }
// shift the stack
// void readArray( ) {
//     wxString      inFileName = "/groups/lingli/Documents/cisTEM/build/tomoalign_Intel-gpu-debug-static/src/peak_analysis/shifted_mapx.txt";
//     std::ifstream inFile;
//     inFile.open(inFileName.c_str( ));
//     if ( inFile.is_open( ) ) {
//         wxPrintf("file is open\n");
//         float myarray[10][5760];
//         for ( int j = 0; j < 2; j++ ) {
//             for ( int i = 0; i < 5760; i++ ) {
//                 inFile >> myarray[j][i];
//                 wxPrintf("j i value %i %i %g\n", j, i, myarray[j][i]);
//             }
//         }
//     }
// }

void NikoTestApp::DoInteractiveUserInput( ) {
    UserInput* my_input = new UserInput("TrimStack", 1.0);

    wxString input_image          = my_input->GetFilenameFromUser("Input image file name", "Name of input image file *.mrc", "input.mrc", true);
    wxString peak_filename        = my_input->GetFilenameFromUser("Peak filename", "The file containing peak for each patch, *.txt", "peak_01.txt", true);
    wxString coordinates_filename = my_input->GetFilenameFromUser("Coordinates (PLT) filename", "The input particle coordinates, in Imagic-style PLT forlmat", "coos.plt", true);
    // int      output_stack_box_size = my_input->GetIntFromUser("Box size for output candidate particle images (pixels)", "In pixels. Give 0 to skip writing particle images to disk.", "256", 0);
    // float    rotation_angle        = my_input->GetFloatFromUser("rotation angle of the tomography", "phi in degrees, 0.0 for none rotation", "0.0");
    // wxString shifts_filename       = my_input->GetFilenameFromUser("Shifts filename", "The shifts, *.txt", "shifts.txt", true);
    delete my_input;

    my_current_job.Reset(3);
    my_current_job.ManualSetArguments("ttt", input_image.ToUTF8( ).data( ), peak_filename.ToUTF8( ).data( ), coordinates_filename.ToUTF8( ).data( ));
}

bool NikoTestApp::DoCalculation( ) {
    wxString input_image = my_current_job.arguments[0].ReturnStringArgument( );
    wxString peakfile    = my_current_job.arguments[1].ReturnStringArgument( );
    wxString patchfile   = my_current_job.arguments[2].ReturnStringArgument( );
    MRCFile  input_test(input_image.ToStdString( ), false);

    wxString    outputpath    = "/groups/lingli/Documents/cisTEM/build/tomoalign_Intel-gpu-debug-static/src/tilt05_peak_analysis/";
    std::string outputpathstd = "/groups/lingli/Documents/cisTEM/build/tomoalign_Intel-gpu-debug-static/src/tilt05_peak_analysis/";
    // MRCFile input_test("/groups/lingli/Documents/cisTEM/build/tomoalign_Intel-gpu-debug-static/src/Test-ximina-20220928/outputstack.mrc", false);
    // MRCFile input_test("/groups/lingli/Documents/cisTEM/build/tomoalign_Intel-gpu-debug-static/src/peak_analysis/sample_img.mrc", false);
    // MRCFile input_test("/groups/lingli/Documents/cisTEM/build/tomoalign_Intel-gpu-debug-static/src/unstacked_Ximena_raw/image_000.mrc", false);
    // wxString coordinates_filename = my_input->GetFilenameFromUser("Coordinates (PLT) filename", "The input particle coordinates, in Imagic-style PLT forlmat", "coos.plt", true);
    // wxString coordinates_filename = my_current_job.arguments[2].ReturnStringArgument( );
    NumericTextFile *patch_positions, *peak_positions;
    // wxString         patchfile = "/groups/lingli/Documents/cisTEM/build/tomoalign_Intel-gpu-debug-static/src/patchlocations.plt";
    // wxString         peakfile  = "/groups/lingli/Documents/cisTEM/build/tomoalign_Intel-gpu-debug-static/src/peak_positions.txt";
    // wxString patchfile = "/groups/lingli/Documents/cisTEM/build/tomoalign_Intel-gpu-debug-static/src/coord_TS17.plt";
    // wxString peakfile  = "/groups/lingli/Documents/cisTEM/build/tomoalign_Intel-gpu-debug-static/src/peak_01.txt";

    patch_positions = new NumericTextFile(patchfile, OPEN_TO_READ, 2);
    peak_positions  = new NumericTextFile(peakfile, OPEN_TO_READ, 2);

    // NumericTextFile *shift_filex, *shift_filey;
    // NumericTextFile peak_position;
    // readArray( );
    wxPrintf("start\n");

    Image sample_img;
    Image interp_img, interp_img_tmp;

    int image_x_dim, image_y_dim;
    image_x_dim = input_test.ReturnXSize( );
    image_y_dim = input_test.ReturnYSize( );
    int patch_no;
    patch_no            = patch_positions->number_of_lines;
    int     patch_x_num = 3, patch_y_num = 2;
    int     bin      = 1;
    float** peaks_x  = NULL;
    float** peaks_y  = NULL;
    float** patchs_x = NULL;
    float** patchs_y = NULL;

    Allocate2DFloatArray(peaks_x, patch_y_num, patch_x_num);
    Allocate2DFloatArray(peaks_y, patch_y_num, patch_x_num);
    Allocate2DFloatArray(patchs_x, patch_y_num, patch_x_num);
    Allocate2DFloatArray(patchs_y, patch_y_num, patch_x_num);
    for ( int i = 0; i < patch_y_num; i++ ) {
        float tmppeak[2];
        float tmppatch[2];
        for ( int j = 0; j < patch_x_num; j++ ) {
            patch_positions->ReadLine(tmppatch);
            peak_positions->ReadLine(tmppeak);
            patchs_x[i][j] = tmppatch[0] + 1440;
            patchs_y[i][j] = tmppatch[1] + 1024;
            peaks_x[i][j]  = -tmppeak[0] * bin + patchs_x[i][j];
            peaks_y[i][j]  = -tmppeak[1] * bin + patchs_y[i][j];

            wxPrintf("patchpeaks x y %i, %i, %g, %g, %g, %g\n", i, j, peaks_x[i][j], peaks_y[i][j], patchs_x[i][j], patchs_y[i][j]);
        }
    }

    float** shifted_map_x;
    float** shifted_map_y;
    float** original_map_x;
    float** original_map_y;
    Allocate2DFloatArray(shifted_map_x, image_y_dim, image_x_dim);
    Allocate2DFloatArray(shifted_map_y, image_y_dim, image_x_dim);
    Allocate2DFloatArray(original_map_x, image_y_dim, image_x_dim);
    Allocate2DFloatArray(original_map_y, image_y_dim, image_x_dim);

    // initialize the pixel coordinates
    for ( int i = 0; i < image_y_dim; i++ ) {
        for ( int j = 0; j < image_x_dim; j++ ) {
            shifted_map_x[i][j]  = j;
            original_map_x[i][j] = j;
            shifted_map_y[i][j]  = i;
            original_map_y[i][j] = i;
        }
    }

    // calculate shifte amount along x
    for ( int i = 0; i < patch_y_num - 1; i++ ) {
        for ( int j = 0; j < patch_x_num - 1; j++ ) {
            int   interval_x_no     = int(patchs_x[i][j + 1] - patchs_x[i][j]);
            int   interval_y_no     = -int(patchs_y[i + 1][j] - patchs_y[i][j]);
            float SHX_interval_0_1  = (peaks_x[i][j + 1] - peaks_x[i][j]) / interval_x_no;
            float SHX_interval_2_3  = (peaks_x[i + 1][j + 1] - peaks_x[i + 1][j]) / interval_x_no;
            float SHX_interval_diff = (SHX_interval_0_1 - SHX_interval_2_3) / interval_y_no;

            float SHY_interval_2_0  = -(peaks_y[i + 1][j] - peaks_y[i][j]) / interval_y_no;
            float SHY_interval_3_1  = -(peaks_y[i + 1][j + 1] - peaks_y[i][j + 1]) / interval_y_no;
            float SHY_interval_diff = (SHY_interval_3_1 - SHY_interval_2_0) / interval_x_no;

            float ref_SHX_interval = SHX_interval_2_3;
            float ref_SHX_location = peaks_x[i + 1][j];
            float ref_SHY_interval = SHY_interval_2_0;
            float ref_SHY_location = peaks_y[i + 1][j];

            float ref_SH_x = peaks_x[i + 1][j];
            float ref_SH_y = peaks_y[i + 1][j];

            int   ystart       = int(patchs_y[i + 1][j]);
            int   yend         = int(patchs_y[i][j]);
            int   xstart       = int(patchs_x[i + 1][j]);
            int   xend         = int(patchs_x[i + 1][j + 1]);
            float slop_along_y = (peaks_x[i][j] - peaks_x[i + 1][j]) / (yend - ystart);
            float slop_along_x = (peaks_y[i + 1][j + 1] - peaks_y[i + 1][j]) / (xend - xstart);

            int yindex_start = ystart;
            int yindex_end   = yend;
            int xindex_start = xstart;
            int xindex_end   = xend;
            if ( i == 0 )
                yindex_end = image_y_dim;
            if ( i == patch_y_num - 2 )
                yindex_start = 0;
            if ( j == 0 )
                xindex_start = 0;
            if ( j == patch_x_num - 2 )
                xindex_end = image_x_dim;

            for ( int yindex = yindex_start; yindex < yindex_end; yindex++ ) {
                float current_interval_x      = ref_SHX_interval + SHX_interval_diff * (yindex - ystart);
                float current_start_locationx = ref_SH_x + slop_along_y * (yindex - ystart);
                for ( int xindex = xindex_start; xindex < xindex_end; xindex++ ) {
                    shifted_map_x[yindex][xindex] = current_start_locationx + (xindex - xstart) * current_interval_x;
                    shifted_map_y[yindex][xindex] = yindex;
                    float current_interval_y      = ref_SHY_interval + SHY_interval_diff * (xindex - xstart);
                    float current_start_locationy = ref_SH_y + slop_along_x * (xindex - xstart);
                    shifted_map_y[yindex][xindex] = current_start_locationy + (yindex - ystart) * current_interval_y;
                }
            }
        }
    }

    wxString      shifted_mapx_file = outputpath + "shifted_x.txt";
    wxString      shifted_mapy_file = outputpath + "shifted_y.txt";
    std::ofstream xoFile, yoFile;

    // wxPrintf("1\n");
    // for ( int i = 0; i < 10; i++ ) {
    //     for ( int j = 0; j < 10; j++ ) {
    //         shifted_map_x[i][j] = j;
    //         shifted_map_y[i][j] = i;
    //     }
    // }

    xoFile.open(shifted_mapx_file.c_str( ));
    yoFile.open(shifted_mapy_file.c_str( ));
    if ( xoFile.is_open( ) && yoFile.is_open( ) ) {
        wxPrintf("files are open\n");
        // float myarray[10][5760];
        for ( int i = 0; i < image_y_dim; i++ ) {
            for ( int j = 0; j < image_x_dim; j++ ) {
                xoFile << shifted_map_x[i][j] << '\t';
                yoFile << shifted_map_y[i][j] << '\t';
            }
            xoFile << '\n';
            yoFile << '\n';
        }
    }
    xoFile.close( );
    yoFile.close( );

    wxPrintf("1\n");
    sample_img.ReadSlice(&input_test, 1);
    // image_no = input_test.ReturnNumberOfSlices( );
    // image_x_dim = sample_img.logical_x_dimension;
    // image_y_dim = sample_img.logical_y_dimension;
    image_x_dim = input_test.ReturnXSize( );
    image_y_dim = input_test.ReturnYSize( );

    wxPrintf("image dim: %i, %i\n", image_x_dim, image_y_dim);
    interp_img.Allocate(image_x_dim, image_y_dim, true);
    interp_img_tmp.Allocate(image_x_dim, image_y_dim, true);
    interp_img.SetToConstant(sample_img.ReturnAverageOfRealValuesOnEdges( ));
    interp_img_tmp.SetToConstant(sample_img.ReturnAverageOfRealValuesOnEdges( ));
    wxPrintf("2\n");
    // float* shifted_map = new float[image_y_dim][4092][2];
    wxPrintf("image dim: %i, %i\n", image_x_dim, image_y_dim);
    int totalpixels = image_x_dim * image_y_dim;
    wxPrintf("3 total pixels %i\n", totalpixels);
    // float shifted_mapx[totalpixels], shifted_mapy[totalpixels];
    float* shifted_mapx      = new float[totalpixels];
    float* shifted_mapy      = new float[totalpixels];
    float* interpolated_mapx = new float[totalpixels];
    float* interpolated_mapy = new float[totalpixels];

    wxPrintf("3\n");
    wxPrintf("start loading shifted text\n");

    // load array from file
    // wxString      shift_filex = "/groups/lingli/Documents/cisTEM/build/tomoalign_Intel-gpu-debug-static/src/peak_analysis/shifted_mapx.txt";
    // wxString      shift_filey = "/groups/lingli/Documents/cisTEM/build/tomoalign_Intel-gpu-debug-static/src/peak_analysis/shifted_mapy.txt";
    wxString      shift_filex = outputpath + "shifted_x.txt";
    wxString      shift_filey = outputpath + "shifted_y.txt";
    std::ifstream xFile, yFile;

    wxPrintf("1\n");

    xFile.open(shift_filex.c_str( ));
    yFile.open(shift_filey.c_str( ));

    if ( xFile.is_open( ) && yFile.is_open( ) ) {
        wxPrintf("files are open\n");
        // float myarray[10][5760];
        for ( int pix = 0; pix < totalpixels; pix++ ) {
            xFile >> shifted_mapx[pix];
            yFile >> shifted_mapy[pix];
        }
    }
    wxPrintf("shifting files are loaded \n");
    xFile.close( );
    yFile.close( );
    // load array from file end

    // int len = sizeof(shifted_mapx) / sizeof(shifted_mapx[0]);
    int len = *(&shifted_mapx + 1) - shifted_mapx;
    // std::cout << "the size" << std::sizeof(shifted_mapx[0]);
    // wxPrintf("len %i %i %i\n", sizeof(shifted_mapx), sizeof(shifted_mapx[0]), len);
    wxPrintf("len %i\n", len);
    sample_img.Distortion(&interp_img, shifted_mapx, shifted_mapy);
    interp_img.WriteSlicesAndFillHeader(outputpathstd + "interp.mrc", 1);

    delete &sample_img;
    delete &interp_img;
    delete[] shifted_mapx;
    delete[] shifted_mapy;
    Deallocate2DFloatArray(shifted_map_x, image_y_dim);
    Deallocate2DFloatArray(shifted_map_y, image_y_dim);
    Deallocate2DFloatArray(original_map_x, image_y_dim);
    Deallocate2DFloatArray(original_map_y, image_y_dim);

    return true;
}

// //-------------------------------------------test the interpolation---------------------------------------
// bool NikoTestApp::DoCalculation( ) {
//     // MRCFile input_test("/groups/lingli/Documents/cisTEM/build/tomoalign_Intel-gpu-debug-static/src/Test-ximina-20220928/outputstack.mrc", false);
//     MRCFile input_test("/groups/lingli/Documents/cisTEM/build/tomoalign_Intel-gpu-debug-static/src/peak_analysis/sample_img.mrc", false);
//     // NumericTextFile shift_file("/groups/lingli/Documents/cisTEM/build/tomoalign_Intel-gpu-debug-static/src/peak_analysis/newshifts.txt");
//     // NumericTextFile *shift_filex, *shift_filey;
//     // readArray( );
//     wxPrintf("start\n");
//     // shift_filex = new NumericTextFile("/groups/lingli/Documents/cisTEM/build/tomoalign_Intel-gpu-debug-static/src/peak_analysis/shifted_mapx.txt", OPEN_TO_READ, 5760);
//     // wxPrintf("1\n");
//     // shift_filey = new NumericTextFile("/groups/lingli/Documents/cisTEM/build/tomoalign_Intel-gpu-debug-static/src/peak_analysis/shifted_mapy.txt", OPEN_TO_READ, 5760);

//     Image sample_img;
//     Image interp_img, interp_img_tmp;

//     int image_x_dim, image_y_dim;
//     int image_no;
//     wxPrintf("1\n");
//     sample_img.ReadSlice(&input_test, 1);
//     image_no = input_test.ReturnNumberOfSlices( );
//     // image_x_dim = sample_img.logical_x_dimension;
//     // image_y_dim = sample_img.logical_y_dimension;
//     image_x_dim = input_test.ReturnXSize( );
//     image_y_dim = input_test.ReturnYSize( );

//     wxPrintf("image dim: %i, %i\n", image_x_dim, image_y_dim);
//     interp_img.Allocate(image_x_dim, image_y_dim, true);
//     interp_img_tmp.Allocate(image_x_dim, image_y_dim, true);
//     interp_img.SetToConstant(sample_img.ReturnAverageOfRealValuesOnEdges( ));
//     interp_img_tmp.SetToConstant(sample_img.ReturnAverageOfRealValuesOnEdges( ));
//     wxPrintf("2\n");
//     // float* shifted_map = new float[image_y_dim][4092][2];
//     wxPrintf("image dim: %i, %i\n", image_x_dim, image_y_dim);
//     int totalpixels = image_x_dim * image_y_dim;
//     wxPrintf("3 total pixels %i\n", totalpixels);
//     // float shifted_mapx[totalpixels], shifted_mapy[totalpixels];
//     float* shifted_mapx      = new float[totalpixels];
//     float* shifted_mapy      = new float[totalpixels];
//     float* interpolated_mapx = new float[totalpixels];
//     float* interpolated_mapy = new float[totalpixels];

//     wxPrintf("3\n");
//     wxPrintf("start loading shifted text\n");

//     // load array
//     wxString      shift_filex = "/groups/lingli/Documents/cisTEM/build/tomoalign_Intel-gpu-debug-static/src/peak_analysis/shifted_mapx.txt";
//     wxString      shift_filey = "/groups/lingli/Documents/cisTEM/build/tomoalign_Intel-gpu-debug-static/src/peak_analysis/shifted_mapy.txt";
//     std::ifstream xFile, yFile;

//     wxPrintf("1\n");

//     xFile.open(shift_filex.c_str( ));
//     yFile.open(shift_filey.c_str( ));

//     if ( xFile.is_open( ) && yFile.is_open( ) ) {
//         wxPrintf("files are open\n");
//         // float myarray[10][5760];
//         for ( int pix = 0; pix < totalpixels; pix++ ) {
//             xFile >> shifted_mapx[pix];
//             yFile >> shifted_mapy[pix];
//         }
//     }
//     wxPrintf("shifting files are loaded \n");
//     xFile.close( );
//     yFile.close( );
//     // int len = sizeof(shifted_mapx) / sizeof(shifted_mapx[0]);
//     int len = *(&shifted_mapx + 1) - shifted_mapx;
//     // std::cout << "the size" << std::sizeof(shifted_mapx[0]);
//     // wxPrintf("len %i %i %i\n", sizeof(shifted_mapx), sizeof(shifted_mapx[0]), len);
//     wxPrintf("len %i\n", len);
//     sample_img.Distortion(&interp_img, shifted_mapx, shifted_mapy);
//     interp_img.WriteSlicesAndFillHeader("interp.mrc", 1);

//     delete[] shifted_mapx;
//     delete[] shifted_mapy;

//     return true;
// }

// //------------------------------------------- end test the interpolation---------------------------------------

// // test the rotation operation
// bool NikoTestApp::DoCalculation( ) {
//     MRCFile         input_test("/groups/lingli/Documents/cisTEM/build/tomoalign_Intel-gpu-debug-static/src/input_bf_stretch31_0.030.mrc", false);
//     Image           Test_image;
//     Image           Small_image;
//     Image           Rotated_image;
//     AnglesAndShifts AandS;
//     Small_image.Allocate(200, 100, true);

//     // Test_image.SetToConstant(1.0);
//     Test_image.ReadSlice(&input_test, 1);
//     Test_image.ClipInto(&Small_image, Test_image.ReturnAverageOfRealValues( ));
//     Small_image.ForwardFFT( );
//     Small_image.GaussianLowPassRadiusFilter(0.2, 0.01);
//     Small_image.BackwardFFT( );
//     Small_image.WriteSlicesAndFillHeader("test.mrc", 1);
//     // AandS.Init(0, 0, 86.3, 10, 20);
//     // wxPrintf("phi: %g\n", AandS.ReturnPhiAngle( ));
//     // Rotated_image.Allocate(200, 200, true);
//     // Small_image.Rotate2D(Rotated_image, AandS);
//     // Rotated_image.WriteSlicesAndFillHeader("rotated.mrc", 1);
//     Small_image.Rotate2DInPlace(86.3);
//     Small_image.WriteSlicesAndFillHeader("rotated.mrc", 1);
//     Small_image.RealSpaceIntegerShift(10, 50);
//     Small_image.WriteSlicesAndFillHeader("rotatedshifted.mrc", 1);
//     return true;
// }

// bool NikoTestApp::DoCalculation( ) {
//     wxPrintf("hello1\n");

//     Image peak_image, input_image;
//     // MRCFile input_peak("/groups/lingli/Documents/cisTEM/build/tomoalign_Intel-gpu-debug-static/src/Test/peak102_0.050_0.030.mrc", false);
//     // MRCFile input_peak("/groups/lingli/Downloads/TS17/test_coarsealgin/49peak.mrc", false);
//     // MRCFile          input_stack(input_imgstack.ToStdString( ), false)
//     // MRCFile input_peak("/groups/lingli/Downloads/TS17/test_coarsealgin/peaks.mrc", false);
//     MRCFile          input_peak("/groups/lingli/Downloads/TS17/test_coarsealign_1/image_peak.mrc", false);
//     NumericTextFile *tilt_angle_file, *peak_points, *shift_file, *peak_points_raw;
//     // tilt_angle_file = new NumericTextFile('angle_filename/groups/lingli/Downloads/TS17/TS17.rawtlt', OPEN_TO_READ, 1);
//     // peak_points     = new NumericTextFile(outputpath + "peakpoints_newcurved.txt", OPEN_TO_WRITE, 4);
//     // peak_points_raw = new NumericTextFile(outputpath + "peakpoints_pk_img.txt", OPEN_TO_WRITE, 4);
//     // shift_file      = new NumericTextFile(outputpath + "shifts_newcurved.txt", OPEN_TO_WRITE, 3);

//     int   image_no = tilt_angle_file->number_of_lines;
//     float tilts[image_no];
//     float stretch[image_no];

//     float shifts[image_no][2];
//     float peaks[image_no][2];
//     for ( int i = 0; i < image_no; i++ ) {
//         tilt_angle_file->ReadLine(&tilts[i]);
//         // wxPrintf("angle %i ; % g\n", i, tilts[i]);
//     }
//     for ( int i = 0; i < image_no; i++ )
//         tilts[i] = (tilts[i]) / 180.0 * PI;

//     // wxString angle_filename = my_current_job.arguments[1].ReturnStringArgument( );
//     // MRCFile ref_file("/groups/lingli/Documents/cisTEM/build/tomoalign_Intel-gpu-debug-static/src/Output/ref34_0.030.mrc", false);
//     // MRCFile input_file("/groups/lingli/Documents/cisTEM/build/tomoalign_Intel-gpu-debug-static/src/Output/input35_0.030_R0.000.mrc", false);

//     //     MRCFile input_stack(input_imgstack.ToStdString( ), false);
//     // int X_maskcenter, Y_maskcenter;
//     // MRCFile input_test("/groups/lingli/Documents/cisTEM/build/tomoalign_Intel-gpu-debug-static/src/Output/input_bf_stretch35_0.030.mrc", false);

//     int X_dim = input_peak.ReturnXSize( );
//     int Y_dim = input_peak.ReturnYSize( );

//     // float input_array[X_dim][Y_dim];
//     float xpeak[10];
//     float ypeak[10];
//     float peak[10];
//     float width;
//     float widthMin;
//     int   maxPeaks    = 10;
//     float minStrength = 0.05;
//     width             = 10;
//     widthMin          = 1;
//     float fs[6], fiv[6];
//     float a11, a12, a21, a22;
//     // float theta = 50.98;
//     // float ref   = 47.99;
//     float phi = 86.3;

//     wxPrintf("X_dim = %i, Y_dim = %i \n", X_dim, Y_dim);
//     wxPrintf("hello2\n");
//     // float str = fabs(cosf(ref / 180 * PI) / cosf(theta / 180 * PI));
//     // // float str = fabs(cosf(theta / 180 * PI) / cosf(ref / 180 * PI));
//     // // str = fabs((cosf(ref / 180 * PI)) / cosf(theta / 180 * PI));
//     // wxPrintf("str: %g\n", str);
//     // // rotmagstrToAmat(0, 1.0, str, phi, &a11, &a12, &a21, &a22);
//     // rotmagstrToAmat(0, 1.0, str, phi, &a11, &a12, &a21, &a22);
//     // // rotmagstrToAmat(theta, 1.0, str, phi, &a11, &a12, &a21, &a22);
//     // // rotmagstrToAmat(ref - theta, 1.0, str, phi, &a11, &a12, &a21, &a22);
//     // // rotmagstrToAmat(1.0, 1.0, str, 0.0, &a11, &a12, &a21, &a22);
//     // // fs[0] = a11, fs[1] = a12, fs[2] = 0.0;
//     // // fs[3] = a21, fs[4] = a22, fs[5] = 0.0;
//     // // note that the fortran matrix is column dominate, c++ is row dominate
//     // // fs[0] = a11, fs[1] = a12, fs[4] = 0.0;

//     // fs[0] = a11, fs[2] = a12, fs[4] = 0.0;
//     // fs[1] = a21, fs[3] = a22, fs[5] = 0.0;

//     // float fs_a[6], fiv_a[6];

//     // rotmagstrToAmat(0, 1.0, 1, -phi, &a11, &a12, &a21, &a22);
//     // fs_a[0] = a11, fs_a[2] = a12, fs_a[4] = 0.0;
//     // fs_a[1] = a21, fs_a[3] = a22, fs_a[5] = 0.0;

//     // xfInvert(fs_a, fiv_a, 2);

//     // // float unX1, unY1;
//     // // float xorg = -1173.334 / 4, yorg = 1680.392 / 4;
//     // // xfApply(fs, 0.0, 0.0, xorg, yorg, &unX1, &unY1, 2);
//     // // wxPrintf("un1, un2 %g, %g, %g, %g\n", unX1, unY1, unX1 * 4, unY1 * 4);

//     // xfInvert(fs, fiv, 2);

//     // wxPrintf("a11 a12 a21 a22 %g, %g, %g, %g \n", a11, a12, a21, a22);
//     // wxPrintf("fs              %g, %g, %g, %g, %g, %g \n", fs[0], fs[1], fs[2], fs[3], fs[4], fs[5]);
//     // wxPrintf("finv            %g, %g, %g, %g, %g, %g \n", fiv[0], fiv[1], fiv[2], fiv[3], fiv[4], fiv[5]);
//     // wxPrintf("fs_a              %g, %g, %g, %g, %g, %g \n", fs_a[0], fs_a[1], fs_a[2], fs_a[3], fs_a[4], fs_a[5]);
//     // float finv_c[6];
//     // finv_c[0] = fiv[0], finv_c[1] = fiv[2], finv_c[2] = fiv[4];
//     // finv_c[3] = fiv[1], finv_c[4] = fiv[3], finv_c[5] = fiv[5];

//     // padded_dimensions_x = ReturnClosestFactorizedUpper(pad_factor * input_file_2d.ReturnXSize( ), 3);
//     // padded_dimensions_y = ReturnClosestFactorizedUpper(pad_factor * input_file_2d.ReturnYSize( ), 3);

//     //   input_volume.Allocate(input_file_3d.ReturnXSize( ), input_file_3d.ReturnYSize( ), input_file_3d.ReturnZSize( ), true);
//     // circlemask_image.Allocate(X_dim, Y_dim, true);
//     peak_image.Allocate(X_dim, Y_dim, true);
//     // peak_image.ReadSlice(&input_peak, 1);
//     // wxPrintf("hello3\n");
//     // input_image.Allocate(input_test.ReturnXSize( ), input_test.ReturnYSize( ), true);
//     // input_image.ReadSlice(&input_test, 1);
//     // wxPrintf("hello4\n");
//     // int raw_X_dim = 1440;
//     // int raw_Y_dim = 1023;

//     // float mask_radius_x = raw_X_dim / 2.0 - raw_X_dim / 10.0;
//     // float mask_radius_y = raw_Y_dim / 2.0 - raw_Y_dim / 10.0;
//     // float mask_radius_z = 1;
//     // // float mask_edge     = std::max(raw_image_dim_x / bin, raw_image_dim_y / bin) / 4.0;
//     // // float mask_edge = 192;
//     // float mask_edge = std::max(raw_X_dim, raw_Y_dim) / 10.0;
//     // // float wanted_taper_edge_x  = std::max(raw_X_dim, raw_Y_dim) / 10.0;
//     // // float wanted_taper_edge_y  = std::max(raw_X_dim, raw_Y_dim) / 10.0;
//     // float wanted_taper_edge_x  = raw_Y_dim / 10.0;
//     // float wanted_taper_edge_y  = raw_Y_dim / 10.0;
//     // float wanted_mask_radius_x = raw_X_dim / 2.0;
//     // float wanted_mask_radius_y = raw_Y_dim / 2.0;
//     // input_image.TaperLinear(wanted_taper_edge_x, wanted_taper_edge_y, 1, mask_radius_x, mask_radius_y, 0);
//     // // input_image.TaperEdges( );
//     // input_image.WriteSlicesAndFillHeader("/groups/lingli/Documents/cisTEM/build/tomoalign_Intel-gpu-debug-static/src/Output/input_bf_stretch35_0.030_taperLinear.mrc", 1);

//     // Image empty_image, padded_image;
//     // empty_image.Allocate(raw_X_dim, raw_Y_dim, true);
//     // empty_image.SetToConstant(1.0);
//     // padded_image.Allocate(X_dim, Y_dim, true);
//     // empty_image.ClipInto(&padded_image);
//     // empty_image.CopyFrom(&padded_image);
//     // empty_image.WriteSlicesAndFillHeader("/groups/lingli/Documents/cisTEM/build/tomoalign_Intel-gpu-debug-static/src/Output/empty.mrc", 1);

//     // empty_image.TaperLinear(wanted_taper_edge_x, wanted_taper_edge_y, 1, mask_radius_x, mask_radius_y, 0);
//     // empty_image.WriteSlicesAndFillHeader("/groups/lingli/Documents/cisTEM/build/tomoalign_Intel-gpu-debug-static/src/Output/emptyLinearTaper.mrc", 1);
//     // Peak cistem_peak;
//     // cistem_peak = peak_image.FindPeakWithIntegerCoordinates( );
//     // wxPrintf("cisTEM peak %g, %g, %g,\n", cistem_peak.x, cistem_peak.y, cistem_peak.value);
//     // // wxPrintf("hello3\n");
//     // XCorrPeakFindWidth(&peak_image.real_values[0], X_dim + 2, Y_dim, xpeak, ypeak,
//     //                    peak, &width, &widthMin, maxPeaks,
//     //    minStrength);
//     // //    XCorrPeakFindWidth()
//     // Image image_ref;
//     // Image image_cur, image_shifted;
//     // image_ref.Allocate(ref_file.ReturnXSize( ), ref_file.ReturnYSize( ), true);
//     // image_cur.Allocate(input_file.ReturnXSize( ), input_file.ReturnYSize( ), true);
//     // image_cur.ReadSlice(&input_file, 1);
//     // image_ref.ReadSlice(&ref_file, 1);
//     // float ccc1, ccc2, ccc3, ccc4;
//     // float peakmax = peak[0];
//     float theta = 51;
//     float ref;
//     float tmppeak[4];
//     // float ref   = 48;
//     // for ( int i = 19; i < 35; i++ ) {
//     // for ( int i = 2; i < 19; i++ ) {
//     for ( int i = 0; i < 34; i++ ) {
//         // // if ( theta < 0 ) {
//         // // theta = 0 + (i - 1) * 3;
//         // theta = theta + 3;
//         // ref   = theta - 3;
//         // // }

//         // // theta = theta + i * 3;
//         // // ref = ref - (17 - 1 - i) * 3;
//         wxPrintf("--------image %i ----------\n", i);
//         // wxPrintf("current nd ref: %g, %g\n", theta, ref);
//         // float str = fabs(cosf(ref / 180 * PI) / cosf(theta / 180 * PI));
//         // // float str = fabs(cosf(theta / 180 * PI) / cosf(ref / 180 * PI));
//         // // str = fabs((cosf(ref / 180 * PI)) / cosf(theta / 180 * PI));
//         // wxPrintf("str: %g\n", str);
//         // // rotmagstrToAmat(0, 1.0, str, phi, &a11, &a12, &a21, &a22);
//         // rotmagstrToAmat(0, 1.0, str, phi, &a11, &a12, &a21, &a22);
//         // // rotmagstrToAmat(theta, 1.0, str, phi, &a11, &a12, &a21, &a22);
//         // // rotmagstrToAmat(ref - theta, 1.0, str, phi, &a11, &a12, &a21, &a22);
//         // // rotmagstrToAmat(1.0, 1.0, str, 0.0, &a11, &a12, &a21, &a22);
//         // // fs[0] = a11, fs[1] = a12, fs[2] = 0.0;
//         // // fs[3] = a21, fs[4] = a22, fs[5] = 0.0;
//         // // note that the fortran matrix is column dominate, c++ is row dominate
//         // // fs[0] = a11, fs[1] = a12, fs[4] = 0.0;

//         // fs[0] = a11, fs[2] = a12, fs[4] = 0.0;
//         // fs[1] = a21, fs[3] = a22, fs[5] = 0.0;

//         // // float fs_a[6], fiv_a[6];

//         // // rotmagstrToAmat(0, 1.0, 1, -phi, &a11, &a12, &a21, &a22);
//         // // fs_a[0] = a11, fs_a[2] = a12, fs_a[4] = 0.0;
//         // // fs_a[1] = a21, fs_a[3] = a22, fs_a[5] = 0.0;

//         // // xfInvert(fs_a, fiv_a, 2);

//         // // float unX1, unY1;
//         // // float xorg = -1173.334 / 4, yorg = 1680.392 / 4;
//         // // xfApply(fs, 0.0, 0.0, xorg, yorg, &unX1, &unY1, 2);
//         // // wxPrintf("un1, un2 %g, %g, %g, %g\n", unX1, unY1, unX1 * 4, unY1 * 4);

//         // xfInvert(fs, fiv, 2);

//         // wxPrintf("a11 a12 a21 a22 %g, %g, %g, %g \n", a11, a12, a21, a22);
//         // wxPrintf("fs              %g, %g, %g, %g, %g, %g \n", fs[0], fs[1], fs[2], fs[3], fs[4], fs[5]);
//         // wxPrintf("finv            %g, %g, %g, %g, %g, %g \n", fiv[0], fiv[1], fiv[2], fiv[3], fiv[4], fiv[5]);
//         // // wxPrintf("fs_a              %g, %g, %g, %g, %g, %g \n", fs_a[0], fs_a[1], fs_a[2], fs_a[3], fs_a[4], fs_a[5]);
//         // float finv_c[6];
//         // finv_c[0] = fiv[0], finv_c[1] = fiv[2], finv_c[2] = fiv[4];
//         // finv_c[3] = fiv[1], finv_c[4] = fiv[3], finv_c[5] = fiv[5];

//         peak_image.ReadSlice(&input_peak, i + 1);
//         wxPrintf("hello3\n");
//         Peak cistem_peak;
//         cistem_peak = peak_image.FindPeakWithIntegerCoordinates( );
//         wxPrintf("cisTEM peak %g, %g, %g,\n", cistem_peak.x, cistem_peak.y, cistem_peak.value);
//         // tmppeak[0] = i, tmppeak[1] = peak.x, tmppeak[2] = peak.y, tmppeak[3] = peak.value;
//         // wxPrintf("hello3\n");
//         // XCorrPeakFindWidth(&peak_image.real_values[0], X_dim + 2, Y_dim, xpeak, ypeak,
//         //                    peak, &width, &widthMin, maxPeaks, minStrength);
//         // wxPrintf("peaks: %g, %g, %g\n", xpeak[0] - X_dim / 2, ypeak[0] - Y_dim / 2, peak[i]);
//         // wxPrintf("peaks: %g, %g, %g\n", (xpeak[0] - X_dim / 2) * 4, (ypeak[0] - Y_dim / 2) * 4, peak[0]);
//         // float unX1, unY1, unX2, unY2;
//         // xfApply(fiv, 0.0, 0.0, xpeak[0] - X_dim / 2, ypeak[0] - Y_dim / 2, &unX1, &unY1, 2);
//         // wxPrintf("ind un1, un2 %i %5g %5g %5g %5g\n", i, unX1, unY1, unX1 * 4, unY1 * 4);

//         // wxPrintf("peaks: %g, %g, %g\n", xpeak[0], ypeak[0], peak[0]);
//         // // wxPrintf("peaks: %g, %g, %g\n", xpeak[i] - X_dim / 2, ypeak[i] - Y_dim / 2, peak[i]);
//         // // ccc1 = image_cur.ReturnCorrelationCoefficientUnnormalized(image_ref, wxMax(xpeak[i] - X_dim / 2, ypeak[i] - Y_dim / 2));
//         // // image_cur.WriteSlicesAndFillHeader("../src/Output/test1.mrc", 1);
//         // image_shifted.CopyFrom(&image_cur);
//         // image_shifted.PhaseShift(-(xpeak[i] - X_dim / 2), -(ypeak[i] - Y_dim / 2), 0);
//         // image_shifted.WriteSlicesAndFillHeader("shift1_phase.mrc", 1);
//         // // image_shifted.CopyFrom(&image_cur);
//         // // image_shifted.RealSpaceIntegerShift((xpeak[i] - X_dim / 2), (ypeak[i] - Y_dim / 2), 0);
//         // // image_shifted.WriteSlicesAndFillHeader("shift2_real.mrc", 1);

//         // // image_shifted.RealSpaceIntegerShift(-(xpeak[i] - X_dim / 2), -(ypeak[i] - Y_dim / 2), 0);
//         // // ccc1        = image_shifted.ReturnCorrelationCoefficientNormalized(image_ref, wxMax(X_dim / 8, Y_dim / 8));
//         // // float sizex = raw_X_dim / 2 - mask_edge - abs(xpeak[i] - X_dim / 2) / 2;
//         // // float sizey = raw_Y_dim / 2 - mask_edge - abs(ypeak[i] - Y_dim / 2) / 2;
//         // float sizex = raw_X_dim / 2 - mask_edge / 2;
//         // float sizey = raw_Y_dim / 2 - mask_edge / 2;

//         // wxPrintf("sizex sizey: %g, %g \n", sizex, sizey);
//         // // ccc1 = image_shifted.ReturnCorrelationCoefficientNormalizedRectangle(image_ref, sizex, sizey);
//         // // ccc2 = image_shifted.ReturnCorrelationCoefficientNormalized(image_ref, wxMax(X_dim / 8, Y_dim / 8));
//         // // ccc3 = image_shifted.ReturnCorrelationCoefficientNormalizedRectangle(image_ref, raw_X_dim / 2, raw_Y_dim / 2);
//         // // float testresult[2];
//         // int num, num_tmp;
//         // ccc3 = image_cur.ReturnCorrelationCoefficientNormalizedAtPeak(image_ref, &num, xpeak[i] - X_dim / 2, ypeak[i] - Y_dim / 2, 0, sizex, sizey);
//         // // ccc3 = image_cur.ReturnCorrelationCoefficientNormalizedAtPeak(image_shifted, &num, xpeak[i] - X_dim / 2, ypeak[i] - Y_dim / 2, 0, sizex, sizey);
//         // // overlap = float(nsum) / ((nxPad - 2 * nxCCTrimA) * (nyPad - 2 * nyCCTrimA));
//         // // ccc3               = testresult[0];
//         // // float num;

//         // float overlap = num / (2 * sizex) / (2 * sizey);

//         // float overlapPower = 6;
//         // float weight       = 1. / (1 + powf(wxMax(0.1, (wxMin(10.0, 0.125 / overlap))), overlapPower));
//         // wxPrintf("num overlap, overlap 0.125/overlap weight %i,%g, %g, %g\n", num, overlap, 0.125 / overlap, weight);

//         // float wgtccc3 = ccc3 * 1. / (1.0 + powf(wxMax(0.1, wxMin(10.0, 0.125 / overlap)), overlapPower));
//         // // wxPrintf("overlap overlap power, wgtc")
//         // // float wgtCCC       = ccc * 1. / (1. + &wxMax(0.1, min(10., overlapCrit / overlap)) * *overlapPower);
//         // // ccc1 = image_cur.ReturnCorrelationCoefficientNormalizedRectangle(image_ref, sizex, sizey);
//         // // ccc2 = image_cur.ReturnCorrelationCoefficientNormalized(image_ref, wxMax(X_dim / 8, Y_dim / 8));
//         // // ccc3 = image_cur.ReturnCorrelationCoefficientNormalizedRectangle(image_ref, raw_X_dim / 2, raw_Y_dim / 2);

//         // // ccc2 = image_shifted.ReturnCorrelationCoefficientUnnormalized(image_ref, wxMax(abs(xpeak[i] - X_dim / 2), abs(ypeak[i] - Y_dim / 2)));
//         // // image_cur.WriteSlicesAndFillHeader("../src/Output/test2.mrc", 1);
//         // // wxPrintf("coefs 1 2: %g, %g\n", ccc1, ccc2);
//         // float ratio = peak[i] / peakmax;
//         // // wxPrintf("peaks: %5g, %5g, %10.5g, %10.5g, %10.5g, %10.5g,%10.5g,%10.5g\n", xpeak[i] - X_dim / 2, ypeak[i] - Y_dim / 2, peak[i], ccc1, ccc2, ccc3, ccc4, ratio);
//         // wxPrintf("peaks: %5g, %5g, %10.5g, %10.5g, %10.5g,%10.5g\n", xpeak[i] - X_dim / 2, ypeak[i] - Y_dim / 2, peak[i], ccc3, wgtccc3, ratio);
//         // wxPrintf("peaks: %g, %g, %g\n", xpeak[i] - X_dim / 2, ypeak[i] - Y_dim / 2, peak[i]);
//         // wxPrintf("peaks: %g, %g, %g\n", (xpeak[i] - X_dim / 2) * 4, (ypeak[i] - Y_dim / 2) * 4, peak[i]);
//         // wxPrintf("peaks: %g, %g, %g\n", xpeak[i], ypeak[i], peak[i]);

//         // wxPrintf("peaks: %g, %g, %g\n", xpeak[0] - X_dim / 2, ypeak[0] - Y_dim / 2, peak[0]);
//         // wxPrintf("peaks: %g, %g, %g\n", xpeak[1] - X_dim / 2, ypeak[1] - Y_dim / 2, peak[1]);
//         // wxPrintf("peaks: %g, %g, %g\n", xpeak[2] - X_dim / 2, ypeak[2] - Y_dim / 2, peak[2]);
//         // wxPrintf("peaks: %g, %g, %g\n", xpeak[3] - X_dim / 2, ypeak[3] - Y_dim / 2, peak[3]);
//         // wxPrintf("peaks: %g, %g, %g\n", xpeak[4] - X_dim / 2, ypeak[4] - Y_dim / 2, peak[4]);
//         // wxPrintf("peaks: %g, %g, %g\n", xpeak[5] - X_dim / 2, ypeak[5] - Y_dim / 2, peak[5]);
//         // wxPrintf("peaks: %g, %g, %g\n", xpeak[6] - X_dim / 2, ypeak[6] - Y_dim / 2, peak[6]);
//         // wxPrintf("peaks: %g, %g, %g\n", xpeak[7] - X_dim / 2, ypeak[7] - Y_dim / 2, peak[7]);
//         // wxPrintf("peaks: %g, %g, %g\n", xpeak[8] - X_dim / 2, ypeak[8] - Y_dim / 2, peak[8]);
//         // wxPrintf("peaks: %g, %g, %g\n", xpeak[9] - X_dim / 2, ypeak[9] - Y_dim / 2, peak[9]);
//         // float unX1, unY1, unX2, unY2;

//         // xfApply(fs, 0.0, 0.0, xpeak[i] - X_dim / 2, ypeak[i] - Y_dim / 2, &unX2, &unY2, 2);
//         // // xfApply(fs, 0.0, 0.0, xpeak[i], ypeak[i], &unX2, &unY2, 2);
//         // wxPrintf("un1, un2 %g, %g, %g, %g\n", unX2, unY2, unX2 * 4, unY2 * 4);

//         // xfApply(fiv, 0.0, 0.0, xpeak[i] - X_dim / 2, ypeak[i] - Y_dim / 2, &unX1, &unY1, 2);
//         // wxPrintf("un1, un2 %g, %g, %g, %g\n", unX1, unY1, unX1 * 4, unY1 * 4);
//         // xfApply(finv_c, 0.0, 0.0, xpeak[i] - X_dim / 2, ypeak[i] - Y_dim / 2, &unX2, &unY2, 2);

//         // xfApply(fs, float(X_dim) / 2.0, float(Y_dim) / 2.0, xpeak[i], ypeak[i], &unX2, &unY2, 2);
//         // wxPrintf("un1, un2 %g, %g, %g, %g\n", unX2, unY2, unX2 * 4, unY2 * 4);
//         // wxPrintf("un1, un2 %g, %g, %g, %g\n", unX2 - X_dim / 2.0, unY2 - Y_dim / 2.0, unX2 * 4, unY2 * 4);

//         // xfApply(fiv, float(X_dim) / 2.0, float(Y_dim) / 2.0, xpeak[i], ypeak[i], &unX2, &unY2, 2);
//         // wxPrintf("un1, un2 %g, %g, %g, %g\n", unX2, unY2, unX2 * 4, unY2 * 4);
//         // wxPrintf("un1, un2 %g, %g, %g, %g\n", unX2 - X_dim / 2.0, unY2 - Y_dim / 2.0, unX2 * 4, unY2 * 4);
//         // call xfapply(fsInv, 0., 0., xpeak, ypeak, unstretchDx, unstretchDy)

//         // float x1 = unX1, y1 = unY1;
//         // xfApply(fs_a, 0, 0, x1, y1, &unX2, &unY2, 2);
//         // wxPrintf("un1, un2 %g, %g, %g, %g\n", unX2, unY2, unX2 * 4, unY2 * 4);
//         // wxPrintf("un1, un2 %g, %g, %g, %g\n", unX2 - X_dim / 2.0, unY2 - Y_dim / 2.0, unX2 * 4, unY2 * 4);
//     }

//     // input_array = peak_image.real_values;
//     // delete xpeak;
//     // delete ypeak;
//     // delete peak;
//     return true;
// }

// bool NikoTestApp::DoCalculation( ) {
//     wxPrintf("Hello world4\n");
//     // // int X_maskcenter          = my_current_job.arguments[1].ReturnIntegerArgument( );
//     wxString input_imgstack = my_current_job.arguments[0].ReturnStringArgument( );
//     // // wxString output_stack_filename = my_current_job.arguments[1].ReturnStringArgument( );
//     wxString angle_filename = my_current_job.arguments[1].ReturnStringArgument( );
//     int      img_index      = my_current_job.arguments[2].ReturnIntegerArgument( );
//     // wxString coordinates_filename  = my_current_job.arguments[2].ReturnStringArgument( );
//     // int      output_stack_box_size = my_current_job.arguments[3].ReturnIntegerArgument( );

//     NumericTextFile* tilt_angle_file;
//     tilt_angle_file = new NumericTextFile(angle_filename, OPEN_TO_READ, 1);

//     //stack manipulate
//     // MRCFile input_stack("/groups/lingli/Downloads/CTEM_tomo1/10064/proc_07112022/clip/tomo1_ali.mrc", false);
//     MRCFile input_stack(input_imgstack.ToStdString( ), false);
//     // MRCFile output_stack(output_stack_filename.ToStdString( ), true);
//     // output_stack.OpenFile(output_stack_filename.ToStdString( ), true);
//     int image_no = input_stack.ReturnNumberOfSlices( );
//     // MRCFile input_stack("/groups/lingli/Downloads/CTEM_tomo1/10064/proc_07112022/clip/tomo1_ali.mrc", false);
//     // wxPrintf("image number in the stack: %i\n", image_no);

//     Image current_image;
//     Image cos_image;
//     // MRCFile      output_file("output.mrc", true);
//     // ProgressBar* my_progress = new ProgressBar(input_stack.ReturnNumberOfSlices( ));
//     int my_x;
//     int my_y;
//     // Image box;

//     // box.Allocate(output_stack_box_size, output_stack_box_size, 1, true);
//     // int number_of_particles = input_coos_file->number_of_lines;

//     //write a if statement to judge if the number of coordinates in the coord file equals to image_no
//     // int number_of_patchgroups = input_coos_file->number_of_lines;
//     // float temp_array[3];
//     float temp_angle[1];
//     int   x_at_centertlt, y_at_centertlt;
//     // int   col = 2;

//     // float    temp_array[number_of_patchgroups][2];
//     // MRCFile* patch = new MRCFile[number_of_patchgroups];
//     // for ( int patch_counter = 0; patch_counter < number_of_patchgroups; patch_counter++ ) {
//     //     input_coos_file->ReadLine(temp_array[patch_counter]);
//     //     // string tmpstring = std::to_string(patch_counter);
//     //     // tmpstring = patch_counter.ToStdString( );
//     //     // MRCFile tmpstring(std::to_string(patch_counter) + ".mrc", true);
//     //     // string zz                     = std::to_string(patch_counter) + ".mrc";
//     //     // patch[patch_counter] = MRCFile(wxString::Format("%g.mrc", patch_counter).ToStdString( ), true);
//     //     patch[patch_counter].OpenFile(wxString::Format("%i.mrc", patch_counter).ToStdString( ), true);
//     // }

//     //     string zz, zzname;
//     // zz     = std::to_string(patch_counter) + ".mrc";
//     // zzname = std::to_string(patch_counter);
//     // MRCFile zzname(zz, true);
//     // wxPrintf("number of patch groups: %i\n\n", number_of_patchgroups);
//     for ( long image_counter = 0; image_counter < img_index; image_counter++ ) {
//         // current_image.ReadSlice(&input_stack, image_counter + 1);
//         // float image_mean = current_image.ReturnAverageOfRealValues( );

//         tilt_angle_file->ReadLine(temp_angle);
//         // my_image.crop( );
//     }
//     current_image.ReadSlice(&input_stack, img_index);
//     float xdim     = current_image.logical_x_dimension;
//     float xdim_cos = xdim * cosf(temp_angle[0] / 180.0 * PI);
//     float ydim     = current_image.logical_y_dimension;
//     wxPrintf("dimensions %g %g %g\n", xdim, xdim_cos, ydim);
//     cos_image.Allocate(int(xdim_cos) + 1, int(ydim), 1, true);
//     current_image.ClipInto(&cos_image, 0.0, false, 1.0, int(xdim_cos), int(ydim), 0);
//     wxPrintf("dimensions %i %i\n", cos_image.logical_x_dimension, cos_image.logical_y_dimension);

//     float wantedvalue;
//     wantedvalue = 1.0;
//     // cos_image.AddByLinearInterpolationFourier2D( xdim, ydim, 1.0);

//     // cos_image.AddByLinearInterpolationReal(xdim, ydim, wantedvalue, wantedvalue);

//     cos_image.WriteSlicesAndFillHeader("costest1.mrc", 1);

//     // for ( int patch_counter = 0; patch_counter < number_of_patchgroups; patch_counter++ ) {

//     //     // input_coos_file->ReadLine(temp_array);
//     //     // x_at_centertlt = temp_array[0];
//     //     // y_at_centertlt = temp_array[1];
//     //     x_at_centertlt = temp_array[patch_counter][0];
//     //     y_at_centertlt = temp_array[patch_counter][1];
//     //     my_x           = int(x_at_centertlt * cosf(PI * temp_angle[0] / 180.0));
//     //     my_y           = y_at_centertlt;
//     //     current_image.ClipInto(&box, image_mean, false, 1.0, int(my_x), int(my_y), 0);

//     //     // wxPrintf("x=%i, y=%i\n", my_x, my_y);
//     //     // string zz = std::to_string(patch_counter) + ".mrc";
//     //     box.WriteSlice(&patch[patch_counter], image_counter + 1);
//     //     // wxPrintf("%.0f .mrc", patch_counter)
//     //     // box.WriteSlice(zz.str( ), image_counter + 1);
//     // }
//     // my_progress->Update(image_counter + 1);
//     return true;
// }

// delete my_progress;
// delete input_coos_file;
// delete[] patch;
// delete temp_array;

/*    square mask part
    Image masked_image;
    // Image circlemask_image;
    Image squaremask;

    wxString input_2d     = my_current_job.arguments[0].ReturnStringArgument( );
    int      X_maskcenter = my_current_job.arguments[1].ReturnIntegerArgument( );
    int      Y_maskcenter = my_current_job.arguments[2].ReturnIntegerArgument( );
    // float    SquareMaskSize = my_current_job.arguments[3].ReturnFloatArgument( );
    int SquareMaskSize = my_current_job.arguments[3].ReturnIntegerArgument( );

    wxPrintf("Hello world\n");

    MRCFile input_file_2d(input_2d.ToStdString( ), false);
    Image   input_image;

    int X_dim, Y_dim;

    // int X_maskcenter, Y_maskcenter;
    X_dim = input_file_2d.ReturnXSize( );
    Y_dim = input_file_2d.ReturnYSize( );
    // X_maskcenter = X_dim / 4;
    // Y_maskcenter = Y_dim / 4;
    wxPrintf("X_dim = %i, Y_dim = %i \n", X_dim, Y_dim);
    wxPrintf("X_maskcenter = %i, Y_maskcenter = %i \n", X_maskcenter, Y_maskcenter);
    wxPrintf("masksize = %i\n\n", SquareMaskSize);

    // padded_dimensions_x = ReturnClosestFactorizedUpper(pad_factor * input_file_2d.ReturnXSize( ), 3);
    // padded_dimensions_y = ReturnClosestFactorizedUpper(pad_factor * input_file_2d.ReturnYSize( ), 3);

    //   input_volume.Allocate(input_file_3d.ReturnXSize( ), input_file_3d.ReturnYSize( ), input_file_3d.ReturnZSize( ), true);
    // circlemask_image.Allocate(X_dim, Y_dim, true);
    input_image.Allocate(X_dim, Y_dim, true);
    input_image.ReadSlice(&input_file_2d, 1);

    squaremask.Allocate(X_dim, Y_dim, true);
    squaremask.SetToConstant(1.0);
    squaremask.SquareMaskWithValue(SquareMaskSize, 0.0, false, X_maskcenter, Y_maskcenter, 0);
    // GaussianLowPassFilter(float sigma)
    squaremask.ForwardFFT( );
    squaremask.GaussianLowPassFilter(0.01);
    squaremask.BackwardFFT( );

    masked_image.CopyFrom(&input_image);
    masked_image.Normalize(10.0);
    masked_image.WriteSlicesAndFillHeader("normalized.mrc", 1);

    // masked_image.ForwardFFT( );
    // squaremask.ForwardFFT( );
    masked_image.MultiplyPixelWise(squaremask);

    input_image.physical_address_of_box_center_x = X_maskcenter;
    input_image.physical_address_of_box_center_y = Y_maskcenter;
    input_image.CosineRectangularMask(SquareMaskSize / 2, SquareMaskSize / 2 + 50, 0, 50, false, false, 0.0);

    input_image.WriteSlicesAndFillHeader("cosinrectangularmask.mrc", 1);

    // zz = masked_image.ApplyMask(squaremask, 80, 10, 0.5, 10);
    masked_image.WriteSlicesAndFillHeader("imageapplysquaremask.mrc", 1);
    squaremask.WriteSlicesAndFillHeader("squaretest.mrc", 1);

    input_image.CalculateCrossCorrelationImageWith(&masked_image);
    input_image.WriteSlicesAndFillHeader("ccn-temp.mrc", 1);

    Peak peaks;
    peaks.x = 10;
    peaks.y = 10;
    wxPrintf("peak position: x = %g, y = %g \n\n", peaks.x, peaks.y, peaks.value);

    peaks = input_image.FindPeakWithIntegerCoordinates(0, 400, 10);
    wxPrintf("peak position: x = %g, y = %g ,value = %g\n\n", peaks.x, peaks.y, peaks.value);

    input_image.RealSpaceIntegerShift(15, 20, 0);
    peaks = input_image.FindPeakWithIntegerCoordinates(0, 400, 10);
    wxPrintf("peak position: x = %g, y = %g ,value = %g\n\n", peaks.x, peaks.y, peaks.value);

       square msk part end */

// padded_image.Allocate(padded_dimensions_x, padded_dimensions_y, true);

// masked_image.Allocate(input_file_2d.ReturnXSize( ), input_file_2d.ReturnYSize( ), true);

// //  input_volume.ReadSlices(&input_file_3d, 1, input_file_3d.ReturnZSize( ));
// input_image.ReadSlice(&input_file_2d, 1);
// input_image.Normalize(10);

// circlemask_image.ReadSlice(&mask_file_2d, 1);

// sigma = sqrtf(input_image.ReturnVarianceOfRealValues( ));
// // temp
// wxPrintf("xdim= %i mean=%g \n\n", input_file_2d.ReturnXSize( ), input_image.ReturnAverageOfRealValues( ));
// wxPrintf("sigma= %g \n\n", sigma);

// masked_image.CopyFrom(&input_image);
// float zz;
// zz = masked_image.ApplyMask( );

//masked_image.SquareMaskWithValue(500, 0);
// masked_image.TriangleMask(300); //this create a mask, not apply mask to the image
// masked_image.CircleMask(200); //this mask the original image by a circle
// masked_image.WriteSlicesAndFillHeader("circlemask.mrc", 1);
// masked_image.CircleMaskWithValue(200, -10); //this mask the original image by a circle
// masked_image.WriteSlicesAndFillHeader("circlemaskn10.mrc", 1);
// float zz;
// zz = masked_image.ApplyMask(masked_image, 20);
// zz = masked_image.ApplyMask(circlemask_image, 80, 10, 0.5, 10);
// masked_image.WriteSlicesAndFillHeader("imageapplymask.mrc", 1);
// wxPrintf("zzzzzzzzz%g\n\n", zz);

/*    ///-----------------------------------------------------------------------------------------------------------
    input_image.ClipIntoLargerRealSpace2D(&padded_image);
    padded_image.AddGaussianNoise(10.0 * sigma);
    padded_image.WriteSlice(&output_file, 1);
    padded_image.ForwardFFT( );

    input_image.AddSlices(input_volume);
    //	input_image.QuickAndDirtyWriteSlice("junk.mrc", 1);
    count = 1;
    output_image.CopyFrom(&padded_image);
    temp_image.CopyFrom(&input_image);
    temp_image.Resize(padded_dimensions_x, padded_dimensions_y, 1);
    //	temp_image2.CopyFrom(&input_image);
    //	temp_image2.Resize(padded_dimensions_x, padded_dimensions_y, 1);
    //	temp_image2.RealSpaceIntegerShift(input_image.logical_x_dimension, input_image.logical_y_dimension, 0);
    //	temp_image.AddImage(&temp_image2);
    //	temp_image.QuickAndDirtyWriteSlice("junk.mrc", 1);
    temp_image.ForwardFFT( );
    output_image.ConjugateMultiplyPixelWise(temp_image);
    output_image.SwapRealSpaceQuadrants( );
    output_image.BackwardFFT( );
    peak = output_image.real_values[output_image.logical_x_dimension / 2 + (output_image.logical_x_dimension + output_image.padding_jump_value) * output_image.logical_y_dimension / 2];
    wxPrintf("\nPeak with whole projection = %g background = %g\n\n", peak, output_image.ReturnVarianceOfRealValues(float(2 * input_image.logical_x_dimension), 0.0, 0.0, 0.0, true));
    //	wxPrintf("\nPeak with whole projection = %g\n\n", output_image.ReturnMaximumValue());
    output_image.WriteSlice(&output_file, 2);
    sum_of_peaks = 0.0;
    for ( i = 1; i <= 3; i += 2 ) {
        for ( j = 1; j <= 3; j += 2 ) {
            output_image.CopyFrom(&padded_image);
            temp_image.CopyFrom(&input_image);
            temp_image.SquareMaskWithValue(float(input_image.logical_x_dimension) / 2.0, 0.0, false, i * input_image.logical_x_dimension / 4, j * input_image.logical_x_dimension / 4);
            temp_image.Resize(padded_dimensions_x, padded_dimensions_y, 1);
            temp_image.ForwardFFT( );
            output_image.ConjugateMultiplyPixelWise(temp_image);
            output_image.SwapRealSpaceQuadrants( );
            output_image.BackwardFFT( );
            peak = output_image.real_values[output_image.logical_x_dimension / 2 + (output_image.logical_x_dimension + output_image.padding_jump_value) * output_image.logical_y_dimension / 2];
            //			peak = output_image.ReturnMaximumValue();
            wxPrintf("Quarter peak = %i %i %g\n", i, j, peak);
            sum_of_peaks += peak;
            count++;
            output_image.WriteSlice(&output_file, 1 + count);
        }
    }
    wxPrintf("\nSum of quarter peaks = %g\n\n", sum_of_peaks);

    sum_of_peaks = 0.0;
    for ( i = 0; i < 4; i++ ) {
        output_image.CopyFrom(&padded_image);
        input_image.AddSlices(input_volume, i * input_volume.logical_z_dimension / 4 + 1, (i + 1) * input_volume.logical_z_dimension / 4);
        //		input_image.QuickAndDirtyWriteSlice("junk.mrc", i + 1);
        temp_image.CopyFrom(&input_image);
        temp_image.Resize(padded_dimensions_x, padded_dimensions_y, 1);
        temp_image.ForwardFFT( );
        output_image.ConjugateMultiplyPixelWise(temp_image);
        output_image.SwapRealSpaceQuadrants( );
        output_image.BackwardFFT( );
        peak = output_image.real_values[output_image.logical_x_dimension / 2 + (output_image.logical_x_dimension + output_image.padding_jump_value) * output_image.logical_y_dimension / 2];
        //		peak = output_image.ReturnMaximumValue();
        wxPrintf("Slice peak = %i %g\n", i + 1, peak);
        sum_of_peaks += peak;
        count++;
        output_image.WriteSlice(&output_file, 1 + count);
    }
    wxPrintf("\nSum of slice peaks = %g\n", sum_of_peaks);
    ///------------------------------- */

/*	wxPrintf("\nDoing 1000 FFTs %i x %i\n", output_image.logical_x_dimension, output_image.logical_y_dimension);
	for (i = 0; i < 1000; i++)
	{
		output_image.is_in_real_space = false;
		output_image.SetToConstant(1.0);
		output_image.BackwardFFT();
	}
	wxPrintf("\nFinished\n");
*/

/*	int i, j;
	int slice_thickness;
	int first_slice, last_slice, middle_slice;
	long offset;
	long pixel_counter;
	float bfactor = 20.0;
	float mask_radius = 75.0;
	float pixel_size = 1.237;
//	float pixel_size = 0.97;
	float bfactor_res_limit = 8.0;
	float resolution_limit = 3.8;
//	float resolution_limit = 3.0;
	float cosine_edge = 5.0;
	float bfactor_pixels;

	MRCFile input_file("input.mrc", false);
	MRCFile output_file_2D("output2D.mrc", true);
	MRCFile output_file_3D("output3D.mrc", true);
	Image input_image;
	Image output_image;
	Image output_image_3D;

	Curve power_spectrum;
	Curve number_of_terms;

	UserInput my_input("NikoTest", 1.00);
	pixel_size = my_input.GetFloatFromUser("Pixel size of images (A)", "Pixel size of input images in Angstroms", "1.0", 0.0);
	mask_radius = my_input.GetFloatFromUser("Mask radius (A)", "Radius of mask to be applied to input 3D map, in Angstroms", "100.0", 0.0);
	bfactor = my_input.GetFloatFromUser("B-Factor (A^2)", "B-factor to be applied to dampen the 3D map after spectral flattening, in Angstroms squared", "20.0");
	bfactor_res_limit = my_input.GetFloatFromUser("Low resolution limit for spectral flattening (A)", "The resolution at which spectral flattening starts being applied, in Angstroms", "8.0", 0.0);
	resolution_limit = my_input.GetFloatFromUser("High resolution limit (A)", "Resolution of low-pass filter applied to final output maps, in Angstroms", "3.0", 0.0);

	slice_thickness = myroundint(resolution_limit / pixel_size);
	input_image.Allocate(input_file.ReturnXSize(), input_file.ReturnYSize(), slice_thickness, true);
	output_image.Allocate(input_file.ReturnXSize(), input_file.ReturnYSize(), true);
	output_image_3D.Allocate(input_file.ReturnXSize(), input_file.ReturnYSize(), input_file.ReturnZSize(), true);

	wxPrintf("\nCalculating 3D spectrum...\n");

	power_spectrum.SetupXAxis(0.0, 0.5 * sqrtf(3.0), int((output_image_3D.logical_x_dimension / 2.0 + 1.0) * sqrtf(3.0) + 1.0));
	number_of_terms.SetupXAxis(0.0, 0.5 * sqrtf(3.0), int((output_image_3D.logical_x_dimension / 2.0 + 1.0) * sqrtf(3.0) + 1.0));
	output_image_3D.ReadSlices(&input_file, 1, input_file.ReturnZSize());
	output_image_3D.CosineMask(mask_radius / pixel_size, 10.0 / pixel_size);

	first_slice = int((input_file.ReturnZSize() - slice_thickness + 1) / 2.0);
	last_slice = first_slice + slice_thickness;
	pixel_counter = 0;
	for (j = first_slice; j < last_slice; j++)
	{
		offset = j * (output_image_3D.logical_x_dimension + output_image_3D.padding_jump_value) * output_image_3D.logical_y_dimension;
		for (i = 0; i < (output_image_3D.logical_x_dimension + output_image_3D.padding_jump_value) * output_image_3D.logical_y_dimension; i++) {input_image.real_values[pixel_counter] = output_image_3D.real_values[i + offset]; pixel_counter++;}
	}

	output_image_3D.ForwardFFT();
	output_image_3D.Compute1DPowerSpectrumCurve(&power_spectrum, &number_of_terms);
	power_spectrum.SquareRoot();
	wxPrintf("Done with 3D spectrum. Starting slice estimation...\n");

//	input_image.ReadSlices(&input_file, first_slice, last_slice);
	bfactor_pixels = bfactor / pixel_size / pixel_size;
	input_image.ForwardFFT();
	input_image.ApplyBFactorAndWhiten(power_spectrum, bfactor_pixels, bfactor_pixels, pixel_size / bfactor_res_limit);
//	input_image.ApplyBFactor(bfactor_pixels);
//	input_image.CosineMask(pixel_size / resolution_limit, cosine_edge / input_file.ReturnXSize());
	input_image.CosineMask(pixel_size / resolution_limit - pixel_size / 40.0, pixel_size / 20.0);
	input_image.BackwardFFT();

	middle_slice = int(slice_thickness / 2.0);
	offset = middle_slice * (input_file.ReturnXSize() + input_image.padding_jump_value) * input_file.ReturnYSize();
	pixel_counter = 0;
	for (i = 0; i < (input_file.ReturnXSize() + input_image.padding_jump_value) * input_file.ReturnYSize(); i++) {output_image.real_values[pixel_counter] = input_image.real_values[i + offset]; pixel_counter++;}
//	output_image.ForwardFFT();
//	output_image.CosineMask(pixel_size / resolution_limit - pixel_size / 40.0, pixel_size / 20.0);
//	output_image.BackwardFFT();
	output_image.WriteSlice(&output_file_2D, 1);
	wxPrintf("Done with slices. Starting 3D B-factor application...\n");

	output_image_3D.ApplyBFactorAndWhiten(power_spectrum, bfactor_pixels, bfactor_pixels, pixel_size / bfactor_res_limit);
	output_image_3D.CosineMask(pixel_size / resolution_limit - pixel_size / 40.0, pixel_size / 20.0);
	output_image_3D.BackwardFFT();
	output_image_3D.WriteSlices(&output_file_3D, 1, input_file.ReturnZSize());
	wxPrintf("Done with 3D B-factor application.\n");

	int i;
	int min_class;
	int max_class;
	int count;
	float temp_float;
	float input_parameters[17];

	MRCFile input_file("input.mrc", false);
	MRCFile output_file("output.mrc", true);
	Image input_image;
	Image padded_image;
	Image ctf_image;
	Image sum_image;
	CTF ctf;
	AnglesAndShifts rotation_angle;

	input_image.Allocate(input_file.ReturnXSize(), input_file.ReturnYSize(), true);
	padded_image.Allocate(4 * input_file.ReturnXSize(), 4 * input_file.ReturnYSize(), true);
	ctf_image.Allocate(input_file.ReturnXSize(), input_file.ReturnYSize(), false);
	sum_image.Allocate(input_file.ReturnXSize(), input_file.ReturnYSize(), false);
	sum_image.SetToConstant(0.0);

	FrealignParameterFile input_par_file("input.par", OPEN_TO_READ);
	input_par_file.ReadFile();

//	count = 0;
//	for (i = 1; i <= input_par_file.number_of_lines; i++)
//	{
//		if (i % 100 == 1) wxPrintf("Working on line %i\n", i);
//		input_par_file.ReadLine(input_parameters);
//		input_image.ReadSlice(&input_file, int(input_parameters[0] + 0.5));
//		count++;
//		input_image.WriteSlice(&output_file, count);
//	}

	for (i = 1; i <= input_par_file.number_of_lines; i++)
	{
		if (i % 100 == 1) wxPrintf("Rotating image %i\n", i);
		input_par_file.ReadLine(input_parameters);
		input_image.ReadSlice(&input_file, i);
		input_image.RealSpaceIntegerShift(-input_parameters[4], -input_parameters[5]);
		input_image.ForwardFFT();
		input_image.ClipInto(&padded_image);
		padded_image.BackwardFFT();
		rotation_angle.GenerateRotationMatrix2D(-input_parameters[1]);
		padded_image.Rotate2DSample(input_image, rotation_angle);
		input_image.WriteSlice(&output_file, i);
		if (input_parameters[7] == 2)
		{
			ctf.Init(300.0, 0.0, 0.1, input_parameters[8], input_parameters[9], input_parameters[10], 0.0, 0.0, 0.0, 1.0, input_parameters[11]);
			ctf_image.CalculateCTFImage(ctf);
			input_image.ForwardFFT();
			input_image.PhaseFlipPixelWise(ctf_image);
			sum_image.AddImage(&input_image);
		}
//		if (i == 1001) break;
	}
	sum_image.QuickAndDirtyWriteSlice("sum.mrc", 1);
*/

/*	FrealignParameterFile input_par_file("input.par", OPEN_TO_READ);
	FrealignParameterFile output_par_file("output.par", OPEN_TO_WRITE);
	input_par_file.ReadFile(true);
	input_par_file.ReduceAngles();
	min_class = myroundint(input_par_file.ReturnMin(7));
	max_class = myroundint(input_par_file.ReturnMax(7));
	for (i = min_class; i <= max_class; i++)
	{
		temp_float = input_par_file.ReturnDistributionMax(2, i);
		if (temp_float != 0.0) wxPrintf("theta max, sigma = %i %g %g\n", i, temp_float, input_par_file.ReturnDistributionSigma(2, temp_float, i));
//		input_par_file.SetParameters(2, temp_float, i);
		temp_float = input_par_file.ReturnDistributionMax(3, i);
		if (temp_float != 0.0) wxPrintf("phi max, sigma = %i %g %g\n", i, temp_float, input_par_file.ReturnDistributionSigma(3, temp_float, i));
//		input_par_file.SetParameters(3, temp_float, i);
	} */
//	for (i = 1; i <= input_par_file.number_of_lines; i++)
//	{
//		input_par_file.ReadLine(input_parameters);
//		output_par_file.WriteLine(input_parameters);
//	}

//	MRCFile input_file("input.mrc", false);
//	MRCFile output_file("output.mrc", true);
//	Image input_image;
//	Image filtered_image;
//	Image kernel;

//	input_image.Allocate(input_file.ReturnXSize(), input_file.ReturnYSize(), input_file.ReturnZSize(), true);
//	filtered_image.Allocate(input_file.ReturnXSize(), input_file.ReturnYSize(), input_file.ReturnZSize(), true);
//	kernel.Allocate(input_file.ReturnXSize(), input_file.ReturnYSize(), input_file.ReturnZSize(), true);
//	input_image.ReadSlices(&input_file,1,input_image.logical_z_dimension);

/*	kernel.SetToConstant(1.0);
	kernel.CosineMask(8.0, 8.0, false, true, 0.0);
//	kernel.real_values[0] = 1.0;
	temp_float = kernel.ReturnAverageOfRealValues() * kernel.number_of_real_space_pixels;
//	wxPrintf("average = %g\n", temp_float);
//	kernel.WriteSlices(&output_file,1,input_image.logical_z_dimension);
	kernel.ForwardFFT();
	kernel.SwapRealSpaceQuadrants();
	kernel.MultiplyByConstant(float(kernel.number_of_real_space_pixels) / temp_float);
//	kernel.CosineMask(0.03, 0.03, true);

	input_image.SetMinimumValue(0.0);
	filtered_image.CopyFrom(&input_image);
	filtered_image.ForwardFFT();
	filtered_image.MultiplyPixelWise(kernel);
//	filtered_image.CosineMask(0.01, 0.02);
	filtered_image.BackwardFFT();
//	filtered_image.MultiplyByConstant(0.3);
	input_image.SubtractImage(&filtered_image);
*/
//	input_image.SetToConstant(1.0);
//	input_image.CorrectSinc(45.0, 1.0, true, 0.0);
//	for (i = 0; i < input_image.real_memory_allocated; i++) if (input_image.real_values[i] < 0.0) input_image.real_values[i] = -log(-input_image.real_values[i] + 1.0);
//	input_image.WriteSlices(&output_file,1,input_image.logical_z_dimension);
//	temp_float = -420.5; wxPrintf("%g\n", fmodf(temp_float, 360.0));
//	filtered_image.WriteSlices(&output_file,1,input_image.logical_z_dimension);

// return true;
// }
