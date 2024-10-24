#include <dlib/dlib/queue.h>
#include "../../core/core_headers.h"
#include <vector>
#include <cmath>
#include <algorithm>
#include <dlib/dlib/matrix.h>
#include <dlib/dlib/statistics.h>
// #include <numeric>
// #include <iostream>
#include "movieframespline.h"
// #include "movieframespline_quad.h"

using namespace std;
using namespace cistem;
using namespace dlib;
typedef matrix<double, 0, 1> column_vector;

struct Point {
    double x, y;

    double distance(const Point& other) const {
        return std::sqrt((x - other.x) * (x - other.x) + (y - other.y) * (y - other.y));
    }
};

Point findNearestNeighbor(const Point& point, const std::vector<Point>& points) {
    Point  nearest;
    double minDistance = std::numeric_limits<double>::max( );

    for ( const auto& p : points ) {
        double dist = point.distance(p);
        if ( dist < minDistance ) {
            minDistance = dist;
            nearest     = p;
        }
    }
    return nearest;
}

// #include <iostream>
// #include <vector>

std::vector<double> diff(const std::vector<double>& data) {
    std::vector<double> differences;
    for ( size_t i = 1; i < data.size( ); ++i ) {
        differences.push_back(data[i] - data[i - 1]);
    }
    return differences;
}

std::vector<int> FindCommonElements(const std::vector<int>& vec1, const std::vector<int>& vec2) {
    std::vector<int>        commonElements;
    std::unordered_set<int> set1(vec1.begin( ), vec1.end( ));

    for ( const auto& num : vec2 ) {
        if ( set1.count(num) > 0 ) {
            commonElements.push_back(num);
        }
    }

    return commonElements;
}

// #include <iostream>
// #include <vector>

double calculatePercentile(const std::vector<double>& data, double percentile) {
    if ( data.empty( ) )
        return 0;

    std::vector<double> sortedData = data;
    std::sort(sortedData.begin( ), sortedData.end( ));

    double index      = (percentile / 100.0) * (sortedData.size( ) - 1);
    int    lowerIndex = static_cast<int>(index);
    int    upperIndex = lowerIndex + 1;

    if ( upperIndex >= sortedData.size( ) ) {
        return sortedData[lowerIndex];
    }

    double fraction = index - lowerIndex;
    return sortedData[lowerIndex] + fraction * (sortedData[upperIndex] - sortedData[lowerIndex]);
}

double calculateMean(const std::vector<double>& data) {
    double sum = 0.0;
    for ( double value : data ) {
        sum += value;
    }
    return data.size( ) > 0 ? sum / data.size( ) : 0.0;
}

double calculateStdDev(const std::vector<double>& data) {
    double mean     = calculateMean(data);
    double variance = 0.0;

    for ( double value : data ) {
        variance += std::pow(value - mean, 2);
    }
    variance = variance / data.size( );

    return std::sqrt(variance);
}

std::vector<double> findOutliersUpperBound(const std::vector<double>& data) {
    std::vector<double> sortedData = data;
    std::sort(sortedData.begin( ), sortedData.end( ));

    double Q1  = calculatePercentile(sortedData, 25);
    double Q3  = calculatePercentile(sortedData, 75);
    double IQR = Q3 - Q1;

    double lowerBound = Q1 - 1.5 * IQR;
    double upperBound = Q3 + 1.5 * IQR;

    std::vector<double> outliers;
    for ( double value : data ) {
        // if (value < lowerBound || value > upperBound) {
        //     outliers.push_back(value);
        // }
        if ( value > upperBound ) {
            outliers.push_back(value);
        }
    }

    return outliers;
}

std::vector<int> findOutlierUpperBoundIndices(const std::vector<double>& data) {
    std::vector<double> sortedData = data;

    double Q1  = calculatePercentile(sortedData, 25);
    double Q3  = calculatePercentile(sortedData, 75);
    double IQR = Q3 - Q1;

    double lowerBound = Q1 - 1.5 * IQR;
    double upperBound = Q3 + 1.5 * IQR;

    std::vector<int> outlierIndices;
    for ( size_t i = 0; i < data.size( ); ++i ) {
        // if (data[i] < lowerBound || data[i] > upperBound) {
        //     outlierIndices.push_back(i); // Add the index of the outlier
        // }
        if ( data[i] > upperBound ) {
            outlierIndices.push_back(i); // Add the index of the outlier
        }
    }

    return outlierIndices;
}

std::vector<column_vector> PCAReduction(const std::vector<column_vector>& data, double target_dim) {
    vector_normalizer_pca<column_vector> pca, pcaa;
    pca.train(data, 0.99);
    // wxPrintf("size data.nr eps %li %li %f\n", data[0].size( ), data[0].nr( ), target_dim / data[0].nr( ));
    // wxPrintf("size data r and c %li %li\n", data[0].nr( ), data[0].nc( ));

    // std::vector<column_vector> new_data;
    std::vector<column_vector> output_data;
    wxPrintf("datasize %li \n", data.size( ));
    // new_data.reserve(data.size( ));
    output_data.reserve(data.size( ));
    for ( size_t i = 0; i < data.size( ); ++i ) {
        // new_data.emplace_back(pca(data[i]));
        output_data.emplace_back(rowm(pca(data[i]), range(0, target_dim - 1)));
    }

    // wxPrintf("std size %li\n", pca.std_devs( ).size( ));
    // wxPrintf("std %f\n", pca.std_devs( )(0));
    // wxPrintf("mean %f\n", pca.means( )(0));
    // wxPrintf("pca_matrix size %li\n", pca.pca_matrix( ).size( ));

    return output_data;
}

//==================================================== functions ====================================================================

std::vector<int> FixOutliers(matrix<double>* patch_peaksx, matrix<double>* patch_peaksy, matrix<double> patch_locations_x, matrix<double> patch_locations_y, int patch_no_x, int patch_no_y, int step_size_x, int step_size_y, int image_no, bool write_out_the_outliers, std::string output_path, std::string file_pref) {
    int patch_no = patch_no_x * patch_no_y;
    // /*  this section fix the outliers using nearest neighbor ---------------------------------------------------
    wxPrintf("fixing outlier fix\n");
    std::vector<double> used_index(patch_no);
    std::vector<int>    missed_index; // it should be in increasing order
    matrix<float>       truefalse;
    // truefalse.set_size(patch_num_y, patch_num_x);
    truefalse = ones_matrix<float>(patch_no_y, patch_no_x);

    std::vector<double>              differences;
    std::vector<std::vector<double>> diffx_vec(patch_no, std::vector<double>(image_no - 1, 0));
    std::vector<std::vector<double>> diffy_vec(patch_no, std::vector<double>(image_no - 1, 0));
    std::vector<double>              stdx_vec(patch_no);
    std::vector<double>              stdy_vec(patch_no);
    std::vector<int>                 outlier_x_ind;
    std::vector<int>                 outlier_y_ind;

    for ( int ii = 0; ii < patch_no_y; ii++ ) {
        for ( int jj = 0; jj < patch_no_x; jj++ ) {
            for ( int j = 0; j < image_no - 1; j++ ) {
                diffx_vec[ii * patch_no_x + jj][j] = patch_peaksx[j + 1](ii, jj) - patch_peaksx[j](ii, jj);
                diffy_vec[ii * patch_no_x + jj][j] = patch_peaksy[j + 1](ii, jj) - patch_peaksy[j](ii, jj);
            }
        }
    }

    for ( int i = 0; i < patch_no; i++ ) {
        stdx_vec[i] = calculateStdDev(diffx_vec[i]);
        stdy_vec[i] = calculateStdDev(diffy_vec[i]);
    }

    outlier_x_ind = findOutlierUpperBoundIndices(stdx_vec);
    outlier_y_ind = findOutlierUpperBoundIndices(stdy_vec);

    outlier_x_ind.insert(outlier_x_ind.end( ), outlier_y_ind.begin( ), outlier_y_ind.end( ));
    missed_index.insert(missed_index.end( ), outlier_x_ind.begin( ), outlier_x_ind.end( ));
    std::sort(missed_index.begin( ), missed_index.end( ));
    auto last = std::unique(missed_index.begin( ), missed_index.end( ));
    missed_index.erase(last, missed_index.end( ));

    wxPrintf("outliers:\n");
    for ( int i = 0; i < missed_index.size( ); i++ ) {
        wxPrintf("%d\t", missed_index[i]);
    }
    wxPrintf("\n");

    if ( write_out_the_outliers ) {
        float           outlier;
        NumericTextFile outputfile(output_path + "outliers.txt", OPEN_TO_WRITE, 1);
        for ( int i = 0; i < missed_index.size( ); i++ ) {
            // wxPrintf("%d\t", missed_index[i]);
            outlier = missed_index[i];
            outputfile.WriteLine(&outlier);
        }
        outputfile.Close( );
    }

    for ( int i = 0; i < missed_index.size( ); i++ ) {
        int c = missed_index[i] % patch_no_x;
        int r = (missed_index[i] - c) / patch_no_x;
        // cout << "column and row are " << c << " " << r << endl;
        truefalse(r, c) = 0;
    }
    cout << "truefalse map" << endl
         << truefalse << endl;
    std::vector<Point> points;
    std::vector<Point> badpoints;
    for ( int i = 0; i < patch_no_y; i++ ) {
        for ( int j = 0; j < patch_no_x; j++ ) {
            if ( truefalse(i, j) == 0 ) {
                badpoints.push_back({patch_locations_x(j), patch_locations_y(i)});
            }
            else {
                Point temppoint;
                temppoint = {patch_locations_x(j), patch_locations_y(i)};
                points.push_back({patch_locations_x(j), patch_locations_y(i)});
            }
        }
    }
    int pointslength    = points.size( );
    int badpointslength = badpoints.size( );
    cout << pointslength << endl;
    // for ( int i = 0; i < pointslength; i++ ) {
    //     cout << points[i].x << " " << points[i].y << endl;
    // }
    cout << "bad points" << endl;
    for ( int i = 0; i < badpointslength; i++ ) {
        cout << badpoints[i].x << endl;
    }

    matrix<double>* patch_peaksx_tmp;
    matrix<double>* patch_peaksy_tmp;
    patch_peaksx_tmp = new matrix<double>[image_no];
    patch_peaksy_tmp = new matrix<double>[image_no];
    for ( int i = 0; i < image_no; i++ ) {
        patch_peaksx_tmp[i].set_size(patch_no_y, patch_no_x);
        patch_peaksy_tmp[i].set_size(patch_no_y, patch_no_x);
        patch_peaksx_tmp[i] = patch_peaksx[i];
        patch_peaksy_tmp[i] = patch_peaksy[i];
    }

    for ( int i = 0; i < badpointslength; i++ ) {
        Point nearest = findNearestNeighbor(badpoints[i], points);
        int   jj      = nearest.x / step_size_x / 2;
        int   ii      = nearest.y / step_size_y / 2;
        int   jjb     = badpoints[i].x / step_size_x / 2;
        int   iib     = badpoints[i].y / step_size_y / 2;
        for ( int image_ind = 0; image_ind < image_no; image_ind++ ) {
            patch_peaksx[image_ind](iib, jjb) = patch_peaksx_tmp[image_ind](ii, jj);
            patch_peaksy[image_ind](iib, jjb) = patch_peaksy_tmp[image_ind](ii, jj);
        }
    }
    // cout << "before fix" << endl;
    // cout << patch_peaksx_tmp[image_no - 1] << endl;
    // cout << "fixed" << endl;
    // // cout << reshape(shiftsplinex.z_on_knot, patch_num_x, patch_num_y) << endl;
    // cout << patch_peaksx[image_no - 1] << endl;

    // Spline3d.UpdateDiscreteValues(patch_peaksx);
    // Spline3dy.UpdateDiscreteValues(patch_peaksy);
    return missed_index;
}

// void CalculateDiffSquare(Image** ref_stack, Image** patch_stack, float unitless_bfactor, int patch_no, int image_no, bool write_out_the_map, std::string output_path, std::string file_pref) {
float CalculateDiffSquare(Image** patch_stack, int patch_no, int image_no, bool write_out_the_map, int max_thread, std::string output_path, std::string file_pref) {

    int   patch_dim   = patch_stack[0][0].logical_x_dimension;
    int   central_pix = (patch_stack[0][0].padding_jump_value + patch_dim) * patch_dim / 2 + patch_dim / 2;
    Image sum_of_images, sum_of_images_minus_current, diff_image, diff_image_copy;
    Peak  peak;
    // int    max_thread = 8;
    float *peak_sum, *pixmult;
    peak_sum                  = new float[patch_no];
    pixmult                   = new float[patch_no];
    float           total_Sum = 0.0;
    NumericTextFile outputpeaks(wxString::Format("%s%s_peaks.txt", output_path, file_pref).ToStdString( ), OPEN_TO_WRITE, 3);
    float           temp_array[3];
    wxPrintf("calculating diff square\n");
#pragma omp parallel for num_threads(max_thread) private(sum_of_images, sum_of_images_minus_current, diff_image, diff_image_copy, temp_array)
    for ( int patch_index = 0; patch_index < patch_no; patch_index++ ) {
        sum_of_images.Allocate(patch_dim, patch_dim, false);
        sum_of_images.SetToConstant(0.0);
        wxPrintf("patch %d\n", patch_index);
        peak_sum[patch_index] = 0;
        pixmult[patch_index]  = 0;

        for ( int image_counter = 0; image_counter < image_no; image_counter++ ) {
            sum_of_images.AddImage(&patch_stack[patch_index][image_counter]);
            if ( write_out_the_map ) {
                patch_stack[patch_index][image_counter].QuickAndDirtyWriteSlice(wxString::Format("%s%s_%02ishifted.mrc", output_path, file_pref, patch_index).ToStdString( ), image_counter + 1);
            }
        }
        for ( int image_counter = 0; image_counter < image_no; image_counter++ ) {
            sum_of_images_minus_current.Allocate(sum_of_images.logical_x_dimension, sum_of_images.logical_y_dimension, false);
            sum_of_images_minus_current.CopyFrom(&sum_of_images);
            sum_of_images_minus_current.SubtractImage(&patch_stack[patch_index][image_counter]);
            sum_of_images_minus_current.DivideByConstant(image_no - 1);
            if ( write_out_the_map ) {
                sum_of_images_minus_current.QuickAndDirtyWriteSlice(wxString::Format("%s%s_%02i.mrc", output_path, file_pref, patch_index).ToStdString( ), image_counter + 1);
            }

            diff_image.Allocate(sum_of_images.logical_x_dimension, sum_of_images.logical_y_dimension, false);
            diff_image_copy.Allocate(sum_of_images.logical_x_dimension, sum_of_images.logical_y_dimension, false);
            diff_image.CopyFrom(&sum_of_images_minus_current);
            diff_image.SubtractImage(&patch_stack[patch_index][image_counter]);
            diff_image_copy.CopyFrom(&diff_image);
            diff_image.CalculateCrossCorrelationImageWith(&diff_image_copy); //both in fourier space
            peak_sum[patch_index] += diff_image.real_values[central_pix];
            // wxPrintf("here?");
            // diff_image.BackwardFFT( );

            // diff_image.ForwardFFT( );
            // diff_image_copy.ForwardFFT( );
            // diff_image.CalculateCrossCorrelationImageWith(&diff_image_copy); //both in fourier space

            if ( write_out_the_map ) {
                diff_image.QuickAndDirtyWriteSlice(wxString::Format("%s%s_%02iCCmap.mrc", output_path, file_pref, patch_index).ToStdString( ), image_counter + 1);
            }
            // peak = sum_of_images_minus_current.FindPeakAtOriginFast2D();
            // peak = diff_image.FindPeakWithIntegerCoordinates( );
            // wxPrintf("peak value and pixel %f %f\n", peak.value, diff_image.real_values[central_pix]);
            // peak_sum[patch_index] += peak.value; // * patch_dim * patch_dim;
            // peak_sum[patch_index] += diff_image.real_values[central_pix];
            diff_image_copy.BackwardFFT( );
            pixmult[patch_index] += diff_image_copy.ReturnPixelWiseProduct(diff_image_copy);
            diff_image.Deallocate( );
            diff_image_copy.Deallocate( );
            sum_of_images_minus_current.Deallocate( );
            // wxPrintf("peak: x y value %g, %g, %g\n", peak.x, peak.y, peak.value);
            // wxPrintf("correlatoin and scaled peak.value %f %g\n", cc_value, peak.value * patch_dim * patch_dim);
        }
        temp_array[0] = patch_index;
        temp_array[1] = peak_sum[patch_index];
        temp_array[2] = pixmult[patch_index];
        outputpeaks.WriteLine(temp_array);
        // total_Sum += peak_sum[patch_index];
        total_Sum += pixmult[patch_index];
        // wxPrintf("here?1\n");
        // tmpimg.Deallocate( );
        sum_of_images.Deallocate( );
    }
    outputpeaks.Close( );
    total_Sum = total_Sum * patch_dim * patch_dim;
    wxPrintf("calculating diff square done\n");
    delete[] peak_sum;
    return total_Sum;
};

void Generate_CoeffSpline(bicubicsplinestack ccmap_stack, Image** patch_stack, float unitless_bfactor, int patch_no, int image_no, int max_thread, bool write_out_the_ccmap, std::string output_path, std::string file_pref, bool* disable_patches) {
    // void Generate_CoeffSpline(Image** patch_stack, float unitless_bfactor, int patch_no, int image_no, bool write_out_the_ccmap, std::string output_path, std::string file_pref) {

    int quater_patch_dim = ccmap_stack.m;
    int valid_index      = 0;
    for ( int patch_index = 0; patch_index < patch_no; patch_index++ ) {
        if ( disable_patches[patch_index] ) {
            continue;
        }
        valid_index = patch_index;
        break;
    }
    int   patch_dim = patch_stack[valid_index][0].logical_x_dimension; //based on patches are square
    Image sum_of_images, sum_of_images_minus_current, tmpimg, img_bfactor;

    // tmpimg.Allocate(quater_patch_dim, quater_patch_dim, false);
    // sum_of_images.Allocate(patch_dim, patch_dim, false);

    wxPrintf("------------------------------------generating coeffcient map spline------------------------------------\n");
    // int max_thread = 8;
#pragma omp parallel for num_threads(max_thread) private(sum_of_images, sum_of_images_minus_current, img_bfactor, tmpimg)
    for ( int patch_index = 0; patch_index < patch_no; patch_index++ ) {
        if ( disable_patches[patch_index] ) {
            wxPrintf("patch %d disabled\n", patch_index);
            continue;
        }
        tmpimg.Allocate(quater_patch_dim, quater_patch_dim, false);
        sum_of_images.Allocate(patch_dim, patch_dim, false);
        sum_of_images.SetToConstant(0.0);
        wxPrintf("patch %d\n", patch_index);
        // apply b factor to the patch stack and generate image sum
        // wxPrintf("check the patch stack values: \n");
        // for ( int ii = 0; ii < quater_patch_dim; ii += 50 ) {
        //     wxPrintf("patch_stack value, %f, %f, %f\n", patch_stack[0][0].real_values[ii]);
        // }
        for ( int image_counter = 0; image_counter < image_no; image_counter++ ) {
            img_bfactor.CopyFrom(&patch_stack[patch_index][image_counter]);
            // img_bfactor.ForwardFFT( );
            img_bfactor.ApplyBFactor(unitless_bfactor);
            sum_of_images.AddImage(&img_bfactor);
        }
        // generate correlation map
        for ( int image_counter = 0; image_counter < image_no; image_counter++ ) {
            sum_of_images_minus_current.CopyFrom(&sum_of_images);
            // each frame apply b factor
            img_bfactor.CopyFrom(&patch_stack[patch_index][image_counter]);
            // img_bfactor.ForwardFFT( );
            img_bfactor.ApplyBFactor(unitless_bfactor);
            sum_of_images_minus_current.SubtractImage(&img_bfactor);

            // // both apply b factor
            // img_bfactor.CopyFrom(&patch_stack[patch_index][image_counter]);
            // img_bfactor.ForwardFFT( );
            // sum_of_images_minus_current.SubtractImage(&img_bfactor);
            // img_bfactor.ApplyBFactor(unitless_bfactor);

            // // sum apply b factor, single image no bfactor
            // img_bfactor.CopyFrom(&patch_stack[patch_index][image_counter]);
            // sum_of_images_minus_current.SubtractImage(&img_bfactor);
            // sum_of_images_minus_current.ApplyBFactor(unitless_bfactor);
            // img_bfactor.ForwardFFT( );

            // sum_of_images_minus_current.ForwardFFT( );
            // img_bfactor.ForwardFFT( );
            // sum_of_images_minus_current.ApplyBFactor(unitless_bfactor);
            sum_of_images_minus_current.CalculateCrossCorrelationImageWith(&img_bfactor); //both in fourier space

            if ( write_out_the_ccmap ) {
                sum_of_images_minus_current.QuickAndDirtyWriteSlice(wxString::Format("%s%s_%02i.mrc", output_path, file_pref, patch_index).ToStdString( ), image_counter + 1);
            }
            sum_of_images_minus_current.ClipInto(&tmpimg, 0);
            if ( write_out_the_ccmap ) {
                tmpimg.QuickAndDirtyWriteSlice(wxString::Format("%s%s_%02icroped.mrc", output_path, file_pref, patch_index).ToStdString( ), image_counter + 1);
            }
            // wxPrintf("img %d\n", image_counter);
            // tmpimg.BackwardFFT( );
            // wxPrintf("check the real values\n");
            int pixel_counter = 0;
            int spline_ind    = patch_index * image_no + image_counter;
            ccmap_stack.spline_stack[spline_ind].z_on_knot.set_size(quater_patch_dim * quater_patch_dim, 1);

            for ( int ii = 0; ii < quater_patch_dim; ii++ ) {
                for ( int jj = 0; jj < quater_patch_dim; jj++ ) {
                    ccmap_stack.spline_stack[spline_ind].z_on_knot(ii * quater_patch_dim + jj) = tmpimg.real_values[pixel_counter];
                    pixel_counter++;
                }
                pixel_counter += tmpimg.padding_jump_value;
                // wxPrintf(" \n");
            }
            // wxPrintf("here?\n");
            ccmap_stack.UpdateSingleSpline(spline_ind);
            // wxPrintf(" out function qz %f %f %f\n", ccmap_stack.spline_stack[patch_index * image_no + image_counter].Qz2d(0, 0), ccmap_stack.spline_stack[patch_index * image_no + image_counter].Qz2d(10, 10), ccmap_stack.spline_stack[patch_index * image_no + image_counter].Qz2d(30, 30));

            // pirnt to check
            // for ( int ii = 0; ii < quater_patch_dim; ii += 50 ) {
            //     for ( int jj = 0; jj < quater_patch_dim; jj += 50 ) {
            //         wxPrintf("on knot, img, interp, %f, %f, %f\n", ccmap_stack.spline_stack[spline_ind].z_on_knot(ii * quater_patch_dim + jj), tmpimg.real_values[ii, jj], ccmap_stack.spline_stack[spline_ind].ApplySplineFunc(jj, ii));
            //     }
            // }
            // wxPrintf("input to continue\n");
            // int i;
            // cin >> i;
        }
        // wxPrintf("here?1\n");
        tmpimg.Deallocate( );
        sum_of_images.Deallocate( );
    }
    // for ( int ii = 0; ii < quater_patch_dim; ii++ ) {
    //     float shx, shy;
    //     wxPrintf("input your shiftx and y\n");
    //     cin >> shx;
    //     cin >> shy;
    //     cout << ccmap_stack.spline_stack[0].ApplySplineFunc(shx, shy) << endl;
    //     wxPrintf("result: %f\n", ccmap_stack.spline_stack[0].ApplySplineFunc(shx, shy));
    // }
    wxPrintf("generating spline coefficient map finished.\n");
}

void Generate_RefStack(Image** patch_stack, Image** ref_stack, int patch_no, int image_no) {
    int   patch_dim = patch_stack[0][0].logical_x_dimension; //based on patches are square
    Image sum_of_images, sum_of_images_minus_current;
    int   max_thread = 8;

#pragma omp parallel for num_threads(max_thread) private(sum_of_images)
    for ( int patch_index = 0; patch_index < patch_no; patch_index++ ) {

        sum_of_images.Allocate(patch_dim, patch_dim, false);
        sum_of_images.SetToConstant(0.0);
        wxPrintf("patch %d\n", patch_index);

        for ( int image_counter = 0; image_counter < image_no; image_counter++ ) {
            sum_of_images.AddImage(&patch_stack[patch_index][image_counter]);
        }

        for ( int image_counter = 0; image_counter < image_no; image_counter++ ) {
            ref_stack[patch_index][image_counter].CopyFrom(&sum_of_images);
            ref_stack[patch_index][image_counter].SubtractImage(&patch_stack[patch_index][image_counter]);
            ref_stack[patch_index][image_counter].DivideByConstant(float(image_no - 1));
        }
        sum_of_images.Deallocate( );
    }
}

void write_joins(std::string output_path, std::string join_file_pref, column_vector Join1d) {
    std::ofstream      xoFile, yoFile;
    char               buffer[200];
    std::string        join_file_name;
    std::ofstream      joinfile;
    std::ostringstream oss;
    oss << "%s%s%s" << output_path.c_str( ), join_file_pref.c_str( ), ".txt";
    std::snprintf(buffer, sizeof(buffer), "%s%s%s", output_path.c_str( ), join_file_pref.c_str( ), ".txt");
    join_file_name = buffer;
    // cout << "file " << join_file_name << endl;
    joinfile.open(join_file_name.c_str( ));

    for ( int i = 0; i < Join1d.size( ); i++ ) {
        joinfile << Join1d(i) << '\n';
    }
    joinfile.close( );
}

matrix<double> read_joins(std::string join_file_pref, std::string output_path, int joinsize) {
    wxPrintf("reading the joins from\n");
    std::ifstream      ijoinfile;
    char               buffer[200];
    std::ostringstream oss;
    std::string        join_file;
    matrix<double>     Join1d;
    // column_vector Join1d;
    // Join1d = ones_matrix<double>(knot_on_x * knot_on_y * knot_on_z * 2, 1);
    // std::string join_file_pref = "Joins_R1";
    oss << "%s%s%s" << output_path.c_str( ), join_file_pref.c_str( ), ".txt";
    std::snprintf(buffer, sizeof(buffer), "%s%s%s", output_path.c_str( ), join_file_pref.c_str( ), ".txt");
    // join_file = oss.str( );
    join_file = buffer;
    cout << "file " << join_file << endl;
    wxPrintf("file %s\n", join_file);
    ijoinfile.open(join_file.c_str( ));
    Join1d.set_size(joinsize, 1);
    // int tmpsize  = Join1d.size( );
    // wxPrintf("join1d size %d\n", tmpsize);
    for ( int i = 0; i < joinsize; i++ ) {
        ijoinfile >> Join1d(i);
    }
    ijoinfile.close( );
    wxPrintf("finish reading -----\n");
    return Join1d;
}

int patch_trimming_basedon_locations(Image* input_stack, Image** patch_stack, int number_of_images, int patch_num_x, int patch_num_y, int output_stack_box_size, std::string outpath, std::string file_pref, int max_threads, bool mean_padding, bool write_trimmed_files, float** patch_locations, bool* disable_patch) {

    int number_of_patchgroups = patch_num_x * patch_num_y;

    MRCFile* patch = new MRCFile[number_of_patchgroups];

    int  image_dim_x;
    int  image_dim_y;
    bool input_stack_is_in_real_space = input_stack[0].is_in_real_space;

    image_dim_x = input_stack[0].logical_x_dimension;
    image_dim_y = input_stack[0].logical_y_dimension;

    if ( ! input_stack_is_in_real_space ) {
        for ( int image_counter = 0; image_counter < number_of_images; image_counter++ ) {
            input_stack[image_counter].BackwardFFT( );
        }
    }

    //write a if statement to judge if the number of coordinates in the coord file equals to image_no

    // int step_size_x   = myroundint(float(image_dim_x) / float(patch_num_x) / 2); // check whether myroundint is correct
    // int step_size_y   = myroundint(float(image_dim_y) / float(patch_num_y) / 2);
    int patch_counter = 0;

    for ( int patch_y_ind = 0; patch_y_ind < patch_num_y; patch_y_ind++ ) {
        for ( int patch_x_ind = 0; patch_x_ind < patch_num_x; patch_x_ind++ ) {
            // patch[patch_counter].OpenFile(wxString::Format("%s%04i.mrc", outpath, patch_counter).ToStdString( ), true);
            if ( write_trimmed_files ) {
                patch[patch_counter].OpenFile(wxString::Format("%s%s_%04i.mrc", outpath, file_pref, patch_counter).ToStdString( ), true);
            }
            patch_counter += 1;
        }
    }

    wxPrintf("number of patch groups: %i\n\n", number_of_patchgroups);
    Image sum_of_images;
    sum_of_images.Allocate(image_dim_x, image_dim_y, 1);
    sum_of_images.SetToConstant(0.0);
    for ( int image_counter = 0; image_counter < number_of_images; image_counter++ ) {
        sum_of_images.AddImage(&input_stack[image_counter]);
    }
    Image mask_image;
    mask_image.CopyFrom(&sum_of_images);
    mask_image.ForwardFFT( );
    mask_image.GaussianLowPassFilter(0.01);
    mask_image.BackwardFFT( );
    mask_image.QuickAndDirtyWriteSlice("/tmp/gauss.mrc", 1);

    mask_image.Binarise(0.1 * mask_image.ReturnMaximumValue( ));

    // for ( int image_counter = 0; image_counter < number_of_images; image_counter++ ) {
#pragma omp parallel for default(shared) num_threads(max_threads)
    for ( int patch_counter = 0; patch_counter < number_of_patchgroups; patch_counter++ ) {
        wxPrintf("current patch: %i\n", patch_counter);
        Image mask_patch;
        mask_patch.Allocate(output_stack_box_size, output_stack_box_size, 1, true);
        float my_x = patch_locations[patch_counter][0] - image_dim_x / 2.0;
        float my_y = patch_locations[patch_counter][1] - image_dim_y / 2.0;
        mask_image.ClipInto(&mask_patch, 0.0, false, 1.0, int(my_x), int(my_y), 0);
        if ( mask_patch.ReturnAverageOfRealValues( ) < 0.99 ) {
            disable_patch[patch_counter] = true;
            wxPrintf("patch %i is disabled\n", patch_counter);
            continue;
        }
        for ( int image_counter = 0; image_counter < number_of_images; image_counter++ ) {
            float image_mean;
            if ( mean_padding ) {
                // wxPrintf(" yest mean padding");
                image_mean = input_stack[image_counter].ReturnAverageOfRealValues( );
            }
            else {
                image_mean = 0.0;
            }

            // wxPrintf("index trim center x y %d %f %f", patch_counter, my_x, my_y);

            patch_stack[patch_counter][image_counter].Allocate(output_stack_box_size, output_stack_box_size, 1, true);

            input_stack[image_counter].ClipInto(&patch_stack[patch_counter][image_counter], image_mean, false, 1.0, int(my_x), int(my_y), 0);
            if ( write_trimmed_files ) {
                patch_stack[patch_counter][image_counter].WriteSlice(&patch[patch_counter], image_counter + 1);
            }
        }
        patch[patch_counter].CloseFile( );
    }

    // delete input_coos_file;
    delete[] patch;

    if ( ! input_stack_is_in_real_space ) {
        // wxPrintf("input stack is not in real space\n");
#pragma omp parallel for default(shared) num_threads(max_threads)
        for ( int patch_ind = 0; patch_ind < number_of_patchgroups; patch_ind++ ) {
            if ( disable_patch[patch_ind] ) {
                continue;
            }
            for ( int image_ind = 0; image_ind < number_of_images; image_ind++ ) {
                patch_stack[patch_ind][image_ind].ForwardFFT(true);
                patch_stack[patch_ind][image_ind].ZeroCentralPixel( );
            }
        }
        for ( int image_counter = 0; image_counter < number_of_images; image_counter++ ) {
            input_stack[image_counter].ForwardFFT(true);
            input_stack[image_counter].ZeroCentralPixel( );
        }
    }

    // delete[] input_stack_real_space;
    wxPrintf("Done Patch Trimming\n");
    // return patch_stack;
    return number_of_patchgroups;
};
