// The contents of this file are in the public domain. See LICENSE_FOR_EXAMPLE_PROGRAMS.txt

/*
    This is an example illustrating the use of the matrix object 
    from the dlib C++ Library.
*/
#include <dlib/dlib/queue.h>
#include "../../core/core_headers.h"
#include <iostream>
#include <dlib/dlib/matrix.h>
#include <dlib/dlib/statistics.h>
#include <vector>
#include <dlib/dlib/matrix/matrix_abstract.h>

using namespace dlib;
using namespace std;

// ----------------------------------------------------------------------------------------

// void PCAReduction(const std::vector<Matrix>& data, double target_dim) {
//     dlib::vector_normalizer_pca<Matrix> pca;
//     pca.train(data, target_dim / data[0].nr( ));

//     std::vector<matrix> new_data;
//     new_data.reserve(data.size( ));
//     for ( size_t i = 0; i < data.size( ); ++i ) {
//         new_data.emplace_back(pca(data[i]));
//     }
//     for ( size_t r = 0; r < new_data.size( ); ++r ) {
//         Matrix vec = new_data(r);
//         double x   = vec(0, 0);
//         double y   = vec(1, 0);
//     }
// }

// void PCAReduction(const std::vector<matrix<double, 0, 1>>& data, double target_dim) {
//     vector_normalizer_pca<matrix<double, 0, 1>> pca, pcaa;
//     // pca.train(data, 0.99);
//     pca.train(data, target_dim / data[0].nr( ));
//     wxPrintf("size data.nr eps %li %li %f\n", data[0].size( ), data[0].nr( ), target_dim / data[0].nr( ));
//     wxPrintf("size data r and c %li %li\n", data[0].nr( ), data[0].nc( ));
//     // pca.train(data, target_dim / data[0].nc( ));
//     // pca.train(data, target_dim / 75);
//     // wxPrintf("data.nr eps %li %f\n", data[0].nc( ), target_dim / 75);
//     // wxPrintf("data.nr eps %li %f\n", data[0].nc( ), target_dim / data[0].nc( ));

//     std::vector<matrix<double, 0, 1>> new_data;
//     wxPrintf("datasize %li \n", data.size( ));
//     new_data.reserve(data.size( ));
//     for ( size_t i = 0; i < data.size( ); ++i ) {
//         new_data.emplace_back(pca(data[i]));
//         // wxPrintf("size pca %f\n", pca(data[i]));
//         // wxPrintf("size pca %li \n", (pca(data[i])).size( ));
//         // wxPrintf("size pca %li %f\n", (pca(data[i])).size( ), pca(data[i]));
//         // pcaa.train(data[i], 0.6);
//         // wxPrintf("tttt----------- %li, %f %f\n", (pcaa(data[i])).size( ), pcaa(data[i])(0), pcaa(data[i])(1));
//     }

//     for ( size_t r = 0; r < new_data.size( ); ++r ) {
//         matrix<double, 0, 1> vec = new_data[r];
//         double               x   = vec(0);
//         double               y   = vec(1);
//         double               z   = vec(2);
//         wxPrintf("new_data size: %li\n", new_data[r].size( ));
//         // wxPrintf("index x y %li %f %f %f\n", r, x, y, z);
//         // wxPrintf("new data x y %li %f %f %f\n", r, new_data[r](0), new_data[r](1), new_data[r](2));
//         wxPrintf("%li %f %f %f\n", r, new_data[r](0), new_data[r](1), new_data[r](2));
//         // wxPrintf("std size %li\n", pca.std_devs( ).size( ));
//         // wxPrintf("std %f\n", pca.std_devs( )(0));
//         // wxPrintf("pca_matrix size %li\n", pca.pca_matrix( ).size( ));
//         // wxPrintf("pca_matrix  %f %f \n", pca.pca_matrix( )(0), pca.pca_matrix( )(1));
//     }
//     wxPrintf("std size %li\n", pca.std_devs( ).size( ));
//     wxPrintf("std %f\n", pca.std_devs( )(0));
//     wxPrintf("mean %f\n", pca.means( )(0));
//     wxPrintf("pca_matrix size %li\n", pca.pca_matrix( ).size( ));
//     // for ( size_t r = 0; r < pca.pca_matrix( ).size( ); ++r ) {
//     //     wxPrintf("pca_matrix  %f\n", pca.pca_matrix( )(r));
//     // }
//     // wxPrintf("pca_matrix  %f %f \n", pca.pca_matrix( )(0), pca.pca_matrix( )(1));
// }

typedef matrix<double, 0, 1> column_vector;

std::vector<column_vector> PCAReduction(const std::vector<column_vector>& data, double target_dim) {
    vector_normalizer_pca<column_vector> pca, pcaa;
    pca.train(data, 0.99);
    // pca.train(data, target_dim / data[0].nr( ));
    wxPrintf("size data.nr eps %li %li %f\n", data[0].size( ), data[0].nr( ), target_dim / data[0].nr( ));
    wxPrintf("size data r and c %li %li\n", data[0].nr( ), data[0].nc( ));
    // pca.train(data, target_dim / data[0].nc( ));
    // pca.train(data, target_dim / 75);
    // wxPrintf("data.nr eps %li %f\n", data[0].nc( ), target_dim / 75);
    // wxPrintf("data.nr eps %li %f\n", data[0].nc( ), target_dim / data[0].nc( ));

    std::vector<column_vector> new_data;
    std::vector<column_vector> output_data;
    wxPrintf("datasize %li \n", data.size( ));
    new_data.reserve(data.size( ));
    for ( size_t i = 0; i < data.size( ); ++i ) {
        new_data.emplace_back(pca(data[i]));
        //  tmp = rowm(Q_1d, range(currentid * (this->knot_no_z), (currentid + 1) * (this->knot_no_z) - 1));
        output_data.emplace_back(rowm(pca(data[i]), range(0, target_dim - 1)));
        // wxPrintf("size of reduced %li\n", rowm(pca(data[i]), range(0, target_dim - 1)).size( ));
        // wxPrintf("size pca %f\n", pca(data[i]));
        // wxPrintf("size pca %li \n", (pca(data[i])).size( ));
        // wxPrintf("size pca %li %f\n", (pca(data[i])).size( ), pca(data[i]));
        // pcaa.train(data[i], 0.6);
        // wxPrintf("tttt----------- %li, %f %f\n", (pcaa(data[i])).size( ), pcaa(data[i])(0), pcaa(data[i])(1));
    }

    // for ( size_t r = 0; r < new_data.size( ); ++r ) {
    //     matrix<double, 0, 1> vec = new_data[r];
    //     double               x   = vec(0);
    //     double               y   = vec(1);
    //     double               z   = vec(2);
    //     wxPrintf("new_data size: %li\n", new_data[r].size( ));
    //     // wxPrintf("index x y %li %f %f %f\n", r, x, y, z);
    //     // wxPrintf("new data x y %li %f %f %f\n", r, new_data[r](0), new_data[r](1), new_data[r](2));
    //     wxPrintf("%li %f %f %f\n", r, new_data[r](0), new_data[r](1), new_data[r](2));
    //     // wxPrintf("std size %li\n", pca.std_devs( ).size( ));
    //     // wxPrintf("std %f\n", pca.std_devs( )(0));
    //     // wxPrintf("pca_matrix size %li\n", pca.pca_matrix( ).size( ));
    //     // wxPrintf("pca_matrix  %f %f \n", pca.pca_matrix( )(0), pca.pca_matrix( )(1));
    // }
    wxPrintf("std size %li\n", pca.std_devs( ).size( ));
    wxPrintf("std %f\n", pca.std_devs( )(0));
    wxPrintf("mean %f\n", pca.means( )(0));
    wxPrintf("pca_matrix size %li\n", pca.pca_matrix( ).size( ));
    // for ( size_t r = 0; r < pca.pca_matrix( ).size( ); ++r ) {
    //     wxPrintf("pca_matrix  %f\n", pca.pca_matrix( )(r));
    // }
    // wxPrintf("pca_matrix  %f %f \n", pca.pca_matrix( )(0), pca.pca_matrix( )(1));
    // return new_data;
    return output_data;
}

int main( ) {
    wxString      input_path = "/data/lingli/Lingli_20221028/grid2_process/MotCor202301/ER_HOXB8_ribosome/movies/s_records_15-25_quad/patch_12_08_b2_cspline_control_firstframezero/";
    std::ifstream File;
    wxString      shift_file_string = "shift";
    wxString      shift_file;
    // wxString      shift_filey;
    int                        patch_no_x = 12;
    int                        patch_no_y = 8;
    int                        patch_no   = patch_no_x * patch_no_y;
    int                        image_no   = 75;
    std::vector<column_vector> shiftx;
    std::vector<column_vector> shifty;
    std::vector<column_vector> shiftxt;
    std::vector<column_vector> shiftyt;
    std::vector<column_vector> PCAshiftx;
    std::vector<column_vector> PCAshifty;
    std::vector<column_vector> PCAshiftxt;
    std::vector<column_vector> PCAshiftyt;

    column_vector tmpx;
    column_vector tmpy;

    // tmpx.set_size(1, image_no);
    // tmpy.set_size(1, image_no);
    // tmpx.set_size(image_no, 1);
    // tmpy.set_size(image_no, 1);
    tmpx.set_size(patch_no, 1);
    tmpy.set_size(patch_no, 1);
    // matrix<double>*                   shiftx = new matrix<double>[patch_no];
    // matrix<double>*                   shifty = new matrix<double>[patch_no];
    //   patch_peaksx = new matrix<double>[number_of_input_images];
    // matrix<double>* shifty;
    // shiftx.set_size(patch_no, image_no);
    // shifty.set_size(patch_no, image_no);

    float** patch_shifts_x = new float*[patch_no];
    for ( int patch_ind = 0; patch_ind < patch_no; patch_ind++ ) {
        patch_shifts_x[patch_ind] = new float[image_no];
    }

    float** patch_shifts_y = new float*[patch_no];
    for ( int patch_ind = 0; patch_ind < patch_no; patch_ind++ ) {
        patch_shifts_y[patch_ind] = new float[image_no];
    }
    // load array from file
    /*
    for ( int patch_ind = 0; patch_ind < patch_no; patch_ind++ ) {
        shift_file = wxString::Format(input_path + "%04i_" + shift_file_string + ".txt", patch_ind);
        // wxPrintf("shiftfiles is %s, \n", shift_file);
        File.open(shift_file.c_str( ));

        if ( File.is_open( ) ) {
            // wxPrintf("files are open\n");
            for ( int image_ind = 0; image_ind < image_no; image_ind++ ) {
                File >> patch_shifts_x[patch_ind][image_ind];
                File >> patch_shifts_y[patch_ind][image_ind];
                // shiftx(patch_ind, image_ind) = patch_shifts_x[patch_ind][image_ind];
                // shifty(patch_ind, image_ind) = patch_shifts_y[patch_ind][image_ind];
                // shiftx[patch_ind](image_ind) = patch_shifts_x[patch_ind][image_ind];
                // shifty[patch_ind](image_ind) = patch_shifts_y[patch_ind][image_ind];
            }
        }
        // double meanvalue = mean(tmpx);
        // double stdvalue  = stddev(tmpx);
        // wxPrintf("mean %f %f\n", mean(tmpx), stddev(tmpx));
        // tmpx = (tmpx - meanvalue) / stdvalue;
        // wxPrintf("first tmpx %f\n", tmpx(0));

        File.close( );
        // load array from file end
    }

    for ( int image_ind = 0; image_ind < image_no; image_ind++ ) {
        for ( int patch_ind = 0; patch_ind < patch_no; patch_ind++ ) {
            tmpx(patch_ind) = patch_shifts_x[patch_ind][image_ind];
            tmpy(patch_ind) = patch_shifts_y[patch_ind][image_ind];
        }
        shiftx.push_back(tmpx);
        shifty.push_back(tmpy);
    }
    // */
    //stack the patches

    for ( int patch_ind = 0; patch_ind < patch_no; patch_ind++ ) {
        shift_file = wxString::Format(input_path + "%04i_" + shift_file_string + ".txt", patch_ind);
        // wxPrintf("shiftfiles is %s, \n", shift_file);
        File.open(shift_file.c_str( ));

        if ( File.is_open( ) ) {
            // wxPrintf("files are open\n");
            for ( int image_ind = 0; image_ind < image_no; image_ind++ ) {
                File >> patch_shifts_x[patch_ind][image_ind];
                File >> patch_shifts_y[patch_ind][image_ind];
                // shiftx(patch_ind, image_ind) = patch_shifts_x[patch_ind][image_ind];
                // shifty(patch_ind, image_ind) = patch_shifts_y[patch_ind][image_ind];
                // shiftx[patch_ind](image_ind) = patch_shifts_x[patch_ind][image_ind];
                // shifty[patch_ind](image_ind) = patch_shifts_y[patch_ind][image_ind];
                tmpx(image_ind) = patch_shifts_x[patch_ind][image_ind];
                tmpy(image_ind) = patch_shifts_y[patch_ind][image_ind];
            }
        }
        // double meanvalue = mean(tmpx);
        // double stdvalue  = stddev(tmpx);
        // wxPrintf("mean %f %f\n", mean(tmpx), stddev(tmpx));
        // tmpx = (tmpx - meanvalue) / stdvalue;
        // wxPrintf("first tmpx %f\n", tmpx(0));
        shiftx.push_back(tmpx);
        shifty.push_back(tmpy);
        File.close( );
        // load array from file end
    }

    for ( int image_ind = 0; image_ind < image_no; image_ind++ ) {
        for ( int patch_ind = 0; patch_ind < patch_no; patch_ind++ ) {
            tmpx(patch_ind) = patch_shifts_x[patch_ind][image_ind];
            tmpy(patch_ind) = patch_shifts_y[patch_ind][image_ind];
        }
        shiftxt.push_back(tmpx);
        shiftyt.push_back(tmpy);
    }
    PCAshiftx  = PCAReduction(shiftx, 3);
    PCAshiftxt = PCAReduction(shiftxt, 3);
    wxPrintf("size of shift t %li\n", PCAshiftxt.size( ));
    for ( size_t r = 0; r < PCAshiftx.size( ); ++r ) {
        matrix<double, 0, 1> vec = PCAshiftx[r];
        double               x   = vec(0);
        // double               y   = vec(1);
        // double               z   = vec(2);
        wxPrintf("new_data size: %li\n", PCAshiftx[r].size( ));
        // wxPrintf("index x y %li %f %f %f\n", r, x, y, z);
        // wxPrintf("new data x y %li %f %f %f\n", r, new_data[r](0), new_data[r](1), new_data[r](2));
        wxPrintf("%li %f %f %f\n", r, PCAshiftx[r](0), PCAshiftx[r](1), PCAshiftx[r](2));
        // wxPrintf("%li %f \n", r, PCAshiftx[r]);
        // wxPrintf("std size %li\n", pca.std_devs( ).size( ));
        // wxPrintf("std %f\n", pca.std_devs( )(0));
        // wxPrintf("pca_matrix size %li\n", pca.pca_matrix( ).size( ));
        // wxPrintf("pca_matrix  %f %f \n", pca.pca_matrix( )(0), pca.pca_matrix( )(1));
    }

    // PCAReduction(shifty, 1);

    // vector_normalizer_pca.train(tmpx, 1 / 75);
}