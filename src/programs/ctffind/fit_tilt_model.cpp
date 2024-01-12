#include "../../core/core_headers.h"

class
        FitTiltModel : public MyApp {

  public:
    bool DoCalculation( );
    void DoInteractiveUserInput( );

  private:
};

IMPLEMENT_APP(FitTiltModel)

void FitTiltModel::DoInteractiveUserInput( ) {
    UserInput*  my_input    = new UserInput("FitTiltModel", 1.0);
    std::string input_file  = "";
    std::string rawtlt_file = "";
    std::string output_path = "";
    float       tem_axis_angle;
    input_file     = my_input->GetFilenameFromUser("txt file containing ctffind tilt angle and axis direction", "second column is tilt angle and thrid column is axis direction", "TiltAngle_AxisDirction", false);
    rawtlt_file    = my_input->GetFilenameFromUser("tlt file containing the raw tilt values", "one column storing tilt angle", "rawtlt.tlt", false);
    tem_axis_angle = my_input->GetFloatFromUser("axis direction ", "the axis direction of the microscope", "178.4", 0.0);
    output_path    = my_input->GetFilenameFromUser("path to the output files", "/outpath/", "/data/output/", false);
    my_current_job.ManualSetArguments("ttft", input_file.c_str( ), rawtlt_file.c_str( ), tem_axis_angle, output_path.c_str( ));
};

using namespace std;
using namespace cistem;

std::vector<std::vector<double>> CTFRotationMatrix(double phi, double theta);
double                           CalcRMSE(const std::vector<double>& data, std::vector<int> indexes);
void                             MatrixToAngleZXZ(const std::vector<std::vector<double>>& R, double* theta, double* phi);
std::vector<double>              multiplyMatWithVec(const std::vector<std::vector<double>>& matrix, const std::vector<double>& vec);
std::vector<std::vector<double>> multiplyMatrices(const std::vector<std::vector<double>>& matrix1, const std::vector<std::vector<double>>& matrix2);
void                             adjust_ctffind_range_for_plot(std::vector<double> theta_series);
// void                             range_adjust_for_plot(std::vector<double>& theta_serie, std::vector<double>& phi_series, double tem_phi);

int                 image_no;
std::vector<int>    UpdatedIndices, outlier_indexes;
std::vector<double> ctffind_phi, ctffind_theta;
float*              raw_tilt;

double optim_function(void* pt2Object, double values[]) {
    std::vector<std::vector<double>>* ctf_rot_mat;
    std::vector<std::vector<double>>* tem_rot_mat;
    ctf_rot_mat = new std::vector<std::vector<double>>[image_no];
    tem_rot_mat = new std::vector<std::vector<double>>[image_no];

    double                           phi_zero   = values[1];
    double                           theta_zero = values[2];
    double                           phi_tem    = values[3];
    double                           rmse;
    float                            norm_mt;
    std::vector<double>              res;
    std::vector<int>                 index_vec;
    std::vector<double>              z_vec, ctf_vec, fit_vec, diff_vec;
    std::vector<std::vector<double>> zero_rot, tmp_mat;

    z_vec = {0, 0, 1};
    for ( int image_ind = 0; image_ind < image_no; image_ind++ ) {
        ctf_rot_mat[image_ind] = CTFRotationMatrix(ctffind_phi[image_ind], ctffind_theta[image_ind]);
        tem_rot_mat[image_ind] = CTFRotationMatrix(phi_tem, raw_tilt[image_ind]);
    }
    zero_rot = CTFRotationMatrix(phi_zero, theta_zero);

    res.clear( );
    for ( int image_ind = 0; image_ind < image_no; image_ind++ ) {
        ctf_vec = multiplyMatWithVec(ctf_rot_mat[image_ind], z_vec);
        // tmp_mat = multiplyMatrices(tem_rot_mat[image_ind], zero_rot);
        tmp_mat = multiplyMatrices(zero_rot, tem_rot_mat[image_ind]);
        fit_vec = multiplyMatWithVec(tmp_mat, z_vec);

        float sum_num = 0;
        for ( int k = 0; k < 3; k++ ) {
            sum_num += pow((ctf_vec[k] - fit_vec[k]), 2);
        }
        norm_mt = powf(sum_num, 0.5);
        res.push_back(norm_mt);
    }

    for ( int i = 0; i < image_no; i++ ) {
        index_vec.push_back(i);
    }

    int flag = 1;
    do {
        UpdatedIndices.clear( );
        outlier_indexes.clear( );
        rmse = CalcRMSE(res, index_vec);
        for ( size_t i = 0; i < index_vec.size( ); ++i ) {
            if ( res[index_vec[i]] / rmse <= 3.0 ) {
                UpdatedIndices.push_back(index_vec[i]);
            }
            else {
                outlier_indexes.push_back(index_vec[i]);
            }
        }
        index_vec.clear( );
        index_vec = UpdatedIndices;

        if ( outlier_indexes.size( ) == 0 ) {
            flag = 0;
        }
    } while ( flag == 1 );

    int count = 0;
    for ( int i = 0; i < image_no; ++i ) {
        if ( UpdatedIndices[count] == i ) {
            count++;
            continue;
        }
        else {
            outlier_indexes.push_back(i);
        }
    }
    delete[] ctf_rot_mat;
    delete[] tem_rot_mat;
    return rmse;
}

bool FitTiltModel::DoCalculation( ) {
    const std::string input_filename = my_current_job.arguments[0].ReturnStringArgument( );
    const std::string rawtltfile     = my_current_job.arguments[1].ReturnStringArgument( );
    float             input_tem_axis = my_current_job.arguments[2].ReturnFloatArgument( );
    const std::string outpath        = my_current_job.arguments[3].ReturnStringArgument( );

    int*   index;
    double fitted_phi_zero, fitted_theta_zero, fitted_phi_tem;
    double rmse;

    // std::vector<double>               abs_error_theta, abs_error_phi;
    std::vector<std::vector<double>>* exp_rot;
    std::vector<std::vector<double>>  zero_rot, tmp_mat;
    std::vector<double>               exp_theta, exp_phi;
    NumericTextFile                   inputfile(input_filename, OPEN_TO_READ, 3);
    NumericTextFile                   rawtlt(rawtltfile, OPEN_TO_READ, 1);

    // Read Inputs
    wxPrintf("read txt file by numeric txtfile\n");
    image_no = inputfile.number_of_lines;
    wxPrintf("number of tilts: %d\n", image_no);
    float temp_array[3];

    index    = new int[image_no];
    raw_tilt = new float[image_no];
    for ( int image_ind = 0; image_ind < image_no; image_ind++ ) {
        inputfile.ReadLine(temp_array);
        index[image_ind] = temp_array[0];
        ctffind_phi.push_back(360 - temp_array[1]);
        ctffind_theta.push_back(-temp_array[2]);
        rawtlt.ReadLine(&raw_tilt[image_ind]);
    }
    rawtlt.Close( );
    inputfile.Close( );

    DownhillSimplex simplex_minimzer(3);

    double ranges[4];
    double start_values[4];
    double min_values[4];

    ranges[0] = 0.0f;
    ranges[1] = 180.0f;
    ranges[2] = 180.0f;
    ranges[3] = 0.00f;

    start_values[0] = 0.0f;
    start_values[2] = 10;
    start_values[3] = 178.4f;
    start_values[3] = input_tem_axis;
    start_values[1] = start_values[3];

    simplex_minimzer.SetIinitalValues(start_values, ranges);

    simplex_minimzer.initial_values[1][1] = start_values[1] * simplex_minimzer.value_scalers[1] + ranges[1] * simplex_minimzer.value_scalers[1] * sqrtf(8.0f / 9.0f);
    simplex_minimzer.initial_values[1][2] = start_values[2] * simplex_minimzer.value_scalers[2];
    simplex_minimzer.initial_values[1][3] = start_values[3] * simplex_minimzer.value_scalers[3] - ranges[3] * simplex_minimzer.value_scalers[3] / 3.0f;

    simplex_minimzer.initial_values[2][1] = start_values[1] * simplex_minimzer.value_scalers[1] - ranges[1] * simplex_minimzer.value_scalers[1] * sqrtf(2.0f / 9.0f);
    simplex_minimzer.initial_values[2][2] = start_values[2] * simplex_minimzer.value_scalers[2] + ranges[2] * simplex_minimzer.value_scalers[2] * sqrtf(2.0f / 3.0f);
    simplex_minimzer.initial_values[2][3] = start_values[3] * simplex_minimzer.value_scalers[3] - ranges[3] * simplex_minimzer.value_scalers[3] / 3.0f;

    simplex_minimzer.initial_values[3][1] = start_values[1] * simplex_minimzer.value_scalers[1] - ranges[1] * simplex_minimzer.value_scalers[1] * sqrtf(2.0f / 9.0f);
    simplex_minimzer.initial_values[3][2] = start_values[2] * simplex_minimzer.value_scalers[2] - ranges[2] * simplex_minimzer.value_scalers[2] * sqrtf(2.0f / 3.0f);
    simplex_minimzer.initial_values[3][3] = start_values[3] * simplex_minimzer.value_scalers[3] - ranges[3] * simplex_minimzer.value_scalers[3] / 3.0f;

    simplex_minimzer.initial_values[4][1] = start_values[1] * simplex_minimzer.value_scalers[1];
    simplex_minimzer.initial_values[4][2] = start_values[2] * simplex_minimzer.value_scalers[2];
    simplex_minimzer.initial_values[4][3] = start_values[3] * simplex_minimzer.value_scalers[3] + ranges[3] * simplex_minimzer.value_scalers[3];

    simplex_minimzer.MinimizeFunction(this, optim_function);
    simplex_minimzer.GetMinimizedValues(min_values);

    wxPrintf("\nfitted result: phi_zero, theta_zero, phi_tem\n");
    wxPrintf(" %f, %f, %f\n", min_values[1], min_values[2], min_values[3]);

    fitted_phi_zero   = min_values[1];
    fitted_theta_zero = min_values[2];
    fitted_phi_tem    = min_values[3];

    if ( abs(fitted_phi_zero - input_tem_axis) > 145 ) {
        if ( fitted_phi_zero > input_tem_axis ) {
            fitted_phi_zero -= 180;
            fitted_theta_zero *= -1;
        }
        else {
            fitted_phi_zero += 180;
            fitted_theta_zero *= -1;
        }
    }

    wxPrintf("\nadjusted fitted result: phi_zero, theta_zero, phi_tem\n");
    wxPrintf(" %f, %f, %f\n", fitted_phi_zero, fitted_theta_zero, fitted_phi_tem);

    rmse = optim_function(this, min_values);
    wxPrintf("rmse is %f\n", rmse);

    zero_rot = CTFRotationMatrix(fitted_phi_zero, fitted_theta_zero);

    exp_rot = new std::vector<std::vector<double>>[image_no];

    exp_theta.resize(image_no);
    exp_phi.resize(image_no);

    for ( int image_ind = 0; image_ind < image_no; image_ind++ ) {
        tmp_mat            = CTFRotationMatrix(fitted_phi_tem, raw_tilt[image_ind]);
        exp_rot[image_ind] = multiplyMatrices(zero_rot, tmp_mat);
        MatrixToAngleZXZ(exp_rot[image_ind], &exp_theta[image_ind], &exp_phi[image_ind]);
    }

    // save results to files :
    wxPrintf("\nfitted tilt and axis direction\n");
    for ( int image_ind = 0; image_ind < image_no; image_ind++ ) {
        wxPrintf("%d %f %f\n", image_ind, exp_theta[image_ind], exp_phi[image_ind]);
    }

    wxPrintf("\noutlier indexes: ");
    NumericTextFile OutlierIndexFile(outpath + "outlier_index.txt", OPEN_TO_WRITE, 1);
    if ( outlier_indexes.size( ) == 0 ) {
        wxPrintf("\n--- all data pairs were used for fitting, no outliers --- \n");
    }
    else {
        for ( float value : outlier_indexes ) {
            wxPrintf("%f\t", value);
            OutlierIndexFile.WriteLine(&value);
        }
        wxPrintf("\n");
    }
    OutlierIndexFile.Close( );

    NumericTextFile outputfile(outpath + "fitted_parameter.txt", OPEN_TO_WRITE, 3);
    temp_array[0] = fitted_phi_zero;
    temp_array[1] = fitted_theta_zero;
    temp_array[2] = fitted_phi_tem;
    outputfile.WriteLine(temp_array);
    outputfile.Close( );

    NumericTextFile outputangle(outpath + "fitted_tilt_and_axis_angle.txt", OPEN_TO_WRITE, 3);
    for ( int i = 0; i < image_no; i++ ) {
        temp_array[0] = index[i];
        temp_array[1] = exp_theta[i];
        temp_array[2] = exp_phi[i];
        outputangle.WriteLine(temp_array);
    }
    outputangle.Close( );

    adjust_ctffind_range_for_plot(exp_theta);
    // range_adjust_for_plot(exp_theta, exp_phi, input_tem_axis);
    // range_adjust_for_plot(ctffind_theta, ctffind_phi, input_tem_axis);
    NumericTextFile  absolutionerror(outpath + "abserror_tilt_and_axis_angle.txt", OPEN_TO_WRITE, 3);
    float            sum_theta_error = 0;
    float            sum_phi_error   = 0;
    int              count           = 0;
    int              outlier_count   = 0;
    std::vector<int> excluded_index;
    for ( int i = 0; i < image_no; i++ ) {
        // wxPrintf("%d %f %f %d %f %f\n", i, exp_theta[i], exp_phi[i], i, ctffind_theta[i], ctffind_phi[i]);
        temp_array[0] = index[i];
        temp_array[1] = abs(exp_theta[i] - ctffind_theta[i]);
        temp_array[2] = abs(exp_phi[i] - ctffind_phi[i]);
        absolutionerror.WriteLine(temp_array);
        if ( outlier_indexes[outlier_count] == i && outlier_count < outlier_indexes.size( ) ) {
            excluded_index.push_back(outlier_indexes[outlier_count]);
            outlier_count++;
            continue;
        }
        if ( abs(exp_theta[i]) > 5 && ctffind_theta[i] > 5 ) {
            sum_theta_error += temp_array[1];
            sum_phi_error += temp_array[2];
            count++;
        }
        else {
            excluded_index.push_back(i);
        }
    }
    absolutionerror.Close( );

    wxPrintf("\nthe excluded index for calculating the mean abs error are:\n");
    for ( int value : excluded_index ) {
        wxPrintf(" %d\t", value);
    }
    wxPrintf("\n");
    float mean_theta_error = sum_theta_error / count;
    float mean_phi_error   = sum_phi_error / count;
    wxPrintf("\nthe mean abs error for theta and phi are: %f %f \n", mean_theta_error, mean_phi_error);

    delete[] index;
    delete[] raw_tilt;
    delete[] exp_rot;

    return true;
}

void MatrixToAngleZXZ(const std::vector<std::vector<double>>& R, double* theta, double* phi) {
    std::vector<double> z_vec;
    std::vector<double> vec_norm;
    double              rtoa = 57.2957795131;
    double              theta_phi_pair[2];

    z_vec = {0, 0, 1};

    vec_norm = multiplyMatWithVec(R, z_vec);
    *theta   = (double)acos(vec_norm[2]) * rtoa;

    if ( vec_norm[1] == 0 ) {
        *phi = 90;
    }
    else {
        *phi = (double)atan(vec_norm[0] / vec_norm[1]) * rtoa;
    }
    if ( *phi < 0 ) {
        *phi = *phi + 180;
    }
}

std::vector<std::vector<double>> CTFRotationMatrix(double phi, double theta) {

    std::vector<std::vector<double>> R(3, std::vector<double>(3, 0));

    double ator   = 0.0174532925;
    double cosphi = (double)cos(ator * phi);
    double sinphi = (double)sin(ator * phi);
    double costh  = (double)cos(ator * theta);
    double sinth  = (double)sin(ator * theta);

    R[0][0] = cosphi * cosphi + sinphi * sinphi * costh;
    R[0][1] = -cosphi * sinphi + cosphi * sinphi * costh;
    R[0][2] = -sinphi * sinth;
    R[1][0] = -cosphi * sinphi + cosphi * sinphi * costh;
    R[1][1] = cosphi * cosphi * costh + sinphi * sinphi;
    R[1][2] = -cosphi * sinth;
    R[2][0] = sinphi * sinth;
    R[2][1] = cosphi * sinth;
    R[2][2] = costh;
    return R;
};

double CalcRMSE(const std::vector<double>& data, std::vector<int> indexes) {
    std::vector<double> sortedData = data;
    std::sort(sortedData.begin( ), sortedData.end( ));

    double rmse = 0;
    for ( size_t i = 0; i < indexes.size( ); ++i ) {
        rmse = rmse + powf(data[indexes[i]], 2);
    }

    rmse = rmse / indexes.size( );
    rmse = powf(rmse, 0.5);

    return rmse;
};

std::vector<double> multiplyMatWithVec(const std::vector<std::vector<double>>& matrix, const std::vector<double>& vec) {
    if ( matrix.empty( ) || matrix[0].size( ) != vec.size( ) ) {
        throw std::invalid_argument("Matrix and vector dimensions do not match for multiplication.");
    }
    std::vector<double> result(matrix.size( ), 0);
    for ( size_t i = 0; i < matrix.size( ); ++i ) {
        for ( size_t j = 0; j < vec.size( ); ++j ) {
            result[i] += matrix[i][j] * vec[j];
        }
    }
    return result;
};

std::vector<std::vector<double>> multiplyMatrices(const std::vector<std::vector<double>>& matrix1, const std::vector<std::vector<double>>& matrix2) {
    if ( matrix1.empty( ) || matrix2.empty( ) || matrix1[0].size( ) != matrix2.size( ) ) {
        throw std::invalid_argument("Matrix dimensions do not match for multiplication.");
    }
    std::vector<std::vector<double>> result(matrix1.size( ), std::vector<double>(matrix2[0].size( ), 0));
    for ( size_t i = 0; i < matrix1.size( ); ++i ) {
        for ( size_t j = 0; j < matrix2[0].size( ); ++j ) {
            for ( size_t k = 0; k < matrix1[0].size( ); ++k ) {
                result[i][j] += matrix1[i][k] * matrix2[k][j];
            }
        }
    }
    return result;
};

void adjust_ctffind_range_for_plot(std::vector<double> theta_series) { //, std::vector<double>& phi_series, double tem_phi) {
    int count_pred    = std::count_if(theta_series.begin( ), theta_series.end( ), [](int x) { return x < 0; });
    int count_ctffind = std::count_if(ctffind_theta.begin( ), ctffind_theta.end( ), [](int x) { return x < 0; });
    if ( (count_pred - image_no / 2) * (count_ctffind - image_no / 2) < 0 ) {
        for ( int i = 0; i < image_no; i++ ) {
            ctffind_theta[i] *= -1;
            ctffind_phi[i] -= 180;
        }
    }
};

// void range_adjust_for_plot(std::vector<double>& theta_series, std::vector<double>& phi_series, double tem_phi) {
//     double imod_rot = tem_phi - 90;
//     int    count    = std::count_if(theta_series.begin( ), theta_series.end( ), [](int x) { return x < 0; });

//     if ( count > image_no / 2 ) {
//         for ( auto& element : theta_series ) {
//             element *= -1;
//         }
//         for ( auto& element : phi_series ) {
//             element += 180;
//         }
//     }
//     // if ( imod_rot > 45 ) {
//     //     for ( auto& element : phi_series ) {
//     //         element -= 90;
//     //     }
//     // }

//     for ( int i = 0; i < image_no; i++ ) {
//         while ( phi_series[i] > 360 ) {
//             phi_series[i] -= 360;
//         }

//         while ( phi_series[i] < 0 ) {
//             phi_series[i] += 360;
//         }

//         if ( phi_series[i] < (imod_rot) / 2.0 ) {
//             phi_series[i] += 180;
//             theta_series[i] *= -1;
//         }
//     }

//     // if ( imod_rot > 45 ) {
//     //     for ( auto& element : phi_series ) {
//     //         element += 90;
//     //     }
//     // }
// };
