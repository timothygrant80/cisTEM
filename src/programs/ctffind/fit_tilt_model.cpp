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
    input_file              = my_input->GetFilenameFromUser("txt file containing ctffind tilt angle and axis direction", "second column is tilt angle and thrid column is axis direction", "TiltAngle_AxisDirction", false);
    rawtlt_file             = my_input->GetFilenameFromUser("tlt file containing the raw tilt values", "one column storing tilt angle", "rawtlt.tlt", false);
    output_path             = my_input->GetFilenameFromUser("path to the output files", "/outpath/", "/data/output/", false);
    my_current_job.ManualSetArguments("ttt", input_file.c_str( ), rawtlt_file.c_str( ), output_path.c_str( ));
};

using namespace std;
using namespace cistem;

std::vector<std::vector<double>> CTFRotationMatrix(double phi, double theta);
double                           CalcRMSE(const std::vector<double>& data, std::vector<int> indexes);
void                             MatrixToAngleZXZ(const std::vector<std::vector<double>>& R, double* theta, double* phi);
std::vector<double>              multiplyMatWithVec(const std::vector<std::vector<double>>& matrix, const std::vector<double>& vec);
std::vector<std::vector<double>> multiplyMatrices(const std::vector<std::vector<double>>& matrix1, const std::vector<std::vector<double>>& matrix2);

std::vector<std::vector<double>>* ctf_rot_mat;
std::vector<std::vector<double>>* tem_rot_mat;
std::vector<int>                  UpdatedIndices, outlier_indexes;
float*                            raw_tilt;
double*                           ctffind_phi;
double*                           ctffind_theta;
int                               image_no;

double optim_function(void* pt2Object, double values[]) {
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
        tmp_mat = multiplyMatrices(tem_rot_mat[image_ind], zero_rot);
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

    return rmse;
}

bool FitTiltModel::DoCalculation( ) {
    const std::string input_filename = my_current_job.arguments[0].ReturnStringArgument( );
    const std::string rawtltfile     = my_current_job.arguments[1].ReturnStringArgument( );
    const std::string outpath        = my_current_job.arguments[2].ReturnStringArgument( );

    int*    index;
    double* exp_theta;
    double* exp_phi;

    std::vector<std::vector<double>>* exp_rot;
    std::vector<std::vector<double>>  zero_rot;

    NumericTextFile inputfile(input_filename, OPEN_TO_READ, 3);
    NumericTextFile rawtlt(rawtltfile, OPEN_TO_READ, 1);

    // Read Inputs
    wxPrintf("read txt file by numeric txtfile\n");
    image_no = inputfile.number_of_lines;
    wxPrintf("number of tilts: %d\n", image_no);
    float temp_array[3];

    index         = new int[image_no];
    ctffind_phi   = new double[image_no];
    ctffind_theta = new double[image_no];
    raw_tilt      = new float[image_no];
    for ( int image_ind = 0; image_ind < image_no; image_ind++ ) {
        inputfile.ReadLine(temp_array);
        index[image_ind]         = temp_array[0];
        ctffind_phi[image_ind]   = temp_array[1];
        ctffind_theta[image_ind] = temp_array[2];
        rawtlt.ReadLine(&raw_tilt[image_ind]);
    }
    rawtlt.Close( );
    inputfile.Close( );

    ctf_rot_mat = new std::vector<std::vector<double>>[image_no];
    tem_rot_mat = new std::vector<std::vector<double>>[image_no];

    DownhillSimplex simplex_minimzer(3);

    double ranges[4];
    double start_values[4];
    double min_values[4];

    ranges[0] = 0.0f;
    ranges[1] = 180.0f;
    ranges[2] = 180.0f;
    ranges[3] = 180.0f;

    start_values[0] = 0.0f;
    start_values[1] = 90;
    start_values[2] = 10;
    start_values[3] = 90.0f;
    // check the object function value with the initila guess
    // float test      = optim_function(this, start_values);
    // wxPrintf("initial %f\n", test);

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

    wxPrintf("fitted result: phi_zero, theta_zero, phi_tem\n");
    wxPrintf(" %f, %f, %f\n", min_values[1], min_values[2], min_values[3]);
    // check the object function value after refinement
    // test = optim_function(this, min_values);
    // wxPrintf("final %f\n", test);

    double fitted_phi_zero   = min_values[1];
    double fitted_theta_zero = min_values[2];
    double fitted_phi_tem    = min_values[3];

    zero_rot = CTFRotationMatrix(fitted_phi_zero, fitted_theta_zero);

    exp_rot = new std::vector<std::vector<double>>[image_no];

    exp_theta = new double[image_no];
    exp_phi   = new double[image_no];
    for ( int image_ind = 0; image_ind < image_no; image_ind++ ) {
        ctf_rot_mat[image_ind] = CTFRotationMatrix(fitted_phi_tem, raw_tilt[image_ind]);
        exp_rot[image_ind]     = multiplyMatrices(ctf_rot_mat[image_ind], zero_rot);
        MatrixToAngleZXZ(exp_rot[image_ind], &exp_theta[image_ind], &exp_phi[image_ind]);
    }

    // save results to files :
    wxPrintf(" fitted tilt and axis direction\n");
    for ( int image_ind = 0; image_ind < image_no; image_ind++ ) {
        wxPrintf("%d %f %f\n", image_ind, exp_theta[image_ind], exp_phi[image_ind]);
    }
    if ( outlier_indexes.size( ) == 0 ) {
        wxPrintf("--- all data pairs were used for fitting, no outliers --- \n");
    }
    else {
        NumericTextFile OutlierIndexFile(outpath + "outlier_index.txt", OPEN_TO_WRITE, 1);
        wxPrintf("--- outliers' indexes are: ");
        for ( float value : outlier_indexes ) {
            wxPrintf("%d\t", value);
            OutlierIndexFile.WriteLine(&value);
        }
        wxPrintf("\n");
        OutlierIndexFile.Close( );
    }

    NumericTextFile outputfile(outpath + "fitted_parameter.txt", OPEN_TO_WRITE, 3);
    temp_array[0] = min_values[1];
    temp_array[1] = min_values[2];
    temp_array[2] = min_values[3];
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

    delete[] index;
    delete[] ctffind_phi;
    delete[] ctffind_theta;
    delete[] raw_tilt;
    delete[] ctf_rot_mat;
    delete[] tem_rot_mat;
    delete[] exp_rot;
    delete[] exp_theta;
    delete[] exp_phi;

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