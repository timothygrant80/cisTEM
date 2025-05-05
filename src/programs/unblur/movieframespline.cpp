#include <dlib/dlib/queue.h>
#include "../../core/core_headers.h"
#include <iostream>
#include <vector>
#include <numeric>
#include <dlib/dlib/matrix.h>
// #include "./cubicspline.h"
// #include "./bicubicspline.h"
// #include "movieframespline_control.h"
#include "movieframespline.h"

void MovieFrameSpline::Initialize(int knot_no_along_z, int row_no, int column_no, int frame_no, int image_x_dim, int image_y_dim, float spline_knotz_distance, float spline_knotx_distance, float spline_knoty_distance) {
    wxPrintf("initial\n");
    this->frame_no    = frame_no;
    this->image_x_dim = image_x_dim;
    this->image_y_dim = image_y_dim;

    this->spline_patch_dimx = spline_knotx_distance;
    this->spline_patch_dimy = spline_knoty_distance;
    this->spline_patch_dimz = spline_knotz_distance;

    this->knot_no_y  = row_no;
    this->knot_no_x  = column_no;
    this->knot_no_z  = knot_no_along_z;
    this->total_knot = knot_no_along_z * row_no * column_no;

    this->value_on_knot = new matrix<double>*[row_no];
    for ( int i = 0; i < row_no; i++ ) {
        this->value_on_knot[i] = new matrix<double>[column_no];
    }

    this->Spline1d = new cubicspline*[row_no];
    for ( int i = 0; i < row_no; i++ ) {
        this->Spline1d[i] = new cubicspline[column_no];
    }
    // this->Spline1d = new quadspline*[row_no + 2];
    // for ( int i = 0; i < row_no + 2; i++ ) {
    //     this->Spline1d[i] = new quadspline[column_no + 2];
    // }

    for ( int i = 0; i < row_no; i++ ) {
        for ( int j = 0; j < column_no; j++ ) {
            this->Spline1d[i][j].InitializeSpline(knot_no_along_z, spline_patch_dimz);
        }
    }

    this->Spline2d = new bicubicspline[this->frame_no];
    for ( int i = 0; i < frame_no; i++ ) {
        this->Spline2d[i].InitializeSpline(row_no, column_no, spline_patch_dimy, spline_patch_dimx);
    }
    // this->phixy    = this->Spline2d[0].phi;
    // this->invphixy = this->Spline2d[0].invphi;
    wxPrintf("done initial\n");
    this->smooth_interp = new matrix<double>[this->frame_no];
    // this->discrete_values = new matrix<double>[this->frame_no];
}

void MovieFrameSpline::InitializeForward(int knot_no_along_z, int row_no, int column_no, int frame_no, int image_x_dim, int image_y_dim, float spline_knotz_distance, float spline_knotx_distance, float spline_knoty_distance) {
    this->frame_no    = frame_no;
    this->image_x_dim = image_x_dim;
    this->image_y_dim = image_y_dim;

    this->spline_patch_dimx = spline_knotx_distance;
    this->spline_patch_dimy = spline_knoty_distance;
    this->spline_patch_dimz = spline_knotz_distance;

    this->knot_no_y  = row_no;
    this->knot_no_x  = column_no;
    this->knot_no_z  = knot_no_along_z;
    this->total_knot = knot_no_along_z * row_no * column_no;

    this->value_on_knot = new matrix<double>*[row_no];
    for ( int i = 0; i < row_no; i++ ) {
        this->value_on_knot[i] = new matrix<double>[column_no];
    }

    this->Spline1d = new cubicspline*[row_no];
    for ( int i = 0; i < row_no; i++ ) {
        this->Spline1d[i] = new cubicspline[column_no];
    }
    // this->Spline1d = new quadspline*[row_no];
    // for ( int i = 0; i < row_no; i++ ) {
    //     this->Spline1d[i] = new quadspline[column_no];
    // }

    for ( int i = 0; i < row_no; i++ ) {
        for ( int j = 0; j < column_no; j++ ) {
            this->Spline1d[i][j].InitializeSplineForwardModel(knot_no_along_z, spline_patch_dimz);
        }
    }

    // this->phiz    = this->Spline1d[0][0].phi;
    // this->invphiz = this->Spline1d[0][0].invphi;

    this->Spline2d = new bicubicspline[this->frame_no];
    for ( int i = 0; i < frame_no; i++ ) {
        this->Spline2d[i].InitializeSplineForwardModel(row_no, column_no, spline_patch_dimy, spline_patch_dimx);
    }
    this->phixy    = this->Spline2d[0].phi;
    this->invphixy = this->Spline2d[0].invphi;
    // this->smooth_interp   = new matrix<double>[this->frame_no];
    // this->discrete_values = new matrix<double>[this->frame_no];
}

void MovieFrameSpline::Deallocate( ) {

    delete[] this->Spline2d;
    for ( int i = 0; i < this->knot_no_y; i++ ) {
        delete[] this->Spline1d[i];
        // delete[] this->value_on_knot[i];
    }
    if ( this->value_on_knot ) {
        for ( int i = 0; i < this->knot_no_y; i++ ) {
            delete[] this->value_on_knot[i];
        }
    }
    delete[] this->Spline1d;
    delete[] this->value_on_knot;
    delete[] this->smooth_interp;
    if ( this->discrete_values ) {
        delete[] this->discrete_values;
    }
}

void MovieFrameSpline::Destroy( ) {

    delete[] this->Spline2d;
    for ( int i = 0; i < this->knot_no_y; i++ ) {
        delete[] this->Spline1d[i];
        // delete[] this->value_on_knot[i];
    }
    if ( this->value_on_knot ) {
        for ( int i = 0; i < this->knot_no_y; i++ ) {
            delete[] this->value_on_knot[i];
        }
    }
    delete[] this->Spline1d;
    delete[] this->value_on_knot;
    delete[] this->smooth_interp;
    delete[] this->discrete_values;

    if ( this->smooth_interp ) {
        this->smooth_interp = nullptr;
    }
    if ( this->discrete_values ) {
        this->discrete_values = nullptr;
    }
    this->Spline2d      = nullptr;
    this->Spline1d      = nullptr;
    this->value_on_knot = nullptr;
    // this->smooth_interp   = nullptr;
    // this->discrete_values = nullptr;
}

void MovieFrameSpline::Update3DSplineOnKnotValue(matrix<double>** value_on_knot) {
    for ( int i = 0; i < this->knot_no_y; i++ ) {
        delete[] this->value_on_knot[i];
    }
    delete[] this->value_on_knot;

    this->value_on_knot = value_on_knot; //when it is quadratic b-spline, value_on_knot is value_on_control_points
}

void MovieFrameSpline::CopyFrom(MovieFrameSpline other_MovieFrameSpline) {
    *this = other_MovieFrameSpline;
}

void MovieFrameSpline::Update3DSplineInterpMapping(matrix<double> x_vector, matrix<double> y_vector, matrix<double> z_vector) {
    this->x                  = x_vector;
    this->y                  = y_vector;
    this->z                  = z_vector;
    this->Mapping_Mat_Row_no = y_vector.size( );
    this->Mapping_Mat_Col_no = x_vector.size( );

    matrix<double> Value_On_Image;
    Value_On_Image.set_size(this->knot_no_y, this->knot_no_x);

    for ( int i = 0; i < this->knot_no_y; i++ ) {
        for ( int j = 0; j < this->knot_no_x; j++ ) {
            this->Spline1d[i][j].UpdateSpline(this->value_on_knot[i][j]); //here is indeed update the qy for quadratic spline
            this->Spline1d[i][j].FineGridCurve = this->Spline1d[i][j].SplineCurve(this->z, this->Spline1d[i][j].Qy);
            this->Spline1d[i][j].MappingMat    = this->Spline1d[i][j].MappingMatrix(this->z);
        }
    }
    for ( int frame_ind = 0; frame_ind < this->frame_no; frame_ind++ ) {
        for ( int i = 0; i < this->knot_no_y; i++ ) {
            for ( int j = 0; j < this->knot_no_x; j++ ) {
                Value_On_Image(i, j) = this->Spline1d[i][j].FineGridCurve(frame_ind);
            }
        }
        this->Spline2d[frame_ind].UpdateSpline(Value_On_Image);
        this->Spline2d[frame_ind].MappingMat = this->Spline2d[frame_ind].MappingMatrix(x_vector, y_vector);
    }

    this->MappingMat_xy = this->Spline2d[0].MappingMatrix(this->x, this->y);
    this->MappingMat_z  = this->Spline1d[0][0].MappingMatrix(this->z);
}

void MovieFrameSpline::Update3DSplineInterpFrames(matrix<double> z_vector) {
    // wxPrintf("not enter mapping update?\n");

    // this->x                  = x_vector;
    // this->y                  = y_vector;
    this->z = z_vector;
    // this->Mapping_Mat_Row_no = y_vector.size( );
    // this->Mapping_Mat_Col_no = x_vector.size( );
    this->Mapping_Mat_z_no = z_vector.size( );

    // wxPrintf("in mapping update?\n");
    // matrix<double> Control_On_Image;
    // Control_On_Image.set_size(this->knot_no_y + 2, this->knot_no_x + 2);
    // wxPrintf("1sd?\n");
    this->Spline1d[0][0].MappingMat = this->Spline1d[0][0].MappingMatrix(this->z);
    // wxPrintf("2sd?\n");
    // this->Spline2d[0].MappingMat = this->Spline2d[0].MappingMatrix(x_vector, y_vector);
    // this->MappingMat_xy          = this->Spline2d[0].MappingMat;
    this->MappingMat_z = this->Spline1d[0][0].MappingMat;
    for ( int i = 0; i < this->knot_no_y; i++ ) {
        for ( int j = 0; j < this->knot_no_x; j++ ) {
            // wxPrintf("update1d\n");
            // this->Spline1d[i][j].UpdateSpline(this->value_on_knot[i][j]); //here is indeed update the qy for quadratic spline
            // this->Spline1d[i][j].FineGridCurve = this->Spline1d[i][j].SplineCurve(this->z, this->Spline1d[i][j].Qy);
            this->Spline1d[i][j].MappingMat     = this->MappingMat_z;
            this->Spline1d[i][j].Mapping_Mat_no = this->Mapping_Mat_z_no;
            // this->Spline1d[i][j].MappingMat = this->Spline1d[i][j].MappingMatrix(this->z);
        }
    }
    // for ( int frame_ind = 1; frame_ind < this->frame_no; frame_ind++ ) {
    //     // wxPrintf("update2d\n");
    //     this->Spline2d[frame_ind].MappingMat         = this->MappingMat_xy;
    //     this->Spline2d[frame_ind].Mapping_Mat_Row_no = this->Mapping_Mat_Row_no;
    //     this->Spline2d[frame_ind].Mapping_Mat_Col_no = this->Mapping_Mat_Col_no;
    //     // this->Spline2d[frame_ind].MappingMat = this->Spline2d[frame_ind].MappingMatrix(x_vector, y_vector);
    //     // this->Spline2d[frame_ind].MappingMat = this->Spline2d[frame_ind].MappingMatrix(this->x, this->y);
    //     // wxPrintf(" mappin mat row and col %d %d\n", int(this->Spline2d[frame_ind].Mapping_Mat_Row_no), int(this->Spline2d[frame_ind].Mapping_Mat_Col_no));
    // }
}

void MovieFrameSpline::Update3DSplineInterpMappingControl(matrix<double> x_vector, matrix<double> y_vector, matrix<double> z_vector) {
    // wxPrintf("not enter mapping update?\n");

    this->x                  = x_vector;
    this->y                  = y_vector;
    this->z                  = z_vector;
    this->Mapping_Mat_Row_no = y_vector.size( );
    this->Mapping_Mat_Col_no = x_vector.size( );
    this->Mapping_Mat_z_no   = z_vector.size( );

    // wxPrintf("in mapping update?\n");
    // matrix<double> Control_On_Image;
    // Control_On_Image.set_size(this->knot_no_y + 2, this->knot_no_x + 2);
    // wxPrintf("1sd?\n");
    this->Spline1d[0][0].MappingMat = this->Spline1d[0][0].MappingMatrix(this->z);
    // wxPrintf("2sd?\n");
    this->Spline2d[0].MappingMat = this->Spline2d[0].MappingMatrix(x_vector, y_vector);
    this->MappingMat_xy          = this->Spline2d[0].MappingMat;
    this->MappingMat_z           = this->Spline1d[0][0].MappingMat;
    for ( int i = 0; i < this->knot_no_y; i++ ) {
        for ( int j = 0; j < this->knot_no_x; j++ ) {
            // wxPrintf("update1d\n");
            // this->Spline1d[i][j].UpdateSpline(this->value_on_knot[i][j]); //here is indeed update the qy for quadratic spline
            // this->Spline1d[i][j].FineGridCurve = this->Spline1d[i][j].SplineCurve(this->z, this->Spline1d[i][j].Qy);
            this->Spline1d[i][j].MappingMat     = this->MappingMat_z;
            this->Spline1d[i][j].Mapping_Mat_no = this->Mapping_Mat_z_no;
            // this->Spline1d[i][j].MappingMat = this->Spline1d[i][j].MappingMatrix(this->z);
        }
    }
    for ( int frame_ind = 1; frame_ind < this->frame_no; frame_ind++ ) {
        // wxPrintf("update2d\n");
        this->Spline2d[frame_ind].MappingMat         = this->MappingMat_xy;
        this->Spline2d[frame_ind].Mapping_Mat_Row_no = this->Mapping_Mat_Row_no;
        this->Spline2d[frame_ind].Mapping_Mat_Col_no = this->Mapping_Mat_Col_no;
        // this->Spline2d[frame_ind].MappingMat = this->Spline2d[frame_ind].MappingMatrix(x_vector, y_vector);
        // this->Spline2d[frame_ind].MappingMat = this->Spline2d[frame_ind].MappingMatrix(this->x, this->y);
        // wxPrintf(" mappin mat row and col %d %d\n", int(this->Spline2d[frame_ind].Mapping_Mat_Row_no), int(this->Spline2d[frame_ind].Mapping_Mat_Col_no));
    }
}

// void MovieFrameSpline::Update3DSplineInterpMappingControl(matrix<double> x_vector, matrix<double> y_vector, matrix<double> z_vector) {
//     wxPrintf("not enter mapping update?\n");

//     this->x                  = x_vector;
//     this->y                  = y_vector;
//     this->z                  = z_vector;
//     this->Mapping_Mat_Row_no = y_vector.size( );
//     this->Mapping_Mat_Col_no = x_vector.size( );

//     // matrix<double> Value_On_Image;
//     // Value_On_Image.set_size(this->knot_no_y, this->knot_no_x);
//     wxPrintf("in mapping update?\n");
//     // matrix<double> Control_On_Image;
//     // Control_On_Image.set_size(this->knot_no_y + 2, this->knot_no_x + 2);
//     // wxPrintf("1sd?\n");
//     // this->Spline1d[0][0].MappingMat = this->Spline1d[0][0].MappingMatrix(this->z);
//     // wxPrintf("2sd?\n");
//     // this->Spline2d[0].MappingMat = this->Spline2d[0].MappingMatrix(x_vector, y_vector);
//     // this->MappingMat_xy          = this->Spline2d[0].MappingMat;
//     // this->MappingMat_z           = this->Spline1d[0][0].MappingMat;
//     for ( int i = 0; i < this->knot_no_y; i++ ) {
//         for ( int j = 0; j < this->knot_no_x; j++ ) {
//             wxPrintf("update1d\n");
//             // this->Spline1d[i][j].UpdateSpline(this->value_on_knot[i][j]); //here is indeed update the qy for quadratic spline
//             // this->Spline1d[i][j].FineGridCurve = this->Spline1d[i][j].SplineCurve(this->z, this->Spline1d[i][j].Qy);
//             // this->Spline1d[i][j].MappingMat = this->MappingMat_z;
//             this->Spline1d[i][j].MappingMat = this->Spline1d[i][j].MappingMatrix(this->z);
//         }
//     }
//     for ( int frame_ind = 1; frame_ind < this->frame_no; frame_ind++ ) {
//         // for ( int i = 0; i < this->knot_no_y + 2; i++ ) {
//         //     for ( int j = 0; j < this->knot_no_x + 2; j++ ) {
//         //         Control_On_Image(i, j) = this->Spline1d[i][j].FineGridCurve(frame_ind);
//         //     }
//         // }
//         // this->Spline2d[frame_ind].UpdateSpline2dControlPoints(Control_On_Image);
//         wxPrintf("update2d\n");
//         // this->Spline2d[frame_ind].MappingMat = this->MappingMat_xy;
//         // this->Spline2d[frame_ind].MappingMat = this->Spline2d[frame_ind].MappingMatrix(x_vector, y_vector);
//         this->Spline2d[frame_ind].MappingMat = this->Spline2d[frame_ind].MappingMatrix(this->x, this->y);
//         wxPrintf(" mappin mat row and col %d %d\n", int(this->Spline2d[frame_ind].Mapping_Mat_Row_no), int(this->Spline2d[frame_ind].Mapping_Mat_Col_no));
//     }

//     this->MappingMat_xy = this->Spline2d[0].MappingMatrix(this->x, this->y);
//     this->MappingMat_z  = this->Spline1d[0][0].MappingMatrix(this->z);
//     // this->MappingMat_xy = this->Spline2d[0].MappingMat;
//     // this->MappingMat_z  = this->Spline1d[0][0].MappingMat;
// }

void MovieFrameSpline::Update3DSpline1dInput(matrix<double> value_on_knot_1d) {
    matrix<double> tmp;
    matrix<double> Value_On_Image;
    // matrix<double>* grid_curves
    Value_On_Image.set_size(this->knot_no_y, this->knot_no_x);
    // cout << "interpspline" << value_on_knot_1d << endl;
    for ( int i = 0; i < this->knot_no_y; i++ ) {
        for ( int j = 0; j < this->knot_no_x; j++ ) {
            // tmp = rowm(value_on_knot_2d, (i * this->knot_no_x + j));
            int currentid = i * this->knot_no_x + j;
            // cout << "range " << range(currentid * this->knot_no_z, (currentid + 1) * this->knot_no_z - 1);
            // for cubic spline
            // tmp = rowm(value_on_knot_1d, range(currentid * this->knot_no_z, (currentid + 1) * this->knot_no_z - 1)); //, (currentid + 1) * this->knot_no_z - 1);
            // this->Spline1d[i][j].UpdateSpline(tmp);
            // for quadratic spline
            tmp = rowm(value_on_knot_1d, range(currentid * (this->knot_no_z + 1), (currentid + 1) * (this->knot_no_z + 1) - 1)); //, (currentid + 1) * this->knot_no_z - 1);
            this->Spline1d[i][j].UpdateSpline(tmp);
            this->Spline1d[i][j].FineGridCurve = this->Spline1d[i][j].SplineCurve(this->z, this->Spline1d[i][j].Qy);
            // double len                         = this->value_on_knot[i][j].size( );
            // grid_curves[i * this->knot_no_x + j] = this->Spline1d[i][j].ApplyMappingMat(Spline1d[i][j].Qy);
        }
    }

    for ( int frame_ind = 0; frame_ind < this->frame_no; frame_ind++ ) {
        for ( int i = 0; i < this->knot_no_y; i++ ) {
            for ( int j = 0; j < this->knot_no_x; j++ ) {
                Value_On_Image(i, j) = this->Spline1d[i][j].FineGridCurve(frame_ind);
            }
        }
        this->Spline2d[frame_ind].UpdateSpline(Value_On_Image);
        // Spline2d[frame_ind].MappingMat = Spline2d[frame_ind].MappingMatrix(x_vector, y_vector);
    }
}

void MovieFrameSpline::Update3DSpline1dInputControlPoints(matrix<double> Q_1d) {
    matrix<double> tmp;
    matrix<double> Value_On_Image;
    // matrix<double>* grid_curves
    // Value_On_Image.set_size(this->knot_no_y, this->knot_no_x);
    matrix<double> Control_value_On_Image;
    // Control_value_On_Image.set_size(this->knot_no_y + 2, this->knot_no_x + 2);
    Control_value_On_Image.set_size(this->knot_no_y, this->knot_no_x);
    // cout << "interpspline" << value_on_knot_1d << endl;
    for ( int i = 0; i < this->knot_no_y; i++ ) {
        for ( int j = 0; j < this->knot_no_x; j++ ) {
            // tmp = rowm(value_on_knot_2d, (i * this->knot_no_x + j));
            int currentid = i * (this->knot_no_x) + j;
            // cout << "range " << range(currentid * this->knot_no_z, (currentid + 1) * this->knot_no_z - 1);
            // for cubic spline
            // tmp = rowm(value_on_knot_1d, range(currentid * this->knot_no_z, (currentid + 1) * this->knot_no_z - 1)); //, (currentid + 1) * this->knot_no_z - 1);
            // this->Spline1d[i][j].UpdateSpline(tmp);
            // for quadratic spline
            // tmp = rowm(Q_1d, range(currentid * (this->knot_no_z + 1), (currentid + 1) * (this->knot_no_z + 1) - 1)); //, (currentid + 1) * this->knot_no_z - 1);
            // for cubic spline
            // tmp = rowm(Q_1d, range(currentid * (this->knot_no_z + 2), (currentid + 1) * (this->knot_no_z + 2) - 1)); //, (currentid + 1) * this->knot_no_z - 1);
            // for cubic splien
            tmp = rowm(Q_1d, range(currentid * (this->knot_no_z), (currentid + 1) * (this->knot_no_z) - 1)); //, (currentid + 1) * this->knot_no_z - 1);
            // wxPrintf("before update spline 1d control points\n");
            this->Spline1d[i][j].UpdateSpline1dControlPoints(tmp);
            // this->Spline1d[i][j].UpdateSpline(tmp);

            this->Spline1d[i][j].FineGridCurve = this->Spline1d[i][j].SplineCurve(this->z, this->Spline1d[i][j].Qy);
            // double len                         = this->value_on_knot[i][j].size( );
            // grid_curves[i * this->knot_no_x + j] = this->Spline1d[i][j].ApplyMappingMat(Spline1d[i][j].Qy);
        }
    }

    for ( int frame_ind = 0; frame_ind < this->frame_no; frame_ind++ ) {
        for ( int i = 0; i < this->knot_no_y; i++ ) {
            for ( int j = 0; j < this->knot_no_x; j++ ) {
                // Value_On_Image(i, j) = this->Spline1d[i][j].FineGridCurve(frame_ind);
                Control_value_On_Image(i, j) = this->Spline1d[i][j].FineGridCurve(frame_ind);
            }
        }
        // wxPrintf("before update spline 2d control points\n");

        this->Spline2d[frame_ind].UpdateSpline2dControlPoints(Control_value_On_Image);
        // wxPrintf("updated spline 2d control points %f\n", this->Spline2d[frame_ind].Qz2d(0, 0));

        // Spline2d[frame_ind].MappingMat = Spline2d[frame_ind].MappingMatrix(x_vector, y_vector);
    }
}

void MovieFrameSpline::Update3DSpline(matrix<double>** value_on_knot) {

    matrix<double> Value_On_Image;
    Value_On_Image.set_size(this->knot_no_y, this->knot_no_x);
    this->value_on_knot = value_on_knot;

    for ( int i = 0; i < this->knot_no_y; i++ ) {
        for ( int j = 0; j < this->knot_no_x; j++ ) {
            this->Spline1d[i][j].UpdateSpline(this->value_on_knot[i][j]);
            this->Spline1d[i][j].FineGridCurve = this->Spline1d[i][j].SplineCurve(this->z, this->Spline1d[i][j].Qy);
            // Spline1d[i][j].MappingMat    = Spline1d[i][j].MappingMatrix(this->z);
        }
    }
    for ( int frame_ind = 0; frame_ind < this->frame_no; frame_ind++ ) {
        for ( int i = 0; i < this->knot_no_y; i++ ) {
            for ( int j = 0; j < this->knot_no_x; j++ ) {
                Value_On_Image(i, j) = this->Spline1d[i][j].FineGridCurve(frame_ind);
            }
        }
        this->Spline2d[frame_ind].UpdateSpline(Value_On_Image);
        // this->Spline2d[frame_ind].MappingMat = this->Spline2d[frame_ind].MappingMatrix(x_vector, y_vector);
    }
}

matrix<double>* MovieFrameSpline::SmoothInterp( ) {
    // matrix<double>* interp;
    if ( this->smooth_interp != NULL ) {
        // wxPrintf("bf delete\n");
        delete[] this->smooth_interp;
        // wxPrintf("af delete\n");
    }
    // wxPrintf("bf delete\n");
    // delete[] this->smooth_interp;
    // wxPrintf("af delete\n");

    this->smooth_interp = new matrix<double>[this->frame_no];
    for ( int frame_ind = 0; frame_ind < this->frame_no; frame_ind++ ) {
        // if ( frame_ind == 0 ) {
        //     // wxPrintf("before apply mapping mat\n");
        //     wxPrintf("before apply mapping mat %f\n", this->Spline2d[frame_ind].Qz2d(0, 0));
        // }
        // wxPrintf("bf delete\n");
        this->smooth_interp[frame_ind] = this->Spline2d[frame_ind].ApplyMappingMat(this->Spline2d[frame_ind].Qz2d);
        // if ( frame_ind == 0 ) {
        //     // wxPrintf("before apply mapping mat\n");
        //     wxPrintf("after apply mapping mat %f\n", this->Spline2d[frame_ind].Qz2d(0, 1));
        // }
    }
    // interp = this->smooth_interp;
    // // this->smooth_interp = interp;
    // return interp;
    return this->smooth_interp;
}

matrix<double>* MovieFrameSpline::KnotToInterp(matrix<double> value_on_knot_1d) {
    matrix<double>* interp;
    this->Update3DSpline1dInput(value_on_knot_1d);
    interp = this->SmoothInterp( );
    return interp;
}

matrix<double>* MovieFrameSpline::ControlToInterp(matrix<double> Q_1d) {
    // matrix<double>* interp;
    this->Update3DSpline1dInputControlPoints(Q_1d);
    // interp = this->SmoothInterp( );
    // return interp;
    return this->SmoothInterp( );
}

// matrix<double>* MovieFrameSpline::KnotToInterp(matrix<double> value_on_knot_1d) {
//     matrix<double>* interp;
//     this->Update3DSpline1dInput(value_on_knot_1d);
//     interp = this->SmoothInterp( );
//     return interp;
// }

// matrix<double>* MovieFrameSpline::ControlToInterp(matrix<double> Q_1d) {
//     matrix<double>* interp;
//     // wxPrintf("before update 3dspline 1d input\n");
//     this->Update3DSpline1dInputControlPoints(Q_1d);
//     // wxPrintf("before smooth\n");
//     interp = this->SmoothInterp( );
//     return interp;
// }

void MovieFrameSpline::UpdateDiscreteValues(matrix<double>* Discret_Values_For_Smooth) {
    // matrix<double>* discrete_values;
    delete[] this->discrete_values;
    this->discrete_values = new matrix<double>[this->frame_no];

    for ( int frame_ind = 0; frame_ind < this->frame_no; frame_ind++ ) {
        // this->discrete_values[frame_ind].set_size(this->Mapping_Mat_Col_no,this->Mapping_Mat_Row_no);
        this->discrete_values[frame_ind] = Discret_Values_For_Smooth[frame_ind];
    }
    // wxPrintf("discrete values %f\n", this->discrete_values[0](0, 0));
}

double MovieFrameSpline::Apply3DSplineFunc(double x, double y, int image_index) {
    double value;
    value = this->Spline2d[image_index].ApplySplineFunc(x, y);
    return value;
}

double MovieFrameSpline::OptimizationKnotObejctFast(matrix<double> value_on_knot_1d) {
    matrix<double>* interp;
    // cout << "size" << value_on_knot_1d.size( ) << endl;
    // matrix<double>  value_on_knot_2d = reshape(value_on_knot_1d, this->knot_no_x * this->knot_no_y, this->knot_no_z);
    double error, current_error;
    interp = KnotToInterp(value_on_knot_1d);
    error  = 0.0;
    for ( int k = 0; k < this->frame_no; k++ ) {
        current_error = 0;
        for ( int i = 0; i < this->Mapping_Mat_Row_no; i++ ) {
            for ( int j = 0; j < this->Mapping_Mat_Col_no; j++ ) {
                // current_error = current_error + powf(std::abs(smooth_interp[k](i, j) - this->discrete_values[k](i, j)), 2);
                // error         = error + std::abs(smooth_interp[k](i, j) - this->discrete_values[k](i, j)) / this->frame_no;
                error = error + powf(std::abs(interp[k](i, j) - this->discrete_values[k](i, j)), 2) / this->frame_no;
                // cout << " image " << k << endl;
                // cout << " interp and discret val, error " << smooth_interp[k](i, j) << " " << this->discrete_values[k](i, j) << " " << powf(std::abs(smooth_interp[k](i, j) - this->discrete_values[k](i, j)), 2) << endl;
            }
        }
    }

    return error;
}

double MovieFrameSpline::OptimizationKnotObejctFromControlPoints(matrix<double> value_on_control_1d) {
    matrix<double>* interp;
    // cout << "size" << value_on_knot_1d.size( ) << endl;
    // matrix<double>  value_on_knot_2d = reshape(value_on_knot_1d, this->knot_no_x * this->knot_no_y, this->knot_no_z);
    double error, current_error;
    // interp = KnotToInterp(value_on_knot_1d);
    // wxPrintf("before control to interp\n");
    interp = ControlToInterp(value_on_control_1d);
    // wxPrintf("after control to interp\n");
    error = 0.0;
    for ( int k = 0; k < this->frame_no; k++ ) {
        current_error = 0;
        for ( int i = 0; i < this->Mapping_Mat_Row_no; i++ ) {
            for ( int j = 0; j < this->Mapping_Mat_Col_no; j++ ) {
                // current_error = current_error + powf(std::abs(smooth_interp[k](i, j) - this->discrete_values[k](i, j)), 2);
                // error         = error + std::abs(smooth_interp[k](i, j) - this->discrete_values[k](i, j)) / this->frame_no;
                // wxPrintf("here?\n");
                // wxPrintf("%f\n", interp[k](i, j));
                // wxPrintf("%f\n", this->discrete_values[k](i, j));
                error = error + powf(std::abs(interp[k](i, j) - this->discrete_values[k](i, j)), 2) / this->frame_no;
                // cout << " image " << k << endl;
                // cout << " interp and discret val, error " << smooth_interp[k](i, j) << " " << this->discrete_values[k](i, j) << " " << powf(std::abs(smooth_interp[k](i, j) - this->discrete_values[k](i, j)), 2) << endl;
            }
        }
    }
    // wxPrintf("error calculated\n");
    interp = nullptr;
    return error;
}