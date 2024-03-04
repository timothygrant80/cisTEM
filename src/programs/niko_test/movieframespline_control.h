#include <dlib/dlib/queue.h>
#include "../../core/core_headers.h"
#include <iostream>
#include <vector>
#include <numeric>
#include <dlib/dlib/matrix.h>
// #include "cubicspline.h"
#include "quadspline.h"
#include "bicubicspline.h"

using namespace std;
using namespace dlib;
using namespace cistem;

class MovieFrameSpline {

  public:
    int knot_no_x;
    int knot_no_y;
    int knot_no_z; // this is the knot no along the frame sequece dimension
    int total_knot;

    double spline_patch_dimx;
    double spline_patch_dimy;
    double spline_patch_dimz;

    matrix<double>  x;
    matrix<double>  y;
    matrix<double>  z;
    matrix<double>* discrete_values;
    matrix<double>* smooth_interp; // smooth_interp[i] is the smooth surface of i th frame

    int image_x_dim;
    int image_y_dim;
    int frame_no;

    // cubicspline**    Spline1d;
    quadspline**     Spline1d;
    bicubicspline*   Spline2d;
    matrix<double>   phiz;
    matrix<double>   phixy;
    matrix<double>   invphiz;
    matrix<double>   invphixy;
    matrix<double>   MappingMat_z;
    matrix<double>   MappingMat_xy;
    long             Mapping_Mat_Row_no;
    long             Mapping_Mat_Col_no;
    matrix<double>** value_on_knot; // shape: knot_no_y, knot_no_x, knot_no_z

    void            Initialize(int knot_no_along_z, int row_no, int column_no, int frame_no, int image_x_dim, int image_y_dim, float spline_knotz_distance, float spline_knotx_distance, float spline_knoty_distance);
    void            InitializeForward(int knot_no_along_z, int row_no, int column_no, int frame_no, int image_x_dim, int image_y_dim, float spline_knotz_distance, float spline_knotx_distance, float spline_knoty_distance);
    void            CopyFrom(MovieFrameSpline other_MovieFrameSpline);
    void            Update3DSplineOnKnotValue(matrix<double>** value_on_knot);
    void            Update3DSplineInterpMapping(matrix<double> x_vector, matrix<double> y_vector, matrix<double> z_vector);
    void            Update3DSplineInterpMappingControl(matrix<double> x_vector, matrix<double> y_vector, matrix<double> z_vector);
    void            Update3DSpline1dInput(matrix<double> value_on_knot_1d);
    void            Update3DSpline1dInputControlPoints(matrix<double> Q_1d);
    void            Update3DSpline(matrix<double>** value_on_knot);
    matrix<double>* SmoothInterp( );
    matrix<double>* KnotToInterp(matrix<double> value_on_knot_1d);
    matrix<double>* ControlToInterp(matrix<double> Q_1d);
    void            UpdateDiscreteValues(matrix<double>* Discret_Values_For_Smooth);
    // void            Update3DSpline1dInputControlPoints(matrix<double>* Discret_Values_For_Smooth);
    double Apply3DSplineFunc(double x, double y, int image_index);
    double OptimizationKnotObejctFast(matrix<double> value_on_knot_1d);

    double OptimizationKnotObejctFromControlPoints(matrix<double> value_on_control_1d);

    void Deallocate( );
};
