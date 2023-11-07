#include <dlib/dlib/queue.h>
#include <iostream>
#include <vector>
#include <numeric>
#include <dlib/dlib/matrix.h>

using namespace std;
using namespace dlib;

class bicubicspline {

  public:
    int            m = 4; // row no.
    int            n = 4; // column no.
    matrix<double> z_on_knot;
    float          spline_patch_dim_y = 1;
    float          spline_patch_dim_x = 1;

    int            total; // remains to be check
    long           totaldim;
    matrix<double> phi;
    matrix<double> Qz2d;
    matrix<double> MappingMat;
    // matrix<double> UsedIndex;
    std::vector<double> UsedIndex;
    matrix<double>      x;
    matrix<double>      y;
    matrix<double>      z;
    matrix<double>      FineGridSurface;
    long                Mapping_Mat_Row_no;
    long                Mapping_Mat_Col_no;
    // bicubicspline( );
    // ~bicubicspline( ); //what is this in Image() and ~Image() ????

    void           InitializeSpline(int row_no, int column_no, float spline_patch_dimy, float spline_patch_dimx);
    void           UpdateSpline(matrix<double> z_on_knot);
    void           UpdateSplineInterpMapping(matrix<double> x, matrix<double> y, matrix<double> z);
    void           InitializeSplineModel(int row_no, int column_no, float spline_patch_dimy, float spline_patch_dimx, matrix<double> x, matrix<double> y, matrix<double> z, matrix<double> z_on_knot);
    void           CalcPhi( );
    matrix<double> CalcQz(matrix<double> z_on_knot);
    double         ApplySpline(double u, double v, int pu, int pv, matrix<double> Qz2d);
    matrix<double> ZOnGrid(matrix<double> Qz2d);
    matrix<double> SplineSurface(matrix<double> x, matrix<double> y, matrix<double> Qz2d);
    matrix<double> MappingMatrix(matrix<double> x, matrix<double> y);
    matrix<double> ApplyMappingMat(matrix<double> Qz1d);
    double         OptimizationObejct(matrix<double> Qz1d);
    double         OptimizationKnotObejct(matrix<double> join1d);
};

