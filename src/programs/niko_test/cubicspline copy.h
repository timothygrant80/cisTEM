#include <dlib/dlib/queue.h>
#include <iostream>
#include <vector>
#include <numeric>
#include <dlib/dlib/matrix.h>

using namespace std;
using namespace dlib;

class cubicspline {

  public:
    int            n = 4; // knot no.
    matrix<double> y_on_knot;
    float          spline_patch_dim = 1;

    int            total; // remains to be check
    long           totaldim;
    matrix<double> phi;
    matrix<double> invphi;
    matrix<double> Qy;
    matrix<double> MappingMat;
    // matrix<double> UsedIndex;
    std::vector<double> UsedIndex;
    matrix<double>      x;
    matrix<double>      y;
    // matrix<double> z;
    matrix<double> FineGridCurve;
    long           Mapping_Mat_no;

    void InitializeSpline(int knot_no, float spline_patch_dim);
    void InitializeSplineForwardModel(int knot_no, float spline_patch_dim);

    void UpdateSpline(matrix<double> y_on_knot);
    void UpdateSpline1dControlPoints(matrix<double> Qy1d_updated);

    void           UpdateSplineInterpMapping(matrix<double> x);
    void           InitializeSplineModel(int knot_no, float spline_patch_dim, float spline_patch_dimx, matrix<double> x, matrix<double> y, matrix<double> y_on_knot);
    void           CalcPhi( );
    matrix<double> CalcQy(matrix<double> y_on_knot);
    double         ApplySpline(double v, int pv, matrix<double> Qy);
    matrix<double> YOnGrid(matrix<double> Qy);
    matrix<double> SplineCurve(matrix<double> x, matrix<double> Qy);
    matrix<double> MappingMatrix(matrix<double> x);
    matrix<double> ApplyMappingMat(matrix<double> Qy);
    matrix<double> ApplyMappingMatWithMat(matrix<double> MappingMat, matrix<double> Qy);
    double         ApplySplineFunc(double xp);
};