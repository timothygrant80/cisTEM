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
    matrix<double> invphi;
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

    void           InitializeSpline(int row_no, int column_no, float spline_patch_dimy, float spline_patch_dimx);
    void           UpdateSpline(matrix<double> z_on_knot);
    void           UpdateSplineInterpMapping(matrix<double> x, matrix<double> y, matrix<double> z);
    void           InitializeSplineModel(int row_no, int column_no, float spline_patch_dimy, float spline_patch_dimx, matrix<double> x, matrix<double> y, matrix<double> z, matrix<double> z_on_knot);
    void           CalcPhi( );
    matrix<double> CalcQz(matrix<double> z_on_knot);
    void           UpdateQz( );
    matrix<double> CalcQzWithInvPhi(matrix<double> InvPhi);
    // matrix<double> CalcQzWithInvPhi(matrix<double> z_on_knot, matrix<double> InvPhi);

    double         ApplySpline(double u, double v, int pu, int pv, matrix<double> Qz2d);
    matrix<double> ZOnGrid(matrix<double> Qz2d);
    matrix<double> SplineSurface(matrix<double> x, matrix<double> y, matrix<double> Qz2d);
    matrix<double> MappingMatrix(matrix<double> x, matrix<double> y);
    // matrix<double> ApplyMappingMat(matrix<double> Qz1d);
    matrix<double> ApplyMappingMat(matrix<double> Qz2d);
    double         OptimizationObejct(matrix<double> Qz1d);
    double         OptimizationKnotObejct(matrix<double> join1d);
    double         ApplySplineFunc(double xp, double yp);
};

class bicubicsplinestack {

  public:
    int    m; // row no.
    int    n; // column no.
    int    total_knot;
    int    total;
    double totaldim;
    int    spline_no;
    float  spline_patch_dim_y = 1;
    float  spline_patch_dim_x = 1;

    matrix<double> z_on_knot;

    matrix<double> phi;
    matrix<double> invphi;
    matrix<double> MappingMat;

    matrix<double> x;
    matrix<double> y;
    matrix<double> z;
    bicubicspline* spline_stack;

    long Mapping_Mat_Row_no;
    long Mapping_Mat_Col_no;

    // bicubicspline( );
    // ~bicubicspline( ); //what is this in Image() and ~Image() ????

    void InitializeSplineStack(int row_no, int column_no, int spline_no, float spline_patch_dimy, float spline_patch_dimx) {

        this->spline_no          = spline_no;
        this->m                  = row_no;
        this->n                  = column_no;
        this->total_knot         = this->m * this->n;
        this->total              = (this->m + 2) * (this->n + 2);
        this->totaldim           = (this->m + 2) * (this->n + 2);
        this->spline_patch_dim_x = spline_patch_dimx;
        this->spline_patch_dim_y = spline_patch_dimy;

        this->spline_stack = new bicubicspline[this->spline_no];

        this->spline_stack[0].InitializeSpline(this->m, this->n, this->spline_patch_dim_y, this->spline_patch_dim_x);
        this->phi    = this->spline_stack[0].phi;
        this->invphi = this->spline_stack[0].invphi;

        for ( int i = 1; i < this->spline_no; i++ ) {
            this->spline_stack[i].m                  = this->m;
            this->spline_stack[i].n                  = this->n;
            this->spline_stack[i].total              = this->total;
            this->spline_stack[i].totaldim           = this->totaldim;
            this->spline_stack[i].spline_patch_dim_x = spline_stack[0].spline_patch_dim_x;
            this->spline_stack[i].spline_patch_dim_y = spline_stack[0].spline_patch_dim_y;
        }
    }

    void UpdateSplineInterpMapping(matrix<double> x, matrix<double> y, matrix<double> z) {
        this->x                  = x;
        this->y                  = y;
        this->z                  = z;
        this->Mapping_Mat_Row_no = y.size( );
        this->Mapping_Mat_Col_no = x.size( );
        this->MappingMat         = this->spline_stack[0].MappingMatrix(x, y);
    }

    // void UpdateSingleSpline(matrix<double> z_on_knot, int spline_index) {
    void UpdateSingleSpline(int spline_index) {
        // this->spline_stack[spline_index].z_on_knot = z_on_knot;
        this->spline_stack[spline_index].Qz2d      = this->spline_stack[spline_index].CalcQzWithInvPhi(this->invphi);
        // this->spline_stack[spline_index].Qz2d = this->spline_stack[spline_index].CalcQz(z_on_knot);
        // wxPrintf("check function");
        // this->spline_stack[spline_index].Qz2d.set_size(this->m + 2, this->n + 2);
        // this->spline_stack[spline_index].UpdateQz( );
    }
};

// class bicubicsplinestack {

//   public:
//     int            m = 4; // row no.
//     int            n = 4; // column no.
//     int            frame_no;

//     matrix<double> z_on_knot;
//     // float          spline_patch_dim_y = 1;
//     // float          spline_patch_dim_x = 1;

//     // int            total; // remains to be check
//     // long           totaldim;
//     matrix<double> phi;
//     matrix<double> invphi;
//     // matrix<double> Qz2d;
//     matrix<double> MappingMat;
//     // matrix<double> UsedIndex;
//     // std::vector<double> UsedIndex;
//     matrix<double>      x;
//     matrix<double>      y;
//     matrix<double>      z;
//     // matrix<double>      FineGridSurface;
//     long                Mapping_Mat_Row_no;
//     long                Mapping_Mat_Col_no;
//     // bicubicspline( );
//     // ~bicubicspline( ); //what is this in Image() and ~Image() ????

//     void           InitializeSplineStack(int row_no, int column_no, float spline_patch_dimy, float spline_patch_dimx);

//     void           UpdateSpline(matrix<double> z_on_knot);
//     void           UpdateSplineInterpMapping(matrix<double> x, matrix<double> y, matrix<double> z);
//     void           InitializeSplineModel(int row_no, int column_no, float spline_patch_dimy, float spline_patch_dimx, matrix<double> x, matrix<double> y, matrix<double> z, matrix<double> z_on_knot);
//     void           CalcPhi( );
//     matrix<double> CalcQz(matrix<double> z_on_knot);
//     matrix<double> CalcQzWithInvPhi(matrix<double> InvPhi);
//     // matrix<double> CalcQzWithInvPhi(matrix<double> z_on_knot, matrix<double> InvPhi);

//     double         ApplySpline(double u, double v, int pu, int pv, matrix<double> Qz2d);
//     matrix<double> ZOnGrid(matrix<double> Qz2d);
//     matrix<double> SplineSurface(matrix<double> x, matrix<double> y, matrix<double> Qz2d);
//     matrix<double> MappingMatrix(matrix<double> x, matrix<double> y);
//     // matrix<double> ApplyMappingMat(matrix<double> Qz1d);
//     matrix<double> ApplyMappingMat(matrix<double> Qz2d);
//     double         OptimizationObejct(matrix<double> Qz1d);
//     double         OptimizationKnotObejct(matrix<double> join1d);
//     double         ApplySplineFunc(double xp, double yp);
// };