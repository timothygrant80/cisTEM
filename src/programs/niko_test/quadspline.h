#include <dlib/dlib/queue.h>
#include <iostream>
#include <vector>
#include <numeric>
#include <dlib/dlib/matrix.h>

using namespace std;
using namespace dlib;

class quadspline {

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

    // void           InitializeSpline(int knot_no, float spline_patch_dim);
    void InitializeSpline(int knot_no, float spline_patch_dim) {
        this->n                = knot_no;
        this->spline_patch_dim = spline_patch_dim;
        this->total            = (this->n + 1);
        this->totaldim         = (this->n + 1);
        // CalcPhi( );
        this->Qy.set_size(this->totaldim, 1);
    }

    // void UpdateSplineControlPoints(matrix<double> Updated_Qy) {
    //     // this->y_on_knot = y_on_knot;
    //     // this->Qy        = CalcQy(y_on_knot);
    //     this->Qy = Updated_Qy;
    // }
    void UpdateSpline(matrix<double> Updated_Qy) {
        // this->y_on_knot = y_on_knot;
        // this->Qy        = CalcQy(y_on_knot);
        this->Qy = Updated_Qy;
    }

    void InitializeSplineModel(int knot_no, float spline_patch_dim, float spline_patch_dimx, matrix<double> x, matrix<double> y, matrix<double> Input_Qy) {
        this->n                = knot_no;
        this->spline_patch_dim = spline_patch_dim;
        this->x                = x;
        this->y                = y;
        // this->y_on_knot        = y_on_knot;
        this->Qy       = Input_Qy;
        this->total    = (this->n + 1);
        this->totaldim = (this->n + 1);
        // this->Mapping_Mat_no   = x.size( );
        // CalcPhi( );
        // this->Qy            = CalcQy(y_on_knot);
        // this->MappingMat    = MappingMatrix(x);
        this->FineGridCurve = SplineCurve(x, this->Qy);
    }

    double ApplySpline(double v, int pv, matrix<double> Qy) {
        double param_y;
        param_y = 1.0 / 2.0 * ((powf((1 - v), 2)) * Qy(pv) + (-2 * powf(v, 2) + 2 * v + 1) * Qy(pv + 1) + powf(v, 2) * Qy(pv + 2));
        return param_y;
    }

    matrix<double> SplineCurve(matrix<double> x, matrix<double> Qy) {
        int            lenx = x.size( );
        matrix<double> y_curve;
        y_curve.set_size(lenx, 1);
        matrix<double> PVX;
        double         v;
        PVX = x / this->spline_patch_dim;
        if ( PVX(lenx - 1) > this->n ) {
            cout << " the data set exists the boundary of the spline model" << endl;
        }
        int x_index_start = 0;
        int x_index_end;

        for ( int pv = 0; pv < this->n - 1; pv++ ) {
            // cout << "mark1" << endl;
            if ( x_index_start > lenx - 1 )
                break;
            while ( PVX(x_index_start) < pv )
                x_index_start++;
            x_index_end = x_index_start;

            if ( PVX(x_index_start) >= (pv + 1) )
                continue;

            while ( PVX(x_index_end) < (pv + 1) && x_index_end <= lenx - 1 ) {
                x_index_end++;
            }
            if ( x_index_end > x_index_start )
                x_index_end -= 1;

            for ( int j = x_index_start; j <= x_index_end; j++ ) {
                v          = PVX(j) - pv;
                y_curve(j) = ApplySpline(v, pv, Qy);
            }

            x_index_start = x_index_end + 1;
        }

        if ( (x_index_start <= (lenx - 1)) && ((PVX(x_index_start) - (this->n - 1)) == 0) ) {
            int pv                 = this->n - 2;
            v                      = PVX(x_index_start) - pv;
            y_curve(x_index_start) = ApplySpline(v, pv, Qy);
        }

        return y_curve;
    }

    matrix<double> MappingMatrix(matrix<double> x) {
        int lenx = x.size( );
        // matrix<double> y_curve;
        matrix<double> MappingMat;
        MappingMat.set_size(lenx, 3);
        this->Mapping_Mat_no = lenx;
        // y_curve.set_size(lenx, 1);
        int            count = 0;
        matrix<double> PVX;
        double         v;
        PVX = x / this->spline_patch_dim;
        if ( PVX(lenx - 1) > this->n ) {
            cout << " the data set exists the boundary of the spline model" << endl;
        }
        int x_index_start = 0;
        int x_index_end;

        for ( int pv = 0; pv < this->n - 1; pv++ ) {
            // cout << "mark1" << endl;
            if ( x_index_start > lenx - 1 )
                break;
            while ( PVX(x_index_start) < pv )
                x_index_start++;
            x_index_end = x_index_start;

            if ( PVX(x_index_start) >= (pv + 1) )
                continue;

            while ( PVX(x_index_end) < (pv + 1) && x_index_end <= lenx - 1 ) {
                x_index_end++;
            }
            if ( x_index_end > x_index_start )
                x_index_end -= 1;

            for ( int j = x_index_start; j <= x_index_end; j++ ) {
                v                    = PVX(j) - pv;
                MappingMat(count, 0) = j;
                MappingMat(count, 1) = pv;
                MappingMat(count, 2) = v;
                count++;
                // y_curve(j) = ApplySpline(v, pv, Qy);
            }

            x_index_start = x_index_end + 1;
        }

        if ( (x_index_start <= (lenx - 1)) && ((PVX(x_index_start) - (this->n - 1)) == 0) ) {
            int pv               = this->n - 2;
            v                    = PVX(x_index_start) - pv;
            MappingMat(count, 0) = x_index_start;
            MappingMat(count, 1) = pv;
            MappingMat(count, 2) = v;
            count++;
            // y_curve(x_index_start) = ApplySpline(v, pv, Qy);
        }

        return MappingMat;
    }

    // void           UpdateSpline(matrix<double> y_on_knot);
    // void           UpdateSplineInterpMapping(matrix<double> x);
    // void           InitializeSplineModel(int knot_no, float spline_patch_dim, float spline_patch_dimx, matrix<double> x, matrix<double> y, matrix<double> y_on_knot);
    // void           CalcPhi( );
    // matrix<double> CalcQy(matrix<double> y_on_knot);
    // double         ApplySpline(double v, int pv, matrix<double> Qy);
    // matrix<double> YOnGrid(matrix<double> Qy);
    // matrix<double> SplineCurve(matrix<double> x, matrix<double> Qy);
    // matrix<double> MappingMatrix(matrix<double> x);
    // matrix<double> ApplyMappingMat(matrix<double> Qy);
    // matrix<double> ApplyMappingMatWithMat(matrix<double> MappingMat, matrix<double> Qy);
    // double         ApplySplineFunc(double xp);
};