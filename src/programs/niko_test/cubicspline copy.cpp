#include <dlib/dlib/queue.h>
#include <iostream>
#include <vector>
#include <numeric>
#include <dlib/dlib/matrix.h>
#include "cubicspline.h"

using namespace std;
using namespace dlib;

void cubicspline::InitializeSpline(int knot_no, float spline_patch_dim) {
    // wxPrintf("initial3 cubicspline\n");
    this->n                = knot_no;
    this->spline_patch_dim = spline_patch_dim;
    this->total            = (this->n + 2);
    this->totaldim         = (this->n + 2);
    this->Qy.set_size(this->totaldim, 1);
}

void cubicspline::InitializeSplineForwardModel(int knot_no, float spline_patch_dim) {
    this->n                = knot_no;
    this->spline_patch_dim = spline_patch_dim;
    this->total            = (this->n + 2);
    this->totaldim         = (this->n + 2);
    CalcPhi( );
}

void cubicspline::UpdateSpline(matrix<double> y_on_knot) {
    this->y_on_knot = y_on_knot;
    this->Qy        = CalcQy(y_on_knot);
}

void cubicspline::UpdateSpline1dControlPoints(matrix<double> Qy1d_updated) {
    // this->z_on_knot = z_on_knot;
    // matrix<double> Qy;
    // this->Qy.set_size(this->totaldim, 1);
    // set boundary condition
    this->Qy(0)                  = Qy1d_updated(0) * 2 - Qy1d_updated(1);
    this->Qy(this->totaldim - 1) = Qy1d_updated(this->n - 1) * 2 - Qy1d_updated(this->n - 2);
    for ( int i = 1; i < this->totaldim - 1; i++ ) {
        this->Qy(i) = Qy1d_updated(i - 1);
    }
}

void cubicspline::UpdateSplineInterpMapping(matrix<double> x) {
    this->x              = x;
    this->Mapping_Mat_no = x.size( );
    this->MappingMat     = MappingMatrix(x);
}

void cubicspline::InitializeSplineModel(int knot_no, float spline_patch_dim, float spline_patch_dimx, matrix<double> x, matrix<double> y, matrix<double> y_on_knot) {
    this->n                = knot_no;
    this->spline_patch_dim = spline_patch_dim;
    this->x                = x;
    this->y                = y;
    this->y_on_knot        = y_on_knot;
    this->total            = (this->n + 2);
    this->totaldim         = (this->n + 2);
    this->Mapping_Mat_no   = x.size( );
    CalcPhi( );
    this->Qy            = CalcQy(y_on_knot);
    this->MappingMat    = MappingMatrix(x);
    this->FineGridCurve = SplineCurve(x, this->Qy);
}

void cubicspline::CalcPhi( ) {
    // this->phi.set_size(this->totaldim, this->totaldim);
    this->phi = zeros_matrix<double>(this->totaldim, this->totaldim);
    for ( int i = 0; i < this->n; i++ ) {
        this->phi(i + 1, i)     = 1;
        this->phi(i + 1, i + 1) = 4;
        this->phi(i + 1, i + 2) = 1;
    }

    // # end condition constraints
    this->phi(0, 0)                     = 1;
    this->phi(0, 1)                     = -2;
    this->phi(0, 2)                     = 1;
    this->phi(this->n + 1, this->n - 1) = 1;
    this->phi(this->n + 1, this->n)     = -2;
    this->phi(this->n + 1, this->n + 1) = 1;
    this->invphi.set_size(this->totaldim, this->totaldim);
    this->invphi = inv(this->phi);
}

matrix<double> cubicspline::CalcQy(matrix<double> y_on_knot) {
    // void CalcQy(matrix<double> z) {
    matrix<double> Qy1d;
    // matrix<double> Qz2d;
    matrix<double> invphi;
    Qy1d.set_size(this->totaldim, 1);
    // Qz2d.set_size((this->m + 2), (this->n + 2));
    // float Py[total];

    matrix<double> Py;
    // matrix<double> Qz1d, Qz2d;
    // Py.set_size(1, total * total);
    Py.set_size(this->totaldim, 1);
    Py(0) = 0;
    for ( int i = 1; i <= this->n; i++ ) {
        Py(i) = y_on_knot(i - 1);
    }
    Py(this->n + 1) = 0;

    invphi.set_size(this->totaldim, this->totaldim);
    invphi = inv(phi);

    Qy1d = invphi * Py * 6;

    return Qy1d;
}

double cubicspline::ApplySpline(double v, int pv, matrix<double> Qy) {
    double param_y;
    param_y = 1.0 / 6.0 * ((powf((1 - v), 3)) * Qy(pv) + (3 * powf(v, 3) - 6 * powf(v, 2) + 4) * Qy(pv + 1) + (-3 * powf(v, 3) + 3 * powf(v, 2) + 3 * v + 1) * Qy(pv + 2) + powf(v, 3) * Qy(pv + 3));
    return param_y;
}

matrix<double> cubicspline::YOnGrid(matrix<double> Qy) {
    matrix<double> GridY;
    GridY.set_size(this->n, 1);
    for ( int i = 0; i < this->n; i++ ) {
        GridY(i) = ApplySpline(0.0, i, Qy);
    }
    GridY(this->n - 1) = ApplySpline(1.0, this->n - 2, Qy);

    return GridY;
}

matrix<double> cubicspline::SplineCurve(matrix<double> x, matrix<double> Qy) {
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

matrix<double> cubicspline::MappingMatrix(matrix<double> x) {
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

matrix<double> cubicspline::ApplyMappingMat(matrix<double> Qy) {
    // Qz2d = reshape(Qy, (this->m + 2), (this->n + 2));
    matrix<double> smoothcurve;
    smoothcurve.set_size(this->Mapping_Mat_no, 1);
    int total_no = this->Mapping_Mat_no;
    int pv;

    float v;
    int   knot;

    for ( int i = 0; i < total_no; i++ ) {
        knot = this->MappingMat(i, 0);
        pv   = this->MappingMat(i, 1);
        v    = this->MappingMat(i, 2);

        smoothcurve(knot) = ApplySpline(v, pv, Qy);
    }
    return smoothcurve;
}

matrix<double> cubicspline::ApplyMappingMatWithMat(matrix<double> MappingMat, matrix<double> Qy) {
    // Qz2d = reshape(Qy, (this->m + 2), (this->n + 2));
    matrix<double> smoothcurve;
    smoothcurve.set_size(MappingMat.size( ), 1);
    int total_no = MappingMat.size( );
    int pv;

    float v;
    int   knot;

    for ( int i = 0; i < total_no; i++ ) {
        knot = MappingMat(i, 0);
        pv   = MappingMat(i, 1);
        v    = MappingMat(i, 2);

        smoothcurve(knot) = ApplySpline(v, pv, Qy);
    }
    return smoothcurve;
}

double cubicspline::ApplySplineFunc(double xp) {
    double yp;
    double PVX;
    double v;
    PVX = xp / this->spline_patch_dim;
    int pv;

    pv = int(PVX);
    // if ( PVX == this->n - 1 ) {
    if ( PVX <= this->n - 1 && int(PVX) == this->n - 1 ) {
        pv = this->n - 2;
    }
    if ( PVX > this->n - 1 ) {
        cout << "the point exceed the spline curve range." << endl;
    }
    v  = PVX - pv;
    yp = ApplySpline(v, pv, this->Qy);
    return yp;
}

// double OptimizationObejct(matrix<double> Qz1d) {}

// double OptimizationKnotObejct(matrix<double> join1d) {}
