#include <dlib/dlib/queue.h>
#include <iostream>
#include <vector>
#include <numeric>
#include <dlib/dlib/matrix.h>
#include "quadspline.h"

using namespace std;
using namespace dlib;

void quadspline::InitializeSpline(int knot_no, float spline_patch_dim) {
    this->n                = knot_no;
    this->spline_patch_dim = spline_patch_dim;
    this->total            = (this->n + 1);
    this->totaldim         = (this->n + 1);
    CalcPhi( );
}

void quadspline::CalcPhi( ) {
    // this->phi.set_size(this->totaldim, this->totaldim);
    this->phi = zeros_matrix<double>(this->totaldim, this->totaldim);
    for ( int i = 0; i < this->total; i++ ) {
        this->phi(i, i)     = 1;
        this->phi(i, i + 1) = 1;
    }

    // // # end condition constraints
    // this->phi(0, 0)                     = 1;
    // this->phi(0, 1)                     = -2;
    // this->phi(0, 2)                     = 1;
    // this->phi(this->n + 1, this->n - 1) = 1;
    // this->phi(this->n + 1, this->n)     = -2;
    // this->phi(this->n + 1, this->n + 1) = 1;
    this->invphi.set_size(this->totaldim, this->totaldim);
    this->invphi = inv(this->phi);
}