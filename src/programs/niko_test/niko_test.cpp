// The contents of this file are in the public domain. See LICENSE_FOR_EXAMPLE_PROGRAMS.txt
/*

    This is an example illustrating the use the general purpose non-linear
    optimization routines from the dlib C++ Library.

    The library provides implementations of many popular algorithms such as L-BFGS
    and BOBYQA.  These algorithms allow you to find the minimum or maximum of a
    function of many input variables.  This example walks though a few of the ways
    you might put these routines to use.

*/

// #include <dlib/optimization.h>
// #include <dlib/global_optimization.h>
// #include <iostream>
#include <dlib/dlib/queue.h>
#include "../../core/core_headers.h"
#include <iostream>
// #include "../../core/dlib/queue.h"

// #include <string>
// #include <fstream>
// #include "../../core/dlib/optimization.h"
// #include "../../core/dlib/global_optimization.h"
#include "dlib/dlib/optimization.h"
#include "dlib/dlib/global_optimization.h"
// #include "../../core/dlib-part/optimization.h"
// #include "../../core/dlib-part/global_optimization.h"
// #include <iostream>
// #include <vector>

using namespace std;
using namespace dlib;

// ----------------------------------------------------------------------------------------

// In dlib, most of the general purpose solvers optimize functions that take a
// column vector as input and return a double.  So here we make a typedef for a
// variable length column vector of doubles.  This is the type we will use to
// represent the input to our objective functions which we will be minimizing.
typedef matrix<double, 0, 1> column_vector;

// ----------------------------------------------------------------------------------------
// Below we create a few functions.  When you get down into main() you will see that
// we can use the optimization algorithms to find the minimums of these functions.
// ----------------------------------------------------------------------------------------

double rosen(const column_vector& m)
/*
    This function computes what is known as Rosenbrock's function.  It is 
    a function of two input variables and has a global minimum at (1,1).
    So when we use this function to test out the optimization algorithms
    we will see that the minimum found is indeed at the point (1,1). 
*/
{
    const double x = m(0);
    const double y = m(1);

    // compute Rosenbrock's function and return the result
    return 100.0 * pow(y - x * x, 2) + pow(1 - x, 2);
}

// This is a helper function used while optimizing the rosen() function.
const column_vector rosen_derivative(const column_vector& m)
/*!
    ensures
        - returns the gradient vector for the rosen function
!*/
{
    const double x = m(0);
    const double y = m(1);

    // make us a column vector of length 2
    column_vector res(2);

    // now compute the gradient vector
    res(0) = -400 * x * (y - x * x) - 2 * (1 - x); // derivative of rosen() with respect to x
    res(1) = 200 * (y - x * x); // derivative of rosen() with respect to y
    return res;
}

// This function computes the Hessian matrix for the rosen() fuction.  This is
// the matrix of second derivatives.
matrix<double> rosen_hessian(const column_vector& m) {
    const double x = m(0);
    const double y = m(1);

    matrix<double> res(2, 2);

    // now compute the second derivatives
    res(0, 0) = 1200 * x * x - 400 * y + 2; // second derivative with respect to x
    res(1, 0) = res(0, 1) = -400 * x; // derivative with respect to x and y
    res(1, 1)             = 200; // second derivative with respect to y
    return res;
}

// ----------------------------------------------------------------------------------------

class rosen_model {
    /*!
        This object is a "function model" which can be used with the
        find_min_trust_region() routine.  
    !*/

  public:
    typedef ::column_vector column_vector;
    typedef matrix<double>  general_matrix;

    double operator( )(
            const column_vector& x) const { return rosen(x); }

    void get_derivative_and_hessian(
            const column_vector& x,
            column_vector&       der,
            general_matrix&      hess) const {
        der  = rosen_derivative(x);
        hess = rosen_hessian(x);
    }
};

// ----------------------------------------------------------------------------------------

int main( ) try {
    // int main( ) {
    // Set the starting point to (4,8).  This is the point the optimization algorithm
    // will start out from and it will move it closer and closer to the function's
    // minimum point.   So generally you want to try and compute a good guess that is
    // somewhat near the actual optimum value.
    column_vector starting_point = {4, 8};

    // The first example below finds the minimum of the rosen() function and uses the
    // analytical derivative computed by rosen_derivative().  Since it is very easy to
    // make a mistake while coding a function like rosen_derivative() it is a good idea
    // to compare your derivative function against a numerical approximation and see if
    // the results are similar.  If they are very different then you probably made a
    // mistake.  So the first thing we do is compare the results at a test point:
    cout << "Difference between analytic derivative and numerical approximation of derivative: "
         << length(derivative(rosen)(starting_point) - rosen_derivative(starting_point)) << endl;

    cout << "Find the minimum of the rosen function()" << endl;
    // Now we use the find_min() function to find the minimum point.  The first argument
    // to this routine is the search strategy we want to use.  The second argument is the
    // stopping strategy.  Below I'm using the objective_delta_stop_strategy which just
    // says that the search should stop when the change in the function being optimized
    // is small enough.

    // The other arguments to find_min() are the function to be minimized, its derivative,
    // then the starting point, and the last is an acceptable minimum value of the rosen()
    // function.  That is, if the algorithm finds any inputs to rosen() that gives an output
    // value <= -1 then it will stop immediately.  Usually you supply a number smaller than
    // the actual global minimum.  So since the smallest output of the rosen function is 0
    // we just put -1 here which effectively causes this last argument to be disregarded.

    find_min(bfgs_search_strategy( ), // Use BFGS search algorithm
             objective_delta_stop_strategy(1e-7), // Stop when the change in rosen() is less than 1e-7
             rosen, rosen_derivative, starting_point, -1);
    // Once the function ends the starting_point vector will contain the optimum point
    // of (1,1).
    cout << "rosen solution:\n"
         << starting_point << endl;

    // Now let's try doing it again with a different starting point and the version
    // of find_min() that doesn't require you to supply a derivative function.
    // This version will compute a numerical approximation of the derivative since
    // we didn't supply one to it.
    starting_point = {-94, 5.2};
    find_min_using_approximate_derivatives(bfgs_search_strategy( ),
                                           objective_delta_stop_strategy(1e-7),
                                           rosen, starting_point, -1);
    // Again the correct minimum point is found and stored in starting_point
    cout << "rosen solution:\n"
         << starting_point << endl;

    // Here we repeat the same thing as above but this time using the L-BFGS
    // algorithm.  L-BFGS is very similar to the BFGS algorithm, however, BFGS
    // uses O(N^2) memory where N is the size of the starting_point vector.
    // The L-BFGS algorithm however uses only O(N) memory.  So if you have a
    // function of a huge number of variables the L-BFGS algorithm is probably
    // a better choice.
    starting_point = {0.8, 1.3};
    find_min(lbfgs_search_strategy(10), // The 10 here is basically a measure of how much memory L-BFGS will use.
             objective_delta_stop_strategy(1e-7).be_verbose( ), // Adding be_verbose() causes a message to be
             // printed for each iteration of optimization.
             rosen, rosen_derivative, starting_point, -1);

    cout << endl
         << "rosen solution: \n"
         << starting_point << endl;

    starting_point = {-94, 5.2};
    find_min_using_approximate_derivatives(lbfgs_search_strategy(10),
                                           objective_delta_stop_strategy(1e-7),
                                           rosen, starting_point, -1);
    cout << "rosen solution: \n"
         << starting_point << endl;

    // dlib also supports solving functions subject to bounds constraints on
    // the variables.  So for example, if you wanted to find the minimizer
    // of the rosen function where both input variables were in the range
    // 0.1 to 0.8 you would do it like this:
    starting_point = {0.1, 0.1}; // Start with a valid point inside the constraint box.
    find_min_box_constrained(lbfgs_search_strategy(10),
                             objective_delta_stop_strategy(1e-9),
                             rosen, rosen_derivative, starting_point, 0.1, 0.8);
    // Here we put the same [0.1 0.8] range constraint on each variable, however, you
    // can put different bounds on each variable by passing in column vectors of
    // constraints for the last two arguments rather than scalars.

    cout << endl
         << "constrained rosen solution: \n"
         << starting_point << endl;

    // You can also use an approximate derivative like so:
    starting_point = {0.1, 0.1};
    find_min_box_constrained(bfgs_search_strategy( ),
                             objective_delta_stop_strategy(1e-9),
                             rosen, derivative(rosen), starting_point, 0.1, 0.8);
    cout << endl
         << "constrained rosen solution: \n"
         << starting_point << endl;

    // In many cases, it is useful if we also provide second derivative information
    // to the optimizers.  Two examples of how we can do that are shown below.
    starting_point = {0.8, 1.3};
    find_min(newton_search_strategy(rosen_hessian),
             objective_delta_stop_strategy(1e-7),
             rosen,
             rosen_derivative,
             starting_point,
             -1);
    cout << "rosen solution: \n"
         << starting_point << endl;

    // We can also use find_min_trust_region(), which is also a method which uses
    // second derivatives.  For some kinds of non-convex function it may be more
    // reliable than using a newton_search_strategy with find_min().
    starting_point = {0.8, 1.3};
    find_min_trust_region(objective_delta_stop_strategy(1e-7),
                          rosen_model( ),
                          starting_point,
                          10 // initial trust region radius
    );
    cout << "rosen solution: \n"
         << starting_point << endl;

    // Next, let's try the BOBYQA algorithm.  This is a technique specially
    // designed to minimize a function in the absence of derivative information.
    // Generally speaking, it is the method of choice if derivatives are not available
    // and the function you are optimizing is smooth and has only one local optima.  As
    // an example, consider the be_like_target function defined below:
    column_vector target         = {3, 5, 1, 7};
    auto          be_like_target = [&](const column_vector& x) {
        return mean(squared(x - target));
    };
    starting_point = {-4, 5, 99, 3};
    find_min_bobyqa(be_like_target,
                    starting_point,
                    9, // number of interpolation points
                    uniform_matrix<double>(4, 1, -1e100), // lower bound constraint
                    uniform_matrix<double>(4, 1, 1e100), // upper bound constraint
                    10, // initial trust region radius
                    1e-6, // stopping trust region radius
                    100 // max number of objective function evaluations
    );
    cout << "be_like_target solution:\n"
         << starting_point << endl;

    // Finally, let's try the find_min_global() routine.  Like find_min_bobyqa(),
    // this technique is specially designed to minimize a function in the absence
    // of derivative information.  However, it is also designed to handle
    // functions with many local optima.  Where BOBYQA would get stuck at the
    // nearest local optima, find_min_global() won't.  find_min_global() uses a
    // global optimization method based on a combination of non-parametric global
    // function modeling and BOBYQA style quadratic trust region modeling to
    // efficiently find a global minimizer.  It usually does a good job with a
    // relatively small number of calls to the function being optimized.
    //
    // You also don't have to give it a starting point or set any parameters,
    // other than defining bounds constraints.  This makes it the method of
    // choice for derivative free optimization in the presence of multiple local
    // optima.  Its API also allows you to define functions that take a
    // column_vector as shown above or to explicitly use named doubles as
    // arguments, which we do here.
    // ---------------------------------------------
    auto complex_holder_table = [](double x0, double x1) {
        // This function is a version of the well known Holder table test
        // function, which is a function containing a bunch of local optima.
        // Here we make it even more difficult by adding more local optima
        // and also a bunch of discontinuities.

        // add discontinuities
        double sign = 1;
        for ( double j = -4; j < 9; j += 0.5 ) {
            if ( j < x0 && x0 < j + 0.5 )
                x0 += sign * 0.25;
            sign *= -1;
        }
        // Holder table function tilted towards 10,10 and with additional
        // high frequency terms to add more local optima.
        return -(std::abs(sin(x0) * cos(x1) * exp(std::abs(1 - std::sqrt(x0 * x0 + x1 * x1) / pi))) - (x0 + x1) / 10 - sin(x0 * 10) * cos(x1 * 10));
    };

    // To optimize this difficult function all we need to do is call
    // find_min_global()
    auto result = find_min_global(complex_holder_table,
                                  {-10, -10}, // lower bounds
                                  {10, 10}, // upper bounds
                                  std::chrono::milliseconds(500) // run this long
    );

    cout.precision(9);
    // These cout statements will show that find_min_global() found the
    // globally optimal solution to 9 digits of precision:
    cout << "complex holder table function solution y (should be -21.9210397): " << result.y << endl;
    cout << "complex holder table function solution x:\n"
         << result.x << endl;

}

// * /

catch ( std::exception& e ) {

    cout << e.what( ) << endl;
}

// #include "../../core/core_headers.h"
// #include <iostream>

// #include <string>
// #include <fstream>
// #include "../../core/dlib/optimization.h"
// #include "../../core/dlib/global_optimization.h"
// #include <iostream>
// #include <vector>

// class
//         NikoTestApp : public MyApp {

//   public:
//     bool DoCalculation( );
//     void DoInteractiveUserInput( );

//   private:
// };

// IMPLEMENT_APP(NikoTestApp)

// // override the DoInteractiveUserInput

// void NikoTestApp::DoInteractiveUserInput( ) {
//     //     UserInput* my_input = new UserInput("TrimStack", 1.0);

//     //     wxString input_imgstack = my_input->GetFilenameFromUser("Input image file name", "Name of input image file *.mrc", "input.mrc", true);
//     //     // // wxString output_stack_filename = my_input->GetFilenameFromUser("Filename for output stack of particles.", "A stack of particles will be written to disk", "particles.mrc", false);
//     //     wxString angle_filename = my_input->GetFilenameFromUser("Tilt Angle filename", "The tilts, *.tlt", "ang.tlt", true);
//     //     // wxString coordinates_filename  = my_input->GetFilenameFromUser("Coordinates (PLT) filename", "The input particle coordinates, in Imagic-style PLT forlmat", "coos.plt", true);
//     //     int img_index = my_input->GetIntFromUser("Box size for output candidate particle images (pixels)", "In pixels. Give 0 to skip writing particle images to disk.", "256", 0);

//     //     delete my_input;

//     //     my_current_job.Reset(3);
//     //     my_current_job.ManualSetArguments("tti", input_imgstack.ToUTF8( ).data( ), angle_filename.ToUTF8( ).data( ), img_index);
// }

// // override the do calculation method which will be what is actually run..

// /*!
//  * Finds the coordinates of up to the [maxPeaks] highest peaks in [array], which is
//  * dimensioned to [nxdim] by [ny], and returns the positions in [xpeak], [ypeak], and the
//  * peak values in [peak].  If [minStrength] is greater than 0, then only those peaks that
//  * are greater than that fraction of the highest peak will be returned.  In addition, if
//  * [width] and [widthMin] are not NULL, the distance from the
//  * peak to the position at half of the peak height is measured in 8 directions, the
//  * overall mean width of the peak is returned in [width], and the minimum width along one
//  * of the four axes is returned in [widthMin].  The X size of the
//  * image is assumed to be [nxdim] - 2.  The sub-pixel position is determined by fitting
//  * a parabola separately in X and Y to the peak and 2 adjacent points.  Positions
//  * are numbered from zero and coordinates bigger than half the image size are
//  * shifted to be negative.  The positions are thus the amount to shift a second
//  * image in a correlation to align it to the first.  If fewer than [maxPeaks]
//  * peaks are found, then the remaining values in [peaks] will be -1.e30.
//  */
// #define B3DMIN(a, b) ((a) < (b) ? (a) : (b))
// #define B3DMAX(a, b) ((a) > (b) ? (a) : (b))
// #define B3DCLAMP(a, b, c) a = B3DMAX((b), B3DMIN((c), (a)))

// void XCorrPeakFindWidth(float* array, int nxdim, int ny, float* xpeak, float* ypeak,
//                         float* peak, float* width, float* widthMin, int maxPeaks,
//                         float minStrength) {
//     float cx, cy, y1, y2, y3, local, val;
//     float widthTemp[4];
//     int   ixpeak, iypeak, ix, iy, ixBlock, iyBlock, ixStart, ixEnd, iyStart, iyEnd;
//     int   i, j, ixm, ixp, iyb, iybm, iybp, idx, idy;
//     int   nx        = nxdim - 2;
//     int   blockSize = 5;
//     int   nBlocksX  = (nx + blockSize - 1) / blockSize;
//     int   nBlocksY  = (ny + blockSize - 1) / blockSize;
//     int   ixTopPeak = 10 * nx;
//     int   iyTopPeak = 10 * ny;
//     float xLimCen, yLimCen, xLimRadSq, yLimRadSq, threshold = 0.;

//     float sApplyLimits = -1;
//     float sLimitXlo    = -3000;
//     float sLimitYlo    = -3000;
//     float sLimitXhi    = 3000;
//     float sLimitYhi    = 3000;

//     /* If using elliptical limits, compute center and squares of radii */
//     if ( sApplyLimits < 0 ) {
//         xLimCen   = 0.5 * (sLimitXlo + sLimitXhi);
//         cx        = B3DMAX(1., (sLimitXhi - sLimitXlo) / 2.);
//         xLimRadSq = cx * cx;
//         yLimCen   = 0.5 * (sLimitYlo + sLimitYhi);
//         cy        = B3DMAX(1., (sLimitYhi - sLimitYlo) / 2.);
//         yLimRadSq = cy * cy;
//     }

//     /* find peaks */
//     for ( i = 0; i < maxPeaks; i++ ) {
//         peak[i]  = -1.e30f;
//         xpeak[i] = 0.;
//         ypeak[i] = 0.;
//     }

//     /* Look for highest peak if looking for one peak or if there is a minimum strength */
//     if ( maxPeaks < 2 || minStrength > 0. ) {

//         /* Find one peak within the limits */
//         if ( sApplyLimits ) {
//             for ( iy = 0; iy < ny; iy++ ) {
//                 idy = (iy > ny / 2) ? iy - ny : iy;
//                 if ( idy < sLimitYlo || idy > sLimitYhi )
//                     continue;
//                 for ( ix = 0; ix < nx; ix++ ) {
//                     idx = (ix > nx / 2) ? ix - nx : ix;
//                     if ( idx >= sLimitXlo && idx <= sLimitXhi && array[ix + iy * nxdim] > *peak ) {
//                         if ( sApplyLimits < 0 ) {
//                             cx = idx - xLimCen;
//                             cy = idy - yLimCen;
//                             if ( cx * cx / xLimRadSq + cy * cy / yLimRadSq > 1. )
//                                 continue;
//                         }
//                         *peak  = array[ix + iy * nxdim];
//                         ixpeak = ix;
//                         iypeak = iy;
//                     }
//                 }
//             }
//             if ( *peak > -0.9e30 ) {
//                 *xpeak = (float)ixpeak;
//                 *ypeak = (float)iypeak;
//             }
//         }
//         else {

//             /* Or just find the one peak in the whole area */
//             for ( iy = 0; iy < ny; iy++ ) {
//                 for ( ix = iy * nxdim; ix < nx + iy * nxdim; ix++ ) {
//                     if ( array[ix] > *peak ) {
//                         *peak  = array[ix];
//                         ixpeak = ix - iy * nxdim;
//                         iypeak = iy;
//                     }
//                 }
//             }
//             *xpeak = (float)ixpeak;
//             *ypeak = (float)iypeak;
//         }
//         threshold = minStrength * *peak;
//         ixTopPeak = ixpeak;
//         iyTopPeak = iypeak;
//     }

//     /* Now find all requested peaks */
//     if ( maxPeaks > 1 ) {

//         // Check for local peaks by looking at the highest point in each local
//         // block
//         for ( iyBlock = 0; iyBlock < nBlocksY; iyBlock++ ) {

//             // Block start and end in Y
//             iyStart = iyBlock * blockSize;
//             iyEnd   = iyStart + blockSize;
//             if ( iyEnd > ny )
//                 iyEnd = ny;

//             // Test if entire block is outside limits
//             if ( sApplyLimits && (iyStart > ny / 2 || iyEnd <= ny / 2) ) {
//                 idy = (iyStart > ny / 2) ? iyStart - ny : iyStart;
//                 if ( idy > sLimitYhi )
//                     continue;
//                 idy = (iyEnd > ny / 2) ? iyEnd - ny : iyEnd;
//                 if ( idy < sLimitYlo )
//                     continue;
//             }

//             // Loop on X blocks, get start and end in Y
//             for ( ixBlock = 0; ixBlock < nBlocksX; ixBlock++ ) {
//                 ixStart = ixBlock * blockSize;
//                 ixEnd   = ixStart + blockSize;
//                 if ( ixEnd > nx )
//                     ixEnd = nx;

//                 // Test if entire block is outside limits
//                 if ( sApplyLimits && (ixStart > nx / 2 || ixEnd <= nx / 2) ) {
//                     idx = (ixStart > nx / 2) ? ixStart - nx : ixStart;
//                     if ( idx > sLimitXhi )
//                         continue;
//                     idx = (ixEnd > nx / 2) ? ixEnd - nx : ixEnd;
//                     if ( idx < sLimitXlo )
//                         continue;
//                 }

//                 // Loop on every pixel in the block; have to test each pixel
//                 local = -1.e30f;
//                 for ( iy = iyStart; iy < iyEnd; iy++ ) {
//                     if ( sApplyLimits ) {
//                         idy = (iy > ny / 2) ? iy - ny : iy;
//                         if ( idy < sLimitYlo || idy > sLimitYhi )
//                             continue;
//                     }
//                     for ( ix = ixStart; ix < ixEnd; ix++ ) {
//                         if ( sApplyLimits ) {
//                             idx = (ix > nx / 2) ? ix - nx : ix;
//                             if ( idx < sLimitXlo || idx > sLimitXhi )
//                                 continue;

//                             // Apply elliptical test
//                             if ( sApplyLimits < 0 ) {
//                                 cx = idx - xLimCen;
//                                 cy = idy - yLimCen;
//                                 if ( cx * cx / xLimRadSq + cy * cy / yLimRadSq > 1. )
//                                     continue;
//                             }
//                         }
//                         val = array[ix + iy * nxdim];
//                         if ( val > local && val > peak[maxPeaks - 1] && val > threshold ) {
//                             local  = val;
//                             ixpeak = ix;
//                             iypeak = iy;
//                         }
//                     }
//                 }

//                 // evaluate local peak for truly being local.
//                 // Allow equality on one side, otherwise identical adjacent values are lost
//                 if ( local > -0.9e30 ) {
//                     ixm  = (ixpeak + nx - 1) % nx;
//                     ixp  = (ixpeak + 1) % nx;
//                     iyb  = iypeak * nxdim;
//                     iybp = ((iypeak + 1) % ny) * nxdim;
//                     iybm = ((iypeak + ny - 1) % ny) * nxdim;

//                     if ( local > array[ixpeak + iybm] && local >= array[ixpeak + iybp] &&
//                          local > array[ixm + iyb] && local >= array[ixp + iyb] &&
//                          local > array[ixm + iybp] && local >= array[ixp + iybm] &&
//                          local > array[ixp + iybp] && local >= array[ixm + iybm] &&
//                          (ixpeak != ixTopPeak || iypeak != iyTopPeak) ) {

//                         // Insert peak into the list
//                         for ( i = 0; i < maxPeaks; i++ ) {
//                             if ( peak[i] < local ) {
//                                 for ( j = maxPeaks - 1; j > i; j-- ) {
//                                     peak[j]  = peak[j - 1];
//                                     xpeak[j] = xpeak[j - 1];
//                                     ypeak[j] = ypeak[j - 1];
//                                 }
//                                 peak[i]  = local;
//                                 xpeak[i] = (float)ixpeak;
//                                 ypeak[i] = (float)iypeak;
//                                 break;
//                             }
//                         }
//                     }
//                 }
//             }
//         }
//     }

//     // for ( i = 0; i < maxPeaks; i++ ) {
//     //     if ( peak[i] < -0.9e30 )
//     //         continue;

//     // // Add 0.2 just in case float was less than int assigned to it
//     // ixpeak = (int)(xpeak[i] + 0.2);
//     // iypeak = (int)(ypeak[i] + 0.2);

//     // /* simply fit a parabola to the two adjacent points in X or Y */

//     // y1 = array[(ixpeak + nx - 1) % nx + iypeak * nxdim];
//     // y2 = peak[i];
//     // y3 = array[(ixpeak + 1) % nx + iypeak * nxdim];
//     // cx = (float)parabolicFitPosition(y1, y2, y3);

//     // y1 = array[ixpeak + ((iypeak + ny - 1) % ny) * nxdim];
//     // y3 = array[ixpeak + ((iypeak + 1) % ny) * nxdim];
//     // cy = (float)parabolicFitPosition(y1, y2, y3);

//     // /*    return adjusted pixel coordinate */
//     // xpeak[i] = ixpeak + cx;
//     // ypeak[i] = iypeak + cy;
//     // if ( xpeak[i] > nx / 2 )
//     //     xpeak[i] = xpeak[i] - nx;
//     // if ( ypeak[i] > ny / 2 )
//     //     ypeak[i] = ypeak[i] - ny;

//     // /* Return width if non-NULL */
//     // if ( width && widthMin ) {
//     //     widthTemp[0] = peakHalfWidth(array, ixpeak, iypeak, nx, ny, 1, 0) +
//     //                    peakHalfWidth(array, ixpeak, iypeak, nx, ny, -1, 0);
//     //     widthTemp[1] = peakHalfWidth(array, ixpeak, iypeak, nx, ny, 0, 1) +
//     //                    peakHalfWidth(array, ixpeak, iypeak, nx, ny, 0, -1);
//     //     widthTemp[2] = peakHalfWidth(array, ixpeak, iypeak, nx, ny, 1, 1) +
//     //                    peakHalfWidth(array, ixpeak, iypeak, nx, ny, -1, -1);
//     //     widthTemp[3] = peakHalfWidth(array, ixpeak, iypeak, nx, ny, 1, -1) +
//     //                    peakHalfWidth(array, ixpeak, iypeak, nx, ny, -1, 1);
//     //     avgSD(widthTemp, 4, &width[i], &cx, &cy);
//     //     widthMin[i] = B3DMIN(widthTemp[0], widthTemp[1]);
//     //     widthMin[i] = B3DMIN(widthMin[i], widthTemp[2]);
//     //     widthMin[i] = B3DMIN(widthMin[i], widthTemp[3]);
//     // }
//     // }
//     sApplyLimits = 0;
// }

// void rotmagstrToAmat(float theta, float smag, float str, float phi, float* a11,
//                      float* a12, float* a21, float* a22) {
//     double ator = 0.0174532925;
//     float  sinth, costh, sinphi, cosphi, sinphisq, cosphisq, f1, f2, f3;

//     costh    = (float)cos(ator * theta);
//     sinth    = (float)sin(ator * theta);
//     cosphi   = (float)cos(ator * phi);
//     sinphi   = (float)sin(ator * phi);
//     cosphisq = cosphi * cosphi;
//     sinphisq = sinphi * sinphi;
//     f1       = smag * (str * cosphisq + sinphisq);
//     f2       = smag * (str - 1.) * cosphi * sinphi;
//     f3       = smag * (str * sinphisq + cosphisq);
//     *a11     = f1 * costh - f2 * sinth;
//     *a12     = f2 * costh - f3 * sinth;
//     *a21     = f1 * sinth + f2 * costh;
//     *a22     = f2 * sinth + f3 * costh;
// }

// /*!
//  * Takes the inverse of transform [f] and returns the result in [finv], which can be the
//  * same as [f].
//  */
// void xfInvert(float* f, float* finv, int rows) {
//     float tmp[9];
//     float denom   = f[0] * f[rows + 1] - f[rows] * f[1];
//     int   idx     = 2 * rows;
//     int   idy     = 2 * rows + 1;
//     tmp[0]        = f[rows + 1] / denom;
//     tmp[rows]     = -f[rows] / denom;
//     tmp[1]        = -f[1] / denom;
//     tmp[rows + 1] = f[0] / denom;
//     tmp[idx]      = -(tmp[0] * f[idx] + tmp[rows] * f[idy]);
//     tmp[idy]      = -(tmp[1] * f[idx] + tmp[rows + 1] * f[idy]);
//     if ( rows > 2 ) {
//         tmp[2] = tmp[5] = 0.;
//         tmp[8]          = 1.;
//     }
//     for ( idx = 0; idx < 3 * rows; idx++ )
//         finv[idx] = tmp[idx];
// }

// /*!
//  * Applies transform [f] to the point [x], [y], with the center of transformation at
//  * [xcen], [ycen], and returns the result in [xp], [yp], which can be the same as
//  * [x], [y].
//  */
// void xfApply(float* f, float xcen, float ycen, float x, float y, float* xp, float* yp,
//              int rows) {
//     float xadj = x - xcen;
//     float yadj = y - ycen;
//     *xp        = f[0] * xadj + f[rows] * yadj + f[2 * rows] + xcen;
//     *yp        = f[1] * xadj + f[rows + 1] * yadj + f[2 * rows + 1] + ycen;
// }

// double sliceEdgeMean(float* array, int nxdim, int ixlo, int ixhi, int iylo,
//                      int iyhi) {
//     double dmean, sum = 0.;
//     int    ix, iy;
//     for ( ix = ixlo; ix <= ixhi; ix++ )
//         sum += array[ix + iylo * nxdim] + array[ix + iyhi * nxdim];

//     for ( iy = iylo + 1; iy < iyhi; iy++ )
//         sum += array[ixlo + iy * nxdim] + array[ixhi + iy * nxdim];

//     dmean = sum / (2 * (ixhi - ixlo + iyhi - iylo));
//     return dmean;
// }

// // /* Find the point in one direction away from the peak pixel where it falls by half */
// // static float peakHalfWidth(float* array, int ixPeak, int iyPeak, int nx, int ny, int delx,
// //                            int dely) {
// //     int   nxdim = nx + 2;
// //     float peak  = array[ixPeak + iyPeak * nxdim];
// //     int   dist, ix, iy;
// //     float scale   = (float)sqrt((double)delx * delx + dely * dely);
// //     float lastVal = peak, val;

// //     for ( dist = 1; dist < B3DMIN(nx, ny) / 4; dist++ ) {
// //         ix  = (ixPeak + dist * delx + nx) % nx;
// //         iy  = (iyPeak + dist * dely + ny) % ny;
// //         val = array[ix + iy * nxdim];
// //         if ( val < peak / 2. )
// //             return scale * (float)(dist + (lastVal - peak / 2.) / (lastVal - val) - 1.);
// //         lastVal = val;
// //     }
// //     return scale * (float)dist;
// // }
// // shift the stack
// // void readArray( ) {
// //     wxString      inFileName = "/groups/lingli/Documents/cisTEM/build/tomoalign_Intel-gpu-debug-static/src/peak_analysis/shifted_mapx.txt";
// //     std::ifstream inFile;
// //     inFile.open(inFileName.c_str( ));
// //     if ( inFile.is_open( ) ) {
// //         wxPrintf("file is open\n");
// //         float myarray[10][5760];
// //         for ( int j = 0; j < 2; j++ ) {
// //             for ( int i = 0; i < 5760; i++ ) {
// //                 inFile >> myarray[j][i];
// //                 wxPrintf("j i value %i %i %g\n", j, i, myarray[j][i]);
// //             }
// //         }
// //     }
// // }
// using namespace std;
// using namespace dlib;

// // ----------------------------------------------------------------------------------------

// typedef matrix<double, 3, 1>  input_vector;
// typedef matrix<double, 18, 1> parameter_vector;

// // ----------------------------------------------------------------------------------------

// // We will use this function to generate data.  It represents a function of 2 variables
// // and 3 parameters.   The least squares procedure will be used to infer the values of
// // the 3 parameters based on a set of input/output pairs.
// double model(
//         const input_vector&     input,
//         const parameter_vector& params) {
//     const double p0 = params(0);
//     const double p1 = params(1);
//     const double p2 = params(2);

//     const double i0 = input(0);
//     const double i1 = input(1);

//     const double temp = p0 * i0 + p1 * i1 + p2;

//     return temp * temp;
//     // return temp;
// }

// double modeltest(
//         const input_vector&     input,
//         const parameter_vector& params) {
//     const double c0 = params(0);
//     const double c1 = params(1);
//     const double c2 = params(2), c3 = params(3), c4 = params(4), c5 = params(5), c6 = params(6), c7 = params(7);
//     const double c8 = params(8), c9 = params(9), c10 = params(10), c11 = params(11), c12 = params(12), c13 = params(13);
//     const double c14 = params(14), c15 = params(15), c16 = params(16), c17 = params(17);

//     const double x = input(0);
//     const double y = input(1);
//     const double t = input(2);

//     const double temp = c0 * t + c1 * pow(t, 2) + c2 * pow(t, 3) + c3 * x * t + c4 * x * pow(t, 2) + c5 * x * pow(t, 3) + c6 * pow(x, 2) * t + c7 * pow(x, 2) * pow(t, 2) + c8 * pow(x, 2) * pow(t, 3) + c9 * y * t + c10 * y * pow(t, 2) + c11 * y * pow(t, 3) + c12 * pow(y, 2) * t + c13 * pow(y, 2) * pow(t, 2) + c14 * pow(y, 2) * pow(t, 3) + c15 * x * y * t + c16 * x * y * pow(t, 2) + c17 * x * y * pow(t, 3);

//     return temp;
// }

// double residual_test(
//         const std::pair<input_vector, double>& data,
//         const parameter_vector&                params) {
//     return modeltest(data.first, params) - data.second;
// }

// parameter_vector residual_derivative_test(
//         const std::pair<input_vector, double>& data,
//         const parameter_vector&                params) {
//     parameter_vector der;

//     const double c0 = params(0);
//     const double c1 = params(1);
//     const double c2 = params(2), c3 = params(3), c4 = params(4), c5 = params(5), c6 = params(6), c7 = params(7);
//     const double c8 = params(8), c9 = params(9), c10 = params(10), c11 = params(11), c12 = params(12), c13 = params(13);
//     const double c14 = params(14), c15 = params(15), c16 = params(16), c17 = params(17);

//     const double x = data.first(0);
//     const double y = data.first(1);
//     const double t = data.first(2);

//     const double temp = c0 * t + c1 * pow(t, 2) + c2 * pow(t, 3) + c3 * x * t + c4 * x * pow(t, 2) + c5 * x * pow(t, 3) + c6 * pow(x, 2) * t + c7 * pow(x, 2) * pow(t, 2) + c8 * pow(x, 2) * pow(t, 3) + c9 * y * t + c10 * y * pow(t, 2) + c11 * y * pow(t, 3) + c12 * pow(y, 2) * t + c13 * pow(y, 2) * pow(t, 2) + c14 * pow(y, 2) * pow(t, 3) + c15 * x * y * t + c16 * x * y * pow(t, 2) + c17 * x * y * pow(t, 3);
//     der(0)            = t;
//     der(1)            = pow(t, 2);
//     der(2)            = pow(t, 3);
//     der(3)            = x * t;
//     der(4)            = x * pow(t, 2);
//     der(5)            = x * pow(t, 3);
//     der(6)            = pow(x, 2) * t;
//     der(7)            = pow(x, 2) * pow(t, 2);
//     der(8)            = pow(x, 2) * pow(t, 3);
//     der(9)            = y * t;
//     der(10)           = y * pow(t, 2);
//     der(11)           = y * pow(t, 3);
//     der(12)           = pow(y, 2) * t;
//     der(13)           = pow(y, 2) * pow(t, 2);
//     der(14)           = pow(y, 2) * pow(t, 3);
//     der(15)           = x * y * t;
//     der(16)           = x * y * pow(t, 2);
//     der(17)           = x * y * pow(t, 3);
//     // der(0) = i0 * 2;
//     // der(1) = i1 * 2;
//     // der(2) = 1;

//     return der;
// }

// double residual(
//         const std::pair<input_vector, double>& data,
//         const parameter_vector&                params) {
//     return model(data.first, params) - data.second;
// }

// // ----------------------------------------------------------------------------------------

// // This function is the derivative of the residual() function with respect to the parameters.
// parameter_vector residual_derivative(
//         const std::pair<input_vector, double>& data,
//         const parameter_vector&                params) {
//     parameter_vector der;

//     const double p0 = params(0);
//     const double p1 = params(1);
//     const double p2 = params(2);

//     const double i0 = data.first(0);
//     const double i1 = data.first(1);

//     const double temp = p0 * i0 + p1 * i1 + p2;

//     der(0) = i0 * 2 * temp;
//     der(1) = i1 * 2 * temp;
//     der(2) = 2 * temp;
//     // der(0) = i0 * 2;
//     // der(1) = i1 * 2;
//     // der(2) = 1;

//     return der;
// }

// // this is to test the L-BFGS-B method
// // ----------------------------------------------------------------------------------------

// // In dlib, most of the general purpose solvers optimize functions that take a
// // column vector as input and return a double.  So here we make a typedef for a
// // variable length column vector of doubles.  This is the type we will use to
// // represent the input to our objective functions which we will be minimizing.
// typedef matrix<double, 0, 1> column_vector;

// // ----------------------------------------------------------------------------------------
// // Below we create a few functions.  When you get down into main() you will see that
// // we can use the optimization algorithms to find the minimums of these functions.
// // ----------------------------------------------------------------------------------------

// double rosen(const column_vector& m)
// /*
//     This function computes what is known as Rosenbrock's function.  It is
//     a function of two input variables and has a global minimum at (1,1).
//     So when we use this function to test out the optimization algorithms
//     we will see that the minimum found is indeed at the point (1,1).
// */
// {
//     const double x = m(0);
//     const double y = m(1);

//     // compute Rosenbrock's function and return the result
//     return 100.0 * pow(y - x * x, 2) + pow(1 - x, 2);
// }

// // This is a helper function used while optimizing the rosen() function.
// const column_vector rosen_derivative(const column_vector& m)
// /*!
//     ensures
//         - returns the gradient vector for the rosen function
// !*/
// {
//     const double x = m(0);
//     const double y = m(1);

//     // make us a column vector of length 2
//     column_vector res(2);

//     // now compute the gradient vector
//     res(0) = -400 * x * (y - x * x) - 2 * (1 - x); // derivative of rosen() with respect to x
//     res(1) = 200 * (y - x * x); // derivative of rosen() with respect to y
//     return res;
// }

// // This function computes the Hessian matrix for the rosen() fuction.  This is
// // the matrix of second derivatives.
// matrix<double> rosen_hessian(const column_vector& m) {
//     const double x = m(0);
//     const double y = m(1);

//     matrix<double> res(2, 2);

//     // now compute the second derivatives
//     res(0, 0) = 1200 * x * x - 400 * y + 2; // second derivative with respect to x
//     res(1, 0) = res(0, 1) = -400 * x; // derivative with respect to x and y
//     res(1, 1)             = 200; // second derivative with respect to y
//     return res;
// }

// // ----------------------------------------------------------------------------------------

// class rosen_model {
//     /*!
//         This object is a "function model" which can be used with the
//         find_min_trust_region() routine.
//     !*/

//   public:
//     typedef ::column_vector column_vector;
//     typedef matrix<double>  general_matrix;

//     double operator( )(
//             const column_vector& x) const { return rosen(x); }

//     void get_derivative_and_hessian(
//             const column_vector& x,
//             column_vector&       der,
//             general_matrix&      hess) const {
//         der  = rosen_derivative(x);
//         hess = rosen_hessian(x);
//     }
// };

// bool NikoTestApp::DoCalculation( ) try {
//     // Set the starting point to (4,8).  This is the point the optimization algorithm
//     // will start out from and it will move it closer and closer to the function's
//     // minimum point.   So generally you want to try and compute a good guess that is
//     // somewhat near the actual optimum value.
//     column_vector starting_point = {4, 8};

//     // The first example below finds the minimum of the rosen() function and uses the
//     // analytical derivative computed by rosen_derivative().  Since it is very easy to
//     // make a mistake while coding a function like rosen_derivative() it is a good idea
//     // to compare your derivative function against a numerical approximation and see if
//     // the results are similar.  If they are very different then you probably made a
//     // mistake.  So the first thing we do is compare the results at a test point:

//     std::setvbuf(stdout, NULL, _IONBF, 0);
//     std::cout << "Difference between analytic derivative and numerical approximation of derivative: "
//               << length(derivative(rosen)(starting_point) - rosen_derivative(starting_point)) << std::endl;
//     std::cout.flush( );
//     std::cout << "Find the minimum of the rosen function()" << std::endl;
//     std::cout.flush( );
//     // Now we use the find_min() function to find the minimum point.  The first argument
//     // to this routine is the search strategy we want to use.  The second argument is the
//     // stopping strategy.  Below I'm using the objective_delta_stop_strategy which just
//     // says that the search should stop when the change in the function being optimized
//     // is small enough.

//     // The other arguments to find_min() are the function to be minimized, its derivative,
//     // then the starting point, and the last is an acceptable minimum value of the rosen()
//     // function.  That is, if the algorithm finds any inputs to rosen() that gives an output
//     // value <= -1 then it will stop immediately.  Usually you supply a number smaller than
//     // the actual global minimum.  So since the smallest output of the rosen function is 0
//     // we just put -1 here which effectively causes this last argument to be disregarded.

//     find_min(bfgs_search_strategy( ), // Use BFGS search algorithm
//              objective_delta_stop_strategy(1e-7), // Stop when the change in rosen() is less than 1e-7
//              rosen, rosen_derivative, starting_point, -1);
//     // Once the function ends the starting_point vector will contain the optimum point
//     // of (1,1).
//     cout << "rosen solution:\n"
//          << starting_point << endl;

//     // Now let's try doing it again with a different starting point and the version
//     // of find_min() that doesn't require you to supply a derivative function.
//     // This version will compute a numerical approximation of the derivative since
//     // we didn't supply one to it.
//     starting_point = {-94, 5.2};
//     find_min_using_approximate_derivatives(bfgs_search_strategy( ),
//                                            objective_delta_stop_strategy(1e-7),
//                                            rosen, starting_point, -1);
//     // Again the correct minimum point is found and stored in starting_point
//     cout << "rosen solution:\n"
//          << starting_point << endl;

//     // Here we repeat the same thing as above but this time using the L-BFGS
//     // algorithm.  L-BFGS is very similar to the BFGS algorithm, however, BFGS
//     // uses O(N^2) memory where N is the size of the starting_point vector.
//     // The L-BFGS algorithm however uses only O(N) memory.  So if you have a
//     // function of a huge number of variables the L-BFGS algorithm is probably
//     // a better choice.
//     starting_point = {0.8, 1.3};
//     find_min(lbfgs_search_strategy(10), // The 10 here is basically a measure of how much memory L-BFGS will use.
//              objective_delta_stop_strategy(1e-7).be_verbose( ), // Adding be_verbose() causes a message to be
//              // printed for each iteration of optimization.
//              rosen, rosen_derivative, starting_point, -1);

//     cout << endl
//          << "rosen solution: \n"
//          << starting_point << endl;

//     starting_point = {-94, 5.2};
//     find_min_using_approximate_derivatives(lbfgs_search_strategy(10),
//                                            objective_delta_stop_strategy(1e-7),
//                                            rosen, starting_point, -1);
//     cout << "rosen solution: \n"
//          << starting_point << endl;

//     // dlib also supports solving functions subject to bounds constraints on
//     // the variables.  So for example, if you wanted to find the minimizer
//     // of the rosen function where both input variables were in the range
//     // 0.1 to 0.8 you would do it like this:
//     starting_point = {0.1, 0.1}; // Start with a valid point inside the constraint box.
//     find_min_box_constrained(lbfgs_search_strategy(10),
//                              objective_delta_stop_strategy(1e-9),
//                              rosen, rosen_derivative, starting_point, 0.1, 0.8);
//     // Here we put the same [0.1 0.8] range constraint on each variable, however, you
//     // can put different bounds on each variable by passing in column vectors of
//     // constraints for the last two arguments rather than scalars.

//     cout << endl
//          << "constrained rosen solution: \n"
//          << starting_point << endl;

//     // You can also use an approximate derivative like so:
//     starting_point = {0.1, 0.1};
//     find_min_box_constrained(bfgs_search_strategy( ),
//                              objective_delta_stop_strategy(1e-9),
//                              rosen, derivative(rosen), starting_point, 0.1, 0.8);
//     cout << endl
//          << "constrained rosen solution: \n"
//          << starting_point << endl;

//     // In many cases, it is useful if we also provide second derivative information
//     // to the optimizers.  Two examples of how we can do that are shown below.
//     starting_point = {0.8, 1.3};
//     find_min(newton_search_strategy(rosen_hessian),
//              objective_delta_stop_strategy(1e-7),
//              rosen,
//              rosen_derivative,
//              starting_point,
//              -1);
//     cout << "rosen solution: \n"
//          << starting_point << endl;

//     // We can also use find_min_trust_region(), which is also a method which uses
//     // second derivatives.  For some kinds of non-convex function it may be more
//     // reliable than using a newton_search_strategy with find_min().
//     starting_point = {0.8, 1.3};
//     find_min_trust_region(objective_delta_stop_strategy(1e-7),
//                           rosen_model( ),
//                           starting_point,
//                           10 // initial trust region radius
//     );
//     cout << "rosen solution: \n"
//          << starting_point << endl;
//     wxPrintf("rosen solution: %f, %f\n", starting_point(0), starting_point(1));
//     // Next, let's try the BOBYQA algorithm.  This is a technique specially
//     // designed to minimize a function in the absence of derivative information.
//     // Generally speaking, it is the method of choice if derivatives are not available
//     // and the function you are optimizing is smooth and has only one local optima.  As
//     // an example, consider the be_like_target function defined below:
//     column_vector target         = {3, 5, 1, 7};
//     auto          be_like_target = [&](const column_vector& x) {
//         return mean(squared(x - target));
//     };
//     starting_point = {-4, 5, 99, 3};
//     find_min_bobyqa(be_like_target,
//                     starting_point,
//                     9, // number of interpolation points
//                     uniform_matrix<double>(4, 1, -1e100), // lower bound constraint
//                     uniform_matrix<double>(4, 1, 1e100), // upper bound constraint
//                     10, // initial trust region radius
//                     1e-6, // stopping trust region radius
//                     100 // max number of objective function evaluations
//     );
//     cout << "be_like_target solution:\n"
//          << starting_point << endl;
//     wxPrintf("be_like_target solution: %f, %f, %f, %f\n", starting_point(0), starting_point(1), starting_point(2), starting_point(3));
//     // Finally, let's try the find_min_global() routine.  Like find_min_bobyqa(),
//     // this technique is specially designed to minimize a function in the absence
//     // of derivative information.  However, it is also designed to handle
//     // functions with many local optima.  Where BOBYQA would get stuck at the
//     // nearest local optima, find_min_global() won't.  find_min_global() uses a
//     // global optimization method based on a combination of non-parametric global
//     // function modeling and BOBYQA style quadratic trust region modeling to
//     // efficiently find a global minimizer.  It usually does a good job with a
//     // relatively small number of calls to the function being optimized.
//     //
//     // You also don't have to give it a starting point or set any parameters,
//     // other than defining bounds constraints.  This makes it the method of
//     // choice for derivative free optimization in the presence of multiple local
//     // optima.  Its API also allows you to define functions that take a
//     // column_vector as shown above or to explicitly use named doubles as
//     // arguments, which we do here.
//     auto complex_holder_table = [](double x0, double x1) {
//         // This function is a version of the well known Holder table test
//         // function, which is a function containing a bunch of local optima.
//         // Here we make it even more difficult by adding more local optima
//         // and also a bunch of discontinuities.

//         // add discontinuities
//         double sign = 1;
//         for ( double j = -4; j < 9; j += 0.5 ) {
//             if ( j < x0 && x0 < j + 0.5 )
//                 x0 += sign * 0.25;
//             sign *= -1;
//         }
//         // Holder table function tilted towards 10,10 and with additional
//         // high frequency terms to add more local optima.
//         return -(std::abs(sin(x0) * cos(x1) * exp(std::abs(1 - std::sqrt(x0 * x0 + x1 * x1) / pi))) - (x0 + x1) / 10 - sin(x0 * 10) * cos(x1 * 10));
//     };

//     // To optimize this difficult function all we need to do is call
//     // find_min_global()
//     // auto result = find_min_global(complex_holder_table,
//     //                               {-10, -10}, // lower bounds
//     //                               {10, 10}, // upper bounds
//     //                               std::chrono::milliseconds(500) // run this long
//     // );
//     // auto result = find_min_global(rosen,
//     //                               {-10, -10}, // lower bounds
//     //                               {10, 10}, // upper bounds
//     //                               std::chrono::milliseconds(500) // run this long
//     // );
//     // auto result = find_min_global(rosen, {0.1, 0.1}, {2, 2}, max_function_calls(100), 0);
//     auto result = find_min_global(rosen, {0.1, 0.1}, {2, 2}, std::chrono::seconds(5));
//     // auto result = find_min_global(complex_holder_table,
//     //                               {-10, -10}, // lower bounds
//     //                               {10, 10} // upper bounds
//     // );
//     // auto result = find_min_global(complex_holder_table,
//     //                               {-10, -10}, // lower bounds
//     //                               {10, 10}, // upper bounds
//     //                               500 // run this long
//     // );
//     // std::cout.precision(9);
//     // // These cout statements will show that find_min_global() found the
//     // // globally optimal solution to 9 digits of precision:
//     // std::cout << "complex holder table function solution y (should be -21.9210397): " << result.y << endl;
//     // std::cout << "complex holder table function solution x:\n"
//     //           << result.x << endl;
//     return true;

// }

// /*
//     cout.precision(9);
//     // These cout statements will show that find_min_global() found the
//     // globally optimal solution to 9 digits of precision:
//     cout << "complex holder table function solution y (should be -21.9210397): " << result.y << endl;
//     cout << "complex holder table function solution x:\n"
//          << result.x << endl;
//     return true;
// }

// */
// catch ( std::exception& e ) {

//     cout << e.what( ) << endl;
//     return true;
// }

// /* this is to test fitting the shift using least square method
// bool NikoTestApp::DoCalculation( ) {
//     // wxString patch_shifts;

//     // wxString input_path  = "/data/lingli/Lingli_20221028/grid2_process/MotCor202301/PatchMovies_gain/";
//     // wxString output_path = "/data/lingli/Lingli_20221028/draft_tmp";
//     wxString input_path  = "/data/lingli/Lingli_20221028/grid2_process/MotCor202301/TestPatch/";
//     wxString output_path = "/data/lingli/Lingli_20221028/TestPatch/";
//     int      image_no    = 75;
//     int      patch_num_x = 6;
//     int      patch_num_y = 4;
//     int      patch_no    = patch_num_x * patch_num_y;
//     float**  shiftsx     = NULL;
//     float**  shiftsy     = NULL;

//     Allocate2DFloatArray(shiftsx, image_no, patch_no);
//     Allocate2DFloatArray(shiftsy, image_no, patch_no);
//     wxString         shift_file;
//     NumericTextFile* shiftfile;
//     float            shifts[2];
//     float            patch_locations[patch_no][2];
//     for ( int patch_index = 0; patch_index < patch_no; patch_index++ ) {
//         shift_file = wxString::Format(input_path + "%02i_" + "shift.txt", patch_index);
//         shiftfile  = new NumericTextFile(shift_file, OPEN_TO_READ, 2);
//         for ( int image_index = 0; image_index < image_no; image_index++ ) {
//             shiftfile->ReadLine(shifts);
//             shiftsx[image_index][patch_index] = shifts[0];
//             shiftsy[image_index][patch_index] = shifts[1];
//             // wxPrintf("shifts: %f, %f\n", shiftsx[image_index][patch_index], shiftsy[image_index][patch_index]);
//         }
//     }
//     int image_dim_x = 5760;
//     int image_dim_y = 4092;
//     int step_size_x = myroundint(float(image_dim_x) / float(patch_num_x) / 2);
//     int step_size_y = myroundint(float(image_dim_y) / float(patch_num_y) / 2);
//     for ( int patch_y_ind = 0; patch_y_ind < patch_num_y; patch_y_ind++ ) {
//         for ( int patch_x_ind = 0; patch_x_ind < patch_num_x; patch_x_ind++ ) {
//             patch_locations[patch_num_x * patch_y_ind + patch_x_ind][0] = patch_x_ind * step_size_x * 2 + step_size_x;
//             patch_locations[patch_num_x * patch_y_ind + patch_x_ind][1] = image_dim_y - patch_y_ind * step_size_y * 2 - step_size_y;
//             wxPrintf("patch locations: %f, %f\n", patch_locations[patch_num_x * patch_y_ind + patch_x_ind][0], patch_locations[patch_num_x * patch_y_ind + patch_x_ind][1]);
//         }
//     }
//     input_vector                                 input;
//     std::vector<std::pair<input_vector, double>> data_x, data_y;
//     for ( int image_index = 0; image_index < image_no; image_index++ ) {
//         for ( int patch_index = 0; patch_index < patch_no; patch_index++ ) {
//             float time;
//             time     = image_index;
//             input(0) = patch_locations[patch_index][0];
//             input(1) = patch_locations[patch_index][1];
//             input(2) = time;
//             // double outputx = shiftsx[image_index][patch_index];
//             // double outputy = shiftsy[image_index][patch_index];
//             // the following is to set the first image has no shift
//             double outputx = shiftsx[image_index][patch_index] - shiftsx[0][patch_index];
//             double outputy = shiftsy[image_index][patch_index] - shiftsy[0][patch_index];
//             data_x.push_back(make_pair(input, outputx));
//             data_y.push_back(make_pair(input, outputy));
//         }
//     }
//     parameter_vector x, y;
//     x = 1;
//     y = 1;

//     solve_least_squares_lm(objective_delta_stop_strategy(1e-7).be_verbose( ),
//                            residual_test,
//                            residual_derivative_test,
//                            data_x,
//                            x);
//     wxPrintf("x fitted parameters: %f, %f, %f, %f, %f, %f, %f, %f, %f, %f, %f, %f, %f, %f, %f, %f, %f, %f\n", x(0), x(1), x(2), x(3), x(4), x(5), x(6), x(7), x(8), x(9), x(10), x(11), x(12), x(13), x(14), x(15), x(16), x(17));

//     solve_least_squares_lm(objective_delta_stop_strategy(1e-7).be_verbose( ),
//                            residual_test,
//                            residual_derivative_test,
//                            data_y,
//                            y);
//     wxPrintf("y fitted parameters: %f, %f, %f, %f, %f, %f, %f, %f, %f, %f, %f, %f, %f, %f, %f, %f, %f, %f\n", y(0), y(1), y(2), y(3), y(4), y(5), y(6), y(7), y(8), y(9), y(10), y(11), y(12), y(13), y(14), y(15), y(16), y(17));

//     // readin stack and apply distortion

//     // int    image_dim_x    = input_stack[0].logical_x_dimension;
//     // int    image_dim_y    = input_stack[0].logical_y_dimension;
//     int      number_of_images = image_no;
//     wxString inputstack       = input_path + "records_19_b2_gain_div_aligned_frames.mrc";
//     int      totalpixels      = image_dim_x * image_dim_y;

//     float* original_map_x = new float[totalpixels];
//     float* original_map_y = new float[totalpixels];
//     float* shifted_map_x  = new float[totalpixels];

//     float*  shifted_map_y = new float[totalpixels];
//     Image*  input_stack   = new Image[image_no];
//     Image*  distorted_stack;
//     MRCFile input_file(inputstack.ToStdString( ), false);
//     distorted_stack = new Image[image_no];

//     for ( int image_counter = 0; image_counter < image_no; image_counter++ ) {
//         input_stack[image_counter].ReadSlice(&input_file, image_counter + 1);
//     }

//     // initialize the pixel coordinates
//     for ( int i = 0; i < image_dim_y; i++ ) {
//         for ( int j = 0; j < image_dim_x; j++ ) {
//             original_map_x[i * image_dim_x + j] = j;
//             original_map_y[i * image_dim_x + j] = i;
//         }
//     }
//     // input_vector     input;
//     float            time;
//     parameter_vector params_x   = x;
//     parameter_vector params_y   = y;
//     wxString         outputpath = "/data/lingli/Lingli_20221028/grid2_process/MotCor202301/TestPatch/";
//     for ( int image_counter = 0; image_counter < number_of_images; image_counter++ ) {
//         // for ( int image_counter = 10; image_counter < 11; image_counter++ ) {
//         time     = image_counter;
//         input(2) = time;
//         for ( int pix = 0; pix < totalpixels; pix++ ) {
//             input(0)           = original_map_x[pix];
//             input(1)           = original_map_y[pix];
//             shifted_map_x[pix] = modeltest(input, params_x) + original_map_x[pix];
//             shifted_map_y[pix] = modeltest(input, params_y) + original_map_y[pix];
//             if ( pix < 20 ) {
//                 wxPrintf("%f %f %f %f\n", shifted_map_x[pix], shifted_map_y[pix], original_map_x[pix], original_map_y[pix]);
//             }
//         }
//         // distorted_stack[image_counter].Allocate(input_stack[image_counter].logical_x_dimension, input_stack[image_counter].logical_y_dimension, 1, false);
//         distorted_stack[image_counter].Allocate(input_stack[image_counter].logical_x_dimension, input_stack[image_counter].logical_y_dimension, 1, true);
//         distorted_stack[image_counter].SetToConstant(input_stack[image_counter].ReturnAverageOfRealValuesOnEdges( ));
//         // input_stack[image_counter].BackwardFFT( );
//         // input_stack[image_counter].ZeroCentralPixel( );
//         input_stack[image_counter].Distortion(&distorted_stack[image_counter], shifted_map_x, shifted_map_y);
//     }

//     Image sum_image_fixed, distorted_fixed;
//     // sum_image_fixed.Allocate(input_stack[0].logical_x_dimension, input_stack[0].logical_y_dimension, false);
//     // distorted_fixed.Allocate(input_stack[0].logical_x_dimension, input_stack[0].logical_y_dimension, false);
//     sum_image_fixed.Allocate(input_stack[0].logical_x_dimension, input_stack[0].logical_y_dimension, true);
//     distorted_fixed.Allocate(input_stack[0].logical_x_dimension, input_stack[0].logical_y_dimension, true);
//     sum_image_fixed.SetToConstant(0.0);
//     distorted_fixed.SetToConstant(0.0);
//     for ( int image_counter = 0; image_counter < number_of_images; image_counter++ ) {
//         sum_image_fixed.AddImage(&input_stack[image_counter]);
//         distorted_fixed.AddImage(&distorted_stack[image_counter]);
//         input_stack[image_counter].QuickAndDirtyWriteSlice(outputpath.ToStdString( ) + "original_frames.mrc", image_counter + 1);
//         distorted_stack[image_counter].QuickAndDirtyWriteSlice(outputpath.ToStdString( ) + "fixed_frames.mrc", image_counter + 1);
//     }
//     // sum_image_fixed.BackwardFFT( );
//     // distorted_fixed.BackwardFFT( );
//     sum_image_fixed.WriteSlicesAndFillHeader(outputpath.ToStdString( ) + "original.mrc", 1);
//     distorted_fixed.WriteSlicesAndFillHeader(outputpath.ToStdString( ) + "fixed.mrc", 1);

//     delete shiftfile;
//     return true;
// }
// */

// /*

// bool NikoTestApp::DoCalculation( ) {
//     // ----------------------------------------------------------------------------------------

//     // int main( ) {
//     // try {
//     // randomly pick a set of parameters to use in this example
//     const parameter_vector params = 100 * randm(3, 1);
//     // shift_filex = wxString::Format(input_path + "%02i_" + shift_file_x + ".txt", shift_file_index);
//     // shift_filey = wxString::Format(input_path + "%02i_" + shift_file_y + ".txt", shift_file_index);
//     // wxPrintf("shiftfiles are %s, %s, \n", shift_filex, shift_filey);
//     // xFile.open(shift_filex.c_str( ));
//     // yFile.open(shift_filey.c_str( ));

//     // if ( xFile.is_open( ) && yFile.is_open( ) ) {
//     //     wxPrintf("files are open\n");
//     //     // float myarray[10][5760];
//     //     for ( int pix = 0; pix < totalpixels; pix++ ) {
//     //         xFile >> shifted_mapx[pix];
//     //         yFile >> shifted_mapy[pix];
//     //     }
//     // }
//     // wxPrintf("first shift x ,y %f, %f \n", shifted_mapx[0], shifted_mapy[0]);
//     // for ( int i = 0; i < image_y_dim; i++ ) {
//     //     for ( int j = 0; j < image_x_dim; j++ ) {
//     //         shifted_mapx[i * image_x_dim + j] += j;
//     //         shifted_mapy[i * image_x_dim + j] += i;
//     //     }
//     // }
//     // wxPrintf("adjusted first pix position x ,y %f, %f \n", shifted_mapx[0], shifted_mapy[0]);
//     // wxPrintf("shifting files are loaded \n");
//     // xFile.close( );
//     // yFile.close( );
//     wxPrintf("test 1\n");
//     wxPrintf("parames %f\n", trans(params)(0));
//     std::cout << "params: " << trans(params) << endl;
//     std::cout.flush( ); // explicitly flush here
//     // Now let's generate a bunch of input/output pairs according to our model.
//     std::vector<std::pair<input_vector, double>> data_samples;
//     input_vector                                 input;
//     // for ( int i = 0; i < 10; ++i ) {
//     //     input               = 100 * randm(2, 1);
//     //     const double output = model(input, params);

//     //     // save the pair
//     //     data_samples.push_back(make_pair(input, output));
//     // }

//     for ( int i = 1; i < 5; i++ ) {
//         // wxPrintf("i is %i \n", i);
//         // input               = (i, i + 1);
//         // input(0)            = i / 100;
//         // input(1)            = (i + 1) / 100;
//         // int j = i + 1;
//         for ( int j = 1; j < 5; j++ ) {

//             // if ( i % 2 == 0 ) {
//             //     input = (float(i) / 10.0),
//             //     (float(j) / 10.0); // the decimal interval cannot be evenly distribution. other wise it fail. so I used /10.0 and /100.0
//             // }
//             // else {
//             //     input = (float(i) / 100.0),
//             //     (float(j) / 100.0);
//             // }
//             input = (float(i) / 10.0),
//             (float(j) / 10.0);

//             // input(0) = 10 * randm(1, 1);
//             // input(1) = 10 * randm(1, 1);
//             wxPrintf("test %f\n", 10 * randm(1, 1)(0));
//             const double output = model(input, params);
//             // }
//             data_samples.push_back(make_pair(input, output));
//         }
//     }
//     // wxPrintf("input 1 %f\n", input(1)[1]);
//     wxPrintf("input 0 %f\n", input(0));
//     wxPrintf("data_sample 3 %f\n", data_samples[0].first(0));
//     wxPrintf("data_sample 3 %f\n", data_samples[0].first(1));
//     wxPrintf("data_sample 3 %f\n", data_samples[1].first(0));
//     wxPrintf("data_sample 3 %f\n", data_samples[1].first(1));
//     wxPrintf("data_sample 3 %f\n", data_samples[1].second);
//     // wxPrintf("data_sample 3 %f\n", data_samples[0].first(2));
//     // wxPrintf("data_sample 5 %f", data_samples[5]);
//     // Before we do anything, let's make sure that our derivative function defined above matches
//     // the approximate derivative computed using central differences (via derivative()).
//     // If this value is big then it means we probably typed the derivative function incorrectly.
//     cout << "derivative error: " << length(residual_derivative(data_samples[0], params) - derivative(residual)(data_samples[0], params)) << endl;
//     wxPrintf("derivative error: %f,\n", length(residual_derivative(data_samples[0], params) - derivative(residual)(data_samples[0], params)));
//     // Now let's use the solve_least_squares_lm() routine to figure out what the
//     // parameters are based on just the data_samples.
//     parameter_vector x;
//     x = 1;
//     // x = 100;
//     // x(0) = 15;
//     // x(1) = 15;
//     // x(2) = 70;

//     cout << "Use Levenberg-Marquardt" << endl;

//     // Use the Levenberg-Marquardt method to determine the parameters which
//     // minimize the sum of all squared residuals.
//     // solve_least_squares_lm(objective_delta_stop_strategy(1e-7).be_verbose( ),
//     //                        residual,
//     //                        residual_derivative,
//     //                        data_samples,
//     //                        x);
//     solve_least_squares_lm(gradient_norm_stop_strategy(1e-7).be_verbose( ),
//                            residual,
//                            residual_derivative,
//                            data_samples,
//                            x);

//     // Now x contains the solution.  If everything worked it will be equal to params.
//     cout << "inferred parameters: " << trans(x) << endl;
//     cout << "solution error:      " << length(x - params) << endl;
//     cout << endl;
//     wxPrintf("stopstrategy results %f, %f, %f, \n", x(0), x(1), x(2));
//     wxPrintf("solution error %f\n", length(x - params));

//     solve_least_squares_lm(objective_delta_stop_strategy(10).be_verbose( ),
//                            residual,
//                            residual_derivative,
//                            data_samples,
//                            x);

//     // Now x contains the solution.  If everything worked it will be equal to params.
//     cout << "inferred parameters: " << trans(x) << endl;
//     cout << "solution error:      " << length(x - params) << endl;
//     cout << endl;
//     wxPrintf("1 results %f, %f, %f, \n", x(0), x(1), x(2));
//     wxPrintf("solution error %f\n", length(x - params));
//     // x = 1;
//     x(0) = 15;
//     x(1) = 15;
//     x(2) = 55;
//     cout << "Use Levenberg-Marquardt, approximate derivatives" << endl;
//     // If we didn't create the residual_derivative function then we could
//     // have used this method which numerically approximates the derivatives for you.
//     solve_least_squares_lm(objective_delta_stop_strategy(1e-7).be_verbose( ),
//                            residual,
//                            derivative(residual),
//                            data_samples,
//                            x);

//     // Now x contains the solution.  If everything worked it will be equal to params.
//     cout << "inferred parameters: " << trans(x) << endl;
//     cout << "solution error:      " << length(x - params) << endl;
//     cout << endl;
//     wxPrintf("2 results %f, %f, %f \n", x(0), x(1), x(2));
//     wxPrintf("solution error %f\n", length(x - params));
//     // x = 1;
//     x(0) = 15;
//     x(1) = 15;
//     x(2) = 55;
//     cout << "Use Levenberg-Marquardt/quasi-newton hybrid" << endl;
//     // This version of the solver uses a method which is appropriate for problems
//     // where the residuals don't go to zero at the solution.  So in these cases
//     // it may provide a better answer.
//     solve_least_squares(objective_delta_stop_strategy(1e-3).be_verbose( ),
//                         residual,
//                         residual_derivative,
//                         data_samples,
//                         x, 20.0);

//     // Now x contains the solution.  If everything worked it will be equal to params.
//     cout << "inferred parameters: " << trans(x) << endl;
//     cout << "solution error:      " << length(x - params) << endl;
//     wxPrintf("3 results %f, %f, %f, \n", x(0), x(1), x(2));
//     wxPrintf("solution error %f\n", length(x - params));
//     // } catch ( std::exception& e ) {
//     //     cout << e.what( ) << endl;
//     // }

//     return true;
//     // }
// }
// */
// /*

// void NikoTestApp::DoInteractiveUserInput( ) {
//     UserInput* my_input = new UserInput("TrimStack", 1.0);

//     wxString input_image          = my_input->GetFilenameFromUser("Input image file name", "Name of input image file *.mrc", "input.mrc", true);
//     wxString peak_filename        = my_input->GetFilenameFromUser("Peak filename", "The file containing peak for each patch, *.txt", "peak_01.txt", true);
//     wxString coordinates_filename = my_input->GetFilenameFromUser("Coordinates (PLT) filename", "The input particle coordinates, in Imagic-style PLT forlmat", "coos.plt", true);
//     // int      output_stack_box_size = my_input->GetIntFromUser("Box size for output candidate particle images (pixels)", "In pixels. Give 0 to skip writing particle images to disk.", "256", 0);
//     // float    rotation_angle        = my_input->GetFloatFromUser("rotation angle of the tomography", "phi in degrees, 0.0 for none rotation", "0.0");
//     // wxString shifts_filename       = my_input->GetFilenameFromUser("Shifts filename", "The shifts, *.txt", "shifts.txt", true);
//     delete my_input;

//     my_current_job.Reset(3);
//     my_current_job.ManualSetArguments("ttt", input_image.ToUTF8( ).data( ), peak_filename.ToUTF8( ).data( ), coordinates_filename.ToUTF8( ).data( ));
// }

// bool NikoTestApp::DoCalculation( ) {
//     wxString input_image = my_current_job.arguments[0].ReturnStringArgument( );
//     wxString peakfile    = my_current_job.arguments[1].ReturnStringArgument( );
//     wxString patchfile   = my_current_job.arguments[2].ReturnStringArgument( );
//     MRCFile  input_test(input_image.ToStdString( ), false);

//     wxString    outputpath    = "/groups/lingli/Documents/cisTEM/build/tomoalign_Intel-gpu-debug-static/src/tilt04_shift_analysis/";
//     std::string outputpathstd = "/groups/lingli/Documents/cisTEM/build/tomoalign_Intel-gpu-debug-static/src/tilt04_shift_analysis/";
//     // MRCFile input_test("/groups/lingli/Documents/cisTEM/build/tomoalign_Intel-gpu-debug-static/src/Test-ximina-20220928/outputstack.mrc", false);
//     // MRCFile input_test("/groups/lingli/Documents/cisTEM/build/tomoalign_Intel-gpu-debug-static/src/peak_analysis/sample_img.mrc", false);
//     // MRCFile input_test("/groups/lingli/Documents/cisTEM/build/tomoalign_Intel-gpu-debug-static/src/unstacked_Ximena_raw/image_000.mrc", false);
//     // wxString coordinates_filename = my_input->GetFilenameFromUser("Coordinates (PLT) filename", "The input particle coordinates, in Imagic-style PLT forlmat", "coos.plt", true);
//     // wxString coordinates_filename = my_current_job.arguments[2].ReturnStringArgument( );
//     NumericTextFile *patch_positions, *peak_positions;
//     // wxString         patchfile = "/groups/lingli/Documents/cisTEM/build/tomoalign_Intel-gpu-debug-static/src/patchlocations.plt";
//     // wxString         peakfile  = "/groups/lingli/Documents/cisTEM/build/tomoalign_Intel-gpu-debug-static/src/peak_positions.txt";
//     // wxString patchfile = "/groups/lingli/Documents/cisTEM/build/tomoalign_Intel-gpu-debug-static/src/coord_TS17.plt";
//     // wxString peakfile  = "/groups/lingli/Documents/cisTEM/build/tomoalign_Intel-gpu-debug-static/src/peak_01.txt";

//     patch_positions = new NumericTextFile(patchfile, OPEN_TO_READ, 2);
//     peak_positions  = new NumericTextFile(peakfile, OPEN_TO_READ, 2);

//     // NumericTextFile *shift_filex, *shift_filey;
//     // NumericTextFile peak_position;
//     // readArray( );
//     wxPrintf("start\n");

//     Image sample_img;
//     Image interp_img, interp_img_tmp;

//     int image_x_dim, image_y_dim;
//     image_x_dim = input_test.ReturnXSize( );
//     image_y_dim = input_test.ReturnYSize( );
//     int patch_no;
//     patch_no            = patch_positions->number_of_lines;
//     int     patch_x_num = 6, patch_y_num = 4;
//     int     bin      = 1;
//     float** peaks_x  = NULL;
//     float** peaks_y  = NULL;
//     float** patchs_x = NULL;
//     float** patchs_y = NULL;

//     Allocate2DFloatArray(peaks_x, patch_y_num, patch_x_num);
//     Allocate2DFloatArray(peaks_y, patch_y_num, patch_x_num);
//     Allocate2DFloatArray(patchs_x, patch_y_num, patch_x_num);
//     Allocate2DFloatArray(patchs_y, patch_y_num, patch_x_num);
//     for ( int i = 0; i < patch_y_num; i++ ) {
//         float tmppeak[2];
//         float tmppatch[2];
//         for ( int j = 0; j < patch_x_num; j++ ) {
//             patch_positions->ReadLine(tmppatch);
//             peak_positions->ReadLine(tmppeak);
//             patchs_x[i][j] = tmppatch[0] + image_x_dim / 2;
//             patchs_y[i][j] = tmppatch[1] + image_y_dim / 2;
//             peaks_x[i][j]  = tmppeak[0] * bin + patchs_x[i][j];
//             peaks_y[i][j]  = tmppeak[1] * bin + patchs_y[i][j];

//             wxPrintf("patchpeaks x y %i, %i, %g, %g, %g, %g\n", i, j, peaks_x[i][j], peaks_y[i][j], patchs_x[i][j], patchs_y[i][j]);
//         }
//     }

//     float** shifted_map_x;
//     float** shifted_map_y;
//     float** original_map_x;
//     float** original_map_y;
//     Allocate2DFloatArray(shifted_map_x, image_y_dim, image_x_dim);
//     Allocate2DFloatArray(shifted_map_y, image_y_dim, image_x_dim);
//     Allocate2DFloatArray(original_map_x, image_y_dim, image_x_dim);
//     Allocate2DFloatArray(original_map_y, image_y_dim, image_x_dim);

//     // initialize the pixel coordinates
//     for ( int i = 0; i < image_y_dim; i++ ) {
//         for ( int j = 0; j < image_x_dim; j++ ) {
//             shifted_map_x[i][j]  = j;
//             original_map_x[i][j] = j;
//             shifted_map_y[i][j]  = i;
//             original_map_y[i][j] = i;
//         }
//     }

//     // calculate shifte amount along x
//     for ( int i = 0; i < patch_y_num - 1; i++ ) {
//         for ( int j = 0; j < patch_x_num - 1; j++ ) {
//             int   interval_x_no     = int(patchs_x[i][j + 1] - patchs_x[i][j]);
//             int   interval_y_no     = -int(patchs_y[i + 1][j] - patchs_y[i][j]);
//             float SHX_interval_0_1  = (peaks_x[i][j + 1] - peaks_x[i][j]) / interval_x_no;
//             float SHX_interval_2_3  = (peaks_x[i + 1][j + 1] - peaks_x[i + 1][j]) / interval_x_no;
//             float SHX_interval_diff = (SHX_interval_0_1 - SHX_interval_2_3) / interval_y_no;

//             float SHY_interval_2_0  = -(peaks_y[i + 1][j] - peaks_y[i][j]) / interval_y_no;
//             float SHY_interval_3_1  = -(peaks_y[i + 1][j + 1] - peaks_y[i][j + 1]) / interval_y_no;
//             float SHY_interval_diff = (SHY_interval_3_1 - SHY_interval_2_0) / interval_x_no;

//             float ref_SHX_interval = SHX_interval_2_3;
//             float ref_SHX_location = peaks_x[i + 1][j];
//             float ref_SHY_interval = SHY_interval_2_0;
//             float ref_SHY_location = peaks_y[i + 1][j];

//             float ref_SH_x = peaks_x[i + 1][j];
//             float ref_SH_y = peaks_y[i + 1][j];

//             int   ystart       = int(patchs_y[i + 1][j]);
//             int   yend         = int(patchs_y[i][j]);
//             int   xstart       = int(patchs_x[i + 1][j]);
//             int   xend         = int(patchs_x[i + 1][j + 1]);
//             float slop_along_y = (peaks_x[i][j] - peaks_x[i + 1][j]) / (yend - ystart);
//             float slop_along_x = (peaks_y[i + 1][j + 1] - peaks_y[i + 1][j]) / (xend - xstart);

//             int yindex_start = ystart;
//             int yindex_end   = yend;
//             int xindex_start = xstart;
//             int xindex_end   = xend;
//             if ( i == 0 )
//                 yindex_end = image_y_dim;
//             if ( i == patch_y_num - 2 )
//                 yindex_start = 0;
//             if ( j == 0 )
//                 xindex_start = 0;
//             if ( j == patch_x_num - 2 )
//                 xindex_end = image_x_dim;

//             for ( int yindex = yindex_start; yindex < yindex_end; yindex++ ) {
//                 float current_interval_x      = ref_SHX_interval + SHX_interval_diff * (yindex - ystart);
//                 float current_start_locationx = ref_SH_x + slop_along_y * (yindex - ystart);
//                 for ( int xindex = xindex_start; xindex < xindex_end; xindex++ ) {
//                     shifted_map_x[yindex][xindex] = current_start_locationx + (xindex - xstart) * current_interval_x;
//                     shifted_map_y[yindex][xindex] = yindex;
//                     float current_interval_y      = ref_SHY_interval + SHY_interval_diff * (xindex - xstart);
//                     float current_start_locationy = ref_SH_y + slop_along_x * (xindex - xstart);
//                     shifted_map_y[yindex][xindex] = current_start_locationy + (yindex - ystart) * current_interval_y;
//                 }
//             }
//         }
//     }

//     wxString      shifted_mapx_file = outputpath + "shifted_x.txt";
//     wxString      shifted_mapy_file = outputpath + "shifted_y.txt";
//     std::ofstream xoFile, yoFile;

//     // wxPrintf("1\n");
//     // for ( int i = 0; i < 10; i++ ) {
//     //     for ( int j = 0; j < 10; j++ ) {
//     //         shifted_map_x[i][j] = j;
//     //         shifted_map_y[i][j] = i;
//     //     }
//     // }

//     xoFile.open(shifted_mapx_file.c_str( ));
//     yoFile.open(shifted_mapy_file.c_str( ));
//     if ( xoFile.is_open( ) && yoFile.is_open( ) ) {
//         wxPrintf("files are open\n");
//         // float myarray[10][5760];
//         for ( int i = 0; i < image_y_dim; i++ ) {
//             for ( int j = 0; j < image_x_dim; j++ ) {
//                 xoFile << shifted_map_x[i][j] << '\t';
//                 yoFile << shifted_map_y[i][j] << '\t';
//             }
//             xoFile << '\n';
//             yoFile << '\n';
//         }
//     }
//     xoFile.close( );
//     yoFile.close( );

//     wxPrintf("1\n");
//     sample_img.ReadSlice(&input_test, 1);
//     // image_no = input_test.ReturnNumberOfSlices( );
//     // image_x_dim = sample_img.logical_x_dimension;
//     // image_y_dim = sample_img.logical_y_dimension;
//     // image_x_dim = input_test.ReturnXSize( );
//     // image_y_dim = input_test.ReturnYSize( );

//     wxPrintf("image dim: %i, %i\n", image_x_dim, image_y_dim);
//     interp_img.Allocate(image_x_dim, image_y_dim, true);
//     interp_img_tmp.Allocate(image_x_dim, image_y_dim, true);
//     interp_img.SetToConstant(sample_img.ReturnAverageOfRealValuesOnEdges( ));
//     interp_img_tmp.SetToConstant(sample_img.ReturnAverageOfRealValuesOnEdges( ));
//     wxPrintf("2\n");
//     // float* shifted_map = new float[image_y_dim][4092][2];
//     wxPrintf("image dim: %i, %i\n", image_x_dim, image_y_dim);
//     int totalpixels = image_x_dim * image_y_dim;
//     wxPrintf("3 total pixels %i\n", totalpixels);
//     // float shifted_mapx[totalpixels], shifted_mapy[totalpixels];
//     float* shifted_mapx      = new float[totalpixels];
//     float* shifted_mapy      = new float[totalpixels];
//     float* interpolated_mapx = new float[totalpixels];
//     float* interpolated_mapy = new float[totalpixels];

//     wxPrintf("3\n");
//     wxPrintf("start loading shifted text\n");

//     // load array from file
//     // wxString      shift_filex = "/groups/lingli/Documents/cisTEM/build/tomoalign_Intel-gpu-debug-static/src/peak_analysis/shifted_mapx.txt";
//     // wxString      shift_filey = "/groups/lingli/Documents/cisTEM/build/tomoalign_Intel-gpu-debug-static/src/peak_analysis/shifted_mapy.txt";
//     wxString      shift_filex = outputpath + "shifted_x.txt";
//     wxString      shift_filey = outputpath + "shifted_y.txt";
//     std::ifstream xFile, yFile;

//     wxPrintf("1\n");

//     xFile.open(shift_filex.c_str( ));
//     yFile.open(shift_filey.c_str( ));

//     if ( xFile.is_open( ) && yFile.is_open( ) ) {
//         wxPrintf("files are open\n");
//         // float myarray[10][5760];
//         for ( int pix = 0; pix < totalpixels; pix++ ) {
//             xFile >> shifted_mapx[pix];
//             yFile >> shifted_mapy[pix];
//         }
//     }
//     wxPrintf("shifting files are loaded \n");
//     xFile.close( );
//     yFile.close( );
//     // load array from file end

//     // int len = sizeof(shifted_mapx) / sizeof(shifted_mapx[0]);
//     int len = *(&shifted_mapx + 1) - shifted_mapx;
//     // std::cout << "the size" << std::sizeof(shifted_mapx[0]);
//     // wxPrintf("len %i %i %i\n", sizeof(shifted_mapx), sizeof(shifted_mapx[0]), len);
//     wxPrintf("len %i\n", len);
//     sample_img.Distortion(&interp_img, shifted_mapx, shifted_mapy);
//     interp_img.WriteSlicesAndFillHeader(outputpathstd + "interp.mrc", 1);

//     // delete &sample_img;
//     // delete &interp_img;
//     delete[] shifted_mapx;
//     delete[] shifted_mapy;
//     Deallocate2DFloatArray(shifted_map_x, image_y_dim);
//     Deallocate2DFloatArray(shifted_map_y, image_y_dim);
//     Deallocate2DFloatArray(original_map_x, image_y_dim);
//     Deallocate2DFloatArray(original_map_y, image_y_dim);

//     return true;
// }

// */

// // //-------------------------------------------test the interpolation---------------------------------------
// // bool NikoTestApp::DoCalculation( ) {
// //     // MRCFile input_test("/groups/lingli/Documents/cisTEM/build/tomoalign_Intel-gpu-debug-static/src/Test-ximina-20220928/outputstack.mrc", false);
// //     MRCFile input_test("/groups/lingli/Documents/cisTEM/build/tomoalign_Intel-gpu-debug-static/src/peak_analysis/sample_img.mrc", false);
// //     // NumericTextFile shift_file("/groups/lingli/Documents/cisTEM/build/tomoalign_Intel-gpu-debug-static/src/peak_analysis/newshifts.txt");
// //     // NumericTextFile *shift_filex, *shift_filey;
// //     // readArray( );
// //     wxPrintf("start\n");
// //     // shift_filex = new NumericTextFile("/groups/lingli/Documents/cisTEM/build/tomoalign_Intel-gpu-debug-static/src/peak_analysis/shifted_mapx.txt", OPEN_TO_READ, 5760);
// //     // wxPrintf("1\n");
// //     // shift_filey = new NumericTextFile("/groups/lingli/Documents/cisTEM/build/tomoalign_Intel-gpu-debug-static/src/peak_analysis/shifted_mapy.txt", OPEN_TO_READ, 5760);

// //     Image sample_img;
// //     Image interp_img, interp_img_tmp;

// //     int image_x_dim, image_y_dim;
// //     int image_no;
// //     wxPrintf("1\n");
// //     sample_img.ReadSlice(&input_test, 1);
// //     image_no = input_test.ReturnNumberOfSlices( );
// //     // image_x_dim = sample_img.logical_x_dimension;
// //     // image_y_dim = sample_img.logical_y_dimension;
// //     image_x_dim = input_test.ReturnXSize( );
// //     image_y_dim = input_test.ReturnYSize( );

// //     wxPrintf("image dim: %i, %i\n", image_x_dim, image_y_dim);
// //     interp_img.Allocate(image_x_dim, image_y_dim, true);
// //     interp_img_tmp.Allocate(image_x_dim, image_y_dim, true);
// //     interp_img.SetToConstant(sample_img.ReturnAverageOfRealValuesOnEdges( ));
// //     interp_img_tmp.SetToConstant(sample_img.ReturnAverageOfRealValuesOnEdges( ));
// //     wxPrintf("2\n");
// //     // float* shifted_map = new float[image_y_dim][4092][2];
// //     wxPrintf("image dim: %i, %i\n", image_x_dim, image_y_dim);
// //     int totalpixels = image_x_dim * image_y_dim;
// //     wxPrintf("3 total pixels %i\n", totalpixels);
// //     // float shifted_mapx[totalpixels], shifted_mapy[totalpixels];
// //     float* shifted_mapx      = new float[totalpixels];
// //     float* shifted_mapy      = new float[totalpixels];
// //     float* interpolated_mapx = new float[totalpixels];
// //     float* interpolated_mapy = new float[totalpixels];

// //     wxPrintf("3\n");
// //     wxPrintf("start loading shifted text\n");

// //     // load array
// //     wxString      shift_filex = "/groups/lingli/Documents/cisTEM/build/tomoalign_Intel-gpu-debug-static/src/peak_analysis/shifted_mapx.txt";
// //     wxString      shift_filey = "/groups/lingli/Documents/cisTEM/build/tomoalign_Intel-gpu-debug-static/src/peak_analysis/shifted_mapy.txt";
// //     std::ifstream xFile, yFile;

// //     wxPrintf("1\n");

// //     xFile.open(shift_filex.c_str( ));
// //     yFile.open(shift_filey.c_str( ));

// //     if ( xFile.is_open( ) && yFile.is_open( ) ) {
// //         wxPrintf("files are open\n");
// //         // float myarray[10][5760];
// //         for ( int pix = 0; pix < totalpixels; pix++ ) {
// //             xFile >> shifted_mapx[pix];
// //             yFile >> shifted_mapy[pix];
// //         }
// //     }
// //     wxPrintf("shifting files are loaded \n");
// //     xFile.close( );
// //     yFile.close( );
// //     // int len = sizeof(shifted_mapx) / sizeof(shifted_mapx[0]);
// //     int len = *(&shifted_mapx + 1) - shifted_mapx;
// //     // std::cout << "the size" << std::sizeof(shifted_mapx[0]);
// //     // wxPrintf("len %i %i %i\n", sizeof(shifted_mapx), sizeof(shifted_mapx[0]), len);
// //     wxPrintf("len %i\n", len);
// //     sample_img.Distortion(&interp_img, shifted_mapx, shifted_mapy);
// //     interp_img.WriteSlicesAndFillHeader("interp.mrc", 1);

// //     delete[] shifted_mapx;
// //     delete[] shifted_mapy;

// //     return true;
// // }

// // //------------------------------------------- end test the interpolation---------------------------------------

// // // test the rotation operation
// // bool NikoTestApp::DoCalculation( ) {
// //     MRCFile         input_test("/groups/lingli/Documents/cisTEM/build/tomoalign_Intel-gpu-debug-static/src/input_bf_stretch31_0.030.mrc", false);
// //     Image           Test_image;
// //     Image           Small_image;
// //     Image           Rotated_image;
// //     AnglesAndShifts AandS;
// //     Small_image.Allocate(200, 100, true);

// //     // Test_image.SetToConstant(1.0);
// //     Test_image.ReadSlice(&input_test, 1);
// //     Test_image.ClipInto(&Small_image, Test_image.ReturnAverageOfRealValues( ));
// //     Small_image.ForwardFFT( );
// //     Small_image.GaussianLowPassRadiusFilter(0.2, 0.01);
// //     Small_image.BackwardFFT( );
// //     Small_image.WriteSlicesAndFillHeader("test.mrc", 1);
// //     // AandS.Init(0, 0, 86.3, 10, 20);
// //     // wxPrintf("phi: %g\n", AandS.ReturnPhiAngle( ));
// //     // Rotated_image.Allocate(200, 200, true);
// //     // Small_image.Rotate2D(Rotated_image, AandS);
// //     // Rotated_image.WriteSlicesAndFillHeader("rotated.mrc", 1);
// //     Small_image.Rotate2DInPlace(86.3);
// //     Small_image.WriteSlicesAndFillHeader("rotated.mrc", 1);
// //     Small_image.RealSpaceIntegerShift(10, 50);
// //     Small_image.WriteSlicesAndFillHeader("rotatedshifted.mrc", 1);
// //     return true;
// // }

// // bool NikoTestApp::DoCalculation( ) {
// //     wxPrintf("hello1\n");

// //     Image peak_image, input_image;
// //     // MRCFile input_peak("/groups/lingli/Documents/cisTEM/build/tomoalign_Intel-gpu-debug-static/src/Test/peak102_0.050_0.030.mrc", false);
// //     // MRCFile input_peak("/groups/lingli/Downloads/TS17/test_coarsealgin/49peak.mrc", false);
// //     // MRCFile          input_stack(input_imgstack.ToStdString( ), false)
// //     // MRCFile input_peak("/groups/lingli/Downloads/TS17/test_coarsealgin/peaks.mrc", false);
// //     MRCFile          input_peak("/groups/lingli/Downloads/TS17/test_coarsealign_1/image_peak.mrc", false);
// //     NumericTextFile *tilt_angle_file, *peak_points, *shift_file, *peak_points_raw;
// //     // tilt_angle_file = new NumericTextFile('angle_filename/groups/lingli/Downloads/TS17/TS17.rawtlt', OPEN_TO_READ, 1);
// //     // peak_points     = new NumericTextFile(outputpath + "peakpoints_newcurved.txt", OPEN_TO_WRITE, 4);
// //     // peak_points_raw = new NumericTextFile(outputpath + "peakpoints_pk_img.txt", OPEN_TO_WRITE, 4);
// //     // shift_file      = new NumericTextFile(outputpath + "shifts_newcurved.txt", OPEN_TO_WRITE, 3);

// //     int   image_no = tilt_angle_file->number_of_lines;
// //     float tilts[image_no];
// //     float stretch[image_no];

// //     float shifts[image_no][2];
// //     float peaks[image_no][2];
// //     for ( int i = 0; i < image_no; i++ ) {
// //         tilt_angle_file->ReadLine(&tilts[i]);
// //         // wxPrintf("angle %i ; % g\n", i, tilts[i]);
// //     }
// //     for ( int i = 0; i < image_no; i++ )
// //         tilts[i] = (tilts[i]) / 180.0 * PI;

// //     // wxString angle_filename = my_current_job.arguments[1].ReturnStringArgument( );
// //     // MRCFile ref_file("/groups/lingli/Documents/cisTEM/build/tomoalign_Intel-gpu-debug-static/src/Output/ref34_0.030.mrc", false);
// //     // MRCFile input_file("/groups/lingli/Documents/cisTEM/build/tomoalign_Intel-gpu-debug-static/src/Output/input35_0.030_R0.000.mrc", false);

// //     //     MRCFile input_stack(input_imgstack.ToStdString( ), false);
// //     // int X_maskcenter, Y_maskcenter;
// //     // MRCFile input_test("/groups/lingli/Documents/cisTEM/build/tomoalign_Intel-gpu-debug-static/src/Output/input_bf_stretch35_0.030.mrc", false);

// //     int X_dim = input_peak.ReturnXSize( );
// //     int Y_dim = input_peak.ReturnYSize( );

// //     // float input_array[X_dim][Y_dim];
// //     float xpeak[10];
// //     float ypeak[10];
// //     float peak[10];
// //     float width;
// //     float widthMin;
// //     int   maxPeaks    = 10;
// //     float minStrength = 0.05;
// //     width             = 10;
// //     widthMin          = 1;
// //     float fs[6], fiv[6];
// //     float a11, a12, a21, a22;
// //     // float theta = 50.98;
// //     // float ref   = 47.99;
// //     float phi = 86.3;

// //     wxPrintf("X_dim = %i, Y_dim = %i \n", X_dim, Y_dim);
// //     wxPrintf("hello2\n");
// //     // float str = fabs(cosf(ref / 180 * PI) / cosf(theta / 180 * PI));
// //     // // float str = fabs(cosf(theta / 180 * PI) / cosf(ref / 180 * PI));
// //     // // str = fabs((cosf(ref / 180 * PI)) / cosf(theta / 180 * PI));
// //     // wxPrintf("str: %g\n", str);
// //     // // rotmagstrToAmat(0, 1.0, str, phi, &a11, &a12, &a21, &a22);
// //     // rotmagstrToAmat(0, 1.0, str, phi, &a11, &a12, &a21, &a22);
// //     // // rotmagstrToAmat(theta, 1.0, str, phi, &a11, &a12, &a21, &a22);
// //     // // rotmagstrToAmat(ref - theta, 1.0, str, phi, &a11, &a12, &a21, &a22);
// //     // // rotmagstrToAmat(1.0, 1.0, str, 0.0, &a11, &a12, &a21, &a22);
// //     // // fs[0] = a11, fs[1] = a12, fs[2] = 0.0;
// //     // // fs[3] = a21, fs[4] = a22, fs[5] = 0.0;
// //     // // note that the fortran matrix is column dominate, c++ is row dominate
// //     // // fs[0] = a11, fs[1] = a12, fs[4] = 0.0;

// //     // fs[0] = a11, fs[2] = a12, fs[4] = 0.0;
// //     // fs[1] = a21, fs[3] = a22, fs[5] = 0.0;

// //     // float fs_a[6], fiv_a[6];

// //     // rotmagstrToAmat(0, 1.0, 1, -phi, &a11, &a12, &a21, &a22);
// //     // fs_a[0] = a11, fs_a[2] = a12, fs_a[4] = 0.0;
// //     // fs_a[1] = a21, fs_a[3] = a22, fs_a[5] = 0.0;

// //     // xfInvert(fs_a, fiv_a, 2);

// //     // // float unX1, unY1;
// //     // // float xorg = -1173.334 / 4, yorg = 1680.392 / 4;
// //     // // xfApply(fs, 0.0, 0.0, xorg, yorg, &unX1, &unY1, 2);
// //     // // wxPrintf("un1, un2 %g, %g, %g, %g\n", unX1, unY1, unX1 * 4, unY1 * 4);

// //     // xfInvert(fs, fiv, 2);

// //     // wxPrintf("a11 a12 a21 a22 %g, %g, %g, %g \n", a11, a12, a21, a22);
// //     // wxPrintf("fs              %g, %g, %g, %g, %g, %g \n", fs[0], fs[1], fs[2], fs[3], fs[4], fs[5]);
// //     // wxPrintf("finv            %g, %g, %g, %g, %g, %g \n", fiv[0], fiv[1], fiv[2], fiv[3], fiv[4], fiv[5]);
// //     // wxPrintf("fs_a              %g, %g, %g, %g, %g, %g \n", fs_a[0], fs_a[1], fs_a[2], fs_a[3], fs_a[4], fs_a[5]);
// //     // float finv_c[6];
// //     // finv_c[0] = fiv[0], finv_c[1] = fiv[2], finv_c[2] = fiv[4];
// //     // finv_c[3] = fiv[1], finv_c[4] = fiv[3], finv_c[5] = fiv[5];

// //     // padded_dimensions_x = ReturnClosestFactorizedUpper(pad_factor * input_file_2d.ReturnXSize( ), 3);
// //     // padded_dimensions_y = ReturnClosestFactorizedUpper(pad_factor * input_file_2d.ReturnYSize( ), 3);

// //     //   input_volume.Allocate(input_file_3d.ReturnXSize( ), input_file_3d.ReturnYSize( ), input_file_3d.ReturnZSize( ), true);
// //     // circlemask_image.Allocate(X_dim, Y_dim, true);
// //     peak_image.Allocate(X_dim, Y_dim, true);
// //     // peak_image.ReadSlice(&input_peak, 1);
// //     // wxPrintf("hello3\n");
// //     // input_image.Allocate(input_test.ReturnXSize( ), input_test.ReturnYSize( ), true);
// //     // input_image.ReadSlice(&input_test, 1);
// //     // wxPrintf("hello4\n");
// //     // int raw_X_dim = 1440;
// //     // int raw_Y_dim = 1023;

// //     // float mask_radius_x = raw_X_dim / 2.0 - raw_X_dim / 10.0;
// //     // float mask_radius_y = raw_Y_dim / 2.0 - raw_Y_dim / 10.0;
// //     // float mask_radius_z = 1;
// //     // // float mask_edge     = std::max(raw_image_dim_x / bin, raw_image_dim_y / bin) / 4.0;
// //     // // float mask_edge = 192;
// //     // float mask_edge = std::max(raw_X_dim, raw_Y_dim) / 10.0;
// //     // // float wanted_taper_edge_x  = std::max(raw_X_dim, raw_Y_dim) / 10.0;
// //     // // float wanted_taper_edge_y  = std::max(raw_X_dim, raw_Y_dim) / 10.0;
// //     // float wanted_taper_edge_x  = raw_Y_dim / 10.0;
// //     // float wanted_taper_edge_y  = raw_Y_dim / 10.0;
// //     // float wanted_mask_radius_x = raw_X_dim / 2.0;
// //     // float wanted_mask_radius_y = raw_Y_dim / 2.0;
// //     // input_image.TaperLinear(wanted_taper_edge_x, wanted_taper_edge_y, 1, mask_radius_x, mask_radius_y, 0);
// //     // // input_image.TaperEdges( );
// //     // input_image.WriteSlicesAndFillHeader("/groups/lingli/Documents/cisTEM/build/tomoalign_Intel-gpu-debug-static/src/Output/input_bf_stretch35_0.030_taperLinear.mrc", 1);

// //     // Image empty_image, padded_image;
// //     // empty_image.Allocate(raw_X_dim, raw_Y_dim, true);
// //     // empty_image.SetToConstant(1.0);
// //     // padded_image.Allocate(X_dim, Y_dim, true);
// //     // empty_image.ClipInto(&padded_image);
// //     // empty_image.CopyFrom(&padded_image);
// //     // empty_image.WriteSlicesAndFillHeader("/groups/lingli/Documents/cisTEM/build/tomoalign_Intel-gpu-debug-static/src/Output/empty.mrc", 1);

// //     // empty_image.TaperLinear(wanted_taper_edge_x, wanted_taper_edge_y, 1, mask_radius_x, mask_radius_y, 0);
// //     // empty_image.WriteSlicesAndFillHeader("/groups/lingli/Documents/cisTEM/build/tomoalign_Intel-gpu-debug-static/src/Output/emptyLinearTaper.mrc", 1);
// //     // Peak cistem_peak;
// //     // cistem_peak = peak_image.FindPeakWithIntegerCoordinates( );
// //     // wxPrintf("cisTEM peak %g, %g, %g,\n", cistem_peak.x, cistem_peak.y, cistem_peak.value);
// //     // // wxPrintf("hello3\n");
// //     // XCorrPeakFindWidth(&peak_image.real_values[0], X_dim + 2, Y_dim, xpeak, ypeak,
// //     //                    peak, &width, &widthMin, maxPeaks,
// //     //    minStrength);
// //     // //    XCorrPeakFindWidth()
// //     // Image image_ref;
// //     // Image image_cur, image_shifted;
// //     // image_ref.Allocate(ref_file.ReturnXSize( ), ref_file.ReturnYSize( ), true);
// //     // image_cur.Allocate(input_file.ReturnXSize( ), input_file.ReturnYSize( ), true);
// //     // image_cur.ReadSlice(&input_file, 1);
// //     // image_ref.ReadSlice(&ref_file, 1);
// //     // float ccc1, ccc2, ccc3, ccc4;
// //     // float peakmax = peak[0];
// //     float theta = 51;
// //     float ref;
// //     float tmppeak[4];
// //     // float ref   = 48;
// //     // for ( int i = 19; i < 35; i++ ) {
// //     // for ( int i = 2; i < 19; i++ ) {
// //     for ( int i = 0; i < 34; i++ ) {
// //         // // if ( theta < 0 ) {
// //         // // theta = 0 + (i - 1) * 3;
// //         // theta = theta + 3;
// //         // ref   = theta - 3;
// //         // // }

// //         // // theta = theta + i * 3;
// //         // // ref = ref - (17 - 1 - i) * 3;
// //         wxPrintf("--------image %i ----------\n", i);
// //         // wxPrintf("current nd ref: %g, %g\n", theta, ref);
// //         // float str = fabs(cosf(ref / 180 * PI) / cosf(theta / 180 * PI));
// //         // // float str = fabs(cosf(theta / 180 * PI) / cosf(ref / 180 * PI));
// //         // // str = fabs((cosf(ref / 180 * PI)) / cosf(theta / 180 * PI));
// //         // wxPrintf("str: %g\n", str);
// //         // // rotmagstrToAmat(0, 1.0, str, phi, &a11, &a12, &a21, &a22);
// //         // rotmagstrToAmat(0, 1.0, str, phi, &a11, &a12, &a21, &a22);
// //         // // rotmagstrToAmat(theta, 1.0, str, phi, &a11, &a12, &a21, &a22);
// //         // // rotmagstrToAmat(ref - theta, 1.0, str, phi, &a11, &a12, &a21, &a22);
// //         // // rotmagstrToAmat(1.0, 1.0, str, 0.0, &a11, &a12, &a21, &a22);
// //         // // fs[0] = a11, fs[1] = a12, fs[2] = 0.0;
// //         // // fs[3] = a21, fs[4] = a22, fs[5] = 0.0;
// //         // // note that the fortran matrix is column dominate, c++ is row dominate
// //         // // fs[0] = a11, fs[1] = a12, fs[4] = 0.0;

// //         // fs[0] = a11, fs[2] = a12, fs[4] = 0.0;
// //         // fs[1] = a21, fs[3] = a22, fs[5] = 0.0;

// //         // // float fs_a[6], fiv_a[6];

// //         // // rotmagstrToAmat(0, 1.0, 1, -phi, &a11, &a12, &a21, &a22);
// //         // // fs_a[0] = a11, fs_a[2] = a12, fs_a[4] = 0.0;
// //         // // fs_a[1] = a21, fs_a[3] = a22, fs_a[5] = 0.0;

// //         // // xfInvert(fs_a, fiv_a, 2);

// //         // // float unX1, unY1;
// //         // // float xorg = -1173.334 / 4, yorg = 1680.392 / 4;
// //         // // xfApply(fs, 0.0, 0.0, xorg, yorg, &unX1, &unY1, 2);
// //         // // wxPrintf("un1, un2 %g, %g, %g, %g\n", unX1, unY1, unX1 * 4, unY1 * 4);

// //         // xfInvert(fs, fiv, 2);

// //         // wxPrintf("a11 a12 a21 a22 %g, %g, %g, %g \n", a11, a12, a21, a22);
// //         // wxPrintf("fs              %g, %g, %g, %g, %g, %g \n", fs[0], fs[1], fs[2], fs[3], fs[4], fs[5]);
// //         // wxPrintf("finv            %g, %g, %g, %g, %g, %g \n", fiv[0], fiv[1], fiv[2], fiv[3], fiv[4], fiv[5]);
// //         // // wxPrintf("fs_a              %g, %g, %g, %g, %g, %g \n", fs_a[0], fs_a[1], fs_a[2], fs_a[3], fs_a[4], fs_a[5]);
// //         // float finv_c[6];
// //         // finv_c[0] = fiv[0], finv_c[1] = fiv[2], finv_c[2] = fiv[4];
// //         // finv_c[3] = fiv[1], finv_c[4] = fiv[3], finv_c[5] = fiv[5];

// //         peak_image.ReadSlice(&input_peak, i + 1);
// //         wxPrintf("hello3\n");
// //         Peak cistem_peak;
// //         cistem_peak = peak_image.FindPeakWithIntegerCoordinates( );
// //         wxPrintf("cisTEM peak %g, %g, %g,\n", cistem_peak.x, cistem_peak.y, cistem_peak.value);
// //         // tmppeak[0] = i, tmppeak[1] = peak.x, tmppeak[2] = peak.y, tmppeak[3] = peak.value;
// //         // wxPrintf("hello3\n");
// //         // XCorrPeakFindWidth(&peak_image.real_values[0], X_dim + 2, Y_dim, xpeak, ypeak,
// //         //                    peak, &width, &widthMin, maxPeaks, minStrength);
// //         // wxPrintf("peaks: %g, %g, %g\n", xpeak[0] - X_dim / 2, ypeak[0] - Y_dim / 2, peak[i]);
// //         // wxPrintf("peaks: %g, %g, %g\n", (xpeak[0] - X_dim / 2) * 4, (ypeak[0] - Y_dim / 2) * 4, peak[0]);
// //         // float unX1, unY1, unX2, unY2;
// //         // xfApply(fiv, 0.0, 0.0, xpeak[0] - X_dim / 2, ypeak[0] - Y_dim / 2, &unX1, &unY1, 2);
// //         // wxPrintf("ind un1, un2 %i %5g %5g %5g %5g\n", i, unX1, unY1, unX1 * 4, unY1 * 4);

// //         // wxPrintf("peaks: %g, %g, %g\n", xpeak[0], ypeak[0], peak[0]);
// //         // // wxPrintf("peaks: %g, %g, %g\n", xpeak[i] - X_dim / 2, ypeak[i] - Y_dim / 2, peak[i]);
// //         // // ccc1 = image_cur.ReturnCorrelationCoefficientUnnormalized(image_ref, wxMax(xpeak[i] - X_dim / 2, ypeak[i] - Y_dim / 2));
// //         // // image_cur.WriteSlicesAndFillHeader("../src/Output/test1.mrc", 1);
// //         // image_shifted.CopyFrom(&image_cur);
// //         // image_shifted.PhaseShift(-(xpeak[i] - X_dim / 2), -(ypeak[i] - Y_dim / 2), 0);
// //         // image_shifted.WriteSlicesAndFillHeader("shift1_phase.mrc", 1);
// //         // // image_shifted.CopyFrom(&image_cur);
// //         // // image_shifted.RealSpaceIntegerShift((xpeak[i] - X_dim / 2), (ypeak[i] - Y_dim / 2), 0);
// //         // // image_shifted.WriteSlicesAndFillHeader("shift2_real.mrc", 1);

// //         // // image_shifted.RealSpaceIntegerShift(-(xpeak[i] - X_dim / 2), -(ypeak[i] - Y_dim / 2), 0);
// //         // // ccc1        = image_shifted.ReturnCorrelationCoefficientNormalized(image_ref, wxMax(X_dim / 8, Y_dim / 8));
// //         // // float sizex = raw_X_dim / 2 - mask_edge - abs(xpeak[i] - X_dim / 2) / 2;
// //         // // float sizey = raw_Y_dim / 2 - mask_edge - abs(ypeak[i] - Y_dim / 2) / 2;
// //         // float sizex = raw_X_dim / 2 - mask_edge / 2;
// //         // float sizey = raw_Y_dim / 2 - mask_edge / 2;

// //         // wxPrintf("sizex sizey: %g, %g \n", sizex, sizey);
// //         // // ccc1 = image_shifted.ReturnCorrelationCoefficientNormalizedRectangle(image_ref, sizex, sizey);
// //         // // ccc2 = image_shifted.ReturnCorrelationCoefficientNormalized(image_ref, wxMax(X_dim / 8, Y_dim / 8));
// //         // // ccc3 = image_shifted.ReturnCorrelationCoefficientNormalizedRectangle(image_ref, raw_X_dim / 2, raw_Y_dim / 2);
// //         // // float testresult[2];
// //         // int num, num_tmp;
// //         // ccc3 = image_cur.ReturnCorrelationCoefficientNormalizedAtPeak(image_ref, &num, xpeak[i] - X_dim / 2, ypeak[i] - Y_dim / 2, 0, sizex, sizey);
// //         // // ccc3 = image_cur.ReturnCorrelationCoefficientNormalizedAtPeak(image_shifted, &num, xpeak[i] - X_dim / 2, ypeak[i] - Y_dim / 2, 0, sizex, sizey);
// //         // // overlap = float(nsum) / ((nxPad - 2 * nxCCTrimA) * (nyPad - 2 * nyCCTrimA));
// //         // // ccc3               = testresult[0];
// //         // // float num;

// //         // float overlap = num / (2 * sizex) / (2 * sizey);

// //         // float overlapPower = 6;
// //         // float weight       = 1. / (1 + powf(wxMax(0.1, (wxMin(10.0, 0.125 / overlap))), overlapPower));
// //         // wxPrintf("num overlap, overlap 0.125/overlap weight %i,%g, %g, %g\n", num, overlap, 0.125 / overlap, weight);

// //         // float wgtccc3 = ccc3 * 1. / (1.0 + powf(wxMax(0.1, wxMin(10.0, 0.125 / overlap)), overlapPower));
// //         // // wxPrintf("overlap overlap power, wgtc")
// //         // // float wgtCCC       = ccc * 1. / (1. + &wxMax(0.1, min(10., overlapCrit / overlap)) * *overlapPower);
// //         // // ccc1 = image_cur.ReturnCorrelationCoefficientNormalizedRectangle(image_ref, sizex, sizey);
// //         // // ccc2 = image_cur.ReturnCorrelationCoefficientNormalized(image_ref, wxMax(X_dim / 8, Y_dim / 8));
// //         // // ccc3 = image_cur.ReturnCorrelationCoefficientNormalizedRectangle(image_ref, raw_X_dim / 2, raw_Y_dim / 2);

// //         // // ccc2 = image_shifted.ReturnCorrelationCoefficientUnnormalized(image_ref, wxMax(abs(xpeak[i] - X_dim / 2), abs(ypeak[i] - Y_dim / 2)));
// //         // // image_cur.WriteSlicesAndFillHeader("../src/Output/test2.mrc", 1);
// //         // // wxPrintf("coefs 1 2: %g, %g\n", ccc1, ccc2);
// //         // float ratio = peak[i] / peakmax;
// //         // // wxPrintf("peaks: %5g, %5g, %10.5g, %10.5g, %10.5g, %10.5g,%10.5g,%10.5g\n", xpeak[i] - X_dim / 2, ypeak[i] - Y_dim / 2, peak[i], ccc1, ccc2, ccc3, ccc4, ratio);
// //         // wxPrintf("peaks: %5g, %5g, %10.5g, %10.5g, %10.5g,%10.5g\n", xpeak[i] - X_dim / 2, ypeak[i] - Y_dim / 2, peak[i], ccc3, wgtccc3, ratio);
// //         // wxPrintf("peaks: %g, %g, %g\n", xpeak[i] - X_dim / 2, ypeak[i] - Y_dim / 2, peak[i]);
// //         // wxPrintf("peaks: %g, %g, %g\n", (xpeak[i] - X_dim / 2) * 4, (ypeak[i] - Y_dim / 2) * 4, peak[i]);
// //         // wxPrintf("peaks: %g, %g, %g\n", xpeak[i], ypeak[i], peak[i]);

// //         // wxPrintf("peaks: %g, %g, %g\n", xpeak[0] - X_dim / 2, ypeak[0] - Y_dim / 2, peak[0]);
// //         // wxPrintf("peaks: %g, %g, %g\n", xpeak[1] - X_dim / 2, ypeak[1] - Y_dim / 2, peak[1]);
// //         // wxPrintf("peaks: %g, %g, %g\n", xpeak[2] - X_dim / 2, ypeak[2] - Y_dim / 2, peak[2]);
// //         // wxPrintf("peaks: %g, %g, %g\n", xpeak[3] - X_dim / 2, ypeak[3] - Y_dim / 2, peak[3]);
// //         // wxPrintf("peaks: %g, %g, %g\n", xpeak[4] - X_dim / 2, ypeak[4] - Y_dim / 2, peak[4]);
// //         // wxPrintf("peaks: %g, %g, %g\n", xpeak[5] - X_dim / 2, ypeak[5] - Y_dim / 2, peak[5]);
// //         // wxPrintf("peaks: %g, %g, %g\n", xpeak[6] - X_dim / 2, ypeak[6] - Y_dim / 2, peak[6]);
// //         // wxPrintf("peaks: %g, %g, %g\n", xpeak[7] - X_dim / 2, ypeak[7] - Y_dim / 2, peak[7]);
// //         // wxPrintf("peaks: %g, %g, %g\n", xpeak[8] - X_dim / 2, ypeak[8] - Y_dim / 2, peak[8]);
// //         // wxPrintf("peaks: %g, %g, %g\n", xpeak[9] - X_dim / 2, ypeak[9] - Y_dim / 2, peak[9]);
// //         // float unX1, unY1, unX2, unY2;

// //         // xfApply(fs, 0.0, 0.0, xpeak[i] - X_dim / 2, ypeak[i] - Y_dim / 2, &unX2, &unY2, 2);
// //         // // xfApply(fs, 0.0, 0.0, xpeak[i], ypeak[i], &unX2, &unY2, 2);
// //         // wxPrintf("un1, un2 %g, %g, %g, %g\n", unX2, unY2, unX2 * 4, unY2 * 4);

// //         // xfApply(fiv, 0.0, 0.0, xpeak[i] - X_dim / 2, ypeak[i] - Y_dim / 2, &unX1, &unY1, 2);
// //         // wxPrintf("un1, un2 %g, %g, %g, %g\n", unX1, unY1, unX1 * 4, unY1 * 4);
// //         // xfApply(finv_c, 0.0, 0.0, xpeak[i] - X_dim / 2, ypeak[i] - Y_dim / 2, &unX2, &unY2, 2);

// //         // xfApply(fs, float(X_dim) / 2.0, float(Y_dim) / 2.0, xpeak[i], ypeak[i], &unX2, &unY2, 2);
// //         // wxPrintf("un1, un2 %g, %g, %g, %g\n", unX2, unY2, unX2 * 4, unY2 * 4);
// //         // wxPrintf("un1, un2 %g, %g, %g, %g\n", unX2 - X_dim / 2.0, unY2 - Y_dim / 2.0, unX2 * 4, unY2 * 4);

// //         // xfApply(fiv, float(X_dim) / 2.0, float(Y_dim) / 2.0, xpeak[i], ypeak[i], &unX2, &unY2, 2);
// //         // wxPrintf("un1, un2 %g, %g, %g, %g\n", unX2, unY2, unX2 * 4, unY2 * 4);
// //         // wxPrintf("un1, un2 %g, %g, %g, %g\n", unX2 - X_dim / 2.0, unY2 - Y_dim / 2.0, unX2 * 4, unY2 * 4);
// //         // call xfapply(fsInv, 0., 0., xpeak, ypeak, unstretchDx, unstretchDy)

// //         // float x1 = unX1, y1 = unY1;
// //         // xfApply(fs_a, 0, 0, x1, y1, &unX2, &unY2, 2);
// //         // wxPrintf("un1, un2 %g, %g, %g, %g\n", unX2, unY2, unX2 * 4, unY2 * 4);
// //         // wxPrintf("un1, un2 %g, %g, %g, %g\n", unX2 - X_dim / 2.0, unY2 - Y_dim / 2.0, unX2 * 4, unY2 * 4);
// //     }

// //     // input_array = peak_image.real_values;
// //     // delete xpeak;
// //     // delete ypeak;
// //     // delete peak;
// //     return true;
// // }

// // bool NikoTestApp::DoCalculation( ) {
// //     wxPrintf("Hello world4\n");
// //     // // int X_maskcenter          = my_current_job.arguments[1].ReturnIntegerArgument( );
// //     wxString input_imgstack = my_current_job.arguments[0].ReturnStringArgument( );
// //     // // wxString output_stack_filename = my_current_job.arguments[1].ReturnStringArgument( );
// //     wxString angle_filename = my_current_job.arguments[1].ReturnStringArgument( );
// //     int      img_index      = my_current_job.arguments[2].ReturnIntegerArgument( );
// //     // wxString coordinates_filename  = my_current_job.arguments[2].ReturnStringArgument( );
// //     // int      output_stack_box_size = my_current_job.arguments[3].ReturnIntegerArgument( );

// //     NumericTextFile* tilt_angle_file;
// //     tilt_angle_file = new NumericTextFile(angle_filename, OPEN_TO_READ, 1);

// //     //stack manipulate
// //     // MRCFile input_stack("/groups/lingli/Downloads/CTEM_tomo1/10064/proc_07112022/clip/tomo1_ali.mrc", false);
// //     MRCFile input_stack(input_imgstack.ToStdString( ), false);
// //     // MRCFile output_stack(output_stack_filename.ToStdString( ), true);
// //     // output_stack.OpenFile(output_stack_filename.ToStdString( ), true);
// //     int image_no = input_stack.ReturnNumberOfSlices( );
// //     // MRCFile input_stack("/groups/lingli/Downloads/CTEM_tomo1/10064/proc_07112022/clip/tomo1_ali.mrc", false);
// //     // wxPrintf("image number in the stack: %i\n", image_no);

// //     Image current_image;
// //     Image cos_image;
// //     // MRCFile      output_file("output.mrc", true);
// //     // ProgressBar* my_progress = new ProgressBar(input_stack.ReturnNumberOfSlices( ));
// //     int my_x;
// //     int my_y;
// //     // Image box;

// //     // box.Allocate(output_stack_box_size, output_stack_box_size, 1, true);
// //     // int number_of_particles = input_coos_file->number_of_lines;

// //     //write a if statement to judge if the number of coordinates in the coord file equals to image_no
// //     // int number_of_patchgroups = input_coos_file->number_of_lines;
// //     // float temp_array[3];
// //     float temp_angle[1];
// //     int   x_at_centertlt, y_at_centertlt;
// //     // int   col = 2;

// //     // float    temp_array[number_of_patchgroups][2];
// //     // MRCFile* patch = new MRCFile[number_of_patchgroups];
// //     // for ( int patch_counter = 0; patch_counter < number_of_patchgroups; patch_counter++ ) {
// //     //     input_coos_file->ReadLine(temp_array[patch_counter]);
// //     //     // string tmpstring = std::to_string(patch_counter);
// //     //     // tmpstring = patch_counter.ToStdString( );
// //     //     // MRCFile tmpstring(std::to_string(patch_counter) + ".mrc", true);
// //     //     // string zz                     = std::to_string(patch_counter) + ".mrc";
// //     //     // patch[patch_counter] = MRCFile(wxString::Format("%g.mrc", patch_counter).ToStdString( ), true);
// //     //     patch[patch_counter].OpenFile(wxString::Format("%i.mrc", patch_counter).ToStdString( ), true);
// //     // }

// //     //     string zz, zzname;
// //     // zz     = std::to_string(patch_counter) + ".mrc";
// //     // zzname = std::to_string(patch_counter);
// //     // MRCFile zzname(zz, true);
// //     // wxPrintf("number of patch groups: %i\n\n", number_of_patchgroups);
// //     for ( long image_counter = 0; image_counter < img_index; image_counter++ ) {
// //         // current_image.ReadSlice(&input_stack, image_counter + 1);
// //         // float image_mean = current_image.ReturnAverageOfRealValues( );

// //         tilt_angle_file->ReadLine(temp_angle);
// //         // my_image.crop( );
// //     }
// //     current_image.ReadSlice(&input_stack, img_index);
// //     float xdim     = current_image.logical_x_dimension;
// //     float xdim_cos = xdim * cosf(temp_angle[0] / 180.0 * PI);
// //     float ydim     = current_image.logical_y_dimension;
// //     wxPrintf("dimensions %g %g %g\n", xdim, xdim_cos, ydim);
// //     cos_image.Allocate(int(xdim_cos) + 1, int(ydim), 1, true);
// //     current_image.ClipInto(&cos_image, 0.0, false, 1.0, int(xdim_cos), int(ydim), 0);
// //     wxPrintf("dimensions %i %i\n", cos_image.logical_x_dimension, cos_image.logical_y_dimension);

// //     float wantedvalue;
// //     wantedvalue = 1.0;
// //     // cos_image.AddByLinearInterpolationFourier2D( xdim, ydim, 1.0);

// //     // cos_image.AddByLinearInterpolationReal(xdim, ydim, wantedvalue, wantedvalue);

// //     cos_image.WriteSlicesAndFillHeader("costest1.mrc", 1);

// //     // for ( int patch_counter = 0; patch_counter < number_of_patchgroups; patch_counter++ ) {

// //     //     // input_coos_file->ReadLine(temp_array);
// //     //     // x_at_centertlt = temp_array[0];
// //     //     // y_at_centertlt = temp_array[1];
// //     //     x_at_centertlt = temp_array[patch_counter][0];
// //     //     y_at_centertlt = temp_array[patch_counter][1];
// //     //     my_x           = int(x_at_centertlt * cosf(PI * temp_angle[0] / 180.0));
// //     //     my_y           = y_at_centertlt;
// //     //     current_image.ClipInto(&box, image_mean, false, 1.0, int(my_x), int(my_y), 0);

// //     //     // wxPrintf("x=%i, y=%i\n", my_x, my_y);
// //     //     // string zz = std::to_string(patch_counter) + ".mrc";
// //     //     box.WriteSlice(&patch[patch_counter], image_counter + 1);
// //     //     // wxPrintf("%.0f .mrc", patch_counter)
// //     //     // box.WriteSlice(zz.str( ), image_counter + 1);
// //     // }
// //     // my_progress->Update(image_counter + 1);
// //     return true;
// // }

// // delete my_progress;
// // delete input_coos_file;
// // delete[] patch;
// // delete temp_array;

// /*    square mask part
//     Image masked_image;
//     // Image circlemask_image;
//     Image squaremask;

//     wxString input_2d     = my_current_job.arguments[0].ReturnStringArgument( );
//     int      X_maskcenter = my_current_job.arguments[1].ReturnIntegerArgument( );
//     int      Y_maskcenter = my_current_job.arguments[2].ReturnIntegerArgument( );
//     // float    SquareMaskSize = my_current_job.arguments[3].ReturnFloatArgument( );
//     int SquareMaskSize = my_current_job.arguments[3].ReturnIntegerArgument( );

//     wxPrintf("Hello world\n");

//     MRCFile input_file_2d(input_2d.ToStdString( ), false);
//     Image   input_image;

//     int X_dim, Y_dim;

//     // int X_maskcenter, Y_maskcenter;
//     X_dim = input_file_2d.ReturnXSize( );
//     Y_dim = input_file_2d.ReturnYSize( );
//     // X_maskcenter = X_dim / 4;
//     // Y_maskcenter = Y_dim / 4;
//     wxPrintf("X_dim = %i, Y_dim = %i \n", X_dim, Y_dim);
//     wxPrintf("X_maskcenter = %i, Y_maskcenter = %i \n", X_maskcenter, Y_maskcenter);
//     wxPrintf("masksize = %i\n\n", SquareMaskSize);

//     // padded_dimensions_x = ReturnClosestFactorizedUpper(pad_factor * input_file_2d.ReturnXSize( ), 3);
//     // padded_dimensions_y = ReturnClosestFactorizedUpper(pad_factor * input_file_2d.ReturnYSize( ), 3);

//     //   input_volume.Allocate(input_file_3d.ReturnXSize( ), input_file_3d.ReturnYSize( ), input_file_3d.ReturnZSize( ), true);
//     // circlemask_image.Allocate(X_dim, Y_dim, true);
//     input_image.Allocate(X_dim, Y_dim, true);
//     input_image.ReadSlice(&input_file_2d, 1);

//     squaremask.Allocate(X_dim, Y_dim, true);
//     squaremask.SetToConstant(1.0);
//     squaremask.SquareMaskWithValue(SquareMaskSize, 0.0, false, X_maskcenter, Y_maskcenter, 0);
//     // GaussianLowPassFilter(float sigma)
//     squaremask.ForwardFFT( );
//     squaremask.GaussianLowPassFilter(0.01);
//     squaremask.BackwardFFT( );

//     masked_image.CopyFrom(&input_image);
//     masked_image.Normalize(10.0);
//     masked_image.WriteSlicesAndFillHeader("normalized.mrc", 1);

//     // masked_image.ForwardFFT( );
//     // squaremask.ForwardFFT( );
//     masked_image.MultiplyPixelWise(squaremask);

//     input_image.physical_address_of_box_center_x = X_maskcenter;
//     input_image.physical_address_of_box_center_y = Y_maskcenter;
//     input_image.CosineRectangularMask(SquareMaskSize / 2, SquareMaskSize / 2 + 50, 0, 50, false, false, 0.0);

//     input_image.WriteSlicesAndFillHeader("cosinrectangularmask.mrc", 1);

//     // zz = masked_image.ApplyMask(squaremask, 80, 10, 0.5, 10);
//     masked_image.WriteSlicesAndFillHeader("imageapplysquaremask.mrc", 1);
//     squaremask.WriteSlicesAndFillHeader("squaretest.mrc", 1);

//     input_image.CalculateCrossCorrelationImageWith(&masked_image);
//     input_image.WriteSlicesAndFillHeader("ccn-temp.mrc", 1);

//     Peak peaks;
//     peaks.x = 10;
//     peaks.y = 10;
//     wxPrintf("peak position: x = %g, y = %g \n\n", peaks.x, peaks.y, peaks.value);

//     peaks = input_image.FindPeakWithIntegerCoordinates(0, 400, 10);
//     wxPrintf("peak position: x = %g, y = %g ,value = %g\n\n", peaks.x, peaks.y, peaks.value);

//     input_image.RealSpaceIntegerShift(15, 20, 0);
//     peaks = input_image.FindPeakWithIntegerCoordinates(0, 400, 10);
//     wxPrintf("peak position: x = %g, y = %g ,value = %g\n\n", peaks.x, peaks.y, peaks.value);

//        square msk part end */

// // padded_image.Allocate(padded_dimensions_x, padded_dimensions_y, true);

// // masked_image.Allocate(input_file_2d.ReturnXSize( ), input_file_2d.ReturnYSize( ), true);

// // //  input_volume.ReadSlices(&input_file_3d, 1, input_file_3d.ReturnZSize( ));
// // input_image.ReadSlice(&input_file_2d, 1);
// // input_image.Normalize(10);

// // circlemask_image.ReadSlice(&mask_file_2d, 1);

// // sigma = sqrtf(input_image.ReturnVarianceOfRealValues( ));
// // // temp
// // wxPrintf("xdim= %i mean=%g \n\n", input_file_2d.ReturnXSize( ), input_image.ReturnAverageOfRealValues( ));
// // wxPrintf("sigma= %g \n\n", sigma);

// // masked_image.CopyFrom(&input_image);
// // float zz;
// // zz = masked_image.ApplyMask( );

// //masked_image.SquareMaskWithValue(500, 0);
// // masked_image.TriangleMask(300); //this create a mask, not apply mask to the image
// // masked_image.CircleMask(200); //this mask the original image by a circle
// // masked_image.WriteSlicesAndFillHeader("circlemask.mrc", 1);
// // masked_image.CircleMaskWithValue(200, -10); //this mask the original image by a circle
// // masked_image.WriteSlicesAndFillHeader("circlemaskn10.mrc", 1);
// // float zz;
// // zz = masked_image.ApplyMask(masked_image, 20);
// // zz = masked_image.ApplyMask(circlemask_image, 80, 10, 0.5, 10);
// // masked_image.WriteSlicesAndFillHeader("imageapplymask.mrc", 1);
// // wxPrintf("zzzzzzzzz%g\n\n", zz);

// /*    ///-----------------------------------------------------------------------------------------------------------
//     input_image.ClipIntoLargerRealSpace2D(&padded_image);
//     padded_image.AddGaussianNoise(10.0 * sigma);
//     padded_image.WriteSlice(&output_file, 1);
//     padded_image.ForwardFFT( );

//     input_image.AddSlices(input_volume);
//     //	input_image.QuickAndDirtyWriteSlice("junk.mrc", 1);
//     count = 1;
//     output_image.CopyFrom(&padded_image);
//     temp_image.CopyFrom(&input_image);
//     temp_image.Resize(padded_dimensions_x, padded_dimensions_y, 1);
//     //	temp_image2.CopyFrom(&input_image);
//     //	temp_image2.Resize(padded_dimensions_x, padded_dimensions_y, 1);
//     //	temp_image2.RealSpaceIntegerShift(input_image.logical_x_dimension, input_image.logical_y_dimension, 0);
//     //	temp_image.AddImage(&temp_image2);
//     //	temp_image.QuickAndDirtyWriteSlice("junk.mrc", 1);
//     temp_image.ForwardFFT( );
//     output_image.ConjugateMultiplyPixelWise(temp_image);
//     output_image.SwapRealSpaceQuadrants( );
//     output_image.BackwardFFT( );
//     peak = output_image.real_values[output_image.logical_x_dimension / 2 + (output_image.logical_x_dimension + output_image.padding_jump_value) * output_image.logical_y_dimension / 2];
//     wxPrintf("\nPeak with whole projection = %g background = %g\n\n", peak, output_image.ReturnVarianceOfRealValues(float(2 * input_image.logical_x_dimension), 0.0, 0.0, 0.0, true));
//     //	wxPrintf("\nPeak with whole projection = %g\n\n", output_image.ReturnMaximumValue());
//     output_image.WriteSlice(&output_file, 2);
//     sum_of_peaks = 0.0;
//     for ( i = 1; i <= 3; i += 2 ) {
//         for ( j = 1; j <= 3; j += 2 ) {
//             output_image.CopyFrom(&padded_image);
//             temp_image.CopyFrom(&input_image);
//             temp_image.SquareMaskWithValue(float(input_image.logical_x_dimension) / 2.0, 0.0, false, i * input_image.logical_x_dimension / 4, j * input_image.logical_x_dimension / 4);
//             temp_image.Resize(padded_dimensions_x, padded_dimensions_y, 1);
//             temp_image.ForwardFFT( );
//             output_image.ConjugateMultiplyPixelWise(temp_image);
//             output_image.SwapRealSpaceQuadrants( );
//             output_image.BackwardFFT( );
//             peak = output_image.real_values[output_image.logical_x_dimension / 2 + (output_image.logical_x_dimension + output_image.padding_jump_value) * output_image.logical_y_dimension / 2];
//             //			peak = output_image.ReturnMaximumValue();
//             wxPrintf("Quarter peak = %i %i %g\n", i, j, peak);
//             sum_of_peaks += peak;
//             count++;
//             output_image.WriteSlice(&output_file, 1 + count);
//         }
//     }
//     wxPrintf("\nSum of quarter peaks = %g\n\n", sum_of_peaks);

//     sum_of_peaks = 0.0;
//     for ( i = 0; i < 4; i++ ) {
//         output_image.CopyFrom(&padded_image);
//         input_image.AddSlices(input_volume, i * input_volume.logical_z_dimension / 4 + 1, (i + 1) * input_volume.logical_z_dimension / 4);
//         //		input_image.QuickAndDirtyWriteSlice("junk.mrc", i + 1);
//         temp_image.CopyFrom(&input_image);
//         temp_image.Resize(padded_dimensions_x, padded_dimensions_y, 1);
//         temp_image.ForwardFFT( );
//         output_image.ConjugateMultiplyPixelWise(temp_image);
//         output_image.SwapRealSpaceQuadrants( );
//         output_image.BackwardFFT( );
//         peak = output_image.real_values[output_image.logical_x_dimension / 2 + (output_image.logical_x_dimension + output_image.padding_jump_value) * output_image.logical_y_dimension / 2];
//         //		peak = output_image.ReturnMaximumValue();
//         wxPrintf("Slice peak = %i %g\n", i + 1, peak);
//         sum_of_peaks += peak;
//         count++;
//         output_image.WriteSlice(&output_file, 1 + count);
//     }
//     wxPrintf("\nSum of slice peaks = %g\n", sum_of_peaks);
//     ///------------------------------- */

// /*	wxPrintf("\nDoing 1000 FFTs %i x %i\n", output_image.logical_x_dimension, output_image.logical_y_dimension);
// 	for (i = 0; i < 1000; i++)
// 	{
// 		output_image.is_in_real_space = false;
// 		output_image.SetToConstant(1.0);
// 		output_image.BackwardFFT();
// 	}
// 	wxPrintf("\nFinished\n");
// */

// /*	int i, j;
// 	int slice_thickness;
// 	int first_slice, last_slice, middle_slice;
// 	long offset;
// 	long pixel_counter;
// 	float bfactor = 20.0;
// 	float mask_radius = 75.0;
// 	float pixel_size = 1.237;
// //	float pixel_size = 0.97;
// 	float bfactor_res_limit = 8.0;
// 	float resolution_limit = 3.8;
// //	float resolution_limit = 3.0;
// 	float cosine_edge = 5.0;
// 	float bfactor_pixels;

// 	MRCFile input_file("input.mrc", false);
// 	MRCFile output_file_2D("output2D.mrc", true);
// 	MRCFile output_file_3D("output3D.mrc", true);
// 	Image input_image;
// 	Image output_image;
// 	Image output_image_3D;

// 	Curve power_spectrum;
// 	Curve number_of_terms;

// 	UserInput my_input("NikoTest", 1.00);
// 	pixel_size = my_input.GetFloatFromUser("Pixel size of images (A)", "Pixel size of input images in Angstroms", "1.0", 0.0);
// 	mask_radius = my_input.GetFloatFromUser("Mask radius (A)", "Radius of mask to be applied to input 3D map, in Angstroms", "100.0", 0.0);
// 	bfactor = my_input.GetFloatFromUser("B-Factor (A^2)", "B-factor to be applied to dampen the 3D map after spectral flattening, in Angstroms squared", "20.0");
// 	bfactor_res_limit = my_input.GetFloatFromUser("Low resolution limit for spectral flattening (A)", "The resolution at which spectral flattening starts being applied, in Angstroms", "8.0", 0.0);
// 	resolution_limit = my_input.GetFloatFromUser("High resolution limit (A)", "Resolution of low-pass filter applied to final output maps, in Angstroms", "3.0", 0.0);

// 	slice_thickness = myroundint(resolution_limit / pixel_size);
// 	input_image.Allocate(input_file.ReturnXSize(), input_file.ReturnYSize(), slice_thickness, true);
// 	output_image.Allocate(input_file.ReturnXSize(), input_file.ReturnYSize(), true);
// 	output_image_3D.Allocate(input_file.ReturnXSize(), input_file.ReturnYSize(), input_file.ReturnZSize(), true);

// 	wxPrintf("\nCalculating 3D spectrum...\n");

// 	power_spectrum.SetupXAxis(0.0, 0.5 * sqrtf(3.0), int((output_image_3D.logical_x_dimension / 2.0 + 1.0) * sqrtf(3.0) + 1.0));
// 	number_of_terms.SetupXAxis(0.0, 0.5 * sqrtf(3.0), int((output_image_3D.logical_x_dimension / 2.0 + 1.0) * sqrtf(3.0) + 1.0));
// 	output_image_3D.ReadSlices(&input_file, 1, input_file.ReturnZSize());
// 	output_image_3D.CosineMask(mask_radius / pixel_size, 10.0 / pixel_size);

// 	first_slice = int((input_file.ReturnZSize() - slice_thickness + 1) / 2.0);
// 	last_slice = first_slice + slice_thickness;
// 	pixel_counter = 0;
// 	for (j = first_slice; j < last_slice; j++)
// 	{
// 		offset = j * (output_image_3D.logical_x_dimension + output_image_3D.padding_jump_value) * output_image_3D.logical_y_dimension;
// 		for (i = 0; i < (output_image_3D.logical_x_dimension + output_image_3D.padding_jump_value) * output_image_3D.logical_y_dimension; i++) {input_image.real_values[pixel_counter] = output_image_3D.real_values[i + offset]; pixel_counter++;}
// 	}

// 	output_image_3D.ForwardFFT();
// 	output_image_3D.Compute1DPowerSpectrumCurve(&power_spectrum, &number_of_terms);
// 	power_spectrum.SquareRoot();
// 	wxPrintf("Done with 3D spectrum. Starting slice estimation...\n");

// //	input_image.ReadSlices(&input_file, first_slice, last_slice);
// 	bfactor_pixels = bfactor / pixel_size / pixel_size;
// 	input_image.ForwardFFT();
// 	input_image.ApplyBFactorAndWhiten(power_spectrum, bfactor_pixels, bfactor_pixels, pixel_size / bfactor_res_limit);
// //	input_image.ApplyBFactor(bfactor_pixels);
// //	input_image.CosineMask(pixel_size / resolution_limit, cosine_edge / input_file.ReturnXSize());
// 	input_image.CosineMask(pixel_size / resolution_limit - pixel_size / 40.0, pixel_size / 20.0);
// 	input_image.BackwardFFT();

// 	middle_slice = int(slice_thickness / 2.0);
// 	offset = middle_slice * (input_file.ReturnXSize() + input_image.padding_jump_value) * input_file.ReturnYSize();
// 	pixel_counter = 0;
// 	for (i = 0; i < (input_file.ReturnXSize() + input_image.padding_jump_value) * input_file.ReturnYSize(); i++) {output_image.real_values[pixel_counter] = input_image.real_values[i + offset]; pixel_counter++;}
// //	output_image.ForwardFFT();
// //	output_image.CosineMask(pixel_size / resolution_limit - pixel_size / 40.0, pixel_size / 20.0);
// //	output_image.BackwardFFT();
// 	output_image.WriteSlice(&output_file_2D, 1);
// 	wxPrintf("Done with slices. Starting 3D B-factor application...\n");

// 	output_image_3D.ApplyBFactorAndWhiten(power_spectrum, bfactor_pixels, bfactor_pixels, pixel_size / bfactor_res_limit);
// 	output_image_3D.CosineMask(pixel_size / resolution_limit - pixel_size / 40.0, pixel_size / 20.0);
// 	output_image_3D.BackwardFFT();
// 	output_image_3D.WriteSlices(&output_file_3D, 1, input_file.ReturnZSize());
// 	wxPrintf("Done with 3D B-factor application.\n");

// 	int i;
// 	int min_class;
// 	int max_class;
// 	int count;
// 	float temp_float;
// 	float input_parameters[17];

// 	MRCFile input_file("input.mrc", false);
// 	MRCFile output_file("output.mrc", true);
// 	Image input_image;
// 	Image padded_image;
// 	Image ctf_image;
// 	Image sum_image;
// 	CTF ctf;
// 	AnglesAndShifts rotation_angle;

// 	input_image.Allocate(input_file.ReturnXSize(), input_file.ReturnYSize(), true);
// 	padded_image.Allocate(4 * input_file.ReturnXSize(), 4 * input_file.ReturnYSize(), true);
// 	ctf_image.Allocate(input_file.ReturnXSize(), input_file.ReturnYSize(), false);
// 	sum_image.Allocate(input_file.ReturnXSize(), input_file.ReturnYSize(), false);
// 	sum_image.SetToConstant(0.0);

// 	FrealignParameterFile input_par_file("input.par", OPEN_TO_READ);
// 	input_par_file.ReadFile();

// //	count = 0;
// //	for (i = 1; i <= input_par_file.number_of_lines; i++)
// //	{
// //		if (i % 100 == 1) wxPrintf("Working on line %i\n", i);
// //		input_par_file.ReadLine(input_parameters);
// //		input_image.ReadSlice(&input_file, int(input_parameters[0] + 0.5));
// //		count++;
// //		input_image.WriteSlice(&output_file, count);
// //	}

// 	for (i = 1; i <= input_par_file.number_of_lines; i++)
// 	{
// 		if (i % 100 == 1) wxPrintf("Rotating image %i\n", i);
// 		input_par_file.ReadLine(input_parameters);
// 		input_image.ReadSlice(&input_file, i);
// 		input_image.RealSpaceIntegerShift(-input_parameters[4], -input_parameters[5]);
// 		input_image.ForwardFFT();
// 		input_image.ClipInto(&padded_image);
// 		padded_image.BackwardFFT();
// 		rotation_angle.GenerateRotationMatrix2D(-input_parameters[1]);
// 		padded_image.Rotate2DSample(input_image, rotation_angle);
// 		input_image.WriteSlice(&output_file, i);
// 		if (input_parameters[7] == 2)
// 		{
// 			ctf.Init(300.0, 0.0, 0.1, input_parameters[8], input_parameters[9], input_parameters[10], 0.0, 0.0, 0.0, 1.0, input_parameters[11]);
// 			ctf_image.CalculateCTFImage(ctf);
// 			input_image.ForwardFFT();
// 			input_image.PhaseFlipPixelWise(ctf_image);
// 			sum_image.AddImage(&input_image);
// 		}
// //		if (i == 1001) break;
// 	}
// 	sum_image.QuickAndDirtyWriteSlice("sum.mrc", 1);
// */

// /*	FrealignParameterFile input_par_file("input.par", OPEN_TO_READ);
// 	FrealignParameterFile output_par_file("output.par", OPEN_TO_WRITE);
// 	input_par_file.ReadFile(true);
// 	input_par_file.ReduceAngles();
// 	min_class = myroundint(input_par_file.ReturnMin(7));
// 	max_class = myroundint(input_par_file.ReturnMax(7));
// 	for (i = min_class; i <= max_class; i++)
// 	{
// 		temp_float = input_par_file.ReturnDistributionMax(2, i);
// 		if (temp_float != 0.0) wxPrintf("theta max, sigma = %i %g %g\n", i, temp_float, input_par_file.ReturnDistributionSigma(2, temp_float, i));
// //		input_par_file.SetParameters(2, temp_float, i);
// 		temp_float = input_par_file.ReturnDistributionMax(3, i);
// 		if (temp_float != 0.0) wxPrintf("phi max, sigma = %i %g %g\n", i, temp_float, input_par_file.ReturnDistributionSigma(3, temp_float, i));
// //		input_par_file.SetParameters(3, temp_float, i);
// 	} */
// //	for (i = 1; i <= input_par_file.number_of_lines; i++)
// //	{
// //		input_par_file.ReadLine(input_parameters);
// //		output_par_file.WriteLine(input_parameters);
// //	}

// //	MRCFile input_file("input.mrc", false);
// //	MRCFile output_file("output.mrc", true);
// //	Image input_image;
// //	Image filtered_image;
// //	Image kernel;

// //	input_image.Allocate(input_file.ReturnXSize(), input_file.ReturnYSize(), input_file.ReturnZSize(), true);
// //	filtered_image.Allocate(input_file.ReturnXSize(), input_file.ReturnYSize(), input_file.ReturnZSize(), true);
// //	kernel.Allocate(input_file.ReturnXSize(), input_file.ReturnYSize(), input_file.ReturnZSize(), true);
// //	input_image.ReadSlices(&input_file,1,input_image.logical_z_dimension);

// /*	kernel.SetToConstant(1.0);
// 	kernel.CosineMask(8.0, 8.0, false, true, 0.0);
// //	kernel.real_values[0] = 1.0;
// 	temp_float = kernel.ReturnAverageOfRealValues() * kernel.number_of_real_space_pixels;
// //	wxPrintf("average = %g\n", temp_float);
// //	kernel.WriteSlices(&output_file,1,input_image.logical_z_dimension);
// 	kernel.ForwardFFT();
// 	kernel.SwapRealSpaceQuadrants();
// 	kernel.MultiplyByConstant(float(kernel.number_of_real_space_pixels) / temp_float);
// //	kernel.CosineMask(0.03, 0.03, true);

// 	input_image.SetMinimumValue(0.0);
// 	filtered_image.CopyFrom(&input_image);
// 	filtered_image.ForwardFFT();
// 	filtered_image.MultiplyPixelWise(kernel);
// //	filtered_image.CosineMask(0.01, 0.02);
// 	filtered_image.BackwardFFT();
// //	filtered_image.MultiplyByConstant(0.3);
// 	input_image.SubtractImage(&filtered_image);
// */
// //	input_image.SetToConstant(1.0);
// //	input_image.CorrectSinc(45.0, 1.0, true, 0.0);
// //	for (i = 0; i < input_image.real_memory_allocated; i++) if (input_image.real_values[i] < 0.0) input_image.real_values[i] = -log(-input_image.real_values[i] + 1.0);
// //	input_image.WriteSlices(&output_file,1,input_image.logical_z_dimension);
// //	temp_float = -420.5; wxPrintf("%g\n", fmodf(temp_float, 360.0));
// //	filtered_image.WriteSlices(&output_file,1,input_image.logical_z_dimension);

// // return true;
// // }
