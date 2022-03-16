// <class_header>
// <class>DownhillSimplex</class>
//
// <author>Timothy Grant</author>
//
// <date>20061007</date>
//
// <description>
// Downhill simplex function minimisation, this is all stolen from
// numerical recipes.  you have to set the initial_values, call
// minimise function, and the minimised result will be placed in
// minimised values
//
//                              TO USE
//
// You must have a setup DownhillSimplex object, either by constructing
// with the number of required dimensions (default is 1), or by setting
// the number of dimensions and then calling setup.
//
// You must then input your starting values, they are set to 0 as default.
// There are number_of_dimensions squared starting values, or polongs on
// the simplex.  for example if you have 3 dimensions x, y, z
//
// initial_values[0][0] = first x
// initial_values[0][1] = first y
// initial_values[0][2] = first z
//
// initial_values[1][0] = second x
// initial_values[1][1] = second y
// initial_values[1][2] = second z
//
// and so on, and so on.  When these are set, you can call the
// minimise function routine, in which you must specify the function
// to minimise.  You setup this function yourself, and it must take
// a double[number_of_dimensions] input (the values currently being
// evaluated) and return a double (the result of the evaluation).
//
// The function will hopefully return the values in minimised_values
// that lead to the minimum value.  (This is a minimization, if you
// are looking for a maximum, return 1 / your result
// </description>
// </class_header>

class DownhillSimplex {

    ///////////////////////////////////////////////////////////////////////////////
    //    Private stuff, these are the functions stolen from NR, they don't      //
    //    need to be called directly.                                            //
    ///////////////////////////////////////////////////////////////////////////////

  private:
    void   amoeba(double** p, double y[], long ndim, double ftol, double funk(double[]), long* nfunk);
    void   amoeba(double** p, double y[], long ndim, double ftol, void* pt2Object, double (*callback)(void* pt2Object, double[]), long* nfunk);
    double amotry(double** p, double y[], double psum[], long ndim, double funk(double[]), long ihi, double fac);
    double amotry(double** p, double y[], double psum[], long ndim, void* pt2Object, double (*callback)(void* pt2Object, double[]), long ihi, double fac);

    double*  dvector(long nl, long nh);
    double** dmatrix(long nrl, long nrh, long ncl, long nch);

    void free_dvector(double* v, long nl, long nh);
    void free_dmatrix(double** m, long nrl, long nrh, long ncl, long nch);

    double** p; // input values
    double*  y; // evaluated results

  public:
    double** initial_values;
    double*  minimised_values;
    double*  value_scalers;

    long number_of_dimensions;

    wxDateTime time_start;
    wxDateTime time_end;

    double tolerance;
    void   MinimizeFunction(void* pt2Object, double (*callback)(void* pt2Object, double[]));
    void   Setup( );

  public:
    //   Constructors

    DownhillSimplex( );
    DownhillSimplex(long set_number_of_dimensions);

    //   Destructors

    ~DownhillSimplex( );

    //   Functions

    void MinimizeFunction(double function_to_min(double[]));
    void SetIinitalValues(double* intial_values, double* wanted_range);
    void GetMinimizedValues(double output_values[]);

    wxTimeSpan ReturnTimeSpanOfMinimization( ) { return time_end - time_start; }
};
