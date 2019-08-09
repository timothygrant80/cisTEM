#include "core_headers.h"

DownhillSimplex::DownhillSimplex()
{

    // assume one dimension

    number_of_dimensions = 3;
    tolerance = 1.0e-8;
    Setup();

}

DownhillSimplex::DownhillSimplex(long set_number_of_dimensions)
{
	MyDebugAssertTrue(set_number_of_dimensions >= 3, "Simplex must have at least 3 dimensions");
     number_of_dimensions = set_number_of_dimensions;
     tolerance = 1.0e-8;
     Setup();
}

DownhillSimplex::~DownhillSimplex()
{

  delete[] minimised_values;
  delete[] value_scalers;

  free_dmatrix(initial_values, 1, number_of_dimensions + 2, 1, number_of_dimensions + 1);

}

void DownhillSimplex::Setup()
{
  minimised_values = new double[number_of_dimensions + 2];
  value_scalers = new double[number_of_dimensions + 2];
  initial_values=dmatrix(1, number_of_dimensions + 2, 1, number_of_dimensions + 1);
}


void DownhillSimplex::GetMinimizedValues(double output_values[])
{
	for( int i=0; i < number_of_dimensions + 1 ; i++)
	{
		output_values[i] = minimised_values[i] / value_scalers[i];
	}
}

void DownhillSimplex::SetIinitalValues(double *wanted_intial_values, double *wanted_range)
{
	int i,j;

	for( i=1; i< number_of_dimensions + 1; i++)
	{
		if (wanted_intial_values[i] == 0) value_scalers[i] = 1.0f;
		else value_scalers[i] = 1 / wanted_intial_values[i];
		initial_values[1][i] = wanted_intial_values[i] * value_scalers[i];
	}

	for( i=1; i< number_of_dimensions + 1; i++)
	{
	  initial_values[2][i] = wanted_intial_values[i] * value_scalers[i] + wanted_range[i] * value_scalers[i];
	}

	for( i=1; i< number_of_dimensions + 1; i++)
	{
	  initial_values[3][i] = wanted_intial_values[i] * value_scalers[i] - wanted_range[i] * value_scalers[i];
	}

	for( i=4; i< number_of_dimensions + 2; i++)
	{
		for( j=1; j< number_of_dimensions + 1; j++)
		{

			if (global_random_number_generator.GetUniformRandom() >= 0)
			{
				initial_values[i][j] = wanted_intial_values[j] * value_scalers[i] + wanted_range[j] * value_scalers[i];
			}
			else
			{
				initial_values[i][j] = wanted_intial_values[j] * value_scalers[i] - wanted_range[j] * value_scalers[i];
			}

		}
	}

}

void DownhillSimplex::amoeba(double **p, double y[], long ndim, double ftol, double funk(double []), long *nfunk)
{
    long i,ihi,ilo,inhi,j,mpts=ndim+1;
    double rtol,sum,swap,ysave,ytry,*psum, *psum_scaled;

    psum=dvector(1,ndim);
    psum_scaled=dvector(1,ndim);

    *nfunk=0;
    for (j=1;j<=ndim;j++)
    {
      for (sum=0.0,i=1;i<=mpts;i++) sum += p[i][j];
      psum[j]=sum;
    }
    for (;;) {
        ilo=1;
        ihi = y[1]>y[2] ? (inhi=2,1) : (inhi=1,2);
        for (i=1;i<=mpts;i++) {
           if (y[i] <= y[ilo]) ilo=i;
           if (y[i] > y[ihi]) {
              inhi=ihi;
              ihi=i;
	    } else if (y[i] > y[inhi] && i != ihi) inhi=i;
           }
	rtol=2.0*fabs(y[ihi]-y[ilo])/(fabs(y[ihi])+fabs(y[ilo])+1.0e-10);
	if (rtol < ftol) {
	        {swap=y[1];y[1]=y[ilo];y[ilo]=swap;}
		for (i=1;i<=ndim;i++) {swap=p[1][i];p[1][i]=p[ilo][i];p[ilo][i]=swap;}
		 break;
	}
	*nfunk +=2;
	ytry=amotry(p,y,psum,ndim,funk,ihi,-1.0);
	if (ytry<=y[ilo])
	   ytry=amotry(p,y,psum,ndim,funk,ihi,2.0);///////////////////
	else if (ytry >= y[inhi]) {
	     ysave=y[ihi];
	     ytry=amotry(p,y,psum,ndim,funk,ihi,0.5);
	     if (ytry >= ysave) {
		for (i=1;i<=mpts;i++) {
		    if (i != ilo) {
			for (j=1;j<=ndim;j++)
			{
			    p[i][j]=psum[j]=0.5*(p[i][j]+p[ilo][j]);
			    psum_scaled[j] = psum[j] / value_scalers[j];
			}

			y[i]=funk(psum_scaled);
		    }
		}
		*nfunk += ndim;
		 for (j=1;j<=ndim;j++)
                 {
		   for (sum=0.0,i=1;i<=mpts;i++) sum += p[i][j];
                   psum[j]=sum;
                 }

	      }
	    } else --(*nfunk);
	}

	free_dvector(psum,1,ndim);
	free_dvector(psum_scaled,1,ndim);
}

void DownhillSimplex::amoeba(double **p, double y[], long ndim, double ftol, void *pt2Object, double (*callback)(void* pt2Object, double []), long *nfunk)
{
    long i,ihi,ilo,inhi,j,mpts=ndim+1;
    double rtol,sum,swap,ysave,ytry,*psum, *psum_scaled;

    psum=dvector(1,ndim);
    psum_scaled=dvector(1,ndim);
    *nfunk=0;
    for (j=1;j<=ndim;j++)
    {
      for (sum=0.0,i=1;i<=mpts;i++) sum += p[i][j];
      psum[j]=sum;
    }
    for (;;) {
        ilo=1;
        ihi = y[1]>y[2] ? (inhi=2,1) : (inhi=1,2);
        for (i=1;i<=mpts;i++) {
           if (y[i] <= y[ilo]) ilo=i;
           if (y[i] > y[ihi]) {
              inhi=ihi;
              ihi=i;
	    } else if (y[i] > y[inhi] && i != ihi) inhi=i;
           }
	rtol=2.0*fabs(y[ihi]-y[ilo])/(fabs(y[ihi])+fabs(y[ilo])+1.0e-10);
	if (rtol < ftol) {
	        {swap=y[1];y[1]=y[ilo];y[ilo]=swap;}
		for (i=1;i<=ndim;i++) {swap=p[1][i];p[1][i]=p[ilo][i];p[ilo][i]=swap;}
		 break;
	}
	*nfunk +=2;
	ytry=amotry(p,y,psum,ndim,pt2Object,callback,ihi,-1.0);
	if (ytry<=y[ilo])
	   ytry=amotry(p,y,psum,ndim,pt2Object,callback,ihi,2.0);///////////////////
	else if (ytry >= y[inhi]) {
	     ysave=y[ihi];
	     ytry=amotry(p,y,psum,ndim,pt2Object,callback,ihi,0.5);
	     if (ytry >= ysave) {
		for (i=1;i<=mpts;i++) {
		    if (i != ilo) {
			for (j=1;j<=ndim;j++)
			{
			    p[i][j]=psum[j]=0.5*(p[i][j]+p[ilo][j]);
				psum_scaled[j] = psum[j] / value_scalers[j];
			}
			y[i]=callback(pt2Object,psum_scaled);
		    }
		}
		*nfunk += ndim;
		 for (j=1;j<=ndim;j++)
                 {
		   for (sum=0.0,i=1;i<=mpts;i++) sum += p[i][j];
                   psum[j]=sum;
                 }

	      }
	    } else --(*nfunk);
	}

	free_dvector(psum,1,ndim);
	free_dvector(psum_scaled,1,ndim);
}

double DownhillSimplex::amotry(double **p,double y[], double psum[], long ndim, double funk(double []), long ihi, double fac)
{
    long j;
    double fac1,fac2,ytry,*ptry,*ptry_scaled;

    ptry=dvector(1,ndim);
    ptry_scaled=dvector(1, ndim);

    fac1=(1.0-fac)/ndim;
    fac2=fac1-fac;
    for (j=1;j<=ndim;j++) ptry[j]=psum[j]*fac1-p[ihi][j]*fac2;
    for (j=1;j<=ndim;j++) ptry_scaled[j] = ptry[j] / value_scalers[j];
    ytry=funk(ptry_scaled);
    if (ytry < y[ihi]) {
	y[ihi]=ytry;
	for (j=1;j<=ndim;j++) {
		psum[j] +=ptry[j]-p[ihi][j];
		p[ihi][j]=ptry[j];
	    }
	}
	free_dvector(ptry,1,ndim);
	free_dvector(ptry_scaled,1,ndim);
//prlongf("f=%lf,",ytry);
	return ytry;
}

double DownhillSimplex::amotry(double **p,double y[], double psum[], long ndim, void *pt2Object, double (*callback)(void* pt2Object, double []), long ihi, double fac)
{
    long j;
    double fac1,fac2,ytry,*ptry,*ptry_scaled;

    ptry=dvector(1,ndim);
    ptry_scaled=dvector(1,ndim);

    fac1=(1.0-fac)/ndim;
    fac2=fac1-fac;
    for (j=1;j<=ndim;j++) ptry[j]=psum[j]*fac1-p[ihi][j]*fac2;
    for (j=1;j<=ndim;j++) ptry_scaled[j]=ptry[j] / value_scalers[j];
    ytry=callback(pt2Object,ptry_scaled);
    if (ytry < y[ihi]) {
	y[ihi]=ytry;
	for (j=1;j<=ndim;j++) {
		psum[j] +=ptry[j]-p[ihi][j];
		p[ihi][j]=ptry[j];
	    }
	}
	free_dvector(ptry,1,ndim);
	free_dvector(ptry_scaled,1,ndim);
//prlongf("f=%lf,",ytry);
	return ytry;
}
void DownhillSimplex::MinimizeFunction(double function_to_min(double []))
{
	time_start = wxDateTime::Now();
  long nfunk = 0; // hold the number of evaluations
  double *scaled_values = new double[number_of_dimensions + 1];

  y = dvector(1, number_of_dimensions + 2);
  p = dmatrix(1, number_of_dimensions + 2, 1, number_of_dimensions +1);

  // fill p... blank minimised values

  for(long i=1; i<= number_of_dimensions+1 ; i++)
  {
    for (long i2 = 1; i2 <= number_of_dimensions +1; i2++)
    {
      p[i][i2] = initial_values[i][i2];
    }

    minimised_values[i] = 0;
  }

  // evaluate the starting values...

  for(long i=1; i<= number_of_dimensions+1 ; i++)
  {
	  for (int j = 1; j < number_of_dimensions + 1; j++)
	  {
		  scaled_values[j] = p[i][j] / value_scalers[j];
	  }
    y[i] = function_to_min(scaled_values);
  }

  // do the minimisation

  amoeba(p, y, number_of_dimensions, tolerance, function_to_min, &nfunk);

  // do another (should be cheap)

  amoeba(p, y, number_of_dimensions, tolerance, function_to_min, &nfunk);

  // return the best values (middle of the simplex)

  for(int i=1; i< number_of_dimensions; i++)
  {
    minimised_values[i] = p[1][i];
  }
  time_end = wxDateTime::Now();
  delete [] scaled_values;
}


void DownhillSimplex::MinimizeFunction(void *pt2Object, double (*callback)(void* pt2Object, double []))
{
	time_start = wxDateTime::Now();
  long nfunk = 0; // hold the number of evaluations
  double *scaled_values = new double[number_of_dimensions + 1];

  y = dvector(1, number_of_dimensions + 2);
  p = dmatrix(1, number_of_dimensions + 2, 1, number_of_dimensions +1);

  // fill p... blank minimised values

  for(long i=1; i<= number_of_dimensions+1 ; i++)
  {
    for (long i2 = 1; i2 <= number_of_dimensions +1; i2++)
    {
      p[i][i2] = initial_values[i][i2];
    }

    minimised_values[i] = 0;
  }

  // evaluate the starting values...

  for(long i=1; i<= number_of_dimensions+1 ; i++)
  {
	  for (int j = 1; j < number_of_dimensions + 1; j++)
	  {
		 scaled_values[j] = p[i][j] / value_scalers[j];
	  }

	  y[i] = callback(pt2Object, scaled_values);

  }

  // do the minimisation

  amoeba(p, y, number_of_dimensions, tolerance, pt2Object, callback, &nfunk);

  // do another (should be cheap)

  amoeba(p, y, number_of_dimensions, tolerance, pt2Object, callback, &nfunk);

  // return the best values (middle of the simplex)

  for(long i=1; i<= number_of_dimensions; i++)
  {
    minimised_values[i] = p[1][i];
  }

  time_end = wxDateTime::Now();
  delete [] scaled_values;

}

double *DownhillSimplex::dvector(long nl,long nh)
{
	double *v;

	v=(double *)malloc((unsigned) (nh-nl+1)*sizeof(double));
	if (!v) {
		MyPrintWithDetails("Allocation failure in dvector()\n");
		DEBUG_ABORT;
	}
	return v-nl;
}

double **DownhillSimplex::dmatrix(long nrl, long nrh, long ncl, long nch)
{
        long i;
	double **m;

	m=(double **) malloc((unsigned) (nrh-nrl+1)*sizeof(double*));
	if (!m) {
		MyPrintWithDetails("Allocation failure in dmatrix()\n");
		DEBUG_ABORT;
	}
	m -= nrl;

	for(i=nrl;i<=nrh;i++) {
		m[i]=(double *) malloc((unsigned) (nch-ncl+1)*sizeof(double));
		if (!m[i]) {
			MyPrintWithDetails("Allocation failure in dmatrix()\n");
			DEBUG_ABORT;
		}
		m[i] -= ncl;
	}
	return m;
}

void DownhillSimplex::free_dvector(double *v, long nl, long nh)
{
	free((char*) (v+nl));
}

void DownhillSimplex::free_dmatrix(double **m, long nrl, long nrh, long ncl, long nch)
{
	long i;

	for(i=nrh;i>=nrl;i--) free((char*) (m[i]+ncl));
	free((char*) (m+nrl));
}

