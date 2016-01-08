#include "core_headers.h"

DownhillSimplex::DownhillSimplex()
{

    // assume one dimension

    number_of_dimensions = 1;
    tolerance = 1.0e-5;
    Setup();

}

DownhillSimplex::DownhillSimplex(long set_number_of_dimensions)
{
     number_of_dimensions = set_number_of_dimensions;
     tolerance = 1.0e-5;
     Setup();
}

DownhillSimplex::~DownhillSimplex()
{

  delete[] minimised_values;
  free_dmatrix(initial_values, 1, number_of_dimensions + 2, 1, number_of_dimensions + 1);

}

void DownhillSimplex::Setup()
{
  minimised_values = new double[number_of_dimensions + 2];
  initial_values=dmatrix(1, number_of_dimensions + 2, 1, number_of_dimensions + 1);
}



void DownhillSimplex::amoeba(double **p, double y[], long ndim, double ftol, double funk(double []), long *nfunk)
{
    long i,ihi,ilo,inhi,j,mpts=ndim+1;
    double rtol,sum,swap,ysave,ytry,*psum;

    psum=dvector(1,ndim);
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
			    p[i][j]=psum[j]=0.5*(p[i][j]+p[ilo][j]);
			y[i]=funk(psum);
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
}

void DownhillSimplex::amoeba(double **p, double y[], long ndim, double ftol, void *pt2Object, double (*callback)(void* pt2Object, double []), long *nfunk)
{
    long i,ihi,ilo,inhi,j,mpts=ndim+1;
    double rtol,sum,swap,ysave,ytry,*psum;

    psum=dvector(1,ndim);
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
			    p[i][j]=psum[j]=0.5*(p[i][j]+p[ilo][j]);
			y[i]=callback(pt2Object,psum);
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
}

double DownhillSimplex::amotry(double **p,double y[], double psum[], long ndim, double funk(double []), long ihi, double fac)
{
    long j;
    double fac1,fac2,ytry,*ptry;

    ptry=dvector(1,ndim);
    fac1=(1.0-fac)/ndim;
    fac2=fac1-fac;
    for (j=1;j<=ndim;j++) ptry[j]=psum[j]*fac1-p[ihi][j]*fac2;
    ytry=funk(ptry);
    if (ytry < y[ihi]) {
	y[ihi]=ytry;
	for (j=1;j<=ndim;j++) {
		psum[j] +=ptry[j]-p[ihi][j];
		p[ihi][j]=ptry[j];
	    }
	}
	free_dvector(ptry,1,ndim);
//prlongf("f=%lf,",ytry);
	return ytry;
}

double DownhillSimplex::amotry(double **p,double y[], double psum[], long ndim, void *pt2Object, double (*callback)(void* pt2Object, double []), long ihi, double fac)
{
    long j;
    double fac1,fac2,ytry,*ptry;

    ptry=dvector(1,ndim);
    fac1=(1.0-fac)/ndim;
    fac2=fac1-fac;
    for (j=1;j<=ndim;j++) ptry[j]=psum[j]*fac1-p[ihi][j]*fac2;
    ytry=callback(pt2Object,ptry);
    if (ytry < y[ihi]) {
	y[ihi]=ytry;
	for (j=1;j<=ndim;j++) {
		psum[j] +=ptry[j]-p[ihi][j];
		p[ihi][j]=ptry[j];
	    }
	}
	free_dvector(ptry,1,ndim);
//prlongf("f=%lf,",ytry);
	return ytry;
}
void DownhillSimplex::minimise_function(double function_to_min(double []))
{

  long nfunk = 0; // hold the number of evaluations

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
    y[i] = function_to_min(p[i]);
  }

  // do the minimisation

  amoeba(p, y, number_of_dimensions, tolerance, function_to_min, &nfunk);

  // do another (should be cheap)

  amoeba(p, y, number_of_dimensions, tolerance, function_to_min, &nfunk);

  // return the best values (middle of the simplex)

  for(long i=1; i<= number_of_dimensions; i++)
  {
    minimised_values[i] = p[1][i];
  }


}


void DownhillSimplex::minimise_function(void *pt2Object, double (*callback)(void* pt2Object, double []))
{
  long nfunk = 0; // hold the number of evaluations

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
    y[i] = callback(pt2Object, p[i]);
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

}

double *DownhillSimplex::dvector(long nl,long nh)
{
	double *v;

	v=(double *)malloc((unsigned) (nh-nl+1)*sizeof(double));
	if (!v) {
		MyPrintWithDetails("Allocation failure in dvector()\n");
		abort();
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
		abort();
	}
	m -= nrl;

	for(i=nrl;i<=nrh;i++) {
		m[i]=(double *) malloc((unsigned) (nch-ncl+1)*sizeof(double));
		if (!m[i]) {
			MyPrintWithDetails("Allocation failure in dmatrix()\n");
			abort();
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

