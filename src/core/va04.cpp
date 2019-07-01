/* va04_ctffind.f -- translated by f2c (version 20100827).
   You must link the resulting object file with libf2c:
	on Microsoft Windows system, link with libf2c.lib;
	on Linux or Unix systems, link with .../path/to/libf2c.a -lm
	or, if you install libf2c.a in a standard place, with -lf2c -lm
	-- in that order, at the end of the command line, as in
		cc *.o -lf2c -lm
	Source for libf2c is in /netlib/f2c/libf2c.zip, e.g.,

		http://www.netlib.org/f2c/libf2c.zip
*/
#include <wx/wx.h>
#ifdef __cplusplus
extern "C" {
#endif
#include "f2c.h"


/* Table of constant values */

static int c__1 = 1;


/* ****************************************************************************** */

/* Subroutine */ int calcfx_(int *nx, float *xpar, float *rf, float *ain,
	float *cs, float *wl, float *wgh1, float *wgh2, float *thetatr, float *
	rmin2, float *rmax2, int *nxyz, float *hw, float *dast)
{

/*     CALCULATES NEW VALUE FOR F TO INPUT TO SUBROUTINE VA04A */

/* ****************************************************************************** */



/*      RF=-EVALCTF(CS,WL,WGH1,WGH2,XPAR(1),XPAR(2),XPAR(3), */
/*     +		  THETATR,HW,AIN,NXYZ,RMIN2,RMAX2,DAST) */

    /* Parameter adjustments */
    --nxyz;
    --ain;
    --xpar;

    /* Function Body */
    *rf = (float)-1.;
    return 0;
} /* calcfx_ */

/* ************************************************************************** */
///* Subroutine */ int va04a_(real *x, real *e, integer *n, real *f, real *
//	escale, integer *iprint, integer *icon, integer *maxit, real *ain,
//	real *cs, real *wl, real *wgh1, real *wgh2, real *thetatr, real *
//	rmin2, real *rmax2, integer *nxyz, real *hw, real *dast)
//{

//int va04a_(integer *n, real *e, real *escale, int *num_function_calls, float (*target_function) (void* parameters, float[]), void *parameters, real *f,
//			integer *iprint, integer *icon, integer *maxit, real *x)
int va04a_(int *n, float *e, float *escale, int *num_function_calls, float (*target_function) (void* parameters, float[]), void *parameters, float *f,
			int *iprint, int *icon, int *maxit, float *x)

{

    /* Format strings */
    char fmt_19[] = "(5x,\002VA04A MAXIMUM CHANGE DOES NOT ALTER FUNCTION\002)";
    char fmt_52[] = "(/1x,\002ITERATION\002,i5,i15,\002 FUNCTION VALUES\002,10x,\002F =\002,e21.14/(5e24.14))";
    char fmt_80[] = "(5x,\002VA04A ACCURACY LIMITED BY ERRORS IN F\002)";

    /* System generated locals */
    int i__1, i__2, maxx = 100 * *maxit;
    double r__1, r__2;

    /* Builtin functions */
    int s_wsfe(cilist *), e_wsfe();
//    double r_sign(real *, real *), sqrt(doublereal);
    int do_fio(int *, char *, ftnlen);

    /* Local variables */
    int icnt = 0;
    double a, b, d__, xs[11];
    int i__, j, k;
// work array w must have at least dimension n * (n + 3)
    double w[154], da, db, fa, dd, fb, fc, dc, di, fi, dl;
    int jj;
    double fp;
    int is;
    double aaa;
    int ind, jjj, jil, inn, ixp;
    double tmp, sum, dacc, dmag;
    int nfcc;
    double dmax__, scer, ddmag, fkeep, fhold, ddmax;
    int iline, idirn, iterc, itone;
    double fprev;
    extern /* Subroutine */ int calcfx_(int *, float *, float *, float *,
	    float *, float *, float *, float *, float *, float *, float *, int *,
	     float *, float *);
    int isgrad;

    /* Fortran I/O blocks */
    cilist io___32 = { 0, 6, 0, fmt_19, 0 };
    cilist io___42 = { 0, 6, 0, fmt_52, 0 };
    cilist io___45 = { 0, 6, 0, fmt_80, 0 };


/* ************************************************************************** */
/*  STANDARD FORTRAN 66 (A VERIFIED PFORT SUBROUTINE) */
/* 	W[N*(N+3)] */
    /* Parameter adjustments */
    //--nxyz;
    //--ain;
    --e;
    --x;

    /* Function Body */
    ddmag = *escale * (float).1;
    scer = (float).05 / *escale;
    jj = *n * *n + *n;
    jjj = jj + *n;
    k = *n + 1;
    nfcc = 1;
    ind = 1;
    inn = 1;
    i__1 = *n;
    for (i__ = 1; i__ <= i__1; i__++) {xs[i__] = x[i__];};
    for (i__ = 1; i__ <= i__1; ++i__) {
	i__2 = *n;
	for (j = 1; j <= i__2; ++j) {
	    w[k - 1] = (float)0.;
	    if (i__ - j != 0) {
		goto L4;
	    } else {
		goto L3;
	    }
L3:
	    w[k - 1] = (r__1 = e[i__], dabs(r__1));
	    w[i__ - 1] = *escale;
L4:
	    ++k;
/* L2: */
	}
/* L1: */
    }
    iterc = 1;
    isgrad = 2;
    //calcfx_(n, &x[1], f, &ain[1], cs, wl, wgh1, wgh2, thetatr, rmin2, rmax2, &
	//    nxyz[1], hw, dast);
    icnt++;
	if (icnt > maxx) goto L999;
    *f = target_function(parameters, x + 1);
    fkeep = dabs(*f) + dabs(*f);
L5:
    itone = 1;
    fp = *f;
    sum = (float)0.;
    ixp = jj;
    i__1 = *n;
    for (i__ = 1; i__ <= i__1; ++i__) {
	++ixp;
	w[ixp - 1] = x[i__];
/* L6: */
    }
    idirn = *n + 1;
    iline = 1;
L7:
    dmax__ = w[iline - 1];
    dacc = dmax__ * scer;
/* Computing MIN */
    r__1 = ddmag, r__2 = dmax__ * (float).1;
    dmag = dmin(r__1,r__2);
/* Computing MAX */
    r__1 = dmag, r__2 = dacc * (float)20.;
    dmag = dmax(r__1,r__2);
    ddmax = dmag * (float)10.;
    switch (itone) {
	case 1:  goto L70;
	case 2:  goto L70;
	case 3:  goto L71;
    }
L70:
    dl = (float)0.;
    d__ = dmag;
    fprev = *f;
    is = 5;
    fa = *f;
    da = dl;
L8:
    dd = d__ - dl;
    dl = d__;
L58:
    k = idirn;
    i__1 = *n;
    for (i__ = 1; i__ <= i__1; ++i__) {
	x[i__] += dd * w[k - 1];
	++k;
/* L9: */
    }
    //calcfx_(n, &x[1], f, &ain[1], cs, wl, wgh1, wgh2, thetatr, rmin2, rmax2, &
	//    nxyz[1], hw, dast);
    icnt++;
	if (icnt > maxx) goto L999;
    *f = target_function(parameters, x + 1);

    ++nfcc;
    switch (is) {
	case 1:  goto L10;
	case 2:  goto L11;
	case 3:  goto L12;
	case 4:  goto L13;
	case 5:  goto L14;
	case 6:  goto L96;
    }
L14:
    if ((r__1 = *f - fa) < (float)0.) {
	goto L15;
    } else if (r__1 == 0) {
	goto L16;
    } else {
	goto L24;
    }
L16:
    if (dabs(d__) - dmax__ <= (float)0.) {
	goto L17;
    } else {
	goto L18;
    }
L17:
    d__ += d__;
    goto L8;
L18:
    //s_wsfe(&io___32);
    //e_wsfe();
	// Removed this statement since it does not seem to be very informative
    // wxPrintf("Warning(VA04): maximum change does not alter target function\n");
    goto L20;
L15:
    fb = *f;
    db = d__;
    goto L21;
L24:
    fb = fa;
    db = da;
    fa = *f;
    da = d__;
L21:
    switch (isgrad) {
	case 1:  goto L83;
	case 2:  goto L23;
    }
L23:
    d__ = db + db - da;
    is = 1;
    goto L8;
L83:
    d__ = (da + db - (fa - fb) / (da - db)) * (float).5;
    is = 4;
    if ((da - d__) * (d__ - db) >= (float)0.) {
	goto L8;
    } else {
	goto L25;
    }
L25:
    is = 1;
    if ((r__1 = d__ - db, dabs(r__1)) - ddmax <= (float)0.) {
	goto L8;
    } else {
	goto L26;
    }
L26:
    r__1 = db - da;
    d__ = db + copysign(ddmax, r__1); //d__ = db + r_sign(&ddmax, &r__1);
    is = 1;
    ddmax += ddmax;
    ddmag += ddmag;
    if (ddmax - dmax__ <= (float)0.) {
	goto L8;
    } else {
	goto L27;
    }
L27:
    ddmax = dmax__;
    goto L8;
L13:
    if (*f - fa >= (float)0.) {
	goto L23;
    } else {
	goto L28;
    }
L28:
    fc = fb;
    dc = db;
L29:
    fb = *f;
    db = d__;
    goto L30;
L12:
    if (*f - fb <= (float)0.) {
	goto L28;
    } else {
	goto L31;
    }
L31:
    fa = *f;
    da = d__;
    goto L30;
L11:
    if (*f - fb >= (float)0.) {
	goto L10;
    } else {
	goto L32;
    }
L32:
    fa = fb;
    da = db;
    goto L29;
L71:
    dl = (float)1.;
    ddmax = (float)5.;
    fa = fp;
    da = (float)-1.;
    fb = fhold;
    db = (float)0.;
    d__ = (float)1.;
L10:
    fc = *f;
    dc = d__;
L30:
    a = (db - dc) * (fa - fc);
    b = (dc - da) * (fb - fc);
    if ((a + b) * (da - dc) <= (float)0.) {
	goto L33;
    } else {
	goto L34;
    }
L33:
    fa = fb;
    da = db;
    fb = fc;
    db = dc;
    goto L26;
L34:
    d__ = (a * (db + dc) + b * (da + dc)) * (float).5 / (a + b);
    di = db;
    fi = fb;
    if (fb - fc <= (float)0.) {
	goto L44;
    } else {
	goto L43;
    }
L43:
    di = dc;
    fi = fc;
L44:
    switch (itone) {
	case 1:  goto L86;
	case 2:  goto L86;
	case 3:  goto L85;
    }
L85:
    itone = 2;
    goto L45;
L86:
    if ((r__1 = d__ - di, dabs(r__1)) - dacc <= (float)0.) {
	goto L41;
    } else {
	goto L93;
    }
L93:
    if ((r__1 = d__ - di, dabs(r__1)) - dabs(d__) * (float).03 <= (float)0.) {
	goto L41;
    } else {
	goto L45;
    }
L45:
    if ((da - dc) * (dc - d__) >= (float)0.) {
	goto L46;
    } else {
	goto L47;
    }
L46:
    fa = fb;
    da = db;
    fb = fc;
    db = dc;
    goto L25;
L47:
    is = 2;
    if ((db - d__) * (d__ - dc) >= (float)0.) {
	goto L8;
    } else {
	goto L48;
    }
L48:
    is = 3;
    goto L8;
L41:
    *f = fi;
    d__ = di - dl;
    dd = sqrt((dc - db) * (dc - da) * (da - db) / (a + b));
    i__1 = *n;
    for (i__ = 1; i__ <= i__1; ++i__) {
	x[i__] += d__ * w[idirn - 1];
	w[idirn - 1] = dd * w[idirn - 1];
	++idirn;
/* L49: */
    }
    if (dd == (float)0.) {
	dd = (float)1e-10;
    }
    w[iline - 1] /= dd;
    ++iline;
    if (*iprint - 1 != 0) {
	goto L51;
    } else {
	goto L50;
    }
L50:
    switch (*iprint) {
	case 1:  goto L51;
	case 2:  goto L53;
    }
L51:
    switch (itone) {
	case 1:  goto L55;
	case 2:  goto L38;
    }
L55:
    if (fprev - *f - sum >= (float)0.) {
	goto L95;
    } else {
	goto L94;
    }
L95:
    sum = fprev - *f;
    jil = iline;
L94:
    if (idirn - jj <= 0) {
	goto L7;
    } else {
	goto L84;
    }
L84:
    switch (ind) {
	case 1:  goto L92;
	case 2:  goto L72;
    }
L92:
    fhold = *f;
    is = 6;
    ixp = jj;
    i__1 = *n;
    for (i__ = 1; i__ <= i__1; ++i__) {
	++ixp;
	w[ixp - 1] = x[i__] - w[ixp - 1];
/* L59: */
    }
    dd = (float)1.;
    goto L58;
L96:
    switch (ind) {
	case 1:  goto L112;
	case 2:  goto L87;
    }
L112:
    if (fp - *f <= (float)0.) {
	goto L37;
    } else {
	goto L91;
    }
L91:
/* Computing 2nd power */
    r__1 = fp - *f;
    d__ = (fp + *f - fhold * (float)2.) * (float)2. / (r__1 * r__1);
/* Computing 2nd power */
    r__1 = fp - fhold - sum;
    if (d__ * (r__1 * r__1) - sum >= (float)0.) {
	goto L37;
    } else {
	goto L87;
    }
L87:
    j = jil * *n + 1;
    if (j - jj <= 0) {
	goto L60;
    } else {
	goto L61;
    }
L60:
    i__1 = jj;
    for (i__ = j; i__ <= i__1; ++i__) {
	k = i__ - *n;
	w[k - 1] = w[i__ - 1];
/* L62: */
    }
    i__1 = *n;
    for (i__ = jil; i__ <= i__1; ++i__) {
	w[i__ - 2] = w[i__ - 1];
/* L97: */
    }
L61:
    idirn -= *n;
    itone = 3;
    k = idirn;
    ixp = jj;
    aaa = (float)0.;
    i__1 = *n;
    for (i__ = 1; i__ <= i__1; ++i__) {
	++ixp;
	w[k - 1] = w[ixp - 1];
	if (aaa - (r__1 = w[k - 1] / e[i__], dabs(r__1)) >= (float)0.) {
	    goto L67;
	} else {
	    goto L66;
	}
L66:
	aaa = (r__1 = w[k - 1] / e[i__], dabs(r__1));
L67:
	++k;
/* L65: */
    }
    ddmag = (float)1.;
    if (aaa == (float)0.) {
	aaa = (float)1e-10;
    }
    w[*n - 1] = *escale / aaa;
    iline = *n;
    goto L7;
L37:
    ixp = jj;
    aaa = (float)0.;
    *f = fhold;
    i__1 = *n;
    for (i__ = 1; i__ <= i__1; ++i__) {
	++ixp;
	x[i__] -= w[ixp - 1];
	if (aaa * (r__1 = e[i__], dabs(r__1)) - (r__2 = w[ixp - 1], dabs(r__2)
		) >= (float)0.) {
	    goto L99;
	} else {
	    goto L98;
	}
L98:
	aaa = (r__1 = w[ixp - 1] / e[i__], dabs(r__1));
L99:
	;
    }
    goto L72;
L38:
    aaa *= di + (float)1.;
    switch (ind) {
	case 1:  goto L72;
	case 2:  goto L106;
    }
L72:
    if (*iprint - 2 >= 0) {
	goto L50;
    } else {
	goto L53;
    }
L53:
    switch (ind) {
	case 1:  goto L109;
	case 2:  goto L88;
    }
L109:
    if (aaa - (float).1 <= (float)0.) {
	goto L89;
    } else {
	goto L76;
    }
L89:
    switch (*icon) {
	case 1:  goto L20;
	case 2:  goto L116;
    }
L116:
    ind = 2;
    switch (inn) {
	case 1:  goto L100;
	case 2:  goto L101;
    }
L100:
    inn = 2;
    k = jjj;
    i__1 = *n;
    for (i__ = 1; i__ <= i__1; ++i__) {
	++k;
	w[k - 1] = x[i__];
	x[i__] += e[i__] * (float)10.;
/* L102: */
    }
    fkeep = *f;
    //calcfx_(n, &x[1], f, &ain[1], cs, wl, wgh1, wgh2, thetatr, rmin2, rmax2, &
	//    nxyz[1], hw, dast);
    icnt++;
	if (icnt > maxx) goto L999;
    *f = target_function(parameters, x + 1);
    ++nfcc;
    ddmag = (float)0.;
    goto L108;
L76:
    if (*f - fp >= (float)0.) {
	goto L78;
    } else {
	goto L35;
    }
L78:
    //s_wsfe(&io___45);
    //e_wsfe();
	// Removed this statement since it does not seem to be very informative
    // wxPrintf("Warning(VA04): accuracy limited by errors in target function\n");
    goto L20;
L88:
    ind = 1;
L35:
    tmp = fp - *f;
    if (tmp > (float)0.) {
	ddmag = sqrt(tmp) * (float).4;
    } else {
	ddmag = (float)0.;
    }
    isgrad = 1;
L108:
    ++iterc;
    if (iterc - *maxit <= 0) {
	goto L5;
    } else {
	goto L81;
    }
L81:
/*   81 WRITE(6,82) MAXIT */
/* L82: */
    if (*f - fkeep <= (float)0.) {
	goto L20;
    } else {
	goto L110;
    }
L110:
    *f = fkeep;
    i__1 = *n;
    for (i__ = 1; i__ <= i__1; ++i__) {
	++jjj;
	x[i__] = w[jjj - 1];
/* L111: */
    }
    goto L20;
L101:
    jil = 1;
    fp = fkeep;
    if ((r__1 = *f - fkeep) < (float)0.) {
	goto L105;
    } else if (r__1 == 0) {
	goto L78;
    } else {
	goto L104;
    }
L104:
    jil = 2;
    fp = *f;
    *f = fkeep;
L105:
    ixp = jj;
    i__1 = *n;
    for (i__ = 1; i__ <= i__1; ++i__) {
	++ixp;
	k = ixp + *n;
	switch (jil) {
	    case 1:  goto L114;
	    case 2:  goto L115;
	}
L114:
	w[ixp - 1] = w[k - 1];
	goto L113;
L115:
	w[ixp - 1] = x[i__];
	x[i__] = w[k - 1];
L113:
	;
    }
    jil = 2;
    goto L92;
L106:
    if (aaa - (float).1 <= (float)0.) {
	goto L20;
    } else {
	goto L107;
    }
L20:
    return 0;
L107:
    inn = 1;
    goto L35;
L999:
	for (i__ = 1; i__ <= i__1; i__++) {x[i__] = xs[i__];};
    *f = target_function(parameters, x + 1);
	// Removed this statement since it does not seem to be very informative
    // wxPrintf("Warning(VA04): Endless loop safety catch, icnt = %i\n", icnt);
    return 0;
} /* va04a_ */

#ifdef __cplusplus
	}
#endif
