#ifndef _src_test_gpu_test_gpu_core_headers_h
#define _src_test_gpu_test_gpu_core_headers_h

void    test_complex_add(Complex* a, Complex* b, Complex* output);
void    test_complex_scale(Complex* a, float output);
Complex test_complex_scale(Complex& a, float output);

#endif