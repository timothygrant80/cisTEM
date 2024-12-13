#ifndef __SRC_PROGRAMS_QUICK_TEST_QUICK_TEST_GPU_H__
#define __SRC_PROGRAMS_QUICK_TEST_QUICK_TEST_GPU_H__

// Object to be used when quickly testing ideas in quick_test.cpp, with a default empty kernel as example

class QuickTestGPU {
  public:
    QuickTestGPU( ) = default;
    void callHelloFromGPU( );
};
#endif // __SRC_PROGRAMS_QUICK_TEST_QUICK_TEST_GPU_H__