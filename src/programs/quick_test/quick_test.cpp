#include <cistem_config.h>

#include "../../core/core_headers.h"
#include "../../constants/constants.h"

#ifdef ENABLEGPU
#include "../../gpu/gpu_core_headers.h"
#include "../../gpu/DeviceManager.h"
#include "../../gpu/TemplateMatchingCore.h"
#include "quick_test_gpu.h"
#else
#include "../../core/core_headers.h"
#endif

#include "../../constants/constants.h"

#include "../../core/scattering_potential.h"

class
        QuickTestApp : public MyApp {

  public:
    bool     DoCalculation( );
    void     DoInteractiveUserInput( );
    wxString symmetry_symbol;
    bool     my_test_1 = false;
    bool     my_test_2 = true;

    std::array<wxString, 2> input_starfile_filename;

  private:
};

IMPLEMENT_APP(QuickTestApp)

// override the DoInteractiveUserInput

void QuickTestApp::DoInteractiveUserInput( ) {

    UserInput* my_input = new UserInput("Unblur", 2.0);

    input_starfile_filename.at(0) = my_input->GetFilenameFromUser("Input starfile filename 1", "", "", false);
    input_starfile_filename.at(1) = my_input->GetFilenameFromUser("Input starfile filename 2", "", "", false);
    symmetry_symbol               = my_input->GetSymmetryFromUser("Particle symmetry", "The assumed symmetry of the particle to be reconstructed", "C1");

    delete my_input;
}

unsigned int GetTransformSize(int kernel_type) {
    return kernel_type * 1;
}

// override the do calculation method which will be what is actually run..
template <int FFT_ALGO_t, unsigned int SizeValue>
void SelectSizeAndTypeWithFold(int kernel_type) {

    // transform_size.P is set when folding over Q values
    unsigned int P = GetTransformSize(kernel_type);
    // Note: the size of the input/output may not match the size of the transform, i.e. transform_size.L <= transform_size.P
    if ( SizeValue == P ) {
        std::cerr << "SizeValue == " << SizeValue << std::endl;
    }

    if constexpr ( ! check_pow2(SizeValue) ) {
        std::cerr << "SizeValue must be a power of 2" << std::endl;
    }
    static_assert(check_pow2_func<SizeValue>( ), "SizeValue must be a power of 2");
}

template <int FFT_ALGO_t, unsigned int... SizeValues>
void SelectSizeAndType(int kernel_type) {
    (SelectSizeAndTypeWithFold<FFT_ALGO_t, SizeValues>(kernel_type), ...);
}

#define MY_INTS_TO_LOOP 64, 128, 256, 1024, 2048, 4096

// template <int FFT_ALGO_t>
// void set_val(int kernel_type) {

//     SelectSizeAndType<FFT_ALGO_t, MY_INTS_TO_LOOP>(kernel_type);
// }

constexpr bool check_pow2(int n) {
    return n == 1 ? true : n % 2 == 0 ? check_pow2(n / 2)
                                      : false;
}

template <bool T, unsigned int n>
EnableIf<T> checkME( ) {
    return;
}

template <unsigned int n>
void check_pow2_func( ) {
    checkME<check_pow2(n), n>( );
}

template <unsigned int... SizeValues>
void MySimpleCheck( ) {
    (check_pow2_func<SizeValues>, ...);
}

template <>
void MySimpleCheck<0>( ) {
    MySimpleCheck<MY_INTS_TO_LOOP>( );
}

bool QuickTestApp::DoCalculation( ) {

#ifdef ENABLEGPU
    // DeviceManager gpuDev;
    // gpuDev.ListDevices( );

    // QuickTestGPU quick_test_gpu;
    // quick_test_gpu.callHelloFromGPU( );
#endif
    constexpr int FFT_ALGO_t = 0;

    std::vector<int> kernel_types = {64, 1, 3, 512, 0, 2};

    std::initializer_list<int> my_checker = {MY_INTS_TO_LOOP};

    for ( auto& val : my_checker ) {
        std::cerr << val << std::endl;
    }
    // MySimpleCheck<MY_INTS_TO_LOOP>( );
    // for ( auto kernel_type : kernel_types ) {
    //     set_val<FFT_ALGO_t>(kernel_type);
    // }
    return true;
}
