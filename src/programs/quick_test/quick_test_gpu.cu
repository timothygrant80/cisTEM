#include "../../gpu/gpu_core_headers.h"
#include "../../gpu/gpu_indexing_functions.h"

#include "quick_test_gpu.h"

__global__ void helloWorldKernel( ) {
    printf("Hello from GPU\n");
}

__global__ void tf_kernel_old(float2* val, float TW, int idx) {
    __sincosf(TW * idx, &val->y, &val->x);
}

__global__ void tf_kernel_no_intrinsic(float2* val, float TW, int idx) {
    double x;
    double y;

    sincos(double(TW) * double(idx), &y, &x);
    val->x = float(x);
    val->y = float(y);
}

__global__ void tf_kernel_new(float2* val, float2 TWa, float2 TWs, int N) {

    val[0] = TWa;
    for ( int i = 1; i < N; i++ ) {
        TWa    = ComplexMul(TWa, TWs);
        val[i] = TWa;
    }
}

void QuickTestGPU::callHelloFromGPU( ) {

    // Normally you cannot print direclty from a GPU kernel, b/c wx steals the default thread.
    // We can temporarily steal the stdout pointer to stderr to print from the GPU kernel.
    // std::string       msg = "Hello from GPU";
    // FlushKernelPrintF flusher(msg);
    // helloWorldKernel<<<1, 1, 0, cudaStreamPerThread>>>( );
    // cudaErr(cudaStreamSynchronize(cudaStreamPerThread));

    std::vector<float> index;
    int                i      = 0;
    constexpr int      len    = 512;
    int                ept    = 64;
    int                stride = len / ept;

    while ( i < len ) {
        index.push_back(i + 1);
        i += stride;
    }

    constexpr float     TW = float(-2.0 * pi_v<double> / double(len));
    std::vector<float2> old_twiddle_factors(index.size( ));
    std::vector<float2> new_twiddle_factors(index.size( ));
    std::vector<float2> ref_twiddle_factors(index.size( ));
    for ( int i = 0; i < old_twiddle_factors.size( ); i++ ) {
        old_twiddle_factors.at(i) = float2{0.f, 0.f};
        new_twiddle_factors.at(i) = float2{0.f, 0.f};
        ref_twiddle_factors.at(i) = float2{0.f, 0.f};
    }

    float2* d_val;
    cudaErr(cudaMalloc(&d_val, sizeof(float2)));

    for ( int i = 0; i < index.size( ); i++ ) {
        float idx = index.at(i);
        precheck;
        tf_kernel_old<<<1, 1>>>(d_val, TW, idx);
        postcheck;
        cudaErr(cudaMemcpy(&old_twiddle_factors[i], d_val, sizeof(float2), cudaMemcpyDeviceToHost));
    }

    for ( int i = 0; i < index.size( ); i++ ) {
        float idx = index.at(i);
        precheck;
        tf_kernel_no_intrinsic<<<1, 1>>>(d_val, TW, idx);
        postcheck;
        cudaErr(cudaMemcpy(&ref_twiddle_factors[i], d_val, sizeof(float2), cudaMemcpyDeviceToHost));
    }

    float2* d_val_new;
    cudaErr(cudaMalloc(&d_val_new, sizeof(float2) * index.size( )));
    float2 TWa, TWs;
    sincos(TW * index.at(0), &TWa.y, &TWa.x);
    sincos(TW * float(stride), &TWs.y, &TWs.x);
    precheck;
    tf_kernel_new<<<1, 1>>>(d_val_new, TWa, TWs, index.size( ));
    postcheck;
    cudaErr(cudaMemcpy(new_twiddle_factors.data( ), d_val_new, sizeof(float2) * index.size( ), cudaMemcpyDeviceToHost));

    for ( auto& val : old_twiddle_factors ) {
        wxPrintf("old twiddle factor = %3.15f %3.15f\n", val.x, val.y);
    }

    for ( auto& val : ref_twiddle_factors ) {
        wxPrintf("ref twiddle factor = %3.15f %3.15f\n", val.x, val.y);
    }

    for ( auto& val : new_twiddle_factors ) {
        wxPrintf("new twiddle factor = %3.15f %3.15f\n", val.x, val.y);
    }

    // print the relative difference between the two
    double avg_diff_x;
    double avg_diff_y;
    for ( int i = 0; i < index.size( ); i++ ) {
        float2 diff = old_twiddle_factors.at(i) - ref_twiddle_factors.at(i);

        avg_diff_x += diff.x;
        avg_diff_y += diff.y;
        // wxPrintf("diff = %3.6g %3.6g\n", diff.x, diff.y);
    }
    avg_diff_x /= index.size( );
    avg_diff_y /= index.size( );
    wxPrintf("\navg diff old = %3.6g %3.6g\n", avg_diff_x, avg_diff_y);

    avg_diff_x = 0;
    avg_diff_y = 0;
    for ( int i = 0; i < index.size( ); i++ ) {
        float2 diff = new_twiddle_factors.at(i) - ref_twiddle_factors.at(i);

        avg_diff_x += diff.x;
        avg_diff_y += diff.y;
        // wxPrintf("diff new = %3.6g %3.6g\n", diff.x, diff.y);
    }
    avg_diff_x /= index.size( );
    avg_diff_y /= index.size( );
    wxPrintf("\navg diff new = %3.6g %3.6g\n", avg_diff_x, avg_diff_y);

    // When the flusher goes out of scope, the stdout pointer is restored automatically.
}
