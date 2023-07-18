
#include "../../../gpu/gpu_core_headers.h"
#include "../../../../include/catch2/catch.hpp"
#include "test_gpu_core_headers.cuh"

TEST_CASE("Gpu Core Headers ", "[Gpu Core Headers]") {

    // Setup to test the complex math options
    Complex c1 = make_float2(1.0f, 2.0f);
    Complex c2 = make_float2(3.0f, 4.0f);
    Complex c3;
    Complex device_result;

    Complex* d_c1;
    Complex* d_c2;
    Complex* d_c3;

    cudaErr(cudaMalloc(&d_c1, sizeof(Complex)));
    cudaErr(cudaMalloc(&d_c2, sizeof(Complex)));
    cudaErr(cudaMalloc(&d_c3, sizeof(Complex)));

    // Copy the test values to device
    cudaErr(cudaMemcpy(d_c1, &c1, sizeof(Complex), cudaMemcpyHostToDevice));
    cudaErr(cudaMemcpy(d_c2, &c2, sizeof(Complex), cudaMemcpyHostToDevice));

    SECTION("testing ComplexAdd") {
        // test ComplexAdd on the host
        c3 = ComplexAdd(c1, c2);
        REQUIRE(c3.x == 4.0f);
        REQUIRE(c3.y == 6.0f);

        // test ComplexAdd on the device
        test_complex_add(d_c1, d_c2, d_c3);

        cudaErr(cudaMemcpy(&device_result, d_c3, sizeof(Complex), cudaMemcpyDeviceToHost));

        REQUIRE(device_result.x == 4.0f);
        REQUIRE(device_result.y == 6.0f);
    }

    SECTION("testing ComplexScale") {
        // test ComplexScale on the host by reference
        c3 = ComplexScale(c1, 2.0f);
        REQUIRE(c3.x == 2.0f);
        REQUIRE(c3.y == 4.0f);

        // test ComplexScale on the device by reference
        c3 = test_complex_scale(*d_c1, 2.0f);
        REQUIRE(c3.x == 2.0f);
        REQUIRE(c3.y == 4.0f);
    }
    SECTION("testing ComplexScale with pointer") {
        // test ComplexScale on the host by reference to pointer
        ComplexScale(&c1, 2.0f);
        REQUIRE(c1.x == 2.0f);
        REQUIRE(c1.y == 4.0f);

        // test ComplexScale on the device by reference to pointer
        test_complex_scale(d_c1, 2.0f);
        cudaErr(cudaMemcpy(&device_result, d_c1, sizeof(Complex), cudaMemcpyDeviceToHost));
        REQUIRE(device_result.x == 2.0f);
        REQUIRE(device_result.y == 4.0f);
    }

    // clean up device mem
    cudaErr(cudaFree(d_c1));
    cudaErr(cudaFree(d_c2));
    cudaErr(cudaFree(d_c3));
}