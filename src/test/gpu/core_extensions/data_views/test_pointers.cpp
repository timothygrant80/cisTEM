
#include "../../../../gpu/core_extensions/data_views/pointers.h"
#include "../../../../../include/catch2/catch.hpp"

TEST_CASE("Pointers constructions ", "[Pointers]") {
    DevicePointerArray<float2> testPtr2;

    testPtr2.resize(4);

    // Initialize some data on the host
    std::vector<float2> test_data(4);
    for ( int i = 0; i < 4; i++ ) {
        test_data[i].x = i;
        test_data[i].y = i + 1;
    }

    // Set device pointers and copy in data
    float2* d_ptr_1;
    float2* d_ptr_2;
    float2* d_ptr_3;
    float2* d_ptr_4;

    cudaErr(cudaMalloc(&d_ptr_1, sizeof(float2)));
    cudaErr(cudaMalloc(&d_ptr_2, sizeof(float2)));
    cudaErr(cudaMalloc(&d_ptr_3, sizeof(float2)));
    cudaErr(cudaMalloc(&d_ptr_4, sizeof(float2)));

    cudaErr(cudaMemcpyAsync(d_ptr_1, &test_data[0], sizeof(float2), cudaMemcpyHostToDevice, cudaStreamPerThread));
    cudaErr(cudaMemcpyAsync(d_ptr_2, &test_data[1], sizeof(float2), cudaMemcpyHostToDevice, cudaStreamPerThread));
    cudaErr(cudaMemcpyAsync(d_ptr_3, &test_data[2], sizeof(float2), cudaMemcpyHostToDevice, cudaStreamPerThread));
    cudaErr(cudaMemcpyAsync(d_ptr_4, &test_data[3], sizeof(float2), cudaMemcpyHostToDevice, cudaStreamPerThread));

    cudaErr(cudaStreamSynchronize(cudaStreamPerThread));

    // Now set the arryay of device pointers
    testPtr2.SetPointer(d_ptr_1, 0);
    testPtr2.SetPointer(d_ptr_2, 1);
    testPtr2.SetPointer(d_ptr_3, 2);
    testPtr2.SetPointer(d_ptr_4, 3);

    // Setup some clean host values to copy into
    std::vector<float2> copy_back(4);

    // Copy out the data
    for ( int i = 0; i < 4; i++ ) {
        cudaErr(cudaMemcpyAsync(&copy_back[i], testPtr2.ptr_array[i], sizeof(float2), cudaMemcpyDeviceToHost, cudaStreamPerThread));
    }

    cudaErr(cudaStreamSynchronize(cudaStreamPerThread));

    // Finally check the values
    for ( int i = 0; i < 4; i++ ) {
        REQUIRE(test_data[i].x == copy_back[i].x);
        REQUIRE(test_data[i].y == copy_back[i].y);
    }

    testPtr2.Deallocate( );
}
