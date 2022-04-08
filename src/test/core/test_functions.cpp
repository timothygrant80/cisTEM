#include "../../core/functions.h"
#include "../../../include/catch2/catch.hpp"

TEST_CASE("ZeroBoolArray", "[ZeroBoolArray]") {
    bool arr[3] = {true, false, true};
    ZeroBoolArray(arr, 3);
    REQUIRE(arr[0] == false);
    REQUIRE(arr[1] == false);
    REQUIRE(arr[2] == false);
}

TEST_CASE("ZeroIntArray", "[ZeroIntArray]") {
    int arr[3] = {1, 2, 3};
    ZeroIntArray(arr, 3);
    REQUIRE(arr[0] == 0);
    REQUIRE(arr[1] == 0);
    REQUIRE(arr[2] == 0);
}

TEST_CASE("ZeroLongArray", "[ZeroLongArray]") {
    long arr[3] = {1, 2, 3};
    ZeroLongArray(arr, 3);
    REQUIRE(arr[0] == 0);
    REQUIRE(arr[1] == 0);
    REQUIRE(arr[2] == 0);
}

TEST_CASE("ZeroFloatArray", "[ZeroFloatArray]") {
    float arr[3] = {1.0f, 2.0f, 3.0f};
    ZeroFloatArray(arr, 3);
    REQUIRE(arr[0] == 0.0f);
    REQUIRE(arr[1] == 0.0f);
    REQUIRE(arr[2] == 0.0f);
}

TEST_CASE("FilenameReplaceExtension", "[FilenameReplaceExtension]") {
    std::string filename  = "test.txt";
    std::string extension = "new";
    std::string expected  = "test.new";
    REQUIRE(FilenameReplaceExtension(filename, extension) == expected);
}

TEST_CASE("FilenameAddSuffix") {
    std::string filename = "test.txt";
    std::string suffix   = "new";
    std::string expected = "test_new.txt";
    REQUIRE(FilenameAddSuffix(filename, suffix) == expected);
}

TEST_CASE("ReturnClosestFactorizedUpper") {
    // TODO: expand these
    std::vector<int> input_sizes  = {255, 310};
    std::vector<int> output_sizes = {256, 512};
    for ( size_t i = 0; i < input_sizes.size( ); i++ ) {
        REQUIRE(ReturnClosestFactorizedUpper(input_sizes[i], 2) == output_sizes[i]);
    }
}

TEST_CASE("ReturnClosestFactorizedLower") {
    // TODO: expand these
    std::vector<int> input_sizes  = {257, 522};
    std::vector<int> output_sizes = {256, 512};
    for ( size_t i = 0; i < input_sizes.size( ); i++ ) {
        REQUIRE(ReturnClosestFactorizedLower(input_sizes[i], 2) == output_sizes[i]);
    }
}
