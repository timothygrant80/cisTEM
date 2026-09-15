#include "../../../core/socket_communication_utils/job_packager.h"
#include "../../../../include/catch2/catch.hpp"

/**
 * @file test_job_packager.cpp
 * @brief Unit tests for job_packager socket communication utilities
 *
 * These tests verify type safety and basic functionality after the fundamental_type::Enum
 * refactoring (uint8_t underlying type + static assertions). Full socket round-trip testing
 * would require mock socket infrastructure and is beyond the scope of these unit tests.
 *
 * See job_packager.cpp Doxygen @warning tags for purple team security findings.
 */

TEST_CASE("fundamental_type::Enum type safety", "[job_packager][type_safety]") {
    using c_ft = cistem::fundamental_type::Enum;

    SECTION("Enum has correct underlying type") {
        // Verify that fundamental_type::Enum is uint8_t (1 byte) as required by wire protocol
        REQUIRE(sizeof(c_ft) == sizeof(uint8_t));
        REQUIRE(sizeof(c_ft) == 1);
    }

    SECTION("All enum values fit in uint8_t") {
        // Verify all enum values are within the valid range for wire protocol
        REQUIRE(c_ft::none_t < 256);
        REQUIRE(c_ft::text_t < 256);
        REQUIRE(c_ft::integer_t < 256);
        REQUIRE(c_ft::float_t < 256);
        REQUIRE(c_ft::bool_t < 256);
        REQUIRE(c_ft::long_t < 256);
        REQUIRE(c_ft::double_t < 256);
        REQUIRE(c_ft::char_t < 256);
        REQUIRE(c_ft::variable_length_t < 256);
        REQUIRE(c_ft::integer_unsigned_t < 256);
    }

    SECTION("Type descriptor casting is safe") {
        // Verify that casting to uint8_t for wire protocol is safe
        c_ft    type_descriptor = c_ft::integer_t;
        uint8_t type_as_byte    = static_cast<uint8_t>(type_descriptor);
        REQUIRE(type_as_byte == type_descriptor);
    }
}

TEST_CASE("RunArgument type storage and retrieval", "[job_packager][RunArgument]") {
    using c_ft = cistem::fundamental_type::Enum;
    RunArgument arg;

    SECTION("String argument") {
        arg.SetStringArgument("test_string");
        REQUIRE(arg.type_of_argument == c_ft::text_t);
        REQUIRE(arg.ReturnStringArgument( ) == "test_string");
    }

    SECTION("Integer argument") {
        arg.SetIntArgument(42);
        REQUIRE(arg.type_of_argument == c_ft::integer_t);
        REQUIRE(arg.ReturnIntegerArgument( ) == 42);
    }

    SECTION("Float argument") {
        arg.SetFloatArgument(3.14f);
        REQUIRE(arg.type_of_argument == c_ft::float_t);
        REQUIRE(arg.ReturnFloatArgument( ) == 3.14f);
    }

    SECTION("Bool argument") {
        arg.SetBoolArgument(true);
        REQUIRE(arg.type_of_argument == c_ft::bool_t);
        REQUIRE(arg.ReturnBoolArgument( ) == true);

        arg.SetBoolArgument(false);
        REQUIRE(arg.ReturnBoolArgument( ) == false);
    }

    SECTION("Encoded byte transfer size is reasonable") {
        // String: 1 (type) + 4 (length) + string bytes
        arg.SetStringArgument("test");
        long transfer_size = arg.ReturnEncodedByteTransferSize( );
        REQUIRE(transfer_size == 1 + 4 + 4); // type + length + 4 chars

        // Integer: 1 (type) + 4 (value)
        arg.SetIntArgument(42);
        transfer_size = arg.ReturnEncodedByteTransferSize( );
        REQUIRE(transfer_size == 1 + 4);

        // Float: 1 (type) + 4 (value)
        arg.SetFloatArgument(3.14f);
        transfer_size = arg.ReturnEncodedByteTransferSize( );
        REQUIRE(transfer_size == 1 + 4);

        // Bool: 1 (type) + 1 (value)
        arg.SetBoolArgument(true);
        transfer_size = arg.ReturnEncodedByteTransferSize( );
        REQUIRE(transfer_size == 1 + 1);
    }
}

TEST_CASE("RunJob basic functionality", "[job_packager][RunJob]") {
    SECTION("Reset allocates correct number of arguments") {
        RunJob job;
        job.Reset(5);
        REQUIRE(job.number_of_arguments == 5);
        REQUIRE(job.arguments != nullptr);
    }

    SECTION("ManualSetArguments parses format string correctly") {
        RunJob job;
        // Format: "tifd" = text, integer, float, double (though double becomes float)
        job.ManualSetArguments("tif", "test_string", 42, 3.14f);

        REQUIRE(job.number_of_arguments == 3);
        REQUIRE(job.arguments[0].ReturnStringArgument( ) == "test_string");
        REQUIRE(job.arguments[1].ReturnIntegerArgument( ) == 42);
        REQUIRE(job.arguments[2].ReturnFloatArgument( ) == 3.14f);
    }

    SECTION("Copy assignment operator works") {
        RunJob job1;
        job1.job_number = 123;
        job1.ManualSetArguments("ti", "test", 456);

        RunJob job2;
        job2 = job1;

        REQUIRE(job2.job_number == 123);
        REQUIRE(job2.number_of_arguments == 2);
        REQUIRE(job2.arguments[0].ReturnStringArgument( ) == "test");
        REQUIRE(job2.arguments[1].ReturnIntegerArgument( ) == 456);
    }

    SECTION("Encoded byte transfer size calculation") {
        RunJob job;
        job.job_number = 1;
        job.ManualSetArguments("tif", "test", 42, 3.14f);

        long transfer_size = job.ReturnEncodedByteTransferSize( );
        // Should be: job_number(4) + number_of_arguments(4) + sum of argument sizes
        // text: 1 + 4 + 4 = 9
        // int: 1 + 4 = 5
        // float: 1 + 4 = 5
        // Total: 4 + 4 + 9 + 5 + 5 = 27
        REQUIRE(transfer_size == 27);
    }
}

TEST_CASE("JobResult basic functionality", "[job_packager][JobResult]") {
    SECTION("SetResult allocates and copies data correctly") {
        float     test_data[] = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f};
        JobResult result;
        result.SetResult(5, test_data);

        REQUIRE(result.result_size == 5);
        REQUIRE(result.result_data != nullptr);
        REQUIRE(result.result_data[0] == 1.0f);
        REQUIRE(result.result_data[4] == 5.0f);
    }

    SECTION("Copy constructor creates independent copy") {
        float     test_data[] = {1.0f, 2.0f, 3.0f};
        JobResult result1(3, test_data);
        JobResult result2(result1);

        REQUIRE(result2.result_size == 3);
        REQUIRE(result2.result_data != result1.result_data); // Different pointers
        REQUIRE(result2.result_data[0] == 1.0f);
        REQUIRE(result2.result_data[2] == 3.0f);
    }

    SECTION("Assignment operator works correctly") {
        float     test_data[] = {1.0f, 2.0f, 3.0f};
        JobResult result1(3, test_data);

        JobResult result2;
        result2 = result1;

        REQUIRE(result2.result_size == 3);
        REQUIRE(result2.result_data != nullptr);
        REQUIRE(result2.result_data[0] == 1.0f);
    }
}

TEST_CASE("JobPackage basic functionality", "[job_packager][JobPackage]") {
    SECTION("Reset initializes job array correctly") {
        RunProfile profile;
        JobPackage package;
        package.Reset(profile, "test_program", 10);

        REQUIRE(package.number_of_jobs == 10);
        REQUIRE(package.number_of_added_jobs == 0);
        REQUIRE(package.jobs != nullptr);
        REQUIRE(package.my_profile.executable_name == "test_program");
    }

    SECTION("AddJob increments counter and populates job") {
        RunProfile profile;
        JobPackage package(profile, "test_program", 5);

        package.AddJob("ti", "first_job", 1);
        REQUIRE(package.number_of_added_jobs == 1);
        REQUIRE(package.jobs[0].arguments[0].ReturnStringArgument( ) == "first_job");

        package.AddJob("ti", "second_job", 2);
        REQUIRE(package.number_of_added_jobs == 2);
        REQUIRE(package.jobs[1].arguments[0].ReturnStringArgument( ) == "second_job");
    }

    SECTION("ReturnNumberOfJobsRemaining works correctly") {
        RunProfile profile;
        JobPackage package(profile, "test_program", 3);

        REQUIRE(package.ReturnNumberOfJobsRemaining( ) == 3);

        package.jobs[0].has_been_run = true;
        REQUIRE(package.ReturnNumberOfJobsRemaining( ) == 2);

        package.jobs[1].has_been_run = true;
        package.jobs[2].has_been_run = true;
        REQUIRE(package.ReturnNumberOfJobsRemaining( ) == 0);
    }

    SECTION("ReturnEncodedByteTransferSize calculation") {
        RunProfile profile;
        profile.executable_name        = "test_exe";
        profile.gui_address            = "localhost";
        profile.number_of_run_commands = 0;

        JobPackage package(profile, "test_exe", 2);
        package.AddJob("i", 1);
        package.AddJob("i", 2);

        long transfer_size = package.ReturnEncodedByteTransferSize( );
        // Should be > 0 and include RunProfile + all jobs
        REQUIRE(transfer_size > 0);
    }
}

/**
 * @note Socket round-trip testing (Send/Receive methods) would require either:
 * 1. Mock TcpSocket infrastructure
 * 2. Real socket server/client setup
 * 3. Refactoring to separate encoding from socket I/O
 *
 * These are beyond the scope of this basic test suite. The Doxygen documentation
 * in job_packager.cpp provides the encoding order specifications that would be
 * verified by full integration tests.
 */
