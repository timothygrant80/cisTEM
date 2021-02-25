#
# CMake for cisTEM
#
# Example usage:
# mdkir build && cd build
# cmake -DBUILD_STATIC_BINARIES=ON -DBUILD_EXPERIMENTAL_FEATURES=OFF ..
#
# or from settings.json in vscode, set same variables
#
project(cisTEM)
cmake_minimum_required(VERSION 3.11)
set(CMAKE_MODULE_PATH "${PROJECT_SOURCE_DIR}/cmake_modules" ${CMAKE_MODULE_PATH})

# specify the C++ standard
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Debug mode
set(CMAKE_C_FLAGS_DEBUG "${CMAKE_C_FLAGS_DEBUG} -DDEBUG")
set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -DDEBUG -O0")

# Experimental build option
option(BUILD_EXPERIMENTAL_FEATURES "Build experimental parts of cisTEM" OFF)
if (BUILD_EXPERIMENTAL_FEATURES)
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -DEXPERIMENTAL")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DEXPERIMENTAL")
endif(BUILD_EXPERIMENTAL_FEATURES)

# OpenMP build option
option(BUILD_OpenMP "Enable OpenMP for multithreading" OFF)
if (BUILD_OpenMP)
    find_package(OpenMP)
    if(OpenMP_FOUND OR OpenMP_CXX_FOUND)
        set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
        set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
    else()
        message(FATAL_ERROR "Build step for OpenMP failed.")
    endif()
endif(BUILD_OpenMP)

#
# Let's link in static libraries
#
option(BUILD_STATIC_BINARIES "Build static binaries to minimize runtime dependencies" OFF)
if (BUILD_STATIC_BINARIES)
    #set(CMAKE_FIND_LIBRARY_SUFFIXES ".a")
    set(BUILD_SHARED_LIBS OFF)
    #set(CMAKE_EXE_LINKER_FLAGS "-static" )
    set(wxWidgets_USE_STATIC ON)
    set(FFTW_USE_STATIC_LIBS ON)
endif(BUILD_STATIC_BINARIES) 


#
# MKL
#
# According to the URL below, proper linking will require:
# target_link_libraries(mytest "-Wl,--start-group" ${MKL_LIBRARIES} "-Wl,--end-group -lpthread -lm -ldl")
# https://stackoverflow.com/questions/56200963/how-to-setup-mkl-on-linux-with-cmake?noredirect=1&lq=1
find_package(MKL)
if(MKL_FOUND)
    message("Using Intel MKL for fast Fourier transforms")
    include_directories(${MKL_INCLUDE_DIRS})
    # We have MKL, so we will use the MKL-provided FFT instead of relying on the FFTW library
    set(FFTW_LIBRARIES ${MKL_LIBRARIES})
else()
    message("Using FFTW for fast Fourier transforms")
    #
    # Setup findFFTW
    #
    # https://github.com/egpbos/findfftw
    #
    # TODO: just add a static FindFFTW.cmake module in cmake_modules/ so that we do not need to be connected to the internet to build
    configure_file(downloadFindFFTW.cmake.in findFFTW-download/CMakeLists.txt)
    execute_process(COMMAND ${CMAKE_COMMAND} -G "${CMAKE_GENERATOR}" .
            RESULT_VARIABLE result
            WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/findFFTW-download )
    if(result)
        message(FATAL_ERROR "CMake step for findFFTW failed: ${result}")
        else()
        message("CMake step for findFFTW completed (${result}).")
    endif()
    execute_process(COMMAND ${CMAKE_COMMAND} --build .
            RESULT_VARIABLE result
            WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/findFFTW-download )
    if(result)
        message(FATAL_ERROR "Build step for findFFTW failed: ${result}")
    endif()
    set(findFFTW_DIR ${CMAKE_CURRENT_BINARY_DIR}/findFFTW-src)
    set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${findFFTW_DIR}")
        
    #
    # Let's look for single-precision FFTW
    #
    find_package(FFTW REQUIRED COMPONENTS FLOAT_LIB)
    include_directories(${FFTW_INCLUDE_DIRS})
endif()


message("Static library suffix: ${CMAKE_STATIC_LIBRARY_SUFFIX}")
message("FFTW libraries: ${FFTW_LIBRARIES}")

#
# Let's look for wxWidgets
#
# How can the user point us to the correct wx-config?
# https://github.com/wxMaxima-developers/wxmaxima/issues/1078
# by setting environment variable WX_CONFIG, assuming cmake 3.11+
#
# To set an environmental variable for cmake configure time in vscode, 
# edit the "cmake.configureEnvironment" part of settings.json
#
# The FindwxWidgets.cmake module will then use this to search for wx-config
# and once it's found it, will remember its path as wxWidgets_CONFIG_EXECUTABLE
#
# More info here: https://gitlab.kitware.com/cmake/cmake/-/issues/17164
# 
# Note: if you first run cmake without this env variable, running this again
# may keep finding the same old wx-config, not the one in the env var. 
# Probably some caching behavior. 
# Solution: delete the CMakeCache.txt file, or possibly nuke the build directory and re-run cmake
# Solution within vscode: run "CMake: Delete cached built settings and reconfigure" from the command palette
#
message("Will search for wxWidgets. To specify which installation to use, set the environment variable WX_CONFIG to the full path of wx-config")
if(DEFINED ENV{WX_CONFIG})
message("Found the following wx-config set by ENV{WX_CONFIG}: $ENV{WX_CONFIG}")
endif()
set(wxWidgets_USE_DEBUG false)	
find_package(wxWidgets REQUIRED COMPONENTS core base std net xml richtext aui)
message("wx include locations: ${wxWidgets_INCLUDE_DIRS}")
include(${wxWidgets_USE_FILE})
include_directories(${wxWidgets_INCLUDE_DIRS})
message(wxWidgets libraries: ${wxWidgets_LIBRARIES})

#
# Let's look for LibTIFF
#
#set(CMAKE_FIND_LIBRARY_SUFFIXES ".a")
find_package(TIFF REQUIRED)
include_directories(${TIFF_INCLUDE_DIRS})
message("TIFF libraries: ${TIFF_LIBRARIES}")

#
# Let's figure out which SVN revision we are building (if indeed we're using SVN at all)
#
find_package(Subversion) 
if(SUBVERSION_FOUND)
    if(EXISTS "${CMAKE_SOURCE_DIR}/.svn")
        Subversion_WC_INFO(${PROJECT_SOURCE_DIR} cisTEM)
        message("Current SVN revision is ${cisTEM_WC_REVISION}")
        execute_process(    COMMAND "${Subversion_SVN_EXECUTABLE}" propget --revprop -r "${cisTEM_WC_REVISION}" git-commit  "https://github.com/ngrigorieff/cisTEM/trunk"
                            OUTPUT_VARIABLE GIT_COMMIT_FROM_SVN )
        string(REGEX REPLACE "\n$" "" GIT_COMMIT_FROM_SVN "${GIT_COMMIT_FROM_SVN}")
        message("Current Git commit from SVN is ${GIT_COMMIT_FROM_SVN}")
        add_definitions(" -DCISTEM_SVN_REV=\"\\\"${cisTEM_WC_REVISION}\\\"\" " " -DCISTEM_GIT_COMMIT=\"\\\"${GIT_COMMIT_FROM_SVN}\\\"\" ")
    endif()
endif() 

#
# Let's see whether we have git
#
find_package(Git)
if(Git_FOUND)
    if(EXISTS "${CMAKE_SOURCE_DIR}/.git")
        execute_process(    COMMAND "${GIT_EXECUTABLE}" log -1 "--format=%H"
                            WORKING_DIRECTORY "${CMAKE_SOURCE_DIR}"
                            OUTPUT_VARIABLE GIT_COMMIT )
        string(REGEX REPLACE "\n$" "" GIT_COMMIT "${GIT_COMMIT}")
        message("Current Git commit is ${GIT_COMMIT}")
        add_definitions(" -DCISTEM_GIT_COMMIT=\"\\\"${GIT_COMMIT}\\\"\" " )
    endif()
endif()

#
# Add the source directory 
#
add_subdirectory(src)
