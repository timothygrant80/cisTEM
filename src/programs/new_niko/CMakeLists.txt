add_executable(niko_test niko_test.cpp)
add_dependencies(niko_test cisTEM_core)

target_link_libraries(niko_test cisTEM_core)
