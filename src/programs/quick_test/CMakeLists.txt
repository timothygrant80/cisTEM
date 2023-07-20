add_executable(quick_test quick_test.cpp)
add_dependencies(quick_test cisTEM_core)

target_link_libraries(quick_test cisTEM_core)
