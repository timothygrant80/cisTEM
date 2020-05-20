add_executable(refine2d refine2d.cpp)
add_dependencies(refine2d cisTEM_core)

target_link_libraries(refine2d cisTEM_core)
