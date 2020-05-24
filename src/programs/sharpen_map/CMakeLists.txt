add_executable(sharpen_map sharpen_map.cpp)
add_dependencies(sharpen_map cisTEM_core)

target_link_libraries(sharpen_map cisTEM_core)
