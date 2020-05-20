add_executable(merge3d merge3d.cpp)
add_dependencies(merge3d cisTEM_core)

target_link_libraries(merge3d cisTEM_core)
