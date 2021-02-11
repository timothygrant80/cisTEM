add_executable(local_resolution local_resolution.cpp)
add_executable(local_resolution_finalize local_resolution_finalize.cpp)
add_dependencies(local_resolution cisTEM_core)
add_dependencies(local_resolution_finalize cisTEM_core)

target_link_libraries(local_resolution cisTEM_core)
target_link_libraries(local_resolution_finalize cisTEM_core)

install(TARGETS local_resolution local_resolution_finalize
        RUNTIME DESTINATION bin)
