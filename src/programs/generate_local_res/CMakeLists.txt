add_executable(generate_local_res generate_local_res.cpp)
add_dependencies(generate_local_res cisTEM_core)

target_link_libraries(generate_local_res cisTEM_core)

install(TARGETS generate_local_res
        RUNTIME DESTINATION bin)
