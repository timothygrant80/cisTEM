add_executable(combine_relion_stacks combine_relion_stacks.cpp)
add_dependencies(combine_relion_stacks cisTEM_core)

target_link_libraries(combine_relion_stacks cisTEM_core)

install(TARGETS combine_relion_stacks
        RUNTIME DESTINATION bin)
