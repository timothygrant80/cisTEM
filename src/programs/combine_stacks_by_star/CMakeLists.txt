add_executable(combine_stacks_by_star combine_stacks_by_star.cpp)
add_dependencies(combine_stacks_by_star cisTEM_core)

target_link_libraries(combine_stacks_by_star cisTEM_core)

install(TARGETS combine_stacks_by_star
        RUNTIME DESTINATION bin)
