add_executable(combine_stacks combine_stacks.cpp)
add_dependencies(combine_stacks cisTEM_core)

target_link_libraries(combine_stacks cisTEM_core)

install(TARGETS combine_stacks
        RUNTIME DESTINATION bin)
