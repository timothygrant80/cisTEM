add_executable(refine_ctf refine_ctf.cpp)
add_dependencies(refine_ctf cisTEM_core)

target_link_libraries(refine_ctf cisTEM_core)

install(TARGETS refine_ctf
        RUNTIME DESTINATION bin)
