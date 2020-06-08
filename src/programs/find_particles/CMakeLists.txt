add_executable(find_particles find_particles.cpp)
add_dependencies(find_particles cisTEM_core)

target_link_libraries(find_particles cisTEM_core)

install(TARGETS find_particles
        RUNTIME DESTINATION bin)
