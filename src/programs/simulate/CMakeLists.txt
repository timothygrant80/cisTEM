add_executable(simulate simulate.cpp wave_function_propagator.cpp scattering_potential.cpp)
add_dependencies(simulate cisTEM_core_experimental cisTEM_core)

target_link_libraries(simulate cisTEM_core_experimental cisTEM_core)

install(TARGETS simulate
        RUNTIME DESTINATION bin)