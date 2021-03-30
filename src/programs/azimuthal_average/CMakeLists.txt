add_executable(azimuthal_average azimuthal_average.cpp)
add_dependencies(azimuthal_average cisTEM_core)

target_link_libraries(azimuthal_average cisTEM_core)

install(TARGETS azimuthal_average
        RUNTIME DESTINATION bin)
