add_executable(cisTEM_job_control guix_job_control.cpp)

add_dependencies(cisTEM_job_control cisTEM_core)

target_link_libraries(cisTEM_job_control    cisTEM_core)

install(TARGETS cisTEM_job_control
        RUNTIME DESTINATION bin)