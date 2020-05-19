message("In the src/core CMakeLists file")

add_library(ctffind_core STATIC 
	functions.cpp
	globals.cpp
	randomnumbergenerator.cpp
	userinput.cpp
	angles_and_shifts.cpp
	parameter_constraints.cpp
	curve.cpp
	image.cpp
	matrix.cpp
	symmetry_matrix.cpp
	abstract_image_file.cpp
	mrc_header.cpp
	mrc_file.cpp
	tiff_file.cpp
	dm_file.cpp
	image_file.cpp
	database.cpp
	project.cpp
	particle.cpp
	job_packager.cpp
	job_tracker.cpp
	run_profiles.cpp
	myapp.cpp
	empirical_distribution.cpp
	ctf.cpp
	numeric_text_file.cpp
	progressbar.cpp
	downhill_simplex.cpp
	brute_force_search.cpp
	va04.cpp
	conjugate_gradient.cpp
	socket_communicator.cpp
	ccl3d.cpp
	template_matching.cpp)

target_link_libraries(ctffind_core fftw3f)
target_link_libraries(ctffind_core ${wxWidgets_LIBRARIES})
target_link_libraries(ctffind_core ${TIFF_LIBRARIES})

