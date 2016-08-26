noinst_HEADERS = core/asset_group.h \
                 core/core_headers.h \
                 core/defines.h \
                 core/functions.h \
                 core/image.h \
                 core/mrc_file.h \
                 core/mrc_header.h \
                 core/dm_file.h \
                 gui/AlignMoviesPanel.h \
                 gui/ErrorDialog.h \
                 gui/MainFrame.h \
                 gui/mathplot.h \
                 gui/MyMovieAssetPanel.h \
                 gui/MovieImportDialog.h \
                 gui/ProjectX_gui.h \
                 gui/icons/action_icon.cpp \
                 gui/icons/assets_icon.cpp \
                 gui/icons/ctf_icon.cpp \
                 gui/icons/image_icon.cpp \
                 gui/icons/movie_align_icon.cpp \
                 gui/icons/movie_icon.cpp \
                 gui/icons/overview_icon.cpp 

noinst_LIBRARIES = libcore.a libguicore.a


bin_PROGRAMS = ProjectX unblur guix_job_control console_test applyctf reconstruct3d merge3d project3d refine3d calc_occ ctffind remove_outlier_pixels resize resample reset_mrc_header estimate_dataset_ssnr find_particles montage extract_particles


libcore_a_SOURCES =    core/functions.cpp \
					   core/globals.cpp \
					   core/randomnumbergenerator.cpp \
                       core/userinput.cpp \
                       core/angles_and_shifts.cpp \
                       core/parameter_constraints.cpp \
                       core/assets.cpp \
                       core/asset_group.cpp \
                       core/curve.cpp \
                       core/image.cpp \
                       core/electron_dose.cpp \
                       core/matrix.cpp \
                       core/symmetry_matrix.cpp \
                       core/mrc_file.cpp \
                       core/mrc_header.cpp \
                       core/dm_file.cpp \
                       core/sqlite/sqlite3.c \
                       core/database.cpp \
                       core/project.cpp \
                       core/reconstruct_3d.cpp \
                       core/reconstructed_volume.cpp \
                       core/resolution_statistics.cpp \
                       core/particle.cpp \
                       core/job_packager.cpp \
                       core/job_tracker.cpp \
                       core/run_profiles.cpp \
                       core/myapp.cpp \
                       core/empirical_distribution.cpp \
                       core/ctf.cpp \
                       core/numeric_text_file.cpp \
                       core/progressbar.cpp \
                       core/downhill_simplex.cpp \
                       core/brute_force_search.cpp \
                       core/va04.cpp \
                       core/conjugate_gradient.cpp \
                       core/euler_search.cpp \
                       core/frealign_parameter_file.cpp \
                       core/refinement_package.cpp \
                       core/refinement.cpp

libcore_a_CXXFLAGS = $(WX_CPPFLAGS_BASE)
libcore_a_CPPFLAGS = $(WX_CPPFLAGS_BASE)

libguicore_a_SOURCES =  core/gui_job_controller.cpp

libguicore_a_CXXFLAGS = $(WX_CPPFLAGS)
libguicore_a_CPPFLAGS = $(WX_CPPFLAGS)

ProjectX_SOURCES = programs/projectx/projectx.cpp gui/gui_functions.cpp gui/my_controls.cpp gui/MyRenameDialog.cpp gui/BitmapPanel.cpp gui/PickingBitmapPanel.cpp gui/job_panel.cpp gui/ProjectX_gui.cpp gui/ResultsDataViewListCtrl.cpp gui/MyMovieFilterDialog.cpp gui/MainFrame.cpp gui/MyAssetParentPanel.cpp gui/MyMovieAssetPanel.cpp gui/MyImageAssetPanel.cpp gui/MyParticlePositionAssetPanel.cpp gui/MyVolumeChooserDialog.cpp gui/MovieImportDialog.cpp gui/MyImageImportDialog.cpp gui/MyVolumeImportDialog.cpp gui/ErrorDialog.cpp gui/AlignMoviesPanel.cpp gui/MyRunProfilesPanel.cpp gui/mathplot.cpp gui/MyMovieAlignResultsPanel.cpp gui/MyFindCTFResultsPanel.cpp gui/PickingResultsPanel.cpp gui/MyAddRunCommandDialog.cpp gui/MyNewProjectWizard.cpp gui/MyNewRefinementPackageWizard.cpp gui/UnblurResultsPanel.cpp gui/ShowCTFResultsPanel.cpp gui/PickingResultsDisplayPanel.cpp gui/CTF1DPanel.cpp gui/FindCTFPanel.cpp gui/FindParticlesPanel.cpp gui/MyVolumeAssetPanel.cpp gui/MyResultsPanel.cpp gui/MyParticlePositionExportDialog.cpp gui/MyFrealignExportDialog.cpp gui/MyRefinementPackageAssetPanel.cpp gui/MyRefine3DPanel.cpp
ProjectX_CXXFLAGS = $(WX_CPPFLAGS)
ProjectX_CPPFLAGS = $(WX_CPPFLAGS)
ProjectX_LDFLAGS = $(WX_LIBS)
ProjectX_LDADD = libguicore.a libcore.a

guix_job_control_SOURCES = programs/guix_job_control/guix_job_control.cpp 
guix_job_control_CXXFLAGS = $(WX_CPPFLAGS_BASE)
guix_job_control_CPPFLAGS = $(WX_CPPFLAGS_BASE)
guix_job_control_LDFLAGS = $(WX_LIBS_BASE)
guix_job_control_LDADD = libcore.a

console_test_SOURCES = programs/console_test/console_test.cpp 
console_test_CXXFLAGS = $(WX_CPPFLAGS_BASE)
console_test_CPPFLAGS = $(WX_CPPFLAGS_BASE)
console_test_LDFLAGS = $(WX_LIBS_BASE)
console_test_LDADD = libcore.a

unblur_SOURCES = programs/unblur/unblur.cpp
unblur_CXXFLAGS = $(WX_CPPFLAGS_BASE)
unblur_CPPFLAGS = $(WX_CPPFLAGS_BASE)
unblur_LDFLAGS = $(WX_LIBS_BASE)
unblur_LDADD = libcore.a

applyctf_SOURCES = programs/applyctf/applyctf.cpp
applyctf_CXXFLAGS = $(WX_CPPFLAGS_BASE)
applyctf_CPPFLAGS = $(WX_CPPFLAGS_BASE)
applyctf_LDFLAGS = $(WX_LIBS_BASE)
applyctf_LDADD = libcore.a

ctffind_SOURCES = programs/ctffind/ctffind.cpp
ctffind_CXXFLAGS = $(WX_CPPFLAGS_BASE)
ctffind_CPPFLAGS = $(WX_CPPFLAGS_BASE)
ctffind_LDFLAGS = $(WX_LIBS_BASE)
ctffind_LDADD = libcore.a

reconstruct3d_SOURCES = programs/reconstruct3d/reconstruct3d.cpp
reconstruct3d_CXXFLAGS = $(WX_CPPFLAGS_BASE)
reconstruct3d_CPPFLAGS = $(WX_CPPFLAGS_BASE)
reconstruct3d_LDFLAGS = $(WX_LIBS_BASE)
reconstruct3d_LDADD = libcore.a

merge3d_SOURCES = programs/merge3d/merge3d.cpp
merge3d_CXXFLAGS = $(WX_CPPFLAGS_BASE)
merge3d_CPPFLAGS = $(WX_CPPFLAGS_BASE)
merge3d_LDFLAGS = $(WX_LIBS_BASE)
merge3d_LDADD = libcore.a

calc_occ_SOURCES = programs/calc_occ/calc_occ.cpp
calc_occ_CXXFLAGS = $(WX_CPPFLAGS_BASE)
calc_occ_CPPFLAGS = $(WX_CPPFLAGS_BASE)
calc_occ_LDFLAGS = $(WX_LIBS_BASE)
calc_occ_LDADD = libcore.a

project3d_SOURCES = programs/project3d/project3d.cpp
project3d_CXXFLAGS = $(WX_CPPFLAGS_BASE)
project3d_CPPFLAGS = $(WX_CPPFLAGS_BASE)
project3d_LDFLAGS = $(WX_LIBS_BASE)
project3d_LDADD = libcore.a

refine3d_SOURCES = programs/refine3d/refine3d.cpp
refine3d_CXXFLAGS = $(WX_CPPFLAGS_BASE)
refine3d_CPPFLAGS = $(WX_CPPFLAGS_BASE)
refine3d_LDFLAGS = $(WX_LIBS_BASE)
refine3d_LDADD = libcore.a

remove_outlier_pixels_SOURCES = programs/remove_outlier_pixels/remove_outlier_pixels.cpp
remove_outlier_pixels_CXXFLAGS = $(WX_CPPFLAGS_BASE)
remove_outlier_pixels_CPPFLAGS = $(WX_CPPFLAGS_BASE)
remove_outlier_pixels_LDFLAGS = $(WX_LIBS_BASE)
remove_outlier_pixels_LDADD = libcore.a

resize_SOURCES = programs/resize/resize.cpp
resize_CXXFLAGS = $(WX_CPPFLAGS_BASE)
resize_CPPFLAGS = $(WX_CPPFLAGS_BASE)
resize_LDFLAGS = $(WX_LIBS_BASE)
resize_LDADD = libcore.a

resample_SOURCES = programs/resample/resample.cpp
resample_CXXFLAGS = $(WX_CPPFLAGS_BASE)
resample_CPPFLAGS = $(WX_CPPFLAGS_BASE)
resample_LDFLAGS = $(WX_LIBS_BASE)
resample_LDADD = libcore.a

estimate_dataset_ssnr_SOURCES = programs/estimate_dataset_ssnr/estimate_dataset_ssnr.cpp
estimate_dataset_ssnr_CXXFLAGS = $(WX_CPPFLAGS_BASE)
estimate_dataset_ssnr_CPPFLAGS = $(WX_CPPFLAGS_BASE)
estimate_dataset_ssnr_LDFLAGS = $(WX_LIBS_BASE)
estimate_dataset_ssnr_LDADD = libcore.a

reset_mrc_header_SOURCES = programs/reset_mrc_header/reset_mrc_header.cpp
reset_mrc_header_CXXFLAGS = $(WX_CPPFLAGS_BASE)
reset_mrc_header_CPPFLAGS = $(WX_CPPFLAGS_BASE)
reset_mrc_header_LDFLAGS = $(WX_LIBS_BASE)
reset_mrc_header_LDADD = libcore.a

find_particles_SOURCES = programs/find_particles/find_particles.cpp
find_particles_CXXFLAGS = $(WX_CPPFLAGS_BASE)
find_particles_CPPFLAGS = $(WX_CPPFLAGS_BASE)
find_particles_LDFLAGS = $(WX_LIBS_BASE)
find_particles_LDADD = libcore.a

montage_SOURCES = programs/montage/montage.cpp
montage_CXXFLAGS = $(WX_CPPFLAGS_BASE)
montage_CPPFLAGS = $(WX_CPPFLAGS_BASE)
montage_LDFLAGS = $(WX_LIBS_BASE)
montage_LDADD = libcore.a

extract_particles_SOURCES = programs/extract_particles/extract_particles.cpp
extract_particles_CXXFLAGS = $(WX_CPPFLAGS_BASE)
extract_particles_CPPFLAGS = $(WX_CPPFLAGS_BASE)
extract_particles_LDFLAGS = $(WX_LIBS_BASE)
extract_particles_LDADD = libcore.a
