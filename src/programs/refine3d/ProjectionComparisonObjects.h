/*

Written by: Ben Himes at some time before now.

The goal for this class is to organize and update a series of pointers during the various cycles and enumerations of particle refinement in refine3d.

The reason this is needed, is partly for convenience but also for efficiency and thread safety. This object stores pointers to several types of image objects, 
optionally gpu image objects as well and is passed by reference (via a pointer) to the conjugate gradient minimizer and the FrealignObjective function.

We also use this to drop in some gpu methods that are ifdef'd here so that the (already hard to read) code in refine3d doesn't get harder to read.

The other primary function of this class is to ensure parity between GpuImage and Image objects. The gpu integration is implemented in a fashion,
to only do what is strictly necessary, while cpu Image objects still handle much of the work. This means both the meta-data defining an image object, as well as 
the actual data itself may fall out of sync with an associated GpuImage object.

*/

#ifndef _SRC_PROGRAMS_REFINE3D_PROJECTION_COMPARISON_OBJECTS_H_
#define _SRC_PROGRAMS_REFINE3D_PROJECTION_COMPARISON_OBJECTS_H_

#include "refine3d_defines.h"

#ifdef ENABLEGPU
#warning "Experimental GPU code from ProjectionComparisonObjects.h will be used in refine3d_gpu"
#else
#warning "CPU code is not yet implemented"
class GpuImage;
#endif

#include "../../constants/constants.h"
using c_img_t = cistem::PCOS_image_type::Enum;

class ProjectionComparisonObjects {

  public:
#ifdef ENABLEGPU
    // the search volumes refer to global (grid search) which happens either in CTF refinement or in global search.
    GpuImage gpu_density_map;
    GpuImage gpu_projection;
    GpuImage gpu_ctf_image;
    GpuImage gpu_particle_image;

    GpuImage gpu_search_density_map;
    GpuImage gpu_search_projection;
    GpuImage gpu_search_ctf_image;
    GpuImage gpu_search_particle_image;

    GpuImage clean_copy;
    GpuImage buffer_cross_terms;
    GpuImage buffer_image_ps;
    GpuImage buffer_projection_ps;

    // #else
    //     // FIXME: shouldn't need to do this to get Cpu to compile - but in some debug steps still accessing the GPU members directly (also FIXME)
    //     Image gpu_density_map;
    //     Image gpu_projection;
    //     Image gpu_ctf_image;
    //     Image gpu_particle_image;

    //     Image gpu_search_density_map;
    //     Image gpu_search_projection;
    //     Image gpu_search_ctf_image;
    //     Image gpu_search_particle_image;

    //     Image clean_copy;

#endif

    bool is_allocated_weighted_correlation_buffers;

    bool current_cpu_pointers_are_for_global_search;
    bool is_allocated_gpu_density_map;
    bool is_allocated_gpu_projection;
    bool is_allocated_gpu_ctf_image;
    bool is_allocated_gpu_particle_image;

    bool is_allocated_gpu_search_density_map;
    bool is_allocated_gpu_search_projection;
    bool is_allocated_gpu_search_ctf_image;
    bool is_allocated_gpu_search_particle_image;

#ifdef CISTEM_DEBUG
    int nprj;
    // Get some extra info to make sure all the allocation/deallocation is working. I.e. even if we succeed (no segfaults and correct results)
    // we still want to be sure we aren't alloc/dealloc or copying data around unecessarily.
    // Note: the density_maps have debug asserts for > 1 allocation
    int n_particle_image_allocations;
    int n_projection_image_allocations;
    int n_ctf_image_allocations;

    int n_search_particle_image_allocations;
    int n_search_projection_image_allocations;
    int n_search_ctf_image_allocations;

    int n_particle_image_HtoD_copies;
    int n_projection_image_HtoD_copies;
    int n_ctf_image_HtoD_copies;

    int n_search_particle_image_HtoD_copies;
    int n_search_projection_image_HtoD_copies;
    int n_search_ctf_image_HtoD_copies;

    int n_calls_to_prep_images;
    int n_calls_to_prep_ctf_images;

    int n_calls_to_prep_search_images;
    int n_calls_to_prep_search_ctf_images;
#endif

    // Intended to be constructed *inside* any parallel region. (Copying is blocked below.)
    // Thread safety is the name of the game. And sure, using omp private() should work if careful, but by mess with it?
    ProjectionComparisonObjects( );
    ~ProjectionComparisonObjects( );
    void Deallocate( );

    // Explicitly define copy and assignment operators to prevent copying of pointers.
    ProjectionComparisonObjects(const ProjectionComparisonObjects& other_pcos);
    ProjectionComparisonObjects& operator=(const ProjectionComparisonObjects& t);
    ProjectionComparisonObjects& operator=(const ProjectionComparisonObjects* t);

    inline void SetInitialAnglesAndShifts(Particle& wanted_particle_local) {
        initial_x_shift     = wanted_particle_local.alignment_parameters.ReturnShiftX( );
        initial_y_shift     = wanted_particle_local.alignment_parameters.ReturnShiftY( );
        initial_psi_angle   = wanted_particle_local.alignment_parameters.ReturnPsiAngle( );
        initial_theta_angle = wanted_particle_local.alignment_parameters.ReturnThetaAngle( );
        initial_phi_angle   = wanted_particle_local.alignment_parameters.ReturnPhiAngle( );
    }

    inline void SetHostPointers(ReconstructedVolume* input_3d_local, Image* projection_image_local, Particle* refine_particle_local, bool is_for_global_search) {
        current_cpu_pointers_are_for_global_search = is_for_global_search ? true : false;

        reference_volume = input_3d_local;
        projection_image = projection_image_local;
        particle         = refine_particle_local;
    }

    // These are used in the projection step. ifndef ENABLEGPU, they are just no-ops.
    float DoGpuProjection( );
    void  PrepareGpuImages(Particle& host_particle, Image& host_projection_image, const bool is_for_global_search, c_img_t image_type = c_img_t::particle_image_t);
    void  PrepareGpuCTFImages(Particle& host_particle, const bool is_for_global_search);
    template <class InputVolumeType>
    void PrepareGpuVolumeProjection(InputVolumeType& input_3d_local, const bool is_for_global_search);

    // In the ctf refinement loop, we want to keep a clean copy of the particle image and just copy it back each loop
    void SetCleanCopyOfParticleImage(const bool is_for_global_search);

    void DeallocateCleanCopyOfParticleImage( );

    void GetCleanCopyOfParticleImage(const bool is_for_global_search);

    void AllocateBuffers(int new_buffer_size);

    Particle*            particle;
    ReconstructedVolume* reference_volume;
    Image*               projection_image;
    bool                 swap_quadrants, apply_shifts, whiten, apply_ctf, absolute_ctf;

    float mask_radius;
    float mask_falloff;
    float x_shift_limit;
    float y_shift_limit;
    float angle_change_limit;

    float initial_x_shift;
    float initial_y_shift;
    float initial_psi_angle;
    float initial_phi_angle;
    float initial_theta_angle;

    int    old_buffer_size;
    float* buffer;
};

#endif // _SRC_PROGRAMS_REFINE3D_PROJECTION_COMPARISON_OBJECTS_H_
