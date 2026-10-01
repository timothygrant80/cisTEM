#include "MaskingService.h"
#include "../core/gui_core_headers.h"

/**
 * @brief Creates and returns a masking thread that will execute the masking and blush inference,
 * if selected by the user.
 * 
 * @param parent The panel generating the masking thread.
 * @param params The input parameters loaded from the GUI into a simplified struct object.
 * @return wxThread* Pointer to the running masking/blush inference thread
 */
wxThread* MaskingService::StartMasking(wxWindow* parent, MaskingParams& params) {
    wxThread* masking_thread = nullptr;

    if ( params.use_user_mask ) {
        masking_thread = new Multiply3DMaskerThread(
                parent,
                params.input_files,
                params.output_files,
                params.mask_filename,
                params.cosine_edge_width,
                params.weight_outside_mask,
                params.low_pass_radius,
                params.pixel_size,
                params.mask_radius,
                params.thread_id);
    }
    else {
        wxPrintf("StartMasking(): params.thread_id == %i\n", params.thread_id);
        masking_thread = new AutoMaskerThread(
                parent,
                params.input_files,
                params.output_files,
                params.pixel_size,
                params.mask_radius,
                params.estimated_resolution);
    }

    if ( masking_thread && masking_thread->Create( ) != wxTHREAD_NO_ERROR ) {
        delete masking_thread;
        return nullptr;
    }

    if ( masking_thread && masking_thread->Run( ) != wxTHREAD_NO_ERROR ) {
        StopAndDestroyMaskingThread(masking_thread);
    }

    return masking_thread;
}

/**
 * @brief 
 * 
 * @tparam PanelType The type of 3D refinement panel that might call the masking thread
 * @param panel The GUI object desiring the masking thread
 */
template <typename PanelType>
void DispatchMasking(PanelType* panel) {
    MaskingParams params;

    // Vals common between panel managers first
    params.pixel_size = panel->my_refinement_manager.input_refinement->resolution_statistics_pixel_size;

    // Common between all panels
    params.thread_id             = panel->next_thread_id++;
    params.stop_flag             = panel->stop_flag;
    panel->active_mask_thread_id = params.thread_id;

    // Auto refine and manual refine have the same naming
    if constexpr ( std::is_same_v<PanelType, AutoRefine3DPanel> || std::is_same_v<PanelType, MyRefine3DPanel> ) {
        params.mask_filename = panel->my_refinement_manager.active_mask_filename;
        params.mask_radius   = panel->my_refinement_manager.active_mask_radius;
        params.use_user_mask = panel->my_refinement_manager.active_should_mask;

        if ( params.use_user_mask ) {
            panel->WriteInfoText("Masking reference reconstruction with selected mask...");
        }
        else {
            panel->WriteInfoText("Automasking reference reconstruction...");
        }

        for ( size_t filename_counter = 0; filename_counter < panel->my_refinement_manager.current_reference_filenames.GetCount( ); filename_counter++ ) {
            wxFileName ref = panel->my_refinement_manager.current_reference_filenames.Item(filename_counter);

            // Make sure output files are going to the right Scratch directory -- AutoRefine3D or ManualRefine3D
            wxString out;
            if constexpr ( std::is_same_v<PanelType, AutoRefine3DPanel> ) {
                out = main_frame->ReturnAutoRefine3DScratchDirectory( ) + ref.GetName( ) + "_masked.mrc";
            }
            else if constexpr ( std::is_same_v<PanelType, MyRefine3DPanel> ) {
                out = main_frame->ReturnRefine3DScratchDirectory( ) + ref.GetName( ) + "_masked.mrc";
            }

            params.input_files.Add(ref.GetFullPath( ));
            params.output_files.Add(out);
        }

        if ( params.use_user_mask ) {
            params.cosine_edge_width   = panel->my_refinement_manager.active_mask_edge;
            params.weight_outside_mask = panel->my_refinement_manager.active_mask_weight;

            if ( panel->my_refinement_manager.active_should_low_pass_filter_mask ) {
                params.low_pass_radius = panel->my_refinement_manager.active_mask_filter_resolution;
            }
            else {
                params.low_pass_radius = 0.0f;
            }
        }
        else {
            float current_res = panel->my_refinement_manager.input_refinement->class_refinement_results[0].class_resolution_statistics.ReturnEstimatedResolution(true);
            if constexpr ( std::is_same_v<PanelType, AutoRefine3DPanel> ) {
                if ( current_res > panel->my_refinement_manager.class_high_res_limits[0] )
                    current_res = panel->my_refinement_manager.class_high_res_limits[0];
            }
            params.estimated_resolution = current_res;
        }
    }

    // AbInitio has distinct differences
    else if constexpr ( std::is_same_v<PanelType, AbInitio3DPanel> ) {
        params.mask_radius          = panel->my_refinement_manager.active_global_mask_radius;
        params.use_user_mask        = false;
        params.estimated_resolution = 0.0;
        params.low_pass_radius      = 0.0;

        for ( size_t filename_counter = 0; filename_counter < panel->my_refinement_manager.current_reference_filenames.GetCount( ); filename_counter++ ) {
            wxFileName ref = panel->my_refinement_manager.current_reference_filenames.Item(filename_counter);
            params.input_files.Add(ref.GetFullPath( ));
            ref.ClearExt( );
            params.output_files.Add(ref.GetFullPath( ) + "_masked.mrc");
        }
    }

    if ( panel->masking_thread ) {
        if ( panel->masking_thread->IsRunning( ) ) {
            panel->masking_thread->Delete( );
        }
        panel->masking_thread->Wait( );
        delete panel->masking_thread;
        panel->masking_thread = nullptr;
    }

    panel->masking_thread = MaskingService::StartMasking(panel, params);

    // If process didn't fail, ensure that subsequent iterations use the proper files as references.
    if ( panel->masking_thread ) {
        panel->my_refinement_manager.current_reference_filenames = params.output_files;
    }
}

template void DispatchMasking<AutoRefine3DPanel>(AutoRefine3DPanel* panel);
template void DispatchMasking<MyRefine3DPanel>(MyRefine3DPanel* panel);
template void DispatchMasking<AbInitio3DPanel>(AbInitio3DPanel* panel);

/**
 * @brief For destroying an existing masking thread upon joinable thread
 * completion or error during thread execution.
 * 
 * @param masking_thread 
 */
void StopAndDestroyMaskingThread(wxThread*& masking_thread) {
    if ( masking_thread ) {
        masking_thread->Delete( );
        masking_thread->Wait( );
        delete masking_thread;
        masking_thread = nullptr;
    }
}