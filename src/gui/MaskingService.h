#ifndef _MASKING_SERVICE_H_
#define _MASKING_SERVICE_H_

#include <wx/window.h>
#include <wx/thread.h>
#include <atomic>
#include <memory>
#include <thread>

class AutoRefinementManager;
class RefinementManager;
class AbInitioManager;

template <typename PanelType>
void DispatchMasking(PanelType* panel);

// void DispatchMasking(AutoRefinementManager* manager);
// void DispatchMasking(RefinementManager* manager);
// void DispatchMasking(AbInitioManager* manager);

struct MaskingParams {
    // Input/output files
    wxArrayString input_files;
    wxArrayString output_files;

    wxString mask_filename;

    // Mode flags
    bool use_user_mask = false;
    bool apply_blush   = false;

    // Mask settings
    float cosine_edge_width   = 0.0f;
    float weight_outside_mask = 0.0f;
    float low_pass_radius     = 0.0f;
    float mask_radius         = 0.0f;

    // Resolution info
    float pixel_size           = 0.0f;
    float estimated_resolution = 0.0f;

    // Threading
    int num_blush_threads = 1;
    int thread_id         = -1;

    // Control
    int                                batch_size = 1;
    std::shared_ptr<std::atomic<bool>> stop_flag  = nullptr;
};

class MaskingService {
  public:
    static wxThread* StartMasking(
            wxWindow*      parent,
            MaskingParams& params);

    //   private:
};

void StopAndDestroyMaskingThread(wxThread*& masking_thread);

#endif