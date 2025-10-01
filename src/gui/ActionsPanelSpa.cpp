#include "../core/gui_core_headers.h"
#include "workflows/SpaWorkflow.h"
#include "workflows/WorkflowRegistry.h"

ActionsPanelSpa::ActionsPanelSpa(wxWindow* parent, wxWindowID id, const wxPoint& pos, const wxSize& size, long style)
    : ActionsPanelParent(parent, id, pos, size, style) {
    // Parent class already connects the event handler, no need to bind again
}

ActionsPanelSpa::~ActionsPanelSpa( ) {
    // CRITICAL: Nullify all global panel pointers to prevent segfaults during workflow switching.
    //
    // When switching workflows (e.g., from Single Particle to Template Matching), the following sequence occurs:
    // 1. The current ActionsPanelSpa and all its child panels are destroyed
    // 2. These panels are wxWidgets children of ActionsBook, so they're automatically deleted
    // 3. However, global pointers to these panels persist and become dangling pointers
    // 4. Various MainFrame::Dirty*() methods may be called during or after workflow switch
    // 5. These methods check panel pointers and try to set dirty flags if non-null
    // 6. Without nullifying here, they would dereference freed memory → segfault
    //
    // This issue was particularly tricky because:
    // - The segfault was inconsistent (depended on memory reuse patterns)
    // - It often occurred several UI operations after the actual workflow switch
    // - The crash location varied (any Dirty*() method could trigger it)
    //
    // By explicitly nullifying these pointers in the destructor, we ensure that:
    // - Dirty*() methods safely skip destroyed panels (null check fails)
    // - The new workflow can create fresh panel instances without conflicts
    // - Memory access violations are prevented during the transition period

    align_movies_panel   = nullptr;
    findctf_panel        = nullptr;
    findparticles_panel  = nullptr;
    classification_panel = nullptr;
    refine_3d_panel      = nullptr;
    refine_ctf_panel     = nullptr;
    auto_refine_3d_panel = nullptr;
    ab_initio_3d_panel   = nullptr;
    generate_3d_panel    = nullptr;
    sharpen_3d_panel     = nullptr;
}

void ActionsPanelSpa::OnActionsBookPageChanged(wxListbookEvent& event) {

    extern MyAlignMoviesPanel*   align_movies_panel;
    extern MyFindCTFPanel*       findctf_panel;
    extern MyFindParticlesPanel* findparticles_panel;
    extern MyRefine2DPanel*      classification_panel;
    extern AbInitio3DPanel*      ab_initio_3d_panel;
    extern AutoRefine3DPanel*    auto_refine_3d_panel;
    extern MyRefine3DPanel*      refine_3d_panel;
    extern RefineCTFPanel*       refine_ctf_panel;
    extern Generate3DPanel*      generate_3d_panel;
    extern Sharpen3DPanel*       sharpen_3d_panel;

#ifdef __WXOSX__
    // Necessary for MacOS to refresh the panels
    if ( event.GetSelection( ) == 0 ) {
        align_movies_panel->Layout( );
        align_movies_panel->Refresh( );
    }
    else if ( event.GetSelection( ) == 1 ) {
        findctf_panel->Layout( );
        findctf_panel->Refresh( );
    }
    else if ( event.GetSelection( ) == 2 ) {
        findparticles_panel->Layout( );
        findparticles_panel->Refresh( );
    }
    else if ( event.GetSelection( ) == 3 ) {
        classification_panel->Layout( );
        classification_panel->Refresh( );
    }
    else if ( event.GetSelection( ) == 4 ) {
        ab_initio_3d_panel->Layout( );
        ab_initio_3d_panel->Refresh( );
    }
    else if ( event.GetSelection( ) == 5 ) {
        auto_refine_3d_panel->Layout( );
        auto_refine_3d_panel->Refresh( );
    }
    else if ( event.GetSelection( ) == 6 ) {
        refine_3d_panel->Layout( );
        refine_3d_panel->Refresh( );
    }
    else if ( event.GetSelection( ) == 7 ) {
        refine_ctf_panel->Layout( );
        refine_ctf_panel->Refresh( );
    }
    else if ( event.GetSelection( ) == 8 ) {
        generate_3d_panel->Layout( );
        generate_3d_panel->Refresh( );
    }
    else if ( event.GetSelection( ) == 9 ) {
        sharpen_3d_panel->Layout( );
        sharpen_3d_panel->Refresh( );
    }
#endif
}
