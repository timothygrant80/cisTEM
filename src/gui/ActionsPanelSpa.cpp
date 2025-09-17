#include "../core/gui_core_headers.h"
#include "workflows/SpaWorkflow.h"
#include "workflows/WorkflowRegistry.h"

ActionsPanelSpa::ActionsPanelSpa(wxWindow* parent, wxWindowID id, const wxPoint& pos, const wxSize& size, long style)
    : ActionsPanelParent(parent, id, pos, size, style) {
    wxPrintf("In actions panel SPA Const\n");
    // Parent class already connects the event handler, no need to bind again
}

ActionsPanelSpa::~ActionsPanelSpa( ) {
    wxPrintf("In the actions panel SPA destructor\n");
    // Set global pointers to nullptr since the panels will be destroyed with ActionsBook
    // This prevents the next workflow from trying to access destroyed panels
    align_movies_panel = nullptr;
    findctf_panel = nullptr;
    findparticles_panel = nullptr;
    classification_panel = nullptr;
    refine_3d_panel = nullptr;
    refine_ctf_panel = nullptr;
    auto_refine_3d_panel = nullptr;
    ab_initio_3d_panel = nullptr;
    generate_3d_panel = nullptr;
    sharpen_3d_panel = nullptr;
}

// TODO: destructor

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
