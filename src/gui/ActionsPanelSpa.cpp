#include "../core/gui_core_headers.h"
#include "workflows/SpaWorkflow.h"
#include "workflows/WorkflowRegistry.h"

ActionsPanelSpa::ActionsPanelSpa(wxWindow* parent, wxWindowID id, const wxPoint& pos, const wxSize& size, long style)
    : ActionsPanelParent(parent, id, pos, size, style) {
    // Bind OnListBookPageChanged from
    Bind(wxEVT_LISTBOOK_PAGE_CHANGED, wxBookCtrlEventHandler(ActionsPanelSpa::OnActionsBookPageChanged), this);
}

// TODO: destructor

void ActionsPanelSpa::OnActionsBookPageChanged(wxBookCtrlEvent& event) {

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
