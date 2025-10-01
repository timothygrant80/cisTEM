#include "../core/gui_core_headers.h"
#include "workflows/TmWorkflow.h"
#include "workflows/WorkflowRegistry.h"

ActionsPanelTm::ActionsPanelTm(wxWindow* parent, wxWindowID id, const wxPoint& pos, const wxSize& size, long style)
    : ActionsPanelParent(parent, id, pos, size, style) {
    // Parent class already connects the event handler, no need to bind again
}

ActionsPanelTm::~ActionsPanelTm( ) {
    // CRITICAL: Nullify all global panel pointers to prevent segfaults during workflow switching.
    //
    // This destructor mirrors the safety mechanism in ActionsPanelSpa::~ActionsPanelSpa().
    // Template Matching workflow uses a different subset of panels than Single Particle,
    // but the same dangling pointer issue applies.
    //
    // Key differences from Single Particle workflow:
    // - Uses match_template_panel and refine_template_panel (TM-specific)
    // - Doesn't use classification, refine_3d, ab_initio panels (SPA-specific)
    // - Shares some common panels (align_movies, findctf, generate_3d, sharpen_3d)
    //
    // The segfault prevention strategy remains the same:
    // 1. These panels are children of ActionsBook and will be auto-deleted
    // 2. Global pointers must be nullified to prevent dangling references
    // 3. MainFrame::Dirty*() methods will safely skip null pointers
    //
    // Note: Only nullify panels that actually exist in this workflow to avoid
    // accidentally clearing pointers that might be managed elsewhere.

    align_movies_panel    = nullptr;
    findctf_panel         = nullptr;
    match_template_panel  = nullptr;
    refine_template_panel = nullptr;
    generate_3d_panel     = nullptr;
    sharpen_3d_panel      = nullptr;
}

void ActionsPanelTm::OnActionsBookPageChanged(wxListbookEvent& event) {

    extern MyAlignMoviesPanel*        align_movies_panel;
    extern MyFindCTFPanel*            findctf_panel;
    extern MatchTemplatePanel*        match_template_panel;
    extern MatchTemplateResultsPanel* match_template_results_panel;
    extern RefineTemplatePanel*       refine_template_panel;
    extern Generate3DPanel*           generate_3d_panel;
    extern Sharpen3DPanel*            sharpen_3d_panel;

#ifdef __WXOSX__
    // Necessary for MacOS to refresh the panels
    if ( event.GetSelection( ) == 0 ) {
        match_template_panel->Layout( );
        match_template_panel->Refresh( );
    }
    else if ( event.GetSelection( ) == 1 ) {
        match_template_results_panel->Layout( );
        match_template_results_panel->Refresh( );
    }
    else if ( event.GetSelection( ) == 2 ) {
        refine_template_panel->Layout( );
        refine_template_panel->Refresh( );
    }
#endif
}
