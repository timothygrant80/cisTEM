#include "../core/gui_core_headers.h"
#include "workflows/TmWorkflow.h"
#include "workflows/WorkflowRegistry.h"

ActionsPanelTm::ActionsPanelTm(wxWindow* parent, wxWindowID id, const wxPoint& pos, const wxSize& size, long style)
    : ActionsPanelParent(parent, id, pos, size, style) {
    wxPrintf("In actions panel TM Const\n");
    // Parent class already connects the event handler, no need to bind again
}

ActionsPanelTm::~ActionsPanelTm( ) {
    wxPrintf("In the actions panel TM destructor\n");
    // Set global pointers to nullptr since the panels will be destroyed with ActionsBook
    // This prevents the next workflow from trying to access destroyed panels
    align_movies_panel = nullptr;
    findctf_panel = nullptr;
    match_template_panel = nullptr;
    refine_template_panel = nullptr;
    generate_3d_panel = nullptr;
    sharpen_3d_panel = nullptr;
}

// TODO: destructor

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
