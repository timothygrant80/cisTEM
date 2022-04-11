#include "../core/gui_core_headers.h"
#ifdef EXPERIMENTAL
#include "wx/webview.h"
#include "WebViewPanel.h"
#endif

MyExperimentalPanel::MyExperimentalPanel(wxWindow* parent, wxWindowID id, const wxPoint& pos, const wxSize& size, long style)
    : ExperimentalPanel(parent, id, pos, size, style) {
    // Bind OnListBookPageChanged from
    Bind(wxEVT_LISTBOOK_PAGE_CHANGED, wxBookCtrlEventHandler(MyExperimentalPanel::OnExperimentalBookPageChanged), this);
}

// TODO: destructor

void MyExperimentalPanel::OnExperimentalBookPageChanged(wxBookCtrlEvent& event) {
    extern RefineTemplateDevPanel* refine_template_dev_panel;
    extern WebViewPanel* web_view_panel;

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
