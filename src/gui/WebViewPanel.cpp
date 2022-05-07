//#include "../core/core_headers.h"
#include "../core/cistem_constants.h"
#include "../core/gui_core_headers.h"
#include "wx/webview.h"
#include "WebViewPanel.h"

// extern MyMovieAssetPanel *movie_asset_panel;
//extern MyImageAssetPanel*         image_asset_panel;
//extern MyVolumeAssetPanel*        volume_asset_panel;
//extern MyRunProfilesPanel*        run_profiles_panel;
//extern MyMainFrame*               main_frame;
//extern MatchTemplateResultsPanel* match_template_results_panel;

WebViewPanel::WebViewPanel(wxWindow* parent)
    : WebViewPanelParent(parent) {
    // Set variables
    m_browser = wxWebView::New(this, wxID_ANY, "http://jojoelfe.github.io/webgl-ctf");
    bSizer43->Add(m_browser, wxSizerFlags( ).Expand( ).Proportion(1));
}
