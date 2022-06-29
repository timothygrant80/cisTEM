#ifndef _gui_WebViewPanel_h_
#define _gui_WebViewPanel_h_

#if wxUSE_WEBVIEW
#warning "wxWebView is enabled"
#else
#warning "wxWebView is disabled"
#endif

// Note this will not build because wxUSE_WEBVIEW is not defined, so the following header is effectively skipped.
// There must be a missing backend component in the build of wxWidgets?

#include <wx/webview.h>

// #include "/opt/WX/intel-dynamic/include/wx-3.0/wx/webview.h"

class WebViewPanel : public WebViewPanelParent {

  public:
    WebViewPanel(wxWindow* parent);
    wxWebView* m_browser;
};

#endif
