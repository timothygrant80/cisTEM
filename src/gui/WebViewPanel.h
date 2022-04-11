#ifndef _gui_WebViewPanel_h_
#define _gui_WebViewPanel_h_

class WebViewPanel : public WebViewPanelParent {
    

  public:
    WebViewPanel(wxWindow* parent);
    wxWebView* m_browser;
    
};

#endif
