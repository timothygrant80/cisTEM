#ifndef __DISPLAYREFINEMENTRESULTS_PANEL_H__
#define __DISPLAYREFINEMENTRESULTS_PANEL_H__

class DisplayRefinementResultsPanel : public DisplayRefinementResultsPanelParent {
  public:
    DisplayRefinementResultsPanel(wxWindow* parent, wxWindowID id = wxID_ANY, const wxPoint& pos = wxDefaultPosition, const wxSize& size = wxDefaultSize, long style = wxTAB_TRAVERSAL, const wxString& name = wxPanelNameStr);

    void Clear( );
};

class DisplayCTFRefinementResultsPanel : public DisplayCTFRefinementResultsPanelParent {
  public:
    DisplayCTFRefinementResultsPanel(wxWindow* parent, wxWindowID id = wxID_ANY, const wxPoint& pos = wxDefaultPosition, const wxSize& size = wxDefaultSize, long style = wxTAB_TRAVERSAL, const wxString& name = wxPanelNameStr);

    void Clear( );
};

#endif
