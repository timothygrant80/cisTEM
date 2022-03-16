#ifndef __MyOverviewPanel__
#define __MyOverviewPanel__

class MyOverviewPanel : public OverviewPanel {
  public:
    MyOverviewPanel(wxWindow* parent, wxWindowID id = wxID_ANY, const wxPoint& pos = wxDefaultPosition, const wxSize& size = wxSize(500, 300), long style = wxTAB_TRAVERSAL);

    void SetWelcomeInfo( );
    void SetProjectInfo( );
    void OnInfoURL(wxTextUrlEvent& event);
};

#endif // __MyOverviewPanel__
