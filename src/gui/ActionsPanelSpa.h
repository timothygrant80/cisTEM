#ifndef gui_ActionsPanelSpa_h_
#define gui_ActionsPanelSpa_h_

class ActionsPanelSpa : public ActionsPanelParent {
  public:
    ActionsPanelSpa(wxWindow* parent, wxWindowID id = wxID_ANY, const wxPoint& pos = wxDefaultPosition, const wxSize& size = wxSize(500, 300), long style = wxTAB_TRAVERSAL);
    ~ActionsPanelSpa( ); // to add debug prints to confirm destruction.
    virtual void OnActionsBookPageChanged(wxListbookEvent& event) override;
};

#endif
