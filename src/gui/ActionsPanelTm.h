#ifndef _gui_ActionsPanelTm_h_
#define _gui_ActionsPanelTm_h_

class ActionsPanelTm : public ActionsPanelParent {
  public:
    ActionsPanelTm(wxWindow* parent, wxWindowID id = wxID_ANY, const wxPoint& pos = wxDefaultPosition, const wxSize& size = wxSize(500, 300), long style = wxTAB_TRAVERSAL);
    ~ActionsPanelTm( ); // to add debug prints to confirm destruction.

    virtual void OnActionsBookPageChanged(wxListbookEvent& event) override;
};

#endif
