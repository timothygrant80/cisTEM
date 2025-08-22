#ifndef _SRC_GUI_RESULTS_PANEL_TM_H_
#define _SRC_GUI_RESULTS_PANEL_TM_H_

// Temporary forward declaration
class ResultsPanelParent;

class ResultsPanelTm : public ResultsPanelParent {
  public:
    ResultsPanelTm(wxWindow* parent, wxWindowID id = wxID_ANY, const wxPoint& pos = wxDefaultPosition, const wxSize& size = wxSize(500, 300), long style = wxTAB_TRAVERSAL);
    void OnResultsBookPageChanged(wxBookCtrlEvent& event);
};

#endif