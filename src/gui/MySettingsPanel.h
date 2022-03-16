class MySettingsPanel : public SettingsPanel {
  public:
    MySettingsPanel(wxWindow* parent, wxWindowID id = wxID_ANY, const wxPoint& pos = wxDefaultPosition, const wxSize& size = wxSize(500, 300), long style = wxTAB_TRAVERSAL);
    void OnSettingsBookPageChanged(wxBookCtrlEvent& event);
};
