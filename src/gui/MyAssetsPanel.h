class MyAssetsPanel : public AssetsPanel {
  public:
    MyAssetsPanel(wxWindow* parent, wxWindowID id = wxID_ANY, const wxPoint& pos = wxDefaultPosition, const wxSize& size = wxSize(500, 300), long style = wxTAB_TRAVERSAL);
    void OnAssetsBookPageChanged(wxBookCtrlEvent& event);
};
