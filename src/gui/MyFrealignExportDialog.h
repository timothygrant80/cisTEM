#ifndef __MyFrealignExportDialog__
#define __MyFrealignExportDialog__

class MyFrealignExportDialog : public FrealignExportDialog {
  public:
    MyFrealignExportDialog(wxWindow* parent);

    void OnCancelButtonClick(wxCommandEvent& event);
    void OnExportButtonClick(wxCommandEvent& event);
    void OnOutputImageStackFileChanged(wxFileDirPickerEvent& event);
};

#endif
