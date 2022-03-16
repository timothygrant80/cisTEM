#ifndef __MyParticlePositionExportDialog__
#define __MyParticlePositionExportDialog__

class MyParticlePositionExportDialog : public ParticlePositionExportDialog {
  public:
    MyParticlePositionExportDialog(wxWindow* parent);

    void OnCancelButtonClick(wxCommandEvent& event);
    void OnExportButtonClick(wxCommandEvent& event);
    void OnDirChanged(wxFileDirPickerEvent& event);
};

#endif
