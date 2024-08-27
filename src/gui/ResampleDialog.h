#ifndef __BINNING_DIALOG_H__
#define __BINNING_DIALOG_H__

class ResampleDialog : public ResampleDialogParent {
  public:
    CombinedPackageClassSelectionPanel*   class_selection_panel;
    CombinedPackageRefinementSelectPanel* refinement_selection_panel;
    ClassVolumeSelectPanel*               initial_reference_panel;

    ResampleDialog(wxWindow* parent, bool resampling_volume);
    ~ResampleDialog( );

    void OnOK(wxCommandEvent& event);
    void OnCancel(wxCommandEvent& event);
    void OnBoxSizeSpinCtrl(wxCommandEvent& event);
    void OnBoxSizeTextEnter(wxCommandEvent& event);

  private:
    bool  resampling_volume;
    float resample_pixel_size;
    int   resample_box_size;
    int   previously_entered_box_size;
};
#endif