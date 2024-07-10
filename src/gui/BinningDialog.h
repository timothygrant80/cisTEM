#ifndef __BINNING_DIALOG_H__
#define __BINNING_DIALOG_H__

class BinningDialog : public BinningDialogParent {
  public:
    CombinedPackageClassSelectionPanel*   class_selection_panel;
    CombinedPackageRefinementSelectPanel* refinement_selection_panel;
    ClassVolumeSelectPanel*               initial_reference_panel;

    BinningDialog(wxWindow* parent);
    ~BinningDialog( );

    void OnOK(wxCommandEvent& event);
    void OnCancel(wxCommandEvent& event);
    void OnDesiredPixelSizeChanged(wxCommandEvent& event);

  private:
    float actual_pixel_size;
    float previously_entered_pixel_size;
    int   actual_box_size;
};
#endif