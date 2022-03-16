#ifndef __ClassumSelectionCopyFromDialog__
#define __ClassumSelectionCopyFromDialog__

class ClassumSelectionCopyFromDialog : public ClassumSelectionCopyFromDialogParent {
  public:
    ClassumSelectionCopyFromDialog(wxWindow* parent);
    void OnOKButtonClick(wxCommandEvent& event);
    void OnCancelButtonClick(wxCommandEvent& event);
    void FillWithSelections(int number_of_classes);

    int ReturnSelectedPosition( ) { return selected_selection_array_position; };

    wxArrayInt original_array_positions;
    int        selected_selection_array_position;
};

#endif // __ClassumSelectionCopyFromDialog__
