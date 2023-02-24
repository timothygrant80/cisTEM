#include <wx/textctrl.h>
#include <wx/checkbox.h>
#include <wx/spinctrl.h>
#include <wx/datectrl.h>
#include <wx/filedlg.h>

#ifndef __MyControls__
#define __MyControls__

class NoFocusBitmapButton : public wxBitmapButton {
  public:
    NoFocusBitmapButton(wxWindow* parent, wxWindowID id, const wxBitmap& bitmap, const wxPoint& pos = wxDefaultPosition, const wxSize& size = wxDefaultSize, long style = wxBU_AUTODRAW, const wxValidator& validator = wxDefaultValidator, const wxString& name = wxButtonNameStr)
        : wxBitmapButton(parent, id, bitmap, pos, size, style, validator, name) {
        Bind(wxEVT_SET_FOCUS, &NoFocusBitmapButton::OnFocus, this);
    }

    bool AcceptsFocus( ) const { return false; }

    bool AcceptsFocusFromKeyboard( ) const { return false; }

    void OnFocus(wxFocusEvent& event) {
        Freeze( );
        Disable( );
        Enable( );
        Thaw( );
    } // nasty hack, I don't want these buttons to focus
};

class MemoryComboBox : public wxOwnerDrawnComboBox {
  public:
    MemoryComboBox(wxWindow* parent, wxWindowID id, const wxString& value = wxEmptyString, const wxPoint& pos = wxDefaultPosition, const wxSize& size = wxDefaultSize, int n = 0, const wxString choices[] = NULL, long style = 0, const wxValidator& validator = wxDefaultValidator, const wxString& name = wxComboBoxNameStr);
    ~MemoryComboBox( );

    wxArrayLong   associated_ids;
    wxArrayString associated_text;

    long selected_id_on_last_clear;
    long currently_selected_id;

    void Clear( );
    void Reset( );
    void AddMemoryItem(wxString wanted_text, long wanted_id);
    void AddMemoryItems(wxArrayString wanted_texts, wxArrayLong wanted_ids);

    void SetSelection(int n);
    void SetSelectionWithEvent(int n);

    bool FillWithRunProfiles( );
    bool FillWithRefinementPackages( );
    bool FillWithVolumeAssets(bool include_generate_from_params = false, bool always_select_newest = false);
#ifdef EXPERIMENTAL
    bool FillWithAtomicCoordinatesAssets(bool include_generate_from_params = false, bool always_select_newest = false);
#endif
    bool FillWithMovieGroups(bool include_all_movies_group = true);
    bool FillWithImageGroups(bool include_all_images_group = true);
    bool FillWithTMJobs(bool include_all_images_group = true);
    bool FillWithImages(long wanted_image_group);
    bool FillWithClassifications(long wanted_refinement_package, bool include_new_classification, bool always_select_newest = false);
    bool FillWithRefinements(long wanted_refinement_package, bool always_select_newest = false);
    bool FillWithClassAverageSelections(bool always_select_newest = false, long wanted_refinement_package_id = -1);

    void OnComboBox(wxCommandEvent& event);
};

class NumericTextCtrl : public wxTextCtrl {

  public:
    NumericTextCtrl(wxWindow* parent, wxWindowID id = wxID_ANY, const wxString& value = wxEmptyString, const wxPoint& pos = wxDefaultPosition, const wxSize& size = wxDefaultSize, long style = 0, const wxValidator& validator = wxDefaultValidator, const wxString& name = wxTextCtrlNameStr);
    ~NumericTextCtrl( );

    float max_value;
    float min_value;
    float previous_value;
    int   precision;

    void SetPrecision(int wanted_precision);
    void ChangeValueFloat(float wanted_float);

    void OnKeyPress(wxKeyEvent& event);
    void OnEnterPressed(wxCommandEvent& event);
    void OnFocusLost(wxFocusEvent& event);

    void  SetMinMaxValue(float wanted_min_value, float wanted_max_value);
    void  CheckValues( );
    float ReturnValue( );
};

class AutoWrapStaticText : public wxStaticText {

  public:
    bool has_autowrapped;
    AutoWrapStaticText(wxWindow* parent, wxWindowID id, const wxString& label, const wxPoint& pos = wxDefaultPosition, const wxSize& size = wxDefaultSize, long style = 0, const wxString& name = wxStaticTextNameStr);
    ~AutoWrapStaticText( );

    void AutoWrap( );
    void OnSize(wxSizeEvent& event);
};

class ClassVolumeSelectPanel : public wxPanel {
  public:
    wxBoxSizer*                  MainSizer;
    wxStaticText*                ClassText;
    VolumeAssetPickerComboPanel* VolumeComboBox;

    int class_number;

    ClassVolumeSelectPanel(wxWindow* parent, wxWindowID id = wxID_ANY, const wxPoint& pos = wxDefaultPosition, const wxSize& size = wxDefaultSize, long style = wxTAB_TRAVERSAL);
    ~ClassVolumeSelectPanel( );
};

class FilterItem : public wxPanel {

  public:
    wxWindow*   my_parent;
    wxBoxSizer* my_sizer;
    wxCheckBox* field_checkbox;

    FilterItem(wxString field_name, wxWindow* parent, wxWindowID id = wxID_ANY, const wxPoint& pos = wxDefaultPosition, const wxSize& size = wxDefaultSize, long style = wxTAB_TRAVERSAL);
    ~FilterItem( );

    virtual void CheckBoxClicked(wxCommandEvent& event) { event.Skip( ); };

    virtual bool IsChecked( ) { return field_checkbox->IsChecked( ); };
};

class IntegerFilterItem : public FilterItem {

  public:
    wxSpinCtrl*   LowSpinCtrl;
    wxStaticText* ToText;
    wxSpinCtrl*   HighSpinCtrl;

    IntegerFilterItem(wxString field_name, wxWindow* parent, wxWindowID id = wxID_ANY, const wxPoint& pos = wxDefaultPosition, const wxSize& size = wxDefaultSize, long style = wxTAB_TRAVERSAL);
    ~IntegerFilterItem( );

    void CheckBoxClicked(wxCommandEvent& event);

    void LowControlChanged(wxCommandEvent& event);
    void HighControlChanged(wxCommandEvent& event);

    int GetLowValue( );
    int GetHighValue( );
};

class FloatFilterItem : public FilterItem {

  public:
    NumericTextCtrl* LowTextCtrl;
    wxStaticText*    ToText;
    NumericTextCtrl* HighTextCtrl;

    float previous_low_value;
    float previous_high_value;

    FloatFilterItem(wxString field_name, wxWindow* parent, wxWindowID id = wxID_ANY, const wxPoint& pos = wxDefaultPosition, const wxSize& size = wxDefaultSize, long style = wxTAB_TRAVERSAL);
    ~FloatFilterItem( );

    void CheckBoxClicked(wxCommandEvent& event);

    void LowControlChanged(wxCommandEvent& event);
    void HighControlChanged(wxCommandEvent& event);
    void LowKillFocus(wxFocusEvent& event);
    void HighKillFocus(wxFocusEvent& event);
    void CheckLowValue( );
    void CheckHighValue( );

    float GetLowValue( );
    float GetHighValue( );
};

class DateFilterItem : public FilterItem {

  public:
    wxDatePickerCtrl* LowDateCtrl;
    wxStaticText*     ToText;
    wxDatePickerCtrl* HighDateCtrl;

    DateFilterItem(wxString field_name, wxWindow* parent, wxWindowID id = wxID_ANY, const wxPoint& pos = wxDefaultPosition, const wxSize& size = wxDefaultSize, long style = wxTAB_TRAVERSAL);
    ~DateFilterItem( );

    void CheckBoxClicked(wxCommandEvent& event);

    void LowControlChanged(wxCommandEvent& event);
    void HighControlChanged(wxCommandEvent& event);

    int GetLowValue( );
    int GetHighValue( );
};

// Dialogs..

class ProperOverwriteCheckSaveDialog : public wxFileDialog {
  public:
    wxString extension_lowercase;
    wxString extension_uppercase;

    wxString overidden_path;

    ProperOverwriteCheckSaveDialog(wxWindow* parent, const wxString& message, const wxString& wildcard, const wxString wanted_extension);
    ~ProperOverwriteCheckSaveDialog( );

    wxString ReturnProperPath( );

    void OnSave(wxCommandEvent& event);
};

// virtual listctrl
class ContentsList : public wxListCtrl {
  public:
    ContentsList(wxWindow* parent, wxWindowID id, const wxPoint& pos = wxDefaultPosition, const wxSize& size = wxDefaultSize, long style = wxLC_ICON, const wxValidator& validator = wxDefaultValidator, const wxString& name = wxListCtrlNameStr);
    wxString OnGetItemText(long item, long column) const;

    int ReturnGuessAtColumnTextWidth(int wanted_column);
};

class RefinementPackageListControl : public wxListCtrl {
  public:
    RefinementPackageListControl(wxWindow* parent, wxWindowID id, const wxPoint& pos = wxDefaultPosition, const wxSize& size = wxDefaultSize, long style = wxLC_ICON, const wxValidator& validator = wxDefaultValidator, const wxString& name = wxListCtrlNameStr);
    wxString OnGetItemText(long item, long column) const;

    int ReturnGuessAtColumnTextWidth( );
};

class ContainedParticleListControl : public wxListCtrl {
  public:
    ContainedParticleListControl(wxWindow* parent, wxWindowID id, const wxPoint& pos = wxDefaultPosition, const wxSize& size = wxDefaultSize, long style = wxLC_ICON, const wxValidator& validator = wxDefaultValidator, const wxString& name = wxListCtrlNameStr);
    wxString OnGetItemText(long item, long column) const;

    int ReturnGuessAtColumnTextWidth(int wanted_column);
};

class ReferenceVolumesListControl : public wxListCtrl {
  public:
    ReferenceVolumesListControl(wxWindow* parent, wxWindowID id, const wxPoint& pos = wxDefaultPosition, const wxSize& size = wxDefaultSize, long style = wxLC_ICON, const wxValidator& validator = wxDefaultValidator, const wxString& name = wxListCtrlNameStr);
    virtual wxString OnGetItemText(long item, long column) const;

    int ReturnGuessAtColumnTextWidth(int wanted_column);
};

class ReferenceVolumesListControlRefinement : public ReferenceVolumesListControl {
  public:
    ReferenceVolumesListControlRefinement(wxWindow* parent, wxWindowID id, const wxPoint& pos = wxDefaultPosition, const wxSize& size = wxDefaultSize, long style = wxLC_ICON, const wxValidator& validator = wxDefaultValidator, const wxString& name = wxListCtrlNameStr);
    wxString OnGetItemText(long item, long column) const;

    RefinementPackagePickerComboPanel* refinement_package_picker_to_use;
};

class RefinementParametersDialog;

class RefinementParametersListCtrl : public wxListCtrl {

    RefinementParametersDialog* parent_dialog;

  public:
    RefinementParametersListCtrl(wxWindow* parent, wxWindowID id, const wxPoint& pos = wxDefaultPosition, const wxSize& size = wxDefaultSize, long style = wxLC_ICON, const wxValidator& validator = wxDefaultValidator, const wxString& name = wxListCtrlNameStr);
    virtual wxString OnGetItemText(long item, long column) const;

    int  ReturnGuessAtColumnTextWidth(int wanted_column);
    void SetParent(RefinementParametersDialog* wanted_parent);
};

class ClassificationSelectionListCtrl : public wxListCtrl {
  public:
    ClassificationSelectionListCtrl(wxWindow* parent, wxWindowID id, const wxPoint& pos = wxDefaultPosition, const wxSize& size = wxDefaultSize, long style = wxLC_ICON, const wxValidator& validator = wxDefaultValidator, const wxString& name = wxListCtrlNameStr);
    wxString OnGetItemText(long item, long column) const;

    int ReturnGuessAtColumnTextWidth(int wanted_column);

    long ReturnCurrentSelection( ) { return current_selection; };

    int ReturnCurrentSelectionOriginalArrayPosition( ) {
        if ( current_selection == -1 )
            return -1;
        else
            return original_classum_selection_array_positions.Item(current_selection);
    };

    void OnClearAll(wxListEvent& event);

    void Fill(long refinement_package_asset_id, long classification_id, bool select_latest = false);

    long current_selection;
    long current_selection_id;
    long selection_id_upon_clear;

    long position_being_edited;

    ArrayofClassificationSelections all_valid_selections;
    wxArrayInt                      original_classum_selection_array_positions;
};

class OneSecondProgressDialog : public wxProgressDialog {

    long time_of_last_update;

  public:
    OneSecondProgressDialog(const wxString& title, const wxString& message, int maximum = 100, wxWindow* parent = NULL, int style = wxPD_APP_MODAL | wxPD_AUTO_HIDE);
    bool Update(int value, const wxString& newmsg = wxEmptyString, bool* skip = NULL);
};

class CombinedPackageClassSelectionPanel : public wxPanel {
  public:
    wxComboBox*   ClassComboBox;
    wxStaticText* ClassText;
    wxBoxSizer*   MainSizer;
    wxBoxSizer*   bSizer989;

    CombinedPackageClassSelectionPanel(wxWindow* parent, wxWindowID id = wxID_ANY, const wxPoint& pos = wxDefaultPosition, const wxSize& size = wxDefaultSize, long style = wxTAB_TRAVERSAL);
    ~CombinedPackageClassSelectionPanel( );
    void FillComboBox(long wanted_refinement_package);
};

class CombinedPackageRefinementSelectPanel : public wxPanel {
  public:
    RefinementPickerComboPanel* RefinementComboBox;
    wxStaticText*               RefinementText;
    wxBoxSizer*                 MainSizer;
    wxBoxSizer*                 bSizer989;

    CombinedPackageRefinementSelectPanel(wxWindow* parent, wxWindowID id = wxID_ANY, const wxPoint& pos = wxDefaultPosition, const wxSize& size = wxDefaultSize, long style = wxTAB_TRAVERSAL);
    ~CombinedPackageRefinementSelectPanel( );

  private:
    bool has_random_parameters;
};

#endif
