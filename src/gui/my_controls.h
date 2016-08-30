#include <wx/textctrl.h>
#include <wx/checkbox.h>
#include <wx/spinctrl.h>
#include <wx/datectrl.h>
#include <wx/filedlg.h>


#ifndef __MyControls__
#define __MyControls__

class NumericTextCtrl : public wxTextCtrl
{

public :

	NumericTextCtrl(wxWindow *parent, wxWindowID id = wxID_ANY, const wxString &value=wxEmptyString, const wxPoint &pos=wxDefaultPosition, const wxSize &size=wxDefaultSize, long style=0, const wxValidator &validator=wxDefaultValidator, const wxString &name=wxTextCtrlNameStr);
	~NumericTextCtrl();

	float max_value;
	float min_value;

	float previous_value;

	void OnKeyPress( wxKeyEvent& event );
	void OnEnterPressed( wxCommandEvent& event);
	void OnFocusLost( wxFocusEvent& event );

	void SetMinMaxValue(float wanted_min_value, float wanted_max_value);
	void CheckValues();
	float ReturnValue();

};

class AutoWrapStaticText : public wxStaticText
{

public :

	bool has_autowrapped;
	AutoWrapStaticText (wxWindow *parent, wxWindowID id, const wxString &label, const wxPoint &pos=wxDefaultPosition, const wxSize &size=wxDefaultSize, long style=0, const wxString &name=wxStaticTextNameStr);
	~AutoWrapStaticText();

	void AutoWrap();
	void OnSize(wxSizeEvent& event);
};


class ClassVolumeSelectPanel : public wxPanel
{
	public:

		wxBoxSizer* MainSizer;
		wxStaticText* ClassText;
		wxComboBox* VolumeComboBox;

		int class_number;

		ClassVolumeSelectPanel( wxWindow* parent, wxWindowID id = wxID_ANY, const wxPoint& pos = wxDefaultPosition, const wxSize& size = wxSize( 500,300 ), long style = wxTAB_TRAVERSAL );
		~ClassVolumeSelectPanel();
};


class FilterItem : public wxPanel
{

	public:

	wxWindow   *my_parent;
	wxBoxSizer *my_sizer;
	wxCheckBox *field_checkbox;


	FilterItem( wxString field_name, wxWindow* parent, wxWindowID id = wxID_ANY, const wxPoint& pos = wxDefaultPosition, const wxSize& size = wxDefaultSize, long style = wxTAB_TRAVERSAL );
	~FilterItem();
	virtual void CheckBoxClicked( wxCommandEvent& event) {event.Skip();};
	virtual bool IsChecked () {return field_checkbox->IsChecked();};


};

class IntegerFilterItem : public FilterItem
{

public :

	wxSpinCtrl *LowSpinCtrl;
	wxStaticText *ToText;
	wxSpinCtrl *HighSpinCtrl;

	IntegerFilterItem(wxString field_name, wxWindow* parent, wxWindowID id = wxID_ANY, const wxPoint& pos = wxDefaultPosition, const wxSize& size = wxDefaultSize, long style = wxTAB_TRAVERSAL );
	~IntegerFilterItem();

	void CheckBoxClicked( wxCommandEvent& event);

	void LowControlChanged( wxCommandEvent& event);
	void HighControlChanged( wxCommandEvent& event);

	int GetLowValue();
	int GetHighValue();


};

class FloatFilterItem : public FilterItem
{

public :

	NumericTextCtrl *LowTextCtrl;
	wxStaticText *ToText;
	NumericTextCtrl *HighTextCtrl;

	float previous_low_value;
	float previous_high_value;

	FloatFilterItem(wxString field_name, wxWindow* parent, wxWindowID id = wxID_ANY, const wxPoint& pos = wxDefaultPosition, const wxSize& size = wxDefaultSize, long style = wxTAB_TRAVERSAL );
	~FloatFilterItem();

	void CheckBoxClicked( wxCommandEvent& event);

	void LowControlChanged( wxCommandEvent& event);
	void HighControlChanged( wxCommandEvent& event);
	void LowKillFocus( wxFocusEvent& event );
	void HighKillFocus( wxFocusEvent& event );
	void CheckLowValue();
	void CheckHighValue();

	float GetLowValue();
	float GetHighValue();


};

class DateFilterItem : public FilterItem
{

public :

	wxDatePickerCtrl *LowDateCtrl;
	wxStaticText *ToText;
	wxDatePickerCtrl *HighDateCtrl;

	DateFilterItem(wxString field_name, wxWindow* parent, wxWindowID id = wxID_ANY, const wxPoint& pos = wxDefaultPosition, const wxSize& size = wxDefaultSize, long style = wxTAB_TRAVERSAL );
	~DateFilterItem();

	void CheckBoxClicked( wxCommandEvent& event);

	void LowControlChanged( wxCommandEvent& event);
	void HighControlChanged( wxCommandEvent& event);

	int GetLowValue();
	int GetHighValue();

};

// Dialogs..

class ProperOverwriteCheckSaveDialog : public wxFileDialog
{
public :
	wxString extension_lowercase;
	wxString extension_uppercase;
	ProperOverwriteCheckSaveDialog(wxWindow *parent, const wxString &message, const wxString &wildcard, const wxString wanted_extension);
	~ProperOverwriteCheckSaveDialog();
	void OnSave( wxCommandEvent& event);
};

// virtual listctrl
class ContentsList: public wxListCtrl{
	public:
		ContentsList(wxWindow *parent, wxWindowID id, const wxPoint &pos=wxDefaultPosition, const wxSize &size=wxDefaultSize, long style=wxLC_ICON, const wxValidator &validator=wxDefaultValidator, const wxString &name=wxListCtrlNameStr);
		wxString OnGetItemText(long item, long column) const;

		int ReturnGuessAtColumnTextWidth(int wanted_column);
};

class RefinementPackageListControl: public wxListCtrl{
	public:
	RefinementPackageListControl(wxWindow *parent, wxWindowID id, const wxPoint &pos=wxDefaultPosition, const wxSize &size=wxDefaultSize, long style=wxLC_ICON, const wxValidator &validator=wxDefaultValidator, const wxString &name=wxListCtrlNameStr);
	wxString OnGetItemText(long item, long column) const;

	int ReturnGuessAtColumnTextWidth();
};

class ContainedParticleListControl: public wxListCtrl{
	public:
	ContainedParticleListControl(wxWindow *parent, wxWindowID id, const wxPoint &pos=wxDefaultPosition, const wxSize &size=wxDefaultSize, long style=wxLC_ICON, const wxValidator &validator=wxDefaultValidator, const wxString &name=wxListCtrlNameStr);
	wxString OnGetItemText(long item, long column) const;

	int ReturnGuessAtColumnTextWidth(int wanted_column);
};

class ReferenceVolumesListControl: public wxListCtrl{
	public:
	ReferenceVolumesListControl(wxWindow *parent, wxWindowID id, const wxPoint &pos=wxDefaultPosition, const wxSize &size=wxDefaultSize, long style=wxLC_ICON, const wxValidator &validator=wxDefaultValidator, const wxString &name=wxListCtrlNameStr);
	wxString OnGetItemText(long item, long column) const;

	int ReturnGuessAtColumnTextWidth(int wanted_column);

};

class RefinementParametersListCtrl: public wxListCtrl{
	public:
	RefinementParametersListCtrl(wxWindow *parent, wxWindowID id, const wxPoint &pos=wxDefaultPosition, const wxSize &size=wxDefaultSize, long style=wxLC_ICON, const wxValidator &validator=wxDefaultValidator, const wxString &name=wxListCtrlNameStr);
	wxString OnGetItemText(long item, long column) const;

	int ReturnGuessAtColumnTextWidth(int wanted_column);

};



#endif
