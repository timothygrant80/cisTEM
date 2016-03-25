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

#endif
