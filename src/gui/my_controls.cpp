//#include "../core/core_headers.h"
#include "../core/gui_core_headers.h"


extern MyRefinementPackageAssetPanel *refinement_package_asset_panel;
extern MyRefinementResultsPanel *refinement_results_panel;
extern Refine2DResultsPanel *refine2d_results_panel;

MemoryComboBox::MemoryComboBox(wxWindow *parent, wxWindowID id, const wxString &value, const wxPoint &pos, const wxSize &size, int n, const wxString choices[], long style, const wxValidator &validator, const wxString &name)
:
wxComboBox(parent, id, value, pos, size, n, choices, style, validator, name)
{
	associated_ids.clear();
	selected_id_on_last_clear -10;
	currently_selected_id = -10;

	Bind(wxEVT_COMBOBOX, &MemoryComboBox::OnComboBox, this);

}

MemoryComboBox::~MemoryComboBox()
{
	Unbind(wxEVT_COMBOBOX, &MemoryComboBox::OnComboBox, this);
}

void MemoryComboBox::OnComboBox(wxCommandEvent& event)
{
	if (GetSelection() >= 0) currently_selected_id = associated_ids.Item(GetSelection());
	event.Skip();
}

void MemoryComboBox::Clear()
{
	associated_ids.clear();
	selected_id_on_last_clear = currently_selected_id;
	currently_selected_id = -10;
	wxComboBox::Clear();
}

void MemoryComboBox::Reset()
{
	associated_ids.clear();
	selected_id_on_last_clear = -10;
	currently_selected_id = -10;
	wxComboBox::Clear();
}

void MemoryComboBox::AddMemoryItem(wxString wanted_text, long wanted_id)
{
	Append(wanted_text);
	associated_ids.Add(wanted_id);
}


bool MemoryComboBox::FillWithRunProfiles()
{
	extern MyRunProfilesPanel *run_profiles_panel;

	Freeze();
	Clear();
	ChangeValue("");

	long new_selection = 0;
	long new_id = -1;

	for (long counter = 0; counter < run_profiles_panel->run_profile_manager.number_of_run_profiles; counter++)
	{
		AddMemoryItem(run_profiles_panel->run_profile_manager.ReturnProfileName(counter) + wxString::Format(" (%li)", run_profiles_panel->run_profile_manager.ReturnTotalJobs(counter)), run_profiles_panel->run_profile_manager.ReturnProfileID(counter));

		if (run_profiles_panel->run_profile_manager.ReturnProfileID(counter) == selected_id_on_last_clear)
		{
			new_selection = counter;
			new_id = selected_id_on_last_clear;
		}
	}

	if (GetCount() > 0)
	{
		SetSelection(new_selection);
	}

	currently_selected_id = new_id;
	Thaw();

	if (new_id == selected_id_on_last_clear && new_id != -1) return true;
	else return false;
}

bool MemoryComboBox::FillWithMovieGroups(bool include_all_movies_group)
{
	extern MyMovieAssetPanel *movie_asset_panel;

	Freeze();
	Clear();
	ChangeValue("");

	long new_selection = 0;
	long new_id = -1;
	long start_position;

	if (include_all_movies_group == true) start_position = 0;
	else start_position = 1;

	for (long counter = start_position; counter < movie_asset_panel->ReturnNumberOfGroups(); counter++)
	{
		AddMemoryItem(movie_asset_panel->ReturnGroupName(counter) +  " (" + wxString::Format(wxT("%li"), movie_asset_panel->ReturnGroupSize(counter)) + ")", movie_asset_panel->ReturnGroupID(counter));
		if (movie_asset_panel->ReturnGroupID(counter) == selected_id_on_last_clear)
		{
			if (include_all_movies_group == true) new_selection = counter;
			else new_selection = counter -1;

			new_id = selected_id_on_last_clear;
		}
	}

	if (GetCount() > 0)
	{
		SetSelection(new_selection);
	}

	currently_selected_id = new_id;
	Thaw();

	if (new_id == selected_id_on_last_clear && new_id != -1) return true;
	else return false;
}

bool MemoryComboBox::FillWithImageGroups(bool include_all_images_group)
{
	extern MyImageAssetPanel *image_asset_panel;

	Freeze();
	Clear();
	ChangeValue("");

	long new_selection = 0;
	long new_id = -1;
	long start_position;

	if (include_all_images_group == true) start_position = 0;
	else start_position = 1;

	for (long counter = start_position; counter < image_asset_panel->ReturnNumberOfGroups(); counter++)
	{
		AddMemoryItem(image_asset_panel->ReturnGroupName(counter) +  " (" + wxString::Format(wxT("%li"), image_asset_panel->ReturnGroupSize(counter)) + ")", image_asset_panel->ReturnGroupID(counter));
		if (image_asset_panel->ReturnGroupID(counter) == selected_id_on_last_clear)
		{
			if (include_all_images_group == true) new_selection == counter;
			else new_selection = counter - 1;

			new_id = selected_id_on_last_clear;
		}
	}

	if (GetCount() > 0)
	{
		SetSelection(new_selection);
	}

	currently_selected_id = new_id;
	Thaw();

	if (new_id == selected_id_on_last_clear && new_id != -1) return true;
	else return false;

}

bool MemoryComboBox::FillWithRefinementPackages()
{
	extern MyRefinementPackageAssetPanel *refinement_package_asset_panel;
	Freeze();
	Clear();
	ChangeValue("");

	long new_selection = -1;
	long new_id = -1;

	for (long counter = 0; counter < refinement_package_asset_panel->all_refinement_packages.GetCount(); counter++)
	{
		AddMemoryItem(refinement_package_asset_panel->all_refinement_packages.Item(counter).name, refinement_package_asset_panel->all_refinement_packages.Item(counter).asset_id);
		if (refinement_package_asset_panel->all_refinement_packages.Item(counter).asset_id == selected_id_on_last_clear)
		{
			new_selection = counter;
			new_id = selected_id_on_last_clear;
		}
	}

	if (GetCount() > 0)
	{
		if (new_selection == -1) SetSelection(GetCount() - 1);
		else SetSelection(new_selection);
	}

	currently_selected_id = new_id;
	Thaw();

	if (new_id == selected_id_on_last_clear && new_id != -1) return true;
	else return false;
}

bool MemoryComboBox::FillWithClassifications(long wanted_refinement_package, bool include_new_classification)
{
	extern MyRefinementPackageAssetPanel *refinement_package_asset_panel;
	Freeze();
	Clear();
	ChangeValue("");

	long new_selection = -1;
	long new_id = -1;

	if (include_new_classification == true) AddMemoryItem("New Classification", -1);

	for (long counter = 0; counter < refinement_package_asset_panel->all_refinement_packages.Item(wanted_refinement_package).classification_ids.GetCount(); counter++)
	{
		AddMemoryItem(refinement_package_asset_panel->ReturnPointerToShortClassificationInfoByClassificationID(refinement_package_asset_panel->all_refinement_packages.Item(wanted_refinement_package).classification_ids.Item(counter))->name, refinement_package_asset_panel->ReturnPointerToShortClassificationInfoByClassificationID(refinement_package_asset_panel->all_refinement_packages.Item(wanted_refinement_package).classification_ids.Item(counter))->classification_id);
		if (refinement_package_asset_panel->ReturnPointerToShortClassificationInfoByClassificationID(refinement_package_asset_panel->all_refinement_packages.Item(wanted_refinement_package).classification_ids.Item(counter))->classification_id == selected_id_on_last_clear)
		{
			if (include_new_classification == true) new_selection = counter + 1;
			else new_selection = counter;

			new_id = selected_id_on_last_clear;
		}
	}

	if (GetCount() > 0)
	{
		if (new_selection == -1) SetSelection(GetCount() - 1);
		else SetSelection(new_selection);
	}

	currently_selected_id = new_id;
	Thaw();

	if (new_id == selected_id_on_last_clear && new_id != -1) return true;
	else return false;
}

bool MemoryComboBox::FillWithRefinements(long wanted_refinement_package)
{
	extern MyRefinementPackageAssetPanel *refinement_package_asset_panel;
	Freeze();
	Clear();
	ChangeValue("");

	long new_selection = -1;
	long new_id = -1;


	for (long counter = 0; counter < refinement_package_asset_panel->all_refinement_packages[wanted_refinement_package].refinement_ids.GetCount(); counter++)
	{
		AddMemoryItem(refinement_package_asset_panel->ReturnPointerToShortRefinementInfoByRefinementID(refinement_package_asset_panel->all_refinement_packages[wanted_refinement_package].refinement_ids[counter])->name, refinement_package_asset_panel->ReturnPointerToShortRefinementInfoByRefinementID(refinement_package_asset_panel->all_refinement_packages[wanted_refinement_package].refinement_ids[counter])->refinement_id);

		if (refinement_package_asset_panel->ReturnPointerToShortRefinementInfoByRefinementID(refinement_package_asset_panel->all_refinement_packages[wanted_refinement_package].refinement_ids[counter])->refinement_id == selected_id_on_last_clear)
		{
			new_selection = counter;
			new_id = selected_id_on_last_clear;
		}
	}

	if (GetCount() > 0)
	{
		if (new_selection == -1) SetSelection(GetCount() - 1);
		else SetSelection(new_selection);
	}

	currently_selected_id = new_id;
	Thaw();

	if (new_id == selected_id_on_last_clear && new_id != -1) return true;
	else return false;
}

OneSecondProgressDialog::OneSecondProgressDialog(const wxString &title, const wxString &message, int maximum, wxWindow *parent, int style)
:
wxProgressDialog(title, message, maximum, parent, style)
{
	time_of_last_update = time(NULL);
}

bool OneSecondProgressDialog::Update(int value, const wxString & newmsg, bool * skip)
{
	if (time(NULL) - time_of_last_update > 0 || newmsg != wxEmptyString)
	{
		wxProgressDialog::Update(value, newmsg, skip);
		if (newmsg != wxEmptyString) Fit();
		time_of_last_update = time(NULL);
	}

}


NumericTextCtrl::NumericTextCtrl(wxWindow *parent, wxWindowID id, const wxString &value, const wxPoint &pos, const wxSize &size, long style, const wxValidator &validator, const wxString &name)
:
wxTextCtrl(parent, id, value, pos, size, style, validator, name)
{
	// bind change text method..

	Bind(wxEVT_CHAR, &NumericTextCtrl::OnKeyPress, this);
	Bind(wxEVT_TEXT_ENTER, &NumericTextCtrl::OnEnterPressed, this );
	Bind(wxEVT_KILL_FOCUS, &NumericTextCtrl::OnFocusLost, this );

	min_value = -FLT_MAX;
	max_value = FLT_MAX;

	precision = 2;

	double initial_value;
	if (value.ToDouble(&initial_value) == false) initial_value = 0.0;

	previous_value = initial_value;
	ChangeValueFloat(initial_value);
}

NumericTextCtrl::~NumericTextCtrl()
{
	Unbind(wxEVT_CHAR, &NumericTextCtrl::OnKeyPress, this);
	Unbind(wxEVT_TEXT_ENTER, &NumericTextCtrl::OnEnterPressed, this );
	Unbind(wxEVT_KILL_FOCUS, &NumericTextCtrl::OnFocusLost, this );
}

void NumericTextCtrl::ChangeValueFloat(float wanted_float)
{
	if (precision == 2) ChangeValue(wxString::Format("%.2f", wanted_float));
	else
	if (precision == 3) ChangeValue(wxString::Format("%.3f", wanted_float));
	else
	if (precision == 4) ChangeValue(wxString::Format("%.4f", wanted_float));
	else
	if (precision == 1)	ChangeValue(wxString::Format("%.1f", wanted_float));
	else
	if (precision == 0) ChangeValue(wxString::Format("%.0f", wanted_float));
	else
	ChangeValue(wxString::Format("%.2f", wanted_float));

}

void NumericTextCtrl::SetPrecision(int wanted_precision)
{
	precision = wanted_precision;

	double current_value;

	if (GetLineText(0).ToDouble(&current_value) == false) current_value = 0.0;
	ChangeValueFloat(current_value);

}



void NumericTextCtrl::SetMinMaxValue(float wanted_min_value, float wanted_max_value)
{
	MyDebugAssertTrue(wanted_min_value <= wanted_max_value, "max value is less than min value");
	min_value = wanted_min_value;
	max_value = wanted_max_value;
}

float NumericTextCtrl::ReturnValue()
{
	double current_value;
	if (GetLineText(0).ToDouble(&current_value) == false) return 0.0;
	else return float(current_value);

}


void NumericTextCtrl::CheckValues()
{
	double current_value;

	if (GetLineText(0).ToDouble(&current_value) == false)
	{
		ChangeValueFloat(previous_value);
	}
	else
	{
		if (current_value < min_value)
		{
			ChangeValueFloat(min_value);
			previous_value = min_value;
		}
		else
		if (current_value > max_value)
		{
			ChangeValueFloat(max_value);
			previous_value = max_value;
		}
		else
		{
			ChangeValueFloat(current_value);
			previous_value = current_value;
		}
	}
}

void NumericTextCtrl::OnKeyPress( wxKeyEvent& event )
{
	int keycode = event.GetKeyCode();
	bool is_valid_key = false;

	if (keycode > 31 && keycode < 127)
	{
		if (keycode > 47 && keycode < 58) is_valid_key = true;
		else if (keycode > 44 && keycode < 47) is_valid_key = true;
	}
	else is_valid_key = true;

	if (is_valid_key == true) event.Skip();
}

void NumericTextCtrl::OnEnterPressed( wxCommandEvent& event)
{
	CheckValues();
	event.Skip();
}

void NumericTextCtrl::OnFocusLost( wxFocusEvent& event )
{
	CheckValues();
	event.Skip();
}

AutoWrapStaticText::AutoWrapStaticText(wxWindow *parent, wxWindowID id, const wxString &label, const wxPoint &pos, const wxSize &size, long style, const wxString &name)
:
wxStaticText(parent, id, label, pos, size, style, name)
{
	//Bind(wxEVT_SIZE, &AutoWrapStaticText::OnSize, this );
	has_autowrapped = false;
}

AutoWrapStaticText::~AutoWrapStaticText()
{
	//Unbind(wxEVT_SIZE, &AutoWrapStaticText::OnSize, this );
}

void AutoWrapStaticText::AutoWrap()
{
	Wrap(GetClientSize().GetWidth());
	has_autowrapped = true;
}

void AutoWrapStaticText::OnSize(wxSizeEvent& event)
{
	Wrap(GetClientSize().GetWidth());
	event.Skip();
}

FilterItem::FilterItem( wxString field_name, wxWindow* parent, wxWindowID id, const wxPoint& pos, const wxSize& size, long style ) : wxPanel( parent, id, pos, size, style )
{
	my_parent = parent;
	my_sizer = new wxBoxSizer (wxHORIZONTAL );
	field_checkbox =  new wxCheckBox( this, wxID_ANY, field_name + " :", wxDefaultPosition, wxDefaultSize, 0 );
	my_sizer->Add(field_checkbox, 0, wxALIGN_CENTER_VERTICAL|wxALL, 5 );
	my_sizer->Add( 0, 0, 1, wxEXPAND, 5 );

	// add events for the checkbox..

	field_checkbox->Bind(wxEVT_COMMAND_CHECKBOX_CLICKED, &FilterItem::CheckBoxClicked, this );


	SetSizer( my_sizer );
	Layout();
	my_sizer->Fit(this);

}


FilterItem::~FilterItem()
{
	field_checkbox->Unbind(wxEVT_COMMAND_CHECKBOX_CLICKED, &FilterItem::CheckBoxClicked, this );
}


IntegerFilterItem::IntegerFilterItem(wxString field_name, wxWindow* parent, wxWindowID id, const wxPoint& pos, const wxSize& size, long style )
:
FilterItem(field_name, parent, id, pos, size, style)
{

	LowSpinCtrl = new wxSpinCtrl( this, wxID_ANY, wxEmptyString, wxDefaultPosition, wxSize( 120,-1 ), wxSP_ARROW_KEYS, 0, 99999999, 0 );
	my_sizer->Add( LowSpinCtrl, 0, wxALIGN_CENTER_VERTICAL|wxALL, 5 );

	ToText = new wxStaticText( this, wxID_ANY, wxT("To"), wxDefaultPosition, wxDefaultSize, 0 );
	ToText->Wrap( -1 );
	my_sizer->Add( ToText, 0, wxALIGN_CENTER_VERTICAL|wxALL, 5 );

	HighSpinCtrl = new wxSpinCtrl( this, wxID_ANY, wxEmptyString, wxDefaultPosition, wxSize( 120,-1 ), wxSP_ARROW_KEYS, 0, 99999999, 0 );
	my_sizer->Add( HighSpinCtrl, 0, wxALIGN_CENTER_VERTICAL|wxALL, 5 );

	LowSpinCtrl->Enable(false);
	HighSpinCtrl->Enable(false);
	Layout();

	LowSpinCtrl->Bind(wxEVT_SPINCTRL, &IntegerFilterItem::LowControlChanged, this );
	HighSpinCtrl->Bind(wxEVT_SPINCTRL, &IntegerFilterItem::HighControlChanged, this );


}


void IntegerFilterItem::CheckBoxClicked(wxCommandEvent &event)
{
	if (field_checkbox->IsChecked() == true)
	{
		LowSpinCtrl->Enable(true);
		HighSpinCtrl->Enable(true);
	}
	else
	{
		LowSpinCtrl->Enable(false);
		HighSpinCtrl->Enable(false);


	}
}

void IntegerFilterItem::LowControlChanged( wxCommandEvent& event)
{
	//	wxPrintf("Changed\n");

	int current_value = LowSpinCtrl->GetValue();

	if (HighSpinCtrl->GetValue() < current_value) HighSpinCtrl->SetValue(current_value);
}

void IntegerFilterItem::HighControlChanged( wxCommandEvent& event)
{
	//	wxPrintf("Changed\n");

	int current_value = HighSpinCtrl->GetValue();

	if (LowSpinCtrl->GetValue() > current_value) LowSpinCtrl->SetValue(current_value);
}

IntegerFilterItem::~IntegerFilterItem()
{
	LowSpinCtrl->Unbind(wxEVT_SPINCTRL, &IntegerFilterItem::LowControlChanged, this );
	HighSpinCtrl->Unbind(wxEVT_SPINCTRL, &IntegerFilterItem::HighControlChanged, this );
}

int IntegerFilterItem::GetLowValue()
{
	return LowSpinCtrl->GetValue();

}

int IntegerFilterItem::GetHighValue()
{
	return HighSpinCtrl->GetValue();
}


//-------------

FloatFilterItem::FloatFilterItem(wxString field_name, wxWindow* parent, wxWindowID id, const wxPoint& pos, const wxSize& size, long style )
:
FilterItem(field_name, parent, id, pos, size, style)
{

	LowTextCtrl = new NumericTextCtrl( this, wxID_ANY, wxEmptyString, wxDefaultPosition, wxSize( 120,-1 ), wxTE_PROCESS_ENTER );
	my_sizer->Add( LowTextCtrl, 0, wxALIGN_CENTER_VERTICAL|wxALL, 5 );

	ToText = new wxStaticText( this, wxID_ANY, wxT("To"), wxDefaultPosition, wxDefaultSize, 0 );
	ToText->Wrap( -1 );
	my_sizer->Add( ToText, 0, wxALIGN_CENTER_VERTICAL|wxALL, 5 );

	HighTextCtrl = new NumericTextCtrl( this, wxID_ANY, wxEmptyString, wxDefaultPosition, wxSize( 120,-1 ), wxTE_PROCESS_ENTER );
	my_sizer->Add( HighTextCtrl, 0, wxALIGN_CENTER_VERTICAL|wxALL, 5 );

	LowTextCtrl->Enable(false);
	HighTextCtrl->Enable(false);
	Layout();

	LowTextCtrl->Bind(wxEVT_TEXT_ENTER, &FloatFilterItem::LowControlChanged, this );
	HighTextCtrl->Bind(wxEVT_TEXT_ENTER, &FloatFilterItem::HighControlChanged, this );
	LowTextCtrl->Bind(wxEVT_KILL_FOCUS, &FloatFilterItem::LowKillFocus, this );
	HighTextCtrl->Bind(wxEVT_KILL_FOCUS, &FloatFilterItem::HighKillFocus, this );

	LowTextCtrl->ChangeValue("0.0");
	HighTextCtrl->ChangeValue("0.0");

	previous_low_value = 0.0;
	previous_high_value = 0.0;
}


void FloatFilterItem::CheckBoxClicked(wxCommandEvent &event)
{
	if (field_checkbox->IsChecked() == true)
	{
		LowTextCtrl->Enable(true);
		HighTextCtrl->Enable(true);
	}
	else
	{
		LowTextCtrl->Enable(false);
		HighTextCtrl->Enable(false);


	}
}

void FloatFilterItem::CheckLowValue()
{
	double current_low_value;
	double current_high_value;

	if (LowTextCtrl->GetLineText(0).ToDouble(&current_low_value) == false)
	{
		LowTextCtrl->ChangeValue(wxString::Format("%.2f", previous_low_value));
	}
	else
	{
		previous_low_value = float(current_low_value);
		LowTextCtrl->ChangeValue(wxString::Format("%.2f", previous_low_value));

		if (HighTextCtrl->GetLineText(0).ToDouble(&current_high_value) == false)
		{
			HighTextCtrl->ChangeValue(wxString::Format("%.2f", previous_high_value));
		}
		else
		{
			if (current_low_value > current_high_value)
			{
				HighTextCtrl->ChangeValue(wxString::Format("%.2f", current_low_value));
				previous_high_value = current_low_value;
			}
		}

	}

}

void FloatFilterItem::CheckHighValue()
{
	double current_low_value;
	double current_high_value;

	if (HighTextCtrl->GetLineText(0).ToDouble(&current_high_value) == false)
	{
		HighTextCtrl->ChangeValue(wxString::Format("%.2f", previous_high_value));
	}
	else
	{
		previous_high_value = float(current_high_value);
		HighTextCtrl->ChangeValue(wxString::Format("%.2f", previous_high_value));

		if (LowTextCtrl->GetLineText(0).ToDouble(&current_low_value) == false)
		{
			LowTextCtrl->ChangeValue(wxString::Format("%.2f", previous_low_value));
		}
		else
		{
			if (current_high_value < current_low_value)
			{
				LowTextCtrl->ChangeValue(wxString::Format("%.2f", current_high_value));
				previous_low_value = current_high_value;
			}
		}

	}

}

void FloatFilterItem::LowControlChanged( wxCommandEvent& event)
{
	CheckLowValue();
}

void FloatFilterItem::HighControlChanged( wxCommandEvent& event)
{
	CheckHighValue();
}

void FloatFilterItem::LowKillFocus( wxFocusEvent& event )
{
	CheckLowValue();
}

void FloatFilterItem::HighKillFocus( wxFocusEvent& event )
{
	CheckHighValue();
}

float FloatFilterItem::GetLowValue()
{
	double current_value;
	LowTextCtrl->GetLineText(0).ToDouble(&current_value);
	return float(current_value);

}

float FloatFilterItem::GetHighValue()
{
	double current_value;
	HighTextCtrl->GetLineText(0).ToDouble(&current_value);
	return float(current_value);

}

FloatFilterItem::~FloatFilterItem()
{
	LowTextCtrl->Unbind(wxEVT_TEXT_ENTER, &FloatFilterItem::LowControlChanged, this );
	HighTextCtrl->Unbind(wxEVT_TEXT_ENTER, &FloatFilterItem::HighControlChanged, this );
	LowTextCtrl->Unbind(wxEVT_KILL_FOCUS, &FloatFilterItem::LowKillFocus, this );
	HighTextCtrl->Unbind(wxEVT_KILL_FOCUS, &FloatFilterItem::HighKillFocus, this );

}

//------------------

DateFilterItem::DateFilterItem(wxString field_name, wxWindow* parent, wxWindowID id, const wxPoint& pos, const wxSize& size, long style )
:
FilterItem(field_name, parent, id, pos, size, style)
{

	LowDateCtrl = new wxDatePickerCtrl( this, wxID_ANY, wxDefaultDateTime, wxDefaultPosition, wxSize( 120,-1 ), wxDP_DEFAULT );
	my_sizer->Add( LowDateCtrl, 0, wxALIGN_CENTER_VERTICAL|wxALL, 5 );

	ToText = new wxStaticText( this, wxID_ANY, wxT("To"), wxDefaultPosition, wxDefaultSize, 0 );
	ToText->Wrap( -1 );
	my_sizer->Add( ToText, 0, wxALIGN_CENTER_VERTICAL|wxALL, 5 );

	HighDateCtrl = new wxDatePickerCtrl( this, wxID_ANY, wxDefaultDateTime, wxDefaultPosition, wxSize( 120,-1 ), wxDP_DEFAULT );
	my_sizer->Add( HighDateCtrl, 0, wxALIGN_CENTER_VERTICAL|wxALL, 5 );

	LowDateCtrl->Enable(false);
	HighDateCtrl->Enable(false);
	Layout();

	LowDateCtrl->Bind(wxEVT_DATE_CHANGED, &DateFilterItem::LowControlChanged, this );
	HighDateCtrl->Bind(wxEVT_DATE_CHANGED, &DateFilterItem::HighControlChanged, this );


}


void DateFilterItem::CheckBoxClicked(wxCommandEvent &event)
{
	if (field_checkbox->IsChecked() == true)
	{
		LowDateCtrl->Enable(true);
		HighDateCtrl->Enable(true);
	}
	else
	{
		LowDateCtrl->Enable(false);
		HighDateCtrl->Enable(false);

	}
}

void DateFilterItem::LowControlChanged( wxCommandEvent& event)
{
	wxDateTime current_value;

	current_value = LowDateCtrl->GetValue();

	if (current_value > HighDateCtrl->GetValue())
	{
		HighDateCtrl->SetValue(current_value);
	}
}

void DateFilterItem::HighControlChanged( wxCommandEvent& event)
{
	wxDateTime current_value;

	current_value = HighDateCtrl->GetValue();

	if (current_value < LowDateCtrl->GetValue())
	{
		LowDateCtrl->SetValue(current_value);
	}
}


DateFilterItem::~DateFilterItem()
{
	LowDateCtrl->Unbind(wxEVT_DATE_CHANGED, &DateFilterItem::LowControlChanged, this );
	HighDateCtrl->Unbind(wxEVT_DATE_CHANGED, &DateFilterItem::HighControlChanged, this );
}

int DateFilterItem::GetLowValue()
{
	return LowDateCtrl->GetValue().GetTicks();

}

int DateFilterItem::GetHighValue()
{
	return HighDateCtrl->GetValue().GetTicks();
}



ProperOverwriteCheckSaveDialog::ProperOverwriteCheckSaveDialog(wxWindow *parent, const wxString &message, const wxString &wildcard, const wxString wanted_extension)
:
wxFileDialog(parent, message, wxEmptyString, wxEmptyString, wildcard, wxFD_SAVE, wxDefaultPosition, wxDefaultSize, wxFileDialogNameStr)
{

	extension_lowercase = wanted_extension.Lower();
	extension_uppercase = wanted_extension.Upper();
	Bind (wxEVT_BUTTON, &ProperOverwriteCheckSaveDialog::OnSave, this, wxID_OK);

};

ProperOverwriteCheckSaveDialog::~ProperOverwriteCheckSaveDialog()
{
	Unbind (wxEVT_BUTTON, &ProperOverwriteCheckSaveDialog::OnSave, this, wxID_OK);
}

void ProperOverwriteCheckSaveDialog::OnSave(wxCommandEvent &event)
{
	wxString current_path = GetPath();

	if (current_path.EndsWith(extension_lowercase) == false && current_path.EndsWith(extension_uppercase) == false)
	{
		current_path += extension_lowercase;
	}
	SetPath(current_path);

	if (DoesFileExist(current_path) == true)
	{
		wxMessageDialog check_dialog(this, wxString::Format("The File :-\n\n\"%s\"\n\nAlready exists, Do you want to OVERWRITE it?", GetPath()), "Confirm", wxYES_NO|wxICON_WARNING);

		if (check_dialog.ShowModal() ==  wxID_YES) event.Skip();
	}
	else event.Skip();
}


// VIRTUAL LIST CTRL
ContentsList::ContentsList(wxWindow *parent, wxWindowID id, const wxPoint &pos, const wxSize &size, long style, const wxValidator &validator, const wxString &name)
:
wxListCtrl(parent, id, pos, size, style, validator, name)
{

}

wxString ContentsList::OnGetItemText(long item, long column) const
{
	MyAssetParentPanel *parent_panel =  reinterpret_cast < MyAssetParentPanel *> (m_parent->GetParent()->GetParent()); // not very nice code!
	return parent_panel->ReturnItemText(item, column);
	//wxPrintf("Here!\n");
	//	return "Hi"	;
}

int ContentsList::ReturnGuessAtColumnTextWidth(int wanted_column)
{
	wxClientDC dc(this);
	long counter;

	wxListItem result;
	result.SetMask(wxLIST_MASK_TEXT);
	GetColumn(wanted_column, result);

	int max_width = dc.GetTextExtent(result.GetText()).x + 20;

	if (GetItemCount() < 100)
	{
		for ( counter = 0; counter < GetItemCount(); counter++)
		{
			if (dc.GetTextExtent(OnGetItemText(counter, wanted_column)).x + 20 > max_width) max_width = dc.GetTextExtent(OnGetItemText(counter, wanted_column)).x + 20;
		}
	}
	else
	{
		for ( counter = 0; counter < 50; counter++)
		{
			if (dc.GetTextExtent(OnGetItemText(counter, wanted_column)).x + 20 > max_width) max_width = dc.GetTextExtent(OnGetItemText(counter, wanted_column)).x + 20;
		}

		for ( counter = GetItemCount() - 50; counter < GetItemCount(); counter++)
		{
			if (dc.GetTextExtent(OnGetItemText(counter, wanted_column)).x + 20 > max_width) max_width = dc.GetTextExtent(OnGetItemText(counter, wanted_column)).x + 20;
		}

	}

	return max_width;

}

// REFINEMENT PACKAGE LIST

RefinementPackageListControl::RefinementPackageListControl(wxWindow *parent, wxWindowID id, const wxPoint &pos, const wxSize &size, long style, const wxValidator &validator, const wxString &name)
:
wxListCtrl(parent, id, pos, size, style, validator, name)
{

}

wxString RefinementPackageListControl::OnGetItemText(long item, long column) const
{
	MyDebugAssertTrue(column == 0, "Asking for column that shouldn't exist (%li)", column);

	MyRefinementPackageAssetPanel *parent_panel =  reinterpret_cast < MyRefinementPackageAssetPanel *> (m_parent->GetParent()->GetParent()); // not very nice code!

	if (parent_panel->all_refinement_packages.GetCount() > 0)
	{
		return parent_panel->all_refinement_packages.Item(item).name;

	}
	else return "";

}

int RefinementPackageListControl::ReturnGuessAtColumnTextWidth()
{

	wxClientDC dc(this);
	long counter;
	int client_height;
	int client_width;

	int current_width;

	GetClientSize(&client_width, &client_height);
	int max_width = client_width;

	if (GetItemCount() < 100)
	{
		for ( counter = 0; counter < GetItemCount(); counter++)
		{
			if (dc.GetTextExtent(OnGetItemText(counter, 0)).x + 20 > max_width) max_width = dc.GetTextExtent(OnGetItemText(counter, 0)).x + 20;
		}
	}
	else
	{
		for ( counter = 0; counter < 50; counter++)
		{
			if (dc.GetTextExtent(OnGetItemText(counter, 0)).x + 20 > max_width) max_width = dc.GetTextExtent(OnGetItemText(counter, 0)).x + 20;
		}

		for ( counter = GetItemCount() - 50; counter < GetItemCount(); counter++)
		{
			if (dc.GetTextExtent(OnGetItemText(counter, 0)).x + 20 > max_width) max_width = dc.GetTextExtent(OnGetItemText(counter, 0)).x + 20;
		}

	}

	return max_width;

}

// CONTAINED PARTICLES

ContainedParticleListControl::ContainedParticleListControl(wxWindow *parent, wxWindowID id, const wxPoint &pos, const wxSize &size, long style, const wxValidator &validator, const wxString &name)
:
wxListCtrl(parent, id, pos, size, style, validator, name)
{

}

wxString ContainedParticleListControl::OnGetItemText(long item, long column) const
{
	MyRefinementPackageAssetPanel *parent_panel =  reinterpret_cast < MyRefinementPackageAssetPanel *> (m_parent->GetParent()->GetParent()); // not very nice code!

	if (parent_panel->all_refinement_packages.GetCount() > 0 && parent_panel->selected_refinement_package >= 0)
	{
		switch(column)
		{
		    case 0  :
		    	return wxString::Format(wxT("%li"), parent_panel->all_refinement_packages.Item(parent_panel->selected_refinement_package).contained_particles.Item(item).original_particle_position_asset_id);
		       break;
		    case 1  :
		    	return wxString::Format(wxT("%li"), parent_panel->all_refinement_packages.Item(parent_panel->selected_refinement_package).contained_particles.Item(item).parent_image_id);
		       break;
		    case 2  :
		    	return wxString::Format(wxT("%.2f"), parent_panel->all_refinement_packages.Item(parent_panel->selected_refinement_package).contained_particles.Item(item).x_pos);
		       break;
		    case 3  :
		    	return wxString::Format(wxT("%.2f"), parent_panel->all_refinement_packages.Item(parent_panel->selected_refinement_package).contained_particles.Item(item).y_pos);
		       break;
		    case 4  :
		    	return wxString::Format(wxT("%.2f Å"), parent_panel->all_refinement_packages.Item(parent_panel->selected_refinement_package).contained_particles.Item(item).pixel_size);
		       break;
		    case 5  :
				return wxString::Format(wxT("%.2f mm"), parent_panel->all_refinement_packages.Item(parent_panel->selected_refinement_package).contained_particles.Item(item).spherical_aberration);
			   break;
		    case 6  :
				return wxString::Format(wxT("%.2f kV"), parent_panel->all_refinement_packages.Item(parent_panel->selected_refinement_package).contained_particles.Item(item).microscope_voltage);
			   break;
		    case 7  :
		    	return wxString::Format(wxT("%.0f Å"), parent_panel->all_refinement_packages.Item(parent_panel->selected_refinement_package).contained_particles.Item(item).defocus_1);
			   break;
		    case 8  :
		    	return wxString::Format(wxT("%.0f Å"), parent_panel->all_refinement_packages.Item(parent_panel->selected_refinement_package).contained_particles.Item(item).defocus_2);
			   break;
		    case 9  :
		    	return wxString::Format(wxT("%.2f °"), parent_panel->all_refinement_packages.Item(parent_panel->selected_refinement_package).contained_particles.Item(item).defocus_angle);
		    	break;
		    case 10  :
		    	return wxString::Format(wxT("%.2f rad."), parent_panel->all_refinement_packages.Item(parent_panel->selected_refinement_package).contained_particles.Item(item).phase_shift);
		    	break;
		 /*   case 10 :
			   	return wxString::Format(wxT("%.2f °"), parent_panel->all_refinement_packages.Item(parent_panel->selected_refinement_package).contained_particles.Item(item).psi);
			   	break;
		    case 11 :
			    return wxString::Format(wxT("%.2f °"), parent_panel->all_refinement_packages.Item(parent_panel->selected_refinement_package).contained_particles.Item(item).theta);
			   	break;
		    case 12 :
			    return wxString::Format(wxT("%.2f °"), parent_panel->all_refinement_packages.Item(parent_panel->selected_refinement_package).contained_particles.Item(item).phi);
			    break;
		    case 13 :
			    return wxString::Format(wxT("%.2f °"), parent_panel->all_refinement_packages.Item(parent_panel->selected_refinement_package).contained_particles.Item(item).x_shift);
			    break;
		    case 14 :
			    return wxString::Format(wxT("%.2f °"), parent_panel->all_refinement_packages.Item(parent_panel->selected_refinement_package).contained_particles.Item(item).y_shift);
			    break;
*/
		    default :
		       MyPrintWithDetails("Error, asking for column (%li) which does not exist", column);
		       return "";

		}
	}
	else
	{
		return "";
	}
}

int ContainedParticleListControl::ReturnGuessAtColumnTextWidth(int wanted_column)
{
	wxClientDC dc(this);
	long counter;

	wxListItem result;
	result.SetMask(wxLIST_MASK_TEXT);
	GetColumn(wanted_column, result);

	int max_width = dc.GetTextExtent(result.GetText()).x + 20;

	if (GetItemCount() < 100)
	{
		for ( counter = 0; counter < GetItemCount(); counter++)
		{
			if (dc.GetTextExtent(OnGetItemText(counter, wanted_column)).x + 20 > max_width) max_width = dc.GetTextExtent(OnGetItemText(counter, wanted_column)).x + 20;
		}
	}
	else
	{
		for ( counter = 0; counter < 50; counter++)
		{
			if (dc.GetTextExtent(OnGetItemText(counter, wanted_column)).x + 20 > max_width) max_width = dc.GetTextExtent(OnGetItemText(counter, wanted_column)).x + 20;
		}

		for ( counter = GetItemCount() - 50; counter < GetItemCount(); counter++)
		{
			if (dc.GetTextExtent(OnGetItemText(counter, wanted_column)).x + 20 > max_width) max_width = dc.GetTextExtent(OnGetItemText(counter, wanted_column)).x + 20;
		}

	}

	return max_width;
}

// 3D References

ReferenceVolumesListControl::ReferenceVolumesListControl(wxWindow *parent, wxWindowID id, const wxPoint &pos, const wxSize &size, long style, const wxValidator &validator, const wxString &name)
:
wxListCtrl(parent, id, pos, size, style, validator, name)
{

}

wxString ReferenceVolumesListControl::OnGetItemText(long item, long column) const
{
	extern MyVolumeAssetPanel *volume_asset_panel;
	MyRefinementPackageAssetPanel *parent_panel =  reinterpret_cast < MyRefinementPackageAssetPanel *> (m_parent->GetParent()->GetParent()); // not very nice code!

	if (parent_panel->all_refinement_packages.GetCount() > 0 && parent_panel->selected_refinement_package >= 0)
	{
		switch(column)
		{
		    case 0  :
		    	return wxString::Format(wxT("%li"), item + 1);
		       break;
		    case 1  :
		    	if (parent_panel->all_refinement_packages.Item(parent_panel->selected_refinement_package).references_for_next_refinement.Item(item) == -1) return wxT("Generate from params.");
		    	else
		    	return volume_asset_panel->ReturnAssetName(volume_asset_panel->ReturnArrayPositionFromAssetID(parent_panel->all_refinement_packages.Item(parent_panel->selected_refinement_package).references_for_next_refinement.Item(item)));
		       break;
		    default :
		       MyPrintWithDetails("Error, asking for column (%li) which does not exist", column);
		       return "";
		}
	}
	else
	{
		return "";
	}
}

int ReferenceVolumesListControl::ReturnGuessAtColumnTextWidth(int wanted_column)
{
	wxClientDC dc(this);
	long counter;

	wxListItem result;
	result.SetMask(wxLIST_MASK_TEXT);
	GetColumn(wanted_column, result);

	int max_width = dc.GetTextExtent(result.GetText()).x + 20;

	if (GetItemCount() < 100)
	{
		for ( counter = 0; counter < GetItemCount(); counter++)
		{
			if (dc.GetTextExtent(OnGetItemText(counter, wanted_column)).x + 20 > max_width) max_width = dc.GetTextExtent(OnGetItemText(counter, wanted_column)).x + 20;
		}
	}
	else
	{
		for ( counter = 0; counter < 50; counter++)
		{
			if (dc.GetTextExtent(OnGetItemText(counter, wanted_column)).x + 20 > max_width) max_width = dc.GetTextExtent(OnGetItemText(counter, wanted_column)).x + 20;
		}

		for ( counter = GetItemCount() - 50; counter < GetItemCount(); counter++)
		{
			if (dc.GetTextExtent(OnGetItemText(counter, wanted_column)).x + 20 > max_width) max_width = dc.GetTextExtent(OnGetItemText(counter, wanted_column)).x + 20;
		}

	}

	return max_width;
}

// Refinements..

RefinementParametersListCtrl::RefinementParametersListCtrl(wxWindow *parent, wxWindowID id, const wxPoint &pos, const wxSize &size, long style, const wxValidator &validator, const wxString &name)
:
wxListCtrl(parent, id, pos, size, style, validator, name)
{

}

wxString RefinementParametersListCtrl::OnGetItemText(long item, long column) const
{

//	wxPrintf("REturning Value for item = %li, column = %li\n", item, column);

	if (refinement_package_asset_panel->all_refinement_packages.GetCount() > 0 && refinement_results_panel->RefinementPackageComboBox->GetCount() > 0)
	{
		Refinement *current_refinement = refinement_results_panel->currently_displayed_refinement;

		//wxPrintf("getting text for column %li column\n", column);

		switch(column)
		{
		    case 0  : // position_in_stack
		    	return wxString::Format("%li",current_refinement->class_refinement_results[refinement_results_panel->current_class].particle_refinement_results[item].position_in_stack);
		    	break;
		    case 1  : // psi
		    	return wxString::Format("%.2f",current_refinement->class_refinement_results[refinement_results_panel->current_class].particle_refinement_results[item].psi);
		     	break;
		    case 2  : // theta
		    	return wxString::Format("%.2f",current_refinement->class_refinement_results[refinement_results_panel->current_class].particle_refinement_results[item].theta);
		     	break;
		    case 3  : // phi
		    	return wxString::Format("%.2f",current_refinement->class_refinement_results[refinement_results_panel->current_class].particle_refinement_results[item].phi);
		     	break;
		    case 4  : // xshift
		    	return wxString::Format("%.2f",current_refinement->class_refinement_results[refinement_results_panel->current_class].particle_refinement_results[item].xshift);
		     	break;
		    case 5  : // yshift
		    	return wxString::Format("%.2f",current_refinement->class_refinement_results[refinement_results_panel->current_class].particle_refinement_results[item].yshift);
		     	break;
		    case 6  : // defocus1
		    	return wxString::Format("%.0f",current_refinement->class_refinement_results[refinement_results_panel->current_class].particle_refinement_results[item].defocus1);
		     	break;
		    case 7  : // defocus2
		    	return wxString::Format("%.0f",current_refinement->class_refinement_results[refinement_results_panel->current_class].particle_refinement_results[item].defocus2);
		     	break;
		    case 8  : // defocus_angle
		    	return wxString::Format("%.2f",current_refinement->class_refinement_results[refinement_results_panel->current_class].particle_refinement_results[item].defocus_angle);
		     	break;
		    case 9  : // phase shift
		    	return wxString::Format("%.2f",current_refinement->class_refinement_results[refinement_results_panel->current_class].particle_refinement_results[item].phase_shift);
		       	break;
		    case 10  : // occupancy
		    	return wxString::Format("%.1f",current_refinement->class_refinement_results[refinement_results_panel->current_class].particle_refinement_results[item].occupancy);
		     	break;
		    case 11  : // logp
		    	return wxString::Format("%.2f",current_refinement->class_refinement_results[refinement_results_panel->current_class].particle_refinement_results[item].logp);
		     	break;
		    case 12  : // sigma
		    	return wxString::Format("%.2f",current_refinement->class_refinement_results[refinement_results_panel->current_class].particle_refinement_results[item].sigma);
		     	break;
		    case 13 : // score
		    	return wxString::Format("%.2f",current_refinement->class_refinement_results[refinement_results_panel->current_class].particle_refinement_results[item].score);
		     	break;
		    case 14 : // image is active
		    	if (current_refinement->class_refinement_results[refinement_results_panel->current_class].particle_refinement_results[item].image_is_active < 0) return "No";
		    	else return "Yes";
		     	break;
		   default :
		       MyPrintWithDetails("Error, asking for column (%li) which does not exist", column);
		       return "";
		}
	}
	else
	{
		return "";
	}

}

int RefinementParametersListCtrl::ReturnGuessAtColumnTextWidth(int wanted_column)
{

	wxClientDC dc(this);
	long counter;

	wxListItem result;
	result.SetMask(wxLIST_MASK_TEXT);
	GetColumn(wanted_column, result);

	int max_width = dc.GetTextExtent(result.GetText()).x + 20;

	if (GetItemCount() < 100)
	{
		for ( counter = 0; counter < GetItemCount(); counter++)
		{
			if (dc.GetTextExtent(OnGetItemText(counter, wanted_column)).x + 20 > max_width) max_width = dc.GetTextExtent(OnGetItemText(counter, wanted_column)).x + 20;
		}
	}
	else
	{
		for ( counter = 0; counter < 50; counter++)
		{
			if (dc.GetTextExtent(OnGetItemText(counter, wanted_column)).x + 20 > max_width) max_width = dc.GetTextExtent(OnGetItemText(counter, wanted_column)).x + 20;
		}

		for ( counter = GetItemCount() - 50; counter < GetItemCount(); counter++)
		{
			if (dc.GetTextExtent(OnGetItemText(counter, wanted_column)).x + 20 > max_width) max_width = dc.GetTextExtent(OnGetItemText(counter, wanted_column)).x + 20;
		}

	}

	return max_width;
}

// Classification Selection..

ClassificationSelectionListCtrl::ClassificationSelectionListCtrl(wxWindow *parent, wxWindowID id, const wxPoint &pos, const wxSize &size, long style, const wxValidator &validator, const wxString &name)
:
wxListCtrl(parent, id, pos, size, style, validator, name)
{

	current_selection_id = -10;
	selection_id_upon_clear = -10;
	current_selection = -1;

	Bind(wxEVT_LIST_DELETE_ALL_ITEMS, &ClassificationSelectionListCtrl::OnClearAll, this);
}

void ClassificationSelectionListCtrl::OnClearAll(wxListEvent& event)
{
	selection_id_upon_clear == current_selection_id ;
}

void ClassificationSelectionListCtrl::Fill(long refinement_package_asset_id, long classification_id, bool select_latest)
{
	Freeze();
	ClearAll();

	all_valid_selections.Clear();
	original_classum_selection_array_positions.Clear();
	int item_to_select = -1;

	for (int counter = 0; counter < refinement_package_asset_panel->all_classification_selections.GetCount(); counter++)
	{
		if (refinement_package_asset_panel->all_classification_selections.Item(counter).refinement_package_asset_id == refinement_package_asset_id && refinement_package_asset_panel->all_classification_selections.Item(counter).classification_id == classification_id)
		{
			all_valid_selections.Add(refinement_package_asset_panel->all_classification_selections.Item(counter));
			original_classum_selection_array_positions.Add(counter);
			if (refinement_package_asset_panel->all_classification_selections.Item(counter).selection_id == selection_id_upon_clear) item_to_select = counter;
		}
	}

	if (all_valid_selections.GetCount() > 0)
	{
		InsertColumn(0, wxT("Selection"), wxLIST_FORMAT_CENTRE,  wxLIST_AUTOSIZE_USEHEADER );
		InsertColumn(1, wxT("Creation Date"), wxLIST_FORMAT_CENTRE,  wxLIST_AUTOSIZE_USEHEADER );
		InsertColumn(2, wxT("Number Selected"), wxLIST_FORMAT_CENTRE,  wxLIST_AUTOSIZE_USEHEADER );

		SetItemCount(all_valid_selections.GetCount());
		RefreshItems(0, all_valid_selections.GetCount() - 1);

		for (int counter = 0; counter < GetColumnCount(); counter++)
		{
			SetColumnWidth(counter, ReturnGuessAtColumnTextWidth(counter));
		}

		if (select_latest == true)
		{
			SetItemState(all_valid_selections.GetCount() - 1, wxLIST_STATE_SELECTED, wxLIST_STATE_SELECTED);
			current_selection_id = all_valid_selections.Item(all_valid_selections.GetCount() - 1).selection_id;
		}
		else
		if (item_to_select != -1)
		{
			SetItemState(item_to_select, wxLIST_STATE_SELECTED, wxLIST_STATE_SELECTED);
			current_selection_id = selection_id_upon_clear;
		}
		//else
		//{
		//	SetItemState(0, wxLIST_STATE_SELECTED, wxLIST_STATE_SELECTED);
		//	current_selection_id = all_valid_selections.Item(0).selection_id;
		//}


	}

	Thaw();
}

wxString ClassificationSelectionListCtrl::OnGetItemText(long item, long column) const
{

//	wxPrintf("REturning Value for item = %li, column = %li\n", item, column);

	if (all_valid_selections.GetCount() > 0 && refine2d_results_panel->RefinementPackageComboBox->GetCount() > 0)
	{
		ClassificationSelection *current_selection = &all_valid_selections.Item(item);

		//wxPrintf("getting text for column %li column\n", column);

		switch(column)
		{
		    case 0  : // position_in_stack
		    	return current_selection->name;
		    	break;
		    case 1  : // psi
		    	return current_selection->creation_date.FormatISOCombined(' ');
		     	break;
		    case 2  : // theta
		    	return wxString::Format("%i", current_selection->number_of_selections);
		     	break;
		    default :
		       MyPrintWithDetails("Error, asking for column (%li) which does not exist", column);
		       return "";
		}

	}
	else
	{
		return "";
	}

}

int ClassificationSelectionListCtrl::ReturnGuessAtColumnTextWidth(int wanted_column)
{

	wxClientDC dc(this);
	long counter;

	wxListItem result;
	result.SetMask(wxLIST_MASK_TEXT);
	GetColumn(wanted_column, result);

	int max_width = dc.GetTextExtent(result.GetText()).x + 20;

	if (GetItemCount() < 100)
	{
		for ( counter = 0; counter < GetItemCount(); counter++)
		{
			if (dc.GetTextExtent(OnGetItemText(counter, wanted_column)).x + 20 > max_width) max_width = dc.GetTextExtent(OnGetItemText(counter, wanted_column)).x + 20;
		}
	}
	else
	{
		for ( counter = 0; counter < 50; counter++)
		{
			if (dc.GetTextExtent(OnGetItemText(counter, wanted_column)).x + 20 > max_width) max_width = dc.GetTextExtent(OnGetItemText(counter, wanted_column)).x + 20;
		}

		for ( counter = GetItemCount() - 50; counter < GetItemCount(); counter++)
		{
			if (dc.GetTextExtent(OnGetItemText(counter, wanted_column)).x + 20 > max_width) max_width = dc.GetTextExtent(OnGetItemText(counter, wanted_column)).x + 20;
		}

	}

	return max_width;
}




ClassVolumeSelectPanel::ClassVolumeSelectPanel( wxWindow* parent, wxWindowID id, const wxPoint& pos, const wxSize& size, long style ) : wxPanel( parent, id, pos, size, style )
{
	MainSizer = new wxBoxSizer( wxHORIZONTAL );

	wxBoxSizer* bSizer201;
	bSizer201 = new wxBoxSizer( wxHORIZONTAL );

	ClassText = new wxStaticText( this, wxID_ANY, wxT("Class #1 :"), wxDefaultPosition, wxDefaultSize, 0 );
	ClassText->Wrap( -1 );
	bSizer201->Add( ClassText, 0, wxALIGN_CENTER_VERTICAL|wxALL, 5 );

	VolumeComboBox = new wxComboBox( this, wxID_ANY, wxT(""), wxDefaultPosition, wxDefaultSize, 0, NULL, wxCB_READONLY );
	VolumeComboBox->Append("Generate From Params.");
	AppendVolumeAssetsToComboBox(VolumeComboBox);
	VolumeComboBox->SetSelection(0);
	bSizer201->Add( VolumeComboBox, 1, wxALL, 5 );


	MainSizer->Add( bSizer201, 1, 0, 5 );

	class_number = -1;


	this->SetSizer( MainSizer );
	this->Layout();
}

ClassVolumeSelectPanel::~ClassVolumeSelectPanel()
{
}

