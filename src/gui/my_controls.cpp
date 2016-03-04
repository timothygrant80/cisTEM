#include "../core/core_headers.h"
#include "../core/gui_core_headers.h"

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

	double initial_value;
	if (value.ToDouble(&initial_value) == false) initial_value = 0.0;

	previous_value = initial_value;
	ChangeValue(wxString::Format("%.2f", initial_value));

}

NumericTextCtrl::~NumericTextCtrl()
{
	Unbind(wxEVT_CHAR, &NumericTextCtrl::OnKeyPress, this);
	Unbind(wxEVT_TEXT_ENTER, &NumericTextCtrl::OnEnterPressed, this );
	Unbind(wxEVT_KILL_FOCUS, &NumericTextCtrl::OnFocusLost, this );
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
		ChangeValue(wxString::Format("%.2f", previous_value));
	}
	else
	{
		if (current_value < min_value)
		{
			ChangeValue(wxString::Format("%.2f", min_value));
			previous_value = min_value;
		}
		else
		if (current_value > max_value)
		{
			ChangeValue(wxString::Format("%.2f", max_value));
			previous_value = max_value;
		}
		else
		{
			ChangeValue(wxString::Format("%.2f", current_value));
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




