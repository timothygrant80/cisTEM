#include "../core/core_headers.h"
#include "../core/gui_core_headers.h"

extern MyMainFrame *main_frame;

MyRunProfilesPanel::MyRunProfilesPanel( wxWindow* parent )
:
RunProfilesPanel( parent )
{
	selected_profile = -1;
	selected_command = -1;
	FillCommandsBox();

}

void MyRunProfilesPanel::FillProfilesBox()
{
	ProfilesListBox->Freeze();
	ProfilesListBox->ClearAll();
	ProfilesListBox->InsertColumn(0, "Profiles", wxLIST_FORMAT_LEFT, wxLIST_AUTOSIZE);

	if (run_profile_manager.number_of_run_profiles > 0)
	{

		for (long counter = 0; counter < run_profile_manager.number_of_run_profiles; counter++)
		{
			ProfilesListBox->InsertItem(counter, run_profile_manager.run_profiles[counter].name, counter);
		}

		SizeProfilesColumn();

		if (selected_profile >= 0 && selected_profile < run_profile_manager.number_of_run_profiles)
		{
			ProfilesListBox->SetItemState(selected_profile, wxLIST_STATE_SELECTED, wxLIST_STATE_SELECTED);
		}
	}

	ProfilesListBox->Thaw();
}

void MyRunProfilesPanel::FillCommandsBox()
{
	int total_number_of_jobs = 0;

	CommandsListBox->Freeze();
	CommandsListBox->ClearAll();
	CommandsListBox->InsertColumn(0, "Command", wxLIST_FORMAT_LEFT, wxLIST_AUTOSIZE);
	CommandsListBox->InsertColumn(1, "No. Copies", wxLIST_FORMAT_LEFT, wxLIST_AUTOSIZE);

	// Fill it from the run_profile_manager..

	if (run_profile_manager.number_of_run_profiles > 0)
	{
		CommandsPanel->Enable(true);

		MyDebugAssertTrue(selected_profile >= 0 && selected_profile < run_profile_manager.number_of_run_profiles, "selected_profile appears incorrect!");

		ManagerTextCtrl->SetValue(buffer_profile.manager_command);

		for (long counter = 0; counter < buffer_profile.number_of_run_commands; counter++)
		{
			CommandsListBox->InsertItem(counter, buffer_profile.run_commands[counter].command_to_run, counter);
			CommandsListBox->SetItem(counter, 1, wxString::Format(wxT("%i"), buffer_profile.run_commands[counter].number_of_copies));
			total_number_of_jobs += buffer_profile.run_commands[counter].number_of_copies;
		}

		NumberProcessesStaticText->SetLabel(wxString::Format(wxT("%i"), total_number_of_jobs));

		if (ManagerTextCtrl->GetValue().Find("$command") ==  wxNOT_FOUND)
		{
			CommandErrorStaticText->SetLabel("Oops! - Command must contain \"$command\"");
		}
		else
		{
			CommandErrorStaticText->SetLabel("");
		}

		if (selected_command >= 0 && selected_command < buffer_profile.number_of_run_commands && buffer_profile.number_of_run_commands > 0)
		{
			CommandsListBox->SetItemState(selected_command, wxLIST_STATE_SELECTED, wxLIST_STATE_SELECTED);
		}

	}
	else
	{
		ManagerTextCtrl->SetValue("");
		CommandErrorStaticText->SetLabel("");
		NumberProcessesStaticText->SetLabel("");
		CommandsPanel->Enable(false);
	}

	SizeCommandsColumns();

	CommandsListBox->Thaw();
}

void MyRunProfilesPanel::SizeProfilesColumn()
{
	int client_height;
	int client_width;

	int current_width;

	ProfilesListBox->GetClientSize(&client_width, &client_height);
	ProfilesListBox->SetColumnWidth(0, wxLIST_AUTOSIZE);

	current_width = ProfilesListBox->GetColumnWidth(0);

	if (client_width > current_width) ProfilesListBox->SetColumnWidth(0, client_width);

}

void MyRunProfilesPanel::OnRenameProfileClick(wxCommandEvent& event)
{
	MyDebugAssertTrue(selected_profile >= 0 && selected_profile < run_profile_manager.number_of_run_profiles, "Trying to rename an non existent profile!");
	ProfilesListBox->EditLabel(selected_profile);
}

void MyRunProfilesPanel::OnProfilesListItemActivated( wxListEvent& event )
{
	MyDebugAssertTrue(selected_profile >= 0 && selected_profile < run_profile_manager.number_of_run_profiles, "Trying to rename an non existent profile!");
	ProfilesListBox->EditLabel(selected_profile);
}

void MyRunProfilesPanel::SizeCommandsColumns()
{

	int client_height;
	int client_width;

	int current_name_width;
	int current_number_width;

	int remainder;

	CommandsListBox->GetClientSize(&client_width, &client_height);
	CommandsListBox->SetColumnWidth(0, -2);
	CommandsListBox->SetColumnWidth(1, -2);

	old_commands_listbox_client_width = client_width;

	current_name_width = CommandsListBox->GetColumnWidth(0);
	current_number_width = CommandsListBox->GetColumnWidth(1);

	if (current_name_width + current_number_width < client_width)
	{
		remainder = client_width - current_number_width;
		CommandsListBox->SetColumnWidth(0, remainder);
	}
}

void MyRunProfilesPanel::OnProfileLeftDown( wxMouseEvent& event )
{
	VetoInvalidMouse(ProfilesListBox, event);
}

void MyRunProfilesPanel::OnProfileDClick( wxMouseEvent& event )
{
	VetoInvalidMouse(ProfilesListBox, event);
}

void MyRunProfilesPanel::OnCommandLeftDown( wxMouseEvent& event )
{
	VetoInvalidMouse(CommandsListBox, event);
}

void MyRunProfilesPanel::OnCommandDClick( wxMouseEvent& event )
{
	VetoInvalidMouse(CommandsListBox, event);
}


void MyRunProfilesPanel::VetoInvalidMouse( wxListCtrl *wanted_list, wxMouseEvent& event )
{
	// Don't allow clicking on anything other than item, to stop the selection bar changing

	int flags;

	if (wanted_list->HitTest(event.GetPosition(), flags)  !=  wxNOT_FOUND)
	{
		event.Skip();
	}
}

void MyRunProfilesPanel::MouseVeto( wxMouseEvent& event )
{
	//Do nothing

}

void MyRunProfilesPanel::OnEndProfileEdit( wxListEvent& event )
{
	if (event.GetLabel() == wxEmptyString)
	{
		event.Veto();
	}
	else
	{
		SetProfileName(event.GetIndex(), event.GetLabel());
		main_frame->current_project.database.AddOrReplaceRunProfile(run_profile_manager.ReturnProfilePointer(event.GetIndex()));
	}

}

void  MyRunProfilesPanel::OnUpdateIU( wxUpdateUIEvent& event )
{
	if (main_frame->current_project.is_open == false)
	{
		Enable(false);
	}
	else
	{
		Enable(true);
		int commands_listbox_client_height;
		int commands_listbox_client_width;

		CommandsListBox->GetClientSize(&commands_listbox_client_width, &commands_listbox_client_height);

		if (commands_listbox_client_width != old_commands_listbox_client_width)
		{
			SizeCommandsColumns();
		}

		if (run_profile_manager.number_of_run_profiles <= 0 || selected_profile < 0 || selected_profile >= run_profile_manager.number_of_run_profiles)
		{
			RemoveProfileButton->Enable(false);
			RenameProfileButton->Enable(false);
		}
		else
		{
			RemoveProfileButton->Enable(true);
			RenameProfileButton->Enable(true);
		}

		if (buffer_profile.number_of_run_commands <= 0 || selected_command < 0 || selected_command >= buffer_profile.number_of_run_commands)
		{
			RemoveCommandButton->Enable(false);
			EditCommandButton->Enable(false);
		}
		else
		{
			RemoveCommandButton->Enable(true);

			if (selected_command >= 0 && selected_command < buffer_profile.number_of_run_commands)
			{
				EditCommandButton->Enable(true);
			}
			else
			{
				EditCommandButton->Enable(false);
			}

		}

		if (command_panel_has_changed == true) CommandsSaveButton->Enable(true);
		else CommandsSaveButton->Enable(false);
	}

}

void MyRunProfilesPanel::OnAddProfileClick( wxCommandEvent& event )
{
	run_profile_manager.AddBlankProfile();
	main_frame->current_project.database.AddOrReplaceRunProfile(run_profile_manager.ReturnLastProfilePointer());


	FillProfilesBox();
	SetSelectedProfile(run_profile_manager.number_of_run_profiles - 1);


}

void MyRunProfilesPanel::OnRemoveProfileClick( wxCommandEvent& event )
{
	if (selected_profile != -1)
	{
		MyDebugAssertTrue(selected_profile >= 0 && selected_profile < run_profile_manager.number_of_run_profiles, "Trying to remove a profile that doesn't exist!");

		main_frame->current_project.database.DeleteRunProfile(run_profile_manager.run_profiles[selected_profile].id);
		run_profile_manager.RemoveProfile(selected_profile);

		FillProfilesBox();

		if (selected_profile > 0) SetSelectedProfile(selected_profile - 1);
		else
		if (selected_profile < run_profile_manager.number_of_run_profiles) SetSelectedProfile(selected_profile);
		else
		{
			selected_profile = -1;
			FillCommandsBox();

		}
	}
}

void MyRunProfilesPanel::CommandsSaveButtonClick( wxCommandEvent& event )
{
	// check the manager text..

	if (ManagerTextCtrl->GetValue().Find("$command") ==  wxNOT_FOUND)
	{
		CommandErrorStaticText->SetLabel("Oops! - Command must contain \"$command\"");
	}
	else
	{
		Freeze();
		CommandErrorStaticText->SetLabel("");
		Layout();
		Thaw();

		buffer_profile.manager_command = ManagerTextCtrl->GetValue();
		run_profile_manager.run_profiles[selected_profile] = buffer_profile;
		main_frame->current_project.database.AddOrReplaceRunProfile(run_profile_manager.ReturnProfilePointer(selected_profile));

		command_panel_has_changed = false;
	}

}

void MyRunProfilesPanel::SetProfileName(long wanted_group, wxString wanted_name)
{
	run_profile_manager.run_profiles[wanted_group].name = wanted_name;
}

void MyRunProfilesPanel::SetSelectedProfile(long wanted_profile)
{
	//wxPrintf("Selecting %li\n", wanted_profile);
	MyDebugAssertTrue(wanted_profile >= 0 && wanted_profile < run_profile_manager.number_of_run_profiles, "Trying to select a profile that doesn't exist!");

	ProfilesListBox->SetItemState(wanted_profile, wxLIST_STATE_SELECTED, wxLIST_STATE_SELECTED);
	selected_profile = wanted_profile;



	//FillCommandsBox();
}

void MyRunProfilesPanel::SetSelectedCommand(long wanted_command)
{

	MyDebugAssertTrue(wanted_command >= 0 && wanted_command < buffer_profile.number_of_run_commands, "Trying to select a command that doesn't exist!");

	CommandsListBox->SetItemState(wanted_command, wxLIST_STATE_SELECTED, wxLIST_STATE_SELECTED);
	selected_command = wanted_command;
	//wxPrintf("set_command = %li\n", selected_command);

}


void MyRunProfilesPanel::OnProfilesFocusChange( wxListEvent& event )
{

	if (event.GetIndex() >= 0)
	{
		selected_profile = event.GetIndex();

		buffer_profile = run_profile_manager.run_profiles[selected_profile];


		if (run_profile_manager.run_profiles[selected_profile].number_of_run_commands > 0) selected_command = 0;
		else selected_command = -1;

		FillCommandsBox();
		command_panel_has_changed = false;
	}

	event.Skip();
}

void MyRunProfilesPanel::OnCommandsFocusChange( wxListEvent& event )
{

	if (event.GetIndex() >= 0)
	{
		selected_command = event.GetIndex();
	}

	//wxPrintf("selected_command = %li\n", selected_command);

	event.Skip();
}


void MyRunProfilesPanel::ManagerTextChanged( wxCommandEvent& event )
{
	command_panel_has_changed = true;
	event.Skip();
}

void MyRunProfilesPanel::AddCommandButtonClick( wxCommandEvent& event )
{
	buffer_profile.AddCommand("$command", 1);
	FillCommandsBox();

	SetSelectedCommand(buffer_profile.number_of_run_commands - 1);

	command_panel_has_changed = true;
}

void MyRunProfilesPanel::RemoveCommandButtonClick( wxCommandEvent& event )
{
	if (selected_command != -1)
	{
		MyDebugAssertTrue(selected_command >= 0 && selected_command < buffer_profile.number_of_run_commands, "Trying to remove a command that doesn't exist!");

		buffer_profile.RemoveCommand(selected_command);
		FillCommandsBox();

		if (selected_command > 0) SetSelectedCommand(selected_command - 1);
		else
		if (selected_command < buffer_profile.number_of_run_commands) SetSelectedCommand(selected_command);
		else
		{
			selected_command = -1;
		}
	}
}


void MyRunProfilesPanel::OnCommandsActivated( wxListEvent& event )
{
	EditCommand();
}

void MyRunProfilesPanel::EditCommandButtonClick( wxCommandEvent& event )
{
	EditCommand();
}

void MyRunProfilesPanel::EditCommand()
{
	MyDebugAssertTrue(selected_command >= 0 && selected_command < buffer_profile.number_of_run_commands, "Trying to edit a command that doesn't exist!");

	MyAddRunCommandDialog *add_dialog = new MyAddRunCommandDialog(this);

	// Set the current values..

	add_dialog->CommandTextCtrl->SetValue(buffer_profile.run_commands[selected_command].command_to_run);
	add_dialog->NumberCopiesSpinCtrl->SetValue(buffer_profile.run_commands[selected_command].number_of_copies);
	add_dialog->ShowModal();
}

void MyRunProfilesPanel::ImportAllFromDatabase()
{
	RunProfile temp_profile;

	run_profile_manager.RemoveAllProfiles();

	main_frame->current_project.database.BeginAllRunProfilesSelect();

	while (main_frame->current_project.database.last_return_code == SQLITE_ROW)
	{
		temp_profile = main_frame->current_project.database.GetNextRunProfile();
		run_profile_manager.AddProfile(&temp_profile);

	}

	main_frame->current_project.database.EndAllRunProfilesSelect();

	if (run_profile_manager.number_of_run_profiles > 0) selected_profile = 0;
	FillProfilesBox();
	FillCommandsBox();
	command_panel_has_changed = false;


}

void MyRunProfilesPanel::Reset()
{
	run_profile_manager.RemoveAllProfiles();

	FillProfilesBox();
	FillCommandsBox();


}


