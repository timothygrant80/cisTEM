//#include "../core/core_headers.h"
#include "../core/gui_core_headers.h"

extern MyMainFrame* main_frame;

MyRunProfilesPanel::MyRunProfilesPanel(wxWindow* parent)
    : RunProfilesPanel(parent) {
    selected_profile = -1;
    selected_command = -1;
    is_dirty         = false;
    FillCommandsBox( );
}

void MyRunProfilesPanel::FillProfilesBox( ) {
    ProfilesListBox->Freeze( );
    ProfilesListBox->ClearAll( );
    ProfilesListBox->InsertColumn(0, "Profiles", wxLIST_FORMAT_LEFT, wxLIST_AUTOSIZE);

    if ( run_profile_manager.number_of_run_profiles > 0 ) {

        for ( long counter = 0; counter < run_profile_manager.number_of_run_profiles; counter++ ) {
            ProfilesListBox->InsertItem(counter, run_profile_manager.run_profiles[counter].name, counter);
        }

        SizeProfilesColumn( );

        if ( selected_profile >= 0 && selected_profile < run_profile_manager.number_of_run_profiles ) {
            ProfilesListBox->SetItemState(selected_profile, wxLIST_STATE_SELECTED, wxLIST_STATE_SELECTED);
        }
    }

    ProfilesListBox->Thaw( );
}

void MyRunProfilesPanel::FillCommandsBox( ) {
    int total_number_of_jobs = 0;

    CommandsListBox->Freeze( );
    CommandsListBox->ClearAll( );
    CommandsListBox->InsertColumn(0, "Command", wxLIST_FORMAT_LEFT, wxLIST_AUTOSIZE);
    CommandsListBox->InsertColumn(1, "No. Copies", wxLIST_FORMAT_LEFT, wxLIST_AUTOSIZE);
    CommandsListBox->InsertColumn(2, "No. Threads Per Copy", wxLIST_FORMAT_LEFT, wxLIST_AUTOSIZE);
    CommandsListBox->InsertColumn(3, "Override Total No. Copies?", wxLIST_FORMAT_LEFT, wxLIST_AUTOSIZE);
    CommandsListBox->InsertColumn(4, "Overriden Total No. Copies?", wxLIST_FORMAT_LEFT, wxLIST_AUTOSIZE);
    CommandsListBox->InsertColumn(5, "Launch Delay (ms)", wxLIST_FORMAT_LEFT, wxLIST_AUTOSIZE);

    // Fill it from the run_profile_manager..

    if ( run_profile_manager.number_of_run_profiles > 0 ) {
        CommandsPanel->Enable(true);

        MyDebugAssertTrue(selected_profile >= 0 && selected_profile < run_profile_manager.number_of_run_profiles, "selected_profile appears incorrect!");

        ManagerTextCtrl->ChangeValue(buffer_profile.manager_command);

        if ( buffer_profile.gui_address == "" ) {
            GuiAddressStaticText->SetLabel("Automatic");
            GuiAutoButton->Enable(false);
        }
        else {
            GuiAddressStaticText->SetLabel(buffer_profile.gui_address);
            GuiAutoButton->Enable(true);
        }

        if ( buffer_profile.controller_address == "" ) {
            ControllerAddressStaticText->SetLabel("Automatic");
            ControllerAutoButton->Enable(false);
        }
        else {
            ControllerAddressStaticText->SetLabel(buffer_profile.controller_address);
            ControllerAutoButton->Enable(true);
        }

        for ( long counter = 0; counter < buffer_profile.number_of_run_commands; counter++ ) {
            CommandsListBox->InsertItem(counter, buffer_profile.run_commands[counter].command_to_run, counter);
            CommandsListBox->SetItem(counter, 1, wxString::Format(wxT("%i"), buffer_profile.run_commands[counter].number_of_copies));
            CommandsListBox->SetItem(counter, 2, wxString::Format(wxT("%i"), buffer_profile.run_commands[counter].number_of_threads_per_copy));

            if ( buffer_profile.run_commands[counter].override_total_copies == true )
                CommandsListBox->SetItem(counter, 3, "Yes");
            else
                CommandsListBox->SetItem(counter, 3, "No");
            CommandsListBox->SetItem(counter, 4, wxString::Format(wxT("%i"), buffer_profile.run_commands[counter].overriden_number_of_copies));
            CommandsListBox->SetItem(counter, 5, wxString::Format(wxT("%i"), buffer_profile.run_commands[counter].delay_time_in_ms));
        }

        total_number_of_jobs = buffer_profile.ReturnTotalJobs( );

        NumberProcessesStaticText->SetLabel(wxString::Format(wxT("%i"), total_number_of_jobs));

        if ( ManagerTextCtrl->GetValue( ).Find("$command") == wxNOT_FOUND ) {
            CommandErrorStaticText->SetLabel("Oops! - Command must contain \"$command\"");
        }
        else {
            CommandErrorStaticText->SetLabel("");
        }

        if ( selected_command >= 0 && selected_command < buffer_profile.number_of_run_commands && buffer_profile.number_of_run_commands > 0 ) {
            CommandsListBox->SetItemState(selected_command, wxLIST_STATE_SELECTED, wxLIST_STATE_SELECTED);
        }
    }
    else {
        ManagerTextCtrl->ChangeValue("");
        CommandErrorStaticText->SetLabel("");
        NumberProcessesStaticText->SetLabel("");
        CommandsPanel->Enable(false);
        ControllerAddressStaticText->SetLabel("");
        GuiAddressStaticText->SetLabel("");
    }

    SizeCommandsColumns( );

    CommandsListBox->Thaw( );
}

void MyRunProfilesPanel::SizeProfilesColumn( ) {
    int client_height;
    int client_width;

    int current_width;

    ProfilesListBox->GetClientSize(&client_width, &client_height);
    ProfilesListBox->SetColumnWidth(0, wxLIST_AUTOSIZE);

    current_width = ProfilesListBox->GetColumnWidth(0);

    if ( client_width > current_width )
        ProfilesListBox->SetColumnWidth(0, client_width);
}

void MyRunProfilesPanel::OnRenameProfileClick(wxCommandEvent& event) {
    MyDebugAssertTrue(selected_profile >= 0 && selected_profile < run_profile_manager.number_of_run_profiles, "Trying to rename an non existent profile!");
    ProfilesListBox->EditLabel(selected_profile);
}

void MyRunProfilesPanel::OnProfilesListItemActivated(wxListEvent& event) {
    MyDebugAssertTrue(selected_profile >= 0 && selected_profile < run_profile_manager.number_of_run_profiles, "Trying to rename an non existent profile!");
    ProfilesListBox->EditLabel(selected_profile);
}

void MyRunProfilesPanel::SizeCommandsColumns( ) {

    int client_height;
    int client_width;

    int current_name_width;
    int current_number_width;
    int current_number_threads_width;
    int current_should_override_number_width;
    int current_overriden_number_width;
    int current_delay_width;

    int remainder;

    CommandsListBox->GetClientSize(&client_width, &client_height);
    CommandsListBox->SetColumnWidth(0, -2);
    CommandsListBox->SetColumnWidth(1, -2);
    CommandsListBox->SetColumnWidth(2, -2);
    CommandsListBox->SetColumnWidth(3, -2);
    CommandsListBox->SetColumnWidth(4, -2);
    CommandsListBox->SetColumnWidth(5, -2);

    old_commands_listbox_client_width = client_width;

    current_name_width                   = CommandsListBox->GetColumnWidth(0);
    current_number_width                 = CommandsListBox->GetColumnWidth(1);
    current_number_threads_width         = CommandsListBox->GetColumnWidth(2);
    current_should_override_number_width = CommandsListBox->GetColumnWidth(3);
    current_overriden_number_width       = CommandsListBox->GetColumnWidth(4);
    current_delay_width                  = CommandsListBox->GetColumnWidth(5);

    if ( current_name_width + current_number_width + current_delay_width + current_number_threads_width + current_should_override_number_width + current_overriden_number_width < client_width ) {
        remainder = client_width - current_number_width - current_delay_width;
        CommandsListBox->SetColumnWidth(0, remainder);
    }
}

void MyRunProfilesPanel::OnProfileLeftDown(wxMouseEvent& event) {
    VetoInvalidMouse(ProfilesListBox, event);
}

void MyRunProfilesPanel::OnProfileDClick(wxMouseEvent& event) {
    VetoInvalidMouse(ProfilesListBox, event);
}

void MyRunProfilesPanel::OnCommandLeftDown(wxMouseEvent& event) {
    VetoInvalidMouse(CommandsListBox, event);
}

void MyRunProfilesPanel::OnCommandDClick(wxMouseEvent& event) {
    VetoInvalidMouse(CommandsListBox, event);
}

void MyRunProfilesPanel::VetoInvalidMouse(wxListCtrl* wanted_list, wxMouseEvent& event) {
    // Don't allow clicking on anything other than item, to stop the selection bar changing

    int flags;

    if ( wanted_list->HitTest(event.GetPosition( ), flags) != wxNOT_FOUND ) {
        event.Skip( );
    }
}

void MyRunProfilesPanel::MouseVeto(wxMouseEvent& event) {
    //Do nothing
}

void MyRunProfilesPanel::OnEndProfileEdit(wxListEvent& event) {
    if ( event.GetLabel( ) == wxEmptyString ) {
        event.Veto( );
    }
    else {
        SetProfileName(event.GetIndex( ), event.GetLabel( ));
        main_frame->current_project.database.AddOrReplaceRunProfile(run_profile_manager.ReturnProfilePointer(event.GetIndex( )));
        event.Skip( );
    }
}

void MyRunProfilesPanel::OnUpdateUI(wxUpdateUIEvent& event) {
    if ( main_frame->current_project.is_open == false ) {
        Enable(false);
    }
    else {
        Enable(true);
        int commands_listbox_client_height;
        int commands_listbox_client_width;

        CommandsListBox->GetClientSize(&commands_listbox_client_width, &commands_listbox_client_height);

        if ( commands_listbox_client_width != old_commands_listbox_client_width ) {
            SizeCommandsColumns( );
        }

        if ( run_profile_manager.number_of_run_profiles <= 0 || selected_profile < 0 || selected_profile >= run_profile_manager.number_of_run_profiles ) {
            RemoveProfileButton->Enable(false);
            RenameProfileButton->Enable(false);
            DuplicateProfileButton->Enable(false);
            ExportButton->Enable(false);
        }
        else {
            RemoveProfileButton->Enable(true);
            RenameProfileButton->Enable(true);
            DuplicateProfileButton->Enable(true);
            ExportButton->Enable(true);
        }

        if ( buffer_profile.number_of_run_commands <= 0 || selected_command < 0 || selected_command >= buffer_profile.number_of_run_commands ) {
            RemoveCommandButton->Enable(false);
            EditCommandButton->Enable(false);
        }
        else {
            RemoveCommandButton->Enable(true);

            if ( selected_command >= 0 && selected_command < buffer_profile.number_of_run_commands ) {
                EditCommandButton->Enable(true);
            }
            else {
                EditCommandButton->Enable(false);
            }
        }

        if ( command_panel_has_changed == true )
            CommandsSaveButton->Enable(true);
        else
            CommandsSaveButton->Enable(false);

        if ( is_dirty == true ) {
            FillProfilesBox( );
            FillCommandsBox( );
            is_dirty = false;
        }
    }
}

void MyRunProfilesPanel::OnImportButtonClick(wxCommandEvent& event) {
    // get the file to to open..

    wxFileDialog openFileDialog(this, _("Open txt file"), "", "", "txt files (*.txt)|*.txt;*.TXT", wxFD_OPEN | wxFD_FILE_MUST_EXIST);

    if ( openFileDialog.ShowModal( ) != wxID_CANCEL ) {
        if ( ImportRunProfilesFromDisk(openFileDialog.GetPath( )) == false ) {
            wxMessageDialog error_dialog(this, "Error importing File", "Error", wxOK | wxICON_ERROR);
            error_dialog.ShowModal( );
        }
    }
}

void MyRunProfilesPanel::OnDuplicateProfileClick(wxCommandEvent& event) {
    RunProfile copy;
    copy = run_profile_manager.run_profiles[selected_profile];

    copy.id   = run_profile_manager.current_id_number + 1;
    copy.name = "Copy of " + copy.name;
    run_profile_manager.AddProfile(&copy);

    main_frame->current_project.database.AddOrReplaceRunProfile(run_profile_manager.ReturnLastProfilePointer( ));
    main_frame->DirtyRunProfiles( );

    FillProfilesBox( );
    SetSelectedProfile(run_profile_manager.number_of_run_profiles - 1);
}

void MyRunProfilesPanel::OnExportButtonClick(wxCommandEvent& event) {
    // Find out which Profiles they want exported..

    wxString* all_run_profiles = new wxString[run_profile_manager.number_of_run_profiles];

    for ( int counter = 0; counter < run_profile_manager.number_of_run_profiles; counter++ ) {
        all_run_profiles[counter] = run_profile_manager.ReturnProfileName(counter);
    }

    wxMultiChoiceDialog* choice_dialog = new wxMultiChoiceDialog(this, "Export Profile", "Select Profiles to Export", run_profile_manager.number_of_run_profiles, all_run_profiles);
    if ( choice_dialog->ShowModal( ) == wxID_OK ) {
        // get the profiles which the user wants to export..

        wxArrayInt user_selections = choice_dialog->GetSelections( );
        choice_dialog->Destroy( );

        if ( user_selections.GetCount( ) > 0 ) {
            // Get an output file..

            ProperOverwriteCheckSaveDialog saveFileDialog(this, _("Save txt file"), "txt files (*.txt;*.TXT)|*.txt", ".txt");

            if ( saveFileDialog.ShowModal( ) == wxID_OK ) {
                WriteRunProfilesToDisk(saveFileDialog.GetPath( ), user_selections);
            }
        }
    }
    else
        choice_dialog->Destroy( );

    delete[] all_run_profiles;
}

void MyRunProfilesPanel::WriteRunProfilesToDisk(wxString filename, wxArrayInt profiles_to_write) {
    wxTextFile output_file;

    int profile_counter;
    int command_counter;

    if ( DoesFileExist(filename) == true ) {
        output_file.Open(filename);
        output_file.Clear( );
    }
    else {
        output_file.Create(filename);
    }

    output_file.AddLine(wxString::Format("number_of_profiles=%i", int(profiles_to_write.GetCount( ))));

    for ( profile_counter = 0; profile_counter < profiles_to_write.GetCount( ); profile_counter++ ) {
        output_file.AddLine(wxString::Format("profile_%i_name=\"%s\"", profile_counter, run_profile_manager.run_profiles[profiles_to_write.Item(profile_counter)].name));
        output_file.AddLine(wxString::Format("profile_%i_manager_command=\"%s\"", profile_counter, run_profile_manager.run_profiles[profiles_to_write.Item(profile_counter)].manager_command));
        output_file.AddLine(wxString::Format("profile_%i_gui_address=\"%s\"", profile_counter, run_profile_manager.run_profiles[profiles_to_write.Item(profile_counter)].gui_address));
        output_file.AddLine(wxString::Format("profile_%i_controller_address=\"%s\"", profile_counter, run_profile_manager.run_profiles[profiles_to_write.Item(profile_counter)].controller_address));
        output_file.AddLine(wxString::Format("profile_%i_number_of_run_commands=%li", profile_counter, run_profile_manager.run_profiles[profiles_to_write.Item(profile_counter)].number_of_run_commands));

        for ( command_counter = 0; command_counter < run_profile_manager.run_profiles[profiles_to_write.Item(profile_counter)].number_of_run_commands; command_counter++ ) {
            output_file.AddLine(wxString::Format("profile_%i_command_%i_command_to_run=\"%s\"", profile_counter, command_counter, run_profile_manager.run_profiles[profiles_to_write.Item(profile_counter)].run_commands[command_counter].command_to_run));
            output_file.AddLine(wxString::Format("profile_%i_command_%i_number_of_copies=%i", profile_counter, command_counter, run_profile_manager.run_profiles[profiles_to_write.Item(profile_counter)].run_commands[command_counter].number_of_copies));
            output_file.AddLine(wxString::Format("profile_%i_command_%i_number_of_threads_per_copy=%i", profile_counter, command_counter, run_profile_manager.run_profiles[profiles_to_write.Item(profile_counter)].run_commands[command_counter].number_of_threads_per_copy));
            output_file.AddLine(wxString::Format("profile_%i_command_%i_should_override_total_number_of_copies=%i", profile_counter, command_counter, int(run_profile_manager.run_profiles[profiles_to_write.Item(profile_counter)].run_commands[command_counter].override_total_copies)));
            output_file.AddLine(wxString::Format("profile_%i_command_%i_overriden_total_number_of_copies=%i", profile_counter, command_counter, run_profile_manager.run_profiles[profiles_to_write.Item(profile_counter)].run_commands[command_counter].overriden_number_of_copies));
            output_file.AddLine(wxString::Format("profile_%i_command_%i_delay_time_in_ms=%i", profile_counter, command_counter, run_profile_manager.run_profiles[profiles_to_write.Item(profile_counter)].run_commands[command_counter].delay_time_in_ms));
        }
    }

    output_file.Write( );
    output_file.Close( );
}

bool MyRunProfilesPanel::ImportRunProfilesFromDisk(wxString filename) {
    wxTextFile input_file;

    int profile_counter;
    int command_counter;

    wxString line_buffer;
    wxString buffer_command_to_run;
    long     buffer_number_of_run_commands;
    long     buffer_number_of_copies;
    long     buffer_number_of_threads;
    long     buffer_override_total_jobs;
    long     buffer_overriden_total_jobs;
    long     buffer_delay_time_in_ms;
    long     number_of_profiles;
    bool     success;

    if ( DoesFileExist(filename) == false )
        return false;

    input_file.Open(filename);

    // get number of profiles..

    line_buffer = input_file.GetFirstLine( );

    if ( line_buffer.Replace("number_of_profiles=", "") != 1 )
        return false;
    success = line_buffer.ToLong(&number_of_profiles);
    if ( success == false )
        return false;

    RunProfile profiles_buffer[number_of_profiles];

    for ( profile_counter = 0; profile_counter < number_of_profiles; profile_counter++ ) {
        line_buffer = input_file.GetNextLine( );
        if ( line_buffer.Replace(wxString::Format("profile_%i_name=\"", profile_counter), "") != 1 )
            return false;
        profiles_buffer[profile_counter].name = line_buffer.Trim(false).Trim(true).Truncate(line_buffer.Length( ) - 1);

        line_buffer = input_file.GetNextLine( );
        if ( line_buffer.Replace(wxString::Format("profile_%i_manager_command=\"", profile_counter), "") != 1 )
            return false;
        profiles_buffer[profile_counter].manager_command = line_buffer.Trim(false).Trim(true).Truncate(line_buffer.Length( ) - 1);

        line_buffer = input_file.GetNextLine( );
        if ( line_buffer.Replace(wxString::Format("profile_%i_gui_address=\"", profile_counter), "") != 1 )
            return false;
        profiles_buffer[profile_counter].gui_address = line_buffer.Trim(false).Trim(true).Truncate(line_buffer.Length( ) - 1);

        line_buffer = input_file.GetNextLine( );
        if ( line_buffer.Replace(wxString::Format("profile_%i_controller_address=\"", profile_counter), "") != 1 )
            return false;
        profiles_buffer[profile_counter].controller_address = line_buffer.Trim(false).Trim(true).Truncate(line_buffer.Length( ) - 1);

        line_buffer = input_file.GetNextLine( );
        if ( line_buffer.Replace(wxString::Format("profile_%i_number_of_run_commands=", profile_counter), "") != 1 )
            return false;
        success = line_buffer.Trim(false).Trim(true).ToLong(&buffer_number_of_run_commands);
        if ( success == false )
            return false;

        wxPrintf("Got here\n");

        for ( command_counter = 0; command_counter < buffer_number_of_run_commands; command_counter++ ) {
            line_buffer = input_file.GetNextLine( );
            if ( line_buffer.Replace(wxString::Format("profile_%i_command_%i_command_to_run=\"", profile_counter, command_counter), "") != 1 )
                return false;
            buffer_command_to_run = line_buffer.Trim(false).Trim(true).Truncate(line_buffer.Length( ) - 1);

            line_buffer = input_file.GetNextLine( );
            if ( line_buffer.Replace(wxString::Format("profile_%i_command_%i_number_of_copies=", profile_counter, command_counter), "") != 1 )
                return false;
            success = line_buffer.Trim(false).Trim(true).ToLong(&buffer_number_of_copies);
            if ( success == false )
                return false;

            line_buffer = input_file.GetNextLine( );
            if ( line_buffer.Replace(wxString::Format("profile_%i_command_%i_number_of_threads_per_copy=", profile_counter, command_counter), "") != 1 )
                return false;
            success = line_buffer.Trim(false).Trim(true).ToLong(&buffer_number_of_threads);
            if ( success == false )
                return false;

            line_buffer = input_file.GetNextLine( );
            if ( line_buffer.Replace(wxString::Format("profile_%i_command_%i_should_override_total_number_of_copies=", profile_counter, command_counter), "") != 1 )
                return false;
            success = line_buffer.Trim(false).Trim(true).ToLong(&buffer_override_total_jobs);
            if ( success == false )
                return false;

            line_buffer = input_file.GetNextLine( );
            if ( line_buffer.Replace(wxString::Format("profile_%i_command_%i_overriden_total_number_of_copies=", profile_counter, command_counter), "") != 1 )
                return false;
            success = line_buffer.Trim(false).Trim(true).ToLong(&buffer_overriden_total_jobs);
            if ( success == false )
                return false;

            line_buffer = input_file.GetNextLine( );
            if ( line_buffer.Replace(wxString::Format("profile_%i_command_%i_delay_time_in_ms=", profile_counter, command_counter), "") != 1 )
                return false;
            success = line_buffer.Trim(false).Trim(true).ToLong(&buffer_delay_time_in_ms);
            if ( success == false )
                return false;

            profiles_buffer[profile_counter].AddCommand(buffer_command_to_run, int(buffer_number_of_copies), int(buffer_number_of_threads), bool(buffer_override_total_jobs), int(buffer_overriden_total_jobs), int(buffer_delay_time_in_ms));
        }
    }

    // actually add them..

    for ( profile_counter = 0; profile_counter < number_of_profiles; profile_counter++ ) {
        profiles_buffer[profile_counter].id = run_profile_manager.current_id_number;
        run_profile_manager.AddProfile(&profiles_buffer[profile_counter]);
        main_frame->current_project.database.AddOrReplaceRunProfile(run_profile_manager.ReturnLastProfilePointer( ));
        profiles_buffer[profile_counter].id = run_profile_manager.current_id_number++;
    }

    if ( selected_profile < 0 )
        selected_profile = 0;
    main_frame->DirtyRunProfiles( );
    return true;
}

void MyRunProfilesPanel::OnAddProfileClick(wxCommandEvent& event) {
    /*
	run_profile_manager.AddBlankProfile();
	main_frame->current_project.database.AddOrReplaceRunProfile(run_profile_manager.ReturnLastProfilePointer());
	main_frame->DirtyRunProfiles();


	FillProfilesBox();
	SetSelectedProfile(run_profile_manager.number_of_run_profiles - 1);
	*/

    AddDefaultLocalProfile( );
}

void MyRunProfilesPanel::AddDefaultLocalProfile( ) {
    run_profile_manager.AddDefaultLocalProfile( );
    main_frame->current_project.database.AddOrReplaceRunProfile(run_profile_manager.ReturnLastProfilePointer( ));
    main_frame->DirtyRunProfiles( );

    FillProfilesBox( );
    SetSelectedProfile(run_profile_manager.number_of_run_profiles - 1);
}

void MyRunProfilesPanel::OnRemoveProfileClick(wxCommandEvent& event) {
    if ( selected_profile != -1 ) {
        MyDebugAssertTrue(selected_profile >= 0 && selected_profile < run_profile_manager.number_of_run_profiles, "Trying to remove a profile that doesn't exist!");

        main_frame->current_project.database.DeleteRunProfile(run_profile_manager.run_profiles[selected_profile].id);
        main_frame->DirtyRunProfiles( );
        run_profile_manager.RemoveProfile(selected_profile);

        FillProfilesBox( );

        if ( selected_profile > 0 )
            SetSelectedProfile(selected_profile - 1);
        else if ( selected_profile < run_profile_manager.number_of_run_profiles )
            SetSelectedProfile(selected_profile);
        else {
            selected_profile = -1;
            FillCommandsBox( );
        }
    }
}

void MyRunProfilesPanel::GuiAddressAutoClick(wxCommandEvent& event) {
    buffer_profile.gui_address = "";
    command_panel_has_changed  = true;
    FillCommandsBox( );
}

void MyRunProfilesPanel::GuiAddressSpecifyClick(wxCommandEvent& event) {
    wxTextEntryDialog temp_dialog(this, "Wanted GUI Address :-", "Set Address", buffer_profile.gui_address);

    if ( temp_dialog.ShowModal( ) == wxID_OK ) {
        buffer_profile.gui_address = temp_dialog.GetValue( );
        command_panel_has_changed  = true;
        FillCommandsBox( );
    }
}

void MyRunProfilesPanel::ControllerAddressAutoClick(wxCommandEvent& event) {
    buffer_profile.controller_address = "";
    command_panel_has_changed         = true;
    FillCommandsBox( );
}

void MyRunProfilesPanel::ControllerAddressSpecifyClick(wxCommandEvent& event) {
    wxTextEntryDialog temp_dialog(this, "Wanted Controller Address :-", "Set Address", buffer_profile.controller_address);

    if ( temp_dialog.ShowModal( ) == wxID_OK ) {
        buffer_profile.controller_address = temp_dialog.GetValue( );
        command_panel_has_changed         = true;
        FillCommandsBox( );
    }
}

void MyRunProfilesPanel::CommandsSaveButtonClick(wxCommandEvent& event) {
    // check the manager text..

    if ( ManagerTextCtrl->GetValue( ).Find("$command") == wxNOT_FOUND ) {
        CommandErrorStaticText->SetLabel("Oops! - Command must contain \"$command\"");
    }
    else {
        Freeze( );
        CommandErrorStaticText->SetLabel("");
        Layout( );
        Thaw( );

        buffer_profile.manager_command                     = ManagerTextCtrl->GetValue( );
        run_profile_manager.run_profiles[selected_profile] = buffer_profile;
        main_frame->current_project.database.AddOrReplaceRunProfile(run_profile_manager.ReturnProfilePointer(selected_profile));
        main_frame->DirtyRunProfiles( );

        command_panel_has_changed = false;
    }
}

void MyRunProfilesPanel::SetProfileName(long wanted_group, wxString wanted_name) {
    run_profile_manager.run_profiles[wanted_group].name = wanted_name;
    main_frame->DirtyRunProfiles( );
}

void MyRunProfilesPanel::SetSelectedProfile(long wanted_profile) {
    //wxPrintf("Selecting %li\n", wanted_profile);
    MyDebugAssertTrue(wanted_profile >= 0 && wanted_profile < run_profile_manager.number_of_run_profiles, "Trying to select a profile that doesn't exist!");

    ProfilesListBox->SetItemState(wanted_profile, wxLIST_STATE_SELECTED, wxLIST_STATE_SELECTED);
    selected_profile = wanted_profile;

    //FillCommandsBox();
}

void MyRunProfilesPanel::SetSelectedCommand(long wanted_command) {

    MyDebugAssertTrue(wanted_command >= 0 && wanted_command < buffer_profile.number_of_run_commands, "Trying to select a command that doesn't exist!");

    CommandsListBox->SetItemState(wanted_command, wxLIST_STATE_SELECTED, wxLIST_STATE_SELECTED);
    selected_command = wanted_command;
    //wxPrintf("set_command = %li\n", selected_command);
}

void MyRunProfilesPanel::OnProfilesFocusChange(wxListEvent& event) {

    if ( event.GetIndex( ) >= 0 ) {
        selected_profile = event.GetIndex( );

        buffer_profile = run_profile_manager.run_profiles[selected_profile];

        if ( run_profile_manager.run_profiles[selected_profile].number_of_run_commands > 0 )
            selected_command = 0;
        else
            selected_command = -1;

        FillCommandsBox( );
        command_panel_has_changed = false;
    }

    event.Skip( );
}

void MyRunProfilesPanel::OnCommandsFocusChange(wxListEvent& event) {

    if ( event.GetIndex( ) >= 0 ) {
        selected_command = event.GetIndex( );
    }

    //wxPrintf("selected_command = %li\n", selected_command);

    event.Skip( );
}

void MyRunProfilesPanel::ManagerTextChanged(wxCommandEvent& event) {
    command_panel_has_changed      = true;
    buffer_profile.manager_command = ManagerTextCtrl->GetValue( );
    if ( ManagerTextCtrl->GetValue( ).Find("$command") == wxNOT_FOUND ) {
        CommandErrorStaticText->SetLabel("Oops! - Command must contain \"$command\"");
    }
    else {
        CommandErrorStaticText->SetLabel("");
    }
    event.Skip( );
}

void MyRunProfilesPanel::AddCommandButtonClick(wxCommandEvent& event) {
    buffer_profile.AddCommand("$command", 1, 1, false, 0, 10);
    FillCommandsBox( );

    SetSelectedCommand(buffer_profile.number_of_run_commands - 1);

    command_panel_has_changed = true;
}

void MyRunProfilesPanel::RemoveCommandButtonClick(wxCommandEvent& event) {
    if ( selected_command != -1 ) {
        MyDebugAssertTrue(selected_command >= 0 && selected_command < buffer_profile.number_of_run_commands, "Trying to remove a command that doesn't exist!");

        buffer_profile.RemoveCommand(selected_command);
        FillCommandsBox( );

        if ( selected_command > 0 )
            SetSelectedCommand(selected_command - 1);
        else if ( selected_command < buffer_profile.number_of_run_commands )
            SetSelectedCommand(selected_command);
        else {
            selected_command = -1;
        }

        command_panel_has_changed = true;
    }
}

void MyRunProfilesPanel::OnCommandsActivated(wxListEvent& event) {
    EditCommand( );
}

void MyRunProfilesPanel::EditCommandButtonClick(wxCommandEvent& event) {
    EditCommand( );
}

void MyRunProfilesPanel::EditCommand( ) {
    MyDebugAssertTrue(selected_command >= 0 && selected_command < buffer_profile.number_of_run_commands, "Trying to edit a command that doesn't exist!");

    MyAddRunCommandDialog* add_dialog = new MyAddRunCommandDialog(this);

    // Set the current values..

    add_dialog->CommandTextCtrl->SetValue(buffer_profile.run_commands[selected_command].command_to_run);
    add_dialog->NumberCopiesSpinCtrl->SetValue(buffer_profile.run_commands[selected_command].number_of_copies);
    add_dialog->DelayTimeSpinCtrl->SetValue(buffer_profile.run_commands[selected_command].delay_time_in_ms);
    add_dialog->NumberThreadsSpinCtrl->SetValue(buffer_profile.run_commands[selected_command].number_of_threads_per_copy);

    if ( buffer_profile.run_commands[selected_command].override_total_copies == true ) {
        add_dialog->OverrideCheckBox->SetValue(true);
        add_dialog->OverridenNoCopiesSpinCtrl->Enable(true);
    }
    else {
        add_dialog->OverrideCheckBox->SetValue(false);
        add_dialog->OverridenNoCopiesSpinCtrl->Enable(false);
    }

    add_dialog->OverridenNoCopiesSpinCtrl->SetValue(buffer_profile.run_commands[selected_command].overriden_number_of_copies);
    add_dialog->ShowModal( );
}

void MyRunProfilesPanel::ImportAllFromDatabase( ) {
    RunProfile temp_profile;

    run_profile_manager.RemoveAllProfiles( );

    main_frame->current_project.database.BeginAllRunProfilesSelect( );

    while ( main_frame->current_project.database.last_return_code == SQLITE_ROW ) {
        temp_profile = main_frame->current_project.database.GetNextRunProfile( );
        run_profile_manager.AddProfile(&temp_profile);
    }

    main_frame->current_project.database.EndAllRunProfilesSelect( );

    if ( run_profile_manager.number_of_run_profiles > 0 )
        selected_profile = 0;
    FillProfilesBox( );
    FillCommandsBox( );
    command_panel_has_changed = false;
    main_frame->DirtyRunProfiles( );
}

void MyRunProfilesPanel::Reset( ) {
    run_profile_manager.RemoveAllProfiles( );

    FillProfilesBox( );
    FillCommandsBox( );

    main_frame->DirtyRunProfiles( );
}
