#ifndef __MyRunProfilesPanel__
#define __MyRunProfilesPanel__

/** Implementing RunProfilesPanel */
class MyRunProfilesPanel : public RunProfilesPanel {
  public:
    RunProfileManager run_profile_manager;
    long              selected_profile;
    long              selected_command;
    bool              command_panel_has_changed;
    RunProfile        buffer_profile;
    bool              is_dirty;

    /** Constructor */
    MyRunProfilesPanel(wxWindow* parent);
    //// end generated class members

    int old_commands_listbox_client_width;

    void FillProfilesBox( );
    void FillCommandsBox( );

    void SizeProfilesColumn( );
    void SizeCommandsColumns( );

    void EditCommand( );

    void OnUpdateUI(wxUpdateUIEvent& event);

    void OnAddProfileClick(wxCommandEvent& event);
    void OnRenameProfileClick(wxCommandEvent& event);
    void OnRemoveProfileClick(wxCommandEvent& event);
    void AddCommandButtonClick(wxCommandEvent& event);
    void EditCommandButtonClick(wxCommandEvent& event);
    void RemoveCommandButtonClick(wxCommandEvent& event);
    void CommandsSaveButtonClick(wxCommandEvent& event);
    void OnImportButtonClick(wxCommandEvent& event);
    void OnExportButtonClick(wxCommandEvent& event);
    void OnDuplicateProfileClick(wxCommandEvent& event);

    void AddDefaultLocalProfile( );

    void GuiAddressAutoClick(wxCommandEvent& event);
    void GuiAddressSpecifyClick(wxCommandEvent& event);
    void ControllerAddressAutoClick(wxCommandEvent& event);
    void ControllerAddressSpecifyClick(wxCommandEvent& event);

    void OnProfilesListItemActivated(wxListEvent& event);

    void VetoInvalidMouse(wxListCtrl* wanted_list, wxMouseEvent& event);
    void OnProfileLeftDown(wxMouseEvent& event);
    void OnProfileDClick(wxMouseEvent& event);

    void OnCommandLeftDown(wxMouseEvent& event);
    void OnCommandDClick(wxMouseEvent& event);

    void MouseVeto(wxMouseEvent& event);

    void ManagerTextChanged(wxCommandEvent& event);

    void OnCommandsActivated(wxListEvent& event);

    void OnEndProfileEdit(wxListEvent& event);
    void OnProfilesFocusChange(wxListEvent& event);

    void OnCommandsFocusChange(wxListEvent& event);

    void SetProfileName(long wanted_group, wxString wanted_name);
    void SetSelectedProfile(long wanted_profile);
    void SetSelectedCommand(long wanted_command);

    void ImportAllFromDatabase( );
    void WriteRunProfilesToDisk(wxString filename, wxArrayInt profiles_to_write);
    bool ImportRunProfilesFromDisk(wxString filename);
    void Reset( );
};

#endif // __MyRunProfilesPanel__
