//#include "../core/core_headers.h"
#include "../core/gui_core_headers.h"
#include "DatabaseUpdateDialog.h"
#include <wx/richmsgdlg.h>

#define SERVER_ID 100
#define SOCKET_ID 101

extern MyAssetsPanel*                    assets_panel;
extern MyMovieAssetPanel*                movie_asset_panel;
extern MyImageAssetPanel*                image_asset_panel;
extern MyParticlePositionAssetPanel*     particle_position_asset_panel;
extern MyVolumeAssetPanel*               volume_asset_panel;
extern AtomicCoordinatesAssetPanel*      atomic_coordinates_asset_panel;
extern TemplateMatchesPackageAssetPanel* template_matches_package_asset_panel;

extern MyRefinementPackageAssetPanel* refinement_package_asset_panel;

extern ActionsPanelParent* actions_panel;

extern MyAlignMoviesPanel*   align_movies_panel;
extern MyFindCTFPanel*       findctf_panel;
extern MyFindParticlesPanel* findparticles_panel;
extern MyRefine2DPanel*      classification_panel;
extern MyRefine3DPanel*      refine_3d_panel;
extern RefineCTFPanel*       refine_ctf_panel;
extern AutoRefine3DPanel*    auto_refine_3d_panel;
extern AbInitio3DPanel*      ab_initio_3d_panel;
extern Generate3DPanel*      generate_3d_panel;
extern Sharpen3DPanel*       sharpen_3d_panel;

extern MySettingsPanel*    settings_panel;
extern MyRunProfilesPanel* run_profiles_panel;

extern MyResultsPanel*            results_panel;
extern MyMovieAlignResultsPanel*  movie_results_panel;
extern MyFindCTFResultsPanel*     ctf_results_panel;
extern MyPickingResultsPanel*     picking_results_panel;
extern MyRefinementResultsPanel*  refinement_results_panel;
extern Refine2DResultsPanel*      refine2d_results_panel;
extern MatchTemplatePanel*        match_template_panel;
extern MatchTemplateResultsPanel* match_template_results_panel;
extern RefineTemplatePanel*       refine_template_panel;

#ifdef EXPERIMENTAL
extern MyExperimentalPanel*    experimental_panel;
extern RefineTemplateDevPanel* refine_template_dev_panel;
#endif

extern MyOverviewPanel* overview_panel;

MyMainFrame::MyMainFrame(wxWindow* parent)
    : MainFrame(parent) {
    //	tree_root = AssetTree->AddRoot("Assets");

    // Add Movies..
    //movie_branch = AssetTree->AppendItem(tree_root, wxString("Movies (0)"));

    // initialize sockets in main thread.

    wxSocketBase::Initialize( );

    is_fullscreen = false;
    SetupServer( );

    // get port and ip addresses..

    my_port             = ReturnServerPort( );
    my_port_string      = ReturnServerPortString( );
    all_my_ip_addresses = ReturnServerAllIpAddresses( );

    int screen_x_size = wxSystemSettings::GetMetric(wxSYS_SCREEN_X);
    int screen_y_size = wxSystemSettings::GetMetric(wxSYS_SCREEN_Y);
    int x_offset;
    int y_offset;

    // We get to do this because the specific register for each workflow is static, so C++ static initialization
    // phase initializes them before the execution of wxApp::OnInit() or main() run
    for ( const auto& name : WorkflowRegistry::Instance( ).GetWorkflowNames( ) ) {
        wxMenuItem* item = new wxMenuItem(WorkflowMenu, wxID_ANY, name, wxEmptyString, wxITEM_RADIO);
        WorkflowMenu->Append(item);
        WorkflowMenu->Bind(wxEVT_MENU, &MyMainFrame::OnWorkflowMenuSelection, this, item->GetId( ));
    }

    // Defaults into Single Particle which can be later updated by the value stored in the
    // database, which does not happen until after panel construction.
    current_workflow = "Single Particle";

    if ( screen_x_size > 1920 && screen_y_size > 1080 ) {
        x_offset = (screen_x_size - 1920) / 2;
        y_offset = (screen_y_size - 1080) / 2;

        if ( x_offset < 0 )
            x_offset = 0;
        if ( y_offset < 0 )
            y_offset = 0;

        SetSize(x_offset, y_offset, 1920, 1080);
    }
    else {
        Maximize(true);
    }

    Bind(wxEVT_CHAR_HOOK, &MyMainFrame::OnCharHook, this);

    // Set brother event handler, this is a nastly little hack so that the socket communicator can use the event handler, and it will work whether the "brother" is a console app or gui panel.

    brother_event_handler = this;
}

MyMainFrame::~MyMainFrame( ) {
    ShutDownServer( );
    ShutDownSocketMonitor( );
    ClearScratchDirectory( );
}

void MyMainFrame::OnWorkflowMenuSelection(wxCommandEvent& event) {
    wxMenuItem* workflow_item = WorkflowMenu->FindItem(event.GetId( ));
    if ( workflow_item ) {
        const wxString name = workflow_item->GetItemLabelText( );
        if ( name != current_workflow ) {
            current_workflow = name;
            SwitchWorkflowPanels(name);
            if ( current_project.database.is_open ) {
                current_project.database.RecordCurrentWorkflowInDB(current_workflow);
            }
        }
    }
}

void MyMainFrame::OnCharHook(wxKeyEvent& event) {

    if ( event.GetKeyCode( ) == WXK_F11 ) {
        if ( is_fullscreen == true ) {
            ShowFullScreen(false);
            is_fullscreen = false;
        }
        else {
            ShowFullScreen(true);
            is_fullscreen = true;
        }
    }

#ifdef DEBUG

    if ( event.GetKeyCode( ) == WXK_F12 ) {
        SetSize(wxDefaultCoord, wxDefaultCoord, 1855, 1025);
    }
#endif

    event.Skip( );
}

void MyMainFrame::RecalculateAssetBrowser(void) {

    /*	wxTreeItemId current_group_branch;

	unsigned long group_counter;
	unsigned long asset_counter;

	long current_group_size;

	bool movies_expanded;

	// need to know the old layout..

	movies_expanded = AssetTree->IsExpanded(movie_branch);


	AssetTree->Freeze();
	AssetTree->DeleteAllItems();

	tree_root = AssetTree->AddRoot("Assets");

	// Add Movies..
	movie_branch = AssetTree->AppendItem(tree_root, wxString("Movies (") + wxString::Format(wxT("%li"), movie_asset_panel->ReturnNumberOfAssets()) + wxString(")"));

		// Movie Groups

		for (group_counter = 0; group_counter < movie_asset_panel->ReturnNumberOfGroups(); group_counter++)
		{
			current_group_size = movie_asset_panel->ReturnGroupSize(group_counter);

			if (current_group_size > 1)
			{
				current_group_branch = AssetTree->AppendItem (movie_branch, movie_asset_panel->ReturnGroupName(group_counter) + wxString(" (")+ wxString::Format(wxT("%li"), current_group_size) + wxString(")"));

				// add the movies..

				for (asset_counter = 0; asset_counter < current_group_size; asset_counter++)
				{
					AssetTree->AppendItem(current_group_branch, movie_asset_panel->ReturnAssetShortFilename(movie_asset_panel->ReturnGroupMember(group_counter, asset_counter)));
				}

			}
		}

	//Images_Branch = main_frame->AssetTree->AppendItem(Tree_Root, "Images (0)");

	// If they were expanded, expand them

	AssetTree->SetItemBold(movie_branch);

	if (movies_expanded == true) AssetTree->Expand(movie_branch);

	AssetTree->Thaw();*/
}

void MyMainFrame::OnCollapseAll(wxCommandEvent& event) {
    //AssetTree->CollapseAll();
}

void MyMainFrame::OnMenuBookChange(wxBookCtrlEvent& event) {
    // redo groups..

    //align_movies_panel->Refresh();
    //	movie_results_panel->group_combo_is_dirty = true;

    // We we were editing the particle picking results, and we move away from Results, we may need to do some database stuff
    if ( event.GetOldSelection( ) == 3 ) {
        picking_results_panel->UpdateResultsFromBitmapPanel( );
    }

#ifdef __WXOSX__
    /*
	   The below is necessary for the MacOS GUI to behave well.
	   We need to make sure the list books (rows of icons) are drawn properly
	   and also the first panel that will be shown.
	   The other panels will be redrawn explicitely later on when the user
	   clicks around.
	 */
    if ( event.GetSelection( ) == 1 ) {
        assets_panel->AssetsBook->Refresh( );
        movie_asset_panel->Layout( );
        movie_asset_panel->Refresh( );
    }
    else if ( event.GetSelection( ) == 2 ) {
        actions_panel->ActionsBook->Refresh( );
        align_movies_panel->Layout( );
        align_movies_panel->Refresh( );
    }
    else if ( event.GetSelection( ) == 3 ) {
        results_panel->ResultsBook->Refresh( );
        movie_results_panel->Layout( );
        movie_results_panel->Refresh( );
    }
    else if ( event.GetSelection( ) == 4 ) {
        settings_panel->SettingsBook->Refresh( );
        run_profiles_panel->Layout( );
        run_profiles_panel->Refresh( );
    }
#ifdef EXPERIMENTAL
    else if ( event.GetSelection( ) == 5 ) {
        experimental_panel->ExperimentalBook->Refresh( );
        //TODO: Layout and Refresh the first Experimental panel
    }
#endif
#endif
}

void MyMainFrame::ResetAllPanels( ) {
    movie_asset_panel->Reset( );
    image_asset_panel->Reset( );
    volume_asset_panel->Reset( );
    atomic_coordinates_asset_panel->Reset( );
    template_matches_package_asset_panel->Reset( );
    particle_position_asset_panel->Reset( );
    refinement_package_asset_panel->Reset( );

    run_profiles_panel->Reset( );
    movie_results_panel->Clear( );
    ctf_results_panel->Clear( );
    picking_results_panel->Clear( );
    refine2d_results_panel->Clear( );
    refinement_results_panel->Clear( );

    align_movies_panel->Reset( );
    findctf_panel->Reset( );
    findparticles_panel->Reset( );
    classification_panel->Reset( );
    ab_initio_3d_panel->Reset( );
    auto_refine_3d_panel->Reset( );
    refine_3d_panel->Reset( );
    refine_ctf_panel->Reset( );
    generate_3d_panel->Reset( );
    sharpen_3d_panel->Reset( );
    match_template_panel->Reset( );
    match_template_results_panel->Clear( );
    refine_template_panel->Reset( );

    DirtyEverything( );
}

void MyMainFrame::DirtyEverything( ) {
    DirtyMovieGroups( );
    DirtyImageGroups( );
    DirtyRunProfiles( );
    DirtyRefinementPackages( );
    DirtyRefinements( );
    DirtyParticlePositionGroups( );
    DirtyClassificationSelections( );
    DirtyClassifications( );
    DirtyVolumes( );
    DirtyAtomicCoordinates( );
}

void MyMainFrame::DirtyVolumes( ) {
    volume_asset_panel->is_dirty            = true;
    refine_3d_panel->volumes_are_dirty      = true;
    auto_refine_3d_panel->volumes_are_dirty = true;
    sharpen_3d_panel->volumes_are_dirty     = true;
    refine_ctf_panel->volumes_are_dirty     = true;

    if ( current_workflow == "Template Matching" ) {
        match_template_panel->volumes_are_dirty = true;
#ifdef EXPERIMENTAL
        refine_template_panel->volumes_are_dirty = true;
#endif
    }
}

void MyMainFrame::DirtyAtomicCoordinates( ) {
    atomic_coordinates_asset_panel->is_dirty = true;
}

void MyMainFrame::DirtyMovieGroups( ) {
    movie_asset_panel->is_dirty               = true;
    align_movies_panel->group_combo_is_dirty  = true;
    movie_results_panel->group_combo_is_dirty = true;
    image_asset_panel->EnableNewFromParentButton( );
}

void MyMainFrame::DirtyImageGroups( ) {
    image_asset_panel->is_dirty                 = true;
    findctf_panel->group_combo_is_dirty         = true;
    ctf_results_panel->group_combo_is_dirty     = true;
    findparticles_panel->group_combo_is_dirty   = true;
    picking_results_panel->group_combo_is_dirty = true;

    if ( current_workflow == "Template Matching" ) {
        match_template_panel->group_combo_is_dirty         = true;
        match_template_results_panel->group_combo_is_dirty = true;
        refine_template_panel->group_combo_is_dirty        = true;
    }
}

void MyMainFrame::DirtyParticlePositionGroups( ) {
    particle_position_asset_panel->is_dirty = true;
}

void MyMainFrame::DirtyRefinementPackages( ) {
    refinement_package_asset_panel->is_dirty                  = true;
    classification_panel->refinement_package_combo_is_dirty   = true;
    refine_3d_panel->refinement_package_combo_is_dirty        = true;
    refine_ctf_panel->refinement_package_combo_is_dirty       = true;
    auto_refine_3d_panel->refinement_package_combo_is_dirty   = true;
    refinement_results_panel->refinement_package_is_dirty     = true;
    refine2d_results_panel->refinement_package_combo_is_dirty = true;
    ab_initio_3d_panel->refinement_package_combo_is_dirty     = true;
    generate_3d_panel->refinement_package_combo_is_dirty      = true;
}

void MyMainFrame::DirtyTemplateMatchesPackages( ) {
    if ( current_workflow == "Template Matching" )
        template_matches_package_asset_panel->is_dirty = true;
}

void MyMainFrame::DirtyRefinements( ) {
    refine_3d_panel->input_params_combo_is_dirty     = true;
    refine_ctf_panel->input_params_combo_is_dirty    = true;
    refinement_results_panel->input_params_are_dirty = true;
    generate_3d_panel->input_params_combo_is_dirty   = true;
}

void MyMainFrame::DirtyClassifications( ) {
    refine2d_results_panel->input_params_combo_is_dirty = true;
}

void MyMainFrame::DirtyClassificationSelections( ) {
    refine2d_results_panel->classification_selections_are_dirty = true;
    ab_initio_3d_panel->classification_selections_are_dirty     = true;
}

void MyMainFrame::DirtyRunProfiles( ) {
    if ( current_workflow == "Single Particle" ) {
        run_profiles_panel->is_dirty                 = true;
        align_movies_panel->run_profiles_are_dirty   = true;
        findctf_panel->run_profiles_are_dirty        = true;
        findparticles_panel->run_profiles_are_dirty  = true;
        classification_panel->run_profiles_are_dirty = true;
        refine_3d_panel->run_profiles_are_dirty      = true;
        refine_ctf_panel->run_profiles_are_dirty     = true;
        auto_refine_3d_panel->run_profiles_are_dirty = true;
        ab_initio_3d_panel->run_profiles_are_dirty   = true;
        generate_3d_panel->run_profiles_are_dirty    = true;
    }
    else if ( current_workflow == "Template Matching" ) {
        align_movies_panel->run_profiles_are_dirty    = true;
        findctf_panel->run_profiles_are_dirty         = true;
        match_template_panel->run_profiles_are_dirty  = true;
        refine_template_panel->run_profiles_are_dirty = true;
    }
}

// SOCKETS

void MyMainFrame::HandleNewSocketConnection(wxSocketBase* new_connection, unsigned char* identification_code) {
    SETUP_SOCKET_CODES

    if ( new_connection != NULL ) {
        // does this correspond to one of our jobs?

        long current_job = job_controller.ReturnJobNumberFromJobCode(identification_code);

        // delete the code..

        delete[] identification_code;

        if ( current_job == -1 ) {
            MyDebugPrint(" GUI : Unknown JOB ID - Closing Connection\n");

            // incorrect identification - close the connection..
            new_connection->Destroy( );
            new_connection = NULL;
        }
        else {
            MyDebugPrint("Connection from Job #%li", current_job);

            job_controller.job_list[current_job].socket = new_connection;

            // let the correct panel look after it from now on..
            job_controller.job_list[current_job].parent_panel->MonitorSocket(new_connection);

            // tell it it is connected

            WriteToSocket(new_connection, socket_you_are_connected, SOCKET_CODE_SIZE, true, "SendSocketJobType", FUNCTION_DETAILS_AS_WXSTRING);
        }
    }
}

void MyMainFrame::OnHelpLaunch(wxCommandEvent& event) {
    wxLaunchDefaultBrowser("http://www.cistem.org/documentation");
}

void MyMainFrame::OnAboutLaunch(wxCommandEvent& event) {
#include "icons/cisTEM_beta_logo_300.cpp"

    wxLogNull* suppress_png_warnings = new wxLogNull;
    wxBitmap   logo_bmp              = wxBITMAP_PNG_FROM_DATA(cisTEM_beta_logo_300);
    delete suppress_png_warnings;

    AboutDialog about_dialog(this);
    about_dialog.LogoBitmap->SetBitmap(logo_bmp);
    about_dialog.VersionStaticText->SetLabel(wxString::Format("cisTEM version %s", CISTEM_VERSION_TEXT));
    about_dialog.BuildDateText->SetLabel(wxString::Format("Built : %s", __DATE__));
    about_dialog.Fit( );
    about_dialog.ShowModal( );
}

void MyMainFrame::OnFileNewProject(wxCommandEvent& event) {
    StartNewProject( );
}

void MyMainFrame::StartNewProject( ) {
    if ( current_project.is_open ) {
        if ( wxMessageBox("The current project must be closed before opening a new project.\n\nClose it now?", "Please confirm", wxICON_QUESTION | wxYES_NO, this) == wxNO )
            return;

        current_project.Close( );
        SetTitle("cisTEM");
    }

    MyNewProjectWizard* my_wizard = new MyNewProjectWizard(this);
    my_wizard->GetPageAreaSizer( )->Add(my_wizard->m_pages.Item(0));
    if ( my_wizard->RunWizard(my_wizard->m_pages.Item(0)) == false )
        return;
    my_wizard->Destroy( );

    if ( current_project.is_open == true ) {
        SetTitle("cisTEM - [" + current_project.project_name + "]");

        // if there is a default run profiles, import it..

        wxString default_run_profile_path = wxStandardPaths::Get( ).GetExecutablePath( );
        default_run_profile_path          = default_run_profile_path.BeforeLast('/');
        default_run_profile_path += "/default_run_profiles.txt";

        if ( DoesFileExist(default_run_profile_path) == true ) {
            //	wxPrintf("Importing run profiles from '%s'\n", default_run_profile_path);
            run_profiles_panel->ImportRunProfilesFromDisk(default_run_profile_path);
        }
        else {
            // there are no default run profiles.. so lets add a default local..

            run_profiles_panel->AddDefaultLocalProfile( );
            //	wxPrintf("no default run profiles (%s)\n", default_run_profile_path);
        }

        AddProjectToRecentProjects(current_project.database.ReturnFilename( ));
        ClearScratchDirectory( );

        overview_panel->SetProjectInfo( );
    }
    else {
        wxMessageBox(wxString::Format("Error Creating database - Does the file already exist?"), "Cannot create database!", wxICON_ERROR);
    }
}

void MyMainFrame::OpenProject(wxString project_filename) {
    // check for the lock file..

    if ( wxDirExists(project_filename + ".lock") ) {
        wxMessageDialog* my_dialog = new wxMessageDialog(this, "There is a lock file for this database, implying another process is writing to it. If multiple processes access the database, it may lead to corruption! A stale lock file can be leftover due to a crashed process, if you are sure that this lock file is stale then select override to delete it and continue.  If not, then select No until you are sure no other process is connected to the database.\n\nYou should backup your database (and journal) before you proceed\n\nDo you want to override?", "Database locked", wxICON_ERROR | wxYES_NO | wxNO_DEFAULT);
        my_dialog->SetYesNoLabels("Override", "No");

        if ( my_dialog->ShowModal( ) != wxID_YES ) {
            my_dialog->Destroy( );
            return;
        }

        // there is a weird bug, whereby when opening a database with a hot journal and dot-file locking, the databse is not rolled back correctly (at least sometimes).
        // as a kind of hack, if the user has said ok to continue, i'm going to open it with no locking (which should roll back the journal) then close it again.

        current_project.database.Open(project_filename, true);
        current_project.database.Close(false);

        // delete lock file

        wxFileName::Rmdir(project_filename + ".lock", wxPATH_RMDIR_RECURSIVE);
    }

    if ( current_project.OpenProjectFromFile(project_filename) == true ) {
        // check this project is not "locked"

        long     my_process_id = wxGetProcessId( );
        wxString my_hostname   = wxGetFullHostName( );

        long     database_process_id;
        wxString database_hostname;

        if ( current_project.database.DoesTableExist("PROCESS_LOCK") == true ) {
            current_project.database.ReturnProcessLockInfo(database_process_id, database_hostname);

            if ( my_process_id != 0 && database_process_id != -1 && my_process_id != database_process_id ) {
                // if we got here then the database is marked as owned..

                wxMessageDialog* my_dialog = new wxMessageDialog(this, wxString::Format("Database is marked as owned by :- \n\nPID : %li\nHost : %s\n\nEach database should only be opened by one instance of cisTEM at a time, otherwise corruption is possible. If it exists, you should close the other instance before continuing, if this message is the result of a crash etc. then you can overide and continue.\n\nDo you want to overide?", database_process_id, database_hostname), "Database already owned", wxICON_ERROR | wxYES_NO | wxNO_DEFAULT);
                my_dialog->SetYesNoLabels("Override", "No");

                if ( my_dialog->ShowModal( ) != wxID_YES ) {
                    my_dialog->Destroy( );
                    current_project.Close(false);
                    return;
                }

                my_dialog->Destroy( );
            }
        }

        // DO DATABASE VERSION CHECK HERE!

        if ( current_project.integer_database_version > INTEGER_DATABASE_VERSION ) {
            wxMessageDialog* my_dialog = new wxMessageDialog(this, "This database was created in a newer version of cisTEM, and cannot be opened.", "Database from newer version", wxICON_ERROR);
            my_dialog->Destroy( );
            current_project.Close(false, false);
            return;
        }
        // We need to see if we want to update
        else if ( current_project.integer_database_version != INTEGER_DATABASE_VERSION || current_project.cistem_version_text != CISTEM_VERSION_TEXT ) {
            auto     schema_comparison = current_project.database.CheckSchema( );
            wxString changes           = "";
            if ( schema_comparison.first.size( ) == 0 && schema_comparison.second.size( ) == 0 ) {
                wxRichMessageDialog* my_dialog = new wxRichMessageDialog(this, wxString::Format("This project was last opened by a different cisTEM version:-\n\nCurrent Version: \t %s\nProject Version: \t %s\n\nHowever, there seem to be no changes to the file format.\n\nAttempt to open the project?,", CISTEM_VERSION_TEXT, current_project.cistem_version_text), "Database from different cisTEM version?", wxICON_ERROR | wxYES_NO | wxNO_DEFAULT);
                if ( my_dialog->ShowModal( ) == wxID_YES )
                    my_dialog->Destroy( );
                else {
                    my_dialog->Destroy( );
                    current_project.Close(false, false);
                    return;
                }
            }
            // database schemas weren't the same; we're going to have to do a manual update
            else {
                // Logic for getting table differences
                for ( auto& table : schema_comparison.first ) {
                    changes += wxString::Format("Add Table: \t %s\n", table);
                }
                for ( auto& column : schema_comparison.second ) {
                    changes += wxString::Format("In Table: \t %s \tadd column: \t %s\n", std::get<0>(column), std::get<1>(column));
                }

                int user_update_type = 0;

                DatabaseUpdateDialog* update_dialog = new DatabaseUpdateDialog(this, changes);
                // Show dialog and get user choice
                user_update_type = update_dialog->ShowModal( );

                // Create enum to avoid 'magic numbers' -- easier to identify at a glance what is happening in each case this way
                enum UpdateOptions { CANCEL            = 1,
                                     UPDATE_ONLY       = 2,
                                     BACKUP_AND_UPDATE = 3 };

                switch ( user_update_type ) {
                    case CANCEL: {
                        current_project.Close(false, false);
                        return;
                    }
                    case UPDATE_ONLY: {
                        UpdateDatabase(schema_comparison);
                        break;
                    }
                    case BACKUP_AND_UPDATE: {
                        // Create backup filename which simply appends _backup
                        std::string tmp_backup_filename = std::string(current_project.database.ReturnFilename( ));
                        tmp_backup_filename.insert(tmp_backup_filename.size( ) - 3, "_backup");
                        wxFileName backup_db_filename(tmp_backup_filename);

                        // Backup and update
                        bool backup_succeeded = current_project.database.CopyDatabaseFile(backup_db_filename);
                        if ( ! backup_succeeded ) {
                            wxPrintf("Failed to backup database properly. Abandoning update attempt.\n");
                            current_project.Close(false, false);
                            return;
                        }
                        UpdateDatabase(schema_comparison);
                        break;
                    }
                    default: {
                        // Shouldn't really get here, so something went wrong, or user closed popup; do same as CANCEL case for safety
                        current_project.Close(false, false);
                        return;
                    }
                }
            }
        }

        // has the database file been moved?  If so, attempt to convert it..

        if ( current_project.database.database_file.GetPath( ) != current_project.project_directory.GetFullPath( ) ) {
            // database has moved?

            wxMessageDialog* my_dialog = new wxMessageDialog(this, wxString::Format("It looks like this project has been moved :-\n\nCurrent Dir. \t: %s\nStored Dir. \t: %s\n\ncisTEM can attempt to migrate the project, updating all paths to point to the current directory. It is wise to make a backup of the database before trying this.\n\nNote : This will only affect paths contained within the project folder, paths to files outside the project folder will remain unchanged.\n\nAttempt to migrate the project?", current_project.database.database_file.GetPath( ), current_project.project_directory.GetFullPath( )), "Database has moved?", wxICON_ERROR | wxYES_NO | wxNO_DEFAULT);
            my_dialog->SetYesNoLabels("Migrate", "No");

            if ( my_dialog->ShowModal( ) != wxID_YES ) {
                my_dialog->Destroy( );
                current_project.Close(false);
                return;
            }
            else {
                // migrate...
                my_dialog->Destroy( );

                if ( MigrateProject(current_project.project_directory.GetFullPath( ), current_project.database.database_file.GetPath( )) == false ) {
                    // something went wrong

                    wxMessageDialog error(this, "Something went wrong!", "Something went wrong!");
                    current_project.Close(false);
                    return;
                }
                else {
                    // close and reopen
                    current_project.Close(false);
                    if ( current_project.OpenProjectFromFile(project_filename) == false )
                        return;
                }
            }
        }

        // if we got here, we can take ownership and carry on..

        current_project.database.SetProcessLockInfo(my_process_id, my_hostname);

        int counter;
        // Note: the second to last arg must be incremented if additional actions are added below.
        OneSecondProgressDialog* my_dialog = new OneSecondProgressDialog("Open Project", "Opening Project", 12, this);

        movie_asset_panel->ImportAllFromDatabase( );
        my_dialog->Update(1, "Opening project (loading image assets...)");
        image_asset_panel->ImportAllFromDatabase( );
        my_dialog->Update(2, "Opening project (loading particle position assets...)");
        particle_position_asset_panel->ImportAllFromDatabase( );
        my_dialog->Update(3, "Opening project (loading run profiles...)");
        run_profiles_panel->ImportAllFromDatabase( );
        my_dialog->Update(4, "Opening project (loading volume assets...)");
        volume_asset_panel->ImportAllFromDatabase( );
        my_dialog->Update(5, "Opening project (loading Refinement Packages...)");
        refinement_package_asset_panel->ImportAllFromDatabase( );
        //align_movies_panel->Refresh();
        my_dialog->Update(6, "Opening project (loading movie alignment results...)");
        movie_results_panel->FillBasedOnSelectCommand("SELECT DISTINCT MOVIE_ASSET_ID FROM MOVIE_ALIGNMENT_LIST");
        my_dialog->Update(7, "Opening project (loading CTF estimation results...)");
        //	current_project.database.AddCTFIcinessColumnIfNecessary();
        ctf_results_panel->FillBasedOnSelectCommand("SELECT DISTINCT IMAGE_ASSET_ID FROM ESTIMATED_CTF_PARAMETERS");
        //	current_project.database.AddCTFIcinessColumnIfNecessary();
        my_dialog->Update(8, "Opening project (loading Match Template Results...)");

        if ( current_workflow == "Template Matching" ) {
            match_template_results_panel->FillBasedOnSelectCommand("SELECT DISTINCT IMAGE_ASSET_ID FROM TEMPLATE_MATCH_LIST");
            my_dialog->Update(9, "Opening project (loading atomic coordinates assets...)");
            atomic_coordinates_asset_panel->ImportAllFromDatabase( );
            my_dialog->Update(10, "Opening project (loading Template Matches Packages...)");
            template_matches_package_asset_panel->ImportAllFromDatabase( );
        }

        my_dialog->Update(11, "Opening project (finishing...)");
        picking_results_panel->OnProjectOpen( );
        my_dialog->Update(12, "Opening project (all done)");

        SetTitle("cisTEM - [" + current_project.project_name + "]");
        DirtyEverything( );
        my_dialog->Destroy( );

        AddProjectToRecentProjects(project_filename);
        ClearScratchDirectory( );
        overview_panel->SetProjectInfo( );

        // This should never execute, but if somehow current_workflow is empty, log that it was and set a default.
        if ( current_workflow.IsEmpty( ) ) {
            wxLogWarning("The current workflow could not be found; returning Single Particle as default to prevent errors.");
            current_workflow = "Single Particle";
        }

        if ( current_project.database.is_open ) {
            // Need to check if there is already a value recorded first.
            // Know database is open so check what is currently recorded
            if ( current_project.database.CheckIfCurrentWorkflowIsInteger( ) ) {
                const int workflow_val = current_project.database.ReturnSingleIntFromSelectCommand("select CURRENT_WORKFLOW from MASTER_SETTINGS");

                // When using enum logic, these were the values used for setting the respective workflows
                constexpr int old_spa_val = 0;
                constexpr int old_tm_val  = 1;
                if ( workflow_val == old_spa_val ) {
                    current_workflow = "Single Particle";
                }
                else if ( workflow_val == old_tm_val ) {
                    current_workflow = "Template Matching";
                }
            }
            else {
                current_workflow = current_project.database.ReturnSingleStringFromSelectCommand("select CURRENT_WORKFLOW from MASTER_SETTINGS");
            }
            current_project.database.RecordCurrentWorkflowInDB(current_workflow);
            SwitchWorkflowPanels(current_workflow);
            ManuallyUpdateWorkflowMenuCheckBox( );
        }
    }
    else {
        wxMessageBox(wxString::Format("Error Opening database :- \n%s\n\nDoes the file exist?", project_filename), "Cannot open database!", wxICON_ERROR);
        MyPrintWithDetails("An error occured opening the database file..");
    }
}

void MyMainFrame::GetFileAndOpenProject( ) {
    // find a DB file..
    if ( current_project.is_open ) {
        if ( wxMessageBox("The current project must be closed before opening a new project.\n\nClose it now?", "Please confirm", wxICON_QUESTION | wxYES_NO, this) == wxNO )
            return;
        current_project.Close( );
        SetTitle("cisTEM");
    }

    wxFileDialog openFileDialog(this, _("Open db file"), "", "", "DB files (*.db)|*.db", wxFD_OPEN | wxFD_FILE_MUST_EXIST);

    if ( openFileDialog.ShowModal( ) == wxID_CANCEL )
        return;
    OpenProject(openFileDialog.GetPath( ));
    overview_panel->SetProjectInfo( );
}

void MyMainFrame::OnFileOpenProject(wxCommandEvent& event) {
    GetFileAndOpenProject( );
}

void MyMainFrame::OnFileExit(wxCommandEvent& event) {
    Close(true);
}

void MyMainFrame::OnFileCloseProject(wxCommandEvent& event) {
    picking_results_panel->OnProjectClose( );

    current_project.Close( );

    ResetAllPanels( );

    SetTitle("cisTEM");
    MenuBook->SetSelection(0);
    overview_panel->SetWelcomeInfo( );
    overview_panel->InfoText->Show(true);
    ClearScratchDirectory( );
}

void MyMainFrame::OnFileMenuUpdate(wxUpdateUIEvent& event) {
    if ( current_project.is_open == true ) {
        FileMenu->FindItem(FileMenu->FindItem("Close Project"))->Enable(true);
        //		ExportMenu->FindItem(ExportMenu->FindItem("Export coordinates to Imagic"))->Enable(true);
        //		ExportMenu->FindItem(ExportMenu->FindItem("Export particles to Frealign"))->Enable(true);
        //		ExportMenu->FindItem(ExportMenu->FindItem("Export particles to Relion"))->Enable(true);
    }
    else {
        FileMenu->FindItem(FileMenu->FindItem("Close Project"))->Enable(false);
        //		ExportMenu->FindItem(ExportMenu->FindItem("Export coordinates to Imagic"))->Enable(false);
        //		ExportMenu->FindItem(ExportMenu->FindItem("Export particles to Frealign"))->Enable(false);
        //		ExportMenu->FindItem(ExportMenu->FindItem("Export particles to Relion"))->Enable(false);
    }
}

void MyMainFrame::OnExportCoordinatesToImagic(wxCommandEvent& event) {
    MyParticlePositionExportDialog* export_dialog = new MyParticlePositionExportDialog(this);
    export_dialog->ShowModal( );
}

void MyMainFrame::OnExportToFrealign(wxCommandEvent& event) {
    MyFrealignExportDialog* export_dialog = new MyFrealignExportDialog(this);
    export_dialog->ShowModal( );
}

void MyMainFrame::OnExportToRelion(wxCommandEvent& event) {
    MyRelionExportDialog* export_dialog = new MyRelionExportDialog(this);
    export_dialog->ShowModal( );
}

void MyMainFrame::ClearScratchDirectory( ) {
    ClearStartupScratch( );
    ClearRefine2DScratch( );
    ClearRefine3DScratch( );
    ClearAutoRefine3DScratch( );
    ClearGenerate3DScratch( );
    ClearRefineCTFScratch( );
}

void MyMainFrame::ClearRefineCTFScratch( ) {
    if ( wxDir::Exists(ReturnRefineCTFScratchDirectory( )) == true )
        wxFileName::Rmdir(ReturnRefineCTFScratchDirectory( ), wxPATH_RMDIR_RECURSIVE);
    if ( wxDir::Exists(ReturnRefineCTFScratchDirectory( )) == false )
        wxFileName::Mkdir(ReturnRefineCTFScratchDirectory( ));
}

void MyMainFrame::ClearStartupScratch( ) {
    if ( wxDir::Exists(ReturnStartupScratchDirectory( )) == true )
        wxFileName::Rmdir(ReturnStartupScratchDirectory( ), wxPATH_RMDIR_RECURSIVE);
    if ( wxDir::Exists(ReturnStartupScratchDirectory( )) == false )
        wxFileName::Mkdir(ReturnStartupScratchDirectory( ));
}

void MyMainFrame::ClearRefine2DScratch( ) {
    if ( wxDir::Exists(ReturnRefine2DScratchDirectory( )) == true )
        wxFileName::Rmdir(ReturnRefine2DScratchDirectory( ), wxPATH_RMDIR_RECURSIVE);
    if ( wxDir::Exists(ReturnRefine2DScratchDirectory( )) == false )
        wxFileName::Mkdir(ReturnRefine2DScratchDirectory( ));
}

void MyMainFrame::ClearRefine3DScratch( ) {
    if ( wxDir::Exists(ReturnRefine3DScratchDirectory( )) == true )
        wxFileName::Rmdir(ReturnRefine3DScratchDirectory( ), wxPATH_RMDIR_RECURSIVE);
    if ( wxDir::Exists(ReturnRefine3DScratchDirectory( )) == false )
        wxFileName::Mkdir(ReturnRefine3DScratchDirectory( ));
}

void MyMainFrame::ClearAutoRefine3DScratch( ) {
    if ( wxDir::Exists(ReturnAutoRefine3DScratchDirectory( )) == true )
        wxFileName::Rmdir(ReturnAutoRefine3DScratchDirectory( ), wxPATH_RMDIR_RECURSIVE);
    if ( wxDir::Exists(ReturnAutoRefine3DScratchDirectory( )) == false )
        wxFileName::Mkdir(ReturnAutoRefine3DScratchDirectory( ));
}

void MyMainFrame::ClearGenerate3DScratch( ) {
    if ( wxDir::Exists(ReturnGenerate3DScratchDirectory( )) == true )
        wxFileName::Rmdir(ReturnGenerate3DScratchDirectory( ), wxPATH_RMDIR_RECURSIVE);
    if ( wxDir::Exists(ReturnGenerate3DScratchDirectory( )) == false )
        wxFileName::Mkdir(ReturnGenerate3DScratchDirectory( ));
}

wxString MyMainFrame::ReturnScratchDirectory( ) {
    return current_project.scratch_directory.GetFullPath( );
}

wxString MyMainFrame::ReturnStartupScratchDirectory( ) {
    return current_project.scratch_directory.GetFullPath( ) + "/Startup/";
}

wxString MyMainFrame::ReturnRefine2DScratchDirectory( ) {
    return current_project.scratch_directory.GetFullPath( ) + "/Refine2D/";
}

wxString MyMainFrame::ReturnRefine3DScratchDirectory( ) {
    return current_project.scratch_directory.GetFullPath( ) + "/ManualRefine3D/";
}

wxString MyMainFrame::ReturnAutoRefine3DScratchDirectory( ) {
    return current_project.scratch_directory.GetFullPath( ) + "/AutoRefine3D/";
}

wxString MyMainFrame::ReturnGenerate3DScratchDirectory( ) {
    return current_project.scratch_directory.GetFullPath( ) + "/Generate3D/";
}

wxString MyMainFrame::ReturnRefineCTFScratchDirectory( ) {
    return current_project.scratch_directory.GetFullPath( ) + "/RefineCTF/";
}

bool MyMainFrame::MigrateProject(wxString old_project_directory, wxString new_project_directory) {
    // this is very boring.. go through and update all the links in the database..
    // start transaction

    current_project.database.Begin( );

    // Master settings..
    current_project.database.ExecuteSQL(wxString::Format("UPDATE MASTER_SETTINGS SET PROJECT_DIRECTORY = '%s';", new_project_directory).ToUTF8( ).data( ));

    // Movie Assets

    current_project.database.ExecuteSQL(wxString::Format("UPDATE MOVIE_ASSETS SET FILENAME = REPLACE(FILENAME, '%s', '%s');", old_project_directory, new_project_directory).ToUTF8( ).data( ));

    // Image Assets

    current_project.database.ExecuteSQL(wxString::Format("UPDATE IMAGE_ASSETS SET FILENAME = REPLACE(FILENAME, '%s', '%s');", old_project_directory, new_project_directory).ToUTF8( ).data( ));

    // Volume Assets

    current_project.database.ExecuteSQL(wxString::Format("UPDATE VOLUME_ASSETS SET FILENAME = REPLACE(FILENAME, '%s', '%s');", old_project_directory, new_project_directory).ToUTF8( ).data( ));

    // Refinement Package Assets

    current_project.database.ExecuteSQL(wxString::Format("UPDATE REFINEMENT_PACKAGE_ASSETS SET STACK_FILENAME = REPLACE(STACK_FILENAME, '%s', '%s');", old_project_directory, new_project_directory).ToUTF8( ).data( ));

    // Movie alignment list

    current_project.database.ExecuteSQL(wxString::Format("UPDATE MOVIE_ALIGNMENT_LIST SET OUTPUT_FILE = REPLACE(OUTPUT_FILE, '%s', '%s');", old_project_directory, new_project_directory).ToUTF8( ).data( ));

    // Estimated CTF Parameters

    current_project.database.ExecuteSQL(wxString::Format("UPDATE ESTIMATED_CTF_PARAMETERS SET OUTPUT_DIAGNOSTIC_FILE = REPLACE(OUTPUT_DIAGNOSTIC_FILE, '%s', '%s');", old_project_directory, new_project_directory).ToUTF8( ).data( ));

    // Classification List

    current_project.database.ExecuteSQL(wxString::Format("UPDATE CLASSIFICATION_LIST SET CLASS_AVERAGE_FILE = REPLACE(CLASS_AVERAGE_FILE, '%s', '%s');", old_project_directory, new_project_directory).ToUTF8( ).data( ));

    // Commit

    // Template Matching...

    current_project.database.ExecuteSQL(wxString::Format("UPDATE TEMPLATE_MATCH_LIST SET MIP_OUTPUT_FILE = REPLACE(MIP_OUTPUT_FILE, '%s', '%s');", old_project_directory, new_project_directory).ToUTF8( ).data( ));
    current_project.database.ExecuteSQL(wxString::Format("UPDATE TEMPLATE_MATCH_LIST SET SCALED_MIP_OUTPUT_FILE = REPLACE(SCALED_MIP_OUTPUT_FILE, '%s', '%s');", old_project_directory, new_project_directory).ToUTF8( ).data( ));
    current_project.database.ExecuteSQL(wxString::Format("UPDATE TEMPLATE_MATCH_LIST SET PSI_OUTPUT_FILE = REPLACE(PSI_OUTPUT_FILE, '%s', '%s');", old_project_directory, new_project_directory).ToUTF8( ).data( ));
    current_project.database.ExecuteSQL(wxString::Format("UPDATE TEMPLATE_MATCH_LIST SET THETA_OUTPUT_FILE = REPLACE(THETA_OUTPUT_FILE, '%s', '%s');", old_project_directory, new_project_directory).ToUTF8( ).data( ));
    current_project.database.ExecuteSQL(wxString::Format("UPDATE TEMPLATE_MATCH_LIST SET PHI_OUTPUT_FILE = REPLACE(PHI_OUTPUT_FILE, '%s', '%s');", old_project_directory, new_project_directory).ToUTF8( ).data( ));
    current_project.database.ExecuteSQL(wxString::Format("UPDATE TEMPLATE_MATCH_LIST SET DEFOCUS_OUTPUT_FILE = REPLACE(DEFOCUS_OUTPUT_FILE, '%s', '%s');", old_project_directory, new_project_directory).ToUTF8( ).data( ));
    current_project.database.ExecuteSQL(wxString::Format("UPDATE TEMPLATE_MATCH_LIST SET PIXEL_SIZE_OUTPUT_FILE = REPLACE(PIXEL_SIZE_OUTPUT_FILE, '%s', '%s');", old_project_directory, new_project_directory).ToUTF8( ).data( ));
    current_project.database.ExecuteSQL(wxString::Format("UPDATE TEMPLATE_MATCH_LIST SET HISTOGRAM_OUTPUT_FILE = REPLACE(HISTOGRAM_OUTPUT_FILE, '%s', '%s');", old_project_directory, new_project_directory).ToUTF8( ).data( ));
    current_project.database.ExecuteSQL(wxString::Format("UPDATE TEMPLATE_MATCH_LIST SET PROJECTION_RESULT_OUTPUT_FILE = REPLACE(PROJECTION_RESULT_OUTPUT_FILE, '%s', '%s');", old_project_directory, new_project_directory).ToUTF8( ).data( ));

    current_project.database.Commit( );

    // everything should be ok?

    return true;
}

void MyMainFrame::SwitchWorkflowPanels(const wxString& workflow_name) {
    Freeze( );

    int current_page_idx  = MenuBook->GetSelection( );
    int actions_panel_idx = MenuBook->FindPage(actions_panel);
    MenuBook->RemovePage(actions_panel_idx);

    if ( actions_panel ) {
        actions_panel->Destroy( );
        actions_panel = nullptr;
    }

    actions_panel = static_cast<ActionsPanelParent*>(WorkflowRegistry::Instance( ).CreateActionsPanel(workflow_name, this->MenuBook));
    this->MenuBook->InsertPage(actions_panel_idx, actions_panel, "Actions", false, actions_panel_idx);
    this->MenuBook->SetSelection(current_page_idx);

    // TODO: Repeat above logic for any panels that are different between workflows

    actions_panel->Layout( );
    MenuBook->Layout( );
    Layout( );
    Thaw( );
}

void MyMainFrame::UpdateDatabase(std::pair<Database::TableChanges, Database::ColumnChanges>& schema_comparison) {
    // 1. Estimate number of rows that will be updated
    // Total number of particles can be very large; on the order of billions.
    // Use long to ensure enough space.
    long row_updates = 0;
    int  normalized_increments; // This will adjust how long it takes for an update based on the total number of particles that will be processed

    // Limit scope of temporarily needed arrays while estimating total number of increments...
    {
        wxArrayInt ref_pkg_ids = current_project.database.ReturnIntArrayFromSelectCommand("select REFINEMENT_PACKAGE_ASSET_ID from REFINEMENT_PACKAGE_ASSETS");
        for ( int cur_pkg_id : ref_pkg_ids ) {
            row_updates += current_project.database.ReturnSingleIntFromSelectCommand(wxString::Format("select COUNT(*) from REFINEMENT_PACKAGE_CONTAINED_PARTICLES_%i", cur_pkg_id));
        }
        wxArrayString refinement_result_tables = current_project.database.ReturnStringArrayFromSelectCommand("SELECT name FROM sqlite_master WHERE type='table' AND name LIKE 'REFINEMENT_RESULT_%_%'");
        for ( wxString table_name : refinement_result_tables ) {
            row_updates += current_project.database.ReturnSingleIntFromSelectCommand(wxString::Format("SELECT COUNT(*) FROM '%s'", table_name));
        }
        wxArrayString classification_result_tables = current_project.database.ReturnStringArrayFromSelectCommand("SELECT name FROM sqlite_master WHERE type='table' AND name LIKE 'CLASSIFICATION_RESULT_%'");
        for ( wxString table_name : classification_result_tables ) {
            row_updates += current_project.database.ReturnSingleIntFromSelectCommand(wxString::Format("SELECT COUNT(*) FROM '%s'", table_name));
        }
    } // End total increments estimation

    row_updates += schema_comparison.second.size( );

    // Adjusting the normalized increments based on total increments helps prevent GUI from freezing up and gives more accurate
    // estimated time remaining. This may need additional fine tuning.
    {
        long tmp_row_updates = row_updates;
        int  place_value     = 1;

        // Get highest place value
        while ( tmp_row_updates >= 10 ) {
            tmp_row_updates /= 10;
            place_value *= 10;
        }

        // Determine how many progress bar increments there should be based on largest place value
        if ( place_value <= 100000 ) {
            normalized_increments = 100;
        }
        else if ( place_value > 100000 && place_value < 10000000 ) {
            normalized_increments = 1000;
        }
        // 10 million or greater
        else {
            normalized_increments = 10000;
        }
    } // End progress normalization

    // 2. Instantiate progress dialog member variable so it can receive updates
    update_progress_dialog = new OneSecondProgressDialog("Updating Database...", "Starting database update...", normalized_increments, this, wxPD_APP_MODAL | wxPD_REMAINING_TIME | wxPD_AUTO_HIDE | wxPD_SMOOTH);

    // 3. Update, passing MainFrame's UpdateProgressTracker variables
    current_project.database.UpdateSchema(schema_comparison.second, this, row_updates, normalized_increments);
    OnCompletion( );
}

////////////////////////////////////
/// TRACK DATABASE SCHEMA UPDATE
////////////////////////////////////
void MyMainFrame::OnUpdateProgress(int progress, wxString new_msg, bool& should_update_text) {
    if ( ! should_update_text ) {
        update_progress_dialog->Update(progress);
    }
    else {
        update_progress_dialog->Update(progress, new_msg);
        should_update_text = false;
    }
}

void MyMainFrame::OnCompletion( ) {
    if ( update_progress_dialog ) {
        update_progress_dialog->Destroy( );
        update_progress_dialog = nullptr;
    }
}

////////////////////////////////////