//#include "../core/core_headers.h"
#include "../core/gui_core_headers.h"

extern MyImageAssetPanel* image_asset_panel;

MyParticlePositionAssetPanel::MyParticlePositionAssetPanel(wxWindow* parent)
    : MyAssetPanelParent(parent) {
    RenameAssetButton->Show(false);
    DisplayButton->Show(false);
    DisplayButton->Enable(false);
    Layout( );

    Label0Title->SetLabel("");
    Label1Title->SetLabel("I.D. : ");
    Label2Title->SetLabel("Parent Image I.D. : ");
    Label3Title->SetLabel("Pick Job I.D. : ");
    Label4Title->SetLabel("X Position : ");
    Label5Title->SetLabel("Y Position : ");
    Label6Title->SetLabel("");
    Label7Title->SetLabel("");
    Label8Title->SetLabel("");
    Label9Title->SetLabel("");

    UpdateInfo( );

    NewFromParentButton->SetLabel("New from\nimage group");
    EnableNewFromParentButton( );

    AssetTypeText->SetLabel("Particle Positions");

    all_groups_list->groups[0].SetName("All Positions");
    all_assets_list = new ParticlePositionAssetList;
    FillGroupList( );
    FillContentsList( );
}

void MyParticlePositionAssetPanel::EnableNewFromParentButton( ) {
    NewFromParentButton->Enable(image_asset_panel->ReturnNumberOfGroups( ) > 1);
}

MyParticlePositionAssetPanel::~MyParticlePositionAssetPanel( ) {
    delete all_assets_list;
}

void MyParticlePositionAssetPanel::UpdateInfo( ) {
    if ( selected_content >= 0 && selected_group >= 0 && all_groups_list->groups[selected_group].number_of_members > 0 ) {
        Label0Text->SetLabel("");
        Label1Text->SetLabel(wxString::Format(wxT("%i"), all_assets_list->ReturnParticlePositionAssetPointer(all_groups_list->ReturnGroupMember(selected_group, selected_content))->asset_id));
        Label2Text->SetLabel(wxString::Format(wxT("%i"), all_assets_list->ReturnParticlePositionAssetPointer(all_groups_list->ReturnGroupMember(selected_group, selected_content))->parent_id));
        Label3Text->SetLabel(wxString::Format(wxT("%i"), all_assets_list->ReturnParticlePositionAssetPointer(all_groups_list->ReturnGroupMember(selected_group, selected_content))->pick_job_id));
        Label4Text->SetLabel(wxString::Format(wxT("%.2f"), all_assets_list->ReturnParticlePositionAssetPointer(all_groups_list->ReturnGroupMember(selected_group, selected_content))->x_position));
        Label5Text->SetLabel(wxString::Format(wxT("%.2f"), all_assets_list->ReturnParticlePositionAssetPointer(all_groups_list->ReturnGroupMember(selected_group, selected_content))->y_position));
        Label6Text->SetLabel("");
        Label7Text->SetLabel("");
        Label8Text->SetLabel("");
        Label9Text->SetLabel("");
    }
    else {
        Label0Text->SetLabel("");
        Label1Text->SetLabel("-");
        Label2Text->SetLabel("-");
        Label3Text->SetLabel("-");
        Label4Text->SetLabel("-");
        Label5Text->SetLabel("-");
        Label6Text->SetLabel("");
        Label7Text->SetLabel("");
        Label8Text->SetLabel("");
        Label9Text->SetLabel("");
    }
}

ParticlePositionAsset* MyParticlePositionAssetPanel::ReturnAssetPointer(long wanted_asset) {
    return all_assets_list->ReturnParticlePositionAssetPointer(wanted_asset);
}

int MyParticlePositionAssetPanel::ShowDeleteMessageDialog( ) {
    wxMessageDialog check_dialog(this, "This will remove the selected particle positions from your ENTIRE project!  It will also remove them from the relevant particle picking results.\n\nIn general, if the positions were created from within cisTEM (i.e. they weren't imported) you probably don't want to do this, you probably want to remove them using the particle picking results panel. Any changes you make here\n\nAre you sure you want to continue?", "Are you sure?", wxYES_NO | wxICON_WARNING);
    return check_dialog.ShowModal( );
}

int MyParticlePositionAssetPanel::ShowDeleteAllMessageDialog( ) {
    wxMessageDialog check_dialog(this, "This will remove the selected particle positions from your ENTIRE project!  It will also remove them from the relevant particle picking results.\n\nIn general, if the positions were created from within cisTEM (i.e. they weren't imported) you probably don't want to do this, you probably want to remove them using the particle picking results panel.\n\nAre you sure you want to continue?", "Are you sure?", wxYES_NO | wxICON_WARNING);
    return check_dialog.ShowModal( );
}

void MyParticlePositionAssetPanel::CompletelyRemoveAsset(long wanted_asset) {
    long counter;
    long found_position;

    // first off remove it from the asset list..

    RemoveAssetFromDatabase(wanted_asset);
    RemoveAssetFromGroups(wanted_asset);
}

void MyParticlePositionAssetPanel::DoAfterDeletionCleanup( ) {
}

void MyParticlePositionAssetPanel::RemoveAssetFromDatabase(long wanted_asset) {
    main_frame->current_project.database.ExecuteSQL(wxString::Format("DELETE FROM PARTICLE_POSITION_ASSETS WHERE PARTICLE_POSITION_ASSET_ID=%i", all_assets_list->ReturnAssetID(wanted_asset)).ToUTF8( ).data( ));
    all_assets_list->RemoveAsset(wanted_asset);
}

void MyParticlePositionAssetPanel::RemoveFromGroupInDatabase(int wanted_group_id, int wanted_asset_id) {
    main_frame->current_project.database.ExecuteSQL(wxString::Format("DELETE FROM PARTICLE_POSITION_GROUP_%i WHERE PARTICLE_POSITION_ASSET_ID=%i", wanted_group_id, wanted_asset_id).ToUTF8( ).data( ));
}

void MyParticlePositionAssetPanel::InsertGroupMemberToDatabase(int wanted_group, int wanted_asset) {
    MyDebugAssertTrue(wanted_group > 0 && wanted_group < all_groups_list->number_of_groups, "Requesting a group (%i) that doesn't exist!", wanted_group);
    MyDebugAssertTrue(wanted_asset >= 0 && wanted_asset < all_assets_list->number_of_assets, "Requesting an partice position (%i) that doesn't exist!", wanted_asset);

    main_frame->current_project.database.InsertOrReplace(wxString::Format("PARTICLE_POSITION_GROUP_%i", ReturnGroupID(wanted_group)).ToUTF8( ).data( ), "ii", "MEMBER_NUMBER", "PARTICLE_POSITION_ASSET_ID", ReturnGroupSize(wanted_group), ReturnGroupMemberID(wanted_group, wanted_asset));
}

void MyParticlePositionAssetPanel::InsertArrayofGroupMembersToDatabase(long wanted_group, wxArrayLong* wanted_array, OneSecondProgressDialog* progress_dialog) {
    MyDebugAssertTrue(wanted_group > 0 && wanted_group < all_groups_list->number_of_groups, "Requesting a group (%i) that doesn't exist!", wanted_group);

    int current_member_number = main_frame->current_project.database.ReturnSingleIntFromSelectCommand(wxString::Format("SELECT MAX(MEMBER_NUMBER) FROM PARTICLE_POSITION_GROUP_%i", ReturnGroupID(wanted_group)).ToUTF8( ).data( ));
    main_frame->current_project.database.BeginBatchInsert(wxString::Format("PARTICLE_POSITION_GROUP_%i", ReturnGroupID(wanted_group)).ToUTF8( ).data( ), 2, "MEMBER_NUMBER", "PARTICLE_POSITION_ASSET_ID");

    for ( long counter = 0; counter < wanted_array->GetCount( ); counter++ ) {
        MyDebugAssertTrue(wanted_array->Item(counter) >= 0 && wanted_array->Item(counter) < all_assets_list->number_of_assets, "Requesting an asset(%li) that doesn't exist!", wanted_array->Item(counter));
        main_frame->current_project.database.AddToBatchInsert("ii", current_member_number, ReturnGroupMemberID(0, int(wanted_array->Item(counter))));
        current_member_number++;
        if ( progress_dialog != NULL )
            progress_dialog->Update(counter);
    }

    main_frame->current_project.database.EndBatchInsert( );
}

void MyParticlePositionAssetPanel::RemoveAllFromDatabase( ) {
    main_frame->current_project.database.Begin( );
    for ( long counter = 1; counter < all_groups_list->number_of_groups; counter++ ) {
        main_frame->current_project.database.ExecuteSQL(wxString::Format("DROP TABLE PARTICLE_POSITION_GROUP_%i", all_groups_list->groups[counter].id).ToUTF8( ).data( ));
    }

    main_frame->current_project.database.ExecuteSQL("DROP TABLE PARTICLE_POSITION_GROUP_LIST");
    main_frame->current_project.database.ExecuteSQL("DROP TABLE PARTICLE_POSITION_ASSETS");
    main_frame->current_project.database.CreateAllTables( );
    main_frame->current_project.database.Commit( );
}

void MyParticlePositionAssetPanel::RemoveAllGroupMembersFromDatabase(int wanted_group_id) {
    main_frame->current_project.database.Begin( );
    main_frame->current_project.database.ExecuteSQL(wxString::Format("DROP TABLE PARTICLE_POSITION_GROUP_%i", wanted_group_id).ToUTF8( ).data( ));
    main_frame->current_project.database.CreateTable(wxString::Format("PARTICLE_POSITION_GROUP_%i", wanted_group_id).ToUTF8( ).data( ), "ii", "MEMBER_NUMBER", "PARTICLE_POSITION_ASSET_ID");
    main_frame->current_project.database.Commit( );
}

void MyParticlePositionAssetPanel::AddGroupToDatabase(int wanted_group_id, const char* wanted_group_name, int wanted_list_id) {
    main_frame->current_project.database.Begin( );
    main_frame->current_project.database.InsertOrReplace("PARTICLE_POSITION_GROUP_LIST", "iti", "GROUP_ID", "GROUP_NAME", "LIST_ID", wanted_group_id, wanted_group_name, wanted_list_id);
    main_frame->current_project.database.CreateTable(wxString::Format("PARTICLE_POSITION_GROUP_%i", wanted_list_id).ToUTF8( ).data( ), "ii", "MEMBER_NUMBER", "PARTICLE_POSITION_ASSET_ID");
    main_frame->current_project.database.Commit( );
}

void MyParticlePositionAssetPanel::RemoveGroupFromDatabase(int wanted_group_id) {
    main_frame->current_project.database.Begin( );
    main_frame->current_project.database.ExecuteSQL(wxString::Format("DROP TABLE PARTICLE_POSITION_GROUP_%i", wanted_group_id).ToUTF8( ).data( ));
    main_frame->current_project.database.ExecuteSQL(wxString::Format("DELETE FROM PARTICLE_POSITION_GROUP_LIST WHERE GROUP_ID=%i", wanted_group_id));
    main_frame->current_project.database.Commit( );
}

void MyParticlePositionAssetPanel::RenameGroupInDatabase(int wanted_group_id, const char* wanted_name) {
    wxString name = wanted_name;
    name.Replace("'", "''");
    wxString sql_command = wxString::Format("UPDATE PARTICLE_POSITION_GROUP_LIST SET GROUP_NAME='%s' WHERE GROUP_ID=%i", name, wanted_group_id);
    main_frame->current_project.database.ExecuteSQL(sql_command.ToUTF8( ).data( ));
}

void MyParticlePositionAssetPanel::RemoveParticlePositionAssetsWithGivenParentImageID(long parent_image_id) {

    ContentsListBox->Freeze( );

    main_frame->current_project.database.RemoveParticlePositionAssetsPickedFromImageWithGivenID(parent_image_id);

    // Remove assets from groups (this part will probably be slow when there are many and/or large groups

    ParticlePositionAsset* current_asset;

    for ( long asset_counter = all_assets_list->number_of_assets - 1; asset_counter >= 0; asset_counter-- ) {
        current_asset = ReturnAssetPointer(asset_counter);

        //wxPrintf("Asset counter = %li ; Checking whether we need to remove asset with id %i (parent id %i)\n",asset_counter,current_asset->asset_id,current_asset->parent_id);
        if ( current_asset->parent_id == parent_image_id ) {
            //wxPrintf("Removing asset with id %i (parent id %i) (position %li)\n", current_asset->asset_id, current_asset->parent_id, asset_counter);
            //all_assets_list->RemoveAsset(asset_counter);
            RemoveAssetFromGroups(current_asset->asset_id);
        }
    }

    // Remove assets from the "master" asset list
    reinterpret_cast<ParticlePositionAssetList*>(all_assets_list)->RemoveAssetsWithGivenParentImageID(parent_image_id);

    FillContentsList( );

    ContentsListBox->Thaw( );
}

void MyParticlePositionAssetPanel::ImportAllFromDatabase( ) {

    int                   counter;
    ParticlePositionAsset temp_asset;
    AssetGroup            temp_group;

    all_assets_list->RemoveAll( );
    all_groups_list->RemoveAll( );

    // First all assets..

    main_frame->current_project.database.BeginAllParticlePositionAssetsSelect( );

    while ( main_frame->current_project.database.last_return_code == SQLITE_ROW ) {
        temp_asset = main_frame->current_project.database.GetNextParticlePositionAsset( );
        AddAsset(&temp_asset);
    }

    main_frame->current_project.database.EndAllParticlePositionAssetsSelect( );

    // Now the groups..

    int last_found_position = 0;

    main_frame->current_project.database.BeginAllParticlePositionGroupsSelect( );

    while ( main_frame->current_project.database.last_return_code == SQLITE_ROW ) {
        temp_group = main_frame->current_project.database.GetNextParticlePositionGroup( );

        // the members of this group are referenced by asset id's, we need to translate this to array position..

        for ( counter = 0; counter < temp_group.number_of_members; counter++ ) {
            int asset_id                = temp_group.members[counter];
            temp_group.members[counter] = all_assets_list->ReturnArrayPositionFromID(asset_id, last_found_position);
            last_found_position         = temp_group.members[counter];
            if ( last_found_position == -1 ) {
                wxPrintf("Oops. Could not find array position of asset with ID %i\n", asset_id);
            }
        }

        all_groups_list->AddGroup(&temp_group);
        if ( temp_group.id > current_group_number )
            current_group_number = temp_group.id;
    }

    main_frame->current_project.database.EndAllParticlePositionGroupsSelect( );
    FillGroupList( );
    FillContentsList( );
    //is_dirty = true;
}

void MyParticlePositionAssetPanel::FillAssetSpecificContentsList( ) {
    ContentsListBox->SetItemCount(all_groups_list->groups[selected_group].number_of_members);

    ContentsListBox->InsertColumn(0, "I.D.", wxLIST_FORMAT_LEFT, 100);
    ContentsListBox->InsertColumn(1, "Parent Image I.D.", wxLIST_FORMAT_LEFT, 100);
    ContentsListBox->InsertColumn(2, "Pick Job I.D.", wxLIST_FORMAT_LEFT, 100);
    ContentsListBox->InsertColumn(3, "X Position", wxLIST_FORMAT_LEFT, 100);
    ContentsListBox->InsertColumn(4, "Y Position", wxLIST_FORMAT_LEFT, 100);

    /*
		for (long counter = 0; counter < all_groups_list->groups[selected_group].number_of_members; counter++)
		{
			ContentsListBox->InsertItem(counter, wxString::Format(wxT("%i"), all_assets_list->ReturnParticlePositionAssetPointer(all_groups_list->ReturnGroupMember(selected_group, counter))->asset_id, counter));
			ContentsListBox->SetItem(counter, 1, wxString::Format(wxT("%i"), all_assets_list->ReturnParticlePositionAssetPointer(all_groups_list->ReturnGroupMember(selected_group, counter))->parent_id));
			ContentsListBox->SetItem(counter, 2, wxString::Format(wxT("%i"), all_assets_list->ReturnParticlePositionAssetPointer(all_groups_list->ReturnGroupMember(selected_group, counter))->pick_job_id));
			ContentsListBox->SetItem(counter, 3, wxString::Format(wxT("%.2f"), all_assets_list->ReturnParticlePositionAssetPointer(all_groups_list->ReturnGroupMember(selected_group, counter))->x_position));
			ContentsListBox->SetItem(counter, 4, wxString::Format(wxT("%.2f"), all_assets_list->ReturnParticlePositionAssetPointer(all_groups_list->ReturnGroupMember(selected_group, counter))->y_position));

		}*/
}

wxString MyParticlePositionAssetPanel::ReturnItemText(long item, long column) const {
    switch ( column ) {
        case 0:
            return wxString::Format(wxT("%i"), all_assets_list->ReturnParticlePositionAssetPointer(all_groups_list->ReturnGroupMember(selected_group, item))->asset_id);
            break;
        case 1:
            return wxString::Format(wxT("%i"), all_assets_list->ReturnParticlePositionAssetPointer(all_groups_list->ReturnGroupMember(selected_group, item))->parent_id);
            break;
        case 2:
            return wxString::Format(wxT("%i"), all_assets_list->ReturnParticlePositionAssetPointer(all_groups_list->ReturnGroupMember(selected_group, item))->pick_job_id);
            break;
        case 3:
            return wxString::Format(wxT("%.2f Å"), all_assets_list->ReturnParticlePositionAssetPointer(all_groups_list->ReturnGroupMember(selected_group, item))->x_position);
            break;
        case 4:
            return wxString::Format(wxT("%.2f Å"), all_assets_list->ReturnParticlePositionAssetPointer(all_groups_list->ReturnGroupMember(selected_group, item))->y_position);
            break;
        default:
            MyPrintWithDetails("Error, asking for column (%li) which does not exist", column);
            return "";
    }
}

void MyParticlePositionAssetPanel::NewFromParentClick(wxCommandEvent& event) {
    MyDebugAssertTrue(image_asset_panel->ReturnNumberOfGroups( ) > 1, "No image groups to work from. Button should have been disabled");

    // Ask the user which movie group they want to "copy"
    // We can only allow them to copy a group where each member has a corresponding image (output by unblur, with the corresponding parent_id)
    wxArrayString my_choices;
    wxArrayInt    choice_group_numbers;

    for ( long group_counter = 1; group_counter < image_asset_panel->ReturnNumberOfGroups( ); group_counter++ ) {
        my_choices.Add(image_asset_panel->ReturnGroupName(group_counter));
    }

    // Generate a dialog for the user
    wxSingleChoiceDialog* group_choice = new wxSingleChoiceDialog(this, "Make a group from which image group?", "Select Group", my_choices);

    // Assuming the user chose a group, generate a new image group
    int selected_group_number;

    if ( group_choice->ShowModal( ) == wxID_OK ) {
        selected_group_number = group_choice->GetSelection( ) + 1;

        main_frame->current_project.database.Begin( );
        wxString new_group_name = image_asset_panel->ReturnGroupName(selected_group_number);
        current_group_number++;
        all_groups_list->AddGroup(new_group_name);
        all_groups_list->groups[all_groups_list->number_of_groups - 1].id = current_group_number;
        AddGroupToDatabase(current_group_number, new_group_name, current_group_number);

        // ProgressBar..

        wxArrayLong particle_positions_to_add;

        long current_image_asset_id;

        OneSecondProgressDialog* my_progress_dialog = new OneSecondProgressDialog("Creating group", "Adding to group...", image_asset_panel->ReturnGroupSize(selected_group_number), this, wxPD_AUTO_HIDE | wxPD_APP_MODAL | wxPD_REMAINING_TIME);

        for ( long image_counter = 0; image_counter < image_asset_panel->ReturnGroupSize(selected_group_number); image_counter++ ) {
            current_image_asset_id = image_asset_panel->ReturnAssetID(image_asset_panel->ReturnGroupMember(selected_group_number, image_counter));
            for ( long particle_counter = 0; particle_counter < ReturnNumberOfAssets( ); particle_counter++ ) {
                if ( ReturnParentAssetID(particle_counter) == current_image_asset_id ) {
                    all_groups_list->groups[all_groups_list->number_of_groups - 1].AddMember(particle_counter);
                    InsertGroupMemberToDatabase(all_groups_list->number_of_groups - 1, all_groups_list->groups[all_groups_list->number_of_groups - 1].number_of_members - 1);
                }
            }

            my_progress_dialog->Update(image_counter + 1);
        }

        my_progress_dialog->Destroy( );
        main_frame->current_project.database.Commit( );

        FillGroupList( );
        FillContentsList( );
        DirtyGroups( );
        main_frame->RecalculateAssetBrowser( );
    }

    group_choice->Destroy( );
}

void MyParticlePositionAssetPanel::ImportAssetClick(wxCommandEvent& event) {
    // Get a text file which should have asset_id x_pos y_pos

    wxFileDialog openFileDialog(this, _("Open TXT file"), "", "", "TXT files (*.txt)|*.txt;*.txt", wxFD_OPEN | wxFD_FILE_MUST_EXIST);

    if ( openFileDialog.ShowModal( ) == wxID_OK ) {
        wxTextFile        input_file;
        wxString          current_line;
        wxString          current_token;
        wxStringTokenizer current_tokenizer;
        bool              have_errors = false;
        int               token_counter;
        input_file.Open(openFileDialog.GetPath( ));
        MyErrorDialog* my_error = new MyErrorDialog(this);
        long           image_asset_id;

        ParticlePositionAsset temp_asset;
        temp_asset.pick_job_id = -1;
        temp_asset.picking_id  = -1;

        // for each line, we add an asset..

        OneSecondProgressDialog* my_dialog = new OneSecondProgressDialog("Import Assets", "Importing Assets", input_file.GetLineCount( ), this);

        // for database..

        main_frame->current_project.database.BeginParticlePositionAssetInsert( );

        for ( long counter = 0; counter < input_file.GetLineCount( ); counter++ ) {
            current_line = input_file.GetLine(counter);
            current_line.Trim( );

            if ( current_line.IsEmpty( ) == false && current_line.StartsWith("#") == false ) {
                current_tokenizer.SetString(current_line);
                wxPrintf("Current Line = %s, number_tokens = %li\n", current_line, current_tokenizer.CountTokens( ));

                if ( current_tokenizer.CountTokens( ) < 3 ) {
                    my_error->ErrorText->AppendText(wxString::Format(wxT("Line %li contains less than 3 (%li) values and will be ignored\n"), counter, current_tokenizer.CountTokens( )));
                    have_errors = true;
                }
                else {
                    if ( current_tokenizer.CountTokens( ) > 3 ) {
                        my_error->ErrorText->AppendText(wxString::Format(wxT("Line %li contains more than 3 values, only first 3 values will be parsed..\n"), counter));
                        have_errors = true;
                    }

                    // get the first token

                    current_token = current_tokenizer.GetNextToken( );

                    if ( current_token.ToLong(&image_asset_id) == false ) {
                        // it wasn't a number, so is it a valid filename?
                        long array_position = reinterpret_cast<ImageAssetList*>(image_asset_panel->all_assets_list)->FindFile(current_token, true);
                        if ( array_position != -1 )
                            image_asset_id = image_asset_panel->ReturnAssetID(array_position);
                        else
                            image_asset_id = -1;
                    }

                    if ( image_asset_id == -1 ) {
                        my_error->ErrorText->AppendText(wxString::Format(wxT("Line %li, column 1 is not read as a valid Asset ID or filename, and so the line will be ignored\n"), counter));
                        have_errors = true;
                    }
                    else {
                        if ( image_asset_panel->ReturnArrayPositionFromAssetID(image_asset_id) < 0 ) {
                            my_error->ErrorText->AppendText(wxString::Format(wxT("Line %li, column 1 : Asset (%li) is not an existing image asset\nn"), counter, image_asset_id));
                            have_errors = true;
                        }
                        else {
                            temp_asset.parent_id = image_asset_id;
                            current_token        = current_tokenizer.GetNextToken( );
                            if ( current_token.ToDouble(&temp_asset.x_position) == false ) {
                                my_error->ErrorText->AppendText(wxString::Format(wxT("Line %li, column 2 is not read as a valid X position, and so the line will be ignored\n"), counter));
                                have_errors = true;
                            }
                            else {
                                current_token = current_tokenizer.GetNextToken( );
                                if ( current_token.ToDouble(&temp_asset.y_position) == false ) {
                                    my_error->ErrorText->AppendText(wxString::Format(wxT("Line %li, column 3 is not read as a valid Y position, and so the line will be ignored\n"), counter));
                                    have_errors = true;
                                }
                                else // if we get here, we should be ok..
                                {
                                    temp_asset.asset_id = current_asset_number;
                                    AddAsset(&temp_asset);
                                    //main_frame->current_project.database.AddNextParticlePositionAsset(temp_asset.asset_id, temp_asset.parent_id, temp_asset.pick_job_id, temp_asset.x_position, temp_asset.y_position);
                                    main_frame->current_project.database.AddNextParticlePositionAsset(&temp_asset);
                                }
                            }
                        }
                    }
                }

                my_dialog->Update(counter);
            }
        }

        main_frame->current_project.database.EndParticlePositionAssetInsert( );

        my_dialog->Destroy( );

        // errros?

        if ( have_errors == true ) {
            my_error->ShowModal( );
        }

        my_error->Destroy( );

        is_dirty = true;
    }
}
