//#include "../core/core_headers.h"
#include "../core/gui_core_headers.h"

extern MyRefinementPackageAssetPanel *refinement_package_asset_panel;
extern MyRefine3DPanel *refine_3d_panel;

//extern MyImageAssetPanel *image_asset_panel;

AtomicCoordinatesAssetPanel::AtomicCoordinatesAssetPanel( wxWindow* parent )
:
MyAssetParentPanel( parent )
{
	Label0Title->SetLabel("Name : ");
	Label1Title->SetLabel("I.D. : ");
	Label2Title->SetLabel("Simulation 3d Job I.D. : ");
	Label3Title->SetLabel("Pixel Size : ");
	Label4Title->SetLabel("X Size : ");
	Label5Title->SetLabel("Y Size : ");
	Label6Title->SetLabel("Z Size : ");
	Label7Title->SetLabel("");
	Label8Title->SetLabel("");
	Label9Title->SetLabel("");

	UpdateInfo();

	SplitterWindow->Unsplit(LeftPanel);


	AssetTypeText->SetLabel("AtomicCoordinates");

	all_groups_list->groups[0].SetName("All AtomicCoordinates");
	all_assets_list = new AtomicCoordinatesAssetList;
	FillGroupList();
	FillContentsList();
}

AtomicCoordinatesAssetPanel::~AtomicCoordinatesAssetPanel()
{
	delete reinterpret_cast <AtomicCoordinatesAssetList*> (all_assets_list);
}

void AtomicCoordinatesAssetPanel::UpdateInfo()
{
	if (selected_content >= 0 && selected_group >= 0 && all_groups_list->groups[selected_group].number_of_members > 0)
	{
		Label0Text->SetLabel(all_assets_list->ReturnAtomicCoordinatesAssetPointer(all_groups_list->ReturnGroupMember(selected_group, selected_content))->asset_name);
		Label1Text->SetLabel(wxString::Format(wxT("%i"), all_assets_list->ReturnAtomicCoordinatesAssetPointer(all_groups_list->ReturnGroupMember(selected_group, selected_content))->asset_id));
		Label2Text->SetLabel(wxString::Format(wxT("%li"), all_assets_list->ReturnAtomicCoordinatesAssetPointer(all_groups_list->ReturnGroupMember(selected_group, selected_content))->simulation_3d_job_id));
		Label3Text->SetLabel(wxString::Format(wxT("%.4f"), all_assets_list->ReturnAtomicCoordinatesAssetPointer(all_groups_list->ReturnGroupMember(selected_group, selected_content))->pixel_size));
		Label4Text->SetLabel(wxString::Format(wxT("%i"), all_assets_list->ReturnAtomicCoordinatesAssetPointer(all_groups_list->ReturnGroupMember(selected_group, selected_content))->x_size));
		Label5Text->SetLabel(wxString::Format(wxT("%i"), all_assets_list->ReturnAtomicCoordinatesAssetPointer(all_groups_list->ReturnGroupMember(selected_group, selected_content))->y_size));
		Label6Text->SetLabel(wxString::Format(wxT("%i"), all_assets_list->ReturnAtomicCoordinatesAssetPointer(all_groups_list->ReturnGroupMember(selected_group, selected_content))->z_size));
		Label7Text->SetLabel("");
		Label8Text->SetLabel("");
		Label9Text->SetLabel("");
	}
	else
	{
		Label0Text->SetLabel("-");
		Label1Text->SetLabel("-");
		Label2Text->SetLabel("-");
		Label3Text->SetLabel("-");
		Label4Text->SetLabel("-");
		Label5Text->SetLabel("-");
		Label6Text->SetLabel("-");
		Label7Text->SetLabel("");
		Label8Text->SetLabel("");
		Label9Text->SetLabel("");
	}

}


AtomicCoordinatesAsset* AtomicCoordinatesAssetPanel::ReturnAssetPointer(long wanted_asset)
{
	return all_assets_list->ReturnAtomicCoordinatesAssetPointer(wanted_asset);
}

int AtomicCoordinatesAssetPanel::ShowDeleteMessageDialog()
{
	wxMessageDialog check_dialog(this, "This will remove the selected volume(s) from your ENTIRE project!\n\nNote that this is only a database deletion, no data files are deleted.\n\nAre you sure you want to continue?", "Are you sure?", wxYES_NO | wxICON_WARNING);
	return check_dialog.ShowModal();

}

int AtomicCoordinatesAssetPanel::ShowDeleteAllMessageDialog()
{
	wxMessageDialog check_dialog(this, "This will remove the ALL volumes from your ENTIRE project!\n\nNote that this is only a database deletion, no data files are deleted.\n\nAre you sure you want to continue?", "Are you sure?", wxYES_NO | wxICON_WARNING);
	return check_dialog.ShowModal();
}

void AtomicCoordinatesAssetPanel::CompletelyRemoveAsset(long wanted_asset)
{
	long wanted_asset_id = all_assets_list->ReturnAssetID(wanted_asset);
	if (wanted_asset_id >= 0) CompletelyRemoveAssetByID(wanted_asset_id);
}

void AtomicCoordinatesAssetPanel::CompletelyRemoveAssetByID(long wanted_asset_id)
{
	long counter;
	long asset_array_position;

	wxArrayString tables;

	// set an reference to this 3D in startups to 0 all startup jobs that produced this 3D..

	tables = main_frame->current_project.database.ReturnStringArrayFromSelectCommand("SELECT name FROM sqlite_master WHERE name like 'STARTUP_RESULT_%'");

	// go through all the tables and replace this..

	for (counter = 0; counter < tables.GetCount(); counter++)
	{
		main_frame->current_project.database.ExecuteSQL(wxString::Format("UPDATE %s SET ATOMICCOORDINATES_ASSET_ID=0 WHERE ATOMICCOORDINATES_ASSET_ID=%li", tables[counter], wanted_asset_id));
	}

	// refinement_package_current_references

	tables = main_frame->current_project.database.ReturnStringArrayFromSelectCommand("SELECT name FROM sqlite_master WHERE name like 'REFINEMENT_PACKAGE_CURRENT_REFERENCES_%'");

	for (counter = 0; counter < tables.GetCount(); counter++)
	{
		main_frame->current_project.database.ExecuteSQL(wxString::Format("UPDATE %s SET ATOMICCOORDINATES_ASSET_ID=-1 WHERE ATOMICCOORDINATES_ASSET_ID=%li", tables[counter], wanted_asset_id));
	}

	// refinement details

	tables = main_frame->current_project.database.ReturnStringArrayFromSelectCommand("SELECT name FROM sqlite_master WHERE name like 'REFINEMENT_DETAILS_%'");

	for (counter = 0; counter < tables.GetCount(); counter++)
	{
		main_frame->current_project.database.ExecuteSQL(wxString::Format("UPDATE %s SET REFERENCE_ATOMICCOORDINATES_ASSET_ID=0 WHERE REFERENCE_ATOMICCOORDINATES_ASSET_ID=%li", tables[counter], wanted_asset_id));
	}

	// delete from in memory refinement package assets

	refinement_package_asset_panel->RemoveVolumeFromAllRefinementPackages(wanted_asset_id);


	// now delete from volume_assets

	main_frame->current_project.database.ExecuteSQL(wxString::Format("DELETE FROM ATOMICCOORDINATES_ASSETS WHERE ATOMICCOORDINATES_ASSET_ID=%li", wanted_asset_id).ToUTF8().data());


	asset_array_position = all_assets_list->ReturnArrayPositionFromID(wanted_asset_id);
	all_assets_list->RemoveAsset(asset_array_position);
	RemoveAssetFromGroups(asset_array_position);
	main_frame->DirtyVolumes();
}

void AtomicCoordinatesAssetPanel::DoAfterDeletionCleanup()
{
	main_frame->DirtyVolumes();
	refinement_package_asset_panel->ReDrawActiveReferences();
	refine_3d_panel->ReDrawActiveReferences();
}


void AtomicCoordinatesAssetPanel::RemoveAssetFromDatabase(long wanted_asset)
{
	main_frame->current_project.database.ExecuteSQL(wxString::Format("DELETE FROM ATOMICCOORDINATES_ASSETS WHERE ATOMICCOORDINATES_ASSET_ID=%i", all_assets_list->ReturnAssetID(wanted_asset)).ToUTF8().data());
	all_assets_list->RemoveAsset(wanted_asset);
}

void AtomicCoordinatesAssetPanel::RemoveFromGroupInDatabase(int wanted_group_id, int wanted_asset_id)
{
	main_frame->current_project.database.ExecuteSQL(wxString::Format("DELETE FROM ATOMICCOORDINATES_GROUP_%i WHERE ATOMICCOORDINATES_ASSET_ID=%i", wanted_group_id, wanted_asset_id).ToUTF8().data());
}

void AtomicCoordinatesAssetPanel::InsertGroupMemberToDatabase(int wanted_group, int wanted_asset)
{
	MyDebugAssertTrue(wanted_group > 0 && wanted_group < all_groups_list->number_of_groups, "Requesting a group (%i) that doesn't exist!", wanted_group);
	MyDebugAssertTrue(wanted_asset >= 0 && wanted_asset < all_assets_list->number_of_assets, "Requesting an partice position (%i) that doesn't exist!", wanted_asset);

	main_frame->current_project.database.InsertOrReplace(wxString::Format("ATOMICCOORDINATES_GROUP_%i", ReturnGroupID(wanted_group)).ToUTF8().data(), "ii", "MEMBER_NUMBER", "ATOMICCOORDINATES_ASSET_ID", ReturnGroupSize(wanted_group), ReturnGroupMemberID(wanted_group, wanted_asset));
}

void AtomicCoordinatesAssetPanel::InsertArrayofGroupMembersToDatabase(long wanted_group, wxArrayLong *wanted_array, OneSecondProgressDialog *progress_dialog)
{

}

void  AtomicCoordinatesAssetPanel::RemoveAllFromDatabase()
{
	/*
	for (long counter = 1; counter < all_groups_list->number_of_groups; counter++)
	{
		main_frame->current_project.database.ExecuteSQL(wxString::Format("DROP TABLE ATOMICCOORDINATES_GROUP_%i", all_groups_list->groups[counter].id).ToUTF8().data());
	}

	main_frame->current_project.database.ExecuteSQL("DROP TABLE ATOMICCOORDINATES_GROUP_LIST");
	main_frame->current_project.database.CreateVolumeGroupListTable();

	main_frame->current_project.database.ExecuteSQL("DROP TABLE ATOMICCOORDINATES_ASSETS");
	main_frame->current_project.database.CreateAtomicCoordinatesAssetTable();
	*/

	for (long counter = ReturnNumberOfAssets() -1; counter >= 0; counter--)
	{
		CompletelyRemoveAsset(counter);
	}





}

void AtomicCoordinatesAssetPanel::RemoveAllGroupMembersFromDatabase(int wanted_group_id)
{
	main_frame->current_project.database.ExecuteSQL(wxString::Format("DROP TABLE ATOMICCOORDINATES_GROUP_%i", wanted_group_id).ToUTF8().data());
	main_frame->current_project.database.CreateTable(wxString::Format("ATOMICCOORDINATES_GROUP_%i", wanted_group_id).ToUTF8().data(), "ii", "MEMBER_NUMBER", "ATOMICCOORDINATES_POSITION_ASSET_ID");
}

void AtomicCoordinatesAssetPanel::AddGroupToDatabase(int wanted_group_id, const char * wanted_group_name, int wanted_list_id)
{
	main_frame->current_project.database.InsertOrReplace("ATOMICCOORDINATES_GROUP_LIST", "iti", "GROUP_ID", "GROUP_NAME", "LIST_ID", wanted_group_id, wanted_group_name, wanted_list_id);
	main_frame->current_project.database.CreateTable(wxString::Format("ATOMICCOORDINATES_GROUP_%i", wanted_list_id).ToUTF8().data(), "ii", "MEMBER_NUMBER", "ATOMICCOORDINATES_POSITION_ASSET_ID");
}

void AtomicCoordinatesAssetPanel::RemoveGroupFromDatabase(int wanted_group_id)
{
	main_frame->current_project.database.ExecuteSQL(wxString::Format("DROP TABLE ATOMICCOORDINATES_GROUP_%i", wanted_group_id).ToUTF8().data());
	main_frame->current_project.database.ExecuteSQL(wxString::Format("DELETE FROM ATOMICCOORDINATES_GROUP_LIST WHERE GROUP_ID=%i", wanted_group_id));
}

void AtomicCoordinatesAssetPanel::RenameGroupInDatabase(int wanted_group_id, const char *wanted_name)
{
	wxString name = wanted_name;
	name.Replace("'", "''");
	wxString sql_command = wxString::Format("UPDATE ATOMICCOORDINATES_GROUP_LIST SET GROUP_NAME='%s' WHERE GROUP_ID=%i", name, wanted_group_id);
	main_frame->current_project.database.ExecuteSQL(sql_command.ToUTF8().data());

}

void AtomicCoordinatesAssetPanel::RenameAsset(long wanted_asset, wxString wanted_name)
{
	wxString name = wanted_name;
	name.Replace("'", "''");
	all_assets_list->ReturnAtomicCoordinatesAssetPointer(wanted_asset)->asset_name = wanted_name;
	wxString sql_command = wxString::Format("UPDATE ATOMICCOORDINATES_ASSETS SET NAME='%s' WHERE ATOMICCOORDINATES_ASSET_ID=%i", name, all_assets_list->ReturnAtomicCoordinatesAssetPointer(wanted_asset)->asset_id);
	main_frame->current_project.database.ExecuteSQL(sql_command.ToUTF8().data());
	main_frame->DirtyVolumes();

}

void AtomicCoordinatesAssetPanel::ImportAllFromDatabase()
{

	int counter;
	AtomicCoordinatesAsset temp_asset;
	AssetGroup temp_group;

	all_assets_list->RemoveAll();
	all_groups_list->RemoveAll();

	// First all assets..

	main_frame->current_project.database.BeginAllAtomicCoordinatesAssetsSelect();

	while (main_frame->current_project.database.last_return_code == SQLITE_ROW)
	{
		temp_asset = main_frame->current_project.database.GetNextAtomicCoordinatesAsset();
		AddAsset(&temp_asset);
	}

	main_frame->current_project.database.EndAllAtomicCoordinatesAssetsSelect();

	// Now the groups..

	main_frame->current_project.database.BeginAllVolumeGroupsSelect();

	while (main_frame->current_project.database.last_return_code == SQLITE_ROW)
	{
		temp_group = main_frame->current_project.database.GetNextVolumeGroup();

		// the members of this group are referenced by asset id's, we need to translate this to array position..

		for (counter = 0; counter < temp_group.number_of_members; counter++)
		{
			temp_group.members[counter] = all_assets_list->ReturnArrayPositionFromID(temp_group.members[counter]);
		}

		all_groups_list->AddGroup(&temp_group);
		if (temp_group.id > current_group_number) current_group_number = temp_group.id;
	}

	main_frame->current_project.database.EndAllVolumeGroupsSelect();

	main_frame->DirtyVolumes();
}

void AtomicCoordinatesAssetPanel::FillAssetSpecificContentsList()
{
	ContentsListBox->InsertColumn(0, "Name", wxLIST_FORMAT_LEFT,  wxLIST_AUTOSIZE_USEHEADER );
	ContentsListBox->InsertColumn(1, "I.D.", wxLIST_FORMAT_LEFT,  wxLIST_AUTOSIZE_USEHEADER );
	ContentsListBox->InsertColumn(2, "Reconstruction Job I.D.", wxLIST_FORMAT_LEFT,  wxLIST_AUTOSIZE_USEHEADER );
	ContentsListBox->InsertColumn(3, "Pixel Size", wxLIST_FORMAT_LEFT,  wxLIST_AUTOSIZE_USEHEADER );
	ContentsListBox->InsertColumn(4, "X Size", wxLIST_FORMAT_LEFT,  wxLIST_AUTOSIZE_USEHEADER );
	ContentsListBox->InsertColumn(5, "Y Size", wxLIST_FORMAT_LEFT,  wxLIST_AUTOSIZE_USEHEADER );
	ContentsListBox->InsertColumn(6, "Z Size", wxLIST_FORMAT_LEFT,  wxLIST_AUTOSIZE_USEHEADER );
/*

		for (long counter = 0; counter < all_groups_list->groups[selected_group].number_of_members; counter++)
		{
			ContentsListBox->InsertItem(counter, wxString::Format(wxT("%s"), all_assets_list->ReturnAtomicCoordinatesAssetPointer(all_groups_list->ReturnGroupMember(selected_group, counter))->filename.GetFullName(), counter));
			ContentsListBox->SetItem(counter, 1, wxString::Format(wxT("%i"), all_assets_list->ReturnAtomicCoordinatesAssetPointer(all_groups_list->ReturnGroupMember(selected_group, counter))->asset_id));
			ContentsListBox->SetItem(counter, 2, wxString::Format(wxT("%i"), all_assets_list->ReturnAtomicCoordinatesAssetPointer(all_groups_list->ReturnGroupMember(selected_group, counter))->simulation_3d_job_id));
			ContentsListBox->SetItem(counter, 3, wxString::Format(wxT("%.4f"), all_assets_list->ReturnAtomicCoordinatesAssetPointer(all_groups_list->ReturnGroupMember(selected_group, counter))->pixel_size));
			ContentsListBox->SetItem(counter, 4, wxString::Format(wxT("%i"), all_assets_list->ReturnAtomicCoordinatesAssetPointer(all_groups_list->ReturnGroupMember(selected_group, counter))->x_size));
			ContentsListBox->SetItem(counter, 5, wxString::Format(wxT("%i"), all_assets_list->ReturnAtomicCoordinatesAssetPointer(all_groups_list->ReturnGroupMember(selected_group, counter))->y_size));
			ContentsListBox->SetItem(counter, 6, wxString::Format(wxT("%i"), all_assets_list->ReturnAtomicCoordinatesAssetPointer(all_groups_list->ReturnGroupMember(selected_group, counter))->z_size));


		}*/
}

wxString AtomicCoordinatesAssetPanel::ReturnItemText(long item, long column) const
{
	switch(column)
	{
	    case 0  :
	    	return all_assets_list->ReturnAtomicCoordinatesAssetPointer(all_groups_list->ReturnGroupMember(selected_group, item))->asset_name;
	       break;
	    case 1  :
	    	return wxString::Format(wxT("%i"), all_assets_list->ReturnAtomicCoordinatesAssetPointer(all_groups_list->ReturnGroupMember(selected_group, item))->asset_id);
	       break;
	    case 2  :
	    	return wxString::Format(wxT("%li"), all_assets_list->ReturnAtomicCoordinatesAssetPointer(all_groups_list->ReturnGroupMember(selected_group, item))->simulation_3d_job_id);
	       break;
	    case 3  :
	    	return wxString::Format(wxT("%.4f"), all_assets_list->ReturnAtomicCoordinatesAssetPointer(all_groups_list->ReturnGroupMember(selected_group, item))->pixel_size);
	       break;
	    case 4  :
	    	return wxString::Format(wxT("%i"), all_assets_list->ReturnAtomicCoordinatesAssetPointer(all_groups_list->ReturnGroupMember(selected_group, item))->x_size);
	       break;
	    case 5  :
	    	return wxString::Format(wxT("%i"), all_assets_list->ReturnAtomicCoordinatesAssetPointer(all_groups_list->ReturnGroupMember(selected_group, item))->y_size);
		   break;
	    case 6  :
	    	return wxString::Format(wxT("%i"), all_assets_list->ReturnAtomicCoordinatesAssetPointer(all_groups_list->ReturnGroupMember(selected_group, item))->z_size);
		   break;
	    default :
	       MyPrintWithDetails("Error, asking for column (%li) which does not exist", column);
	       return "";
	}
}

bool AtomicCoordinatesAssetPanel::IsFileAnAsset(wxFileName file_to_check)
{
	if (reinterpret_cast <AtomicCoordinatesAssetList*>  (all_assets_list)->FindFile(file_to_check) == -1) return false;
	else return true;
}

void AtomicCoordinatesAssetPanel::ImportAssetClick( wxCommandEvent& event )
{
	MyVolumeImportDialog *import_dialog = new MyVolumeImportDialog(this);
		import_dialog->ShowModal();
		main_frame->DirtyVolumes();

}