//#include "../core/core_headers.h"
#include "../core/gui_core_headers.h"

MyMovieAssetPanel::MyMovieAssetPanel( wxWindow* parent )
:
MyAssetParentPanel( parent )
{
	Label0Title->SetLabel("Name : ");
	Label1Title->SetLabel("I.D. : ");
	Label2Title->SetLabel("No. Frames : ");
	Label3Title->SetLabel("Pixel Size : ");
	Label4Title->SetLabel("X Size : ");
	Label5Title->SetLabel("Y Size : ");
	Label6Title->SetLabel("Total Exp. : ");
	Label7Title->SetLabel("Exp. Per Frame : ");
	Label8Title->SetLabel("Voltage : ");
	Label9Title->SetLabel("Cs : ");

	AssetTypeText->SetLabel("Movies");

	all_groups_list->groups[0].SetName("All Movies");
	all_assets_list = new MovieAssetList;
	FillGroupList();
	FillContentsList();
}

MyMovieAssetPanel::~MyMovieAssetPanel()
{

	delete all_assets_list;
}

void MyMovieAssetPanel::UpdateInfo()
{
	if (selected_content >= 0 && selected_group >= 0 && all_groups_list->groups[selected_group].number_of_members > 0)
	{
		Label0Text->SetLabel(all_assets_list->ReturnAssetPointer(all_groups_list->ReturnGroupMember(selected_group, selected_content))->asset_name);
		Label1Text->SetLabel(wxString::Format(wxT("%i"), all_assets_list->ReturnMovieAssetPointer(all_groups_list->ReturnGroupMember(selected_group, selected_content))->asset_id));
		Label2Text->SetLabel(wxString::Format(wxT("%i"), all_assets_list->ReturnMovieAssetPointer(all_groups_list->ReturnGroupMember(selected_group, selected_content))->number_of_frames));
		Label3Text->SetLabel(wxString::Format(wxT("%.4f Å"), all_assets_list->ReturnMovieAssetPointer(all_groups_list->ReturnGroupMember(selected_group, selected_content))->pixel_size));
		Label4Text->SetLabel(wxString::Format(wxT("%i px"), all_assets_list->ReturnMovieAssetPointer(all_groups_list->ReturnGroupMember(selected_group, selected_content))->x_size));
		Label5Text->SetLabel(wxString::Format(wxT("%i px"), all_assets_list->ReturnMovieAssetPointer(all_groups_list->ReturnGroupMember(selected_group, selected_content))->y_size));
		Label6Text->SetLabel(wxString::Format(wxT("%.2f e¯/Å²"), all_assets_list->ReturnMovieAssetPointer(all_groups_list->ReturnGroupMember(selected_group, selected_content))->total_dose));
		Label7Text->SetLabel(wxString::Format(wxT("%.2f e¯/Å²"), all_assets_list->ReturnMovieAssetPointer(all_groups_list->ReturnGroupMember(selected_group, selected_content))->dose_per_frame));
		Label8Text->SetLabel(wxString::Format(wxT("%.2f kV"), all_assets_list->ReturnMovieAssetPointer(all_groups_list->ReturnGroupMember(selected_group, selected_content))->microscope_voltage));
		Label9Text->SetLabel(wxString::Format(wxT("%.2f mm"), all_assets_list->ReturnMovieAssetPointer(all_groups_list->ReturnGroupMember(selected_group, selected_content))->spherical_aberration));
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
		Label7Text->SetLabel("-");
		Label8Text->SetLabel("-");
		Label9Text->SetLabel("-");
	}

}

bool MyMovieAssetPanel::IsFileAnAsset(wxFileName file_to_check)
{
	if (reinterpret_cast <MovieAssetList*>  (all_assets_list)->FindFile(file_to_check) == -1) return false;
	else return true;
}

MovieAsset* MyMovieAssetPanel::ReturnAssetPointer(long wanted_asset)
{
	return all_assets_list->ReturnMovieAssetPointer(wanted_asset);
}

void MyMovieAssetPanel::RemoveAssetFromDatabase(long wanted_asset)
{
	main_frame->current_project.database.ExecuteSQL(wxString::Format("DELETE FROM MOVIE_ASSETS WHERE MOVIE_ASSET_ID=%i", all_assets_list->ReturnAssetID(wanted_asset)).ToUTF8().data());
	all_assets_list->RemoveAsset(wanted_asset);
}

void MyMovieAssetPanel::RemoveFromGroupInDatabase(int wanted_group_id, int wanted_asset_id)
{
	main_frame->current_project.database.ExecuteSQL(wxString::Format("DELETE FROM MOVIE_GROUP_%i WHERE MOVIE_ASSET_ID=%i", wanted_group_id, wanted_asset_id).ToUTF8().data());
}

void MyMovieAssetPanel::InsertGroupMemberToDatabase(int wanted_group, int wanted_asset)
{
	MyDebugAssertTrue(wanted_group > 0 && wanted_group < all_groups_list->number_of_groups, "Requesting a group (%i) that doesn't exist!", wanted_group);
	MyDebugAssertTrue(wanted_asset >= 0 && wanted_asset < all_assets_list->number_of_assets, "Requesting a movie(%i) that doesn't exist!", wanted_asset);

//	wxPrintf("AssetPanel wanted_group = %i\n", wanted_group);
	main_frame->current_project.database.InsertOrReplace(wxString::Format("MOVIE_GROUP_%i", ReturnGroupID(wanted_group)).ToUTF8().data(), "ii", "MEMBER_NUMBER", "MOVIE_ASSET_ID", ReturnGroupSize(wanted_group), ReturnGroupMemberID(wanted_group, wanted_asset));
}

void MyMovieAssetPanel::InsertArrayofGroupMembersToDatabase(long wanted_group, wxArrayLong *wanted_array, OneSecondProgressDialog *progress_dialog )
{
	MyDebugAssertTrue(wanted_group > 0 && wanted_group < all_groups_list->number_of_groups, "Requesting a group (%i) that doesn't exist!", wanted_group);

	int current_member_number = main_frame->current_project.database.ReturnSingleIntFromSelectCommand(wxString::Format("SELECT MAX(MEMBER_NUMBER) FROM MOVIE_GROUP_%i", ReturnGroupID(wanted_group)).ToUTF8().data());
	main_frame->current_project.database.BeginBatchInsert(wxString::Format("MOVIE_GROUP_%i", ReturnGroupID(wanted_group)).ToUTF8().data(), 2, "MEMBER_NUMBER", "MOVIE_ASSET_ID");

	for (long counter = 0; counter < wanted_array->GetCount(); counter++)
	{
		MyDebugAssertTrue(wanted_array->Item(counter) >= 0 && wanted_array->Item(counter) < all_assets_list->number_of_assets, "Requesting an asset(%li) that doesn't exist!", wanted_array->Item(counter));
		main_frame->current_project.database.AddToBatchInsert("ii", current_member_number, ReturnGroupMemberID(0, int(wanted_array->Item(counter))));
		current_member_number++;
		if (progress_dialog != NULL) progress_dialog->Update(counter);
	}

	main_frame->current_project.database.EndBatchInsert();
}

void  MyMovieAssetPanel::RemoveAllFromDatabase()
{
	for (long counter = 1; counter < all_groups_list->number_of_groups; counter++)
	{
		main_frame->current_project.database.ExecuteSQL(wxString::Format("DROP TABLE MOVIE_GROUP_%i", all_groups_list->groups[counter].id).ToUTF8().data());
	}

	main_frame->current_project.database.ExecuteSQL("DROP TABLE MOVIE_GROUP_LIST");
	main_frame->current_project.database.CreateTable("MOVIE_GROUP_LIST", "pti", "GROUP_ID", "GROUP_NAME", "LIST_ID" );

	main_frame->current_project.database.ExecuteSQL("DROP TABLE MOVIE_ASSETS");
	main_frame->current_project.database.CreateMovieAssetTable();


}

void MyMovieAssetPanel::RemoveAllGroupMembersFromDatabase(int wanted_group_id)
{
	main_frame->current_project.database.ExecuteSQL(wxString::Format("DROP TABLE MOVIE_GROUP_%i", wanted_group_id).ToUTF8().data());
	main_frame->current_project.database.CreateTable(wxString::Format("MOVIE_GROUP_%i", wanted_group_id).ToUTF8().data(), "ii", "MEMBER_NUMBER", "MOVIE_ASSET_ID");
}

void MyMovieAssetPanel::AddGroupToDatabase(int wanted_group_id, const char * wanted_group_name, int wanted_list_id)
{
	main_frame->current_project.database.InsertOrReplace("MOVIE_GROUP_LIST", "iti", "GROUP_ID", "GROUP_NAME", "LIST_ID", wanted_group_id, wanted_group_name, wanted_list_id);
	main_frame->current_project.database.CreateTable(wxString::Format("MOVIE_GROUP_%i", wanted_list_id).ToUTF8().data(), "ii", "MEMBER_NUMBER", "MOVIE_ASSET_ID");
}

void MyMovieAssetPanel::RemoveGroupFromDatabase(int wanted_group_id)
{
	main_frame->current_project.database.ExecuteSQL(wxString::Format("DROP TABLE MOVIE_GROUP_%i", wanted_group_id).ToUTF8().data());
	main_frame->current_project.database.ExecuteSQL(wxString::Format("DELETE FROM MOVIE_GROUP_LIST WHERE GROUP_ID=%i", wanted_group_id));
}

void MyMovieAssetPanel::RenameGroupInDatabase(int wanted_group_id, const char *wanted_name)
{
	wxString sql_command = wxString::Format("UPDATE MOVIE_GROUP_LIST SET GROUP_NAME='%s' WHERE GROUP_ID=%i", wanted_name, wanted_group_id);
	main_frame->current_project.database.ExecuteSQL(sql_command.ToUTF8().data());

}

void MyMovieAssetPanel::RenameAsset(long wanted_asset, wxString wanted_name)
{
	all_assets_list->ReturnMovieAssetPointer(wanted_asset)->asset_name = wanted_name;
	wxString sql_command = wxString::Format("UPDATE MOVIE_ASSETS SET NAME='%s' WHERE MOVIE_ASSET_ID=%i", wanted_name, all_assets_list->ReturnMovieAssetPointer(wanted_asset)->asset_id);
	main_frame->current_project.database.ExecuteSQL(sql_command.ToUTF8().data());

}

void MyMovieAssetPanel::ImportAllFromDatabase()
{
	int counter;
	MovieAsset temp_asset;
	AssetGroup temp_group;

	all_assets_list->RemoveAll();
	all_groups_list->RemoveAll();

	// First all movie assets..

	main_frame->current_project.database.BeginAllMovieAssetsSelect();

	while (main_frame->current_project.database.last_return_code == SQLITE_ROW)
	{
		temp_asset = main_frame->current_project.database.GetNextMovieAsset();
		AddAsset(&temp_asset);
	}

	main_frame->current_project.database.EndAllMovieAssetsSelect();

	// Now the groups..

	main_frame->current_project.database.BeginAllMovieGroupsSelect();

	while (main_frame->current_project.database.last_return_code == SQLITE_ROW)
	{
		temp_group = main_frame->current_project.database.GetNextMovieGroup();

		// the members of this group are referenced by movie id's, we need to translate this to array position..

		for (counter = 0; counter < temp_group.number_of_members; counter++)
		{
			temp_group.members[counter] = all_assets_list->ReturnArrayPositionFromID(temp_group.members[counter]);
		}

		all_groups_list->AddGroup(&temp_group);
		if (temp_group.id > current_group_number) current_group_number = temp_group.id;
	}

	main_frame->current_project.database.EndAllMovieGroupsSelect();

	main_frame->DirtyMovieGroups();
	FillGroupList();
	FillContentsList();
}

void MyMovieAssetPanel::FillAssetSpecificContentsList()
{

		ContentsListBox->InsertColumn(0, "I.D.", wxLIST_FORMAT_LEFT,  wxLIST_AUTOSIZE_USEHEADER );
		ContentsListBox->InsertColumn(1, "Name", wxLIST_FORMAT_LEFT,  wxLIST_AUTOSIZE_USEHEADER );
		ContentsListBox->InsertColumn(2, "X Size", wxLIST_FORMAT_LEFT,  wxLIST_AUTOSIZE_USEHEADER );
		ContentsListBox->InsertColumn(3, "Y Size", wxLIST_FORMAT_LEFT,  wxLIST_AUTOSIZE_USEHEADER );
		ContentsListBox->InsertColumn(4, "No. frames", wxLIST_FORMAT_LEFT,  wxLIST_AUTOSIZE_USEHEADER );
		ContentsListBox->InsertColumn(5, "Pixel size", wxLIST_FORMAT_LEFT,  wxLIST_AUTOSIZE_USEHEADER );
		ContentsListBox->InsertColumn(6, "Exp. per frame", wxLIST_FORMAT_LEFT,  wxLIST_AUTOSIZE_USEHEADER );
		ContentsListBox->InsertColumn(7, "Cs", wxLIST_FORMAT_LEFT,  wxLIST_AUTOSIZE_USEHEADER );
		ContentsListBox->InsertColumn(8, "Voltage", wxLIST_FORMAT_LEFT,  wxLIST_AUTOSIZE_USEHEADER );
		ContentsListBox->InsertColumn(9, "Gain filename", wxLIST_FORMAT_LEFT,  wxLIST_AUTOSIZE_USEHEADER );
		ContentsListBox->InsertColumn(10,"Output bin. factor", wxLIST_FORMAT_LEFT, wxLIST_AUTOSIZE_USEHEADER );
		ContentsListBox->InsertColumn(11,"Correct Mag. Distortion?", wxLIST_FORMAT_LEFT, wxLIST_AUTOSIZE_USEHEADER );
		ContentsListBox->InsertColumn(12,"Dist. angle", wxLIST_FORMAT_LEFT, wxLIST_AUTOSIZE_USEHEADER );
		ContentsListBox->InsertColumn(13,"Dist. major scale", wxLIST_FORMAT_LEFT, wxLIST_AUTOSIZE_USEHEADER );
		ContentsListBox->InsertColumn(14,"Dist. minor scale", wxLIST_FORMAT_LEFT, wxLIST_AUTOSIZE_USEHEADER );

/*
		for (long counter = 0; counter < all_groups_list->groups[selected_group].number_of_members; counter++)
		{
			ContentsListBox->InsertItem(counter, wxString::Format(wxT("%i"), all_assets_list->ReturnMovieAssetPointer(all_groups_list->ReturnGroupMember(selected_group, counter))->asset_id, counter));
			ContentsListBox->SetItem(counter, 1, all_assets_list->ReturnAssetPointer(all_groups_list->ReturnGroupMember(selected_group, counter))->ReturnShortNameString());
			ContentsListBox->SetItem(counter, 2, wxString::Format(wxT("%i"),all_assets_list->ReturnMovieAssetPointer(all_groups_list->ReturnGroupMember(selected_group, counter))->x_size));
			ContentsListBox->SetItem(counter, 3, wxString::Format(wxT("%i"), all_assets_list->ReturnMovieAssetPointer(all_groups_list->ReturnGroupMember(selected_group, counter))->y_size));
			ContentsListBox->SetItem(counter, 4, wxString::Format(wxT("%i"), all_assets_list->ReturnMovieAssetPointer(all_groups_list->ReturnGroupMember(selected_group, counter))->number_of_frames));
			ContentsListBox->SetItem(counter, 5, wxString::Format(wxT("%.4f"),all_assets_list->ReturnMovieAssetPointer(all_groups_list->ReturnGroupMember(selected_group, counter))->pixel_size));
			ContentsListBox->SetItem(counter, 6, wxString::Format(wxT("%.3f"), all_assets_list->ReturnMovieAssetPointer(all_groups_list->ReturnGroupMember(selected_group, counter))->dose_per_frame));
			ContentsListBox->SetItem(counter, 7, wxString::Format(wxT("%.2f"), all_assets_list->ReturnMovieAssetPointer(all_groups_list->ReturnGroupMember(selected_group, counter))->spherical_aberration));
			ContentsListBox->SetItem(counter, 8, wxString::Format(wxT("%.2f"), all_assets_list->ReturnMovieAssetPointer(all_groups_list->ReturnGroupMember(selected_group, counter))->microscope_voltage));


		}*/
}

wxString MyMovieAssetPanel::ReturnItemText(long item, long column) const
{
	switch(column)
	{
	    case 0  :
	    	return wxString::Format(wxT("%i"), all_assets_list->ReturnMovieAssetPointer(all_groups_list->ReturnGroupMember(selected_group, item))->asset_id);
	       break;
	    case 1  :
	    	return all_assets_list->ReturnAssetPointer(all_groups_list->ReturnGroupMember(selected_group, item))->asset_name;
	       break;
	    case 2  :
	    	return wxString::Format(wxT("%i"),all_assets_list->ReturnMovieAssetPointer(all_groups_list->ReturnGroupMember(selected_group, item))->x_size);
	       break;
	    case 3  :
	    	return wxString::Format(wxT("%i"), all_assets_list->ReturnMovieAssetPointer(all_groups_list->ReturnGroupMember(selected_group, item))->y_size);
	       break;
	    case 4  :
	    	return wxString::Format(wxT("%i"), all_assets_list->ReturnMovieAssetPointer(all_groups_list->ReturnGroupMember(selected_group, item))->number_of_frames);
	       break;
	    case 5  :
	    	return wxString::Format(wxT("%.4f"),all_assets_list->ReturnMovieAssetPointer(all_groups_list->ReturnGroupMember(selected_group, item))->pixel_size);
		   break;
	    case 6  :
	    	return wxString::Format(wxT("%.3f"), all_assets_list->ReturnMovieAssetPointer(all_groups_list->ReturnGroupMember(selected_group, item))->dose_per_frame);
		   break;
	    case 7  :
	    	return wxString::Format(wxT("%.2f"), all_assets_list->ReturnMovieAssetPointer(all_groups_list->ReturnGroupMember(selected_group, item))->spherical_aberration);
	    	break;
	    case 8  :
	    	return wxString::Format(wxT("%.2f"), all_assets_list->ReturnMovieAssetPointer(all_groups_list->ReturnGroupMember(selected_group, item))->microscope_voltage);
	    	break;
	    case 9  :
	    	return wxFileName(all_assets_list->ReturnMovieAssetPointer(all_groups_list->ReturnGroupMember(selected_group, item))->gain_filename).GetFullName();
	    	break;
	    case 10:
	    	return wxString::Format(wxT("%.3f"), all_assets_list->ReturnMovieAssetPointer(all_groups_list->ReturnGroupMember(selected_group, item))->output_binning_factor);
	    case 11:
	    	if ( all_assets_list->ReturnMovieAssetPointer(all_groups_list->ReturnGroupMember(selected_group, item))->correct_mag_distortion == true) return "Yes";
	    	else return "No";
	    case 12:
	    	return wxString::Format(wxT("%.2f"), all_assets_list->ReturnMovieAssetPointer(all_groups_list->ReturnGroupMember(selected_group, item))->mag_distortion_angle);
	    case 13:
	    	return wxString::Format(wxT("%.3f"), all_assets_list->ReturnMovieAssetPointer(all_groups_list->ReturnGroupMember(selected_group, item))->mag_distortion_major_scale);
	    case 14:
	    	return wxString::Format(wxT("%.3f"), all_assets_list->ReturnMovieAssetPointer(all_groups_list->ReturnGroupMember(selected_group, item))->mag_distortion_minor_scale);
	    default :
	       MyPrintWithDetails("Error, asking for column (%li) which does not exist", column);
	       return "";
	}
}

void MyMovieAssetPanel::ImportAssetClick( wxCommandEvent& event )
{

	MyMovieImportDialog *import_dialog = new MyMovieImportDialog(this);
	import_dialog->ShowModal();

}

double MyMovieAssetPanel::ReturnAssetPixelSize(long wanted_asset)
{
	return all_assets_list->ReturnMovieAssetPointer(wanted_asset)->pixel_size;
}

double MyMovieAssetPanel::ReturnAssetAccelerationVoltage(long wanted_asset)
{
	return all_assets_list->ReturnMovieAssetPointer(wanted_asset)->microscope_voltage;
}

double MyMovieAssetPanel::ReturnAssetDosePerFrame(long wanted_asset)
{
	return all_assets_list->ReturnMovieAssetPointer(wanted_asset)->dose_per_frame;

}

float MyMovieAssetPanel::ReturnAssetSphericalAbberation(long wanted_asset)
{
	return all_assets_list->ReturnMovieAssetPointer(wanted_asset)->spherical_aberration;

}

wxString MyMovieAssetPanel::ReturnAssetGainFilename(long wanted_asset)
{
	return wxString(all_assets_list->ReturnMovieAssetPointer(wanted_asset)->gain_filename);
}

float MyMovieAssetPanel::ReturnAssetBinningFactor(long wanted_asset)
{
	return all_assets_list->ReturnMovieAssetPointer(wanted_asset)->output_binning_factor;
}

int MyMovieAssetPanel::ReturnAssetID(long wanted_asset)
{
	return all_assets_list->ReturnMovieAssetPointer(wanted_asset)->asset_id;
}

bool MyMovieAssetPanel::ReturnCorrectMagDistortion(long wanted_asset)
{
	return all_assets_list->ReturnMovieAssetPointer(wanted_asset)->correct_mag_distortion;
}

float MyMovieAssetPanel::ReturnMagDistortionAngle(long wanted_asset)
{
	return all_assets_list->ReturnMovieAssetPointer(wanted_asset)->mag_distortion_angle;
}

float MyMovieAssetPanel::ReturnMagDistortionMajorScale(long wanted_asset)
{
	return all_assets_list->ReturnMovieAssetPointer(wanted_asset)->mag_distortion_major_scale;
}

float MyMovieAssetPanel::ReturnMagDistortionMinorScale(long wanted_asset)
{
	return all_assets_list->ReturnMovieAssetPointer(wanted_asset)->mag_distortion_minor_scale;
}

double MyMovieAssetPanel::ReturnAssetPreExposureAmount(long wanted_asset)
{
	return 0.0; // FIX THIS!
}
