//#include "../core/core_headers.h"
#include "../core/gui_core_headers.h"

MyImageAssetPanel::MyImageAssetPanel( wxWindow* parent )
:
MyAssetParentPanel( parent )
{
	Label0Title->SetLabel("Name : ");
	Label1Title->SetLabel("I.D. : ");
	Label2Title->SetLabel("Parent Movie I.D. : ");
	Label3Title->SetLabel("Movie Alignment I.D. : ");
	Label4Title->SetLabel("X Size : ");
	Label5Title->SetLabel("Y Size : ");
	Label6Title->SetLabel("Pixel Size : ");
	Label7Title->SetLabel("Voltage : ");
	Label8Title->SetLabel("Cs : ");
	Label9Title->SetLabel("");

	AssetTypeText->SetLabel("Images");

	NewFromParentButton->SetLabel("New from\nmovie group");
	NewFromParentButton->Enable(true);

	all_groups_list->groups[0].SetName("All Images");
	all_assets_list = new ImageAssetList;
	FillGroupList();
	FillContentsList();
}

MyImageAssetPanel::~MyImageAssetPanel()
{
	delete all_assets_list;
}

void MyImageAssetPanel::UpdateInfo()
{
	if (selected_content >= 0 && selected_group >= 0 && all_groups_list->groups[selected_group].number_of_members > 0)
	{
		Label0Text->SetLabel(all_assets_list->ReturnImageAssetPointer(all_groups_list->ReturnGroupMember(selected_group, selected_content))->asset_name);
		Label1Text->SetLabel(wxString::Format(wxT("%i"), all_assets_list->ReturnImageAssetPointer(all_groups_list->ReturnGroupMember(selected_group, selected_content))->asset_id));
		Label2Text->SetLabel(wxString::Format(wxT("%i"), all_assets_list->ReturnImageAssetPointer(all_groups_list->ReturnGroupMember(selected_group, selected_content))->parent_id));
		Label3Text->SetLabel(wxString::Format(wxT("%i"), all_assets_list->ReturnImageAssetPointer(all_groups_list->ReturnGroupMember(selected_group, selected_content))->alignment_id));
		Label4Text->SetLabel(wxString::Format(wxT("%i px"), all_assets_list->ReturnImageAssetPointer(all_groups_list->ReturnGroupMember(selected_group, selected_content))->x_size));
		Label5Text->SetLabel(wxString::Format(wxT("%i px"), all_assets_list->ReturnImageAssetPointer(all_groups_list->ReturnGroupMember(selected_group, selected_content))->y_size));
		Label6Text->SetLabel(wxString::Format(wxT("%.2f Å"), all_assets_list->ReturnImageAssetPointer(all_groups_list->ReturnGroupMember(selected_group, selected_content))->pixel_size));
		Label7Text->SetLabel(wxString::Format(wxT("%.2f kV"), all_assets_list->ReturnImageAssetPointer(all_groups_list->ReturnGroupMember(selected_group, selected_content))->microscope_voltage));
		Label8Text->SetLabel(wxString::Format(wxT("%.2f mm"), all_assets_list->ReturnImageAssetPointer(all_groups_list->ReturnGroupMember(selected_group, selected_content))->spherical_aberration));
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
		Label7Text->SetLabel("-");
		Label8Text->SetLabel("-");
		Label9Text->SetLabel("");
	}

}

bool MyImageAssetPanel::IsFileAnAsset(wxFileName file_to_check)
{
	if (reinterpret_cast <ImageAssetList*>  (all_assets_list)->FindFile(file_to_check) == -1) return false;
	else return true;
}

ImageAsset* MyImageAssetPanel::ReturnAssetPointer(long wanted_asset)
{
	return all_assets_list->ReturnImageAssetPointer(wanted_asset);
}

void MyImageAssetPanel::RemoveAssetFromDatabase(long wanted_asset)
{
	main_frame->current_project.database.ExecuteSQL(wxString::Format("DELETE FROM IMAGE_ASSETS WHERE IMAGE_ASSET_ID=%i", all_assets_list->ReturnAssetID(wanted_asset)).ToUTF8().data());
	all_assets_list->RemoveAsset(wanted_asset);
}

void MyImageAssetPanel::RemoveFromGroupInDatabase(int wanted_group_id, int wanted_asset_id)

{
	main_frame->current_project.database.ExecuteSQL(wxString::Format("DELETE FROM IMAGE_GROUP_%i WHERE IMAGE_ASSET_ID=%i", wanted_group_id, wanted_asset_id).ToUTF8().data());
}

void MyImageAssetPanel::InsertGroupMemberToDatabase(int wanted_group, int wanted_asset)
{
	MyDebugAssertTrue(wanted_group > 0 && wanted_group < all_groups_list->number_of_groups, "Requesting a group (%i) that doesn't exist!", wanted_group);
	MyDebugAssertTrue(wanted_asset >= 0 && wanted_asset < all_assets_list->number_of_assets, "Requesting an image (%i) that doesn't exist!", wanted_asset);

	main_frame->current_project.database.InsertOrReplace(wxString::Format("IMAGE_GROUP_%i", ReturnGroupID(wanted_group)).ToUTF8().data(), "ii", "MEMBER_NUMBER", "IMAGE_ASSET_ID", ReturnGroupSize(wanted_group), ReturnGroupMemberID(wanted_group, wanted_asset));
}

void  MyImageAssetPanel::RemoveAllFromDatabase()
{
	for (long counter = 1; counter < all_groups_list->number_of_groups; counter++)
	{
		main_frame->current_project.database.ExecuteSQL(wxString::Format("DROP TABLE IMAGE_GROUP_%i", all_groups_list->groups[counter].id).ToUTF8().data());
	}

	main_frame->current_project.database.ExecuteSQL("DROP TABLE IMAGE_GROUP_LIST");
	main_frame->current_project.database.CreateTable("IMAGE_GROUP_LIST", "pti", "GROUP_ID", "GROUP_NAME", "LIST_ID" );

}

void MyImageAssetPanel::RemoveAllGroupMembersFromDatabase(int wanted_group_id)
{
	main_frame->current_project.database.ExecuteSQL(wxString::Format("DROP TABLE IMAGE_GROUP_%i", wanted_group_id).ToUTF8().data());
	main_frame->current_project.database.CreateTable(wxString::Format("IMAGE_GROUP_%i", wanted_group_id).ToUTF8().data(), "ii", "MEMBER_NUMBER", "IMAGE_ASSET_ID");
}

void MyImageAssetPanel::AddGroupToDatabase(int wanted_group_id, const char * wanted_group_name, int wanted_list_id)
{
	main_frame->current_project.database.InsertOrReplace("IMAGE_GROUP_LIST", "iti", "GROUP_ID", "GROUP_NAME", "LIST_ID", wanted_group_id, wanted_group_name, wanted_list_id);
	main_frame->current_project.database.CreateTable(wxString::Format("IMAGE_GROUP_%i", wanted_list_id).ToUTF8().data(), "pi", "MEMBER_NUMBER", "IMAGE_ASSET_ID");
}

void MyImageAssetPanel::RemoveGroupFromDatabase(int wanted_group_id)
{
	main_frame->current_project.database.ExecuteSQL(wxString::Format("DROP TABLE IMAGE_GROUP_%i", wanted_group_id).ToUTF8().data());
	main_frame->current_project.database.ExecuteSQL(wxString::Format("DELETE FROM IMAGE_GROUP_LIST WHERE GROUP_ID=%i", wanted_group_id));
}

void MyImageAssetPanel::RenameGroupInDatabase(int wanted_group_id, const char *wanted_name)
{
	wxString sql_command = wxString::Format("UPDATE IMAGE_GROUP_LIST SET GROUP_NAME='%s' WHERE GROUP_ID=%i", wanted_name, wanted_group_id);
	main_frame->current_project.database.ExecuteSQL(sql_command.ToUTF8().data());

}

void MyImageAssetPanel::RenameAsset(long wanted_asset, wxString wanted_name)
{
	all_assets_list->ReturnImageAssetPointer(wanted_asset)->asset_name = wanted_name;
	wxString sql_command = wxString::Format("UPDATE IMAGE_ASSETS SET NAME='%s' WHERE IMAGE_ASSET_ID=%i", wanted_name, all_assets_list->ReturnImageAssetPointer(wanted_asset)->asset_id);
	main_frame->current_project.database.ExecuteSQL(sql_command.ToUTF8().data());

}

void MyImageAssetPanel::ImportAllFromDatabase()
{
	int counter;
	ImageAsset temp_asset;
	AssetGroup temp_group;

	all_assets_list->RemoveAll();
	all_groups_list->RemoveAll();

	// First all assets..

	main_frame->current_project.database.BeginAllImageAssetsSelect();

	while (main_frame->current_project.database.last_return_code == SQLITE_ROW)
	{
		temp_asset = main_frame->current_project.database.GetNextImageAsset();
		AddAsset(&temp_asset);
	}

	main_frame->current_project.database.EndAllImageAssetsSelect();

	// Now the groups..

	main_frame->current_project.database.BeginAllImageGroupsSelect();

	while (main_frame->current_project.database.last_return_code == SQLITE_ROW)
	{
		temp_group = main_frame->current_project.database.GetNextImageGroup();

		// the members of this group are referenced by asset id's, we need to translate this to array position..

		for (counter = 0; counter < temp_group.number_of_members; counter++)
		{
			temp_group.members[counter] = all_assets_list->ReturnArrayPositionFromID(temp_group.members[counter]);
		}

		all_groups_list->AddGroup(&temp_group);
		if (temp_group.id > current_group_number) current_group_number = temp_group.id;
	}

	main_frame->current_project.database.EndAllImageGroupsSelect();
	FillGroupList();
	FillContentsList();
}

void MyImageAssetPanel::FillAssetSpecificContentsList()
{

		ContentsListBox->InsertColumn(0, "I.D.", wxLIST_FORMAT_LEFT,  wxLIST_AUTOSIZE_USEHEADER );
		ContentsListBox->InsertColumn(1, "Name", wxLIST_FORMAT_LEFT,  wxLIST_AUTOSIZE_USEHEADER );
		ContentsListBox->InsertColumn(2, "Parent I.D.", wxLIST_FORMAT_LEFT,  wxLIST_AUTOSIZE_USEHEADER );
		ContentsListBox->InsertColumn(3, "Align. I.D.", wxLIST_FORMAT_LEFT,  wxLIST_AUTOSIZE_USEHEADER );
		ContentsListBox->InsertColumn(4, "X Size", wxLIST_FORMAT_LEFT,  wxLIST_AUTOSIZE_USEHEADER );
		ContentsListBox->InsertColumn(5, "Y Size", wxLIST_FORMAT_LEFT,  wxLIST_AUTOSIZE_USEHEADER );
		//ContentsListBox->InsertColumn(4, "No. frames", wxLIST_FORMAT_LEFT,  wxLIST_AUTOSIZE_USEHEADER );
		ContentsListBox->InsertColumn(6, "Pixel size", wxLIST_FORMAT_LEFT,  wxLIST_AUTOSIZE_USEHEADER );
		//ContentsListBox->InsertColumn(6, "Exp. per frame", wxLIST_FORMAT_LEFT,  wxLIST_AUTOSIZE_USEHEADER );
		ContentsListBox->InsertColumn(7, "Cs", wxLIST_FORMAT_LEFT,  wxLIST_AUTOSIZE_USEHEADER );
		ContentsListBox->InsertColumn(8, "Voltage", wxLIST_FORMAT_LEFT,  wxLIST_AUTOSIZE_USEHEADER );


/*		for (long counter = 0; counter < all_groups_list->groups[selected_group].number_of_members; counter++)
		{
			ContentsListBox->InsertItem(counter, wxString::Format(wxT("%i"), all_assets_list->ReturnImageAssetPointer(all_groups_list->ReturnGroupMember(selected_group, counter))->asset_id, counter));
			ContentsListBox->SetItem(counter, 1, all_assets_list->ReturnAssetPointer(all_groups_list->ReturnGroupMember(selected_group, counter))->ReturnShortNameString());
			ContentsListBox->SetItem(counter, 2, wxString::Format(wxT("%i"), all_assets_list->ReturnImageAssetPointer(all_groups_list->ReturnGroupMember(selected_group, counter))->parent_id));
			ContentsListBox->SetItem(counter, 3, wxString::Format(wxT("%i"), all_assets_list->ReturnImageAssetPointer(all_groups_list->ReturnGroupMember(selected_group, counter))->alignment_id));
			ContentsListBox->SetItem(counter, 4, wxString::Format(wxT("%i"),all_assets_list->ReturnImageAssetPointer(all_groups_list->ReturnGroupMember(selected_group, counter))->x_size));
			ContentsListBox->SetItem(counter, 5, wxString::Format(wxT("%i"), all_assets_list->ReturnImageAssetPointer(all_groups_list->ReturnGroupMember(selected_group, counter))->y_size));
		//	ContentsListBox->SetItem(counter, 4, wxString::Format(wxT("%i"), all_assets_list->ReturnMovieAssetPointer(all_groups_list->ReturnGroupMember(selected_group, counter))->number_of_frames));
			ContentsListBox->SetItem(counter, 6, wxString::Format(wxT("%.3f"),all_assets_list->ReturnImageAssetPointer(all_groups_list->ReturnGroupMember(selected_group, counter))->pixel_size));
//			ContentsListBox->SetItem(counter, 6, wxString::Format(wxT("%.3f"), all_assets_list->ReturnMovieAssetPointer(all_groups_list->ReturnGroupMember(selected_group, counter))->dose_per_frame));
			ContentsListBox->SetItem(counter, 7, wxString::Format(wxT("%.2f"), all_assets_list->ReturnImageAssetPointer(all_groups_list->ReturnGroupMember(selected_group, counter))->spherical_aberration));
			ContentsListBox->SetItem(counter, 8, wxString::Format(wxT("%.2f"), all_assets_list->ReturnImageAssetPointer(all_groups_list->ReturnGroupMember(selected_group, counter))->microscope_voltage));


		}*/
}

wxString MyImageAssetPanel::ReturnItemText(long item, long column) const
{
	switch(column)
	{
	    case 0  :
	    	return wxString::Format(wxT("%i"), all_assets_list->ReturnImageAssetPointer(all_groups_list->ReturnGroupMember(selected_group, item))->asset_id);
	       break;
	    case 1  :
	    	return all_assets_list->ReturnImageAssetPointer(all_groups_list->ReturnGroupMember(selected_group, item))->asset_name;
	       break;
	    case 2  :
	    	return wxString::Format(wxT("%i"), all_assets_list->ReturnImageAssetPointer(all_groups_list->ReturnGroupMember(selected_group, item))->parent_id);
	       break;
	    case 3  :
	    	return wxString::Format(wxT("%i"), all_assets_list->ReturnImageAssetPointer(all_groups_list->ReturnGroupMember(selected_group, item))->alignment_id);
	       break;
	    case 4  :
	    	return wxString::Format(wxT("%i"),all_assets_list->ReturnImageAssetPointer(all_groups_list->ReturnGroupMember(selected_group, item))->x_size);
	       break;
	    case 5  :
	    	return wxString::Format(wxT("%i"), all_assets_list->ReturnImageAssetPointer(all_groups_list->ReturnGroupMember(selected_group, item))->y_size);
		   break;
	    case 6  :
	    	return wxString::Format(wxT("%.3f"),all_assets_list->ReturnImageAssetPointer(all_groups_list->ReturnGroupMember(selected_group, item))->pixel_size);
		   break;
	    case 7  :
	    	return wxString::Format(wxT("%.2f"), all_assets_list->ReturnImageAssetPointer(all_groups_list->ReturnGroupMember(selected_group, item))->spherical_aberration);
	    	break;
	    case 8  :
	    	return wxString::Format(wxT("%.2f"), all_assets_list->ReturnImageAssetPointer(all_groups_list->ReturnGroupMember(selected_group, item))->microscope_voltage);
	    	break;

	    default :
	       MyPrintWithDetails("Error, asking for column (%li) which does not exist", column);
	       return "";
	}
}

void MyImageAssetPanel::ImportAssetClick( wxCommandEvent& event )
{

	MyImageImportDialog *import_dialog = new MyImageImportDialog(this);
	import_dialog->ShowModal();

}

void MyImageAssetPanel::NewFromParentClick( wxCommandEvent & event )
{
	MyDebugAssertTrue(movie_asset_panel->ReturnNumberOfGroups() > 1,"No movie groups to work from. Button should have been disabled");

	// Ask the user which movie group they want to "copy"
	// We can only allow them to copy a group where each member has a corresponding image (output by unblur, with the corresponding parent_id)
	wxArrayString my_choices;
	wxArrayInt choice_group_numbers;
	bool all_movies_have_children[movie_asset_panel->ReturnNumberOfGroups()-1];
	for (long group_counter = 1; group_counter < movie_asset_panel->ReturnNumberOfGroups(); group_counter ++ )
	{
		// check whether all of the members of this group of movies have image children
		int movie_ids[movie_asset_panel->ReturnGroupSize(group_counter)];
		bool movie_has_child[movie_asset_panel->ReturnGroupSize(group_counter)];
		for (long counter = 0; counter < movie_asset_panel->ReturnGroupSize(group_counter); counter ++ )
		{
			movie_ids[counter] = movie_asset_panel->ReturnGroupMemberID(group_counter, counter);
			movie_has_child[counter] = false;
			//
		}
		int current_parent_id;
		for (long image_counter = 0; image_counter < ReturnNumberOfAssets(); image_counter ++ )
		{
			current_parent_id = ReturnAssetPointer(image_counter)->parent_id;
			for (long movie_counter = 0; movie_counter < movie_asset_panel->ReturnGroupSize(group_counter); movie_counter ++ )
			{
				if (movie_ids[movie_counter] == current_parent_id) movie_has_child[movie_counter] = true;
			}
		}
		all_movies_have_children[group_counter] = true;
		for (long counter = 0; counter < movie_asset_panel->ReturnGroupSize(group_counter); counter ++ )
		{
			if (! movie_has_child[counter]) all_movies_have_children[group_counter] = false;
		}

		// if all movies in the current group have image asset children, add to the list the user can choose from
		if (all_movies_have_children[group_counter])
		{
			my_choices.Add(movie_asset_panel->ReturnGroupName(group_counter));
			choice_group_numbers.Add(group_counter);
		}
	}

	// Generate a dialog for the user
	wxSingleChoiceDialog	*group_choice = new wxSingleChoiceDialog(this, "Copy which group of movies?", "Select Group", my_choices);

	// Assuming the user chose a group, generate a new image group
	int selected_group_number;
	if (group_choice->ShowModal() ==  wxID_OK)
	{
		wxPrintf("Would add group now\n");
		selected_group_number = choice_group_numbers.Item(group_choice->GetSelection());
		wxPrintf("Using movie group %i as parent, which has %li members\n",selected_group_number,movie_asset_panel->ReturnGroupSize(selected_group_number));

		// Create a new group
		{
			wxString new_group_name = movie_asset_panel->ReturnGroupName(selected_group_number);
			current_group_number++;
			all_groups_list->AddGroup(new_group_name);
			all_groups_list->groups[all_groups_list->number_of_groups-1].id  = current_group_number;
			AddGroupToDatabase(current_group_number, new_group_name, current_group_number);
			FillGroupList();
			DirtyGroups();
		}

		// Work out who belongs in that group
		// TODO: progress bar
		bool image_has_parent_in_group[ReturnNumberOfAssets()];
		for (long image_counter = 0; image_counter < ReturnNumberOfAssets(); image_counter ++ ) { image_has_parent_in_group[image_counter] = false; }
		MovieAsset *current_movie_asset;
		ImageAsset *current_image_asset;
		for (long movie_counter = 0; movie_counter < movie_asset_panel->ReturnGroupSize(selected_group_number); movie_counter ++ )
		{
			current_movie_asset = movie_asset_panel->ReturnAssetPointer(movie_asset_panel->ReturnGroupMember(selected_group_number, movie_counter));
			for (long image_counter = 0; image_counter < ReturnNumberOfAssets(); image_counter ++ )
			{
				current_image_asset = all_assets_list->ReturnImageAssetPointer(image_counter);
				if (current_image_asset->parent_id == current_movie_asset->asset_id) image_has_parent_in_group[image_counter] = true;
			}
		}

		// Add members to the new group
		// TODO: progress bar
		for (long image_counter = 0; image_counter < ReturnNumberOfAssets(); image_counter ++ )
		{
			if (image_has_parent_in_group[image_counter])
			{
				AddArrayItemToGroup(all_groups_list->number_of_groups-1,image_counter);
			}
		}

		FillGroupList();
		FillContentsList();
		DirtyGroups();
		main_frame->RecalculateAssetBrowser();

	}

	group_choice->Destroy();

}
