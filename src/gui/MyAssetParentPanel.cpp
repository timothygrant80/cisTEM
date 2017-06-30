//#include "../core/core_headers.h"
#include "../core/gui_core_headers.h"

extern MyMainFrame *main_frame;
extern wxTreeItemId Tree_Root;

MyAssetParentPanel::MyAssetParentPanel( wxWindow* parent )
:
AssetParentPanel( parent )
{

	all_groups_list = new AssetGroupList;
	selected_group = 0;
	selected_content = -1;
	highlighted_item = -1;

	current_asset_number = 0;
	current_group_number = 0;

	should_veto_motion = true;

	GroupListBox->SetDropTarget(new GroupDropTarget(GroupListBox, this));

	bool should_veto_motion = true;
	name_is_being_edited = false;
	is_dirty = false;
}

MyAssetParentPanel::~MyAssetParentPanel()
{
	delete all_groups_list;
}




void MyAssetParentPanel::CompletelyRemoveAsset(long wanted_asset)
{
	long counter;
	long found_position;

	// first off remove it from the asset list..

	RemoveAssetFromDatabase(wanted_asset);

	RemoveAssetFromGroups(wanted_asset);
}


void MyAssetParentPanel::RemoveAssetFromGroups(long wanted_asset)
{
	long counter;
	long found_position;


	// we never actually want to remove members from the all_assets group, but we need to make note that there is 1 less asset.

	all_groups_list->groups[0].number_of_members--;

	// if the removed asset was referenced in other groups we need to remove it..

	for (counter = 1; counter < all_groups_list->number_of_groups; counter++)
	{
		found_position = all_groups_list->groups[counter].FindMember(wanted_asset);

		if (found_position != -1)
		{
			RemoveFromGroupInDatabase(ReturnGroupID(counter), ReturnGroupMemberID(counter, found_position));
		}

	}

	all_groups_list->RemoveAssetFromExtraGroups(wanted_asset);

	// because we have shifted the asset list up, any item with a number greater than this item needs to be subtracted by 1

	all_groups_list->ShiftMembersDueToAssetRemoval(wanted_asset);

	DirtyGroups();
}

void MyAssetParentPanel::OnDisplayButtonClick( wxCommandEvent& event )
{
	DisplaySelectedItems();
}

void MyAssetParentPanel::DisplaySelectedItems()
{
	if (DisplayButton->IsEnabled() == true)
	{
		wxString execution_command = wxStandardPaths::Get().GetExecutablePath();
		execution_command = execution_command.BeforeLast('/');
		execution_command += "/display";

		// get all selected..

		long item = -1;

		for ( ;; )
		{
			item = ContentsListBox->GetNextItem(item, wxLIST_NEXT_ALL, wxLIST_STATE_SELECTED);
			if ( item == -1 )
			break;

			execution_command += " ";
			execution_command += all_assets_list->ReturnAssetFullFilename(all_groups_list->ReturnGroupMember(selected_group, item));

		}

		execution_command += "&";
		//wxPrintf("Launching %s\n", execution_command);
		system(execution_command.ToUTF8().data());
	}
}

void MyAssetParentPanel::RemoveAssetClick( wxCommandEvent& event )
{
	// How many assets are selected?

	long number_selected = ContentsListBox->GetSelectedItemCount();
	long selected_asset;
	long number_removed = 0;
	long removed_counter;

	long item;
	long adjusted_item;

	long *already_removed = new long[number_selected];

	long currently_selected_group = selected_group;

	if (number_selected > 0)
	{
		if (selected_group == 0)
		{
			wxMessageDialog *check_dialog = new wxMessageDialog(this, "This will remove the selected assets from your ENTIRE project! Are you sure you want to continue?", "Are you sure?", wxYES_NO);

			if (check_dialog->ShowModal() ==  wxID_YES)
			{
				item = -1;

				for ( ;; )
				{
					item = ContentsListBox->GetNextItem(item, wxLIST_NEXT_ALL, wxLIST_STATE_SELECTED);

					if ( item == -1 )
						break;

					adjusted_item = item;

					// the problem is, that depending on what has been removed - the item from the contentslistbox may not point to the
					// correct place after the first one has been removed.

					for (removed_counter = 0; removed_counter < number_removed; removed_counter++)
					{
						if (already_removed[removed_counter] < item) adjusted_item--;
					}

					CompletelyRemoveAsset(adjusted_item);
					already_removed[number_removed] = item;
					number_removed++;


				}
			}
		}
		else
		{
			long item = -1;

			main_frame->current_project.database.Begin();
			for ( ;; )
			{
					item = ContentsListBox->GetNextItem(item, wxLIST_NEXT_ALL, wxLIST_STATE_SELECTED);
					if ( item == -1 )
					break;

					adjusted_item = item;

					// the problem is, that depending on what has been removed - the item from the contentslistbox may not point to the
					// correct place after the first one has been removed.

					for (removed_counter = 0; removed_counter < number_removed; removed_counter++)
					{
						if (already_removed[removed_counter] < item) adjusted_item--;
					}

					// remove from the database..

					RemoveFromGroupInDatabase(ReturnGroupID(selected_group), ReturnGroupMemberID(selected_group, adjusted_item));

					// now from the gui

					all_groups_list->groups[selected_group].RemoveMember(adjusted_item);

					already_removed[number_removed] = item;
					number_removed++;
			}

			main_frame->current_project.database.Commit();

		}

		FillGroupList();
		SetSelectedGroup(currently_selected_group);
		FillContentsList();
		main_frame->RecalculateAssetBrowser();
		UpdateInfo();
		DirtyGroups();
	}

	delete [] already_removed;

}

void MyAssetParentPanel::AddContentItemToGroup(long wanted_group, long wanted_content_item)
{
	MyDebugAssertTrue(wanted_group > 0 && wanted_group < all_groups_list->number_of_groups, "Requesting a group (%li) that doesn't exist!", wanted_group)
	MyDebugAssertTrue(wanted_content_item >= 0 && wanted_content_item < all_assets_list->number_of_assets, "Requesting an asset(%li) that doesn't exist!", wanted_content_item)

	long selected_asset = all_groups_list->ReturnGroupMember(selected_group, wanted_content_item);

	AddArrayItemToGroup(wanted_group, selected_asset);
}

void MyAssetParentPanel::AddArrayItemToGroup(long wanted_group, long wanted_array_item)
{
	MyDebugAssertTrue(wanted_group > 0 && wanted_group < all_groups_list->number_of_groups, "Requesting a group (%li) that doesn't exist!", wanted_group)
	MyDebugAssertTrue(wanted_array_item >= 0 && wanted_array_item < all_assets_list->number_of_assets, "Requesting an asset(%li) that doesn't exist!", wanted_array_item)

	if (all_groups_list->groups[wanted_group].FindMember(wanted_array_item) == -1)
	{
		all_groups_list->groups[wanted_group].AddMember(wanted_array_item);
		InsertGroupMemberToDatabase(wanted_group, all_groups_list->groups[wanted_group].number_of_members - 1);
		DirtyGroups();
	}
}

void MyAssetParentPanel::AddArrayofArrayItemsToGroup(long wanted_group, wxArrayLong *array_of_wanted_items, OneSecondProgressDialog *progress_dialog)
{
	MyDebugAssertTrue(wanted_group > 0 && wanted_group < all_groups_list->number_of_groups, "Requesting a group (%li) that doesn't exist!", wanted_group)

	wxArrayLong array_to_add_to_database; // copied so that duplicates are not added to the database..

	for (long counter = 0; counter < array_of_wanted_items->GetCount(); counter++)
	{
		MyDebugAssertTrue(array_of_wanted_items->Item(counter) >= 0 && array_of_wanted_items->Item(counter) < all_assets_list->number_of_assets, "Requesting an asset(%li) that doesn't exist!", array_of_wanted_items->Item(counter))
		if (all_groups_list->groups[wanted_group].FindMember(array_of_wanted_items->Item(counter)) == -1)
		{
			all_groups_list->groups[wanted_group].AddMember(array_of_wanted_items->Item(counter));
			//InsertGroupMemberToDatabase(wanted_group, all_groups_list->groups[wanted_group].number_of_members - 1);
			array_to_add_to_database.Add(array_of_wanted_items->Item(counter));
		}
	}

	InsertArrayofGroupMembersToDatabase(wanted_group, &array_to_add_to_database, progress_dialog);
	DirtyGroups();


}

void MyAssetParentPanel::AddSelectedAssetClick( wxCommandEvent& event )
{
	// How many assets are selected?

	long number_selected = ContentsListBox->GetSelectedItemCount();

		if (number_selected > 0)
		{
			if (all_groups_list->number_of_groups > 1)
			{
				// selected group..

				wxArrayString my_choices;

				for (long counter = 1; counter < ReturnNumberOfGroups(); counter++)
				{
					my_choices.Add(all_groups_list->groups[counter].name);
				}

				wxSingleChoiceDialog	*group_choice = new wxSingleChoiceDialog(this, "Add Selected Assets to which group(s)?", "Select Groups", my_choices);

				if (group_choice->ShowModal() ==  wxID_OK)
				{
					main_frame->current_project.database.Begin();
					long item = -1;

					for ( ;; )
					{
						item = ContentsListBox->GetNextItem(item, wxLIST_NEXT_ALL, wxLIST_STATE_SELECTED);
						if ( item == -1 )
						break;

						AddContentItemToGroup(group_choice->GetSelection() + 1, item);
					}
					main_frame->current_project.database.Commit();
				}

				group_choice->Destroy();
			}

			FillGroupList();
			FillContentsList();
			main_frame->RecalculateAssetBrowser();
			//UpdateInfo();
		}

}

void MyAssetParentPanel::RemoveAllAssetsClick( wxCommandEvent& event )
{
	long counter;

	if (selected_group == 0)
	{
		if (all_assets_list->number_of_assets > 0)
		{
			wxMessageDialog *check_dialog = new wxMessageDialog(this, "This will remove ALL assets and groups from your ENTIRE project! Are you sure you want to continue?", "Are you really really sure?", wxYES_NO);

			if (check_dialog->ShowModal() ==  wxID_YES)
			{
				// delete from database..

				RemoveAllFromDatabase();

				// delete from gui..

				Reset();
			}
		}
	}
	else
	{
		// from the database..

		RemoveAllGroupMembersFromDatabase(ReturnGroupID(selected_group));

		// and from the gui

		all_groups_list->groups[selected_group].RemoveAll();

		FillGroupList();
		FillContentsList();
		//CheckActiveButtons();
		main_frame->RecalculateAssetBrowser();
		DirtyGroups();
	}
}

void MyAssetParentPanel::NewGroupClick( wxCommandEvent& event )
{
	// Add a new Group - called New Group

	current_group_number++;

	all_groups_list->AddGroup("New Group");
	all_groups_list->groups[all_groups_list->number_of_groups - 1].id = current_group_number;

	AddGroupToDatabase(current_group_number, "New Group", current_group_number);

	// Database.

	// How many movies are selected?
/*
	long number_selected = ContentsListBox->GetSelectedItemCount();


	if (number_selected > 0)
	{
			long item = -1;

			for ( ;; )
			{
			    item = ContentsListBox->GetNextItem(item, wxLIST_NEXT_ALL, wxLIST_STATE_SELECTED);
			    if ( item == -1 )
			        break;


			    AddContentItemToGroup(all_groups_list->ReturnNumberOfGroups() - 1, item);
			}
			//UpdateInfo();
	}
*/
	FillGroupList();
	DirtyGroups();
	//SetSelectedGroup(all_groups_list->ReturnNumberOfGroups() - 1);

	main_frame->RecalculateAssetBrowser();
}


void MyAssetParentPanel::RemoveGroupClick( wxCommandEvent& event )
{
	if (selected_group != 0)
	{
		// database..

		RemoveGroupFromDatabase(ReturnGroupID(selected_group));

		// gui

		all_groups_list->RemoveGroup(selected_group);
		SetSelectedGroup(selected_group - 1);
	}

	FillGroupList();
	DirtyGroups();
	main_frame->RecalculateAssetBrowser();
}

void MyAssetParentPanel::RenameGroupClick( wxCommandEvent& event )
{
	MyDebugAssertTrue(selected_group >= 0 && selected_group < all_groups_list->number_of_groups, "Trying to rename an non existent group!");
	GroupListBox->EditLabel(selected_group);
}

void MyAssetParentPanel::InvertGroupClick( wxCommandEvent& event )
{
	//
	int group_to_invert = selected_group;
	int id_of_group_to_invert = all_groups_list->groups[group_to_invert].id;

	// Build bool array to work out which assets currently belong to the group, and invert the logic
	bool *group_membership = new bool[all_assets_list->number_of_assets];

	for (long counter = 0; counter < all_assets_list->number_of_assets; counter++)
	{
		group_membership[counter] = true;
	}
	for (long counter = 0; counter < all_groups_list->groups[group_to_invert].number_of_members; counter ++ )
	{
		group_membership[all_groups_list->groups[group_to_invert].members[counter]] = false;
	}

	/* Alexis old code, I have changed it to the code below which i believe is considerably faster.  Also, It now adds a copy of the group.
	// Create a new group
	current_group_number++;
	all_groups_list->AddGroup(all_groups_list->groups[group_to_invert].name);
	all_groups_list->groups[all_groups_list->number_of_groups - 1].id = current_group_number;
	AddGroupToDatabase(current_group_number,all_groups_list->groups[group_to_invert].name,current_group_number);


	FillGroupList();

	// Add its members
	// TODO: progress bar
	for (long counter = 0; counter < all_assets_list->number_of_assets; counter++)
	{
		if (group_membership[counter])
		{
			AddArrayItemToGroup(all_groups_list->number_of_groups-1, counter);
		}
	}

	// Remove the original group
	if (group_to_invert != 0)
	{
		RemoveGroupFromDatabase(id_of_group_to_invert);

		all_groups_list->RemoveGroup(group_to_invert);

	}*/

	wxArrayLong included_group_members;

	for (long counter = 0; counter < all_assets_list->number_of_assets; counter++)
	{
		if (group_membership[counter])
		{
			included_group_members.Add(counter);
		}
	}

	current_group_number++;
	all_groups_list->AddGroup("Inverse of " + all_groups_list->groups[group_to_invert].name);
	all_groups_list->groups[all_groups_list->number_of_groups - 1].id = current_group_number;
	AddGroupToDatabase(current_group_number,all_groups_list->groups[all_groups_list->number_of_groups - 1].name,current_group_number);

	// add to databse with progressbar

	OneSecondProgressDialog *progress_bar = new OneSecondProgressDialog("Invert", "Inverting group", included_group_members.GetCount(), this, wxPD_APP_MODAL);
	AddArrayofArrayItemsToGroup(all_groups_list->number_of_groups - 1, &included_group_members, progress_bar);

	delete [] group_membership;
	FillGroupList();
	DirtyGroups();
	SetSelectedGroup(all_groups_list->number_of_groups-1);
	progress_bar->Destroy();

}

void MyAssetParentPanel::NewFromParentClick ( wxCommandEvent & event )
{
	MyDebugAssertTrue(false,"Oops, this method should never be called. You need to override it in the panel class");
}

void MyAssetParentPanel::OnGroupActivated( wxListEvent& event )
{
	GroupListBox->EditLabel(event.GetIndex());
}

void MyAssetParentPanel::OnAssetActivated( wxListEvent& event )
{
	DisplaySelectedItems();
}

void MyAssetParentPanel::AddAsset(Asset *asset_to_add)
{
	// Firstly add the asset to the Asset list

	all_assets_list->AddAsset(asset_to_add);
	all_groups_list->AddMemberToGroup(0, all_assets_list->number_of_assets - 1);

	if (asset_to_add->asset_id >= current_asset_number) current_asset_number = asset_to_add->asset_id + 1;

	//FillContentsList();
}

void MyAssetParentPanel::SetSelectedGroup(long wanted_group)
{
	MyDebugAssertTrue(wanted_group >= 0 && wanted_group <= all_groups_list->number_of_groups, "Trying to select a group that doesn't exist!");

	GroupListBox->SetItemState(wanted_group, wxLIST_STATE_SELECTED, wxLIST_STATE_SELECTED);

	if (wanted_group != selected_group)
	{
		selected_group = wanted_group;
		selected_content = 0;
		//FillGroupList();
		FillContentsList();

	}
}

unsigned long MyAssetParentPanel::ReturnNumberOfAssets()
{
	return all_assets_list->number_of_assets;
}

unsigned long MyAssetParentPanel::ReturnNumberOfGroups()
{
	return all_groups_list->number_of_groups;
}

void MyAssetParentPanel::SetGroupName(long wanted_group, wxString wanted_name)
{
	if (all_groups_list->groups[wanted_group].name != wanted_name)
	{
		all_groups_list->groups[wanted_group].name = wanted_name;
		RenameGroupInDatabase(ReturnGroupID(wanted_group), wanted_name);
		DirtyGroups();
	}
}

wxString MyAssetParentPanel::ReturnGroupName(long wanted_group)
{
	return all_groups_list->groups[wanted_group].name;
}

int MyAssetParentPanel::ReturnGroupID(long wanted_group)
{
	return all_groups_list->groups[wanted_group].id;
}

wxString MyAssetParentPanel::ReturnAssetShortFilename(long wanted_asset)
{
	return all_assets_list->ReturnAssetPointer(wanted_asset)->ReturnShortNameString();
}

wxString MyAssetParentPanel::ReturnAssetLongFilename(long wanted_asset)
{
	return all_assets_list->ReturnAssetPointer(wanted_asset)->ReturnFullPathString();
}

long MyAssetParentPanel::ReturnGroupMember(long wanted_group, long wanted_member)
{
	return all_groups_list->groups[wanted_group].members[wanted_member];
}

int MyAssetParentPanel::ReturnGroupMemberID(long wanted_group, long wanted_member)
{
	return all_assets_list->ReturnAssetPointer(all_groups_list->groups[wanted_group].members[wanted_member])->asset_id;
}


long MyAssetParentPanel::ReturnGroupSize(long wanted_group)
{
	return all_groups_list->groups[wanted_group].number_of_members;
}

void MyAssetParentPanel::SizeGroupColumn()
{
	int client_height;
	int client_width;

	int current_width;

	GroupListBox->GetClientSize(&client_width, &client_height);
	GroupListBox->SetColumnWidth(0, wxLIST_AUTOSIZE);

	current_width = GroupListBox->GetColumnWidth(0);

	if (client_width > current_width) GroupListBox->SetColumnWidth(0, client_width);

}

void MyAssetParentPanel::FillGroupList()
{
	//wxColor my_grey(50,50,50);

	wxFont current_font;

	GroupListBox->Freeze();
	GroupListBox->ClearAll();
	GroupListBox->InsertColumn(0, "Files", wxLIST_FORMAT_LEFT, wxLIST_AUTOSIZE);

	GroupListBox->InsertItem(0, all_groups_list->groups[0].name + " (" + wxString::Format(wxT("%li"), all_groups_list->groups[0].number_of_members) + ")", 0);

	current_font = GroupListBox->GetFont();
	current_font.MakeBold();
	GroupListBox->SetItemFont(0, current_font);

	for (long counter = 1; counter < all_groups_list->number_of_groups; counter++)
	{
		GroupListBox->InsertItem(counter, all_groups_list->groups[counter].name + " (" + wxString::Format(wxT("%li"), all_groups_list->groups[counter].number_of_members) + ")" , counter);
	}

	SizeGroupColumn();

	if (selected_group >= 0 && selected_group < all_groups_list->number_of_groups)
	{
		GroupListBox->SetItemState(selected_group, wxLIST_STATE_SELECTED, wxLIST_STATE_SELECTED);
	}
	else
	{
		selected_group = 0;
		GroupListBox->SetItemState(selected_group, wxLIST_STATE_SELECTED, wxLIST_STATE_SELECTED);
	}
	GroupListBox->Thaw();
}

void MyAssetParentPanel::OnBeginEdit( wxListEvent& event )
{
	//wxPrintf("Begin Label = %s\n", event.GetLabel());

	// this is a bit of a hack, as sometimes weird things happen and end edit is called by itself, then
	// the label gets the number added to it.

	name_is_being_edited = true;

	if (event.GetIndex() == 0) event.Veto();
	else
	{
		GroupListBox->SetItemText(event.GetIndex(), all_groups_list->groups[event.GetIndex()].name);
		event.Skip();
	}
}

void MyAssetParentPanel::OnEndEdit( wxListEvent& event )
{

	if (event.GetLabel() == wxEmptyString)
	{
		GroupListBox->SetItemText(event.GetIndex(), all_groups_list->groups[event.GetIndex()].name + " (" + wxString::Format(wxT("%li"), all_groups_list->groups[event.GetIndex()].number_of_members) + ")");
		event.Veto();
	}
	else
	{
		if (name_is_being_edited == true)
		{
			if (all_groups_list->groups[event.GetIndex()].name != event.GetLabel())
			{
				SetGroupName(event.GetIndex(), event.GetLabel());
				event.Veto();
				GroupListBox->SetItemText(event.GetIndex(), all_groups_list->groups[event.GetIndex()].name + " (" + wxString::Format(wxT("%li"), all_groups_list->groups[event.GetIndex()].number_of_members) + ")");
		//		GroupListBox->Refresh();
				//FillGroupList();

			}

			name_is_being_edited = false;
		}
		else event.Veto();
	}


}


void MyAssetParentPanel::RenameAssetClick( wxCommandEvent& event )
{


	MyRenameDialog *rename_dialog = new MyRenameDialog(this);
	wxTextCtrl *text_ctrl_pointer;

	if (rename_dialog->ShowModal() == wxID_OK)
	{
		// do the renaming..

		wxWindowList all_children = rename_dialog->RenameScrollPanel->GetChildren();

		for (long counter = 0; counter <  all_children.GetCount(); counter++)
		{
			if (all_children.Item(counter)->GetData()->GetClassInfo()->GetClassName() == wxString("wxTextCtrl"))
			{
				text_ctrl_pointer = reinterpret_cast <wxTextCtrl *> (all_children.Item(counter)->GetData());

				if (text_ctrl_pointer->GetValue() != wxEmptyString)
				{
					RenameAsset(rename_dialog->selected_assets_array_position[counter], text_ctrl_pointer->GetValue());
				}


			}
		}

		rename_dialog->Destroy();
		is_dirty = true;
	}
}





void MyAssetParentPanel::SizeContentsColumn(int column_number)
{
/*	int header_width, text_width;

	ContentsListBox->SetColumnWidth(column_number, wxLIST_AUTOSIZE);
	text_width = ContentsListBox->GetColumnWidth(column_number);
	ContentsListBox->SetColumnWidth(column_number, wxLIST_AUTOSIZE_USEHEADER);
	header_width = ContentsListBox->GetColumnWidth(column_number);

	if (text_width > header_width) ContentsListBox->SetColumnWidth(column_number, wxLIST_AUTOSIZE);
	*/

	ContentsListBox->SetColumnWidth(column_number,  ContentsListBox->ReturnGuessAtColumnTextWidth(column_number));


}

void MyAssetParentPanel::FillContentsList()
{
	if ( selected_group >= 0  )
	{
		ContentsListBox->Freeze();
		ContentsListBox->ClearAll();

		ContentsListBox->SetItemCount(all_groups_list->groups[selected_group].number_of_members);

		FillAssetSpecificContentsList();

		if (all_groups_list->groups[selected_group].number_of_members > 0) ContentsListBox->RefreshItems(0, all_groups_list->groups[selected_group].number_of_members - 1);

		for (int counter = 0; counter < ContentsListBox->GetColumnCount(); counter++)
		{
			SizeContentsColumn(counter);
		}

		if (selected_content >= 0 && selected_content < all_groups_list->groups[selected_group].number_of_members)
		{
			ContentsListBox->SetItemState(selected_content, wxLIST_STATE_SELECTED, wxLIST_STATE_SELECTED);
		}
		else
		{
			if (all_groups_list->groups[selected_group].number_of_members > 0)
			{
				selected_content = 0;
				ContentsListBox->SetItemState(selected_content, wxLIST_STATE_SELECTED, wxLIST_STATE_SELECTED);
				UpdateInfo();
			}
		}


		ContentsListBox->Thaw();
	}
}


void MyAssetParentPanel::OnGroupFocusChange( wxListEvent& event )
{
	if (event.GetIndex() != selected_group)
	{
		selected_group = event.GetIndex();
		selected_content = -1;
		FillContentsList();
	}

	event.Skip();
}

void MyAssetParentPanel::OnContentsSelected( wxListEvent& event )
{
	selected_content = event.GetIndex();

	//wxPrintf("Selected Group = %li\n", selected_group);
	//wxPrintf("Selected Content = %li\n", selected_content);

	event.Skip();

	UpdateInfo();
	//CheckActiveButtons();

}

void MyAssetParentPanel::OnBeginContentsDrag( wxListEvent& event )
{
	long source_item = event.GetIndex();

	//wxPrintf("Dragging item #%li\n", source_item);

	wxTextDataObject my_data(wxString::Format(wxT("%li"), source_item));
	wxDropSource dragSource( this );
	dragSource.SetData( my_data );
	dragSource.DoDragDrop( true );

	event.Skip();
}

void MyAssetParentPanel::MouseVeto( wxMouseEvent& event )
{
	//Do nothing

}

void MyAssetParentPanel::MouseCheckContentsVeto( wxMouseEvent& event )
{
	VetoInvalidMouse(ContentsListBox, event);

}

void MyAssetParentPanel::MouseCheckGroupsVeto( wxMouseEvent& event )
{
	VetoInvalidMouse(GroupListBox, event);
}

void MyAssetParentPanel::VetoInvalidMouse( wxListCtrl *wanted_list, wxMouseEvent& event )
{
	// Don't allow clicking on anything other than item, to stop the selection bar changing

	int flags;

	if (wanted_list->HitTest(event.GetPosition(), flags)  !=  wxNOT_FOUND)
	{
		should_veto_motion = false;
		event.Skip();
	}
	else should_veto_motion = true;
}

void MyAssetParentPanel::OnMotion(wxMouseEvent& event)
{
	if (should_veto_motion == false) event.Skip();
}

/*
bool MyAssetParentPanel::IsFileAnAsset(wxFileName file_to_check)
{
	if (all_assets_list->FindFile(file_to_check) == -1) return false;
	else return true;

	MyPrintWithDetails("This should never be called!!");
	abort();
}*/

int MyAssetParentPanel::ReturnArrayPositionFromParentID(int wanted_id)
{
	return all_assets_list->ReturnArrayPositionFromParentID(wanted_id);
}

int MyAssetParentPanel::ReturnArrayPositionFromAssetID(int wanted_id)
{
	return all_assets_list->ReturnArrayPositionFromID(wanted_id);
}

int MyAssetParentPanel::ReturnAssetID(int wanted_asset)
{
	return all_assets_list->ReturnAssetID(wanted_asset);
}

wxString MyAssetParentPanel::ReturnAssetName(long wanted_asset)
{
	return all_assets_list->ReturnAssetName(wanted_asset);
}


bool MyAssetParentPanel::DragOverGroups(wxCoord x, wxCoord y)
{
	const wxPoint drop_position(x, y);
	int flags;
	long dropped_group = GroupListBox->HitTest(drop_position, flags);

	if (dropped_group > 0)
	{
		if (dropped_group != highlighted_item)
		{
			if (highlighted_item != -1)
			{
				GroupListBox->SetItemBackgroundColour(highlighted_item, GroupListBox->GetBackgroundColour());

			}

			GroupListBox->SetItemBackgroundColour(dropped_group, *wxLIGHT_GREY);
			highlighted_item = dropped_group;

		}

		return true;
	}
	else
	{
		if (highlighted_item != -1)
		{
			GroupListBox->SetItemBackgroundColour(highlighted_item, GroupListBox->GetBackgroundColour());
			highlighted_item = -1;

		}

		return false;
	}

}



void MyAssetParentPanel::Reset()
{

	all_assets_list->RemoveAll();
	all_groups_list->RemoveAll();


	SetSelectedGroup(0);
	selected_content = -1;

	FillGroupList();
	FillContentsList();
	main_frame->RecalculateAssetBrowser();
	UpdateInfo();
	DirtyGroups();

}

void MyAssetParentPanel::EnableNewFromParentButton()
{
	//NewFromParentButton->Enable(false);
}

void MyAssetParentPanel::OnUpdateUI( wxUpdateUIEvent& event )
{
	if (main_frame->current_project.is_open == false)
	{
		Enable(false);
	}
	else
	{
		Enable(true);

		// We leave it up to the panel itself to decide whether to enable the "New from parent" button
		EnableNewFromParentButton();

		if (selected_group == 0)
		{
			RemoveGroupButton->Enable(false);
			RenameGroupButton->Enable(false);
			RenameGroupButton->Enable(false);
			InvertGroupButton->Enable(false);
		}
		else
		{
			RemoveGroupButton->Enable(true);
			RenameGroupButton->Enable(true);
			InvertGroupButton->Enable(true);
		}

		if (ContentsListBox->GetItemCount() < 1)
		{
			RemoveAllAssetsButton->Enable(false);
			RemoveSelectedAssetButton->Enable(false);
			AddSelectedAssetButton->Enable(false);
			RenameAssetButton->Enable(false);
			DisplayButton->Enable(false);
		}
		else
		{
			RemoveAllAssetsButton->Enable(true);

			if (ContentsListBox->GetSelectedItemCount() < 1)
			{
				RemoveSelectedAssetButton->Enable(false);
				AddSelectedAssetButton->Enable(false);
				RenameAssetButton->Enable(false);
				DisplayButton->Enable(false);
			}
			else
			{
				RemoveSelectedAssetButton->Enable(true);
				RenameAssetButton->Enable(true);
				DisplayButton->Enable(true);

				if (ReturnNumberOfGroups() > 1) AddSelectedAssetButton->Enable(true);
				else AddSelectedAssetButton->Enable(false);
			}
		}

		if (is_dirty == true)
		{
			is_dirty = false;
			FillGroupList();
			FillContentsList();
		}

	}
}


// Drag and Drop

GroupDropTarget::GroupDropTarget(wxListCtrl *owner, MyAssetParentPanel *asset_panel)
{
	my_owner = owner;
	my_panel = asset_panel;
	my_data = new wxTextDataObject;
	SetDataObject(my_data);
}

wxDragResult GroupDropTarget::OnData(wxCoord x, wxCoord y, wxDragResult defResult)
{
	long dropped_group;
	long dragged_item;
	long selected_asset;
	int flags;

	GetData();

	wxString dropped_text = my_data->GetText();
	const wxPoint drop_position(x, y);
	dropped_group = my_owner->HitTest(drop_position, flags);

	//  Add the specified image to the specified group..
//	wxPrintf("dropped_group = %li\n", dropped_group);
	if (dropped_group > 0)
	{
		if (my_panel->ContentsListBox->GetSelectedItemCount() == 1)
		{

			dropped_text.ToLong(&dragged_item);
			selected_asset = my_panel->ReturnGroupMember(my_panel->selected_group, dragged_item);


			if (my_panel->all_groups_list->groups[dropped_group].FindMember(selected_asset) == -1)
			{
				my_panel->all_groups_list->groups[dropped_group].AddMember(selected_asset);
				my_panel->InsertGroupMemberToDatabase(dropped_group, my_panel->all_groups_list->groups[dropped_group].number_of_members - 1);
			}
		}
		else
		{
			// work out all the selected items in the contents list and add them to the dragged group..

			long item = -1;

			main_frame->current_project.database.Begin();

			for ( ;; )
			{
				item = my_panel->ContentsListBox->GetNextItem(item, wxLIST_NEXT_ALL, wxLIST_STATE_SELECTED);
				if ( item == -1 )
				break;

				selected_asset = my_panel->all_groups_list->ReturnGroupMember(my_panel->selected_group, item);

				if (my_panel->all_groups_list->groups[dropped_group].FindMember(selected_asset) == -1)
				{
					my_panel->all_groups_list->groups[dropped_group].AddMember(selected_asset);
					my_panel->InsertGroupMemberToDatabase(dropped_group, my_panel->all_groups_list->groups[dropped_group].number_of_members - 1);
				}
			}

			main_frame->current_project.database.Commit();

		}


		my_panel->DirtyGroups();
		//my_panel->FillGroupList();
		main_frame->RecalculateAssetBrowser();
		return wxDragCopy;
	}

	else return wxDragNone;
}

wxDragResult GroupDropTarget::OnDragOver (wxCoord x, wxCoord y, wxDragResult defResult)
{
	if (my_panel->DragOverGroups(x, y) == true) return wxDragCopy;
	else return  wxDragCancel;

}

void GroupDropTarget::OnLeave ()
{
	if (my_panel->highlighted_item != -1)
	{
		my_panel->GroupListBox->SetItemBackgroundColour(my_panel->highlighted_item, my_panel->GroupListBox->GetBackgroundColour());
		my_panel->highlighted_item = -1;
	}
}


bool GroupDropTarget::OnDrop(wxCoord x, wxCoord y)//, const wxString& dropped_text)
{
	const wxPoint drop_position(x, y);
	int flags;
	long dropped_group = my_owner->HitTest(drop_position, flags);

	if (dropped_group > 0) return true;
	else return false;
}



