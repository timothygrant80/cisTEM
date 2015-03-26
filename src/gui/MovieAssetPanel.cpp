#include "MovieAssetPanel.h"
#include "MovieImportDialog.h"
#include <wx/log.h>
#include <iostream>


extern MyMainFrame *main_frame;
extern wxTreeItemId Tree_Root;
extern MyMovieAssetPanel *movie_asset_panel;
extern MyAlignMoviesPanel *align_movies_panel;

#ifdef DEBUG
#define MyDebugPrint(...)	wxLogDebug(__VA_ARGS__); wxPrintf("From %s:%i\n%s\n", __FILE__,__LINE__,__PRETTY_FUNCTION__);
#else
#define MyDebugPrint(...)
#endif

MyMovieAssetPanel::MyMovieAssetPanel( wxWindow* parent )
:
MovieAssetPanel( parent )
{

	selected_group = 0;
	FillGroupList();
	FillContentsList();
	highlighted_item = -1;

	name_is_being_edited = false;

	GroupListBox->SetDropTarget(new GroupDropTarget(GroupListBox));
}


void MyMovieAssetPanel::ImportMovieClick( wxCommandEvent& event )
{

	MyMovieImportDialog *import_dialog = new MyMovieImportDialog(this);
	import_dialog->ShowModal();

}

void MyMovieAssetPanel::RemoveMovieClick( wxCommandEvent& event )
{
	// How many movies are selected?

	long number_selected = ContentsListBox->GetSelectedItemCount();
	long selected_movie;


	if (number_selected > 0)
	{
		if (selected_group == 0)
		{
			wxMessageDialog *check_dialog = new wxMessageDialog(this, "This will remove the selected movies from your ENTIRE project! Are you sure you want to continue?", "Are you sure?", wxYES_NO);

			if (check_dialog->ShowModal() ==  wxID_YES)
			{
				long item = -1;

				for ( ;; )
				{
					item = ContentsListBox->GetNextItem(item, wxLIST_NEXT_ALL, wxLIST_STATE_SELECTED);
					if ( item == -1 )
						break;

					all_assets_list.RemoveMovie(item);
					all_groups_list.groups[0].RemoveMember(item);
					all_groups_list.RemoveMemberFromAllExtraGroups(item);
				}
			}
		}
		else
		{
			long item = -1;

			for ( ;; )
			{
					item = ContentsListBox->GetNextItem(item, wxLIST_NEXT_ALL, wxLIST_STATE_SELECTED);
					if ( item == -1 )
					break;

					selected_movie = all_groups_list.ReturnGroupMember(selected_group, item);

					all_groups_list.groups[selected_group].RemoveMember(selected_movie);
			}

		}

		FillGroupList();
		FillContentsList();
		main_frame->RecalculateAssetBrowser();
		//UpdateMovieInfo();



	}

}

void MyMovieAssetPanel::AddContentItemToGroup(long wanted_group, long wanted_content_item)
{

	long selected_movie = all_groups_list.ReturnGroupMember(selected_group, wanted_content_item);
	MyDebugPrint("%li\n", selected_movie);
	if (all_groups_list.groups[wanted_group].FindMember(selected_movie) == -1)
	{
		all_groups_list.groups[wanted_group].AddMember(selected_movie);
	}
}

void MyMovieAssetPanel::AddSelectedClick( wxCommandEvent& event )
{
	// How many movies are selected?

	long number_selected = ContentsListBox->GetSelectedItemCount();
	long selected_movie;


	if (number_selected > 0)
	{
		if (all_groups_list.number_of_groups > 1)
		{
			// selected group..

			wxArrayString my_choices;

			for (long counter = 1; counter < ReturnNumberOfGroups(); counter++)
			{
				my_choices.Add(all_groups_list.groups[counter].name);
			}

			wxSingleChoiceDialog	*group_choice = new wxSingleChoiceDialog(this, "Add Selected Movies to which group(s)?", "Select Groups", my_choices);

			if (group_choice->ShowModal() ==  wxID_OK)
			{
				long item = -1;

				for ( ;; )
				{
					item = ContentsListBox->GetNextItem(item, wxLIST_NEXT_ALL, wxLIST_STATE_SELECTED);
					if ( item == -1 )
						break;

					AddContentItemToGroup(group_choice->GetSelection() + 1, selected_movie);
				}
			}

			group_choice->Destroy();
		}

		FillGroupList();
		FillContentsList();
		main_frame->RecalculateAssetBrowser();
		//UpdateMovieInfo();
	}

}

void MyMovieAssetPanel::RemoveAllClick( wxCommandEvent& event )
{

	if (selected_group == 0)
	{
		if (all_assets_list.number_of_assets > 0)
		{
			wxMessageDialog *check_dialog = new wxMessageDialog(this, "This will remove ALL movies from your ENTIRE project! Are you sure you want to continue?", "Are you really really sure?", wxYES_NO);

				if (check_dialog->ShowModal() ==  wxID_YES)
				{
					all_assets_list.RemoveAll();

					for (long counter = 0; counter < all_groups_list.number_of_groups; counter++)
					{
						all_groups_list.groups[counter].RemoveAll();
					}
				}


				SetSelectedGroup(0);
				CheckActiveButtons();
				main_frame->RecalculateAssetBrowser();
		}
	}
	else
	{
		all_groups_list.groups[selected_group].RemoveAll();
		FillGroupList();
		FillContentsList();
		CheckActiveButtons();
		main_frame->RecalculateAssetBrowser();
	}


	MyDebugPrint("Remove All Clicked!");




}

void MyMovieAssetPanel:: NewGroupClick( wxCommandEvent& event )
{
	// Add a new Group - called New Group

	all_groups_list.AddGroup("New Group");

	// How many movies are selected?

	long number_selected = ContentsListBox->GetSelectedItemCount();


	if (number_selected > 0)
	{
			long item = -1;

			for ( ;; )
			{
			    item = ContentsListBox->GetNextItem(item, wxLIST_NEXT_ALL, wxLIST_STATE_SELECTED);
			    if ( item == -1 )
			        break;


			    AddContentItemToGroup(all_groups_list.ReturnNumberOfGroups() - 1, item);
			}
			//UpdateMovieInfo();
	}

	FillGroupList();
	SetSelectedGroup(all_groups_list.ReturnNumberOfGroups() - 1);

	main_frame->RecalculateAssetBrowser();
}


void MyMovieAssetPanel:: RemoveGroupClick( wxCommandEvent& event )
{
	if (selected_group != 0)
	{
		all_groups_list.RemoveGroup(selected_group);
		SetSelectedGroup(selected_group - 1);
	}

	FillGroupList();
	main_frame->RecalculateAssetBrowser();
}

void MyMovieAssetPanel::AddAsset(MovieAsset *asset_to_add)
{
	// Firstly add the asset to the Asset list

	all_assets_list.AddMovie(asset_to_add);
	all_groups_list.AddMemberToGroup(0, all_assets_list.number_of_assets - 1);

	//FillContentsList();
}

void MyMovieAssetPanel::SetSelectedGroup(long wanted_group)
{
	wxLogDebug("Setting group #%li\n", wanted_group);
	MyDebugPrint("There are #%li groups in total.\n", all_groups_list.number_of_groups);

	if (wanted_group < 0 || wanted_group >= all_groups_list.number_of_groups)
	{
		wxPrintf("Error! Trying to select a group that doesn't exist!\n\n");
		exit(-1);
	}
	else
	{
		GroupListBox->SetItemState(wanted_group, wxLIST_STATE_SELECTED, wxLIST_STATE_SELECTED);
		selected_group = wanted_group;
		selected_content = 0;
		//FillGroupList();
		FillContentsList();
	}

	CheckActiveButtons();

}

void MyMovieAssetPanel::CheckActiveButtons()
{
	if (selected_group == 0)
	{
		RemoveGroupButton->Enable(false);
	}
	else
	{
		RemoveGroupButton->Enable(true);
	}

	if (ContentsListBox->GetItemCount() < 1)
	{
		RemoveAllMoviesButton->Enable(false);
		RemoveSelectedMovieButton->Enable(false);
		AddSelectedButton->Enable(false);
	}
	else
	{
		RemoveAllMoviesButton->Enable(true);

		if (ContentsListBox->GetSelectedItemCount() < 1)
		{
			RemoveSelectedMovieButton->Enable(false);
			AddSelectedButton->Enable(false);
		}
		else
		{
			RemoveSelectedMovieButton->Enable(true);

			if (ReturnNumberOfGroups() > 1) AddSelectedButton->Enable(true);
			else AddSelectedButton->Enable(false);
		}
	}




}


unsigned long MyMovieAssetPanel::ReturnNumberOfAssets()
{
	return all_assets_list.number_of_assets;
}

unsigned long MyMovieAssetPanel::ReturnNumberOfGroups()
{
	return all_groups_list.number_of_groups;
}

void MyMovieAssetPanel::SetGroupName(long wanted_group, wxString wanted_name)
{
	all_groups_list.groups[wanted_group].name = wanted_name;
}

wxString MyMovieAssetPanel::ReturnGroupName(long wanted_group)
{
	return all_groups_list.groups[wanted_group].name;
}

wxString MyMovieAssetPanel::ReturnAssetShortFilename(long wanted_asset)
{
	return all_assets_list.assets[wanted_asset].ReturnShortNameString();
}

wxString MyMovieAssetPanel::ReturnAssetLongFilename(long wanted_asset)
{
	return all_assets_list.assets[wanted_asset].ReturnFullPathString();
}

long MyMovieAssetPanel::ReturnGroupMember(long wanted_group, long wanted_member)
{
	return all_groups_list.groups[wanted_group].members[wanted_member];
}

long MyMovieAssetPanel::ReturnGroupSize(long wanted_group)
{
	return all_groups_list.groups[wanted_group].number_of_members;
}

void MyMovieAssetPanel::SizeGroupColumn()
{
	int client_height;
	int client_width;

	int current_width;

	GroupListBox->GetClientSize(&client_width, &client_height);
	GroupListBox->SetColumnWidth(0, wxLIST_AUTOSIZE);

	current_width = GroupListBox->GetColumnWidth(0);

	if (client_width > current_width) GroupListBox->SetColumnWidth(0, client_width);

}

void MyMovieAssetPanel::FillGroupList()
{
	//wxColor my_grey(50,50,50);

	wxPrintf("Filling Group List\n");
	wxFont current_font;

	long currently_selected = selected_group;

	GroupListBox->Freeze();
	GroupListBox->ClearAll();
	GroupListBox->InsertColumn(0, "Files", wxLIST_FORMAT_LEFT, wxLIST_AUTOSIZE);

	GroupListBox->InsertItem(0, all_groups_list.groups[0].name + " (" + wxString::Format(wxT("%li"), all_groups_list.groups[0].number_of_members) + ")", 0);

	current_font = GroupListBox->GetFont();
	current_font.MakeBold();
	GroupListBox->SetItemFont(0, current_font);

	for (long counter = 1; counter < all_groups_list.number_of_groups; counter++)
	{
		GroupListBox->InsertItem(counter, all_groups_list.groups[counter].name + " (" + wxString::Format(wxT("%li"), all_groups_list.groups[counter].number_of_members) + ")" , counter);
	}

	SizeGroupColumn();
	GroupListBox->Thaw();
}

void MyMovieAssetPanel::OnBeginEdit( wxListEvent& event )
{
	//wxPrintf("Begin Label = %s\n", event.GetLabel());

	// this is a bit of a hack, as sometimes weird things happen and end edit is called by itself, then
	// the label gets the number added to it.

	name_is_being_edited = true;

	if (event.GetIndex() == 0) event.Veto();
	else
	{
		GroupListBox->SetItemText(event.GetIndex(), all_groups_list.groups[event.GetIndex()].name);
		event.Skip();
	}
}

void MyMovieAssetPanel::OnEndEdit( wxListEvent& event )
{

	if (event.GetLabel() == wxEmptyString)
	{
		GroupListBox->SetItemText(event.GetIndex(), all_groups_list.groups[event.GetIndex()].name + " (" + wxString::Format(wxT("%li"), all_groups_list.groups[event.GetIndex()].number_of_members) + ")");
		event.Veto();
	}
	else
	{
		if (name_is_being_edited == true)
		{

			SetGroupName(event.GetIndex(), event.GetLabel());
			event.Veto();
			GroupListBox->SetItemText(event.GetIndex(), all_groups_list.groups[event.GetIndex()].name + " (" + wxString::Format(wxT("%li"), all_groups_list.groups[event.GetIndex()].number_of_members) + ")");
	//		GroupListBox->Refresh();
			//FillGroupList();
			name_is_being_edited = false;
		}
		else event.Veto();
	}


}


void MyMovieAssetPanel::FillContentsList()
{

	if ( selected_group >= 0  )
	{
		ContentsListBox->Freeze();
		ContentsListBox->ClearAll();
		ContentsListBox->InsertColumn(0, "Files", wxLIST_FORMAT_LEFT, wxLIST_AUTOSIZE);


		for (long counter = 0; counter < all_groups_list.groups[selected_group].number_of_members; counter++)
		{
			ContentsListBox->InsertItem(counter, all_assets_list.assets[ReturnGroupMember(selected_group, counter)].ReturnFullPathString(), counter);
		}

		ContentsListBox->SetColumnWidth(0, wxLIST_AUTOSIZE);
		ContentsListBox->Thaw();
	}
}

void MyMovieAssetPanel::OnGroupFocusChange( wxListEvent& event )
{
	selected_group = event.GetIndex();

	//wxPrintf("Group Change\n\n");
	FillContentsList();
	CheckActiveButtons();
	event.Skip();
}

void MyMovieAssetPanel::OnContentsSelected( wxListEvent& event )
{
	selected_content = event.GetIndex();

	//wxPrintf("Selected Group = %li\n", selected_group);
	//wxPrintf("Selected Content = %li\n", selected_content);

	event.Skip();

	UpdateMovieInfo();
	CheckActiveButtons();

}

void MyMovieAssetPanel::OnBeginContentsDrag( wxListEvent& event )
{
	long source_item = event.GetIndex();

	//wxPrintf("Dragging item #%li\n", source_item);

	wxTextDataObject my_data(wxString::Format(wxT("%li"), source_item));
	wxDropSource dragSource( this );
	dragSource.SetData( my_data );
	dragSource.DoDragDrop( true );

	event.Skip();
}


void MyMovieAssetPanel::UpdateMovieInfo()
{
	if (selected_content >= 0 && selected_group >= 0 && all_groups_list.groups[selected_group].number_of_members > 0)
	{
		FilenameText->SetLabel(all_assets_list.assets[all_groups_list.ReturnGroupMember(selected_group, selected_content)].ReturnShortNameString());
		NumberOfFramesText->SetLabel(wxString::Format(wxT("%li"), all_assets_list.assets[all_groups_list.ReturnGroupMember(selected_group, selected_content)].number_of_frames));
		TotalDoseText->SetLabel(wxString::Format(wxT("%.2f e¯/Å²"), all_assets_list.assets[all_groups_list.ReturnGroupMember(selected_group, selected_content)].total_dose));
		PixelSizeText->SetLabel(wxString::Format(wxT("%.2f Å"), all_assets_list.assets[all_groups_list.ReturnGroupMember(selected_group, selected_content)].pixel_size));
	    DosePerFrameText->SetLabel(wxString::Format(wxT("%.2f e¯/Å²"), all_assets_list.assets[all_groups_list.ReturnGroupMember(selected_group, selected_content)].dose_per_frame));

	}
	else
	{
		FilenameText->SetLabel("-");
		NumberOfFramesText->SetLabel("-");
		TotalDoseText->SetLabel("-");
		PixelSizeText->SetLabel("-");
	    DosePerFrameText->SetLabel("-");
	}

}

bool MyMovieAssetPanel::IsFileAnAsset(wxFileName file_to_check)
{
	if (all_assets_list.FindFile(file_to_check) == -1) return false;
	else return true;
}

bool MyMovieAssetPanel::DragOverGroups(wxCoord x, wxCoord y)
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
				movie_asset_panel->GroupListBox->SetItemBackgroundColour(highlighted_item, movie_asset_panel->GroupListBox->GetBackgroundColour());

			}

			movie_asset_panel->GroupListBox->SetItemBackgroundColour(dropped_group, *wxLIGHT_GREY);
			highlighted_item = dropped_group;

		}

		return true;
	}
	else
	{
		if (highlighted_item != -1)
		{
			movie_asset_panel->GroupListBox->SetItemBackgroundColour(highlighted_item, movie_asset_panel->GroupListBox->GetBackgroundColour());
			highlighted_item = -1;

		}

		return false;
	}

}

// Drag and Drop

GroupDropTarget::GroupDropTarget(wxListCtrl *owner)
{
	my_owner = owner;
	my_data = new wxTextDataObject;
	SetDataObject(my_data);
}

wxDragResult GroupDropTarget::OnData(wxCoord x, wxCoord y, wxDragResult defResult)
{
	long dropped_group;
	long dragged_item;
	long selected_movie;
	int flags;

	GetData();

	wxString dropped_text = my_data->GetText();
	const wxPoint drop_position(x, y);
	dropped_group = my_owner->HitTest(drop_position, flags);

	//  Add the specified image to the specified group..

	if (dropped_group > 0)
	{
		if (movie_asset_panel->ContentsListBox->GetSelectedItemCount() == 1)
		{

			dropped_text.ToLong(&dragged_item);
			selected_movie = movie_asset_panel->all_groups_list.ReturnGroupMember(movie_asset_panel->selected_group, dragged_item);


			if (movie_asset_panel->all_groups_list.groups[dropped_group].FindMember(selected_movie) == -1)
			{
				movie_asset_panel->all_groups_list.groups[dropped_group].AddMember(selected_movie);
			}
		}
		else
		{
			// work out all the selected items in the contents list and add them to the dragged group..

			long item = -1;

			for ( ;; )
			{
				item = movie_asset_panel->ContentsListBox->GetNextItem(item, wxLIST_NEXT_ALL, wxLIST_STATE_SELECTED);
				if ( item == -1 )
				break;

				selected_movie = movie_asset_panel->all_groups_list.ReturnGroupMember(movie_asset_panel->selected_group, item);

				if (movie_asset_panel->all_groups_list.groups[dropped_group].FindMember(selected_movie) == -1)
				{
					movie_asset_panel->all_groups_list.groups[dropped_group].AddMember(selected_movie);
				}
			}

		}

		movie_asset_panel->FillGroupList();
		main_frame->RecalculateAssetBrowser();
		return wxDragCopy;
	}

	else return wxDragNone;
}

wxDragResult GroupDropTarget::OnDragOver (wxCoord x, wxCoord y, wxDragResult defResult)
{
	if (movie_asset_panel->DragOverGroups(x, y) == true) return wxDragCopy;
	else return  wxDragCancel;

}

void GroupDropTarget::OnLeave ()
{
	if (movie_asset_panel->highlighted_item != -1)
	{
		movie_asset_panel->GroupListBox->SetItemBackgroundColour(movie_asset_panel->highlighted_item, movie_asset_panel->GroupListBox->GetBackgroundColour());
		movie_asset_panel->highlighted_item = -1;
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



