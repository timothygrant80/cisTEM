#include "../core/core_headers.h"
#include "../core/gui_core_headers.h"

MyImageAssetPanel::MyImageAssetPanel( wxWindow* parent )
:
MyAssetParentPanel( parent )
{
	Label0Title->SetLabel("Filename : ");
	Label1Title->SetLabel("I.D. : ");
	//Label2Title->SetLabel("No. Frames : ");
	Label2Title->SetLabel("X Size : ");
	Label3Title->SetLabel("Y Size : ");
	Label4Title->SetLabel("Pixel Size : ");
	//Label5Title->SetLabel("Total Exp. : ");
	//Label7Title->SetLabel("Exp. Per Frame : ");
	Label5Title->SetLabel("Voltage : ");
	Label6Title->SetLabel("Cs : ");
	Label7Title->SetLabel("");
	Label8Title->SetLabel("");
	Label9Title->SetLabel("");

	Label7Text->SetLabel("");
	Label8Text->SetLabel("");
	Label9Text->SetLabel("");

	AssetTypeText->SetLabel("Images");

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
		Label0Text->SetLabel(all_assets_list->ReturnAssetPointer(all_groups_list->ReturnGroupMember(selected_group, selected_content))->ReturnFullPathString());
		Label1Text->SetLabel(wxString::Format(wxT("%i"), all_assets_list->ReturnImageAssetPointer(all_groups_list->ReturnGroupMember(selected_group, selected_content))->asset_id));
		//Label2Text->SetLabel(wxString::Format(wxT("%i"), all_assets_list->ReturnMovieAssetPointer(all_groups_list->ReturnGroupMember(selected_group, selected_content))->number_of_frames));
		Label2Text->SetLabel(wxString::Format(wxT("%i px"), all_assets_list->ReturnImageAssetPointer(all_groups_list->ReturnGroupMember(selected_group, selected_content))->x_size));
		Label3Text->SetLabel(wxString::Format(wxT("%i px"), all_assets_list->ReturnImageAssetPointer(all_groups_list->ReturnGroupMember(selected_group, selected_content))->y_size));
		Label4Text->SetLabel(wxString::Format(wxT("%.2f Å"), all_assets_list->ReturnImageAssetPointer(all_groups_list->ReturnGroupMember(selected_group, selected_content))->pixel_size));
		//Label5Text->SetLabel(wxString::Format(wxT("%.2f e¯/Å²"), all_assets_list->ReturnImageAssetPointer(all_groups_list->ReturnGroupMember(selected_group, selected_content))->total_dose));
		//Label7Text->SetLabel(wxString::Format(wxT("%.2f e¯/Å²"), all_assets_list->ReturnMovieAssetPointer(all_groups_list->ReturnGroupMember(selected_group, selected_content))->dose_per_frame));
		Label5Text->SetLabel(wxString::Format(wxT("%.2f kV"), all_assets_list->ReturnImageAssetPointer(all_groups_list->ReturnGroupMember(selected_group, selected_content))->microscope_voltage));
		Label6Text->SetLabel(wxString::Format(wxT("%.2f mm"), all_assets_list->ReturnImageAssetPointer(all_groups_list->ReturnGroupMember(selected_group, selected_content))->spherical_aberration));
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
	main_frame->current_project.database.CreateTable(wxString::Format("IMAGE_GROUP_%i", wanted_list_id).ToUTF8().data(), "ii", "MEMBER_NUMBER", "IMAGE_ASSET_ID");
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

void MyImageAssetPanel::ImportAllFromDatabase()
{
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
		ContentsListBox->InsertColumn(1, "File", wxLIST_FORMAT_LEFT,  wxLIST_AUTOSIZE_USEHEADER );
		ContentsListBox->InsertColumn(2, "X Size", wxLIST_FORMAT_LEFT,  wxLIST_AUTOSIZE_USEHEADER );
		ContentsListBox->InsertColumn(3, "Y Size", wxLIST_FORMAT_LEFT,  wxLIST_AUTOSIZE_USEHEADER );
		//ContentsListBox->InsertColumn(4, "No. frames", wxLIST_FORMAT_LEFT,  wxLIST_AUTOSIZE_USEHEADER );
		ContentsListBox->InsertColumn(4, "Pixel size", wxLIST_FORMAT_LEFT,  wxLIST_AUTOSIZE_USEHEADER );
		//ContentsListBox->InsertColumn(6, "Exp. per frame", wxLIST_FORMAT_LEFT,  wxLIST_AUTOSIZE_USEHEADER );
		ContentsListBox->InsertColumn(5, "Cs", wxLIST_FORMAT_LEFT,  wxLIST_AUTOSIZE_USEHEADER );
		ContentsListBox->InsertColumn(6, "Voltage", wxLIST_FORMAT_LEFT,  wxLIST_AUTOSIZE_USEHEADER );


		for (long counter = 0; counter < all_groups_list->groups[selected_group].number_of_members; counter++)
		{
			ContentsListBox->InsertItem(counter, wxString::Format(wxT("%i"), all_assets_list->ReturnImageAssetPointer(all_groups_list->ReturnGroupMember(selected_group, counter))->asset_id, counter));
			ContentsListBox->SetItem(counter, 1, all_assets_list->ReturnAssetPointer(all_groups_list->ReturnGroupMember(selected_group, counter))->ReturnShortNameString());
			ContentsListBox->SetItem(counter, 2, wxString::Format(wxT("%i"),all_assets_list->ReturnImageAssetPointer(all_groups_list->ReturnGroupMember(selected_group, counter))->x_size));
			ContentsListBox->SetItem(counter, 3, wxString::Format(wxT("%i"), all_assets_list->ReturnImageAssetPointer(all_groups_list->ReturnGroupMember(selected_group, counter))->y_size));
		//	ContentsListBox->SetItem(counter, 4, wxString::Format(wxT("%i"), all_assets_list->ReturnMovieAssetPointer(all_groups_list->ReturnGroupMember(selected_group, counter))->number_of_frames));
			ContentsListBox->SetItem(counter, 4, wxString::Format(wxT("%.3f"),all_assets_list->ReturnImageAssetPointer(all_groups_list->ReturnGroupMember(selected_group, counter))->pixel_size));
//			ContentsListBox->SetItem(counter, 6, wxString::Format(wxT("%.3f"), all_assets_list->ReturnMovieAssetPointer(all_groups_list->ReturnGroupMember(selected_group, counter))->dose_per_frame));
			ContentsListBox->SetItem(counter, 5, wxString::Format(wxT("%.2f"), all_assets_list->ReturnImageAssetPointer(all_groups_list->ReturnGroupMember(selected_group, counter))->spherical_aberration));
			ContentsListBox->SetItem(counter, 6, wxString::Format(wxT("%.2f"), all_assets_list->ReturnImageAssetPointer(all_groups_list->ReturnGroupMember(selected_group, counter))->microscope_voltage));


		}
}

void MyImageAssetPanel::ImportAssetClick( wxCommandEvent& event )
{

	MyImageImportDialog *import_dialog = new MyImageImportDialog(this);
	import_dialog->ShowModal();

}
