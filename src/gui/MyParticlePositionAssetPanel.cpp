#include "../core/core_headers.h"
#include "../core/gui_core_headers.h"

MyParticlePositionAssetPanel::MyParticlePositionAssetPanel( wxWindow* parent )
:
MyAssetParentPanel( parent )
{
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

	UpdateInfo();


	AssetTypeText->SetLabel("Particle Positions");

	all_groups_list->groups[0].SetName("All Positions");
	all_assets_list = new ParticlePositionAssetList;
	FillGroupList();
	FillContentsList();
}

MyParticlePositionAssetPanel::~MyParticlePositionAssetPanel()
{
	delete all_assets_list;
}

void MyParticlePositionAssetPanel::UpdateInfo()
{
	if (selected_content >= 0 && selected_group >= 0 && all_groups_list->groups[selected_group].number_of_members > 0)
	{
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
	else
	{
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

ParticlePositionAsset* MyParticlePositionAssetPanel::ReturnAssetPointer(long wanted_asset)
{
	return all_assets_list->ReturnParticlePositionAssetPointer(wanted_asset);
}

void MyParticlePositionAssetPanel::RemoveAssetFromDatabase(long wanted_asset)
{
	main_frame->current_project.database.ExecuteSQL(wxString::Format("DELETE FROM PARTICLE_POSITION_ASSETS WHERE PARTICLE_POSITION_ASSET_ID=%i", all_assets_list->ReturnAssetID(wanted_asset)).ToUTF8().data());
	all_assets_list->RemoveAsset(wanted_asset);
}

void MyParticlePositionAssetPanel::RemoveFromGroupInDatabase(int wanted_group_id, int wanted_asset_id)
{
	main_frame->current_project.database.ExecuteSQL(wxString::Format("DELETE FROM PARTICLE_POSITION_GROUP_%i WHERE PARTICLE_POSITION_ASSET_ID=%i", wanted_group_id, wanted_asset_id).ToUTF8().data());
}

void MyParticlePositionAssetPanel::InsertGroupMemberToDatabase(int wanted_group, int wanted_asset)
{
	MyDebugAssertTrue(wanted_group > 0 && wanted_group < all_groups_list->number_of_groups, "Requesting a group (%i) that doesn't exist!", wanted_group);
	MyDebugAssertTrue(wanted_asset >= 0 && wanted_asset < all_assets_list->number_of_assets, "Requesting an partice position (%i) that doesn't exist!", wanted_asset);

	main_frame->current_project.database.InsertOrReplace(wxString::Format("PARTICLE_POSITION_GROUP_%i", ReturnGroupID(wanted_group)).ToUTF8().data(), "ii", "MEMBER_NUMBER", "PARTICLE_POSITION_ASSET_ID", ReturnGroupSize(wanted_group), ReturnGroupMemberID(wanted_group, wanted_asset));
}

void  MyParticlePositionAssetPanel::RemoveAllFromDatabase()
{
	for (long counter = 1; counter < all_groups_list->number_of_groups; counter++)
	{
		main_frame->current_project.database.ExecuteSQL(wxString::Format("DROP TABLE PARTICLE_POSITION_GROUP_%i", all_groups_list->groups[counter].id).ToUTF8().data());
	}

	main_frame->current_project.database.ExecuteSQL("DROP TABLE PARTICLE_POSITION_GROUP_LIST");
	main_frame->current_project.database.CreateTable("PARTICLE_POSITION_GROUP_LIST", "pti", "GROUP_ID", "GROUP_NAME", "LIST_ID" );

}

void MyParticlePositionAssetPanel::RemoveAllGroupMembersFromDatabase(int wanted_group_id)
{
	main_frame->current_project.database.ExecuteSQL(wxString::Format("DROP TABLE PARTICLE_POSITION_GROUP_%i", wanted_group_id).ToUTF8().data());
	main_frame->current_project.database.CreateTable(wxString::Format("PARTICLE_POSITION_GROUP_%i", wanted_group_id).ToUTF8().data(), "ii", "MEMBER_NUMBER", "PARTICLE_POSITION_ASSET_ID");
}

void MyParticlePositionAssetPanel::AddGroupToDatabase(int wanted_group_id, const char * wanted_group_name, int wanted_list_id)
{
	main_frame->current_project.database.InsertOrReplace("PARTICLE_POSITION_GROUP_LIST", "iti", "GROUP_ID", "GROUP_NAME", "LIST_ID", wanted_group_id, wanted_group_name, wanted_list_id);
	main_frame->current_project.database.CreateTable(wxString::Format("PARTICLE_POSITION_GROUP_%i", wanted_list_id).ToUTF8().data(), "ii", "MEMBER_NUMBER", "PARTICLE_POSITION_ASSET_ID");
}

void MyParticlePositionAssetPanel::RemoveGroupFromDatabase(int wanted_group_id)
{
	main_frame->current_project.database.ExecuteSQL(wxString::Format("DROP TABLE PARTICLE_POSITION_GROUP_%i", wanted_group_id).ToUTF8().data());
	main_frame->current_project.database.ExecuteSQL(wxString::Format("DELETE FROM PARTICLE_POSITION_GROUP_LIST WHERE GROUP_ID=%i", wanted_group_id));
}

void MyParticlePositionAssetPanel::RenameGroupInDatabase(int wanted_group_id, const char *wanted_name)
{
	wxString sql_command = wxString::Format("UPDATE PARTICLE_POSITION_GROUP_LIST SET GROUP_NAME='%s' WHERE GROUP_ID=%i", wanted_name, wanted_group_id);
	main_frame->current_project.database.ExecuteSQL(sql_command.ToUTF8().data());

}

void MyParticlePositionAssetPanel::ImportAllFromDatabase()
{

	int counter;
	ParticlePositionAsset temp_asset;
	AssetGroup temp_group;

	all_assets_list->RemoveAll();
	all_groups_list->RemoveAll();

	// First all assets..

	main_frame->current_project.database.BeginAllParticlePositionAssetsSelect();

	while (main_frame->current_project.database.last_return_code == SQLITE_ROW)
	{
		temp_asset = main_frame->current_project.database.GetNextParticlePositionAsset();
		AddAsset(&temp_asset);
	}

	main_frame->current_project.database.EndAllParticlePositionAssetsSelect();

	// Now the groups..

	main_frame->current_project.database.BeginAllParticlePositionGroupsSelect();

	while (main_frame->current_project.database.last_return_code == SQLITE_ROW)
	{
		temp_group = main_frame->current_project.database.GetNextParticlePositionGroup();

		// the members of this group are referenced by asset id's, we need to translate this to array position..

		for (counter = 0; counter < temp_group.number_of_members; counter++)
		{
			temp_group.members[counter] = all_assets_list->ReturnArrayPositionFromID(temp_group.members[counter]);
		}

		all_groups_list->AddGroup(&temp_group);
		if (temp_group.id > current_group_number) current_group_number = temp_group.id;
	}

	main_frame->current_project.database.EndAllParticlePositionGroupsSelect();
	FillGroupList();
	FillContentsList();
}

void MyParticlePositionAssetPanel::FillAssetSpecificContentsList()
{

		ContentsListBox->InsertColumn(0, "I.D.", wxLIST_FORMAT_LEFT,  wxLIST_AUTOSIZE_USEHEADER );
		ContentsListBox->InsertColumn(1, "Parent Image I.D.", wxLIST_FORMAT_LEFT,  wxLIST_AUTOSIZE_USEHEADER );
		ContentsListBox->InsertColumn(2, "Pick Job I.D.", wxLIST_FORMAT_LEFT,  wxLIST_AUTOSIZE_USEHEADER );
		ContentsListBox->InsertColumn(3, "X Position", wxLIST_FORMAT_LEFT,  wxLIST_AUTOSIZE_USEHEADER );
		ContentsListBox->InsertColumn(4, "Y Position", wxLIST_FORMAT_LEFT,  wxLIST_AUTOSIZE_USEHEADER );


		for (long counter = 0; counter < all_groups_list->groups[selected_group].number_of_members; counter++)
		{
			ContentsListBox->InsertItem(counter, wxString::Format(wxT("%i"), all_assets_list->ReturnParticlePositionAssetPointer(all_groups_list->ReturnGroupMember(selected_group, counter))->asset_id, counter));
			ContentsListBox->SetItem(counter, 1, wxString::Format(wxT("%i"), all_assets_list->ReturnParticlePositionAssetPointer(all_groups_list->ReturnGroupMember(selected_group, counter))->parent_id));
			ContentsListBox->SetItem(counter, 2, wxString::Format(wxT("%i"), all_assets_list->ReturnParticlePositionAssetPointer(all_groups_list->ReturnGroupMember(selected_group, counter))->pick_job_id));
			ContentsListBox->SetItem(counter, 3, wxString::Format(wxT("%.2f"), all_assets_list->ReturnParticlePositionAssetPointer(all_groups_list->ReturnGroupMember(selected_group, counter))->x_position));
			ContentsListBox->SetItem(counter, 4, wxString::Format(wxT("%.2f"), all_assets_list->ReturnParticlePositionAssetPointer(all_groups_list->ReturnGroupMember(selected_group, counter))->y_position));


		}
}

void MyParticlePositionAssetPanel::ImportAssetClick( wxCommandEvent& event )
{

	//MyImageImportDialog *import_dialog = new MyImageImportDialog(this);
	//import_dialog->ShowModal();

}
