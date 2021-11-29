#include "../core/gui_core_headers.h"


MyRenameDialog::MyRenameDialog( wxWindow* parent )
:
RenameDialog( parent )
{
	MyAssetPanelParent *parent_asset_panel = reinterpret_cast <MyAssetPanelParent *> (parent);

	// How many assets are selected?

	long number_selected = parent_asset_panel->ContentsListBox->GetSelectedItemCount();
	long selected_asset;
	long number_removed = 0;
	long removed_counter;

	long item;
	long adjusted_item;
	long current_array_position;
	long current_asset_id;

	long currently_selected_group = parent_asset_panel->selected_group;




	if (number_selected > 0)
	{
		item = -1;

		for ( ;; )
		{

 			item = parent_asset_panel->ContentsListBox->GetNextItem(item, wxLIST_NEXT_ALL, wxLIST_STATE_SELECTED);

 			if ( item == -1 )
				break;


 			current_array_position = parent_asset_panel->ReturnGroupMember(currently_selected_group, item);
 			selected_assets_array_position.Add(current_array_position);
		}

	}

	// populate the dialog..

	for (long counter = 0; counter < selected_assets_array_position.GetCount(); counter++)
	{
		wxTextCtrl *current_text = new wxTextCtrl( RenameScrollPanel, wxID_ANY, parent_asset_panel->ReturnAssetName(selected_assets_array_position[counter]), wxDefaultPosition, wxDefaultSize, 0 );
		RenameBoxSizer->Add(current_text,  0, wxEXPAND | wxALL, 5 );
	}


	RenameBoxSizer->Fit(RenameScrollPanel);

	SizeAndPosition();



}

void MyRenameDialog::OnCancelClick( wxCommandEvent& event )
{
	Destroy();
}

void MyRenameDialog::OnRenameClick( wxCommandEvent& event )
{
	//BuildSearchCommand();
	EndModal(wxID_OK);
}


void MyRenameDialog::SizeAndPosition()
{
	MainBoxSizer->Layout();
	wxSize input_size = MainBoxSizer->GetMinSize();
	input_size.y+=10;

	int frame_width;
	int frame_height;
	int frame_position_x;
	int frame_position_y;

	main_frame->GetClientSize(&frame_width, &frame_height);
	main_frame->GetPosition(&frame_position_x, &frame_position_y);

	int total_height = input_size.y;

	input_size.x += wxSystemSettings::GetMetric(wxSYS_VSCROLL_X);

	if (total_height > frame_height * 0.9)
	{
		input_size.y -= (total_height - frame_height * 0.9);
	}

	RenameScrollPanel->SetMinSize(input_size);
	RenameScrollPanel->SetSize(input_size);
	RenameBoxSizer->Fit(RenameScrollPanel);
	SetMaxSize(wxSize(frame_width, frame_height * 0.91));
	Layout();
	MainBoxSizer->Fit(this);

	int dialog_height;
	int dialog_width;

	// ok so how big is this dialog now?

	GetSize(&dialog_width, &dialog_height);

	int new_x_pos = (frame_position_x + (frame_width / 2) - (dialog_width / 2));
	int new_y_pos = (frame_position_y + (frame_height / 2) - (dialog_height / 2));

	Move(new_x_pos, new_y_pos);
}
