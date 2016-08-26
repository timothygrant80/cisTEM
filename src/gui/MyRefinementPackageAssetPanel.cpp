#include "../core/gui_core_headers.h"

extern MyVolumeAssetPanel *volume_asset_panel;

MyRefinementPackageAssetPanel::MyRefinementPackageAssetPanel( wxWindow* parent )
:
RefinementPackageAssetPanel( parent )
{
	current_asset_number = 0;
	selected_refinement_package = -1;
	is_dirty = false;

	RefinementPackageListCtrl->InsertColumn(0, "Packages", wxLIST_FORMAT_LEFT, wxLIST_AUTOSIZE);

}

void MyRefinementPackageAssetPanel::OnCreateClick( wxCommandEvent& event )
{
	MyNewRefinementPackageWizard *my_wizard = new MyNewRefinementPackageWizard(this);
	my_wizard->RunWizard(my_wizard->template_page);
}

void MyRefinementPackageAssetPanel::OnRenameClick( wxCommandEvent& event )
{
	if (selected_refinement_package >= 0)
	{
		RefinementPackageListCtrl->EditLabel(selected_refinement_package);
	}

}

void MyRefinementPackageAssetPanel::OnDeleteClick( wxCommandEvent& event )
{
	if (selected_refinement_package >= 0)
	{
		wxMessageDialog *check_dialog = new wxMessageDialog(this, "This will remove the refinement package from your ENTIRE project!\n\nYou probably shouldn't do this if you have used it. Are you sure you want to continue?", "Are you sure?", wxYES_NO);

			if (check_dialog->ShowModal() ==  wxID_YES)
			{
				main_frame->current_project.database.DeleteTable(wxString::Format("REFINEMENT_PACKAGE_CONTAINED_PARTICLES_%li", all_refinement_packages.Item(selected_refinement_package).asset_id));
				main_frame->current_project.database.DeleteTable(wxString::Format("REFINEMENT_PACKAGE_CURRENT_REFERENCES_%li", all_refinement_packages.Item(selected_refinement_package).asset_id));
				main_frame->current_project.database.ExecuteSQL(wxString::Format("DELETE FROM REFINEMENT_PACKAGE_ASSETS WHERE REFINEMENT_PACKAGE_ASSET_ID=%li", all_refinement_packages.Item(selected_refinement_package).asset_id));

				all_refinement_packages.RemoveAt(selected_refinement_package);
				main_frame->DirtyRefinementPackages();
			}
	}
}



void MyRefinementPackageAssetPanel::AddAsset(RefinementPackage *refinement_package)
{
	// add into memory..

	current_asset_number++;
	refinement_package->asset_id = current_asset_number;

	all_refinement_packages.Add(refinement_package);

	// now add it to the database..

	main_frame->current_project.database.AddRefinementPackageAsset(refinement_package);
	main_frame->DirtyRefinementPackages();

}

void MyRefinementPackageAssetPanel::FillRefinementPackages()
{
	Freeze();

	if (all_refinement_packages.GetCount() > 0)
	{

		RefinementPackageListCtrl->SetItemCount(all_refinement_packages.GetCount());
		RefinementPackageListCtrl->SetColumnWidth(0, RefinementPackageListCtrl->ReturnGuessAtColumnTextWidth());

		if (selected_refinement_package >= 0 && selected_refinement_package < all_refinement_packages.GetCount())
		{
			RefinementPackageListCtrl->SetItemState(selected_refinement_package, wxLIST_STATE_SELECTED, wxLIST_STATE_SELECTED);
		}
		else
		{
			selected_refinement_package = 0;
			RefinementPackageListCtrl->SetItemState(selected_refinement_package, wxLIST_STATE_SELECTED, wxLIST_STATE_SELECTED);
		}

		if (all_refinement_packages.GetCount() >  0) RefinementPackageListCtrl->RefreshItems(0, all_refinement_packages.GetCount() - 1);

		StackFileNameText->SetLabel(all_refinement_packages.Item(selected_refinement_package).stack_filename);
		StackBoxSizeText->SetLabel(wxString::Format("%i px", all_refinement_packages.Item(selected_refinement_package).stack_box_size));
		NumberofClassesText->SetLabel(wxString::Format("%i", all_refinement_packages.Item(selected_refinement_package).number_of_classes));
		NumberofRefinementsText->SetLabel(wxString::Format("%i", all_refinement_packages.Item(selected_refinement_package).number_of_run_refinments));
		LastRefinementIDText->SetLabel(wxString::Format("%li", all_refinement_packages.Item(selected_refinement_package).last_refinment_id));
		SymmetryText->SetLabel(all_refinement_packages.Item(selected_refinement_package).symmetry);
		MolecularWeightText->SetLabel(wxString::Format(wxT("%.0f kDa"), all_refinement_packages.Item(selected_refinement_package).estimated_particle_weight_in_kda));
		LargestDimensionText->SetLabel(wxString::Format(wxT("%.0f Å"), all_refinement_packages.Item(selected_refinement_package).estimated_particle_size_in_angstroms));

		// setup the contents panel..

		ContainedParticlesListCtrl->ClearAll();
		ContainedParticlesListCtrl->InsertColumn(0, "Orig. Pos. ID", wxLIST_FORMAT_LEFT);
		ContainedParticlesListCtrl->InsertColumn(1, "Image ID", wxLIST_FORMAT_LEFT);
		ContainedParticlesListCtrl->InsertColumn(2, "X Pos.", wxLIST_FORMAT_LEFT);
		ContainedParticlesListCtrl->InsertColumn(3, "Y Pos.", wxLIST_FORMAT_LEFT);
		ContainedParticlesListCtrl->InsertColumn(4, "Pixel Size", wxLIST_FORMAT_LEFT);
		ContainedParticlesListCtrl->InsertColumn(5, "Cs", wxLIST_FORMAT_LEFT);
		ContainedParticlesListCtrl->InsertColumn(6, "Voltage", wxLIST_FORMAT_LEFT);
		ContainedParticlesListCtrl->InsertColumn(7, "Init. Defocus 1", wxLIST_FORMAT_LEFT);
		ContainedParticlesListCtrl->InsertColumn(8, "Init. Defocus 2", wxLIST_FORMAT_LEFT);
		ContainedParticlesListCtrl->InsertColumn(9, "Init. Defocus Angle", wxLIST_FORMAT_LEFT);
	//	ContainedParticlesListCtrl->InsertColumn(10, wxT("Init. Psi"), wxLIST_FORMAT_LEFT);
	//	ContainedParticlesListCtrl->InsertColumn(11, wxT("Init. Theta"), wxLIST_FORMAT_LEFT);
	//	ContainedParticlesListCtrl->InsertColumn(12, wxT("Init. Phi"), wxLIST_FORMAT_LEFT);
	//	ContainedParticlesListCtrl->InsertColumn(13, "Init. X-shift", wxLIST_FORMAT_LEFT);
	//	ContainedParticlesListCtrl->InsertColumn(14, "Init. Y-Shift", wxLIST_FORMAT_LEFT);


		ContainedParticlesListCtrl->SetItemCount(all_refinement_packages.Item(selected_refinement_package).contained_particles.GetCount());

		if (all_refinement_packages.Item(selected_refinement_package).contained_particles.GetCount() > 0)
		{
			ContainedParticlesListCtrl->RefreshItems(0, all_refinement_packages.Item(selected_refinement_package).contained_particles.GetCount() -1);

			ContainedParticlesListCtrl->SetColumnWidth(0, ContainedParticlesListCtrl->ReturnGuessAtColumnTextWidth(0));
			ContainedParticlesListCtrl->SetColumnWidth(1, ContainedParticlesListCtrl->ReturnGuessAtColumnTextWidth(1));
			ContainedParticlesListCtrl->SetColumnWidth(2, ContainedParticlesListCtrl->ReturnGuessAtColumnTextWidth(2));
			ContainedParticlesListCtrl->SetColumnWidth(3, ContainedParticlesListCtrl->ReturnGuessAtColumnTextWidth(3));
			ContainedParticlesListCtrl->SetColumnWidth(4, ContainedParticlesListCtrl->ReturnGuessAtColumnTextWidth(4));
			ContainedParticlesListCtrl->SetColumnWidth(5, ContainedParticlesListCtrl->ReturnGuessAtColumnTextWidth(5));
			ContainedParticlesListCtrl->SetColumnWidth(6, ContainedParticlesListCtrl->ReturnGuessAtColumnTextWidth(6));
			ContainedParticlesListCtrl->SetColumnWidth(7, ContainedParticlesListCtrl->ReturnGuessAtColumnTextWidth(7));
			ContainedParticlesListCtrl->SetColumnWidth(8, ContainedParticlesListCtrl->ReturnGuessAtColumnTextWidth(8));
			ContainedParticlesListCtrl->SetColumnWidth(9, ContainedParticlesListCtrl->ReturnGuessAtColumnTextWidth(9));
		}

		// 3D references..

		ReDrawActiveReferences();

	}
	else
	{
		RefinementPackageListCtrl->SetItemCount(0);
		ContainedParticlesListCtrl->SetItemCount(0);
		Active3DReferencesListCtrl->SetItemCount(0);

		StackFileNameText->SetLabel("");
		StackBoxSizeText->SetLabel("");
		NumberofClassesText->SetLabel("");
		NumberofRefinementsText->SetLabel("");
		LastRefinementIDText->SetLabel("");
	}

//	RefinementPackageListCtrl->Thaw();
	Thaw();
}

void MyRefinementPackageAssetPanel::ReDrawActiveReferences()
{

	Active3DReferencesListCtrl->ClearAll();
	Active3DReferencesListCtrl->InsertColumn(0, "Class No.", wxLIST_FORMAT_LEFT);
	Active3DReferencesListCtrl->InsertColumn(1, "Reference Volume", wxLIST_FORMAT_LEFT);

	Active3DReferencesListCtrl->SetItemCount(all_refinement_packages.Item(selected_refinement_package).references_for_next_refinement.GetCount());

	if (all_refinement_packages.Item(selected_refinement_package).references_for_next_refinement.GetCount() > 0)
	{
		Active3DReferencesListCtrl->RefreshItems(0, all_refinement_packages.Item(selected_refinement_package).references_for_next_refinement.GetCount() -1);

		Active3DReferencesListCtrl->SetColumnWidth(0, Active3DReferencesListCtrl->ReturnGuessAtColumnTextWidth(0));
		Active3DReferencesListCtrl->SetColumnWidth(1, Active3DReferencesListCtrl->ReturnGuessAtColumnTextWidth(1));
	}

}

void MyRefinementPackageAssetPanel::OnUpdateUI(wxUpdateUIEvent& event)
{
	if (main_frame->current_project.is_open == true)
	{
		Enable(true);

		if (is_dirty == true)
		{
			FillRefinementPackages();
			is_dirty = false;
		}

		if (selected_refinement_package >= 0)
		{
			RenameButton->Enable(true);
			DeleteButton->Enable(true);
		}
		else
		{
			RenameButton->Enable(false);
			DeleteButton->Enable(false);
		}

	}
	else Enable(false);

}


void MyRefinementPackageAssetPanel::ImportAllFromDatabase()
{
	int counter;
	RefinementPackage *temp_package;

	all_refinement_packages.Clear();


	// Now the groups..

	main_frame->current_project.database.BeginAllRefinementPackagesSelect();

	while (main_frame->current_project.database.last_return_code == SQLITE_ROW)
	{
		temp_package = main_frame->current_project.database.GetNextRefinementPackage();
		if (temp_package->asset_id > current_asset_number) current_asset_number = temp_package->asset_id;

		all_refinement_packages.Add(temp_package);

	}

	main_frame->current_project.database.EndAllRefinementPackagesSelect();

	main_frame->DirtyRefinementPackages();
}

void MyRefinementPackageAssetPanel::MouseVeto( wxMouseEvent& event )
{
	//Do nothing

}

void MyRefinementPackageAssetPanel::MouseCheckPackagesVeto( wxMouseEvent& event )
{
	VetoInvalidMouse(RefinementPackageListCtrl, event);

}

void MyRefinementPackageAssetPanel::MouseCheckParticlesVeto( wxMouseEvent& event )
{
	VetoInvalidMouse(ContainedParticlesListCtrl, event);
}

void MyRefinementPackageAssetPanel::VetoInvalidMouse( wxListCtrl *wanted_list, wxMouseEvent& event )
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

void MyRefinementPackageAssetPanel::OnMotion(wxMouseEvent& event)
{
	if (should_veto_motion == false) event.Skip();
}

void MyRefinementPackageAssetPanel::OnPackageFocusChange( wxListEvent& event )
{
	if (event.GetIndex() != selected_refinement_package)
	{
		selected_refinement_package = event.GetIndex();
		is_dirty = true;
	}

	//wxPrintf("Selected refinement package = %li\n", selected_refinement_package);

	event.Skip();
}

void MyRefinementPackageAssetPanel::OnPackageActivated( wxListEvent& event )
{
	RefinementPackageListCtrl->EditLabel(event.GetIndex());
}

void MyRefinementPackageAssetPanel::OnVolumeListItemActivated( wxListEvent& event )
{
	MyVolumeChooserDialog *dialog = new MyVolumeChooserDialog(this);
	dialog->ComboBox->SetSelection(volume_asset_panel->ReturnArrayPositionFromAssetID(all_refinement_packages.Item(selected_refinement_package).references_for_next_refinement.Item(event.GetIndex())) + 1);
	if (dialog->ShowModal() == wxID_OK)
	{
		if (dialog->selected_volume_id != all_refinement_packages.Item(selected_refinement_package).references_for_next_refinement.Item(event.GetIndex()))
		{
			all_refinement_packages.Item(selected_refinement_package).references_for_next_refinement.Item(event.GetIndex()) = dialog->selected_volume_id;
			dialog->Destroy();
			// Change in database..
			main_frame->current_project.database.ExecuteSQL(wxString::Format("UPDATE REFINEMENT_PACKAGE_CURRENT_REFERENCES_%li SET VOLUME_ASSET_ID=%li WHERE CLASS_NUMBER=%li;", all_refinement_packages.Item(selected_refinement_package).asset_id, dialog->selected_volume_id, event.GetIndex() + 1));

			ReDrawActiveReferences();



		}
	}

}

void MyRefinementPackageAssetPanel::OnBeginEdit( wxListEvent& event )
{
	event.Skip();
}

void MyRefinementPackageAssetPanel::OnEndEdit( wxListEvent& event )
{

	if (event.GetLabel() == wxEmptyString)
	{
		RefinementPackageListCtrl->SetItemText(event.GetIndex(), all_refinement_packages.Item(event.GetIndex()).name);
		event.Veto();
	}
	else
	{
		if (all_refinement_packages.Item(event.GetIndex()).name != event.GetLabel())
		{
			all_refinement_packages.Item(event.GetIndex()).name = event.GetLabel();

			// do in database also..

			wxString sql_command = wxString::Format("UPDATE REFINEMENT_PACKAGE_ASSETS SET NAME='%s' WHERE REFINEMENT_PACKAGE_ASSET_ID=%li", all_refinement_packages.Item(event.GetIndex()).name, all_refinement_packages.Item(event.GetIndex()).asset_id);
			main_frame->current_project.database.ExecuteSQL(sql_command.ToUTF8().data());

			main_frame->DirtyRefinementPackages();
			event.Skip();
		}
		else event.Veto();
	}

}

void MyRefinementPackageAssetPanel::Reset()
{
	all_refinement_packages.Clear();
	main_frame->DirtyRefinementPackages();
}


long MyRefinementPackageAssetPanel::ReturnArrayPositionFromAssetID(long wanted_asset_id)
{
	for (long counter = 0; counter < all_refinement_packages.GetCount(); counter++)
	{
		if (all_refinement_packages[counter].asset_id == wanted_asset_id) return counter;
	}

	return -1;

}

