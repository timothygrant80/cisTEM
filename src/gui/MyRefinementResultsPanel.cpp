//#include "../core/core_headers.h"
#include "../core/gui_core_headers.h"

extern MyRefinementPackageAssetPanel *refinement_package_asset_panel;
extern MyRefinementResultsPanel *refinement_results_panel;


MyRefinementResultsPanel::MyRefinementResultsPanel( wxWindow* parent )
:
RefinementResultsPanel( parent )
{
	is_dirty = true;
	current_class = 0;
	FSCPlotPanel->Clear();

	FSCPlotPanel->ClassComboBox->Connect( wxEVT_COMMAND_COMBOBOX_SELECTED, wxCommandEventHandler( MyRefinementResultsPanel::OnClassComboBoxChange ), NULL, this );
}

void MyRefinementResultsPanel::OnClassComboBoxChange( wxCommandEvent& event )
{
	//wxPrintf("Changed\n");
	current_class = FSCPlotPanel->ClassComboBox->GetSelection();
	FillAngles();
	ParameterListCtrl->RefreshItems(0, ParameterListCtrl->GetItemCount() - 1);
	FSCPlotPanel->PlotCurrentClass();
	event.Skip();
}

void MyRefinementResultsPanel::FillRefinementPackageComboBox(void)
{
	RefinementPackageComboBox->Freeze();
	RefinementPackageComboBox->Clear();
	RefinementPackageComboBox->ChangeValue("");
	AppendRefinementPackagesToComboBox(RefinementPackageComboBox);
	RefinementPackageComboBox->SetSelection(RefinementPackageComboBox->GetCount() -1);
	RefinementPackageComboBox->Thaw();

	FillInputParametersComboBox();
}

void MyRefinementResultsPanel::FillInputParametersComboBox(void)
{
	if (RefinementPackageComboBox->GetSelection() >= 0)
	{
		InputParametersComboBox->Freeze();
		InputParametersComboBox->Clear();
		InputParametersComboBox->ChangeValue("");

		for (int counter = 0; counter < refinement_package_asset_panel->all_refinement_packages[RefinementPackageComboBox->GetSelection()].refinement_ids.GetCount(); counter++)
		{
			InputParametersComboBox->Append(refinement_package_asset_panel->ReturnPointerToRefinementByRefinementID(refinement_package_asset_panel->all_refinement_packages[RefinementPackageComboBox->GetSelection()].refinement_ids[counter])->name);

		}

		InputParametersComboBox->SetSelection(InputParametersComboBox->GetCount() - 1);
		InputParametersComboBox->Thaw();

		FillParameterListCtrl();
		FSCPlotPanel->AddRefinement(refinement_package_asset_panel->ReturnPointerToRefinementByRefinementID(refinement_package_asset_panel->all_refinement_packages[refinement_results_panel->RefinementPackageComboBox->GetSelection()].refinement_ids[refinement_results_panel->InputParametersComboBox->GetSelection()]));

	}



}

void MyRefinementResultsPanel::OnUpdateUI( wxUpdateUIEvent& event )
{
	if (main_frame->current_project.is_open == false)
	{
		Enable(false);
		RefinementPackageComboBox->Clear();
		RefinementPackageComboBox->ChangeValue("");
		InputParametersComboBox->Clear();
		InputParametersComboBox->ChangeValue("");
		ParameterListCtrl->ClearAll();
		ParameterListCtrl->SetItemCount(0);
		FSCPlotPanel->Clear();

	}
	else
	{
		Enable(true);

		if (is_dirty == true)
		{
			is_dirty = false;
			FillRefinementPackageComboBox();
			//AngularPlotPanel->Clear();
			FillAngles();
		}
	}
}
void MyRefinementResultsPanel::OnRefinementPackageComboBox( wxCommandEvent& event )
{
	if (RefinementPackageComboBox->GetSelection() >= 0)
	{
		FillInputParametersComboBox();
	}
}

void MyRefinementResultsPanel::OnInputParametersComboBox( wxCommandEvent& event )
{
	if (RefinementPackageComboBox->GetSelection() >= 0)
	{
		FillParameterListCtrl();
		FSCPlotPanel->AddRefinement(refinement_package_asset_panel->ReturnPointerToRefinementByRefinementID(refinement_package_asset_panel->all_refinement_packages[refinement_results_panel->RefinementPackageComboBox->GetSelection()].refinement_ids[refinement_results_panel->InputParametersComboBox->GetSelection()]));
		FillAngles();
		//AngularPlotPanel->Clear();
	}

}

void MyRefinementResultsPanel::OnPlotButtonClick( wxCommandEvent& event )
{
	FillAngles();
}

void MyRefinementResultsPanel::FillAngles()
{
	if (RefinementPackageComboBox->GetSelection() >= 0)
	{
		Refinement *current_refinement;
		current_refinement = refinement_package_asset_panel->ReturnPointerToRefinementByRefinementID( refinement_package_asset_panel->all_refinement_packages[RefinementPackageComboBox->GetSelection()].refinement_ids[InputParametersComboBox->GetCurrentSelection()]);

		//OneSecondProgressDialog *my_dialog = new OneSecondProgressDialog ("Plot Angles", "Plotting Angles", refinement_package_asset_panel->all_refinement_packages[RefinementPackageComboBox->GetSelection()].contained_particles.GetCount(), this);
		AngularPlotPanel->Freeze();
		AngularPlotPanel->Clear();
		AngularPlotPanel->SetSymmetryAndNumber(refinement_package_asset_panel->all_refinement_packages[RefinementPackageComboBox->GetSelection()].symmetry, refinement_package_asset_panel->all_refinement_packages[RefinementPackageComboBox->GetSelection()].contained_particles.GetCount());
		for (long particle_counter = 0; particle_counter < refinement_package_asset_panel->all_refinement_packages[RefinementPackageComboBox->GetSelection()].contained_particles.GetCount(); particle_counter++)
		{
			//wxPrintf("Adding Refinement \n");
			AngularPlotPanel->AddRefinementResult(&current_refinement->class_refinement_results[current_class].particle_refinement_results[particle_counter]);
			//my_dialog->Update(particle_counter + 1);
		}
		AngularPlotPanel->Thaw();
		AngularPlotPanel->Refresh();
	//	delete my_dialog;
	}

}


void MyRefinementResultsPanel::FillParameterListCtrl()
{
	if (RefinementPackageComboBox->GetSelection() >= 0)
	{

		ParameterListCtrl->Freeze();
		ParameterListCtrl->ClearAll();
		ParameterListCtrl->InsertColumn(0, wxT("Position In Stack."), wxLIST_FORMAT_CENTRE,  wxLIST_AUTOSIZE_USEHEADER );
		ParameterListCtrl->InsertColumn(1, wxT("Psi Angle (°)"), wxLIST_FORMAT_CENTRE,  wxLIST_AUTOSIZE_USEHEADER );
		ParameterListCtrl->InsertColumn(2, wxT("Theta Angle (°)"), wxLIST_FORMAT_CENTRE,  wxLIST_AUTOSIZE_USEHEADER );
		ParameterListCtrl->InsertColumn(3, wxT("Phi Angle (°)"), wxLIST_FORMAT_CENTRE,  wxLIST_AUTOSIZE_USEHEADER );
		ParameterListCtrl->InsertColumn(4, wxT("X-Shift (Å)"), wxLIST_FORMAT_CENTRE,  wxLIST_AUTOSIZE_USEHEADER );
		ParameterListCtrl->InsertColumn(5, wxT("Y-Shift (Å)"), wxLIST_FORMAT_CENTRE,  wxLIST_AUTOSIZE_USEHEADER );
		ParameterListCtrl->InsertColumn(6, wxT("Defocus 1 (Å)"), wxLIST_FORMAT_CENTRE,  wxLIST_AUTOSIZE_USEHEADER );
		ParameterListCtrl->InsertColumn(7, wxT("Defocus 2 (Å)"), wxLIST_FORMAT_CENTRE,  wxLIST_AUTOSIZE_USEHEADER );
		ParameterListCtrl->InsertColumn(8, wxT("Defocus Angle (°)"), wxLIST_FORMAT_CENTRE,  wxLIST_AUTOSIZE_USEHEADER );
		ParameterListCtrl->InsertColumn(9, wxT("Occupancy"), wxLIST_FORMAT_CENTRE,  wxLIST_AUTOSIZE_USEHEADER );
		ParameterListCtrl->InsertColumn(10, wxT("logP"), wxLIST_FORMAT_CENTRE,  wxLIST_AUTOSIZE_USEHEADER );
		ParameterListCtrl->InsertColumn(11, wxT("Sigma"), wxLIST_FORMAT_CENTRE,  wxLIST_AUTOSIZE_USEHEADER );
		ParameterListCtrl->InsertColumn(12, wxT("Score"), wxLIST_FORMAT_CENTRE,  wxLIST_AUTOSIZE_USEHEADER );
		ParameterListCtrl->InsertColumn(13, wxT("Score Change"), wxLIST_FORMAT_CENTRE,  wxLIST_AUTOSIZE_USEHEADER );

	    ParameterListCtrl->SetItemCount(refinement_package_asset_panel->all_refinement_packages[RefinementPackageComboBox->GetSelection()].contained_particles.GetCount());
		//ParameterListCtrl->SetItemCount(1);
		//ParameterListCtrl->RefreshItems(0, 0);
		ParameterListCtrl->RefreshItems(0, ParameterListCtrl->GetItemCount() - 1);

		for (int counter = 0; counter < ParameterListCtrl->GetColumnCount(); counter++)
		{
			ParameterListCtrl->SetColumnWidth(counter, ParameterListCtrl->ReturnGuessAtColumnTextWidth(counter));
		}
		ParameterListCtrl->Thaw();
	}

}
