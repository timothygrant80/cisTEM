//#include "../core/core_headers.h"
#include "../core/gui_core_headers.h"

extern MyRefinementPackageAssetPanel *refinement_package_asset_panel;
extern MyRefinementResultsPanel *refinement_results_panel;


MyRefinementResultsPanel::MyRefinementResultsPanel( wxWindow* parent )
:
RefinementResultsPanel( parent )
{
	is_dirty = false;
	input_params_are_dirty = false;
	current_class = 0;
	FSCPlotPanel->Clear();
	currently_displayed_refinement = NULL;

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
	RefinementPackageComboBox->FillWithRefinementPackages();
	FillInputParametersComboBox();
}

void MyRefinementResultsPanel::FillInputParametersComboBox(void)
{
	if (RefinementPackageComboBox->GetSelection() >= 0)
	{
		InputParametersComboBox->FillWithRefinements(RefinementPackageComboBox->GetSelection());

		UpdateCachedRefinement();
		FillParameterListCtrl();
		FSCPlotPanel->AddRefinement(currently_displayed_refinement);
		FillAngles();
	}
}

void MyRefinementResultsPanel::UpdateCachedRefinement()
{
	wxPrintf("refinement package selection = %i, parameter selcetion = %i\n", refinement_results_panel->RefinementPackageComboBox->GetSelection(), refinement_results_panel->InputParametersComboBox->GetSelection());
	if (currently_displayed_refinement == NULL || refinement_package_asset_panel->all_refinement_packages[refinement_results_panel->RefinementPackageComboBox->GetSelection()].refinement_ids[refinement_results_panel->InputParametersComboBox->GetSelection()] != currently_displayed_refinement->refinement_id)
	{
		wxProgressDialog progress_dialog("Please wait", "Retrieving refinement result from database...", 0, this);

		if (currently_displayed_refinement != NULL) delete currently_displayed_refinement;
		currently_displayed_refinement = main_frame->current_project.database.GetRefinementByID(refinement_package_asset_panel->all_refinement_packages[refinement_results_panel->RefinementPackageComboBox->GetSelection()].refinement_ids[refinement_results_panel->InputParametersComboBox->GetSelection()]);
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
		//	FillAngles();
		}

		if (input_params_are_dirty == true)
		{
			input_params_are_dirty = false;
			FillInputParametersComboBox();
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
		UpdateCachedRefinement();
		FillParameterListCtrl();
		FSCPlotPanel->AddRefinement(currently_displayed_refinement);
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
		AngularPlotPanel->Freeze();
		AngularPlotPanel->Clear();
		AngularPlotPanel->SetSymmetryAndNumber(refinement_package_asset_panel->all_refinement_packages[RefinementPackageComboBox->GetSelection()].symmetry, refinement_package_asset_panel->all_refinement_packages[RefinementPackageComboBox->GetSelection()].contained_particles.GetCount());

		for (long particle_counter = 0; particle_counter < refinement_package_asset_panel->all_refinement_packages[RefinementPackageComboBox->GetSelection()].contained_particles.GetCount(); particle_counter++)
		{

			AngularPlotPanel->AddRefinementResult(&currently_displayed_refinement->class_refinement_results[current_class].particle_refinement_results[particle_counter]);

		}
		AngularPlotPanel->Thaw();
		AngularPlotPanel->Refresh();

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
		ParameterListCtrl->InsertColumn(9, wxT("Phase Shift (rad)"), wxLIST_FORMAT_CENTRE,  wxLIST_AUTOSIZE_USEHEADER );
		ParameterListCtrl->InsertColumn(10, wxT("Occupancy"), wxLIST_FORMAT_CENTRE,  wxLIST_AUTOSIZE_USEHEADER );
		ParameterListCtrl->InsertColumn(11, wxT("logP"), wxLIST_FORMAT_CENTRE,  wxLIST_AUTOSIZE_USEHEADER );
		ParameterListCtrl->InsertColumn(12, wxT("Sigma"), wxLIST_FORMAT_CENTRE,  wxLIST_AUTOSIZE_USEHEADER );
		ParameterListCtrl->InsertColumn(13, wxT("Score"), wxLIST_FORMAT_CENTRE,  wxLIST_AUTOSIZE_USEHEADER );
		ParameterListCtrl->InsertColumn(14, wxT("Image Active?"), wxLIST_FORMAT_CENTRE,  wxLIST_AUTOSIZE_USEHEADER );

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
