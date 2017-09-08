//#include "../core/core_headers.h"
#include "../core/gui_core_headers.h"

extern MyImageAssetPanel *image_asset_panel;
extern MyRefinementPackageAssetPanel *refinement_package_asset_panel;

ExportRefinementPackageWizard::ExportRefinementPackageWizard( wxWindow* parent )
:
ExportRefinementPackageWizardParent( parent )
{
	SetPageSize(wxSize(600,400));

	current_package = &refinement_package_asset_panel->all_refinement_packages.Item(refinement_package_asset_panel->selected_refinement_package);

	ParameterSelectPanel->FillComboBox(refinement_package_asset_panel->selected_refinement_package, true);
	//ParameterSelectPanel->AssetComboBox->Bind(wxEVT_COMMAND_COMBOBOX_SELECTED, &ExportRefinementPackageWizard::OnParamsComboBox, this);

	ClassComboBox->Freeze();

	for (int class_counter = 0; class_counter < current_package->number_of_classes; class_counter++)
	{
		ClassComboBox->Append(wxString::Format("Class #%i", class_counter + 1));
	}

	ClassComboBox->SetSelection(0);
}

ExportRefinementPackageWizard::~ExportRefinementPackageWizard()
{
	//ParameterSelectPanel->AssetComboBox->Unbind(wxEVT_COMMAND_COMBOBOX_SELECTED, &ExportRefinementPackageWizard::OnParamsComboBox, this);
}

void ExportRefinementPackageWizard::OnParamsComboBox( wxCommandEvent& event )
{


}

void ExportRefinementPackageWizard::CheckPaths()
{
	if (GetCurrentPage() ==  m_pages.Item(2))
	{
		Freeze();

		EnableNextButton();

		// if the stack file ends in .mrc and we have relion, change to mrcs and vice versa for frealign
		wxFileName current_stack_filename = ParticleStackFileTextCtrl->GetLineText(0);
		wxFileName current_meta_filename = MetaDataFileTextCtrl->GetLineText(0);


		if (ParticleStackFileTextCtrl->GetLineText(0).IsEmpty() == true) DisableNextButton();
		else
		{
			if (FrealignRadioButton->GetValue() == true)
			{
				if (current_stack_filename.GetExt() != "mrc")
				{
					current_stack_filename.SetExt("mrc");
					ParticleStackFileTextCtrl->SetValue(current_stack_filename.GetFullPath());
				}
			}
			else
			if (RelionRadioButton->GetValue() == true)
			{
				if (current_stack_filename.GetExt() != "mrcs")
				{
					current_stack_filename.SetExt("mrcs");
					ParticleStackFileTextCtrl->SetValue(current_stack_filename.GetFullPath());
				}
			}
		}

		if (MetaDataFileTextCtrl->GetLineText(0).IsEmpty() == true) DisableNextButton();
		else
		{
			if (FrealignRadioButton->GetValue() == true)
			{
				if (current_meta_filename.GetExt() != "par")
				{
					current_meta_filename.SetExt("par");
					MetaDataFileTextCtrl->SetValue(current_meta_filename.GetFullPath());
				}
			}
			else
			if (RelionRadioButton->GetValue() == true)
			{
				if (current_meta_filename.GetExt() != "star")
				{
					current_meta_filename.SetExt("star");
					MetaDataFileTextCtrl->SetValue(current_meta_filename.GetFullPath());
				}
			}
		}

		Thaw();

	}

}

void ExportRefinementPackageWizard::OnStackBrowseButtonClick( wxCommandEvent& event )
{
	ProperOverwriteCheckSaveDialog *saveFileDialog;

	if (FrealignRadioButton->GetValue() == true)
	{
		saveFileDialog = new ProperOverwriteCheckSaveDialog(this, _("Save MRC stack file"), "MRC files (*.mrc)|*.mrc", ".mrc");
	}
	else
	if (RelionRadioButton->GetValue() == true)
	{
		saveFileDialog = new ProperOverwriteCheckSaveDialog(this, _("Save MRC stack file"), "MRC files (*.mrcs)|*.mrcs", ".mrcs");
	}

	if (saveFileDialog->ShowModal() == wxID_OK)
	{
		ParticleStackFileTextCtrl->SetValue(saveFileDialog->ReturnProperPath());
	}


//	CheckPaths();
	saveFileDialog->Destroy();

}

void ExportRefinementPackageWizard::OnMetaBrowseButtonClick( wxCommandEvent& event )
{
	ProperOverwriteCheckSaveDialog *saveFileDialog;

	if (FrealignRadioButton->GetValue() == true)
	{
		saveFileDialog = new ProperOverwriteCheckSaveDialog(this, _("Save PAR file"), "PAR files (*.par)|*.par", ".par");
	}
	else
	if (RelionRadioButton->GetValue() == true)
	{
		saveFileDialog = new ProperOverwriteCheckSaveDialog(this, _("Save STAR file"), "STAR files (*.star)|*.star", ".star");
	}

	if (saveFileDialog->ShowModal() == wxID_OK)
	{
		MetaDataFileTextCtrl->SetValue(saveFileDialog->ReturnProperPath());
	}

	CheckPaths();
	saveFileDialog->Destroy();
}

void ExportRefinementPackageWizard::OnPageChanged(  wxWizardEvent& event  )
{
	if (event.GetPage() == m_pages.Item(0))
	{
		EnableNextButton();
	}
	else
	if (event.GetPage() == m_pages.Item(2))
	{
		if (FrealignRadioButton->GetValue() == true)
		{
			MetaFilenameStaticText->SetLabel("Output PAR Filename :-    ");

		}
		else
		if (RelionRadioButton->GetValue() == true)
		{
			MetaFilenameStaticText->SetLabel("Output STAR Filename :-   ");
		}

		CheckPaths();
	}
}

void ExportRefinementPackageWizard::OnPathChange( wxCommandEvent& event )
{
	Freeze();
	EnableNextButton();
	if (ParticleStackFileTextCtrl->GetLineText(0).IsEmpty() == true) DisableNextButton();
	if (MetaDataFileTextCtrl->GetLineText(0).IsEmpty() == true) DisableNextButton();
	Thaw();

}

void ExportRefinementPackageWizard::OnUpdateUI(wxUpdateUIEvent& event)
{


}

void ExportRefinementPackageWizard::OnFinished(  wxWizardEvent& event  )
{
	// get the current refinement package..

	long particle_counter;

	// Are we doing frealign, or relion?

	if (FrealignRadioButton->GetValue() == true) // Frealign
	{
		OneSecondProgressDialog *my_dialog = new OneSecondProgressDialog ("Export To Frealign", "Writing PAR file...", 2, this, wxPD_AUTO_HIDE| wxPD_APP_MODAL);

		// get the refinement..

		Refinement *current_refinement = main_frame->current_project.database.GetRefinementByID(current_package->refinement_ids[ParameterSelectPanel->GetSelection()]);

		// write the parameter file..

		current_refinement->WriteSingleClassFrealignParameterFile(MetaDataFileTextCtrl->GetLineText(0), ClassComboBox->GetSelection());

		my_dialog->Update(1, "Copying Stack...");
		// copy the stack..

		wxCopyFile(current_package->stack_filename, ParticleStackFileTextCtrl->GetLineText(0));

		my_dialog->Destroy();
		delete current_refinement;
	}
	else
	if (RelionRadioButton->GetValue() == true) // Relion
	{
		wxFileName output_stack_filename = ParticleStackFileTextCtrl->GetLineText(0);
		wxFileName relion_star_filename = MetaDataFileTextCtrl->GetLineText(0);

		relion_star_filename.SetExt("star");

		wxTextFile *relion_star_file = new wxTextFile(relion_star_filename.GetFullPath());

		MRCFile input_stack(current_package->stack_filename.ToStdString(), false);
		MRCFile output_stack(output_stack_filename.GetFullPath().ToStdString(), true);
		output_stack.SetPixelSize(current_package->contained_particles[0].pixel_size);
		Image particle_image;

		double particle_radius = current_package->estimated_particle_size_in_angstroms / 2;

		wxString micrograph_filename;

		RefinementPackageParticleInfo current_particle;
		RefinementResult current_refinement_result;

		if (relion_star_file->Exists())
		{
			relion_star_file->Open();
			relion_star_file->Clear();
		}
		else
		{
			relion_star_file->Create();
		}

		// Write headers
		relion_star_file->AddLine(wxString(" "));
		relion_star_file->AddLine(wxString("data_"));
		relion_star_file->AddLine(wxString(" "));
		relion_star_file->AddLine(wxString("loop_"));
		relion_star_file->AddLine(wxString("_rlnMicrographName #1"));
		relion_star_file->AddLine(wxString("_rlnCoordinateX #2"));
		relion_star_file->AddLine(wxString("_rlnCoordinateY #3"));
		relion_star_file->AddLine(wxString("_rlnImageName #4"));
		relion_star_file->AddLine(wxString("_rlnDefocusU #5"));
		relion_star_file->AddLine(wxString("_rlnDefocusV #6"));
		relion_star_file->AddLine(wxString("_rlnDefocusAngle #7"));
		relion_star_file->AddLine(wxString("_rlnPhaseShift #8"));
		relion_star_file->AddLine(wxString("_rlnVoltage #9"));
		relion_star_file->AddLine(wxString("_rlnSphericalAberration #10"));
		relion_star_file->AddLine(wxString("_rlnAmplitudeContrast #11"));
		relion_star_file->AddLine(wxString("_rlnMagnification #12"));
		relion_star_file->AddLine(wxString("_rlnDetectorPixelSize #13"));
		relion_star_file->AddLine(wxString("_rlnAngleRot #14"));
		relion_star_file->AddLine(wxString("_rlnAngleTilt #15"));
		relion_star_file->AddLine(wxString("_rlnAnglePsi #16"));
		relion_star_file->AddLine(wxString("_rlnOriginX #17"));
		relion_star_file->AddLine(wxString("_rlnOriginY #18"));


	/*	// Write headers
		relion_star_file->AddLine(wxString(" "));
		relion_star_file->AddLine(wxString("data_"));
		relion_star_file->AddLine(wxString(" "));
		relion_star_file->AddLine(wxString("loop_"));
		relion_star_file->AddLine(wxString("_rlnMicrographName #1"));
		relion_star_file->AddLine(wxString("_rlnCoordinateX #2"));
		relion_star_file->AddLine(wxString("_rlnCoordinateY #3"));
		relion_star_file->AddLine(wxString("_rlnImageName #4"));
		relion_star_file->AddLine(wxString("_rlnDefocusU #5"));
		relion_star_file->AddLine(wxString("_rlnDefocusV #6"));
		relion_star_file->AddLine(wxString("_rlnDefocusAngle #7"));
		relion_star_file->AddLine(wxString("_rlnPhaseShift #8"));
		relion_star_file->AddLine(wxString("_rlnVoltage #9"));
		relion_star_file->AddLine(wxString("_rlnSphericalAberration #10"));
		relion_star_file->AddLine(wxString("_rlnAmplitudeContrast #11"));
		relion_star_file->AddLine(wxString("_rlnMagnification #12"));
		relion_star_file->AddLine(wxString("_rlnDetectorPixelSize #13"));*/

		OneSecondProgressDialog *my_dialog = new OneSecondProgressDialog ("Export To Relion", "Exporting...", current_package->contained_particles.GetCount(), this);
		Refinement *current_refinement = main_frame->current_project.database.GetRefinementByID(current_package->refinement_ids[ParameterSelectPanel->GetSelection()]);

		for (particle_counter = 0; particle_counter < current_package->contained_particles.GetCount(); particle_counter ++ )
		{

			current_particle = current_package->ReturnParticleInfoByPositionInStack(particle_counter + 1);
			current_refinement_result = current_refinement->ReturnRefinementResultByClassAndPositionInStack(ClassComboBox->GetSelection(), particle_counter + 1);

			particle_image.ReadSlice(&input_stack, particle_counter + 1);
			if (current_package->stack_has_white_protein == false) particle_image.InvertRealValues();
			particle_image.ZeroFloatAndNormalize(1.0, particle_radius /current_particle.pixel_size, true);
			particle_image.WriteSlice(&output_stack, particle_counter + 1);

			// if we have micrograph info, may aswell include it..

			if (current_particle.parent_image_id >= 0)
			{
				micrograph_filename = image_asset_panel->ReturnAssetLongFilename(image_asset_panel->ReturnArrayPositionFromAssetID(current_particle.parent_image_id));
			}
			else
			{
				micrograph_filename = "unkown.mrc";
			}
/*
			relion_star_file->AddLine(wxString::Format("%s %f %f %06li@%s %f %f %f %f %f %f %f %f %f",	micrograph_filename,
																										current_particle.x_pos / current_particle.pixel_size,
																										current_particle.y_pos / current_particle.pixel_size,
																										particle_counter + 1,
																										output_stack_filename.GetFullPath(),
																										current_refinement_result.defocus1,
																										current_refinement_result.defocus2,
																										current_refinement_result.defocus_angle,
																										current_refinement_result.phase_shift,
																										current_particle.microscope_voltage,
																										current_particle.spherical_aberration,
																										current_particle.amplitude_contrast,
																										10000.0f,
																										current_particle.pixel_size));*/

			relion_star_file->AddLine(wxString::Format("%s %f %f %06li@%s %f %f %f %f %f %f %f %f %f %f %f %f %f %f",	micrograph_filename,
																										current_particle.x_pos / current_particle.pixel_size,
																										current_particle.y_pos / current_particle.pixel_size,
																										particle_counter + 1,
																										output_stack_filename.GetFullPath(),
																										current_refinement_result.defocus1,
																										current_refinement_result.defocus2,
																										current_refinement_result.defocus_angle,
																										current_refinement_result.phase_shift,
																										current_particle.microscope_voltage,
																										current_particle.spherical_aberration,
																										current_particle.amplitude_contrast,
																										10000.0f,
																										current_particle.pixel_size,
																										current_refinement_result.phi,
																										current_refinement_result.theta,
																										current_refinement_result.psi,
																										-current_refinement_result.xshift / current_particle.pixel_size,
																										-current_refinement_result.yshift / current_particle.pixel_size));

			my_dialog->Update(particle_counter+1);
		}

		relion_star_file->Write();
		relion_star_file->Close();

		input_stack.CloseFile();
		output_stack.CloseFile();

		delete relion_star_file	;
		delete current_refinement;

		my_dialog->Destroy();


	}

}
