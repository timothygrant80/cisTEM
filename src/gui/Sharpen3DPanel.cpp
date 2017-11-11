#include "../core/gui_core_headers.h"

extern MyVolumeAssetPanel *volume_asset_panel;
extern MyRefinementPackageAssetPanel *refinement_package_asset_panel;

Sharpen3DPanel::Sharpen3DPanel( wxWindow* parent )
:
Sharpen3DPanelParent( parent )
{
	ResultDisplayPanel->Initialise(START_WITH_FOURIER_SCALING | DO_NOT_SHOW_STATUS_BAR | START_WITH_NO_LABEL | FIRST_LOCATION_ONLY);
	GuinierPlot->Initialise("", "", true, 20, 20, 20, 20, false);
	SetInfo();

	volumes_are_dirty = false;
	have_a_result_in_memory = false;
	running_a_job = false;
	active_refinement = NULL;
	active_result = NULL;

	guage_timer = NULL;

	VolumeComboBox->AssetComboBox->Bind(wxEVT_COMMAND_COMBOBOX_SELECTED, &Sharpen3DPanel::OnVolumeComboBox, this);
	Bind(RETURN_SHARPENING_RESULTS_EVT, &Sharpen3DPanel::OnSharpenThreadComplete, this);
	Bind(wxEVT_TIMER, &Sharpen3DPanel::OnGuageTimer, this);
}

void Sharpen3DPanel::OnGuageTimer(wxTimerEvent& event)
{
	ProgressGuage->Pulse();
}

void Sharpen3DPanel::SetInfo()
{

	wxLogNull *suppress_png_warnings = new wxLogNull;

	#include "icons/vsv_sharpening.cpp"
	wxBitmap vsv_sharpening_bmp = wxBITMAP_PNG_FROM_DATA(vsv_sharpening);
	delete suppress_png_warnings;

	InfoText->GetCaret()->Hide();

	InfoText->BeginSuppressUndo();
	InfoText->BeginAlignment(wxTEXT_ALIGNMENT_CENTRE);
	InfoText->BeginBold();
	InfoText->BeginUnderline();
	InfoText->BeginFontSize(14);
	InfoText->WriteText(wxT("Sharpen 3D"));
	InfoText->EndFontSize();
	InfoText->EndBold();
	InfoText->EndUnderline();
	InfoText->Newline();
	InfoText->Newline();
	InfoText->EndAlignment();

	InfoText->BeginAlignment(wxTEXT_ALIGNMENT_LEFT);
	InfoText->WriteText(wxT("The high-resolution signal in 3D reconstructions is usually dampened by various factors, including the envelope function affecting the CTF of the microscope, the modulation transfer function (MTF) of the detector, beam-induced motion, and alignment and interpolation errors introduced during image processing. Structural heterogeneity present in the particles may also contribute. It is common practice to express this dampening by a B-factor, expressed in Å2 and given as exp(-0.25 B/d2) where d is the resolution (in Å) at which the dampening occurs. To visualize the high-resolution details in a reconstructed map, its amplitudes have to be restored by applying a negative B-factor, thereby sharpening the map."));
	InfoText->Newline();
	InfoText->Newline();
	InfoText->EndAlignment();

	InfoText->BeginAlignment(wxTEXT_ALIGNMENT_LEFT);
	InfoText->WriteText(wxT("This panel provides the user with a few parameters to sharpen a map. In the simplest case, a map can be sharpened by providing a negative B-factor. However, because B-factor sharpening involves multiplication with an exponential function, the noise at high resolution can easily be over-amplified. A more robust method to restore the amplitudes at high resolution that also works if the dampening cannot be described with a simple B-factor can be achieved by flattening (i.e. whitening) the amplitude spectrum at high resolution. The panel provides a flexible way to combine B-factor sharpening and spectral flattening to optimize the visibility of high-resolution details in the final map. Optionally, the resolution statistics can be used to apply figure-of-merit (FOM) weighting (Rosenthal & Henderson, 2003) and a 3D mask can be supplied to remove background noise from the map for more accurate sharpening. Finally, the handedness of the map can be inverted if it is wrong and the real-space dampening of densities near the edge of the reconstruction box due to trilinear interpolation used during reconstruction can be corrected. The following figure shows an example of 3-Å map of VSV polymerase (Liang et al. 2015) before and after sharpening"));
	InfoText->Newline();
	InfoText->Newline();
	InfoText->EndAlignment();

	InfoText->BeginAlignment(wxTEXT_ALIGNMENT_CENTRE);
	InfoText->WriteImage(vsv_sharpening_bmp);
	InfoText->Newline();
	InfoText->Newline();
	InfoText->EndAlignment();

	InfoText->BeginAlignment(wxTEXT_ALIGNMENT_CENTRE);
	InfoText->BeginBold();
	InfoText->BeginUnderline();
	InfoText->WriteText(wxT("Program Options"));
	InfoText->EndBold();
	InfoText->EndUnderline();
	InfoText->Newline();
	InfoText->Newline();
	InfoText->EndAlignment();

	InfoText->BeginAlignment(wxTEXT_ALIGNMENT_LEFT);
	InfoText->BeginBold();
	InfoText->WriteText(wxT("Input Volume : "));
	InfoText->EndBold();
	InfoText->WriteText(wxT("The volume (reconstruction) to be sharpened."));
	InfoText->Newline();
	InfoText->BeginBold();
	InfoText->WriteText(wxT("Supply a Mask? "));
	InfoText->EndBold();
	InfoText->WriteText(wxT("Should the volume be masked to make sharpening more accurate? If checked, a volume containing the 3D mask must be selected."));
	InfoText->Newline();
	InfoText->BeginBold();
	InfoText->WriteText(wxT("Flatten From Res. (Å) : "));
	InfoText->EndBold();
	InfoText->WriteText(wxT("The low-resolution limit of the part of the amplitude spectrum to be flattened (whitened). This should normally be a resolution beyond which the influence of the shape transform of the particle is negligible, between 8 – 10 Å."));
	InfoText->Newline();
	InfoText->BeginBold();
	InfoText->WriteText(wxT("Resolution Cut-Off (Å) : "));
	InfoText->EndBold();
	InfoText->WriteText(wxT("The high-resolution limit applied to the sharpened map. The filter edge is given by a cosine function of width specified by 'Filter Edge-Width.'"));
	InfoText->Newline();
	InfoText->BeginBold();
	InfoText->WriteText(wxT("Pre-Cut-Off B-Factor (Å2) : "));
	InfoText->EndBold();
	InfoText->WriteText(wxT("The B-factor to be applied to the low-resolution end of the amplitude spectrum, up to the point given as “Flatten From Res.” A B-factor of -90 Å2 is usually appropriate for cryo-EM maps calculated using data collected on direct detectors."));
	InfoText->Newline();
	InfoText->BeginBold();
	InfoText->WriteText(wxT("Post-Cut-Off B-Factor (Å2) : "));
	InfoText->EndBold();
	InfoText->WriteText(wxT("The B-factor to be applied to the high-resolution end of the amplitude spectrum, from to the point given as “Flatten From Res.” This will apply a B-factor after flattening the spectrum. A value between 0 and 25 Å2 is usually appropriate since the flattening should restore most of the high-resolution signal without the need for further amplification."));
	InfoText->Newline();
	InfoText->BeginBold();
	InfoText->WriteText(wxT("Filter Edge-Width (Å) : "));
	InfoText->EndBold();
	InfoText->WriteText(wxT("The width of the cosine edge of the resolution cut-off applied to the final sharpened map. The width of the cosine is given as 1/w, where w is the value entered here."));
	InfoText->Newline();
	InfoText->BeginBold();
	InfoText->WriteText(wxT("Use FOM Weighting? "));
	InfoText->EndBold();
	InfoText->WriteText(wxT("Should the sharpened map be weighted using a figure of merit (FOM) derived from the resolution statistics describing the map? (see Rosenthal & Henderson, 2003)"));
	InfoText->Newline();
	InfoText->BeginBold();
	InfoText->WriteText(wxT("SSNR Scale Factor : "));
	InfoText->EndBold();
	InfoText->WriteText(wxT("The FOM values are calculated as SSNR/(1+SSNR) where SSNR is the spectral signal-to-noise ratio of the map to be sharpened (calculated as part of the reconstruction). The scale factor allows users to change the effective SSNR values used in this calculation. This can be useful, for example, if the SSNR represents the average signal in a reconstruction but FOM weighting should be performed using a higher SSNR that represents the signal when more disordered parts are excluded from the map. A higher SSNR leads to less filtering that may be more appropriate for high-resolution details in the better parts of a map."));
	InfoText->Newline();
	InfoText->BeginBold();
	InfoText->WriteText(wxT("Inner Mask Radius (Å) : "));
	InfoText->EndBold();
	InfoText->WriteText(wxT("The radius describing the inner bounds of the particle. This is usually set to 0 unless the particle is hollow or has largely disordered density in its center, such as the genome of a spherical virus."));
	InfoText->Newline();
	InfoText->BeginBold();
	InfoText->WriteText(wxT("Outer Mask Radius (Å) : "));
	InfoText->EndBold();
	InfoText->WriteText(wxT("The radius describing the outer bounds of the particle. A spherical mask with this radius will be applied to the map before sharpening unless a 3D mask volume is supplied."));
	InfoText->Newline();
	InfoText->BeginBold();
	InfoText->WriteText(wxT("Invert Handedness?"));
	InfoText->EndBold();
	InfoText->WriteText(wxT("Should the handedness of the sharpened map be inverted? If a reconstruction was initiated using the ab-initio procedure, the handedness will be undetermined and may have to be inverted if found incorrect."));
	InfoText->Newline();
	InfoText->BeginBold();
	InfoText->WriteText(wxT("Correct Gridding Error? "));
	InfoText->EndBold();
	InfoText->WriteText(wxT("Should the reconstruction be corrected for interpolation errors that attenuate the density towards the edges?"));
	InfoText->Newline();
	InfoText->Newline();


	InfoText->BeginAlignment(wxTEXT_ALIGNMENT_CENTRE);
	InfoText->BeginBold();
	InfoText->BeginUnderline();
	InfoText->WriteText(wxT("References"));
	InfoText->EndBold();
	InfoText->EndUnderline();
	InfoText->Newline();
	InfoText->Newline();
	InfoText->EndAlignment();

	InfoText->BeginAlignment(wxTEXT_ALIGNMENT_LEFT);
	InfoText->BeginBold();
	InfoText->WriteText(wxT("Rosenthal, P. B. & Henderson, R.,"));
	InfoText->EndBold();
	InfoText->WriteText(wxT(" 2003. Optimal determination of particle orientation, absolute hand, and contrast loss in single-particle electron cryomicroscopy. J. Mol. Biol. 333, 721-745. "));
	InfoText->BeginURL("https://doi.org/10.1016/j.jmb.2003.07.013");
	InfoText->BeginUnderline();
	InfoText->BeginTextColour(*wxBLUE);

	InfoText->WriteText(wxT("doi:10.1016/j.jmb.2003.07.013"));
	InfoText->EndURL();
	InfoText->EndTextColour();
	InfoText->EndUnderline();
	InfoText->EndAlignment();
	InfoText->Newline();
	InfoText->Newline();

	InfoText->BeginAlignment(wxTEXT_ALIGNMENT_LEFT);
	InfoText->BeginBold();
	InfoText->WriteText(wxT("Liang, B., Li, Z., Jenni, S., Rahmeh, A. A., Morin, B. M., Grant, T., Grigorieff, N., Harrison, S. C., Whelan, S. P.,"));
	InfoText->EndBold();
	InfoText->WriteText(wxT(" 2015. Structure of the L protein of vesicular stomatitis virus from electron cryomicroscopy. Cell 162, 314-327."));
	InfoText->BeginURL("http://dx.doi.org/10.1016/j.cell.2015.06.018");
	InfoText->BeginUnderline();
	InfoText->BeginTextColour(*wxBLUE);
	InfoText->WriteText(wxT("doi:10.1016/j.cell.2015.06.018"));
	InfoText->EndURL();
	InfoText->EndTextColour();
	InfoText->EndUnderline();
	InfoText->EndAlignment();
	InfoText->Newline();
	InfoText->Newline();


}

void Sharpen3DPanel::OnInfoURL(wxTextUrlEvent& event)
{
	 const wxMouseEvent& ev = event.GetMouseEvent();

	 // filter out mouse moves, too many of them
	 if ( ev.Moving() ) return;

	 long start = event.GetURLStart();

	 wxTextAttr my_style;

	 InfoText->GetStyle(start, my_style);

	 // Launch the URL

	 wxLaunchDefaultBrowser(my_style.GetURL());
}

void Sharpen3DPanel::OnUpdateUI( wxUpdateUIEvent& event )
{

	// are there enough members in the selected group.

	if (main_frame->current_project.is_open == false)
	{
		Enable(false);
		InfoPanel->Enable(true);
	}
	else
	if (running_a_job == true) Enable(false);
	else
	{
		Enable(true);

		if (UseMaskCheckBox->GetValue() == true)
		{
			MaskSelectPanel->Enable(true);
			MaskSelectPanel->AssetComboBox->Enable(true);
			InnerMaskRadiusStaticText->Enable(false);
			InnerMaskRadiusTextCtrl->Enable(false);
			OuterMaskRadiusStaticText->Enable(false);
			OuterMaskRadiusTextCtrl->Enable(false);
		}
		else
		{
			MaskSelectPanel->Enable(false);
			MaskSelectPanel->AssetComboBox->Enable(false);

			if (MaskSelectPanel->AssetComboBox->GetCount() > 0)
			{
				MaskSelectPanel->AssetComboBox->Clear();
				MaskSelectPanel->AssetComboBox->ChangeValue("");
			}

			InnerMaskRadiusStaticText->Enable(true);
			InnerMaskRadiusTextCtrl->Enable(true);
			OuterMaskRadiusStaticText->Enable(true);
			OuterMaskRadiusTextCtrl->Enable(true);
		}

		if (VolumeComboBox->GetSelection() >= 0 && running_a_job == false)
		{
			RunJobButton->Enable(true);
		}
		else
		{
			RunJobButton->Enable(false);
		}

		if (active_refinement == NULL)
		{
			UseFSCWeightingStaticText->Enable(false);
			UseFSCWeightingYesButton->Enable(false);
			UseFSCWeightingNoButton->Enable(false);
			SSNRScaleFactorText->Enable(false);
			SSNRScaleFactorTextCtrl->Enable(false);
		}
		else
		{
			UseFSCWeightingStaticText->Enable(true);
			UseFSCWeightingYesButton->Enable(true);
			UseFSCWeightingNoButton->Enable(true);

			if (UseFSCWeightingYesButton->GetValue() == true)
			{
				SSNRScaleFactorText->Enable(true);
				SSNRScaleFactorTextCtrl->Enable(true);
			}
			else
			{
				SSNRScaleFactorText->Enable(false);
				SSNRScaleFactorTextCtrl->Enable(false);
			}
		}


		if (have_a_result_in_memory == true && running_a_job == false)
		{
			ImportResultButton->Enable(true);
			SaveResultButton->Enable(true);
		}
		else
		{
			ImportResultButton->Enable(false);
			SaveResultButton->Enable(false);
		}

		if (volumes_are_dirty == true)
		{
			FillVolumePanels();
			volumes_are_dirty = false;

		}
	}

}

void Sharpen3DPanel::FillVolumePanels()
{
	VolumeComboBox->FillComboBox(false, true);
}

void Sharpen3DPanel::OnUseMaskCheckBox( wxCommandEvent& event )
{
	/*if (UseMaskCheckBox->GetValue() == true)
	{
		auto_mask_value = AutoMaskYesRadioButton->GetValue();
		MaskSelectPanel->FillComboBox();
	}
	else
	{
		if (auto_mask_value == true) AutoMaskYesRadioButton->SetValue(true);
		else AutoMaskNoRadioButton->SetValue(true);
	}*/

	MaskSelectPanel->FillComboBox();
}

void Sharpen3DPanel::OnImportResultClick( wxCommandEvent& event )
{

}

void Sharpen3DPanel::OnSaveResultClick( wxCommandEvent& event )
{
	if (have_a_result_in_memory == true && active_result != NULL && running_a_job == false)
	{
		ProperOverwriteCheckSaveDialog *saveFileDialog;
		saveFileDialog = new ProperOverwriteCheckSaveDialog(this, _("Save mrc file"), "MRC files (*.mrc)|*.mrc", ".mrc");

		if (saveFileDialog->ShowModal() == wxID_CANCEL)
		{
			saveFileDialog->Destroy();
			return;
		}

		VolumeAsset *selected_volume = volume_asset_panel->ReturnAssetPointer(VolumeComboBox->GetSelection());

		MRCFile output_file(saveFileDialog->ReturnProperPath().ToStdString(), true);
		active_result->WriteSlices(&output_file, 1, active_result->logical_z_dimension);
		output_file.SetPixelSize(selected_volume->pixel_size);
		output_file.WriteHeader();

		saveFileDialog->Destroy();
	}
}



void Sharpen3DPanel::OnVolumeComboBox( wxCommandEvent& event )
{
	if (VolumeComboBox->GetSelection() >=0)
	{
		// blank active result..

		if (active_result != NULL)
		{
			delete active_result;
			active_result = NULL;
		}
		have_a_result_in_memory = false;

		if (ResultDisplayPanel->IsShown() == true)
		{
			ResultDisplayPanel->Show(false);
			GuinierPlot->Clear();;
			InfoPanel->Show(true);
			Layout();
		}

		// we need to get an info we have for this volume..
		// first get the refinement_id and class number..

		int found_class;
		long found_refinement_id;
		float inner_mask_radius;
		float outer_mask_radius;

		Refinement *refinement_that_created_3d = NULL;
		VolumeAsset *selected_volume = volume_asset_panel->ReturnAssetPointer(VolumeComboBox->GetSelection());


		if (selected_volume->reconstruction_job_id >= 0)
		{

			sqlite3_stmt *current_statement;
			main_frame->current_project.database.Prepare(wxString::Format("SELECT REFINEMENT_ID, CLASS_NUMBER, INNER_MASK_RADIUS, OUTER_MASK_RADIUS FROM RECONSTRUCTION_LIST WHERE RECONSTRUCTION_ID=%li", selected_volume->reconstruction_job_id), &current_statement);
			main_frame->current_project.database.Step(current_statement);

			found_refinement_id = sqlite3_column_int64(current_statement, 0);
			found_class = sqlite3_column_int(current_statement, 1);
			inner_mask_radius = sqlite3_column_double(current_statement, 2);
			outer_mask_radius = sqlite3_column_double(current_statement, 3);

			active_class = found_class - 1;

			main_frame->current_project.database.Finalize(current_statement);

			// what is the refinement id for this reconstruction_job
			long refinement_id = main_frame->current_project.database.ReturnRefinementIDGivenReconstructionID(selected_volume->reconstruction_job_id);
			ShortRefinementInfo *short_info = refinement_package_asset_panel->ReturnPointerToShortRefinementInfoByRefinementID(refinement_id);

			if (short_info != NULL)
			{
				if (short_info->reconstructed_volume_asset_ids[found_class - 1] == selected_volume->asset_id)
				{
					// this all checks out.. get the refinement
					if (active_refinement == NULL) active_refinement = main_frame->current_project.database.GetRefinementByID(refinement_id, false);
					else
					{
						if (active_refinement->refinement_id != refinement_id) // get the new one
						{
							delete active_refinement;
							active_refinement = main_frame->current_project.database.GetRefinementByID(refinement_id, false);
						}
					}
				}
				else // they don't match, so the 3D for this refinement must have been regenerated, and we have no details
				{
					if (active_refinement != NULL)
					{
						delete active_refinement;
						active_refinement = NULL;
					}

					inner_mask_radius = 0.0f;
					outer_mask_radius = selected_volume->pixel_size * float(selected_volume->x_size) * 0.5;
				}

			}
			else
			{
				if (active_refinement != NULL)
				{
					delete active_refinement;
					active_refinement = NULL;
				}

				inner_mask_radius = 0.0f;
				outer_mask_radius = selected_volume->pixel_size * float(selected_volume->x_size) * 0.5;

			}
		}
		else // we don't have the refinement
		{
			if (active_refinement != NULL)
			{
				delete active_refinement;
				active_refinement = NULL;
			}

			inner_mask_radius = 0.0f;
			outer_mask_radius = selected_volume->pixel_size * float(selected_volume->x_size) * 0.5;

		}


		// set the defaults basically..

		FlattenFromTextCtrl->ChangeValueFloat(8.00f);
		FilterEdgeWidthTextCtrl->ChangeValueFloat(20.0f);
		AdditionalLowBFactorTextCtrl->ChangeValueFloat(-90.0f);
		AdditionalHighBFactorTextCtrl->ChangeValueFloat(0.0f);
		SSNRScaleFactorTextCtrl->ChangeValueFloat(1.0f);
		InnerMaskRadiusTextCtrl->ChangeValueFloat(inner_mask_radius);
		OuterMaskRadiusTextCtrl->ChangeValueFloat(outer_mask_radius);
		InvertHandednessNoButton->SetValue(true);
		CorrectGriddingYesButton->SetValue(true);

		if (active_refinement == NULL) // no refinement
		{
			CutOffResTextCtrl->ChangeValueFloat(3.5f);
			UseFSCWeightingNoButton->SetValue(true);

		}
		else
		{
			CutOffResTextCtrl->ChangeValueFloat(active_refinement->class_refinement_results[active_class].class_resolution_statistics.ReturnEstimatedResolution());
			UseFSCWeightingYesButton->SetValue(true);
		}
	}
}

void Sharpen3DPanel::OnRunButtonClick( wxCommandEvent& event )
{

	SharpenMapThread *sharpen_thread;
	VolumeAsset *selected_volume = volume_asset_panel->ReturnAssetPointer(VolumeComboBox->GetSelection());

	wxString map_filename = selected_volume->filename.GetFullPath();
	float pixel_size = selected_volume->pixel_size;
	float resolution_limit = CutOffResTextCtrl->ReturnValue();
	bool invert_hand = InvertHandednessYesButton->GetValue();
	float inner_mask_radius = InnerMaskRadiusTextCtrl->ReturnValue();
	float outer_mask_radius = OuterMaskRadiusTextCtrl->ReturnValue();
	float start_res_for_whitening = FlattenFromTextCtrl->ReturnValue();
	float additional_low_bfactor = AdditionalLowBFactorTextCtrl->ReturnValue();
	float additional_high_bfactor = AdditionalHighBFactorTextCtrl->ReturnValue();
	float filter_edge = FilterEdgeWidthTextCtrl->ReturnValue();
	bool should_correct_sinc = CorrectGriddingYesButton->GetValue();

	wxString input_mask_filename;
	if (UseMaskCheckBox->GetValue() == true)
	{
		VolumeAsset *mask_volume = volume_asset_panel->ReturnAssetPointer(MaskSelectPanel->GetSelection());
		input_mask_filename = mask_volume->filename.GetFullPath();
	}
	else input_mask_filename = "";

	ResolutionStatistics *input_resolution_statistics;
	if (active_refinement == NULL || UseFSCWeightingYesButton->GetValue() == false) input_resolution_statistics = NULL;
	else input_resolution_statistics = &active_refinement->class_refinement_results[active_class].class_resolution_statistics;

	float statistics_scale_factor = SSNRScaleFactorTextCtrl->ReturnValue();

	sharpen_thread = new SharpenMapThread(this, map_filename, pixel_size, resolution_limit, invert_hand, inner_mask_radius, outer_mask_radius, start_res_for_whitening, additional_low_bfactor, additional_high_bfactor, filter_edge, input_mask_filename, input_resolution_statistics, statistics_scale_factor, should_correct_sinc);

	if ( sharpen_thread->Run() != wxTHREAD_NO_ERROR )
	{
		wxMessageBox( "Error, cannot launch sharpener thread!", "Cannot launch thread", wxICON_ERROR);
		delete sharpen_thread;
		return;
	}
	else
	{
		running_a_job = true;
		ProgressGuage->Pulse();
		if (guage_timer != NULL) delete guage_timer;

		guage_timer = new wxTimer(this, 0);
		guage_timer->Start(100);
	}

}

void Sharpen3DPanel::OnSharpenThreadComplete(ReturnSharpeningResultsEvent& my_event)
{
	if (active_result != NULL) delete active_result;

	active_result = my_event.GetSharpenedImage();
	if (active_result != NULL) have_a_result_in_memory = true;

	Image *original_orth_image = my_event.GetOriginalOrthImage();
	Image *sharpened_orth_image = my_event.GetSharpenedOrthImage();

	Curve *original_curve = my_event.GetOriginalCurve();
	Curve *sharpened_curve = my_event.GetSharpenedCurve();

	GuinierPlot->Clear();

	if (original_curve != NULL && sharpened_curve != NULL)
	{

		float min_y = FLT_MAX;
		float max_y = -FLT_MAX;
		float scale_factor = original_curve->data_y[1] / sharpened_curve->data_y[1];

		// scale the curves

		sharpened_curve->MultiplyByConstant(scale_factor);

		for (int point_counter = 0; point_counter < original_curve->number_of_points; point_counter++)
		{
			if (original_curve->data_x[point_counter] <= 0.5)
			{
				min_y = std::min(min_y, original_curve->data_y[point_counter]);
				min_y = std::min(min_y, sharpened_curve->data_y[point_counter]);

				max_y = std::max(max_y, original_curve->data_y[point_counter]);
				max_y = std::max(max_y, sharpened_curve->data_y[point_counter]);
			}
		}

		GuinierPlot->AddCurve(*original_curve, wxColour(0, 0, 255), "Original");
		GuinierPlot->AddCurve(*sharpened_curve, wxColour(255, 0, 0), "Sharpened");
		GuinierPlot->Draw(0.0f, 0.5f, max_y-10, max_y);
	}

	if (original_curve != NULL) delete original_curve;
	if (sharpened_curve != NULL) delete sharpened_curve;

	if (original_orth_image == NULL)
	{
		if (sharpened_orth_image != NULL) delete sharpened_orth_image;
	}
	else
	if (sharpened_orth_image == NULL) delete original_orth_image;
	else // display them
	{
		Freeze();
		if (ResultDisplayPanel->my_notebook->GetPageCount() != 2)
		{
			ResultDisplayPanel->Clear();
			ResultDisplayPanel->OpenImage(original_orth_image, "Original 3D", true);
			ResultDisplayPanel->OpenImage(sharpened_orth_image, "Sharpened 3D", true);
		}
		else // already have 2 pages..
		{
			ResultDisplayPanel->my_notebook->ChangeSelection(0);
			ResultDisplayPanel->ChangeImage(original_orth_image, "Original 3D", true);
			ResultDisplayPanel->my_notebook->ChangeSelection(1);
			ResultDisplayPanel->ChangeImage(sharpened_orth_image, "Sharpened 3D", true);

		}

		if (ResultDisplayPanel->IsShown() == false)
		{
			ResultDisplayPanel->Show(true);
			InfoPanel->Show(false);
			Layout();
		}

		Thaw();

	}

	if (guage_timer != NULL)
	{
		guage_timer->Stop();
		delete guage_timer;
		guage_timer = NULL;
	}
	ProgressGuage->SetValue(0);
	running_a_job = false;

}

wxThread::ExitCode SharpenMapThread::Entry()
{
	ReturnSharpeningResultsEvent *finished_event = new ReturnSharpeningResultsEvent(RETURN_SHARPENING_RESULTS_EVT); // for sending back the panel

	if (DoesFileExist(map_filename) == true)
	{
		ImageFile *map_file = new ImageFile(map_filename.ToStdString());
		ImageFile *mask_file = NULL;

		if (DoesFileExist(input_mask_filename) == true)
		{
			mask_file = new ImageFile(input_mask_filename.ToStdString());

		}

		Image *volume_to_sharpen = new Image;
		Image *original_orth_image = new Image;
		Image *sharpened_orth_image = new Image;

		volume_to_sharpen->ReadSlices(map_file, 1, map_file->ReturnNumberOfSlices());

		// if we are correcting sinc, do it

		if (correct_sinc == true) volume_to_sharpen->CorrectSinc(outer_mask_radius);

		// calculate an orthogonal view of this..

		original_orth_image->Allocate(map_file->ReturnXSize() * 3, map_file->ReturnYSize(), 1, true);
		volume_to_sharpen->CreateOrthogonalProjectionsImage(original_orth_image, false);

		if (original_orth_image->logical_x_dimension > 1500)
		{
			float scale_factor = original_orth_image->logical_x_dimension / 1500;
			original_orth_image->ForwardFFT();
			original_orth_image->Resize(myroundint(original_orth_image->logical_x_dimension * scale_factor), myroundint(original_orth_image->logical_y_dimension * scale_factor), 1);
			original_orth_image->BackwardFFT();
		}

		finished_event->SetOriginalOrthImage(original_orth_image);

		Image *mask_image = NULL;

		if (mask_file != NULL)
		{
			mask_image = new Image;
			mask_image->ReadSlices(mask_file, 1, mask_file->ReturnNumberOfSlices());
		}
		delete mask_file;

		Curve *original_curve = new Curve;
		Curve *sharpened_curve = new Curve;

		volume_to_sharpen->SharpenMap(pixel_size, resolution_limit, invert_hand, inner_mask_radius, outer_mask_radius, start_res_for_whitening, additional_low_bfactor, additional_high_bfactor, filter_edge, mask_image, input_resolution_statistics, statistics_scale_factor, original_curve, sharpened_curve);

		finished_event->SetSharpenedImage(volume_to_sharpen);
		finished_event->SetOriginalCurve(original_curve);
		finished_event->SetSharpenedCurve(sharpened_curve);

		// calculate an orthogonal view of this..

		sharpened_orth_image->Allocate(map_file->ReturnXSize() * 3, map_file->ReturnYSize(), 1, true);
		volume_to_sharpen->CreateOrthogonalProjectionsImage(sharpened_orth_image, false);

		if (sharpened_orth_image->logical_x_dimension > 1500)
		{
			float scale_factor = sharpened_orth_image->logical_x_dimension / 1500;
			sharpened_orth_image->ForwardFFT();
			sharpened_orth_image->Resize(myroundint(sharpened_orth_image->logical_x_dimension * scale_factor), myroundint(sharpened_orth_image->logical_y_dimension * scale_factor), 1);
			sharpened_orth_image->BackwardFFT();
		}

		finished_event->SetSharpenedOrthImage(sharpened_orth_image);

		delete map_file;
		if (mask_image != NULL) delete mask_image;
	}
	else
	{
		finished_event->SetSharpenedImage(NULL);
		finished_event->SetOriginalOrthImage(NULL);
		finished_event->SetSharpenedOrthImage(NULL);
	}

	wxQueueEvent(main_thread_pointer, finished_event);
}






