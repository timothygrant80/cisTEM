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
//	#include "icons/niko_picture1.cpp"
//	wxBitmap niko_picture1_bmp = wxBITMAP_PNG_FROM_DATA(niko_picture1);

	//#include "icons/niko_picture2.cpp"
	//wxBitmap niko_picture2_bmp = wxBITMAP_PNG_FROM_DATA(niko_picture2);
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
	InfoText->WriteText(wxT("This panel is for sharpening your maps to make them look better! The info below is for Generate 3D, this panel text needs to be written."));
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
	InfoText->WriteText(wxT("Input Refinement Package : "));
	InfoText->EndBold();
	InfoText->WriteText(wxT("The name of the refinement package previously set up in the Assets panel (providing details of particle locations, box size and imaging parameters)."));
	InfoText->Newline();
	InfoText->BeginBold();
	InfoText->WriteText(wxT("Input Parameters : "));
	InfoText->EndBold();
	InfoText->WriteText(wxT("The source of the starting parameters for this reconstruction run."));
	InfoText->Newline();

	InfoText->BeginAlignment(wxTEXT_ALIGNMENT_CENTRE);
	InfoText->BeginBold();
	InfoText->BeginUnderline();
	InfoText->WriteText(wxT("Expert Options"));
	InfoText->EndBold();
	InfoText->EndUnderline();
	InfoText->Newline();
	InfoText->Newline();
	InfoText->EndAlignment();

	InfoText->BeginBold();
	InfoText->WriteText(wxT("Inner/Outer Mask Radius (Å) : "));
	InfoText->EndBold();
	InfoText->WriteText(wxT("Radii describing a spherical mask with an inner and outer radius that will be applied to the final reconstruction and to the half reconstructions to calculate Fourier Shell Correlation curve. The inner radius is normally set to 0.0 but can assume non-zero values to remove density inside a particle if it represents largely disordered features, such as the genomic RNA or DNA of a virus."));
	InfoText->Newline();

	InfoText->BeginBold();
	InfoText->WriteText(wxT("Score to B-factor Constant (Å2) : "));
	InfoText->EndBold();
	InfoText->WriteText(wxT("The particles inserted into a reconstruction will be weighted according to their scores. The weighting function is akin to a B-factor, attenuating high-resolution signal of particles with lower scores more strongly than of particles with higher scores. The B-factor applied to each particle prior to insertion into the reconstruction is calculated as B = (score – average score) * constant * 0.25. Users are encouraged to calculate reconstructions with different values to find a value that produces the highest resolution. Values between 0 and 10 are reasonable (0 will disable weighting)."));
	InfoText->Newline();

	InfoText->BeginBold();
	InfoText->WriteText(wxT("Adjust Score for Defocus? : "));
	InfoText->EndBold();
	InfoText->WriteText(wxT("Scores sometimes depend on the amount of image defocus. A larger defocus amplifies low-resolution features in the image and this may lead to higher particle scores compared to particles from an image with a small defocus. Adjusting the scores for this difference makes sure that particles with smaller defocus are not systematically downweighted by the above B-factor weighting."));
	InfoText->Newline();

	InfoText->BeginBold();
	InfoText->WriteText(wxT("Score Threshold : "));
	InfoText->EndBold();
	InfoText->WriteText(wxT("Particles with a score lower than the threshold will be excluded from the reconstruction. This provides a way to exclude particles that may score low because of misalignment or damage."));
	InfoText->Newline();

	InfoText->BeginBold();
	InfoText->WriteText(wxT("Resolution Limit (Å) : "));
	InfoText->EndBold();
	InfoText->WriteText(wxT("The reconstruction calculation can be accelerated by limiting its resolution. It is important to make sure that the resolution limit entered here is higher than the resolution used for refinement in the following cycle."));
	InfoText->Newline();

	InfoText->BeginBold();
	InfoText->WriteText(wxT("Autocrop Images? : "));
	InfoText->EndBold();
	InfoText->WriteText(wxT("The reconstruction calculation can also be accelerated by cropping the boxes containing the particles. Cropping will slightly reduce the overall quality of the reconstruction due to increased aliasing effects and should not be used when finalizing refinement. However, during refinement, cropping can greatly increase the speed of reconstruction without noticeable impact on the refinement results."));
	InfoText->Newline();

	InfoText->BeginBold();
	InfoText->WriteText(wxT("Also Save Half-Maps? : "));
	InfoText->EndBold();
	InfoText->WriteText(wxT("If yes, the reconstruction half maps will also be saved as '_map1' and '_map2' in the volume assets folder."));
	InfoText->Newline();

	InfoText->BeginBold();
	InfoText->WriteText(wxT("Overwrite Statistics? : "));
	InfoText->EndBold();
	InfoText->WriteText(wxT("If yes, the resolution statistics (essentially the FSC) for the input refinement will be overwritten."));
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
	Freeze();
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
			MaskSelectPanel->AssetComboBox->ChangeValue("");
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
	Thaw();
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
	if (active_refinement == NULL) input_resolution_statistics = NULL;
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






