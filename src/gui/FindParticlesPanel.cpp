//#include "../core/core_headers.h"
#include "../core/gui_core_headers.h"

extern MyMovieAssetPanel *movie_asset_panel;
extern MyImageAssetPanel *image_asset_panel;
extern MyRunProfilesPanel *run_profiles_panel;
extern MyMainFrame *main_frame;
extern MyFindCTFResultsPanel *ctf_results_panel;
extern MyParticlePositionAssetPanel *particle_position_asset_panel;
extern MyPickingResultsPanel *picking_results_panel;
extern MyVolumeAssetPanel *volume_asset_panel;


MyFindParticlesPanel::MyFindParticlesPanel( wxWindow* parent )
:
FindParticlesPanel( parent )
{
	// Set variables

	buffered_results = NULL;

	// Fill combo box..

	//FillGroupComboBox();

	my_job_id = -1;
	running_job = false;

	group_combo_is_dirty = false;
	run_profiles_are_dirty = false;

	SetInfo();
	FillGroupComboBox();
	FillRunProfileComboBox();
	FillPickingAlgorithmComboBox();

	wxSize input_size = ExpertInputSizer->GetMinSize();
	input_size.x += wxSystemSettings::GetMetric(wxSYS_VSCROLL_X);
	input_size.y = -1;
	ExpertOptionsPanel->SetMinSize(input_size);
	ExpertOptionsPanel->SetSize(input_size);

	ExpertToggleButton->Enable(true);

	//result_bitmap.Create(1,1, 24);
	time_of_last_result_update = time(NULL);
	FindParticlesSplitterWindow->Unsplit(LeftPanel);

	ResetDefaults();

	GroupComboBox->AssetComboBox->Bind(wxEVT_COMMAND_COMBOBOX_SELECTED, &MyFindParticlesPanel::OnGroupComboBox, this);
	ImageComboBox->AssetComboBox->Bind(wxEVT_COMMAND_COMBOBOX_SELECTED, &MyFindParticlesPanel::OnImageComboBox, this);
	ReferenceSelectPanel->AssetComboBox->Bind(wxEVT_COMMAND_COMBOBOX_SELECTED, &MyFindParticlesPanel::OnReferenceVolumeComboBox, this);

	ExclusionRadiusNumericCtrl->SetPrecision(0);
	TemplateRadiusNumericCtrl->SetPrecision(0);

	// Symmetries available
	SymmetryComboBox->Append("C1");
  	SymmetryComboBox->Append("C2");
  	SymmetryComboBox->Append("C3");
  	SymmetryComboBox->Append("C4");
  	SymmetryComboBox->Append("D2");
  	SymmetryComboBox->Append("D3");
  	SymmetryComboBox->Append("D4");
  	SymmetryComboBox->Append("I");
  	SymmetryComboBox->Append("I2");
  	SymmetryComboBox->Append("O");
  	SymmetryComboBox->Append("T");
  	SymmetryComboBox->Append("T2");
  	SymmetryComboBox->ChangeValue("C1");
}


void MyFindParticlesPanel::OnInfoURL(wxTextUrlEvent& event)
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

void MyFindParticlesPanel::Reset()
{
	ProgressBar->SetValue(0);
	TimeRemainingText->SetLabel("Time Remaining : ???h:??m:??s");
	CancelAlignmentButton->Show(true);
	FinishButton->Show(false);

	ProgressPanel->Show(false);
	StartPanel->Show(true);
	OutputTextPanel->Show(false);
	output_textctrl->Clear();
	PickingResultsPanel->Show(false);
	PickingResultsPanel->Clear();
	//graph_is_hidden = true;
	InfoPanel->Show(true);
	ReferenceSelectPanel->Clear();

	ExpertToggleButton->Enable(false);

	if (running_job == true)
	{
		main_frame->job_controller.KillJob(my_job_id);

		if (buffered_results != NULL)
		{
			delete [] buffered_results;
			buffered_results = NULL;
		}

		running_job = false;
	}

	Layout();
	LeftPanel->Layout();
	RightPanel->Layout();

}

void MyFindParticlesPanel::ResetDefaults()
{

	ExclusionRadiusNumericCtrl->ChangeValueFloat(120.0f);
	TemplateRadiusNumericCtrl->ChangeValueFloat(80.0f);
	ThresholdPeakHeightNumericCtrl->ChangeValueFloat(6.0f);
	AutoPickRefreshCheckBox->SetValue(false);
	AvoidLowVarianceAreasCheckBox->SetValue(false);
	AvoidHighVarianceAreasCheckBox->SetValue(true);
	LowVarianceThresholdNumericCtrl->ChangeValueFloat(-0.5f);
	HighVarianceThresholdNumericCtrl->ChangeValueFloat(2.0f);
	HighestResolutionNumericCtrl->ChangeValueFloat(20.0f);
	SetMinimumDistanceFromEdgesCheckBox->SetValue(false);
	MinimumDistanceFromEdgesSpinCtrl->SetValue(128);
	AvoidAbnormalLocalMeanAreasCheckBox->SetValue(true);
	NumberOfBackgroundBoxesSpinCtrl->SetValue(50);
	AlgorithmToFindBackgroundChoice->SetSelection(0);
	ProjectionIntervalNumericCtrl->SetValue(wxString::Format("%.1f",2.0 * RotationalAccuracyGivenResolutionRadiusPhaseError(HighestResolutionNumericCtrl->ReturnValue(),TemplateRadiusNumericCtrl->ReturnValue(),120.0)));
	PickingAlgorithmComboBox->SetSelection(ab_initio);
	NumberOfTemplateRotationsSpinCtrl->SetValue(1);
}

void MyFindParticlesPanel::SetInfo()
{

	InfoText->GetCaret()->Hide();

	InfoText->BeginSuppressUndo();
	InfoText->BeginAlignment(wxTEXT_ALIGNMENT_CENTRE);
	InfoText->BeginBold();
	InfoText->BeginUnderline();
	InfoText->BeginFontSize(14);
	InfoText->WriteText(wxT("Particle picking"));
	InfoText->EndFontSize();
	InfoText->EndBold();
	InfoText->EndUnderline();
	InfoText->Newline();
	InfoText->Newline();
	InfoText->EndAlignment();

	InfoText->BeginAlignment(wxTEXT_ALIGNMENT_LEFT);
	InfoText->WriteText(wxT("Individual particles need to be located in each micrograph so that they may be used to compute a 3D reconstruction later. Ideally one would find all the particles and not make any erroneous selections."));
	InfoText->Newline();
	InfoText->Newline();
	InfoText->WriteText(wxT("One can either select (click on) each particle manually, or use the 'ab-initio' mode. In this mode, a circular blob template is generated internally, and then matched against each micrograph, using a matched filter based on the approach described in Sigworth (2004).  This blob approach is very fast, and enables real-time optimization of the picking parameters. Users are encouraged to spend some time finding the parameters that best pick their particles on a number of different images."));
	InfoText->Newline();
	InfoText->Newline();
	InfoText->WriteText(wxT(""));
	InfoText->EndAlignment();

	InfoText->BeginAlignment(wxTEXT_ALIGNMENT_CENTRE);
	InfoText->BeginBold();
	InfoText->BeginUnderline();
	InfoText->WriteText(wxT("Program Options"));
	InfoText->EndBold();
	InfoText->EndUnderline();
	InfoText->Newline();
	InfoText->EndAlignment();

	InfoText->BeginAlignment(wxTEXT_ALIGNMENT_LEFT);
	InfoText->BeginBold();
	InfoText->WriteText(wxT("Input Group : "));
	InfoText->EndBold();
	InfoText->WriteText(wxT("The group of image assets in which to find particles."));
	InfoText->Newline();
	InfoText->BeginBold();
	InfoText->WriteText(wxT("Run Profile : "));
	InfoText->EndBold();
	InfoText->WriteText(wxT("The selected run profile will be used to run the job.  The run profile describes how the job should be run (e.g. how many processors should be used, and on which different computers).  Run profiles are set in the Run Profile panel, located under settings."));
	InfoText->Newline();
	InfoText->BeginBold();
	InfoText->WriteText(wxT("Exclusion Radius (Å) : "));
	InfoText->EndBold();
	InfoText->WriteText(wxT("When a pick is located, no other pick will be made within that distance (in Angstroms). This is the radius of the circles drawn in the preview."));
	InfoText->Newline();
	InfoText->BeginBold();
	InfoText->WriteText(wxT("Template Radius (Å) : "));
	InfoText->EndBold();
	InfoText->WriteText(wxT("When picking with a blob: the radius of the blob template. When picking with a reference volume: the maximum radius of the volume, in Angstroms. The projections will be masked with a soft circle slightly larger than this radius before being used as templates."));
	InfoText->Newline();
	InfoText->BeginBold();
	InfoText->WriteText(wxT("Threshold Peak Hwight : "));
	InfoText->EndBold();
	InfoText->WriteText(wxT("Pick coordinates will be defined as the coordinates of any peak in the search function which exceeds this threshold. In numbers of standard deviations above expected noise variations in the scoring function. See Sigworth (2004) for definition."));
	InfoText->Newline();
	InfoText->BeginBold();
	InfoText->WriteText(wxT("Avoid High-Variance Areas : "));
	InfoText->EndBold();
	InfoText->WriteText(wxT("Avoid areas with abnormally high local variance. This can be effective in avoiding edges of support films or contamination."));
	InfoText->Newline();
	InfoText->EndAlignment();

	InfoText->BeginAlignment(wxTEXT_ALIGNMENT_CENTRE);
	InfoText->BeginBold();
	InfoText->BeginUnderline();
	InfoText->WriteText(wxT("Expert Program Options"));
	InfoText->EndBold();
	InfoText->EndUnderline();
	InfoText->Newline();
	InfoText->EndAlignment();

	InfoText->BeginAlignment(wxTEXT_ALIGNMENT_LEFT);
	InfoText->BeginBold();
	InfoText->WriteText(wxT("Highest Resolution Used in Picking (Å) : "));
	InfoText->EndBold();
	InfoText->WriteText(wxT("The template and micrograph will be resampled (by Fourier cropping) to a pixel size of half the resolution given here. Note that the information in the 'corners' of the Fourier transforms (beyond the Nyquist frequency) remains intact, so that there is some small risk of bias beyond this resolution."));
	InfoText->Newline();
	InfoText->BeginBold();
	InfoText->WriteText(wxT("Set Minimum Distance From Edges (px) : "));
	InfoText->EndBold();
	InfoText->WriteText(wxT("No particle shall be picked closer than this distance from the edges of the micrograph. In pixels."));
	InfoText->Newline();
	InfoText->BeginBold();
	InfoText->WriteText(wxT("Avoid Areas With Abnormal Local Means : "));
	InfoText->EndBold();
	InfoText->WriteText(wxT("Avoid areas with abnormally low or high local mean. This can be effective to avoid picking from, e.g., contaminating ice crystals, support film."));
	InfoText->Newline();
	InfoText->BeginBold();
	InfoText->WriteText(wxT("Number of Background Boxes : "));
	InfoText->EndBold();
	InfoText->WriteText(wxT("Number of background areas to use in estimating the background spectrum. The larger the number of boxes, the more accurate the estimate should be, provided that none of the background boxes contain any particles to be picked."));
	InfoText->Newline();
	InfoText->BeginBold();
	InfoText->WriteText(wxT("Algorithm to Find Background Areas : "));
	InfoText->EndBold();
	InfoText->WriteText(wxT("Testing so far suggests that areas of lowest variance in experimental micrographs should be used to estimate the background spectrum. However, when using synthetic micrographs this can lead to bias in the spectrum estimation and the alternative (areas with local variances near the mean of the distribution of local variances) seems to perform better"));
	InfoText->Newline();
	InfoText->Newline();
	InfoText->EndAlignment();

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
	InfoText->WriteText(wxT("Sigworth F.J."));
	InfoText->EndBold();
	InfoText->WriteText(wxT(" 2004. Classical detection theory and the cryo-EM particle selection problem. J. Struct. Biol. 192, 216–221. "));
	InfoText->BeginURL("http://dx.doi.org/10.1016/j.jsb.2003.10.025");
	InfoText->BeginUnderline();
	InfoText->BeginTextColour(*wxBLUE);
	InfoText->WriteText(wxT("doi:10.1016/j.jsb.2003.10.025"));
	InfoText->EndURL();
	InfoText->EndTextColour();
	InfoText->EndUnderline();
	InfoText->Newline();
	InfoText->Newline();

	InfoText->EndSuppressUndo();


}

void MyFindParticlesPanel::FillGroupComboBox()
{
	GroupComboBox->FillComboBox(true);
}

void MyFindParticlesPanel::FillImageComboBox()
{
	ImageAsset *current_image_asset;
	ImageComboBox->FillComboBox(GroupComboBox->GetSelection());

	if (AutoPickRefreshCheckBox->GetValue())
	{
		wxBusyCursor wait;

		PickingParametersPanel->Freeze();
		ExpertOptionsPanel->Freeze();

		SetAllUserParametersForParticleFinder();

		particle_finder.DoItAll(false);

		DrawResultsFromParticleFinder();

		PickingParametersPanel->Thaw();
		ExpertOptionsPanel->Thaw();
	}
}

wxString MyFindParticlesPanel::ReturnNameOfPickingAlgorithm( const int wanted_algorithm )
{

	wxString string_to_return;

	switch(wanted_algorithm)
	{
	case(ab_initio):
			string_to_return = "blob";
			break;
	case(reference_volume):
			string_to_return = "reference volume";
			break;
	default:
			string_to_return = "unknown";
	}

	return string_to_return;
}

int MyFindParticlesPanel::ReturnNumberOfJobsCurrentlyRunning()
{
	return my_job_tracker.total_running_processes;
}

void MyFindParticlesPanel::FillPickingAlgorithmComboBox()
{

	PickingAlgorithmComboBox->Freeze();
	PickingAlgorithmComboBox->Clear();

	for ( int counter = 0; counter < number_of_picking_algorithms; counter ++ )
	{
		PickingAlgorithmComboBox->Append(ReturnNameOfPickingAlgorithm(counter).Capitalize());
	}

	PickingAlgorithmComboBox->SetSelection(0);
	//PickingAlgorithmComboBox->ChangeValue("Please Select...");

	PickingAlgorithmComboBox->Thaw();
}

// When the user selects a new picking algorithm
void MyFindParticlesPanel::OnPickingAlgorithmComboBox( wxCommandEvent& event )
{
	switch(PickingAlgorithmComboBox->GetCurrentSelection())
	{
	case(-1) :
			// No algorithm selected
			PickingParametersPanel->Hide();
			ExpertToggleButton->Enable(false);
			break;
	case(ab_initio):
			// Ab initio
			ShowPickingParametersPanel();
			ExpertToggleButton->Enable(true);
			OnNewPickingAlgorithm();
			break;
	case(reference_volume):
			// Reference volume
			ShowPickingParametersPanel();
			ExpertToggleButton->Enable(true);
			OnNewPickingAlgorithm();
			break;
	default:
			// This algorithm not implemented yet
			PickingParametersPanel->Hide();
			ExpertToggleButton->Enable(false);
	}

}

void MyFindParticlesPanel::ShowPickingParametersPanel()
{
	PickingParametersPanel->Show(true);

	wxSize ExpertMinSize = ExpertOptionsPanel->GetMinSize();
	wxSize ParamsMinSize = PickingParametersPanel->GetMinSize();
	int splitter_panel_size = std::max(400,ExpertMinSize.x);
	splitter_panel_size = std::max(splitter_panel_size,ParamsMinSize.x) + 10;

	if (FindParticlesSplitterWindow->GetSashPosition() < 0.2 * splitter_panel_size || FindParticlesSplitterWindow->GetSplitMode() != wxSPLIT_VERTICAL || ! FindParticlesSplitterWindow->IsSplit())
	{
		FindParticlesSplitterWindow->SplitVertically(LeftPanel, RightPanel, splitter_panel_size);
		FindParticlesSplitterWindow->SetSashPosition(splitter_panel_size);

		Layout();
		LeftPanel->Layout();
		RightPanel->Layout();
	}

}

void MyFindParticlesPanel::OnAutoPickRefreshCheckBox( wxCommandEvent& event )
{
	if (AutoPickRefreshCheckBox->GetValue())
	{
		// Auto pick refresh is enabled
		TestOnCurrentMicrographButton->Enable(false);

		// Show results panel
		InfoPanel->Show(false);
		PickingResultsPanel->Show(true);
		Layout();
		LeftPanel->Layout();
		RightPanel->Layout();

		// Do the pick
		{
			wxBusyCursor wait;

			PickingParametersPanel->Freeze();
			ExpertOptionsPanel->Freeze();

			if (PickingAlgorithmComboBox->GetSelection() == reference_volume) PrepareTemplatesFromReferenceVolume();
			SetAllUserParametersForParticleFinder();

			particle_finder.DoItAll(false);

			DrawResultsFromParticleFinder();

			PickingParametersPanel->Thaw();
			ExpertOptionsPanel->Thaw();
		}


	}
	else
	{
		// Auto pick refresh is disabled
		TestOnCurrentMicrographButton->Enable(true);
	}
}

int MyFindParticlesPanel::ReturnDefaultMinimumDistanceFromEdges()
{
	float pixel_size = image_asset_panel->ReturnAssetPointer(image_asset_panel->ReturnGroupMember(GroupComboBox->GetSelection(), 0))->pixel_size;
	return int(2.5 * ExclusionRadiusNumericCtrl->ReturnValue() / pixel_size)+1;
}


void MyFindParticlesPanel::OnNewTemplateRadius()
{
	// Enforce min
	if (TemplateRadiusNumericCtrl->ReturnValue() < 1.0) TemplateRadiusNumericCtrl->ChangeValueFloat(1.0);

	// Reset the rotational parameters
	if ( PickingAlgorithmComboBox->GetSelection() == reference_volume )
	{
		ProjectionIntervalNumericCtrl->SetValue(wxString::Format("%.1f",2.0 * RotationalAccuracyGivenResolutionRadiusPhaseError(HighestResolutionNumericCtrl->ReturnValue(),TemplateRadiusNumericCtrl->ReturnValue(),120.0)));
		NumberOfTemplateRotationsSpinCtrl->SetValue(myroundint(360.0f/ProjectionIntervalNumericCtrl->ReturnValue()));
	}

	if (AutoPickRefreshCheckBox->GetValue())
	{
		wxBusyCursor wait;

		PickingParametersPanel->Freeze();
		ExpertOptionsPanel->Freeze();

		if (PickingAlgorithmComboBox->GetSelection() == reference_volume) PrepareTemplatesFromReferenceVolume(); // need to do this because the masking of the projections needs to be redone with the new radius
		SetAllUserParametersForParticleFinder();

		particle_finder.RedoWithNewTemplateRadius();

		DrawResultsFromParticleFinder();

		PickingParametersPanel->Thaw();
		ExpertOptionsPanel->Thaw();
	}
}

void MyFindParticlesPanel::OnNewPickingAlgorithm()
{

	wxPrintf("OnNewPickingAlgorithm\n");
	switch(PickingAlgorithmComboBox->GetCurrentSelection())
	{
	case(ab_initio) :
		wxPrintf("ab initio should set to 1\n");
		NumberOfTemplateRotationsSpinCtrl->SetValue(1);
	case(reference_volume) :
		NumberOfTemplateRotationsSpinCtrl->SetValue(myroundint(360.0f/ProjectionIntervalNumericCtrl->ReturnValue()));
	break;
	default :
		MyDebugAssertTrue(false,"Oops, unknown picking algorithm: %i\n",PickingAlgorithmComboBox->GetCurrentSelection());
	}

	if (AutoPickRefreshCheckBox->GetValue())
	{
		wxBusyCursor wait;

		PickingParametersPanel->Freeze();
		ExpertOptionsPanel->Freeze();

		if (PickingAlgorithmComboBox->GetCurrentSelection() == reference_volume) PrepareTemplatesFromReferenceVolume();

		SetAllUserParametersForParticleFinder();
		
		particle_finder.DoItAll(false);

		DrawResultsFromParticleFinder();

		PickingParametersPanel->Thaw();
		ExpertOptionsPanel->Thaw();
	}
}


void MyFindParticlesPanel::OnNewThresholdPeakHeight()
{
	// Enforce min
	if (ThresholdPeakHeightNumericCtrl->ReturnValue() < 0.1) ThresholdPeakHeightNumericCtrl->ChangeValueFloat(0.1);

	if (AutoPickRefreshCheckBox->GetValue())
	{
		wxBusyCursor wait;

		PickingParametersPanel->Freeze();
		ExpertOptionsPanel->Freeze();

		SetAllUserParametersForParticleFinder();

		particle_finder.RedoWithNewMinimumPeakHeight();

		DrawResultsFromParticleFinder();

		PickingParametersPanel->Thaw();
		ExpertOptionsPanel->Thaw();
	}
}


void MyFindParticlesPanel::OnNewExclusionRadius()
{
	// Enforce min
	if (ExclusionRadiusNumericCtrl->ReturnValue() < 1.0) ExclusionRadiusNumericCtrl->ChangeValueFloat(1.0);

	if (!SetMinimumDistanceFromEdgesCheckBox->IsChecked())
	{
		MinimumDistanceFromEdgesSpinCtrl->SetValue(ReturnDefaultMinimumDistanceFromEdges());
	}

	if (AutoPickRefreshCheckBox->GetValue())
	{
		wxBusyCursor wait;

		PickingParametersPanel->Freeze();
		ExpertOptionsPanel->Freeze();

		SetAllUserParametersForParticleFinder();

		particle_finder.RedoWithNewExclusionRadius();

		DrawResultsFromParticleFinder();

		PickingParametersPanel->Thaw();
		ExpertOptionsPanel->Thaw();
	}
}


void MyFindParticlesPanel::OnNewHighestResolution()
{
	// Enforce min
	float pixel_size = image_asset_panel->ReturnAssetPointer(image_asset_panel->ReturnGroupMember(GroupComboBox->GetSelection(), ImageComboBox->GetSelection()))->pixel_size;
	if (HighestResolutionNumericCtrl->ReturnValue() <  pixel_size * 2.0 ) HighestResolutionNumericCtrl->ChangeValueFloat( pixel_size * 2.0 );

	// Reset the rotational parameters
	if ( PickingAlgorithmComboBox->GetCurrentSelection() == reference_volume )
	{
		ProjectionIntervalNumericCtrl->SetValue(wxString::Format("%.1f",2.0 * RotationalAccuracyGivenResolutionRadiusPhaseError(HighestResolutionNumericCtrl->ReturnValue(),TemplateRadiusNumericCtrl->ReturnValue(),100.0)));
		NumberOfTemplateRotationsSpinCtrl->SetValue(myroundint(360.0f/ProjectionIntervalNumericCtrl->ReturnValue()));
	}

	if (AutoPickRefreshCheckBox->GetValue())
	{
		wxBusyCursor wait;

		PickingParametersPanel->Freeze();
		ExpertOptionsPanel->Freeze();

		SetAllUserParametersForParticleFinder();

		particle_finder.RedoWithNewHighestResolution();

		DrawResultsFromParticleFinder();

		PickingParametersPanel->Thaw();
		ExpertOptionsPanel->Thaw();
	}
}


void MyFindParticlesPanel::OnExclusionRadiusNumericTextEnter( wxCommandEvent& event )
{
	OnNewExclusionRadius();
}

void MyFindParticlesPanel::OnExclusionRadiusNumericTextKillFocus( wxFocusEvent & event )
{
	OnNewExclusionRadius();
}

void MyFindParticlesPanel::OnTemplateRadiusNumericTextKillFocus( wxFocusEvent & event )
{
	OnNewTemplateRadius();
}

void MyFindParticlesPanel::OnThresholdPeakHeightNumericTextKillFocus( wxFocusEvent & event )
{
	OnNewThresholdPeakHeight();
}


void MyFindParticlesPanel::OnExclusionRadiusNumericTextSetFocus( wxFocusEvent & event )
{
	value_on_focus_float = ExclusionRadiusNumericCtrl->ReturnValue();
}

void MyFindParticlesPanel::OnTemplateRadiusNumericTextSetFocus( wxFocusEvent & event )
{
	value_on_focus_float = TemplateRadiusNumericCtrl->ReturnValue();
}

void MyFindParticlesPanel::OnThresholdPeakHeightNumericTextSetFocus( wxFocusEvent & event )
{
	value_on_focus_float = ThresholdPeakHeightNumericCtrl->ReturnValue();
}

void MyFindParticlesPanel::OnTemplateRadiusNumericTextEnter( wxCommandEvent& event )
{
	OnNewTemplateRadius();
}

void MyFindParticlesPanel::OnThresholdPeakHeightNumericTextEnter( wxCommandEvent& event )
{
	OnNewThresholdPeakHeight();
}

void MyFindParticlesPanel::OnHighestResolutionNumericTextEnter( wxCommandEvent & event )
{
	OnNewHighestResolution();
}

void MyFindParticlesPanel::OnHighestResolutionNumericKillFocus( wxFocusEvent & event )
{
	OnNewHighestResolution();
}

void MyFindParticlesPanel::OnHighestResolutionNumericSetFocus( wxFocusEvent & event )
{
	value_on_focus_float = HighestResolutionNumericCtrl->ReturnValue();
}

void MyFindParticlesPanel::OnMinimumDistanceFromEdgesSpinCtrl( wxSpinEvent& event )
{
	if (AutoPickRefreshCheckBox->GetValue())
	{
		wxBusyCursor wait;

		PickingParametersPanel->Freeze();
		ExpertOptionsPanel->Freeze();

		SetAllUserParametersForParticleFinder();

		particle_finder.RedoWithNewMinimumDistanceFromEdges();

		DrawResultsFromParticleFinder();

		PickingParametersPanel->Thaw();
		ExpertOptionsPanel->Thaw();
	}
}

void MyFindParticlesPanel::OnAvoidLowVarianceAreasCheckBox( wxCommandEvent& event )
{
	LowVarianceThresholdNumericCtrl->Enable(AvoidLowVarianceAreasCheckBox->GetValue());
	if (AutoPickRefreshCheckBox->GetValue())
	{
		wxBusyCursor wait;

		PickingParametersPanel->Freeze();
		ExpertOptionsPanel->Freeze();

		SetAllUserParametersForParticleFinder();

		particle_finder.RedoWithNewAvoidLowVarianceAreas();

		DrawResultsFromParticleFinder();

		PickingParametersPanel->Thaw();
		ExpertOptionsPanel->Thaw();
	}
}

void MyFindParticlesPanel::OnAvoidHighVarianceAreasCheckBox( wxCommandEvent& event )
{

	HighVarianceThresholdNumericCtrl->Enable(AvoidHighVarianceAreasCheckBox->GetValue());
	if (AutoPickRefreshCheckBox->GetValue())
	{
		wxBusyCursor wait;

		PickingParametersPanel->Freeze();
		ExpertOptionsPanel->Freeze();

		SetAllUserParametersForParticleFinder();

		particle_finder.RedoWithNewAvoidHighVarianceAreas();

		DrawResultsFromParticleFinder();

		PickingParametersPanel->Thaw();
		ExpertOptionsPanel->Thaw();
	}
}

void MyFindParticlesPanel::OnNewLowVarianceThreshold()
{
	if (AutoPickRefreshCheckBox->GetValue())
	{
		wxBusyCursor wait;

		PickingParametersPanel->Freeze();
		ExpertOptionsPanel->Freeze();

		SetAllUserParametersForParticleFinder();

		particle_finder.RedoWithNewLowVarianceThreshold();

		DrawResultsFromParticleFinder();

		PickingParametersPanel->Thaw();
		ExpertOptionsPanel->Thaw();
	}
}

void MyFindParticlesPanel::OnNewHighVarianceThreshold()
{
	if (AutoPickRefreshCheckBox->GetValue())
	{
		wxBusyCursor wait;

		PickingParametersPanel->Freeze();
		ExpertOptionsPanel->Freeze();

		SetAllUserParametersForParticleFinder();

		particle_finder.RedoWithNewHighVarianceThreshold();

		DrawResultsFromParticleFinder();

		PickingParametersPanel->Thaw();
		ExpertOptionsPanel->Thaw();
	}
}

void MyFindParticlesPanel::OnLowVarianceThresholdNumericTextEnter( wxCommandEvent & event )
{
	OnNewLowVarianceThreshold();
}

void MyFindParticlesPanel::OnHighVarianceThresholdNumericTextEnter( wxCommandEvent & event )
{
	OnNewHighVarianceThreshold();
}

void MyFindParticlesPanel::OnLowVarianceThresholdNumericKillFocus( wxFocusEvent & event )
{
	OnNewLowVarianceThreshold();
}

void MyFindParticlesPanel::OnLowVarianceThresholdNumericSetFocus( wxFocusEvent & event )
{
	value_on_focus_float = LowVarianceThresholdNumericCtrl->ReturnValue();
}

void MyFindParticlesPanel::OnHighVarianceThresholdNumericKillFocus( wxFocusEvent & event )
{
	OnNewHighVarianceThreshold();
}

void MyFindParticlesPanel::OnHighVarianceThresholdNumericSetFocus( wxFocusEvent & event )
{
	value_on_focus_float = HighVarianceThresholdNumericCtrl->ReturnValue();
}


void MyFindParticlesPanel::OnAvoidAbnormalLocalMeanAreasCheckBox( wxCommandEvent& event )
{
	if (AutoPickRefreshCheckBox->GetValue())
	{
		wxBusyCursor wait;

		PickingParametersPanel->Freeze();
		ExpertOptionsPanel->Freeze();

		SetAllUserParametersForParticleFinder();

		particle_finder.RedoWithNewAvoidAbnormalLocalMeanAreas();

		DrawResultsFromParticleFinder();

		PickingParametersPanel->Thaw();
		ExpertOptionsPanel->Thaw();
	}
}

void MyFindParticlesPanel::OnNumberOfBackgroundBoxesSpinCtrl( wxSpinEvent& event )
{
	if (AutoPickRefreshCheckBox->GetValue())
	{
		wxBusyCursor wait;

		PickingParametersPanel->Freeze();
		ExpertOptionsPanel->Freeze();

		SetAllUserParametersForParticleFinder();

		particle_finder.RedoWithNewNumberOfBackgroundBoxes();

		DrawResultsFromParticleFinder();

		PickingParametersPanel->Thaw();
		ExpertOptionsPanel->Thaw();
	}
}

void MyFindParticlesPanel::OnAlgorithmToFindBackgroundChoice( wxCommandEvent& event )
{
	if (AutoPickRefreshCheckBox->GetValue())
	{
		wxBusyCursor wait;

		PickingParametersPanel->Freeze();
		ExpertOptionsPanel->Freeze();

		SetAllUserParametersForParticleFinder();

		particle_finder.RedoWithNewAlgorithmToFindBackground();

		DrawResultsFromParticleFinder();

		PickingParametersPanel->Thaw();
		ExpertOptionsPanel->Thaw();
	}
}

void MyFindParticlesPanel::OnSetMinimumDistanceFromEdgesCheckBox( wxCommandEvent & event )
{
	if ( SetMinimumDistanceFromEdgesCheckBox->GetValue())
	{
		MinimumDistanceFromEdgesSpinCtrl->Enable(true);
	}
	else
	{
		MinimumDistanceFromEdgesSpinCtrl->Enable(false);
		MinimumDistanceFromEdgesSpinCtrl->SetValue(ReturnDefaultMinimumDistanceFromEdges());
		if (AutoPickRefreshCheckBox->GetValue())
		{
			wxBusyCursor wait;

			PickingParametersPanel->Freeze();
			ExpertOptionsPanel->Freeze();

			SetAllUserParametersForParticleFinder();

			particle_finder.RedoWithNewMinimumDistanceFromEdges();

			DrawResultsFromParticleFinder();

			PickingParametersPanel->Thaw();
			ExpertOptionsPanel->Thaw();
		}
	}
	Layout();
	LeftPanel->Layout();
	RightPanel->Layout();
}

void MyFindParticlesPanel::FillRunProfileComboBox()
{
	RunProfileComboBox->FillWithRunProfiles();
}

void MyFindParticlesPanel::OnGroupComboBox( wxCommandEvent& event )
{
	FillImageComboBox();
}

void MyFindParticlesPanel::OnImageComboBox( wxCommandEvent & event )
{
	if (AutoPickRefreshCheckBox->GetValue())
	{
		wxBusyCursor wait;

		PickingParametersPanel->Freeze();
		ExpertOptionsPanel->Freeze();

		SetAllUserParametersForParticleFinder();

		particle_finder.DoItAll(false);

		DrawResultsFromParticleFinder();

		PickingParametersPanel->Thaw();
		ExpertOptionsPanel->Thaw();
	}
}

void MyFindParticlesPanel::OnReferenceVolumeComboBox( wxCommandEvent & event )
{
	
	// Get symmetry of volume asset and set the symmetry dropdown accordingly
	int volume_asset_id = volume_asset_panel->ReturnAssetID(ReferenceSelectPanel->ReturnSelection());
	wxString sym = main_frame->current_project.database.ReturnSymmetryGivenVolumeAssetID(volume_asset_id);
	SymmetryComboBox->SetStringSelection(sym); // if we don't know the sym, an empty string will have been return, and this will fail


	if (AutoPickRefreshCheckBox->GetValue())
	{
		wxBusyCursor wait;

		PickingParametersPanel->Freeze();
		ExpertOptionsPanel->Freeze();

		PrepareTemplatesFromReferenceVolume();
		SetAllUserParametersForParticleFinder();

		particle_finder.DoItAll(false);

		DrawResultsFromParticleFinder();

		PickingParametersPanel->Thaw();
		ExpertOptionsPanel->Thaw();
	}
}

void MyFindParticlesPanel::OnUpdateUI( wxUpdateUIEvent& event )
{
	// are there enough members in the selected group.
	if (main_frame->current_project.is_open == false)
	{
		RunProfileComboBox->Enable(false);
		GroupComboBox->Enable(false);
		PickingAlgorithmComboBox->Enable(false);
		ExpertToggleButton->Enable(false);
		StartPickingButton->Enable(false);
		ReferenceSelectPanel->Enable(false);
		SymmetryComboBox->Enable(false);
		if (PickingParametersPanel->IsShown())
		{
			PickingParametersPanel->Show(false);
			Layout();
			LeftPanel->Layout();
			RightPanel->Layout();
		}
		if (PickingResultsPanel->IsShown())
		{
			PickingResultsPanel->Show(false);
			Layout();
			LeftPanel->Layout();
			RightPanel->Layout();
		}
		if (!InfoPanel->IsShown())
		{
			InfoPanel->Show(true);
			Layout();
			LeftPanel->Layout();
			RightPanel->Layout();
		}
		if (FindParticlesSplitterWindow->IsSplit())
		{
			FindParticlesSplitterWindow->Unsplit(LeftPanel);
			Layout();
			LeftPanel->Layout();
			RightPanel->Layout();
		}

		// Do I need to deal with these too?
		//ExpertOptionsPanel->Show(false);
		//OutputTextPanel->Show(true);



		if (PleaseEstimateCTFStaticText->IsShown())
		{
			PleaseEstimateCTFStaticText->Show(false);
			Layout();
			LeftPanel->Layout();
			RightPanel->Layout();
		}

		if (ReferenceSelectPanel->GetCount() > 0)
		{
			ReferenceSelectPanel->Clear();
			ReferenceSelectPanel->ChangeValue("");
		}
	}
	else // a project is open
	{
		//Enable(true);

		if (group_combo_is_dirty == true)
		{
			FillGroupComboBox();
			FillImageComboBox();
			CheckWhetherGroupsCanBePicked();
			group_combo_is_dirty = false;
		}

		if (run_profiles_are_dirty == true)
		{
			FillRunProfileComboBox();
			run_profiles_are_dirty = false;
		}

		if (running_job == false)
		{
			RunProfileComboBox->Enable(true);
			GroupComboBox->Enable(true);
			ExpertToggleButton->Enable(true);
			ShowPickingParametersPanel();
			if (image_asset_panel->all_groups_list->groups[GroupComboBox->GetSelection()].can_be_picked)
			{
				if (PleaseEstimateCTFStaticText->IsShown())
				{
					PleaseEstimateCTFStaticText->Show(false);
					Layout();
					LeftPanel->Layout();
					RightPanel->Layout();
				}
			}
			else
			{
				if (!PleaseEstimateCTFStaticText->IsShown())
				{
					PleaseEstimateCTFStaticText->Show(true);
					Layout();
					LeftPanel->Layout();
					RightPanel->Layout();
				}
			}

			// Only give a choice of picking algorithm if there is at least one volume available
			if (ReferenceSelectPanel->GetCount() > 0)
			{
				PickingAlgorithmComboBox->Enable(true);
			}
			else
			{
				PickingAlgorithmComboBox->Enable(false);
				PickingAlgorithmComboBox->SetSelection(ab_initio);
			}
			

			if (PickingAlgorithmComboBox->GetCurrentSelection() >= 0)
			{
				ExpertToggleButton->Enable(true);
			}
			if (RunProfileComboBox->GetCount() > 0)
			{
				if (image_asset_panel->ReturnGroupSize(GroupComboBox->GetSelection()) > 0 && run_profiles_panel->run_profile_manager.ReturnTotalJobs(RunProfileComboBox->GetSelection()) > 0 && PickingAlgorithmComboBox->GetCurrentSelection() >= 0 && image_asset_panel->all_groups_list->groups[GroupComboBox->GetSelection()].can_be_picked)
				{
					StartPickingButton->Enable(true);
				}
				else StartPickingButton->Enable(false);
			}
			else
			{
				StartPickingButton->Enable(false);
			}

			if (PickingAlgorithmComboBox->GetCurrentSelection() == reference_volume && ReferenceSelectPanel->GetCount() > 0)
			{
				ReferenceSelectPanel->Enable(true);
				ReferenceVolumeStaticText->Enable(true);
				ProjectionIntervalStaticText->Enable(true);
				ProjectionIntervalNumericCtrl->Enable(true);
				TemplateRadiusStaticText->Enable(true);
				TemplateRadiusNumericCtrl->Enable(true);
				SymmetryStaticText->Enable(true);
				SymmetryComboBox->Enable(true);
				NumberOfTemplateRotationsSpinCtrl->Enable(true);
				NumberOfTemplateRotationsStaticText->Enable(true);
			}
			else
			{
				ReferenceSelectPanel->Enable(false);
				ReferenceVolumeStaticText->Enable(false);
				ProjectionIntervalStaticText->Enable(false);
				ProjectionIntervalNumericCtrl->Enable(false);
				TemplateRadiusStaticText->Enable(true);
				TemplateRadiusNumericCtrl->Enable(true);
				SymmetryStaticText->Enable(false);
				SymmetryComboBox->Enable(false);
				NumberOfTemplateRotationsSpinCtrl->SetValue(1);
				NumberOfTemplateRotationsSpinCtrl->Enable(false);
				NumberOfTemplateRotationsStaticText->Enable(false);
			}

			if (volumes_are_dirty == true)
			{
				ReferenceSelectPanel->FillComboBox();
				volumes_are_dirty = false;
			}
		}
		else // there is a job running
		{
			ExpertToggleButton->Enable(false);
			GroupComboBox->Enable(false);
			PickingAlgorithmComboBox->Enable(false);
			RunProfileComboBox->Enable(false);
			//StartAlignmentButton->SetLabel("Stop Job");
			//StartAlignmentButton->Enable(true);
			ReferenceSelectPanel->Enable(false);
			SymmetryComboBox->Enable(false);
		}

	}




}

void MyFindParticlesPanel::OnExpertOptionsToggle( wxCommandEvent& event )
{
	ExpertOptionsPanel->Show(ExpertToggleButton->GetValue());
	Layout();
	LeftPanel->Layout();
	RightPanel->Layout();
}

void MyFindParticlesPanel::PrepareTemplatesFromReferenceVolume()
{
	MyDebugAssertTrue(PickingAlgorithmComboBox->GetSelection() == reference_volume,"Oops. Doesn't make sense to prepare templates from volume if that's not the algorithm chosen by the user");

	// Choose a filename for the projections
	projections_filename.SetPath(main_frame->current_project.scratch_directory.GetFullPath());
	projections_filename.SetFullName(wxString::Format("picking_templates.mrc"));
	wxPrintf("Filename for projections: %s\n",projections_filename.GetFullPath());

	// Prepare to compute projections
	ImageFile reference_volume_file(volume_asset_panel->ReturnAssetLongFilename(ReferenceSelectPanel->ReturnSelection()).ToStdString());
	ReconstructedVolume reference_3d;
	wxPrintf("Using symmetry: "+SymmetryComboBox->GetValue()+"\n");
	reference_3d.InitWithDimensions(reference_volume_file.ReturnXSize(), reference_volume_file.ReturnYSize(), reference_volume_file.ReturnZSize(), volume_asset_panel->ReturnAssetPointer(ReferenceSelectPanel->ReturnSelection())->pixel_size, SymmetryComboBox->GetValue());
	reference_3d.density_map->ReadSlices(&reference_volume_file, 1, reference_3d.density_map->logical_z_dimension);
	reference_3d.mask_radius = float(reference_volume_file.ReturnXSize()) * 0.45;
	reference_3d.density_map->CorrectSinc(reference_3d.mask_radius / reference_3d.pixel_size);
	reference_3d.PrepareForProjections(0.0,2.0f * reference_3d.pixel_size);

	// Prepare the regular grid of projection directions
	ParameterMap parameter_map; // needed for Euler search init
	parameter_map.SetAllTrue();
	EulerSearch euler_search;
	euler_search.InitGrid(SymmetryComboBox->GetValue(),ProjectionIntervalNumericCtrl->ReturnValue(),0.0f,0.0f,360.0f,360.0f,0.0f,2.0f,parameter_map,1);
	if (SymmetryComboBox->GetValue().StartsWith("C1")) { euler_search.theta_max = 180.0f; }
	euler_search.CalculateGridSearchPositions(false);

	// Padding of the 3D volume (? optional ?)

	// Prepare a projection image and open a file to write them
	Image projection_image;
	projection_image.Allocate(reference_3d.density_map->logical_x_dimension,reference_3d.density_map->logical_y_dimension,false);
	MRCFile projections_file(projections_filename.GetFullPath().ToStdString(),true);

	// Loop over projections
	AnglesAndShifts current_parameters;
	for (int projection_counter = 0; projection_counter < euler_search.number_of_search_positions; projection_counter++)
	{
		current_parameters.Init(euler_search.list_of_search_parameters[projection_counter][0], euler_search.list_of_search_parameters[projection_counter][1],0.0f,0.0f,0.0f);
		reference_3d.density_map->ExtractSlice(projection_image,current_parameters);
		projection_image.complex_values[0] = reference_3d.density_map->complex_values[0];
		projection_image.SwapRealSpaceQuadrants();
		projection_image.BackwardFFT();
		// Add a soft mask (to dampen the aliases)
		projection_image.CosineMask(1.05f * TemplateRadiusNumericCtrl->ReturnValue()/reference_3d.pixel_size + 5.0,10.0);
		// Write the projection to disk
		projection_image.WriteSlice(&projections_file,projection_counter+1);
	}
	projections_file.PrintInfo();
	projections_file.CloseFile();
	wxPrintf("Templates have been written to %s\n", projections_file.filename);
	
}

void MyFindParticlesPanel::SetAllUserParametersForParticleFinder()
{
	ImageAsset * current_image_asset;
	double acceleration_voltage;
	double spherical_aberration;
	double amplitude_contrast;
	double defocus_1;
	double defocus_2;
	double astigmatism_angle;
	double additional_phase_shift;
	double iciness;
	const bool already_have_templates = (PickingAlgorithmComboBox->GetCurrentSelection() == reference_volume);
	const bool average_templates_radially = false;
	const int number_of_template_rotations = NumberOfTemplateRotationsSpinCtrl->GetValue();
	const int output_stack_box_size = 0;
	wxString image_file;
	float pixel_size = 10;
	float minimum_distance_from_edge;

	current_image_asset = image_asset_panel->ReturnAssetPointer(image_asset_panel->ReturnGroupMember(GroupComboBox->GetSelection(), ImageComboBox->GetSelection()));

	main_frame->current_project.database.GetCTFParameters(current_image_asset->ctf_estimation_id,acceleration_voltage,spherical_aberration,amplitude_contrast,defocus_1,defocus_2,astigmatism_angle,additional_phase_shift,iciness);

	// if we have a scaled version of the image, then use that instead..

	wxString small_image_filename = main_frame->current_project.image_asset_directory.GetFullPath();;
	small_image_filename += wxString::Format("/Scaled/%s", current_image_asset->filename.GetFullName());

	if (DoesFileExist(small_image_filename) == true)
	{
		image_file = small_image_filename;
		int largest_dimension =  std::max(current_image_asset->x_size, current_image_asset->y_size);
		float scale_factor = float(SCALED_IMAGE_SIZE) / float(largest_dimension);
		pixel_size = current_image_asset->pixel_size / float(scale_factor);
		minimum_distance_from_edge = MinimumDistanceFromEdgesSpinCtrl->GetValue() * scale_factor;
	}

	if (DoesFileExist(small_image_filename) == false || pixel_size > HighestResolutionNumericCtrl->ReturnValue() / 2)
	{
	//	wxPrintf("not using scaled, pixel size = %f\n", pixel_size);
		image_file =  current_image_asset->filename.GetFullPath();
		pixel_size = current_image_asset->pixel_size;
		minimum_distance_from_edge = MinimumDistanceFromEdgesSpinCtrl->GetValue();


	}


	particle_finder.SetAllUserParameters(   image_file,
											pixel_size,
											current_image_asset->microscope_voltage,
											current_image_asset->spherical_aberration,
											amplitude_contrast,
											additional_phase_shift,
											defocus_1,
											defocus_2,
											astigmatism_angle,
											already_have_templates,
											projections_filename.GetFullPath(),
											average_templates_radially,
											number_of_template_rotations,
											TemplateRadiusNumericCtrl->ReturnValue(),
											ExclusionRadiusNumericCtrl->ReturnValue(),
											HighestResolutionNumericCtrl->ReturnValue(),
											"no_output_stack.mrc",
											output_stack_box_size,
											minimum_distance_from_edge,
											ThresholdPeakHeightNumericCtrl->ReturnValue(),
											AvoidLowVarianceAreasCheckBox->IsChecked(),
											AvoidHighVarianceAreasCheckBox->IsChecked(),
											LowVarianceThresholdNumericCtrl->ReturnValue(),
											HighVarianceThresholdNumericCtrl->ReturnValue(),
											AvoidAbnormalLocalMeanAreasCheckBox->IsChecked(),
											AlgorithmToFindBackgroundChoice->GetSelection(),
											NumberOfBackgroundBoxesSpinCtrl->GetValue(),
											current_image_asset->protein_is_white);
}

void MyFindParticlesPanel::DrawResultsFromParticleFinder()
{
	// Get results in the form of an array of assets
	ArrayOfParticlePositionAssets array_of_assets = particle_finder.ReturnArrayOfParticlePositionAssets();


	ImageAsset * current_image_asset = image_asset_panel->ReturnAssetPointer(image_asset_panel->ReturnGroupMember(GroupComboBox->GetSelection(), ImageComboBox->GetSelection()));
	int image_asset_id = current_image_asset->asset_id;

	double iciness;
	{
		double acceleration_voltage;
		double spherical_aberration;
		double amplitude_contrast;
		double defocus_1;
		double defocus_2;
		double astigmatism_angle;
		double additional_phase_shift;
		main_frame->current_project.database.GetCTFParameters(current_image_asset->ctf_estimation_id,acceleration_voltage,spherical_aberration,amplitude_contrast,defocus_1,defocus_2,astigmatism_angle,additional_phase_shift,iciness);
	}

	PickingResultsPanel->PickingResultsImagePanel->allow_editing_of_coordinates = false;
	PickingResultsPanel->Draw(particle_finder.ReturnMicrographFilename(), array_of_assets, ExclusionRadiusNumericCtrl->ReturnValue(), particle_finder.ReturnOriginalMicrographPixelSize(), particle_finder.ReturnMicrographCTFWithOriginalPixelSize(),image_asset_id,iciness);

}

void MyFindParticlesPanel::OnTestOnCurrentMicrographButtonClick( wxCommandEvent & event )
{

	PickingResultsPanel->Show(true);
	InfoPanel->Show(false);
	Layout();
	LeftPanel->Layout();
	RightPanel->Layout();

	{
		wxBusyCursor wait;

		PickingParametersPanel->Freeze();
		ExpertOptionsPanel->Freeze();

		if (PickingAlgorithmComboBox->GetSelection() == reference_volume) 
		{
			PrepareTemplatesFromReferenceVolume();
		}

		SetAllUserParametersForParticleFinder();

		particle_finder.DoItAll(false);

		DrawResultsFromParticleFinder();

		PickingParametersPanel->Thaw();
		ExpertOptionsPanel->Thaw();
	}

}

void MyFindParticlesPanel::StartPickingClick( wxCommandEvent& event )
{
	MyDebugAssertTrue(buffered_results == NULL, "Error: buffered results not null")

	active_group.CopyFrom(&image_asset_panel->all_groups_list->groups[GroupComboBox->GetSelection()]);

	// Package the job details..

	long counter;
	long number_of_jobs =  active_group.number_of_members; // how many images / movies in the selected group..

	bool ok_number_conversion;

	int number_of_processes;

	int current_asset_id;
	int parent_asset_id;
	int number_of_previous_estimations;

	ImageAsset * current_image_asset;

	wxString buffer_filename;

	std::string input_filename;
	float		pixel_size;
	double 		acceleration_voltage;
	double      spherical_aberration;
	double 		amplitude_contrast;
	double		defocus_1;
	double		defocus_2;
	double		astigmatism_angle;
	double		additional_phase_shift;
	double		iciness;

	bool		already_have_templates = false;
	std::string	templates_filename = "no_templates.mrcs";
	bool		average_templates_radially = true;
	int			number_of_template_rotations = NumberOfTemplateRotationsSpinCtrl->GetValue();
	float		typical_radius = TemplateRadiusNumericCtrl->ReturnValue();
	float		maximum_radius = ExclusionRadiusNumericCtrl->ReturnValue();
	float		highest_resolution_to_use = HighestResolutionNumericCtrl->ReturnValue();
	std::string	output_stack_filename;
	int			output_stack_box_size = 0;
	int			minimum_distance_from_edges = 128;
	float		picking_threshold = ThresholdPeakHeightNumericCtrl->ReturnValue();
	int			number_of_previous_picks;
	bool		avoid_low_variance_areas;
	bool		avoid_high_variance_areas;
	float 		low_variance_threshold_in_fwhm;
	float		high_variance_threshold_in_fwhm;
	bool 		avoid_high_low_mean_areas;
	int			algorithm_to_find_background;
	int			number_of_background_boxes;
	bool 		particles_are_white;

	// allocate space for the buffered results..

	buffered_results = new JobResult[number_of_jobs];

	// read the options form the gui..

	switch(PickingAlgorithmComboBox->GetSelection())
	{
	case(ab_initio) :
		already_have_templates = false;
		templates_filename = "no_templates.mrc";
		average_templates_radially = false;
		number_of_template_rotations = 1;
		typical_radius = TemplateRadiusNumericCtrl->ReturnValue();
		maximum_radius = ExclusionRadiusNumericCtrl->ReturnValue();
	break;
	default :
		MyDebugAssertTrue(false,"Oops, unknown picking algorithm: %i\n",PickingAlgorithmComboBox->GetSelection());
	}

	highest_resolution_to_use = HighestResolutionNumericCtrl->ReturnValue();
	minimum_distance_from_edges = MinimumDistanceFromEdgesSpinCtrl->GetValue();
	picking_threshold = ThresholdPeakHeightNumericCtrl->ReturnValue();
	avoid_low_variance_areas = AvoidLowVarianceAreasCheckBox->GetValue();
	avoid_high_variance_areas = AvoidHighVarianceAreasCheckBox->GetValue();
	low_variance_threshold_in_fwhm = LowVarianceThresholdNumericCtrl->ReturnValue();
	high_variance_threshold_in_fwhm = HighVarianceThresholdNumericCtrl->ReturnValue();
	avoid_high_low_mean_areas = AvoidAbnormalLocalMeanAreasCheckBox->GetValue();
	algorithm_to_find_background = AlgorithmToFindBackgroundChoice->GetSelection();
	number_of_background_boxes = NumberOfBackgroundBoxesSpinCtrl->GetValue();


	current_job_package.Reset(run_profiles_panel->run_profile_manager.run_profiles[RunProfileComboBox->GetSelection()], "find_particles", number_of_jobs);


	OneSecondProgressDialog *my_progress_dialog = new OneSecondProgressDialog ("Preparing Job", "Preparing Job...", number_of_jobs, this, wxPD_REMAINING_TIME | wxPD_AUTO_HIDE| wxPD_APP_MODAL);

	for (counter = 0; counter < number_of_jobs; counter++)
	{

		current_image_asset = image_asset_panel->ReturnAssetPointer(active_group.members[counter]);

		input_filename 			=	current_image_asset->filename.GetFullPath().ToStdString();
		pixel_size				=	current_image_asset->pixel_size;
		acceleration_voltage	=	current_image_asset->microscope_voltage;
		spherical_aberration	=	current_image_asset->spherical_aberration;
		particles_are_white     = 	current_image_asset->protein_is_white;

		main_frame->current_project.database.GetCTFParameters(current_image_asset->ctf_estimation_id,acceleration_voltage,spherical_aberration,amplitude_contrast,defocus_1,defocus_2,astigmatism_angle,additional_phase_shift,iciness);


		number_of_previous_picks = main_frame->current_project.database.ReturnNumberOfPreviousParticlePicksByAssetID(current_image_asset->asset_id);

		output_stack_filename = main_frame->current_project.particle_position_asset_directory.GetFullPath();
		output_stack_filename += wxString::Format("/%s_COOS_%i.mrc", wxFileName::StripExtension(current_image_asset->ReturnShortNameString()),number_of_previous_picks);


		current_job_package.AddJob("tffffffffbtbiffftiifbbffbiib",	input_filename.c_str(), // 0
															pixel_size,
															acceleration_voltage,
															spherical_aberration,
															amplitude_contrast,
															additional_phase_shift, // 5
															defocus_1,
															defocus_2,
															astigmatism_angle,
															already_have_templates,
															templates_filename.c_str(),
															average_templates_radially,
															number_of_template_rotations,
															typical_radius,
															maximum_radius, // 14
															highest_resolution_to_use,
															output_stack_filename.c_str(),
															output_stack_box_size,
															minimum_distance_from_edges,
															picking_threshold,
															avoid_low_variance_areas,
															avoid_high_variance_areas,
															low_variance_threshold_in_fwhm,
															high_variance_threshold_in_fwhm,
															avoid_high_low_mean_areas,
															algorithm_to_find_background,
															number_of_background_boxes,
															particles_are_white
															);

		my_progress_dialog->Update(counter + 1);
	}

	// launch a controller

	my_job_id = main_frame->job_controller.AddJob(this, run_profiles_panel->run_profile_manager.run_profiles[RunProfileComboBox->GetSelection()].manager_command, run_profiles_panel->run_profile_manager.run_profiles[RunProfileComboBox->GetSelection()].gui_address);

	my_progress_dialog->Destroy();

	number_of_particles_picked = 0;

	if (my_job_id != -1)
	{
		SetNumberConnectedTextToZeroAndStartTracking();

		StartPanel->Show(false);
		ProgressPanel->Show(true);

		PickingParametersPanel->Show(false);
		ExpertOptionsPanel->Show(false);
		InfoPanel->Show(false);
		OutputTextPanel->Show(true);
		PickingResultsPanel->Show(true);

		ExpertToggleButton->Enable(false);
		GroupComboBox->Enable(false);
		PickingAlgorithmComboBox->Enable(false);
		Layout();
		LeftPanel->Layout();
		RightPanel->Layout();

		running_job = true;


	}
	ProgressBar->Pulse();

}

void MyFindParticlesPanel::FinishButtonClick( wxCommandEvent& event )
{
	ProgressBar->SetValue(0);
	TimeRemainingText->SetLabel("Time Remaining : ???h:??m:??s");
    CancelAlignmentButton->Show(true);
	FinishButton->Show(false);

	ProgressPanel->Show(false);
	StartPanel->Show(true);
	OutputTextPanel->Show(false);
	output_textctrl->Clear();
	PickingResultsPanel->Show(false);
	PickingResultsPanel->Clear();
	AutoPickRefreshCheckBox->SetValue(false);
	//graph_is_hidden = true;
	InfoPanel->Show(true);

	if (PickingAlgorithmComboBox->GetCurrentSelection() >= 0)
	{
		PickingParametersPanel->Show(true);
		ExpertOptionsPanel->Show(ExpertToggleButton->GetValue());
	}
	else
	{
		ExpertToggleButton->Enable(false);
	}


	running_job = false;
	Layout();
	LeftPanel->Layout();
	RightPanel->Layout();

	//CTFResultsPanel->CTF2DResultsPanel->should_show = false;
	//CTFResultsPanel->CTF2DResultsPanel->Refresh();



}

void MyFindParticlesPanel::TerminateButtonClick( wxCommandEvent& event )
{
	// kill the job, this will kill the socket to terminate downstream processes
	// - this will have to be improved when clever network failure is incorporated


	main_frame->job_controller.KillJob(my_job_id);

	WriteInfoText("Terminated Job");
	TimeRemainingText->SetLabel("Time Remaining : Terminated");
	CancelAlignmentButton->Show(false);
	FinishButton->Show(true);
	ProgressPanel->Layout();

	if (buffered_results != NULL)
	{
		delete [] buffered_results;
		buffered_results = NULL;
	}

	//running_job = false;


}

void MyFindParticlesPanel::WriteInfoText(wxString text_to_write)
{
	output_textctrl->SetDefaultStyle(wxTextAttr(*wxBLACK));
	output_textctrl->AppendText(text_to_write);

	if (text_to_write.EndsWith("\n") == false)	 output_textctrl->AppendText("\n");
}

void MyFindParticlesPanel::WriteErrorText(wxString text_to_write)
{
	 output_textctrl->SetDefaultStyle(wxTextAttr(*wxRED));
	 output_textctrl->AppendText(text_to_write);

	 if (text_to_write.EndsWith("\n") == false)	 output_textctrl->AppendText("\n");
}


// Go over all groups and for each one check whether it's ready to be picked (e.g. that all its images have CTF estimates)
void MyFindParticlesPanel::CheckWhetherGroupsCanBePicked()
{
	int number_of_images_with_ctf_estimates;
	int number_of_images_in_group;
	bool current_image_has_estimate;
	int current_image_id;


	number_of_images_with_ctf_estimates = main_frame->current_project.database.ReturnNumberOfImageAssetsWithCTFEstimates();
	if (number_of_images_with_ctf_estimates > 0)
		{
		int ids_of_images_with_ctf_estimates [number_of_images_with_ctf_estimates];

		main_frame->current_project.database.GetUniqueIDsOfImagesWithCTFEstimations(ids_of_images_with_ctf_estimates,number_of_images_with_ctf_estimates);


		for (int group_counter = 0; group_counter < image_asset_panel->ReturnNumberOfGroups(); group_counter ++ )
		{
			// We start by assuming the current group can be picked
			image_asset_panel->all_groups_list->groups[group_counter].can_be_picked = true;

			number_of_images_in_group = image_asset_panel->ReturnGroupSize(group_counter);

			for ( int counter_in_group = 0; counter_in_group < number_of_images_in_group; counter_in_group ++ )
			{
				current_image_id = image_asset_panel->ReturnGroupMemberID(group_counter,counter_in_group);
				current_image_has_estimate = false;
				for ( int counter_in_estimates = 0; counter_in_estimates < number_of_images_with_ctf_estimates; counter_in_estimates ++ )
				{
					if (ids_of_images_with_ctf_estimates[counter_in_estimates] == current_image_id) current_image_has_estimate = true;
				}
				if (! current_image_has_estimate) {
					// Current group cannot be picked
					image_asset_panel->all_groups_list->groups[group_counter].can_be_picked = false;
				}
			}
		}
	}
	else
	{
		// No images have CTF estimates yet
		for (int group_counter = 0; group_counter < image_asset_panel->ReturnNumberOfGroups(); group_counter ++ )
		{
			image_asset_panel->all_groups_list->groups[group_counter].can_be_picked = false;
		}
	}
}



void  MyFindParticlesPanel::ProcessResult(JobResult *result_to_process, const int &wanted_job_number) // this will have to be overidden in the parent clas when i make it.
{

	if (result_to_process != NULL)
	{
		long current_time = time(NULL);
		int job_number = wanted_job_number;

		job_number = result_to_process->job_number;
		number_of_particles_picked += result_to_process->result_size / 5;

		if (current_time - time_of_last_result_update > 3)
		{
			ArrayOfParticlePositionAssets array_of_assets;
			//wxPrintf("processing result. filename = %s\n",current_job_package.jobs[job_number].arguments[0].ReturnStringArgument());
			wxString image_filename = current_job_package.jobs[job_number].arguments[0].ReturnStringArgument();
			int image_asset_id = -1;
			double iciness = 0.0;
			if (result_to_process)
			{
				image_asset_id = image_asset_panel->ReturnAssetID(active_group.members[result_to_process->job_number]);
				array_of_assets = ParticlePositionsFromJobResults(result_to_process,image_asset_id,1,1,1);
				ImageAsset * current_image_asset = image_asset_panel->ReturnAssetPointer(image_asset_panel->ReturnArrayPositionFromAssetID(image_asset_id));
				double acceleration_voltage,spherical_aberration,amplitude_contrast,defocus_1,defocus_2,astigmatism_angle,additional_phase_shift;
				main_frame->current_project.database.GetCTFParameters(current_image_asset->ctf_estimation_id,acceleration_voltage,spherical_aberration,amplitude_contrast,defocus_1,defocus_2,astigmatism_angle,additional_phase_shift,iciness);
			}
			float radius_in_angstroms = current_job_package.jobs[job_number].arguments[14].ReturnFloatArgument();
			float pixel_size_in_angstroms = current_job_package.jobs[job_number].arguments[1].ReturnFloatArgument();
			PickingResultsPanel->PickingResultsImagePanel->allow_editing_of_coordinates = false;
			CTF ctf_of_current_result;
			ctf_of_current_result.Init( current_job_package.jobs[job_number].arguments[2].ReturnFloatArgument(), //akv
										current_job_package.jobs[job_number].arguments[3].ReturnFloatArgument(), //cs
										current_job_package.jobs[job_number].arguments[4].ReturnFloatArgument(), //amp cont
										current_job_package.jobs[job_number].arguments[6].ReturnFloatArgument(), //def1
										current_job_package.jobs[job_number].arguments[7].ReturnFloatArgument(), //def2
										current_job_package.jobs[job_number].arguments[8].ReturnFloatArgument(), //ang ast
										pixel_size_in_angstroms,
										current_job_package.jobs[job_number].arguments[5].ReturnFloatArgument()); //phase shift
			PickingResultsPanel->Draw(image_filename, array_of_assets, radius_in_angstroms, pixel_size_in_angstroms,ctf_of_current_result,image_asset_id,iciness);

			time_of_last_result_update = time(NULL);
		}

		//my_job_tracker.MarkJobFinished();
		if (my_job_tracker.ShouldUpdate() == true) UpdateProgressBar();

		// store the results..
		buffered_results[result_to_process->job_number] = result_to_process;
	}
	else
	{
		//my_job_tracker.MarkJobFinished();
		if (my_job_tracker.ShouldUpdate() == true) UpdateProgressBar();
/*
		if (my_job_tracker.total_number_of_finished_jobs == my_job_tracker.total_number_of_jobs)
		{
			// job has really finished, so we can write to the database...k
			main_frame->job_controller.KillJob(my_job_id);
			WriteResultToDataBase();

			if (buffered_results != NULL)
			{
				delete [] buffered_results;
				buffered_results = NULL;
			}

			WriteInfoText(wxString::Format("All Jobs have finished. %i particles were picked.", number_of_particles_picked));
			ProgressBar->SetValue(100);
			TimeRemainingText->SetLabel("Time Remaining : All Done!");
			CancelAlignmentButton->Show(false);
			FinishButton->Show(true);
			ProgressPanel->Layout();
		} */
	}

}

void MyFindParticlesPanel::ProcessAllJobsFinished()
{

	MyDebugAssertTrue(my_job_tracker.total_number_of_finished_jobs == my_job_tracker.total_number_of_jobs,"In ProcessAllJobsFinished, but total_number_of_finished_jobs != total_number_of_jobs. Oops.");

	// Update the GUI with project timings
	extern MyOverviewPanel *overview_panel;
	overview_panel->SetProjectInfo();

	//
	WriteResultToDataBase();


	if (buffered_results != NULL)
	{
		delete [] buffered_results;
		buffered_results = NULL;
	}

	// Kill the job (in case it isn't already dead)
	main_frame->job_controller.KillJob(my_job_id);

	WriteInfoText(wxString::Format("All Jobs have finished. %i particles were picked.", number_of_particles_picked));
	ProgressBar->SetValue(100);

	TimeRemainingText->SetLabel(my_job_tracker.ReturnTimeSinceStart().Format("All Done! (%Hh:%Mm:%Ss)"));
	CancelAlignmentButton->Show(false);
	FinishButton->Show(true);
	ProgressPanel->Layout();
}




void MyFindParticlesPanel::WriteResultToDataBase()
{

	long counter;
	int frame_counter;
	int array_location;
	bool have_errors = false;
	int current_asset;
	bool restrain_astigmatism;
	bool find_additional_phase_shift;
	float min_phase_shift;
	float max_phase_shift;
	float phase_shift_step;
	float tolerated_astigmatism;
	wxString current_table_name;
	wxDateTime now = wxDateTime::Now();


	// find the current highest alignment number in the database, then increment by one

	int starting_picking_id = main_frame->current_project.database.ReturnHighestPickingID();
	int picking_id = starting_picking_id + 1;
	int picking_job_id =  main_frame->current_project.database.ReturnHighestPickingJobID() + 1;

	OneSecondProgressDialog *my_progress_dialog = new OneSecondProgressDialog ("Write Results", "Writing results to the database...", my_job_tracker.total_number_of_jobs + particle_position_asset_panel->all_groups_list->number_of_groups + 3, this, wxPD_APP_MODAL);

	// global begin..

	//wxPrintf("\n\n\n\Writing Results to database! - total jobs is %i\n\n\n", my_job_tracker.total_number_of_jobs);

	main_frame->current_project.database.Begin();

	// Record the parameters we used to pick
	main_frame->current_project.database.BeginBatchInsert("PARTICLE_PICKING_LIST",14,
																						"PICKING_ID",
																						"PICKING_JOB_ID",
																						"DATETIME_OF_RUN",
																						"PARENT_IMAGE_ASSET_ID",
																						"PICKING_ALGORITHM",
																						"CHARACTERISTIC_RADIUS",
																						"MAXIMUM_RADIUS",
																						"THRESHOLD_PEAK_HEIGHT",
																						"HIGHEST_RESOLUTION_USED_IN_PICKING",
																						"MIN_DIST_FROM_EDGES",
																						"AVOID_HIGH_VARIANCE",
																						"AVOID_HIGH_LOW_MEAN",
																						"NUM_BACKGROUND_BOXES",
																						"MANUAL_EDIT");
	picking_id = starting_picking_id + 1;
	for (int counter = 0; counter < current_job_package.number_of_jobs; counter ++ )
	{
		main_frame->current_project.database.AddToBatchInsert("iiliirrrriiiii", 		picking_id,
																						picking_job_id,
																						(long int) now.GetAsDOS(),
																						image_asset_panel->ReturnAssetID(active_group.members[counter]),
																						PickingAlgorithmComboBox->GetSelection(),
																						TemplateRadiusNumericCtrl->ReturnValue(),
																						ExclusionRadiusNumericCtrl->ReturnValue(),
																						ThresholdPeakHeightNumericCtrl->ReturnValue(),
																						HighestResolutionNumericCtrl->ReturnValue(),
																						MinimumDistanceFromEdgesSpinCtrl->GetValue(),
																						AvoidHighVarianceAreasCheckBox->GetValue(),
																						AvoidAbnormalLocalMeanAreasCheckBox->GetValue(),
																						NumberOfBackgroundBoxesSpinCtrl->GetValue(),
																						0);
		picking_id ++;
		my_progress_dialog->Update(counter + 1);
	}
	main_frame->current_project.database.EndBatchInsert();

	// Remove group members and assets from the database, one group at a time
	int parent_id;
	wxString sql_command;
	for (int group_counter = 1; group_counter < particle_position_asset_panel->all_groups_list->number_of_groups; group_counter ++)
	{
		for (int job_counter = 0; job_counter < my_job_tracker.total_number_of_jobs; job_counter ++ )
		{
			parent_id = image_asset_panel->ReturnAssetID(active_group.members[job_counter]);
			main_frame->current_project.database.RemoveParticlePositionsWithGivenParentImageIDFromGroup(particle_position_asset_panel->ReturnGroupID(group_counter),parent_id);
		}

		my_progress_dialog->Update( my_job_tracker.total_number_of_jobs + group_counter);
	}

	// Remove from particle_position_assets assets which have a parent_id which is from picking_job_id that we've just done
	main_frame->current_project.database.RemoveParticlePositionAssetsPickedFromImagesAlsoPickedByGivenPickingJobID(picking_job_id);


	// Grab the results and build an array of particle position assets
	ArrayOfParticlePositionAssets array_of_assets;
	ArrayOfParticlePositionAssets temp_array_of_assets;
	int starting_asset_id = 0;
	if (starting_picking_id > 0) starting_asset_id = main_frame->current_project.database.ReturnSingleIntFromSelectCommand(wxString::Format("SELECT MAX(POSITION_ID) FROM PARTICLE_PICKING_RESULTS_%i",picking_job_id-1));
	picking_id = starting_picking_id + 1;
	for (int counter = 0; counter < current_job_package.number_of_jobs; counter ++ )
	{
		temp_array_of_assets = ParticlePositionsFromJobResults(&buffered_results[counter],image_asset_panel->ReturnAssetID(active_group.members[counter]),picking_job_id,picking_id,starting_asset_id);
		WX_APPEND_ARRAY(array_of_assets,temp_array_of_assets);
		starting_asset_id += temp_array_of_assets.GetCount();
		picking_id++;
	}

	my_progress_dialog->Update( current_job_package.number_of_jobs + particle_position_asset_panel->all_groups_list->number_of_groups + 1);

	// Now that we have our array of assets, let's add them to the database
	main_frame->current_project.database.CreateParticlePickingResultsTable(picking_job_id);
	main_frame->current_project.database.AddArrayOfParticlePositionAssetsToResultsTable(picking_job_id,&array_of_assets);
	main_frame->current_project.database.AddArrayOfParticlePositionAssetsToAssetsTable(&array_of_assets);

	my_progress_dialog->Update(current_job_package.number_of_jobs + particle_position_asset_panel->all_groups_list->number_of_groups + 2);

	// global commit..
	main_frame->current_project.database.Commit();


	// At this point, the database should be up-to-date
	particle_position_asset_panel->ImportAllFromDatabase();


	my_progress_dialog->Destroy();


	particle_position_asset_panel->is_dirty = true;
	picking_results_panel->is_dirty = true;

}

ArrayOfParticlePositionAssets MyFindParticlesPanel::ParticlePositionsFromJobResults(JobResult *job_result, const int &parent_image_id, const int &picking_job_id, const int &picking_id, const int &starting_asset_id)
{
	ParticlePositionAsset temp_asset;
	ArrayOfParticlePositionAssets array_of_assets;
	int address_within_results = 0;

	if (job_result->result_size > 0)
	{

		temp_asset.pick_job_id = picking_job_id;
		temp_asset.asset_id = starting_asset_id;

		temp_asset.parent_id = parent_image_id;
		temp_asset.picking_id = picking_id;
		// Loop over picked coordinates

		for (int particle_counter = 0; particle_counter < job_result->result_size / 5; particle_counter ++ )
		{
			address_within_results = particle_counter * 5;
			// Finish setting up the asset. We use an ID that hasn't been used for any other position asset previously.
			temp_asset.asset_id ++;
			temp_asset.x_position = job_result->result_data[address_within_results + 0];
			temp_asset.y_position = job_result->result_data[address_within_results + 1];
			temp_asset.peak_height = job_result->result_data[address_within_results + 2];

			//
			array_of_assets.Add(temp_asset);
		}
	}

	return array_of_assets;

}

void MyFindParticlesPanel::UpdateProgressBar()
{
	ProgressBar->SetValue(my_job_tracker.ReturnPercentCompleted());
	TimeRemainingText->SetLabel(my_job_tracker.ReturnRemainingTime().Format("Time Remaining : %Hh:%Mm:%Ss"));
}

void MyFindParticlesPanel::OnSocketJobResultMsg(JobResult &received_result)
{
	if (received_result.result_size > 0)
	{
		my_job_tracker.MarkJobFinished();
		ProcessResult(&received_result);
	}
}

void MyFindParticlesPanel::SetNumberConnectedText(wxString wanted_text)
{
	NumberConnectedText->SetLabel(wanted_text);
}

void MyFindParticlesPanel::SetTimeRemainingText(wxString wanted_text)
{
	TimeRemainingText->SetLabel(wanted_text);
}

void MyFindParticlesPanel::OnSocketJobFinished(int finished_job_number)
{
	ProcessResult(NULL, finished_job_number);
}

void MyFindParticlesPanel::OnSocketAllJobsFinished()
{
		// Other stuff to do once all jobs finished
		ProcessAllJobsFinished();
}
