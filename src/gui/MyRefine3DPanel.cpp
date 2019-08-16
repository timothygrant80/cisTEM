#include "../core/gui_core_headers.h"

extern MyRefinementPackageAssetPanel *refinement_package_asset_panel;
extern MyRunProfilesPanel *run_profiles_panel;
#ifdef EXPERIMENTAL
extern MyPhenixSettingsPanel *phenix_settings_panel;
#endif
extern MyVolumeAssetPanel *volume_asset_panel;
extern MyRefinementResultsPanel *refinement_results_panel;

wxDEFINE_EVENT(wxEVT_COMMAND_MYTHREAD_COMPLETED, wxThreadEvent);

MyRefine3DPanel::MyRefine3DPanel( wxWindow* parent )
:
Refine3DPanel( parent )
{
	buffered_results = NULL;

	// Fill combo box..

	//FillGroupComboBox();

	my_job_id = -1;
	running_job = false;

//	group_combo_is_dirty = false;
//	run_profiles_are_dirty = false;

	SetInfo();
//	FillGroupComboBox()t
//	FillRunProfileComboBox();

	wxSize input_size = InputSizer->GetMinSize();
	input_size.x += wxSystemSettings::GetMetric(wxSYS_VSCROLL_X);
	input_size.y = -1;
	ExpertPanel->SetMinSize(input_size);
	ExpertPanel->SetSize(input_size);


	// set values //

	/*
	AmplitudeContrastNumericCtrl->SetMinMaxValue(0.0f, 1.0f);
	MinResNumericCtrl->SetMinMaxValue(0.0f, 50.0f);
	MaxResNumericCtrl->SetMinMaxValue(0.0f, 50.0f);
	DefocusStepNumericCtrl->SetMinMaxValue(1.0f, FLT_MAX);
	ToleratedAstigmatismNumericCtrl->SetMinMaxValue(0.0f, FLT_MAX);
	MinPhaseShiftNumericCtrl->SetMinMaxValue(-3.15, 3.15);
	MaxPhaseShiftNumericCtrl->SetMinMaxValue(-3.15, 3.15);
	PhaseShiftStepNumericCtrl->SetMinMaxValue(0.001, 3.15);

	result_bitmap.Create(1,1, 24);
	time_of_last_result_update = time(NULL);*/

	refinement_package_combo_is_dirty = false;
	run_profiles_are_dirty = false;
	input_params_combo_is_dirty = false;
	selected_refinement_package = -1;

	RefinementPackageComboBox->AssetComboBox->Bind(wxEVT_COMMAND_COMBOBOX_SELECTED, &MyRefine3DPanel::OnRefinementPackageComboBox, this);
	Bind(RETURN_PROCESSED_IMAGE_EVT, &MyRefine3DPanel::OnOrthThreadComplete, this);
	Bind(wxEVT_MULTIPLY3DMASKTHREAD_COMPLETED, &MyRefine3DPanel::OnMaskerThreadComplete, this);
	Bind(wxEVT_AUTOMASKERTHREAD_COMPLETED, &MyRefine3DPanel::OnMaskerThreadComplete, this);

	my_refinement_manager.SetParent(this);

	FillRefinementPackagesComboBox();

	long time_of_last_result_update;

	active_orth_thread_id = -1;
	active_mask_thread_id = -1;
	next_thread_id = 1;


}

void MyRefine3DPanel::Reset()
{
	ProgressBar->SetValue(0);
	TimeRemainingText->SetLabel("Time Remaining : ???h:??m:??s");
    CancelAlignmentButton->Show(true);
	FinishButton->Show(false);

	InputParamsPanel->Show(true);
	ProgressPanel->Show(false);
	StartPanel->Show(true);
	OutputTextPanel->Show(false);
	output_textctrl->Clear();
	ShowRefinementResultsPanel->Show(false);
	ShowRefinementResultsPanel->Clear();

	UseMaskCheckBox->SetValue(false);
	LocalRefinementRadio->SetValue(true);
	NumberRoundsSpinCtrl->SetValue(1);
	HighResolutionLimitTextCtrl->ChangeValueFloat(30.0f);

	//CTFResultsPanel->Show(false);
	//graph_is_hidden = true;
	InfoPanel->Show(true);

	RefinementPackageComboBox->Clear();
	InputParametersComboBox->Clear();
	RefinementRunProfileComboBox->Clear();
	ReconstructionRunProfileComboBox->Clear();

	UseMaskCheckBox->SetValue(false);
	ExpertToggleButton->SetValue(false);
	ExpertPanel->Show(false);

	if (running_job == true)
	{
		main_frame->job_controller.KillJob(my_job_id);

		active_mask_thread_id = -1;
		active_orth_thread_id = -1;
		running_job = false;
	}

	if (my_refinement_manager.output_refinement != NULL) delete my_refinement_manager.output_refinement;
	SetDefaults();
	global_delete_refine3d_scratch();
	Layout();
}

void MyRefine3DPanel::SetInfo()
{

	wxLogNull *suppress_png_warnings = new wxLogNull;
//	#include "icons/niko_picture1.cpp"
//	wxBitmap niko_picture1_bmp = wxBITMAP_PNG_FROM_DATA(niko_picture1);

	#include "icons/niko_picture2.cpp"
	wxBitmap niko_picture2_bmp = wxBITMAP_PNG_FROM_DATA(niko_picture2);
	delete suppress_png_warnings;

	InfoText->GetCaret()->Hide();

	InfoText->BeginSuppressUndo();
	InfoText->BeginAlignment(wxTEXT_ALIGNMENT_CENTRE);
	InfoText->BeginBold();
	InfoText->BeginUnderline();
	InfoText->BeginFontSize(14);
	InfoText->WriteText(wxT("3D Refinement & Reconstruction (FrealignX)"));
	InfoText->EndFontSize();
	InfoText->EndBold();
	InfoText->EndUnderline();
	InfoText->Newline();
	InfoText->Newline();
	InfoText->EndAlignment();

	InfoText->BeginAlignment(wxTEXT_ALIGNMENT_LEFT);
	InfoText->WriteText(wxT("The goal of refinement and reconstruction is to obtain 3D maps of the imaged particle at the highest possible resolution. Refinement typically starts with a preexisting structure that serves as a reference to determine initial particle alignment parameters using a global parameter search. In subsequent iterations, these parameters are refined and (optionally) the dataset can be classified into several classes with distinct structural features. This panel allows the user to define a refinement job that includes a set number of iterations (refinement cycles) and number of desired classes to be generated (Lyumkis et al. 2013). The general refinement strategies and options are similar to those available with Frealign and are described in Grigorieff, 2016:"));
	InfoText->Newline();
	InfoText->Newline();
	InfoText->EndAlignment();

/*	InfoText->BeginAlignment(wxTEXT_ALIGNMENT_CENTRE);
	InfoText->WriteImage(niko_picture1_bmp);
	InfoText->Newline();
	InfoText->Newline();
	InfoText->EndAlignment();*/

/*
	InfoText->BeginAlignment(wxTEXT_ALIGNMENT_LEFT);
	InfoText->WriteText(wxT(""));
	InfoText->Newline();
	InfoText->Newline();
	InfoText->EndAlignment();*/

	InfoText->BeginAlignment(wxTEXT_ALIGNMENT_CENTRE);
	InfoText->WriteImage(niko_picture2_bmp);
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
	InfoText->WriteText(wxT("The source of the starting parameters for this refinement run. These can be either set to be random Euler angles and zero X,Y shifts, or they can be the output of a previous refinement (if available)."));
	InfoText->Newline();
	InfoText->BeginBold();
	InfoText->WriteText(wxT("Local Refinement/Global Search : "));
	InfoText->EndBold();
	InfoText->WriteText(wxT("If no starting parameters from a previous refinement are available, they have to be determined in a global search (slow); otherwise it is usually sufficient to perform local refinement (fast)."));
	InfoText->Newline();
	InfoText->BeginBold();
	InfoText->WriteText(wxT("No. of Cycles to Run : "));
	InfoText->EndBold();
	InfoText->WriteText(wxT("The number of refinement cycles to run. For a global search, one is usually sufficient, possibly followed by another one at a later stage in the refinement if the user suspects that the initial reference was limited in quality such that a significant number of particles were misaligned. For local refinement of a single class, typically 3 to 5 cycles are sufficient, possibly followed by another local refinement at increased resolution (see below). If multiple classes are refined, between 30 and 50 cycles should be run to ensure convergence of the classes."));
	InfoText->Newline();
	InfoText->Newline();
	InfoText->EndAlignment();

	InfoText->BeginAlignment(wxTEXT_ALIGNMENT_CENTRE);
	InfoText->BeginBold();
	InfoText->BeginUnderline();
	InfoText->WriteText(wxT("Expert Options"));
	InfoText->EndBold();
	InfoText->EndUnderline();
	InfoText->Newline();
	InfoText->Newline();
	InfoText->EndAlignment();

	InfoText->BeginAlignment(wxTEXT_ALIGNMENT_CENTRE);
	InfoText->BeginUnderline();
	InfoText->WriteText(wxT("General Refinement"));
	InfoText->EndUnderline();
	InfoText->Newline();
	InfoText->Newline();
	InfoText->EndAlignment();

	InfoText->BeginBold();
	InfoText->WriteText(wxT("Low/High-Resolution Limit (Å) : "));
	InfoText->EndBold();
	InfoText->WriteText(wxT("The data used for refinement is usually bandpass-limited to exclude spurious low-resolution features in the particle background (set by the low-resolution limit) and high-resolution noise (set by the high-resolution limit). It is good practice to set the low-resolution limit to 2.5x the approximate particle mask radius. The high-resolution limit should remain significantly below the resolution of the reference used for refinement to enable unbiased resolution estimation using the Fourier Shell Correlation curve."));
	InfoText->Newline();

	InfoText->BeginBold();
	InfoText->WriteText(wxT("Mask Radius (Å) : "));
	InfoText->EndBold();
	InfoText->WriteText(wxT("The radius of the circular mask applied to the input images before refinement starts. This mask should be sufficiently large to include the largest dimension of the particle. When a global search is performed, the radius should be set to include the expected area containing the particle. This area is usually larger than the area defined by the largest dimension of the particle because particles may not be precisely centered."));
	InfoText->Newline();

	InfoText->BeginBold();
	InfoText->WriteText(wxT("Signed CC Resolution Limit (Å) : "));
	InfoText->EndBold();
	InfoText->WriteText(wxT("Particle alignment is done by maximizing a correlation coefficient with the reference. The user has the option to maximize the unsigned correlation coefficient instead (starting at the limit set here) to reduce overfitting (Stewart and Grigorieff, 2004). Overfitting is also reduced by appropriate weighting of the data and this is usually sufficient to achieve good refinement results. The limit set here should therefore be set to 0.0 to maximize the signed correlation at all resolutions, unless there is evidence that there is overfitting. (This feature was formerly known as “FBOOST”.)"));
	InfoText->Newline();

	InfoText->BeginAlignment(wxTEXT_ALIGNMENT_CENTRE);
	InfoText->BeginUnderline();
	InfoText->WriteText(wxT("Global Search"));
	InfoText->EndUnderline();
	InfoText->Newline();
	InfoText->Newline();
	InfoText->EndAlignment();

	InfoText->BeginBold();
	InfoText->WriteText(wxT("Number of Results to Refine : "));
	InfoText->EndBold();
	InfoText->WriteText(wxT("For a global search, an angular grid search is performed and the alignment parameters for the N best matching projections are then refined further in a local refinement. Only the set of parameters yielding the best score (correlation coefficient) is kept. Increasing N will increase the chances of finding the correct particle orientations but will slow down the search. A value of 20 is recommended."));
	InfoText->Newline();

	InfoText->BeginBold();
	InfoText->WriteText(wxT("Also Refine Input Parameters "));
	InfoText->EndBold();
	InfoText->WriteText(wxT("In addition to the N best sets of parameter values found during the grid search, the input set of parameters is also locally refined. Switching this off can help reduce over-fitting that may have biased the input parameters."));
	InfoText->Newline();

	InfoText->BeginBold();
	InfoText->WriteText(wxT("Angular Search Step (°) : "));
	InfoText->EndBold();
	InfoText->WriteText(wxT("The angular step used to generate the search grid for the global search. An appropriate value is suggested by default (depending on particle size and high-resolution limit) but smaller values can be tried if the user suspects that the search misses orientations found in the particle dataset. The smaller the value, the finer the search grid and the slower the search."));
	InfoText->Newline();

	InfoText->BeginBold();
	InfoText->WriteText(wxT("Search Range in X/Y (Å) : "));
	InfoText->EndBold();
	InfoText->WriteText(wxT("The global search can be limited in the X and Y directions (measured from the box center) to ensure that only particles close to the box center are found. This is useful when the particle density is high and particles end up close to each other. In this case, it is usually still possible to align all particles in a cluster of particles (assuming they do not significantly overlap). The values provided here for the search range should be set to exclude the possibility that the same particle is selected twice and counted as two different particles."));
	InfoText->Newline();

	InfoText->BeginAlignment(wxTEXT_ALIGNMENT_CENTRE);
	InfoText->BeginUnderline();
	InfoText->WriteText(wxT("Classification"));
	InfoText->EndUnderline();
	InfoText->Newline();
	InfoText->Newline();
	InfoText->EndAlignment();

	InfoText->BeginBold();
	InfoText->WriteText(wxT("High-Resolution Limit (Å) : "));
	InfoText->EndBold();
	InfoText->WriteText(wxT("The limit set here is analogous to the high-resolution limit set for refinement. It cannot exceed the refinement limit. Setting it to a lower resolution may increase the useful SNR for classification and lead to better separation of particles with different structural features. However, at lower resolution the classification may also become less sensitive to heterogeneity represented by smaller structural features."));
	InfoText->Newline();

	InfoText->BeginBold();
	InfoText->WriteText(wxT("Focused Classification? : "));
	InfoText->EndBold();
	InfoText->WriteText(wxT("Classification can be performed based on structural variability in a defined region of the particle. This is useful when there are multiple regions that have uncorrelated structural variability. Using focused classification, each of these regions can be classified in turn. The focus feature can also be used to reduce noise from other parts of the images and increase the useful SNR for classification. The focus region is defined by a sphere with coordinates and radius in the following four inputs. (This feature was formerly known as “focus_mask”.)"));
	InfoText->Newline();

	InfoText->BeginBold();
	InfoText->WriteText(wxT("Sphere X/Y/Z Co-ordinate and Radius (Å) : "));
	InfoText->EndBold();
	InfoText->WriteText(wxT("These values describe a spherical region inside the particle that contains the structural variability to focus on."));
	InfoText->Newline();

	InfoText->BeginAlignment(wxTEXT_ALIGNMENT_CENTRE);
	InfoText->BeginUnderline();
	InfoText->WriteText(wxT("CTF"));
	InfoText->EndUnderline();
	InfoText->Newline();
	InfoText->Newline();
	InfoText->EndAlignment();

	InfoText->BeginBold();
	InfoText->WriteText(wxT("Refine CTF? : "));
	InfoText->EndBold();
	InfoText->WriteText(wxT("Should the CTF be refined as well? This is only recommended for high-resolution data that yield reconstructions of better than 4 Å resolution, and for particles of sufficient molecular mass (500 kDa and higher)."));
	InfoText->Newline();

	InfoText->BeginBold();
	InfoText->WriteText(wxT("Defocus Search Range (Å) : "));
	InfoText->EndBold();
	InfoText->WriteText(wxT("The range of defocus values to search over for each particle. A search with the step size given in the next input will be performed starting at the defocus values determined in the previous refinement cycle minus the search range, up to values plus the search range. The search steps will be applied to both defocus values, keeping the estimated astigmatism constant."));
	InfoText->Newline();

	InfoText->BeginBold();
	InfoText->WriteText(wxT("Defocus Search Step (Å) : "));
	InfoText->EndBold();
	InfoText->WriteText(wxT("The search step for the defocus search."));
	InfoText->Newline();

	InfoText->BeginAlignment(wxTEXT_ALIGNMENT_CENTRE);
	InfoText->BeginUnderline();
	InfoText->WriteText(wxT("Reconstruction"));
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
	InfoText->WriteText(wxT("Particles with a score lower than the threshold will be excluded from the reconstruction. This provides a way to exclude particles that may score low because of misalignment or damage. A value = 0 will select all particles; 0 < value <= 1 will be interpreted as a percentage; value > 1 will be interpreted as a fixed score threshold."));
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

//	InfoText->EndAlignment();
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
	InfoText->WriteText(wxT("Stewart, A., Grigorieff, N.,"));
	InfoText->EndBold();
	InfoText->WriteText(wxT(" 2004. Noise bias in the refinement of structures derived from single particles. Ultramicroscopy 102, 67-84. "));
	InfoText->BeginURL("http://dx.doi.org/10.1016/j.ultramic.2004.08.008");
	InfoText->BeginUnderline();
	InfoText->BeginTextColour(*wxBLUE);
	InfoText->WriteText(wxT("dio:10.1016/j.ultramic.2004.08.008"));
	InfoText->EndURL();
	InfoText->EndTextColour();
	InfoText->EndUnderline();
	InfoText->EndAlignment();
	InfoText->Newline();
	InfoText->Newline();

	InfoText->BeginAlignment(wxTEXT_ALIGNMENT_LEFT);
	InfoText->BeginBold();
	InfoText->WriteText(wxT("Sindelar, C. V., Grigorieff, N.,"));
	InfoText->EndBold();
	InfoText->WriteText(wxT(" 2012. Optimal noise reduction in 3D reconstructions of single particles using a volume-normalized filter. J. Struct. Biol. 180, 26-38."));
	InfoText->BeginURL("http://dx.doi.org/10.1016/j.jsb.2012.05.005");
	InfoText->BeginUnderline();
	InfoText->BeginTextColour(*wxBLUE);
	InfoText->WriteText(wxT("dio:10.1016/j.jsb.2012.05.005"));
	InfoText->EndURL();
	InfoText->EndTextColour();
	InfoText->EndUnderline();
	InfoText->EndAlignment();
	InfoText->Newline();
	InfoText->Newline();

	InfoText->BeginAlignment(wxTEXT_ALIGNMENT_LEFT);
	InfoText->BeginBold();
	InfoText->WriteText(wxT("Lyumkis, D., Brilot, A. F., Theobald, D. L., Grigorieff, N.,"));
	InfoText->EndBold();
	InfoText->WriteText(wxT(" 2013. Likelihood-based classification of cryo-EM images using FREALIGN. J. Struct. Biol. 183, 377-388."));
	InfoText->BeginURL("http://dx.doi.org/10.1016/j.jsb.2013.07.005");
	InfoText->BeginUnderline();
	InfoText->BeginTextColour(*wxBLUE);
	InfoText->WriteText(wxT("dio:10.1016/j.jsb.2013.07.005"));
	InfoText->EndURL();
	InfoText->EndTextColour();
	InfoText->EndUnderline();
	InfoText->EndAlignment();
	InfoText->Newline();
	InfoText->Newline();

	InfoText->BeginAlignment(wxTEXT_ALIGNMENT_LEFT);
	InfoText->BeginBold();
	InfoText->WriteText(wxT("Grigorieff, N.,"));
	InfoText->EndBold();
	InfoText->WriteText(wxT(" 2016. Frealign: An exploratory tool for single-particle cryo-EM. Methods Enzymol. 579, 191-226. "));
	InfoText->BeginURL("http://dx.doi.org/10.1016/bs.mie.2016.04.013");
	InfoText->BeginUnderline();
	InfoText->BeginTextColour(*wxBLUE);
	InfoText->WriteText(wxT("doi:10.1016/bs.mie.2016.04.013"));
	InfoText->EndURL();
	InfoText->EndTextColour();
	InfoText->EndUnderline();
	InfoText->EndAlignment();
	InfoText->Newline();
	InfoText->Newline();


}

void MyRefine3DPanel::OnInfoURL(wxTextUrlEvent& event)
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


void MyRefine3DPanel::ResetAllDefaultsClick( wxCommandEvent& event )
{
	// TODO : should probably check that the user hasn't changed the defaults yet in the future
	SetDefaults();
}

void MyRefine3DPanel::SetDefaults()
{
	if (RefinementPackageComboBox->GetCount() > 0)
	{
		float calculated_high_resolution_cutoff;
		float local_mask_radius;
		float global_mask_radius;
		float global_angular_step;
		float search_range;

		ExpertPanel->Freeze();

	// calculate high resolution limit..

		long current_input_refinement_id = refinement_package_asset_panel->all_refinement_packages.Item(RefinementPackageComboBox->GetSelection()).refinement_ids[InputParametersComboBox->GetSelection()];
		calculated_high_resolution_cutoff = 30.0;

		for (int class_counter = 0; class_counter < refinement_package_asset_panel->ReturnPointerToShortRefinementInfoByRefinementID(current_input_refinement_id)->number_of_classes; class_counter++)
		{
		//if (refinement_package_asset_panel->ReturnPointerToRefinementByRefinementID(current_input_refinement_id)->class_refinement_results[class_counter].class_resolution_statistics.Return0p5Resolution() > calculated_high_resolution_cutoff) calculated_high_resolution_cutoff = refinement_package_asset_panel->ReturnPointerToRefinementByRefinementID(current_input_refinement_id)->class_refinement_results[class_counter].class_resolution_statistics.Return0p8Resolution();
		}

		local_mask_radius = refinement_package_asset_panel->all_refinement_packages.Item(RefinementPackageComboBox->GetSelection()).estimated_particle_size_in_angstroms * 0.65;
		global_mask_radius = refinement_package_asset_panel->all_refinement_packages.Item(RefinementPackageComboBox->GetSelection()).estimated_particle_size_in_angstroms * 0.8;

		global_angular_step = CalculateAngularStep(calculated_high_resolution_cutoff, local_mask_radius);

		search_range = refinement_package_asset_panel->all_refinement_packages.Item(RefinementPackageComboBox->GetSelection()).estimated_particle_size_in_angstroms * 0.15;
		// Set the values..

		RefinePhiCheckBox->SetValue(true);
		RefineThetaCheckBox->SetValue(true);
		RefinePsiCheckBox->SetValue(true);
		RefineXShiftCheckBox->SetValue(true);
		RefineYShiftCheckBox->SetValue(true);
		RefineOccupanciesCheckBox->SetValue(true);

		float low_res_limit = refinement_package_asset_panel->all_refinement_packages.Item(RefinementPackageComboBox->GetSelection()).estimated_particle_size_in_angstroms * 1.5;
		if (low_res_limit > 300.00) low_res_limit = 300.00;

		auto_mask_value = true;
		LowResolutionLimitTextCtrl->SetValue(wxString::Format("%.2f", low_res_limit));
		HighResolutionLimitTextCtrl->SetValue(wxString::Format("%.2f", calculated_high_resolution_cutoff));
		MaskRadiusTextCtrl->SetValue(wxString::Format("%.2f", local_mask_radius));
		SignedCCResolutionTextCtrl->SetValue("0.00");

		GlobalMaskRadiusTextCtrl->SetValue(wxString::Format("%.2f", global_mask_radius));
		NumberToRefineSpinCtrl->SetValue(20);
		AngularStepTextCtrl->SetValue(wxString::Format("%.2f", global_angular_step));
		SearchRangeXTextCtrl->SetValue(wxString::Format("%.2f", search_range));
		SearchRangeYTextCtrl->SetValue(wxString::Format("%.2f", search_range));

		ClassificationHighResLimitTextCtrl->SetValue(wxString::Format("%.2f", calculated_high_resolution_cutoff));
		SphereClassificatonYesRadio->SetValue(false);
		SphereClassificatonNoRadio->SetValue(true);

		SphereXTextCtrl->SetValue("0.00");
		SphereYTextCtrl->SetValue("0.00");
		SphereZTextCtrl->SetValue("0.00");
		SphereRadiusTextCtrl->SetValue("0.00");

		RefineCTFYesRadio->SetValue(false);
		RefineCTFNoRadio->SetValue(true);
		DefocusSearchRangeTextCtrl->SetValue("500.00");
		DefocusSearchStepTextCtrl->SetValue("50.00");

		InnerMaskRadiusTextCtrl->SetValue("0.00");
		ScoreToWeightConstantTextCtrl->SetValue("2.00");

		AdjustScoreForDefocusYesRadio->SetValue(true);
		AdjustScoreForDefocusNoRadio->SetValue(false);
		ReconstructionScoreThreshold->SetValue("0.00");
		ReconstructionResolutionLimitTextCtrl->SetValue("0.00");
		AutoCropYesRadioButton->SetValue(false);
		AutoCropNoRadioButton->SetValue(true);

		PercentUsedTextCtrl->SetValue("100.00");
		ApplyBlurringNoRadioButton->SetValue(true);
		ApplyBlurringYesRadioButton->SetValue(false);
		SmoothingFactorTextCtrl->SetValue("1.00");

		AutoCenterYesRadioButton->SetValue(false);
		AutoCenterNoRadioButton->SetValue(true);

		MaskEdgeTextCtrl->ChangeValueFloat(10.00);
		MaskWeightTextCtrl->ChangeValueFloat(0.00);
		LowPassMaskYesRadio->SetValue(false);
		LowPassMaskNoRadio->SetValue(true);
		MaskFilterResolutionText->ChangeValueFloat(20.00);

#ifdef EXPERIMENTAL
		MergeMapModelYesRadioButton->SetValue(false);
		MergeMapModelNoRadioButton->SetValue(true);
		MergeMapModelFmodelResolutionTextCtrl->ChangeValueFloat(2.00);
		MergeMapModelBoundaryResolutionTextCtrl->ChangeValueFloat(8.00);
		MergeMapModelBoundaryWidthTextCtrl->ChangeValueFloat(10);
		MergeMapModelModelFilenameTextCtrl->SetValue("");
#endif

		ExpertPanel->Thaw();
	}

}

void MyRefine3DPanel::OnHighResLimitChange( wxCommandEvent& event )
{
	float global_angular_step = CalculateAngularStep(HighResolutionLimitTextCtrl->ReturnValue(), MaskRadiusTextCtrl->ReturnValue());
	AngularStepTextCtrl->SetValue(wxString::Format("%.2f", global_angular_step));
	ClassificationHighResLimitTextCtrl->SetValue(wxString::Format("%.2f", HighResolutionLimitTextCtrl->ReturnValue()));
}

void MyRefine3DPanel::OnUpdateUI( wxUpdateUIEvent& event )
{
	// are there enough members in the selected group.
	if (main_frame->current_project.is_open == false)
	{
		RefinementPackageComboBox->Enable(false);
		InputParametersComboBox->Enable(false);
		RefinementRunProfileComboBox->Enable(false);
		ReconstructionRunProfileComboBox->Enable(false);
		ExpertToggleButton->Enable(false);
		StartRefinementButton->Enable(false);
		LocalRefinementRadio->Enable(false);
		GlobalRefinementRadio->Enable(false);
		NumberRoundsSpinCtrl->Enable(false);
		UseMaskCheckBox->Enable(false);
		MaskSelectPanel->Enable(false);
		HighResolutionLimitTextCtrl->Enable(false);


		if (ExpertPanel->IsShown() == true)
		{
			ExpertToggleButton->SetValue(false);
			ExpertPanel->Show(false);
			Layout();

		}

		if (RefinementPackageComboBox->GetCount() > 0)
		{
			RefinementPackageComboBox->Clear();
			RefinementPackageComboBox->ChangeValue("");

		}

		if (InputParametersComboBox->GetCount() > 0)
		{
			InputParametersComboBox->Clear();
			InputParametersComboBox->ChangeValue("");
		}

		if (ReconstructionRunProfileComboBox->GetCount() > 0)
		{
			ReconstructionRunProfileComboBox->Clear();
			ReconstructionRunProfileComboBox->ChangeValue("");
		}

		if (RefinementRunProfileComboBox->GetCount() > 0)
		{
			RefinementRunProfileComboBox->Clear();
			RefinementRunProfileComboBox->ChangeValue("");
		}

		if (PleaseCreateRefinementPackageText->IsShown())
		{
			PleaseCreateRefinementPackageText->Show(false);
			Layout();
		}

	}
	else
	{
		RefinementRunProfileComboBox->Enable(true);
		ReconstructionRunProfileComboBox->Enable(true);
		HighResolutionLimitTextCtrl->Enable(true);

		if (running_job == false)
		{
			NoCycleStaticText->Enable(true);
			HiResLimitStaticText->Enable(true);
			LocalRefinementRadio->Enable(true);
			GlobalRefinementRadio->Enable(true);
			NumberRoundsSpinCtrl->Enable(true);
			HighResolutionLimitTextCtrl->Enable(true);
			UseMaskCheckBox->Enable(true);
			ExpertToggleButton->Enable(true);

			if (RefinementPackageComboBox->GetCount() > 0)
			{

				RefinementPackageComboBox->Enable(true);
				InputParametersComboBox->Enable(true);

				if (UseMaskCheckBox->GetValue() == true)
				{
					MaskSelectPanel->Enable(true);
				}
				else
				{
					MaskSelectPanel->Enable(false);
					if (MaskSelectPanel->GetCount() > 0)
					{
						MaskSelectPanel->Clear();
						MaskSelectPanel->AssetComboBox->ChangeValue("");
					}
				}

				if (PleaseCreateRefinementPackageText->IsShown())
				{
					PleaseCreateRefinementPackageText->Show(false);
					Layout();
				}

			}
			else
			{
				UseMaskCheckBox->Enable(false);
				MaskSelectPanel->Enable(false);
				MaskSelectPanel->AssetComboBox->ChangeValue("");
				RefinementPackageComboBox->ChangeValue("");
				RefinementPackageComboBox->Enable(false);
				InputParametersComboBox->ChangeValue("");
				InputParametersComboBox->Enable(false);

				if (PleaseCreateRefinementPackageText->IsShown() == false)
				{
					PleaseCreateRefinementPackageText->Show(true);
					Layout();
				}
			}

			if (ExpertToggleButton->GetValue() == true)
			{
				if (GlobalRefinementRadio->GetValue() == false)
				{
					//GlobalResolutionLimitStaticText->Enable(false);
				//	GlobalResolutionLimitTextCtrl->Enable(false);
					GlobalMaskRadiusStaticText->Enable(false);
					GlobalMaskRadiusTextCtrl->Enable(false);
					NumberToRefineSpinCtrl->Enable(false);
					NumberToRefineStaticText->Enable(false);
					AngularStepTextCtrl->Enable(false);
					AngularStepStaticText->Enable(false);
					SearchRangeXTextCtrl->Enable(false);
					SearchRangeXStaticText->Enable(false);
					SearchRangeYTextCtrl->Enable(false);
					SearchRangeYStaticText->Enable(false);
					AlsoRefineInputStaticText1->Enable(false);
					AlsoRefineInputYesRadio->Enable(false);
					AlsoRefineInputNoRadio->Enable(false);
				}
				else
				{
				//	GlobalResolutionLimitStaticText->Enable(true);
				//	GlobalResolutionLimitTextCtrl->Enable(true);
					GlobalMaskRadiusStaticText->Enable(true);
					GlobalMaskRadiusTextCtrl->Enable(true);
					NumberToRefineSpinCtrl->Enable(true);
					NumberToRefineStaticText->Enable(true);
					AngularStepTextCtrl->Enable(true);
					AngularStepStaticText->Enable(true);
					SearchRangeXTextCtrl->Enable(true);
					SearchRangeXStaticText->Enable(true);
					SearchRangeYTextCtrl->Enable(true);
					SearchRangeYStaticText->Enable(true);
					AlsoRefineInputStaticText1->Enable(true);
					AlsoRefineInputYesRadio->Enable(true);
					AlsoRefineInputNoRadio->Enable(true);

				}

				if (SphereClassificatonYesRadio->GetValue() == false)
				{
					SphereXTextCtrl->Enable(false);
					SphereXStaticText->Enable(false);
					SphereYTextCtrl->Enable(false);
					SphereYStaticText->Enable(false);
					SphereZTextCtrl->Enable(false);
					SphereZStaticText->Enable(false);
					SphereRadiusTextCtrl->Enable(false);
					SphereRadiusStaticText->Enable(false);

				}
				else
				{
					SphereXTextCtrl->Enable(true);
					SphereXStaticText->Enable(true);
					SphereYTextCtrl->Enable(true);
					SphereYStaticText->Enable(true);
					SphereZTextCtrl->Enable(true);
					SphereZStaticText->Enable(true);
					SphereRadiusTextCtrl->Enable(true);
					SphereRadiusStaticText->Enable(true);
				}

				if (RefineCTFYesRadio->GetValue() == false)
				{
					DefocusSearchRangeStaticText->Enable(false);
					DefocusSearchRangeTextCtrl->Enable(false);
					DefocusSearchStepTextCtrl->Enable(false);
					DefocusSearchStepStaticText->Enable(false);
				}
				else
				{
					DefocusSearchRangeStaticText->Enable(true);
					DefocusSearchRangeTextCtrl->Enable(true);
					DefocusSearchStepTextCtrl->Enable(true);
					DefocusSearchStepStaticText->Enable(true);
				}

				if (ApplyBlurringYesRadioButton->GetValue() == true)
				{
					SmoothingFactorTextCtrl->Enable(true);
					SmoothingFactorStaticText->Enable(true);
				}
				else
				{
					SmoothingFactorTextCtrl->Enable(false);
					SmoothingFactorStaticText->Enable(false);
				}

				if (UseMaskCheckBox->GetValue() == false)
				{
					MaskEdgeStaticText->Enable(false);
					MaskEdgeTextCtrl->Enable(false);
					MaskWeightStaticText->Enable(false);
					MaskWeightTextCtrl->Enable(false);
					LowPassYesNoStaticText->Enable(false);
					LowPassMaskYesRadio->Enable(false);
					LowPassMaskNoRadio->Enable(false);
					FilterResolutionStaticText->Enable(false);
					MaskFilterResolutionText->Enable(false);

					AutoCenterYesRadioButton->Enable(true);
					AutoCenterNoRadioButton->Enable(true);
					AutoCenterStaticText->Enable(true);

					AutoMaskStaticText->Enable(true);
					AutoMaskYesRadioButton->Enable(true);
					AutoMaskNoRadioButton->Enable(true);

					if (AutoMaskYesRadioButton->GetValue() != auto_mask_value)
					{
						if (auto_mask_value == true) AutoMaskYesRadioButton->SetValue(true);
						else AutoMaskNoRadioButton->SetValue(true);
					}
				}
				else
				{
					AutoCenterYesRadioButton->Enable(false);
					AutoCenterNoRadioButton->Enable(false);
					AutoCenterStaticText->Enable(false);

					AutoMaskStaticText->Enable(false);
					AutoMaskYesRadioButton->Enable(false);
					AutoMaskNoRadioButton->Enable(false);

					if (AutoMaskYesRadioButton->GetValue() != false)
					{
						AutoMaskNoRadioButton->SetValue(true);
					}

					MaskEdgeStaticText->Enable(true);
					MaskEdgeTextCtrl->Enable(true);
					MaskWeightStaticText->Enable(true);
					MaskWeightTextCtrl->Enable(true);
					LowPassYesNoStaticText->Enable(true);
					LowPassMaskYesRadio->Enable(true);
					LowPassMaskNoRadio->Enable(true);

					if (LowPassMaskYesRadio->GetValue() == true)
					{
						FilterResolutionStaticText->Enable(true);
						MaskFilterResolutionText->Enable(true);
					}
					else
					{
						FilterResolutionStaticText->Enable(false);
						MaskFilterResolutionText->Enable(false);
					}




				}



			}

			bool estimation_button_status = false;

			if (RefinementPackageComboBox->GetCount() > 0 && ReconstructionRunProfileComboBox->GetCount() > 0)
			{
				if (run_profiles_panel->run_profile_manager.ReturnTotalJobs(RefinementRunProfileComboBox->GetSelection()) > 1 && run_profiles_panel->run_profile_manager.ReturnTotalJobs(ReconstructionRunProfileComboBox->GetSelection()) > 1)
				{
					if (RefinementPackageComboBox->GetSelection() != wxNOT_FOUND && InputParametersComboBox->GetSelection() != wxNOT_FOUND)
					{
						if (UseMaskCheckBox->GetValue() == false || MaskSelectPanel->AssetComboBox->GetSelection() != wxNOT_FOUND)
						estimation_button_status = true;
					}

				}
			}

			StartRefinementButton->Enable(estimation_button_status);

			if (refinement_package_combo_is_dirty == true)
			{
				FillRefinementPackagesComboBox();
				refinement_package_combo_is_dirty = false;
			}

			if (run_profiles_are_dirty == true)
			{
				FillRunProfileComboBoxes();
				run_profiles_are_dirty = false;
			}

			if (input_params_combo_is_dirty == true)
			{
				FillInputParamsComboBox();
				input_params_combo_is_dirty = false;
			}

			if (volumes_are_dirty == true)
			{
				MaskSelectPanel->FillComboBox();
				volumes_are_dirty = false;
			}
		}
		else
		{
			NoCycleStaticText->Enable(false);
			HiResLimitStaticText->Enable(false);
			RefinementPackageComboBox->Enable(false);
			InputParametersComboBox->Enable(false);
			LocalRefinementRadio->Enable(false);
			ExpertToggleButton->Enable(false);
			GlobalRefinementRadio->Enable(false);
			NumberRoundsSpinCtrl->Enable(false);
			HighResolutionLimitTextCtrl->Enable(false);
			UseMaskCheckBox->Enable(false);
			MaskSelectPanel->Enable(false);

		}
	}

}

void MyRefine3DPanel::OnAutoMaskButton( wxCommandEvent& event )
{
	auto_mask_value = AutoMaskYesRadioButton->GetValue();
}

void MyRefine3DPanel::OnUseMaskCheckBox( wxCommandEvent& event )
{
	if (UseMaskCheckBox->GetValue() == true)
	{
		auto_mask_value = AutoMaskYesRadioButton->GetValue();
		MaskSelectPanel->FillComboBox();
	}
	else
	{
		if (auto_mask_value == true) AutoMaskYesRadioButton->SetValue(true);
		else AutoMaskNoRadioButton->SetValue(true);
	}
	AutoCenterYesRadioButton->SetValue(false);
	AutoCenterNoRadioButton->SetValue(true);
}

void MyRefine3DPanel::OnExpertOptionsToggle( wxCommandEvent& event )
{

	if (ExpertToggleButton->GetValue() == true)
	{
		ExpertPanel->Show(true);
		Layout();
	}
	else
	{
		ExpertPanel->Show(false);
		Layout();
	}
}

#ifdef EXPERIMENTAL
void MyRefine3DPanel::OnMergeMapModelYesRadioButton( wxCommandEvent& event )
{
	MergeMapModelFmodelResolutionStaticText->Enable(true);
	MergeMapModelFmodelResolutionTextCtrl->Enable(true);
	MergeMapModelBoundaryResolutionStaticText->Enable(true);
	MergeMapModelBoundaryResolutionTextCtrl->Enable(true);
	MergeMapModelBoundaryWidthStaticText->Enable(true);
	MergeMapModelBoundaryWidthTextCtrl->Enable(true);
	MergeMapModelModelFilenameStaticText->Enable(true);
	MergeMapModelModelFilenameTextCtrl->Enable(true);
	MergeMapModelModelFilenameBrowseButton->Enable(true);
}

void MyRefine3DPanel::OnMergeMapModelNoRadioButton( wxCommandEvent& event )
{
	MergeMapModelFmodelResolutionStaticText->Enable(false);
	MergeMapModelFmodelResolutionTextCtrl->Enable(false);
	MergeMapModelBoundaryResolutionStaticText->Enable(false);
	MergeMapModelBoundaryResolutionTextCtrl->Enable(false);
	MergeMapModelBoundaryWidthStaticText->Enable(false);
	MergeMapModelBoundaryWidthTextCtrl->Enable(false);
	MergeMapModelModelFilenameStaticText->Enable(false);
	MergeMapModelModelFilenameTextCtrl->Enable(false);
	MergeMapModelModelFilenameBrowseButton->Enable(false);
}

void MyRefine3DPanel::OnModelFileBrowseButtonClick( wxCommandEvent& event )
{
	wxFileDialog openFileDialog(this, _("Select atomic model"), "", "", "PDB, PDBx, mmcif|*.pdb;*.cif", wxFD_OPEN|wxFD_FILE_MUST_EXIST);

	if (openFileDialog.ShowModal() == wxID_OK)
	{
		MergeMapModelModelFilenameTextCtrl->SetValue(openFileDialog.GetPath());
	}
}
#endif

void MyRefine3DPanel::ReDrawActiveReferences()
{
	Active3DReferencesListCtrl->ClearAll();

	if (RefinementPackageComboBox->GetSelection() >= 0)
	{
		Active3DReferencesListCtrl->InsertColumn(0, "Class No.", wxLIST_FORMAT_LEFT);
		Active3DReferencesListCtrl->InsertColumn(1, "Active Reference Volume", wxLIST_FORMAT_LEFT);

		Active3DReferencesListCtrl->SetItemCount(refinement_package_asset_panel->all_refinement_packages.Item(RefinementPackageComboBox->GetSelection()).references_for_next_refinement.GetCount());

		if (refinement_package_asset_panel->all_refinement_packages.Item(RefinementPackageComboBox->GetSelection()).references_for_next_refinement.GetCount() > 0)
		{
			Active3DReferencesListCtrl->RefreshItems(0, refinement_package_asset_panel->all_refinement_packages.Item(RefinementPackageComboBox->GetSelection()).references_for_next_refinement.GetCount() -1);

			Active3DReferencesListCtrl->SetColumnWidth(0, Active3DReferencesListCtrl->ReturnGuessAtColumnTextWidth(0));
			Active3DReferencesListCtrl->SetColumnWidth(1, Active3DReferencesListCtrl->ReturnGuessAtColumnTextWidth(1));
		}
	}
}

void MyRefine3DPanel::FillRefinementPackagesComboBox()
{
	if (RefinementPackageComboBox->FillComboBox() == false) NewRefinementPackageSelected();
}

void MyRefine3DPanel::FillInputParamsComboBox()
{
	if (RefinementPackageComboBox->GetCount() > 0 ) InputParametersComboBox->FillComboBox(RefinementPackageComboBox->GetSelection(), true);
}

void MyRefine3DPanel::NewRefinementPackageSelected()
{
	selected_refinement_package = RefinementPackageComboBox->GetSelection();
	FillInputParamsComboBox();
	SetDefaults();
	ReDrawActiveReferences();
	//wxPrintf("New Refinement Package Selection\n");

}

void MyRefine3DPanel::OnRefinementPackageComboBox( wxCommandEvent& event )
{

	NewRefinementPackageSelected();

}

void MyRefine3DPanel::OnInputParametersComboBox( wxCommandEvent& event )
{
	//SetDefaults();
}

void MyRefine3DPanel::TerminateButtonClick( wxCommandEvent& event )
{
	main_frame->job_controller.KillJob(my_job_id);

	active_mask_thread_id = -1;
	active_orth_thread_id = -1;

	WriteBlueText("Terminated Job");
	TimeRemainingText->SetLabel("Time Remaining : Terminated");
	CancelAlignmentButton->Show(false);
	FinishButton->Show(true);
	ProgressPanel->Layout();
/*
	if (buffered_results != NULL)
	{
		delete [] buffered_results;
		buffered_results = NULL;
	}*/
}

void MyRefine3DPanel::OnVolumeListItemActivated( wxListEvent& event )
{
	MyVolumeChooserDialog *dialog = new MyVolumeChooserDialog(this);

	dialog->ComboBox->SetSelection(volume_asset_panel->ReturnArrayPositionFromAssetID(refinement_package_asset_panel->all_refinement_packages.Item(RefinementPackageComboBox->GetSelection()).references_for_next_refinement.Item(event.GetIndex())) + 1);
	dialog->Fit();
	if (dialog->ShowModal() == wxID_OK)
	{
		if (dialog->selected_volume_id != refinement_package_asset_panel->all_refinement_packages.Item(RefinementPackageComboBox->GetSelection()).references_for_next_refinement.Item(event.GetIndex()))
		{
			refinement_package_asset_panel->all_refinement_packages.Item(RefinementPackageComboBox->GetSelection()).references_for_next_refinement.Item(event.GetIndex()) = dialog->selected_volume_id;
					// Change in database..
			main_frame->current_project.database.ExecuteSQL(wxString::Format("UPDATE REFINEMENT_PACKAGE_CURRENT_REFERENCES_%li SET VOLUME_ASSET_ID=%li WHERE CLASS_NUMBER=%li;", refinement_package_asset_panel->all_refinement_packages.Item(RefinementPackageComboBox->GetSelection()).asset_id, dialog->selected_volume_id, event.GetIndex() + 1));

			ReDrawActiveReferences();
		}
	}
	dialog->Destroy();
}

void MyRefine3DPanel::FinishButtonClick( wxCommandEvent& event )
{
	ProgressBar->SetValue(0);
	TimeRemainingText->SetLabel("Time Remaining : ???h:??m:??s");
    CancelAlignmentButton->Show(true);
	FinishButton->Show(false);

	InputParamsPanel->Show(true);
	ProgressPanel->Show(false);
	StartPanel->Show(true);
	OutputTextPanel->Show(false);
	output_textctrl->Clear();
	ShowRefinementResultsPanel->Show(false);
	ShowRefinementResultsPanel->Clear();
	//CTFResultsPanel->Show(false);
	//graph_is_hidden = true;
	InfoPanel->Show(true);

	if (my_refinement_manager.output_refinement != NULL) delete my_refinement_manager.output_refinement;

	if (ExpertToggleButton->GetValue() == true) ExpertPanel->Show(true);
	else ExpertPanel->Show(false);
	running_job = false;
	Layout();

	//CTFResultsPanel->CTF2DResultsPanel->should_show = false;
	//CTFResultsPanel->CTF2DResultsPanel->Refresh();

}




void MyRefine3DPanel::StartRefinementClick( wxCommandEvent& event )
{
#ifdef EXPERIMENTAL
	if (my_refinement_manager.my_parent->MergeMapModelYesRadioButton->GetValue() == true)
	{
		// Validate paths supplied to merge_map_model before launching
		MyErrorDialog *my_error = new MyErrorDialog(this);
		bool have_errors = false;
		OneSecondProgressDialog *my_progress_dialog = new OneSecondProgressDialog("Validate Parameters", "Validating Parameters...", 2, this, wxPD_AUTO_HIDE|wxPD_APP_MODAL|wxPD_REMAINING_TIME);
		if (! wxDirExists(phenix_settings_panel->buffer_phenix_path))
		{
			my_error->ErrorText->AppendText(wxString::Format(wxT("Can't read this directory for Phenix executables:%s\n"), (phenix_settings_panel->buffer_phenix_path).ToStdString()));
			have_errors = true;
		}
		my_progress_dialog->Update(1);
		if (! wxFileExists(my_refinement_manager.my_parent->MergeMapModelModelFilenameTextCtrl->GetValue()))
		{
			my_error->ErrorText->AppendText(wxString::Format(wxT("Can't read the model: %s\n"), (my_refinement_manager.my_parent->MergeMapModelModelFilenameTextCtrl->GetValue()).ToStdString()));
			have_errors = true;
		}
		my_progress_dialog->Destroy();
		if (have_errors == true)
		{
			my_error->ShowModal();
		}
		my_error->Destroy();
		if (have_errors == false)
		{
			my_refinement_manager.BeginRefinementCycle();
		}
	}
#endif
}

void MyRefine3DPanel::WriteInfoText(wxString text_to_write)
{
	output_textctrl->SetDefaultStyle(wxTextAttr(*wxBLACK));
	output_textctrl->AppendText(text_to_write);

	if (text_to_write.EndsWith("\n") == false)	 output_textctrl->AppendText("\n");
}

void MyRefine3DPanel::WriteBlueText(wxString text_to_write)
{
	output_textctrl->SetDefaultStyle(wxTextAttr(*wxBLUE));
	output_textctrl->AppendText(text_to_write);

	if (text_to_write.EndsWith("\n") == false)	 output_textctrl->AppendText("\n");
}

void MyRefine3DPanel::WriteErrorText(wxString text_to_write)
{
	 output_textctrl->SetDefaultStyle(wxTextAttr(*wxRED));
	 output_textctrl->AppendText(text_to_write);

	 if (text_to_write.EndsWith("\n") == false)	 output_textctrl->AppendText("\n");
}


void MyRefine3DPanel::FillRunProfileComboBoxes()
{
	ReconstructionRunProfileComboBox->FillWithRunProfiles();
	RefinementRunProfileComboBox->FillWithRunProfiles();
}
void MyRefine3DPanel::OnSocketJobResultMsg(JobResult &received_result)
{
	my_refinement_manager.ProcessJobResult(&received_result);


}

void MyRefine3DPanel::OnSocketJobResultQueueMsg(ArrayofJobResults &received_queue)
{
	for (int counter = 0; counter < received_queue.GetCount(); counter++)
	{
		my_refinement_manager.ProcessJobResult(&received_queue.Item(counter));
	}

}

void MyRefine3DPanel::SetNumberConnectedText(wxString wanted_text)
{
	NumberConnectedText->SetLabel(wanted_text);
}

void MyRefine3DPanel::SetTimeRemainingText(wxString wanted_text)
{
	TimeRemainingText->SetLabel(wanted_text);
}

void MyRefine3DPanel::OnSocketAllJobsFinished()
{
	my_refinement_manager.ProcessAllJobsFinished();
}

RefinementManager::RefinementManager()
{
	input_refinement = NULL;
	output_refinement = NULL;

}

void RefinementManager::SetParent(MyRefine3DPanel *wanted_parent)
{
	my_parent = wanted_parent;
}

void RefinementManager::BeginRefinementCycle()
{
	start_with_reconstruction = false;

	active_low_resolution_limit = my_parent->LowResolutionLimitTextCtrl->ReturnValue();
	active_high_resolution_limit = my_parent->HighResolutionLimitTextCtrl->ReturnValue();
	active_mask_radius = my_parent->MaskRadiusTextCtrl->ReturnValue();
	active_signed_cc_limit = my_parent->SignedCCResolutionTextCtrl->ReturnValue();
	active_global_mask_radius = my_parent->GlobalMaskRadiusTextCtrl->ReturnValue();
	active_number_results_to_refine = my_parent->NumberToRefineSpinCtrl->GetValue();
	active_angular_search_step = my_parent->AngularStepTextCtrl->ReturnValue();
	active_search_range_x = my_parent->SearchRangeXTextCtrl->ReturnValue();
	active_search_range_y = my_parent->SearchRangeYTextCtrl->ReturnValue();
	active_classification_high_res_limit = my_parent->ClassificationHighResLimitTextCtrl->ReturnValue();
	active_should_focus_classify = my_parent->SphereClassificatonYesRadio->GetValue();
	active_sphere_x_coord = my_parent->SphereXTextCtrl->ReturnValue();
	active_sphere_y_coord = my_parent->SphereYTextCtrl->ReturnValue();
	active_sphere_z_coord = my_parent->SphereZTextCtrl->ReturnValue();
	active_should_refine_ctf = my_parent->RefineCTFYesRadio->GetValue();
	active_defocus_search_range = my_parent->DefocusSearchRangeTextCtrl->ReturnValue();
	active_defocus_search_step = my_parent->DefocusSearchStepTextCtrl->ReturnValue();
	active_percent_used = my_parent->PercentUsedTextCtrl->ReturnValue();
	active_inner_mask_radius = my_parent->InnerMaskRadiusTextCtrl->ReturnValue();
	active_resolution_limit_rec = my_parent->ReconstructionResolutionLimitTextCtrl->ReturnValue();
	active_score_weight_conversion	= my_parent->ScoreToWeightConstantTextCtrl->ReturnValue();
	active_score_threshold	= my_parent->ReconstructionScoreThreshold->ReturnValue();
	active_adjust_scores = my_parent->AdjustScoreForDefocusYesRadio->GetValue();
	active_crop_images	= my_parent->AutoCropYesRadioButton->GetValue();
	active_should_apply_blurring = my_parent->ApplyBlurringYesRadioButton->GetValue();
	active_smoothing_factor = my_parent->SmoothingFactorTextCtrl->ReturnValue();
	active_sphere_radius = my_parent->SphereRadiusTextCtrl->ReturnValue();
	active_do_global_refinement = my_parent->GlobalRefinementRadio->GetValue();
	active_also_refine_input = my_parent->AlsoRefineInputNoRadio->GetValue();
	active_should_refine_psi = my_parent->RefinePsiCheckBox->GetValue();
	active_should_refine_theta = my_parent->RefineThetaCheckBox->GetValue();
	active_should_refine_phi = my_parent->RefinePhiCheckBox->GetValue();
	active_should_refine_x_shift = my_parent->RefineXShiftCheckBox->GetValue();
	active_should_refine_y_shift = my_parent->RefineYShiftCheckBox->GetValue();
	active_should_mask = my_parent->UseMaskCheckBox->GetValue();
	active_should_auto_mask = my_parent->AutoMaskYesRadioButton->GetValue();
	active_centre_mass = my_parent->AutoCenterYesRadioButton->GetValue();
#ifdef EXPERIMENTAL
	active_should_merge_map_model = my_parent->MergeMapModelYesRadioButton->GetValue();
#endif

	if (my_parent->MaskSelectPanel->ReturnSelection() >= 0) active_mask_asset_id = volume_asset_panel->ReturnAssetID(my_parent->MaskSelectPanel->ReturnSelection());
	else active_mask_asset_id = -1;
	if (my_parent->MaskSelectPanel->ReturnSelection() >= 0)	active_mask_filename = volume_asset_panel->ReturnAssetLongFilename(my_parent->MaskSelectPanel->ReturnSelection());
	else active_mask_filename = "";

#ifdef EXPERIMENTAL
	if (active_should_merge_map_model) active_fmodel_model_filename = my_parent->MergeMapModelModelFilenameTextCtrl->GetValue();
	else active_fmodel_model_filename = "";
	if (active_should_merge_map_model) active_fmodel_resolution = my_parent->MergeMapModelFmodelResolutionTextCtrl->ReturnValue();
	else active_fmodel_resolution = 0;
	if (active_should_merge_map_model) active_boundary_resolution = my_parent->MergeMapModelBoundaryResolutionTextCtrl->ReturnValue();
	else active_boundary_resolution = 0;
#endif

	active_should_low_pass_filter_mask = my_parent->LowPassMaskYesRadio->GetValue();
	active_mask_filter_resolution = my_parent->MaskFilterResolutionText->ReturnValue();
	active_mask_edge = my_parent->MaskEdgeTextCtrl->ReturnValue();
	active_mask_weight = my_parent->MaskWeightTextCtrl->ReturnValue();

	active_refinement_run_profile = run_profiles_panel->run_profile_manager.run_profiles[my_parent->RefinementRunProfileComboBox->GetSelection()];
	active_reconstruction_run_profile = run_profiles_panel->run_profile_manager.run_profiles[my_parent->ReconstructionRunProfileComboBox->GetSelection()];

	number_of_rounds_run = 0;
	number_of_rounds_to_run = my_parent->NumberRoundsSpinCtrl->GetValue();

	active_refinement_package = &refinement_package_asset_panel->all_refinement_packages.Item(my_parent->RefinementPackageComboBox->GetSelection());

	current_refinement_package_asset_id = active_refinement_package->asset_id;
	current_input_refinement_id = active_refinement_package->refinement_ids[my_parent->InputParametersComboBox->GetSelection()];

	int class_counter;
	int number_of_classes = active_refinement_package->number_of_classes;

	wxString blank_string = "";
	current_reference_filenames.Clear();
	current_reference_filenames.Add(blank_string, number_of_classes);

	current_reference_asset_ids.Clear();
	current_reference_asset_ids.Add(-1, number_of_classes);

	// check scratch directory.
	global_delete_refine3d_scratch();

	// get the data..

	for (class_counter = 0; class_counter < active_refinement_package->number_of_classes; class_counter++)
	{
		if (active_refinement_package->references_for_next_refinement[class_counter] == -1) start_with_reconstruction = true;
	}

	my_parent->Freeze();

	my_parent->InputParamsPanel->Show(false);
	my_parent->StartPanel->Show(false);
	my_parent->ProgressPanel->Show(true);
	my_parent->ExpertPanel->Show(false);
	my_parent->InfoPanel->Show(false);
	my_parent->OutputTextPanel->Show(true);
	my_parent->ShowRefinementResultsPanel->Clear();

	if (my_parent->ShowRefinementResultsPanel->LeftRightSplitter->IsSplit() == true) my_parent->ShowRefinementResultsPanel->LeftRightSplitter->Unsplit();
	if (my_parent->ShowRefinementResultsPanel->TopBottomSplitter->IsSplit() == true) my_parent->ShowRefinementResultsPanel->TopBottomSplitter->Unsplit();
	my_parent->ShowRefinementResultsPanel->Show(true);
	my_parent->Layout();

	my_parent->Thaw();

	if (start_with_reconstruction == true)
	{
		input_refinement = main_frame->current_project.database.GetRefinementByID(current_input_refinement_id);
		output_refinement = input_refinement;
		current_output_refinement_id = input_refinement->refinement_id;

		// after this job, the resolution statistics will be real, so update..

		output_refinement->resolution_statistics_are_generated = false;

		SetupReconstructionJob();
		RunReconstructionJob();
	}
	else
	{
		input_refinement = main_frame->current_project.database.GetRefinementByID(current_input_refinement_id);
		output_refinement = new Refinement;

		// we need to set the currently selected reference filenames..

		for (class_counter = 0; class_counter < number_of_classes; class_counter++)
		{
			if (volume_asset_panel->ReturnAssetPointer(volume_asset_panel->ReturnArrayPositionFromAssetID(active_refinement_package->references_for_next_refinement[class_counter]))->x_size != active_refinement_package->stack_box_size ||
				volume_asset_panel->ReturnAssetPointer(volume_asset_panel->ReturnArrayPositionFromAssetID(active_refinement_package->references_for_next_refinement[class_counter]))->y_size != active_refinement_package->stack_box_size ||
				volume_asset_panel->ReturnAssetPointer(volume_asset_panel->ReturnArrayPositionFromAssetID(active_refinement_package->references_for_next_refinement[class_counter]))->z_size != active_refinement_package->stack_box_size ||
				fabsf(volume_asset_panel->ReturnAssetPointer(volume_asset_panel->ReturnArrayPositionFromAssetID(active_refinement_package->references_for_next_refinement[class_counter]))->pixel_size - input_refinement->resolution_statistics_pixel_size) > 0.01f)
			{
				my_parent->WriteErrorText("Error: Reference volume has different dimensions / pixel size from the input stack.  This will currently not work.");
			}

			current_reference_filenames.Item(class_counter) = volume_asset_panel->ReturnAssetLongFilename(volume_asset_panel->ReturnArrayPositionFromAssetID(active_refinement_package->references_for_next_refinement[class_counter]));
			current_reference_asset_ids.Item(class_counter) = volume_asset_panel->ReturnAssetID(volume_asset_panel->ReturnArrayPositionFromAssetID(active_refinement_package->references_for_next_refinement[class_counter]));
		}

#ifdef EXPERIMENTAL
		if (active_should_merge_map_model == true) // TODO: read current boolean value from the GUI
		{
			my_parent->WriteBlueText("Merging the last reference volume with a map calculated from the provided model...\n");
			MergeMapModel();
		}
#endif
		if (my_parent->UseMaskCheckBox->GetValue() == true || my_parent->AutoMaskYesRadioButton->GetValue() == true)
		{
			DoMasking();
		}
		else
		{
			SetupRefinementJob();
			RunRefinementJob();
		}
	}
}


void RefinementManager::RunRefinementJob()
{
	running_job_type = REFINEMENT;
	number_of_received_particle_results = 0;
	number_of_expected_results = input_refinement->number_of_particles * input_refinement->number_of_classes;

	output_refinement->SizeAndFillWithEmpty(input_refinement->number_of_particles, input_refinement->number_of_classes);
	//wxPrintf("Output refinement has %li particles and %i classes\n", output_refinement->number_of_particles, input_refinement->number_of_classes);
	current_output_refinement_id = main_frame->current_project.database.ReturnHighestRefinementID() + 1;

	output_refinement->refinement_id = current_output_refinement_id;
	output_refinement->refinement_package_asset_id = current_refinement_package_asset_id;

	if (active_do_global_refinement == true)
	{
		output_refinement->name = wxString::Format("Global Search #%li", current_output_refinement_id);
	}
	else output_refinement->name = wxString::Format("Local Refinement #%li", current_output_refinement_id);

	output_refinement->resolution_statistics_are_generated = false;
	output_refinement->datetime_of_run = wxDateTime::Now();
	output_refinement->starting_refinement_id = current_input_refinement_id;

	for (int class_counter = 0; class_counter < active_refinement_package->number_of_classes; class_counter++)
	{
		output_refinement->class_refinement_results[class_counter].low_resolution_limit = active_low_resolution_limit;
		output_refinement->class_refinement_results[class_counter].high_resolution_limit = active_high_resolution_limit;
		output_refinement->class_refinement_results[class_counter].mask_radius = active_mask_radius;
		output_refinement->class_refinement_results[class_counter].signed_cc_resolution_limit = active_signed_cc_limit;
		output_refinement->class_refinement_results[class_counter].global_resolution_limit = active_high_resolution_limit;
		output_refinement->class_refinement_results[class_counter].global_mask_radius = active_global_mask_radius;
		output_refinement->class_refinement_results[class_counter].number_results_to_refine = active_number_results_to_refine;
		output_refinement->class_refinement_results[class_counter].angular_search_step = active_angular_search_step;
		output_refinement->class_refinement_results[class_counter].search_range_x = active_search_range_x;
		output_refinement->class_refinement_results[class_counter].search_range_y = active_search_range_y;
		output_refinement->class_refinement_results[class_counter].classification_resolution_limit = active_classification_high_res_limit;
		output_refinement->class_refinement_results[class_counter].should_focus_classify = active_should_focus_classify;
		output_refinement->class_refinement_results[class_counter].sphere_x_coord = active_sphere_x_coord;
		output_refinement->class_refinement_results[class_counter].sphere_y_coord = active_sphere_y_coord;
		output_refinement->class_refinement_results[class_counter].sphere_z_coord = active_sphere_z_coord;
		output_refinement->class_refinement_results[class_counter].sphere_radius = active_sphere_radius;
		output_refinement->class_refinement_results[class_counter].should_refine_ctf = active_should_refine_ctf;
		output_refinement->class_refinement_results[class_counter].defocus_search_range = active_defocus_search_range;
		output_refinement->class_refinement_results[class_counter].defocus_search_step = active_defocus_search_step;

		output_refinement->class_refinement_results[class_counter].should_auto_mask = active_should_auto_mask;
		output_refinement->class_refinement_results[class_counter].should_refine_input_params = active_also_refine_input;
		output_refinement->class_refinement_results[class_counter].should_use_supplied_mask = active_should_mask;
		output_refinement->class_refinement_results[class_counter].mask_asset_id = active_mask_asset_id;
		output_refinement->class_refinement_results[class_counter].mask_edge_width = active_mask_edge;
		output_refinement->class_refinement_results[class_counter].outside_mask_weight = active_mask_weight;
		output_refinement->class_refinement_results[class_counter].should_low_pass_filter_mask = active_should_low_pass_filter_mask;
		output_refinement->class_refinement_results[class_counter].filter_resolution = active_mask_filter_resolution;
	}

	output_refinement->percent_used = active_percent_used;

	output_refinement->resolution_statistics_box_size = input_refinement->resolution_statistics_box_size;
	output_refinement->resolution_statistics_pixel_size = input_refinement->resolution_statistics_pixel_size;

	// launch a controller

	current_job_starttime = time(NULL);
	time_of_last_update = current_job_starttime;
	my_parent->ShowRefinementResultsPanel->AngularPlotPanel->Clear();

	my_parent->WriteBlueText(wxString::Format("Running refinement round %2i of %2i\n", number_of_rounds_run + 1, number_of_rounds_to_run));
	current_job_id = main_frame->job_controller.AddJob(my_parent, active_refinement_run_profile.manager_command, active_refinement_run_profile.gui_address);
	my_parent->my_job_id = current_job_id;

	if (current_job_id != -1)
	{
		long number_of_refinement_processes;
	    if (my_parent->current_job_package.number_of_jobs + 1 < my_parent->current_job_package.my_profile.ReturnTotalJobs()) number_of_refinement_processes = my_parent->current_job_package.number_of_jobs + 1;
	    else number_of_refinement_processes =  my_parent->current_job_package.my_profile.ReturnTotalJobs();

		if (number_of_refinement_processes >= 100000) my_parent->length_of_process_number = 6;
		else
		if (number_of_refinement_processes >= 10000) my_parent->length_of_process_number = 5;
		else
		if (number_of_refinement_processes >= 1000) my_parent->length_of_process_number = 4;
		else
		if (number_of_refinement_processes >= 100) my_parent->length_of_process_number = 3;
		else
		if (number_of_refinement_processes >= 10) my_parent->length_of_process_number = 2;
		else
		my_parent->length_of_process_number = 1;

		if (my_parent->length_of_process_number == 6) my_parent->NumberConnectedText->SetLabel(wxString::Format("%6i / %6li processes connected.", 0, number_of_refinement_processes));
		else
		if (my_parent->length_of_process_number == 5) my_parent->NumberConnectedText->SetLabel(wxString::Format("%5i / %5li processes connected.", 0, number_of_refinement_processes));
		else
		if (my_parent->length_of_process_number == 4) my_parent->NumberConnectedText->SetLabel(wxString::Format("%4i / %4li processes connected.", 0, number_of_refinement_processes));
		else
		if (my_parent->length_of_process_number == 3) my_parent->NumberConnectedText->SetLabel(wxString::Format("%3i / %3li processes connected.", 0, number_of_refinement_processes));
		else
		if (my_parent->length_of_process_number == 2) my_parent->NumberConnectedText->SetLabel(wxString::Format("%2i / %2li processes connected.", 0, number_of_refinement_processes));
		else

		my_parent->NumberConnectedText->SetLabel(wxString::Format("%i / %li processes connected.", 0, number_of_refinement_processes));
		my_parent->TimeRemainingText->SetLabel("Time Remaining : ???h:??m:??s");
		my_parent->Layout();
		my_parent->running_job = true;
		my_parent->my_job_tracker.StartTracking(my_parent->current_job_package.number_of_jobs);

	}




	my_parent->ProgressBar->Pulse();
}

void RefinementManager::SetupMerge3dJob()
{

	int number_of_reconstruction_jobs = active_reconstruction_run_profile.ReturnTotalJobs() - 1;

	int class_counter;

	my_parent->current_job_package.Reset(active_reconstruction_run_profile, "merge3d", active_refinement_package->number_of_classes);

	for (class_counter = 0; class_counter < active_refinement_package->number_of_classes; class_counter++)
	{
		wxString output_reconstruction_1			= "/dev/null";
		wxString output_reconstruction_2			= "/dev/null";
		wxString output_reconstruction_filtered		= main_frame->current_project.volume_asset_directory.GetFullPath() + wxString::Format("/volume_%li_%i.mrc", output_refinement->refinement_id, class_counter + 1);

		current_reference_filenames.Item(class_counter) = output_reconstruction_filtered;

		wxString output_resolution_statistics		= "/dev/null";
		float 	 molecular_mass_kDa					= active_refinement_package->estimated_particle_weight_in_kda;
		float    inner_mask_radius					= active_inner_mask_radius;
		float    outer_mask_radius					= active_mask_radius;
		wxString dump_file_seed_1 					= main_frame->ReturnRefine3DScratchDirectory() + wxString::Format("dump_file_%li_%i_odd_.dump", current_output_refinement_id, class_counter);
		wxString dump_file_seed_2 					= main_frame->ReturnRefine3DScratchDirectory() + wxString::Format("dump_file_%li_%i_even_.dump", current_output_refinement_id, class_counter);

		bool save_orthogonal_views_image = true;
		wxString orthogonal_views_filename = main_frame->current_project.volume_asset_directory.GetFullPath() + wxString::Format("/OrthViews/volume_%li_%i.mrc", output_refinement->refinement_id, class_counter + 1);
		float weiner_nominator = 1.0f;

		my_parent->current_job_package.AddJob("ttttfffttibtif",	output_reconstruction_1.ToUTF8().data(),
															output_reconstruction_2.ToUTF8().data(),
															output_reconstruction_filtered.ToUTF8().data(),
															output_resolution_statistics.ToUTF8().data(),
															molecular_mass_kDa, inner_mask_radius, outer_mask_radius,
															dump_file_seed_1.ToUTF8().data(),
															dump_file_seed_2.ToUTF8().data(),
															class_counter + 1,
															save_orthogonal_views_image,
															orthogonal_views_filename.ToUTF8().data(),
															number_of_reconstruction_jobs, weiner_nominator);
	}
}



void RefinementManager::RunMerge3dJob()
{
	running_job_type = MERGE;

	// start job..

	if (output_refinement->number_of_classes > 1) my_parent->WriteBlueText("Merging and Filtering Reconstructions...");
	else
	my_parent->WriteBlueText("Merging and Filtering Reconstruction...");

	current_job_id = main_frame->job_controller.AddJob(my_parent, active_reconstruction_run_profile.manager_command, active_reconstruction_run_profile.gui_address);
	my_parent->my_job_id = current_job_id;

	if (current_job_id != -1)
	{
		long number_of_refinement_processes;
	    if (my_parent->current_job_package.number_of_jobs + 1 < my_parent->current_job_package.my_profile.ReturnTotalJobs()) number_of_refinement_processes = my_parent->current_job_package.number_of_jobs + 1;
	    else number_of_refinement_processes =  my_parent->current_job_package.my_profile.ReturnTotalJobs();

		if (number_of_refinement_processes >= 100000) my_parent->length_of_process_number = 6;
		else
		if (number_of_refinement_processes >= 10000) my_parent->length_of_process_number = 5;
		else
		if (number_of_refinement_processes >= 1000) my_parent->length_of_process_number = 4;
		else
		if (number_of_refinement_processes >= 100) my_parent->length_of_process_number = 3;
		else
		if (number_of_refinement_processes >= 10) my_parent->length_of_process_number = 2;
		else
		my_parent->length_of_process_number = 1;

		if (my_parent->length_of_process_number == 6) my_parent->NumberConnectedText->SetLabel(wxString::Format("%6i / %6li processes connected.", 0, number_of_refinement_processes));
		else
		if (my_parent->length_of_process_number == 5) my_parent->NumberConnectedText->SetLabel(wxString::Format("%5i / %5li processes connected.", 0, number_of_refinement_processes));
		else
		if (my_parent->length_of_process_number == 4) my_parent->NumberConnectedText->SetLabel(wxString::Format("%4i / %4li processes connected.", 0, number_of_refinement_processes));
		else
		if (my_parent->length_of_process_number == 3) my_parent->NumberConnectedText->SetLabel(wxString::Format("%3i / %3li processes connected.", 0, number_of_refinement_processes));
		else
		if (my_parent->length_of_process_number == 2) my_parent->NumberConnectedText->SetLabel(wxString::Format("%2i / %2li processes connected.", 0, number_of_refinement_processes));
		else

		my_parent->NumberConnectedText->SetLabel(wxString::Format("%i / %li processes connected.", 0, number_of_refinement_processes));

		my_parent->StartPanel->Show(false);
		my_parent->ProgressPanel->Show(true);

		my_parent->ExpertPanel->Show(false);
		my_parent->InfoPanel->Show(false);
		my_parent->OutputTextPanel->Show(true);
			//	CTFResultsPanel->Show(true);

		my_parent->ExpertToggleButton->Enable(false);
		my_parent->RefinementPackageComboBox->Enable(false);
		my_parent->InputParametersComboBox->Enable(false);

		my_parent->TimeRemainingText->SetLabel("Time Remaining : ???h:??m:??s");
		my_parent->Layout();
		my_parent->running_job = true;
		my_parent->my_job_tracker.StartTracking(my_parent->current_job_package.number_of_jobs);

		}

		my_parent->ProgressBar->Pulse();
}


void RefinementManager::SetupReconstructionJob()
{
	wxArrayString written_parameter_files;

	if (start_with_reconstruction == true) written_parameter_files = output_refinement->WritecisTEMStarFiles(main_frame->current_project.parameter_file_directory.GetFullPath() + "/output_par", active_percent_used / 100.0, 1.0);
	else
	written_parameter_files = output_refinement->WritecisTEMStarFiles(main_frame->current_project.parameter_file_directory.GetFullPath() + "/output_par");


	int class_counter;
	long counter;
	int job_counter;
	long number_of_reconstruction_jobs;
	long number_of_reconstruction_processes;
	float current_particle_counter;

	long number_of_particles;
	float particles_per_job;

	// for now, number of jobs is number of processes -1 (master)..

	number_of_reconstruction_processes = active_reconstruction_run_profile.ReturnTotalJobs();
	number_of_reconstruction_jobs = number_of_reconstruction_processes - 1;

	number_of_particles = active_refinement_package->contained_particles.GetCount();

	if (number_of_particles - number_of_reconstruction_jobs < number_of_reconstruction_jobs) particles_per_job = 1;
	else particles_per_job = float(number_of_particles - number_of_reconstruction_jobs) / float(number_of_reconstruction_jobs);

	my_parent->current_job_package.Reset(active_reconstruction_run_profile, "reconstruct3d", number_of_reconstruction_jobs * active_refinement_package->number_of_classes);

	for (class_counter = 0; class_counter < active_refinement_package->number_of_classes; class_counter++)
	{
		current_particle_counter = 1.0;

		for (job_counter = 0; job_counter < number_of_reconstruction_jobs; job_counter++)
		{
			wxString input_particle_stack 		= active_refinement_package->stack_filename;
			wxString input_parameter_file 		= written_parameter_files[class_counter];
			wxString output_reconstruction_1    = "/dev/null";
			wxString output_reconstruction_2			= "/dev/null";
			wxString output_reconstruction_filtered		= "/dev/null";
			wxString output_resolution_statistics		= "/dev/null";
			wxString my_symmetry						= active_refinement_package->symmetry;

			long	 first_particle						= myroundint(current_particle_counter);

			current_particle_counter += particles_per_job;
			if (current_particle_counter > number_of_particles  || job_counter == number_of_reconstruction_jobs - 1) current_particle_counter = number_of_particles;

			long	 last_particle						= myroundint(current_particle_counter);
			current_particle_counter+=1.0;

			//float 	 pixel_size							= active_refinement_package->contained_particles[0].pixel_size;
			//float    voltage_kV							= active_refinement_package->contained_particles[0].microscope_voltage;
			//float 	 spherical_aberration_mm			= active_refinement_package->contained_particles[0].spherical_aberration;
			//float    amplitude_contrast					= active_refinement_package->contained_particles[0].amplitude_contrast;
			float 	 molecular_mass_kDa					= active_refinement_package->estimated_particle_weight_in_kda;
			float    inner_mask_radius					= active_inner_mask_radius;
			float    outer_mask_radius					= active_mask_radius;
			float    resolution_limit_rec				= active_resolution_limit_rec;
			float    score_weight_conversion			= active_score_weight_conversion;
			float    score_threshold					= active_score_threshold;
			bool	 adjust_scores						= active_adjust_scores;
			bool	 invert_contrast					= active_refinement_package->stack_has_white_protein;
			bool	 crop_images						= active_crop_images;
			bool	 dump_arrays						= true;
			wxString dump_file_1 						= main_frame->ReturnRefine3DScratchDirectory() + wxString::Format("dump_file_%li_%i_odd_%i.dump", current_output_refinement_id, class_counter, job_counter +1);
			wxString dump_file_2 						= main_frame->ReturnRefine3DScratchDirectory() + wxString::Format("dump_file_%li_%i_even_%i.dump", current_output_refinement_id, class_counter, job_counter + 1);

			wxString input_reconstruction;
			bool	 use_input_reconstruction;
			float pixel_size_of_reference = active_refinement_package->output_pixel_size;

			if (active_should_apply_blurring == true)
			{
				// do we have a reference..

				if (active_refinement_package->references_for_next_refinement[class_counter] == -1)
				{
					input_reconstruction			= "/dev/null";
					use_input_reconstruction		= false;
				}
				else
				{
					input_reconstruction = current_reference_filenames.Item(class_counter);//volume_asset_panel->ReturnAssetLongFilename(volume_asset_panel->ReturnArrayPositionFromAssetID(refinement_package_asset_panel->all_refinement_packages.Item(my_parent->RefinementPackageComboBox->GetSelection()).references_for_next_refinement[class_counter]));
					use_input_reconstruction = true;
				}


			}
			else
			{
				input_reconstruction			= "/dev/null";
				use_input_reconstruction		= false;
			}

			float    resolution_limit_ref               = active_high_resolution_limit;
			float	 smoothing_factor					= active_smoothing_factor;
			float    padding							= 1.0f;
			bool	 normalize_particles				= true;
			bool	 exclude_blank_edges				= false;
			bool	 split_even_odd						= false;
			bool     centre_mass                        = active_centre_mass;
			int		 max_threads						= 1;



			bool threshold_input_3d = true;

			my_parent->current_job_package.AddJob("ttttttttiiffffffffffbbbbbbbbbbtti",
																		input_particle_stack.ToUTF8().data(),
																		input_parameter_file.ToUTF8().data(),
																		input_reconstruction.ToUTF8().data(),
																		output_reconstruction_1.ToUTF8().data(),
																		output_reconstruction_2.ToUTF8().data(),
																		output_reconstruction_filtered.ToUTF8().data(),
																		output_resolution_statistics.ToUTF8().data(),
																		my_symmetry.ToUTF8().data(),
																		first_particle,
																		last_particle,
																		pixel_size_of_reference,
																		molecular_mass_kDa,
																		inner_mask_radius,
																		outer_mask_radius,
																		resolution_limit_rec,
																		resolution_limit_ref,
																		score_weight_conversion,
																		score_threshold,
																		smoothing_factor,
																		padding,
																		normalize_particles,
																		adjust_scores,
																		invert_contrast,
																		exclude_blank_edges,
																		crop_images,
																		split_even_odd,
																		centre_mass,
																		use_input_reconstruction,
																		threshold_input_3d,
																		dump_arrays,
																		dump_file_1.ToUTF8().data(),
																		dump_file_2.ToUTF8().data(),
																		max_threads);




		}
	}
}


// for now we take the paramter

void RefinementManager::RunReconstructionJob()
{
	running_job_type = RECONSTRUCTION;
	number_of_received_particle_results = 0;
	number_of_expected_results = output_refinement->ReturnNumberOfActiveParticlesInFirstClass() * output_refinement->number_of_classes;

	// in the future store the reconstruction parameters..

	// empty scratch directory..

//	if (wxDir::Exists(main_frame->current_project.scratch_directory.GetFullPath() + "/Refine3D/") == true) wxFileName::Rmdir(main_frame->current_project.scratch_directory.GetFullPath() + "/Refine3D/", wxPATH_RMDIR_RECURSIVE);
//	if (wxDir::Exists(main_frame->current_project.scratch_directory.GetFullPath() + "/Refine3D/") == false) wxFileName::Mkdir(main_frame->current_project.scratch_directory.GetFullPath() + "/Refine3D/");

	// launch a controller


	if (start_with_reconstruction == true)
	{
		if (output_refinement->number_of_classes > 1) my_parent->WriteBlueText("Calculating Initial Reconstructions...");
		else my_parent->WriteBlueText("Calculating Initial Reconstruction...");

	}
	else
	{
		if (output_refinement->number_of_classes > 1) my_parent->WriteBlueText("Calculating Reconstructions...");
		else my_parent->WriteBlueText("Calculating Reconstruction...");

	}

	current_job_id = main_frame->job_controller.AddJob(my_parent, active_reconstruction_run_profile.manager_command, active_reconstruction_run_profile.gui_address);
	my_parent->my_job_id = current_job_id;

	if (current_job_id != -1)
	{
		long number_of_refinement_processes;
	    if (my_parent->current_job_package.number_of_jobs + 1 < my_parent->current_job_package.my_profile.ReturnTotalJobs()) number_of_refinement_processes = my_parent->current_job_package.number_of_jobs + 1;
	    else number_of_refinement_processes =  my_parent->current_job_package.my_profile.ReturnTotalJobs();

		if (number_of_refinement_processes >= 100000) my_parent->length_of_process_number = 6;
		else
		if (number_of_refinement_processes >= 10000) my_parent->length_of_process_number = 5;
		else
		if (number_of_refinement_processes >= 1000) my_parent->length_of_process_number = 4;
		else
		if (number_of_refinement_processes >= 100) my_parent->length_of_process_number = 3;
		else
		if (number_of_refinement_processes >= 10) my_parent->length_of_process_number = 2;
		else
		my_parent->length_of_process_number = 1;

		if (my_parent->length_of_process_number == 6) my_parent->NumberConnectedText->SetLabel(wxString::Format("%6i / %6li processes connected.", 0, number_of_refinement_processes));
		else
		if (my_parent->length_of_process_number == 5) my_parent->NumberConnectedText->SetLabel(wxString::Format("%5i / %5li processes connected.", 0, number_of_refinement_processes));
		else
		if (my_parent->length_of_process_number == 4) my_parent->NumberConnectedText->SetLabel(wxString::Format("%4i / %4li processes connected.", 0, number_of_refinement_processes));
		else
		if (my_parent->length_of_process_number == 3) my_parent->NumberConnectedText->SetLabel(wxString::Format("%3i / %3li processes connected.", 0, number_of_refinement_processes));
		else
		if (my_parent->length_of_process_number == 2) my_parent->NumberConnectedText->SetLabel(wxString::Format("%2i / %2li processes connected.", 0, number_of_refinement_processes));

		my_parent->NumberConnectedText->SetLabel(wxString::Format("%i / %li processes connected.", 0, number_of_refinement_processes));
		my_parent->TimeRemainingText->SetLabel("Time Remaining : ???h:??m:??s");
		my_parent->Layout();
		my_parent->running_job = true;
		my_parent->my_job_tracker.StartTracking(my_parent->current_job_package.number_of_jobs);

	}
		my_parent->ProgressBar->Pulse();
}

void RefinementManager::SetupRefinementJob()
{

	int class_counter;
	long counter;
	long number_of_refinement_jobs;
	int number_of_refinement_processes;
	float current_particle_counter;

	long number_of_particles;
	float particles_per_job;

	// get the last refinement for the currently selected refinement package..

	input_refinement->WritecisTEMStarFiles(main_frame->current_project.parameter_file_directory.GetFullPath() + "/input_par");
	input_refinement->WriteResolutionStatistics(main_frame->current_project.parameter_file_directory.GetFullPath() + "/input_stats");

//	wxPrintf("Input refinement has %li particles\n", input_refinement->number_of_particles);

	// for now, number of jobs is number of processes -1 (master)..

	number_of_refinement_processes = active_refinement_run_profile.ReturnTotalJobs();
	number_of_refinement_jobs = number_of_refinement_processes - 1;

	number_of_particles = active_refinement_package->contained_particles.GetCount();
	if (number_of_particles - number_of_refinement_jobs < number_of_refinement_jobs) particles_per_job = 1;
	else particles_per_job = float(number_of_particles - number_of_refinement_jobs) / float(number_of_refinement_jobs);

	my_parent->current_job_package.Reset(active_refinement_run_profile, "refine3d", number_of_refinement_jobs * active_refinement_package->number_of_classes);

	for (class_counter = 0; class_counter < active_refinement_package->number_of_classes; class_counter++)
	{
		current_particle_counter = 1;

		for (counter = 0; counter < number_of_refinement_jobs; counter++)
		{

			wxString input_particle_images					= active_refinement_package->stack_filename;
			wxString input_parameter_file 					= main_frame->current_project.parameter_file_directory.GetFullPath() + wxString::Format("/input_par_%li_%i.star", current_input_refinement_id, class_counter + 1);
			wxString input_reconstruction					= current_reference_filenames.Item(class_counter);
			wxString input_reconstruction_statistics 		= main_frame->current_project.parameter_file_directory.GetFullPath() + wxString::Format("/input_stats_%li_%i.txt", current_input_refinement_id, class_counter + 1);
			bool	 use_statistics							= true;

			wxString ouput_matching_projections		 		= "";
			//					= "/tmp/output_par.par";
			//wxString ouput_shift_file						= "/tmp/output_shift.shft";
			//wxString output_parameter_file					= "/dev/null";
			wxString ouput_shift_file						= "/dev/null";

			wxString my_symmetry							= active_refinement_package->symmetry;
			long	 first_particle							= myroundint(current_particle_counter);

			current_particle_counter += particles_per_job;
			if (current_particle_counter > number_of_particles  || counter == number_of_refinement_jobs - 1) current_particle_counter = number_of_particles;

			long	 last_particle							= myroundint(current_particle_counter);
			current_particle_counter++;

			float	 percent_used							= active_percent_used / 100.0;

#ifdef DEBUG
			wxString output_parameter_file = wxString::Format("/tmp/output_par_%li_%li_%i.star", first_particle, last_particle, class_counter);
#else
			wxString output_parameter_file = "/dev/null";
#endif

			float 	 output_pixel_size						= active_refinement_package->output_pixel_size;
			float	 molecular_mass_kDa						= active_refinement_package->estimated_particle_weight_in_kda;
			float    mask_radius							= active_mask_radius;
			float    inner_mask_radius                      = active_inner_mask_radius;
			float    low_resolution_limit					= active_low_resolution_limit;
			float    high_resolution_limit					= active_high_resolution_limit;
			float	 signed_CC_limit						= active_signed_cc_limit;
			float	 classification_resolution_limit		= active_classification_high_res_limit;
			float    mask_radius_search						= active_global_mask_radius;
			float	 high_resolution_limit_search			= active_high_resolution_limit;
			float	 angular_step							= active_angular_search_step;
			int		 best_parameters_to_keep				= active_number_results_to_refine;
			float	 max_search_x							= active_search_range_x;
			float	 max_search_y							= active_search_range_y;
			float    mask_center_2d_x						= active_sphere_x_coord;
			float 	 mask_center_2d_y						= active_sphere_y_coord;
			float    mask_center_2d_z						= active_sphere_z_coord;
			float    mask_radius_2d							= active_sphere_radius;

			float	 defocus_search_range					= active_defocus_search_range;
			float	 defocus_step							= active_defocus_search_step;
			float	 padding								= 1.0;

			bool global_search;
			bool local_refinement;
			bool global_local_refinemnent = false;

			if (active_do_global_refinement == true)
			{
				global_search = true;
				local_refinement = false;
			}
			else
			{
				global_search = false;
				local_refinement = true;
			}

			bool ignore_input_parameters = active_also_refine_input;


			bool refine_psi 								= active_should_refine_psi;
			bool refine_theta								= active_should_refine_theta;
			bool refine_phi									= active_should_refine_phi;
			bool refine_x_shift								= active_should_refine_x_shift;
			bool refine_y_shift								= active_should_refine_y_shift;
			bool calculate_matching_projections				= false;
			bool apply_2d_masking							= active_should_focus_classify;
			bool ctf_refinement								= active_should_refine_ctf;
			bool invert_contrast							= active_refinement_package->stack_has_white_protein;

			bool normalize_particles = true;
			bool exclude_blank_edges = false;
			bool normalize_input_3d;

			if (active_should_apply_blurring == true) normalize_input_3d = false;
			else normalize_input_3d = true;

			bool threshold_input_3d = true;
			bool defocus_bias = false;

			int max_threads = 1;

			my_parent->current_job_package.AddJob("ttttbttttiiffffffffffffifffffffffbbbbbbbbbbbbbbbibibb",
																											input_particle_images.ToUTF8().data(),
																											input_parameter_file.ToUTF8().data(),
																											input_reconstruction.ToUTF8().data(),
																											input_reconstruction_statistics.ToUTF8().data(),
																											use_statistics,
																											ouput_matching_projections.ToUTF8().data(),
																											output_parameter_file.ToUTF8().data(),
																											ouput_shift_file.ToUTF8().data(),
																											my_symmetry.ToUTF8().data(),
																											first_particle,
																											last_particle,
																											percent_used,
																											output_pixel_size,
																											molecular_mass_kDa,
																											inner_mask_radius,
																											mask_radius,
																											low_resolution_limit,
																											high_resolution_limit,
																											signed_CC_limit,
																											classification_resolution_limit,
																											mask_radius_search,
																											high_resolution_limit_search,
																											angular_step,
																											best_parameters_to_keep,
																											max_search_x,
																											max_search_y,
																											mask_center_2d_x,
																											mask_center_2d_y,
																											mask_center_2d_z,
																											mask_radius_2d,
																											defocus_search_range,
																											defocus_step,
																											padding,
																											global_search,
																											local_refinement,
																											refine_psi,
																											refine_theta,
																											refine_phi,
																											refine_x_shift,
																											refine_y_shift,
																											calculate_matching_projections,
																											apply_2d_masking,
																											ctf_refinement,
																											normalize_particles,
																											invert_contrast,
																											exclude_blank_edges,
																											normalize_input_3d,
																											threshold_input_3d,
																											max_threads,
																											global_local_refinemnent,
																											class_counter,
																											ignore_input_parameters,
																											defocus_bias);


		}

	}

	/*

	int class_counter;
		long counter;
		long number_of_refinement_jobs;
		int number_of_refinement_processes;
		float current_particle_counter;

		long number_of_particles;
		float particles_per_job;

		// get the last refinement for the currently selected refinement package..

		input_refinement->WriteFrealignParameterFiles(main_frame->current_project.parameter_file_directory.GetFullPath() + "/input_par");
		input_refinement->WriteResolutionStatistics(main_frame->current_project.parameter_file_directory.GetFullPath() + "/input_stats");

	//	wxPrintf("Input refinement has %li particles\n", input_refinement->number_of_particles);

		// for now, number of jobs is number of processes -1 (master)..

		number_of_refinement_processes = run_profiles_panel->run_profile_manager.run_profiles[my_parent->RefinementRunProfileComboBox->GetSelection()].ReturnTotalJobs();
		number_of_refinement_jobs = number_of_refinement_processes - 1;

		number_of_particles = refinement_package_asset_panel->all_refinement_packages.Item(my_parent->RefinementPackageComboBox->GetSelection()).contained_particles.GetCount();
		if (number_of_particles - number_of_refinement_jobs < number_of_refinement_jobs) particles_per_job = 1;
		else particles_per_job = float(number_of_particles - number_of_refinement_jobs) / float(number_of_refinement_jobs);

		my_parent->current_job_package.Reset(run_profiles_panel->run_profile_manager.run_profiles[my_parent->RefinementRunProfileComboBox->GetSelection()], "refine3d", number_of_refinement_jobs * refinement_package_asset_panel->all_refinement_packages.Item(my_parent->RefinementPackageComboBox->GetSelection()).number_of_classes);

		for (class_counter = 0; class_counter < refinement_package_asset_panel->all_refinement_packages.Item(my_parent->RefinementPackageComboBox->GetSelection()).number_of_classes; class_counter++)
		{
			current_particle_counter = 1;

			for (counter = 0; counter < number_of_refinement_jobs; counter++)
			{

				wxString input_particle_images					= refinement_package_asset_panel->all_refinement_packages.Item(my_parent->RefinementPackageComboBox->GetSelection()).stack_filename;
				wxString input_parameter_file 					= main_frame->current_project.parameter_file_directory.GetFullPath() + wxString::Format("/input_par_%li_%i.par", current_input_refinement_id, class_counter + 1);
				wxString input_reconstruction					= volume_asset_panel->ReturnAssetLongFilename(volume_asset_panel->ReturnArrayPositionFromAssetID(refinement_package_asset_panel->all_refinement_packages.Item(my_parent->RefinementPackageComboBox->GetSelection()).references_for_next_refinement[class_counter]));
				wxString input_reconstruction_statistics 		= main_frame->current_project.parameter_file_directory.GetFullPath() + wxString::Format("/input_stats_%li_%i.txt", current_input_refinement_id, class_counter + 1);
				bool	 use_statistics							= true;

				wxString ouput_matching_projections		 		= "";
				//wxString output_parameter_file					= "/tmp/output_par.par";
				//wxString ouput_shift_file						= "/tmp/output_shift.shft";
				wxString output_parameter_file					= "/dev/null";
				wxString ouput_shift_file						= "/dev/null";

				wxString my_symmetry							= refinement_package_asset_panel->all_refinement_packages.Item(my_parent->RefinementPackageComboBox->GetSelection()).symmetry;
				long	 first_particle							= myroundint(current_particle_counter);

				current_particle_counter += particles_per_job;
				if (current_particle_counter > number_of_particles) current_particle_counter = number_of_particles;

				long	 last_particle							= myroundint(current_particle_counter);
				current_particle_counter++;

				float	 percent_used							= my_parent->PercentUsedTextCtrl->ReturnValue() / 100.0;


				// for now we take the paramters of the first image!!!!

				float 	 pixel_size								= refinement_package_asset_panel->all_refinement_packages.Item(my_parent->RefinementPackageComboBox->GetSelection()).contained_particles[0].pixel_size;
				float    voltage_kV								= refinement_package_asset_panel->all_refinement_packages.Item(my_parent->RefinementPackageComboBox->GetSelection()).contained_particles[0].microscope_voltage;
				float 	 spherical_aberration_mm				= refinement_package_asset_panel->all_refinement_packages.Item(my_parent->RefinementPackageComboBox->GetSelection()).contained_particles[0].spherical_aberration;
				float    amplitude_contrast						= refinement_package_asset_panel->all_refinement_packages.Item(my_parent->RefinementPackageComboBox->GetSelection()).contained_particles[0].amplitude_contrast;
				float	 molecular_mass_kDa						= refinement_package_asset_panel->all_refinement_packages.Item(my_parent->RefinementPackageComboBox->GetSelection()).estimated_particle_weight_in_kda;
				float    mask_radius							= my_parent->MaskRadiusTextCtrl->ReturnValue();
				float    low_resolution_limit					= my_parent->LowResolutionLimitTextCtrl->ReturnValue();
				float    high_resolution_limit					= my_parent->HighResolutionLimitTextCtrl->ReturnValue();
				float	 signed_CC_limit						= my_parent->SignedCCResolutionTextCtrl->ReturnValue();
				float	 classification_resolution_limit		= my_parent->ClassificationHighResLimitTextCtrl->ReturnValue();
				float    mask_radius_search						= my_parent->GlobalMaskRadiusTextCtrl->ReturnValue();
				float	 high_resolution_limit_search			= my_parent->HighResolutionLimitTextCtrl->ReturnValue();
				float	 angular_step							= my_parent->AngularStepTextCtrl->ReturnValue();
				int		 best_parameters_to_keep				= my_parent->NumberToRefineSpinCtrl->GetValue();
				float	 max_search_x							= my_parent->SearchRangeXTextCtrl->ReturnValue();
				float	 max_search_y							= my_parent->SearchRangeYTextCtrl->ReturnValue();
				float    mask_center_2d_x						= my_parent->SphereXTextCtrl->ReturnValue();
				float 	 mask_center_2d_y						= my_parent->SphereYTextCtrl->ReturnValue();
				float    mask_center_2d_z						= my_parent->SphereZTextCtrl->ReturnValue();
				float    mask_radius_2d							= my_parent->SphereRadiusTextCtrl->ReturnValue();

				float	 defocus_search_range					= my_parent->DefocusSearchRangeTextCtrl->ReturnValue();
				float	 defocus_step							= my_parent->DefocusSearchStepTextCtrl->ReturnValue();
				float	 padding								= 1.0;

				bool global_search;
				bool local_refinement;

				if (my_parent->GlobalRefinementRadio->GetValue() == true)
				{
					global_search = true;
					local_refinement = false;
				}
				else
				{
					global_search = false;
					local_refinement = true;
				}


				bool refine_psi 								= my_parent->RefinePsiCheckBox->GetValue();
				bool refine_theta								= my_parent->RefineThetaCheckBox->GetValue();
				bool refine_phi									= my_parent->RefinePhiCheckBox->GetValue();
				bool refine_x_shift								= my_parent->RefineXShiftCheckBox->GetValue();
				bool refine_y_shift								= my_parent->RefineYShiftCheckBox->GetValue();
				bool calculate_matching_projections				= false;
				bool apply_2d_masking							= my_parent->SphereClassificatonYesRadio->GetValue();
				bool ctf_refinement								= my_parent->RefineCTFYesRadio->GetValue();
				bool invert_contrast							= false;

				bool normalize_particles = true;
				bool exclude_blank_edges = false;
				bool normalize_input_3d;

				if (my_parent->ApplyBlurringYesRadioButton->GetValue() == true) normalize_input_3d = false;
				else normalize_input_3d = true;

				my_parent->current_job_package.AddJob("ttttbttttiiffffffffffffffifffffffffbbbbbbbbbbbbbbi",
																	input_particle_images.ToUTF8().data(), 				// 0
																	input_parameter_file.ToUTF8().data(), 				// 1
																	input_reconstruction.ToUTF8().data(), 				// 2
																	input_reconstruction_statistics.ToUTF8().data(),	// 3
																	use_statistics, 									// 4
																	ouput_matching_projections.ToUTF8().data(),			// 5
																	output_parameter_file.ToUTF8().data(),				// 6
																	ouput_shift_file.ToUTF8().data(),					// 7
																	my_symmetry.ToUTF8().data(),						// 8
																	first_particle, 									// 9
																	last_particle,										// 10
																	percent_used,										// 11
																	pixel_size, 										// 12
																	voltage_kV,											// 13
																	spherical_aberration_mm,							// 14
																	amplitude_contrast,									// 15
																	molecular_mass_kDa,									// 16
																	mask_radius,										// 17
																	low_resolution_limit,								// 18
																	high_resolution_limit,								// 19
																	signed_CC_limit,									// 20
																	classification_resolution_limit,					// 21
																	mask_radius_search,									// 22
																	high_resolution_limit_search,						// 23
																	angular_step,										// 24
																	best_parameters_to_keep,							// 25
																	max_search_x,										// 26
																	max_search_y,										// 27
																	mask_center_2d_x,									// 28
																	mask_center_2d_y,									// 29
																	mask_center_2d_z,									// 30
																	mask_radius_2d,										// 31
																	defocus_search_range,								// 32
																	defocus_step,										// 33
																	padding,											// 34
																	global_search,										// 35
																	local_refinement,									// 36
																	refine_psi,											// 37
																	refine_theta,										// 38
																	refine_phi, 										// 39
																	refine_x_shift,										// 40
																	refine_y_shift,										// 41
																	calculate_matching_projections,						// 42
																	apply_2d_masking,									// 43
																	ctf_refinement,										// 44
																	normalize_particles,								// 45
																	invert_contrast,									// 46
																	exclude_blank_edges,								// 47
																	normalize_input_3d,									// 48
																	class_counter);										// 49


			}

		}*/
}

void RefinementManager::ProcessJobResult(JobResult *result_to_process)
{
	if (running_job_type == REFINEMENT)
	{

		int current_class = int(result_to_process->result_data[0] + 0.5);
		long current_particle = long(result_to_process->result_data[1] + 0.5) - 1;

		MyDebugAssertTrue(current_particle != -1 && current_class != -1, "Current Particle (%li) or Current Class(%i) = -1!", current_particle, current_class);

	//	wxPrintf("Received a refinement result for class #%i, particle %li\n", current_class + 1, current_particle + 1);
		//wxPrintf("output refinement has %i classes and %li particles\n", output_refinement->number_of_classes, output_refinement->number_of_particles);


		output_refinement->class_refinement_results[current_class].particle_refinement_results[current_particle].position_in_stack = long(result_to_process->result_data[1] + 0.5);
		output_refinement->class_refinement_results[current_class].particle_refinement_results[current_particle].image_is_active = int(result_to_process->result_data[2]);
		output_refinement->class_refinement_results[current_class].particle_refinement_results[current_particle].psi = result_to_process->result_data[3];
		output_refinement->class_refinement_results[current_class].particle_refinement_results[current_particle].theta = result_to_process->result_data[4];
		output_refinement->class_refinement_results[current_class].particle_refinement_results[current_particle].phi = result_to_process->result_data[5];
		output_refinement->class_refinement_results[current_class].particle_refinement_results[current_particle].xshift = result_to_process->result_data[6];
		output_refinement->class_refinement_results[current_class].particle_refinement_results[current_particle].yshift = result_to_process->result_data[7];
		output_refinement->class_refinement_results[current_class].particle_refinement_results[current_particle].defocus1 = result_to_process->result_data[8];
		output_refinement->class_refinement_results[current_class].particle_refinement_results[current_particle].defocus2 = result_to_process->result_data[9];
		output_refinement->class_refinement_results[current_class].particle_refinement_results[current_particle].defocus_angle = result_to_process->result_data[10];
		output_refinement->class_refinement_results[current_class].particle_refinement_results[current_particle].phase_shift = result_to_process->result_data[11];
		output_refinement->class_refinement_results[current_class].particle_refinement_results[current_particle].occupancy = result_to_process->result_data[12];
		output_refinement->class_refinement_results[current_class].particle_refinement_results[current_particle].logp = result_to_process->result_data[13];
		output_refinement->class_refinement_results[current_class].particle_refinement_results[current_particle].sigma = result_to_process->result_data[14];
		output_refinement->class_refinement_results[current_class].particle_refinement_results[current_particle].score = result_to_process->result_data[15];
		output_refinement->class_refinement_results[current_class].particle_refinement_results[current_particle].pixel_size = result_to_process->result_data[17];
		output_refinement->class_refinement_results[current_class].particle_refinement_results[current_particle].microscope_voltage_kv = result_to_process->result_data[18];
		output_refinement->class_refinement_results[current_class].particle_refinement_results[current_particle].microscope_spherical_aberration_mm = result_to_process->result_data[19];
		output_refinement->class_refinement_results[current_class].particle_refinement_results[current_particle].beam_tilt_x = result_to_process->result_data[20];
		output_refinement->class_refinement_results[current_class].particle_refinement_results[current_particle].beam_tilt_y = result_to_process->result_data[21];
		output_refinement->class_refinement_results[current_class].particle_refinement_results[current_particle].image_shift_x = result_to_process->result_data[22];
		output_refinement->class_refinement_results[current_class].particle_refinement_results[current_particle].image_shift_y = result_to_process->result_data[23];
		output_refinement->class_refinement_results[current_class].particle_refinement_results[current_particle].amplitude_contrast = result_to_process->result_data[24];

	/*	wxPrintf("Recieved a result for particle %li, x_shift = %f, y_shift = %f, psi = %f, theta = %f, phi = %f\n",		output_refinement->class_refinement_results[current_class].particle_refinement_results[current_particle].position_in_stack,
																															output_refinement->class_refinement_results[current_class].particle_refinement_results[current_particle].xshift,
																															output_refinement->class_refinement_results[current_class].particle_refinement_results[current_particle].yshift,
																															output_refinement->class_refinement_results[current_class].particle_refinement_results[current_particle].psi,
																															output_refinement->class_refinement_results[current_class].particle_refinement_results[current_particle].theta,
																															output_refinement->class_refinement_results[current_class].particle_refinement_results[current_particle].phi);*/


		number_of_received_particle_results++;
		//wxPrintf("received result!\n");
		long current_time = time(NULL);

		if (number_of_received_particle_results == 1)
		{
			current_job_starttime = current_time;
			time_of_last_update = 0;
			my_parent->ShowRefinementResultsPanel->AngularPlotPanel->SetSymmetryAndNumber(active_refinement_package->symmetry,long(float(output_refinement->number_of_particles) * active_percent_used * 0.01f));
			my_parent->Layout();
		}
		else
		if (current_time != time_of_last_update)
		{
			int current_percentage = float(number_of_received_particle_results) / float(number_of_expected_results) * 100.0;
			time_of_last_update = current_time;
			if (current_percentage > 100) current_percentage = 100;
			my_parent->ProgressBar->SetValue(current_percentage);
			long job_time = current_time - current_job_starttime;
			float seconds_per_job = float(job_time) / float(number_of_received_particle_results - 1);
			long seconds_remaining = float(number_of_expected_results - number_of_received_particle_results) * seconds_per_job;

			TimeRemaining time_remaining;

			if (seconds_remaining > 3600) time_remaining.hours = seconds_remaining / 3600;
			else time_remaining.hours = 0;

			if (seconds_remaining > 60) time_remaining.minutes = (seconds_remaining / 60) - (time_remaining.hours * 60);
			else time_remaining.minutes = 0;

			time_remaining.seconds = seconds_remaining - ((time_remaining.hours * 60 + time_remaining.minutes) * 60);
			my_parent->TimeRemainingText->SetLabel(wxString::Format("Time Remaining : %ih:%im:%is", time_remaining.hours, time_remaining.minutes, time_remaining.seconds));
		}


        // Add this result to the list of results to be plotted onto the angular plot
		if (current_class == 0 && output_refinement->class_refinement_results[current_class].particle_refinement_results[current_particle].image_is_active >= 0)
		{
			my_parent->ShowRefinementResultsPanel->AngularPlotPanel->AddRefinementResult( &output_refinement->class_refinement_results[current_class].particle_refinement_results[current_particle]);
	         // Plot this new result onto the angular plot immediately if it's one of the first few results to come in. Otherwise, only plot at regular intervals.

		     if(my_parent->ShowRefinementResultsPanel->AngularPlotPanel->refinement_results_to_plot.Count() * my_parent->ShowRefinementResultsPanel->AngularPlotPanel->symmetry_matrices.number_of_matrices < 1500 || current_time - my_parent->time_of_last_result_update > 0)
	        {
		        my_parent->ShowRefinementResultsPanel->AngularPlotPanel->Refresh();
	            my_parent->time_of_last_result_update = current_time;
	        }

		}
	}
	else
	if (running_job_type == RECONSTRUCTION)
	{
		//wxPrintf("Got reconstruction job \n");
		number_of_received_particle_results++;
	//	wxPrintf("Received a reconstruction intermmediate result\n");

		long current_time = time(NULL);

		if (number_of_received_particle_results == 1)
		{
			time_of_last_update = 0;
			current_job_starttime = current_time;
		}
		else
		if (current_time - time_of_last_update >= 1)
		{
			time_of_last_update = current_time;
			int current_percentage = float(number_of_received_particle_results) / float(number_of_expected_results) * 100.0;
			if (current_percentage > 100) current_percentage = 100;
			my_parent->ProgressBar->SetValue(current_percentage);
			long job_time = current_time - current_job_starttime;
			float seconds_per_job = float(job_time) / float(number_of_received_particle_results - 1);
			long seconds_remaining = float(number_of_expected_results - number_of_received_particle_results) * seconds_per_job;

			TimeRemaining time_remaining;
			if (seconds_remaining > 3600) time_remaining.hours = seconds_remaining / 3600;
			else time_remaining.hours = 0;

			if (seconds_remaining > 60) time_remaining.minutes = (seconds_remaining / 60) - (time_remaining.hours * 60);
			else time_remaining.minutes = 0;

			time_remaining.seconds = seconds_remaining - ((time_remaining.hours * 60 + time_remaining.minutes) * 60);
			my_parent->TimeRemainingText->SetLabel(wxString::Format("Time Remaining : %ih:%im:%is", time_remaining.hours, time_remaining.minutes, time_remaining.seconds));
		}


	}
	else
	if (running_job_type == MERGE)
	{
	//	wxPrintf("received merge result!\n");

		// add to the correct resolution statistics..

		int number_of_points = result_to_process->result_data[0];
		int class_number = int(result_to_process->result_data[1] + 0.5);
		int array_position = 2;
		float current_resolution;
		float fsc;
		float part_fsc;
		float part_ssnr;
		float rec_ssnr;

		wxPrintf("class_number = %i\n", class_number);
		// add the points..

		output_refinement->class_refinement_results[class_number - 1].class_resolution_statistics.Init(output_refinement->resolution_statistics_pixel_size, output_refinement->resolution_statistics_box_size);

		output_refinement->class_refinement_results[class_number - 1].class_resolution_statistics.FSC.ClearData();
		output_refinement->class_refinement_results[class_number - 1].class_resolution_statistics.part_FSC.ClearData();
		output_refinement->class_refinement_results[class_number - 1].class_resolution_statistics.part_SSNR.ClearData();
		output_refinement->class_refinement_results[class_number - 1].class_resolution_statistics.rec_SSNR.ClearData();


		for (int counter = 0; counter < number_of_points; counter++)
		{
			current_resolution = result_to_process->result_data[array_position];
			array_position++;
			fsc = result_to_process->result_data[array_position];
			array_position++;
			part_fsc = result_to_process->result_data[array_position];
			array_position++;
			part_ssnr = result_to_process->result_data[array_position];
			array_position++;
			rec_ssnr = result_to_process->result_data[array_position];
			array_position++;


			output_refinement->class_refinement_results[class_number - 1].class_resolution_statistics.FSC.AddPoint(current_resolution, fsc);
			output_refinement->class_refinement_results[class_number - 1].class_resolution_statistics.part_FSC.AddPoint(current_resolution, part_fsc);
			output_refinement->class_refinement_results[class_number - 1].class_resolution_statistics.part_SSNR.AddPoint(current_resolution, part_ssnr);
			output_refinement->class_refinement_results[class_number - 1].class_resolution_statistics.rec_SSNR.AddPoint(current_resolution, rec_ssnr);

		}

	}
}



void RefinementManager::ProcessAllJobsFinished()
{

	// Update the GUI with project timings
	extern MyOverviewPanel *overview_panel;
	overview_panel->SetProjectInfo();


	if (running_job_type == REFINEMENT)
	{
		main_frame->job_controller.KillJob(my_parent->my_job_id);

		// calculate occupancies..
		//if (output_refinement->percent_used < 99.99) output_refinement->UpdateOccupancies(false);
		//else output_refinement->UpdateOccupancies(true);
		if (my_parent->RefineOccupanciesCheckBox->GetValue() == true) output_refinement->UpdateOccupancies();
		else output_refinement->UpdateAverageOccupancy();

		SetupReconstructionJob();
		RunReconstructionJob();

	}
	else
	if (running_job_type == RECONSTRUCTION)
	{
		main_frame->job_controller.KillJob(my_parent->my_job_id);
		//wxPrintf("Reconstruction has finished\n");
		SetupMerge3dJob();
		RunMerge3dJob();
	}
	else
	if (running_job_type == MERGE)
	{
		long current_reconstruction_id;

		OrthDrawerThread *result_thread;
		my_parent->active_orth_thread_id = my_parent->next_thread_id;
		my_parent->next_thread_id++;

		if (start_with_reconstruction == true) result_thread = new OrthDrawerThread(my_parent, current_reference_filenames, wxString::Format("Iter. #%i", 0), 1.0f, active_mask_radius / input_refinement->resolution_statistics_pixel_size, my_parent->active_orth_thread_id);
		else result_thread = new OrthDrawerThread(my_parent, current_reference_filenames, wxString::Format("Iter. #%i", number_of_rounds_run + 1), 1.0f, active_mask_radius / input_refinement->resolution_statistics_pixel_size, my_parent->active_orth_thread_id);

		if ( result_thread->Run() != wxTHREAD_NO_ERROR )
		{
			my_parent->WriteErrorText("Error: Cannot start result creation thread, results not displayed");
			delete result_thread;
		}

		int class_counter;

		main_frame->job_controller.KillJob(my_parent->my_job_id);

		VolumeAsset temp_asset;

		temp_asset.pixel_size = output_refinement->resolution_statistics_pixel_size;
		temp_asset.x_size = output_refinement->resolution_statistics_box_size;
		temp_asset.y_size = output_refinement->resolution_statistics_box_size;
		temp_asset.z_size = output_refinement->resolution_statistics_box_size;

		// add the volumes etc to the database..
		main_frame->current_project.database.Begin();
		output_refinement->reference_volume_ids.Clear();
		active_refinement_package->references_for_next_refinement.Clear();

		main_frame->current_project.database.BeginVolumeAssetInsert();

		my_parent->WriteInfoText("");

		for (class_counter = 0; class_counter < output_refinement->number_of_classes; class_counter++)
		{
			temp_asset.asset_id = volume_asset_panel->current_asset_number;

			output_refinement->reference_volume_ids.Add(current_reference_asset_ids[class_counter]);
			current_reference_asset_ids[class_counter] = temp_asset.asset_id;

			if (start_with_reconstruction == true)
			{
				temp_asset.asset_name = wxString::Format("Start Params #%li - Class #%i", current_output_refinement_id, class_counter + 1);

			}
			else
			{
				if (active_do_global_refinement == true)
				{
					temp_asset.asset_name = wxString::Format("Global #%li (Rnd. %i) - Class #%i", current_output_refinement_id, number_of_rounds_run + 1, class_counter + 1);
				}
				else temp_asset.asset_name = wxString::Format("Local #%li (Rnd. %i) - Class #%i", current_output_refinement_id, number_of_rounds_run + 1, class_counter + 1);

			}
			// set the output volume
			output_refinement->class_refinement_results[class_counter].reconstructed_volume_asset_id = temp_asset.asset_id;
			output_refinement->class_refinement_results[class_counter].reconstruction_id = current_reconstruction_id;

			// add the reconstruction job

			current_reconstruction_id = main_frame->current_project.database.ReturnHighestReconstructionID() + 1;
			temp_asset.reconstruction_job_id = current_reconstruction_id;

			main_frame->current_project.database.AddReconstructionJob(current_reconstruction_id, active_refinement_package->asset_id, output_refinement->refinement_id, "", active_inner_mask_radius, active_mask_radius, active_resolution_limit_rec, active_score_weight_conversion, active_adjust_scores, active_crop_images, false, active_should_apply_blurring, active_smoothing_factor, class_counter + 1, long(temp_asset.asset_id));


			temp_asset.filename = main_frame->current_project.volume_asset_directory.GetFullPath() + wxString::Format("/volume_%li_%i.mrc", output_refinement->refinement_id, class_counter + 1);
			output_refinement->reference_volume_ids.Add(temp_asset.asset_id);

			active_refinement_package->references_for_next_refinement.Add(temp_asset.asset_id);
			main_frame->current_project.database.ExecuteSQL(wxString::Format("UPDATE REFINEMENT_PACKAGE_CURRENT_REFERENCES_%li SET VOLUME_ASSET_ID=%i WHERE CLASS_NUMBER=%i", current_refinement_package_asset_id, temp_asset.asset_id, class_counter + 1 ));


			volume_asset_panel->AddAsset(&temp_asset);
			main_frame->current_project.database.AddNextVolumeAsset(temp_asset.asset_id, temp_asset.asset_name, temp_asset.filename.GetFullPath(), temp_asset.reconstruction_job_id, temp_asset.pixel_size, temp_asset.x_size, temp_asset.y_size, temp_asset.z_size);
		}

		main_frame->current_project.database.EndVolumeAssetInsert();

		wxArrayFloat average_occupancies = output_refinement->UpdatePSSNR();

		my_parent->WriteInfoText("");

		if (output_refinement->number_of_classes > 1)
		{
			for (class_counter = 0; class_counter < output_refinement->number_of_classes; class_counter++)
			{
				my_parent->WriteInfoText(wxString::Format(wxT("Est. Res. Class %2i = %2.2f Å (%2.2f %%)"), class_counter + 1, output_refinement->class_refinement_results[class_counter].class_resolution_statistics.ReturnEstimatedResolution(), average_occupancies[class_counter]));
			}
		}
		else
		{
			my_parent->WriteInfoText(wxString::Format(wxT("Est. Res. = %2.2f Å"), output_refinement->class_refinement_results[0].class_resolution_statistics.ReturnEstimatedResolution()));
		}

		my_parent->WriteInfoText("");



		//wxPrintf("Writing to databse\n");
		// write the refinement to the database

		if (start_with_reconstruction == true)
		{
			long point_counter;

			main_frame->DirtyVolumes();
			//volume_asset_panel->is_dirty = true;
			//refinement_package_asset_panel->is_dirty = true;
			//my_parent->input_params_combo_is_dirty = true;
		//	my_parent->SetDefaults();
//			refinement_results_panel->is_dirty = true;

			Refinement *current_refinement = main_frame->current_project.database.GetRefinementByID(output_refinement->refinement_id);
			//update resolution statistics in database and gui..

			main_frame->current_project.database.UpdateRefinementResolutionStatistics(output_refinement);

			for (class_counter = 0; class_counter < current_refinement->number_of_classes; class_counter++)
			{
				current_refinement->class_refinement_results[class_counter].class_resolution_statistics.FSC.ClearData();
				current_refinement->class_refinement_results[class_counter].class_resolution_statistics.part_FSC.ClearData();
				current_refinement->class_refinement_results[class_counter].class_resolution_statistics.part_SSNR.ClearData();
				current_refinement->class_refinement_results[class_counter].class_resolution_statistics.rec_SSNR.ClearData();

				for (point_counter = 0; point_counter < output_refinement->class_refinement_results[class_counter].class_resolution_statistics.FSC.number_of_points; point_counter++)
				{
					current_refinement->class_refinement_results[class_counter].class_resolution_statistics.FSC.AddPoint(output_refinement->class_refinement_results[class_counter].class_resolution_statistics.FSC.data_x[point_counter], output_refinement->class_refinement_results[class_counter].class_resolution_statistics.FSC.data_y[point_counter]);
					current_refinement->class_refinement_results[class_counter].class_resolution_statistics.part_FSC.AddPoint(output_refinement->class_refinement_results[class_counter].class_resolution_statistics.part_FSC.data_x[point_counter], output_refinement->class_refinement_results[class_counter].class_resolution_statistics.part_FSC.data_y[point_counter]);
					current_refinement->class_refinement_results[class_counter].class_resolution_statistics.part_SSNR.AddPoint(output_refinement->class_refinement_results[class_counter].class_resolution_statistics.part_SSNR.data_x[point_counter], output_refinement->class_refinement_results[class_counter].class_resolution_statistics.part_SSNR.data_y[point_counter]);
					current_refinement->class_refinement_results[class_counter].class_resolution_statistics.rec_SSNR.AddPoint(output_refinement->class_refinement_results[class_counter].class_resolution_statistics.rec_SSNR.data_x[point_counter], output_refinement->class_refinement_results[class_counter].class_resolution_statistics.rec_SSNR.data_y[point_counter]);

				}

			}

			my_parent->ShowRefinementResultsPanel->FSCResultsPanel->AddRefinement(output_refinement);

			if (my_parent->ShowRefinementResultsPanel->TopBottomSplitter->IsSplit() == false)
			{
				my_parent->ShowRefinementResultsPanel->TopBottomSplitter->SplitHorizontally(my_parent->ShowRefinementResultsPanel->TopPanel, my_parent->ShowRefinementResultsPanel->BottomPanel);
				my_parent->ShowRefinementResultsPanel->FSCResultsPanel->Show(true);
			}
		}
		else
		{
			// calculate angular distribution histograms
			ArrayofAngularDistributionHistograms all_histograms = output_refinement->ReturnAngularDistributions(active_refinement_package->symmetry);

			for (class_counter = 1; class_counter <= output_refinement->number_of_classes; class_counter++)
			{
				main_frame->current_project.database.AddRefinementAngularDistribution(all_histograms[class_counter - 1], output_refinement->refinement_id, class_counter);
			}

			main_frame->current_project.database.AddRefinement(output_refinement);
			ShortRefinementInfo temp_info;
			temp_info = output_refinement;
			refinement_package_asset_panel->all_refinement_short_infos.Add(temp_info);

			// add this refinment to the refinement package..

			active_refinement_package->last_refinment_id = output_refinement->refinement_id;
			active_refinement_package->refinement_ids.Add(output_refinement->refinement_id);

			main_frame->current_project.database.ExecuteSQL(wxString::Format("UPDATE REFINEMENT_PACKAGE_ASSETS SET LAST_REFINEMENT_ID=%li WHERE REFINEMENT_PACKAGE_ASSET_ID=%li", output_refinement->refinement_id, current_refinement_package_asset_id));
			main_frame->current_project.database.ExecuteSQL(wxString::Format("INSERT INTO REFINEMENT_PACKAGE_REFINEMENTS_LIST_%li (REFINEMENT_NUMBER, REFINEMENT_ID) VALUES (%li, %li);", current_refinement_package_asset_id, main_frame->current_project.database.ReturnSingleLongFromSelectCommand(wxString::Format("SELECT MAX(REFINEMENT_NUMBER) FROM REFINEMENT_PACKAGE_REFINEMENTS_LIST_%li", current_refinement_package_asset_id)) + 1,  output_refinement->refinement_id));

			main_frame->DirtyVolumes();
			main_frame->DirtyRefinements();

//			volume_asset_panel->is_dirty = true;
	//		refinement_package_asset_panel->is_dirty = true;
		//	my_parent->input_params_combo_is_dirty = true;
	//		my_parent->SetDefaults();
			//refinement_results_panel->is_dirty = true;

			my_parent->ShowRefinementResultsPanel->FSCResultsPanel->AddRefinement(output_refinement);

			if (my_parent->ShowRefinementResultsPanel->TopBottomSplitter->IsSplit() == false)
			{
				my_parent->ShowRefinementResultsPanel->TopBottomSplitter->SplitHorizontally(my_parent->ShowRefinementResultsPanel->TopPanel, my_parent->ShowRefinementResultsPanel->BottomPanel);
				my_parent->ShowRefinementResultsPanel->FSCResultsPanel->Show(true);
			}
		}

		main_frame->current_project.database.Commit();
		global_delete_refine3d_scratch();

		my_parent->Layout();


		//wxPrintf("Calling cycle refinement\n");
		main_frame->DirtyVolumes();
		main_frame->DirtyRefinements();
		CycleRefinement();
	}

}

void RefinementManager::DoMasking()
{
	MyDebugAssertTrue(active_should_mask == true || active_should_auto_mask == true, "DoMasking called, when masking not selected!");

	wxArrayString masked_filenames;
	wxString current_masked_filename;
	wxString filename_of_mask = active_mask_filename;

	for (int class_counter = 0; class_counter < current_reference_filenames.GetCount(); class_counter++)
	{
		current_masked_filename = main_frame->ReturnRefine3DScratchDirectory();
		current_masked_filename += wxFileName(current_reference_filenames.Item(class_counter)).GetName();
		current_masked_filename += "_masked.mrc";

		masked_filenames.Add(current_masked_filename);
	}

	if (active_should_mask == true) // user selected masking
	{
		float wanted_cosine_edge_width = active_mask_edge;
		float wanted_weight_outside_mask = active_mask_weight;

		float wanted_low_pass_filter_radius;

		if (active_should_low_pass_filter_mask == true)
		{
			wanted_low_pass_filter_radius = active_mask_filter_resolution;
		}
		else
		{
			wanted_low_pass_filter_radius = 0.0;
		}

		my_parent->active_mask_thread_id = my_parent->next_thread_id;
		my_parent->next_thread_id++;

		Multiply3DMaskerThread *mask_thread = new Multiply3DMaskerThread(my_parent, current_reference_filenames, masked_filenames, filename_of_mask, wanted_cosine_edge_width, wanted_weight_outside_mask, wanted_low_pass_filter_radius, input_refinement->resolution_statistics_pixel_size, my_parent->active_mask_thread_id);

		if ( mask_thread->Run() != wxTHREAD_NO_ERROR )
		{
			my_parent->WriteErrorText("Error: Cannot start masking thread, masking will not be performed");
			delete mask_thread;
		}
		else
		{
			current_reference_filenames = masked_filenames;
			return;
		}
	}
	else
	{
		my_parent->active_mask_thread_id = my_parent->next_thread_id;
		my_parent->next_thread_id++;

		AutoMaskerThread *mask_thread = new AutoMaskerThread(my_parent, current_reference_filenames, masked_filenames, input_refinement->resolution_statistics_pixel_size, active_refinement_package->estimated_particle_size_in_angstroms * 0.75, my_parent->active_mask_thread_id);

		if ( mask_thread->Run() != wxTHREAD_NO_ERROR )
		{
			my_parent->WriteErrorText("Error: Cannot start masking thread, masking will not be performed");
			delete mask_thread;
		}
		else
		{
			current_reference_filenames = masked_filenames;
			return;
		}
	}



}

#ifdef EXPERIMENTAL
void RefinementManager::MergeMapModel()
{
	wxString		current_reference_filename; // reference volume (iterable)
	wxString		current_reference_dir; // parts
	wxString		current_reference_basename; // parts
	wxString		current_reference_ext; // parts
	wxArrayString   merged_filenames; // generated (array)
	wxString		current_merged_filename; // generated (iterable)
	wxString		phenix_bin_dir; // accept from user params
	wxString		model_filename; // accept from user params
	float			model_resolution; // accept from user params
	float			boundary_resolution; // accept from user params

	phenix_bin_dir = phenix_settings_panel->buffer_phenix_path;
	MyDebugAssertTrue(wxDirExists(phenix_bin_dir), "Can't read this directory for Phenix executables:%s",phenix_bin_dir.ToStdString());
	MyDebugAssertTrue(phenix_bin_dir.find_last_of("bin") != wxNOT_FOUND, "Path to Phenix bin directory does not contain \"bin\"");

	for (int class_counter = 0; class_counter < current_reference_filenames.GetCount(); class_counter++)
	{
		current_reference_filename = current_reference_filenames.Item(class_counter);
		wxFileName::SplitPath(current_reference_filenames.Item(class_counter), &current_reference_dir, &current_reference_basename, &current_reference_ext, wxPATH_NATIVE);
		current_merged_filename = main_frame->ReturnRefine3DScratchDirectory() + "/" + current_reference_basename + "_merged.mrc";
		wxPrintf(current_merged_filename + "\n");
		merged_filenames.Add(current_merged_filename);
	}


	// BEGIN Instead of the thread
	Image input_image;
	Image fmodel_image;
	Image scaled_fmodel_image;
	wxString fmodel_filename;
	ImageFile fmodel_file;
	ImageFile input_map_file;
	MRCFile output_map_file;

	float boundary_frequency;
	float half_shell_width;
	float boundary_shell_lower;
	float boundary_shell_upper;
	float reference_amplitude;
	float fmodel_amplitude;
	float amplitude_scale_factor;

	// call the fmodel calculation class with regridding
	model_filename = active_fmodel_model_filename;
	MyDebugAssertTrue(wxFileName::IsFileReadable(model_filename), "Can't read the model:%s",model_filename.ToStdString());
	fmodel_filename = model_filename + ".mrc"; // FIXME TEMP make this nicer
	model_resolution = active_fmodel_resolution;
	boundary_resolution = active_boundary_resolution;
	boundary_frequency = 1/boundary_resolution;
	wxPrintf("boundary_frequency:%.16f\n",boundary_frequency);
	my_parent->WriteBlueText("Merging the last reference volume with a map calculated from the provided model. This could take a minute...\n");
	// TODO: put a spinning wheel of death to let the user know it's working
	FmodelRegrid fmodel_regrid_tool;
	fmodel_regrid_tool.SetupLauncher(phenix_bin_dir, main_frame->ReturnRefine3DScratchDirectory());
	fmodel_regrid_tool.SetAllUserParameters(model_filename, current_reference_filenames.Item(0), fmodel_filename, model_resolution); // any of the references is fine to get the dimensions right
	fmodel_regrid_tool.RunFmodelRegrid();
	MyDebugAssertTrue(wxFileName::IsFileReadable(fmodel_filename), "Can't read the generated map:%s",fmodel_filename.ToStdString());
	fmodel_file.OpenFile(fmodel_filename.ToStdString(), false);
	fmodel_image.ReadSlices(&fmodel_file, 1, fmodel_file.ReturnNumberOfSlices());
	fmodel_image.ForwardFFT();
	fmodel_file.CloseFile();

	// get the average amplitude in the shell at the boundary resolution for scaling purposes, and the shell bounds
	half_shell_width = fmodel_image.fourier_voxel_size_z * 20;
	wxPrintf("half_shell_width:%.16f\n",half_shell_width);
	boundary_shell_lower = boundary_frequency - half_shell_width;
	boundary_shell_upper = boundary_frequency + half_shell_width;
	fmodel_amplitude = fmodel_image.ReturnAverageAmplitudeInShell(boundary_shell_lower, boundary_shell_upper);
	wxPrintf("fmodel_amplitude:%.16f\n",fmodel_amplitude);

	// loop through classes, read each map, merge map and fmodel map, write out the matching merged image
	// loop over the classes. For each, calculate the average amplitude in the shell at the boundary resolution
	// and scale the fmodel amplitudes to match. Then replace the high frequency region of the input image with
	// the scaled fmodel frequencies.

	for (int class_counter = 0; class_counter < current_reference_filenames.GetCount(); class_counter++)
	{
		// read the experimental map
		input_map_file.OpenFile(current_reference_filenames.Item(class_counter).ToStdString(), false);
		//input_map_file.OpenFile("/gne/scratch/u/youngi4/Apof_1p4975px/Assets/Volumes/volume_14_1.mrc", false); // TMP TESTING
		input_image.ReadSlices(&input_map_file, 1, input_map_file.ReturnNumberOfSlices());
		input_map_file.CloseFile();

		// take the high resolution pixels from the scaled_fmodel_image and the rest from the input_image
		input_image.ForwardFFT();
		reference_amplitude = input_image.ReturnAverageAmplitudeInShell(boundary_shell_lower, boundary_shell_upper);
		wxPrintf("reference_amplitude:%.16f\n",reference_amplitude);
		amplitude_scale_factor = reference_amplitude/fmodel_amplitude;
		wxPrintf("amlitude_scale_factor:%.16f\n",amplitude_scale_factor);
		scaled_fmodel_image.CopyFrom(&fmodel_image);
		scaled_fmodel_image.MultiplyByConstant(amplitude_scale_factor);
		input_image.ReplaceHighRes(&scaled_fmodel_image, boundary_frequency);
		input_image.BackwardFFT();

		// write out the new merged map
		output_map_file.OpenFile(merged_filenames.Item(class_counter).ToStdString(), true);
		input_image.WriteSlices(&output_map_file, 1, input_image.logical_z_dimension);
		output_map_file.CloseFile();
		wxPrintf("wrote merged map at %s\n",merged_filenames.Item(class_counter).ToStdString());
	}
	// END Instead of the thread

	current_reference_filenames = merged_filenames;
	return;
}
#endif

void RefinementManager::CycleRefinement()
{
	if (start_with_reconstruction == true)
	{
		output_refinement = new Refinement;
		start_with_reconstruction = false;
#ifdef EXPERIMENTAL
		if (active_should_merge_map_model == true) // TODO: read current boolean value from the GUI
		{
			MergeMapModel();
		}
#endif
		if (active_should_mask == true || active_should_auto_mask == true)
		{
			DoMasking();
		}
		else
		{
			SetupRefinementJob();
			RunRefinementJob();
		}
	}
	else
	{
		number_of_rounds_run++;

		if (number_of_rounds_run < number_of_rounds_to_run)
		{
			current_input_refinement_id = output_refinement->refinement_id;
			delete input_refinement;
			input_refinement = output_refinement;
			output_refinement = new Refinement;

#ifdef EXPERIMENTAL
			if (active_should_merge_map_model == true) // TODO: read current boolean value from the GUI
			{
				MergeMapModel();
			}
#endif
			if (active_should_mask == true || active_should_auto_mask == true)
			{
				DoMasking();
			}
			else
			{
				SetupRefinementJob();
				RunRefinementJob();
			}
		}
		else
		{
			delete input_refinement;
//			delete output_refinement;
			my_parent->WriteBlueText("All refinement cycles are finished!");
			my_parent->CancelAlignmentButton->Show(false);
			my_parent->FinishButton->Show(true);
			my_parent->TimeRemainingText->SetLabel("Time Remaining : Finished!");
			my_parent->ProgressBar->SetValue(100);
			my_parent->ProgressPanel->Layout();
		}
	}

	main_frame->DirtyVolumes();
	main_frame->DirtyRefinements();
}


void MyRefine3DPanel::OnMaskerThreadComplete(wxThreadEvent& my_event)
{
	if (my_event.GetInt() == active_mask_thread_id)	my_refinement_manager.OnMaskerThreadComplete();
}


void MyRefine3DPanel::OnOrthThreadComplete(ReturnProcessedImageEvent& my_event)
{

	Image *new_image = my_event.GetImage();

	if (my_event.GetInt() == active_orth_thread_id)
	{
		if (new_image != NULL)
		{
			ShowRefinementResultsPanel->ShowOrthDisplayPanel->OpenImage(new_image, my_event.GetString(), true);

			if (ShowRefinementResultsPanel->LeftRightSplitter->IsSplit() == false)
			{
				ShowRefinementResultsPanel->LeftRightSplitter->SplitVertically(ShowRefinementResultsPanel->LeftPanel, ShowRefinementResultsPanel->RightPanel, 600);
				Layout();
			}
		}
	}
	else
	{
		delete new_image;
	}

}


void RefinementManager::OnMaskerThreadComplete()
{
	//my_parent->WriteInfoText("Masking Finished");
	SetupRefinementJob();
	RunRefinementJob();
}

