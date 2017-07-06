#include "../core/gui_core_headers.h"

extern MyRefinementPackageAssetPanel *refinement_package_asset_panel;
extern MyRunProfilesPanel *run_profiles_panel;
extern MyVolumeAssetPanel *volume_asset_panel;
extern MyRefinementResultsPanel *refinement_results_panel;

MyRefine2DPanel::MyRefine2DPanel( wxWindow* parent )
:
Refine2DPanel( parent )
{

	wxSize input_size = InputSizer->GetMinSize();
	input_size.x += wxSystemSettings::GetMetric(wxSYS_VSCROLL_X);
	input_size.y = -1;
	ExpertPanel->SetMinSize(input_size);
	ExpertPanel->SetSize(input_size);

	running_job = false;

	SetInfo();
	FillRefinementPackagesComboBox();

	my_classification_manager.SetParent(this);

	ResultDisplayPanel->Initialise(CAN_FFT | START_WITH_FOURIER_SCALING);

	RefinementPackageComboBox->AssetComboBox->Bind(wxEVT_COMMAND_COMBOBOX_SELECTED, &MyRefine2DPanel::OnRefinementPackageComboBox, this);

	/*
	buffered_results = NULL;

	// Fill combo box..

	//FillGroupComboBox();

	my_job_id = -1;


//	group_combo_is_dirty = false;
//	run_profiles_are_dirty = false;FillReifnem


//	FillGroupComboBox();
//	FillRunProfileComboBox();




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

//	refinement_package_combo_is_dirty = false;
//	run_profiles_are_dirty = false;
//	input_params_combo_is_dirty = false;
//	selected_refinement_package = -1;

//	my_refinement_manager.SetParent(this);

//

//	long time_of_last_result_update;


}


void MyRefine2DPanel::WriteInfoText(wxString text_to_write)
{
	output_textctrl->SetDefaultStyle(wxTextAttr(*wxBLACK));
	output_textctrl->AppendText(text_to_write);

	if (text_to_write.EndsWith("\n") == false)	 output_textctrl->AppendText("\n");
}

void MyRefine2DPanel::WriteBlueText(wxString text_to_write)
{
	output_textctrl->SetDefaultStyle(wxTextAttr(*wxBLUE));
	output_textctrl->AppendText(text_to_write);

	if (text_to_write.EndsWith("\n") == false)	 output_textctrl->AppendText("\n");
}

void MyRefine2DPanel::WriteErrorText(wxString text_to_write)
{
	 output_textctrl->SetDefaultStyle(wxTextAttr(*wxRED));
	 output_textctrl->AppendText(text_to_write);

	 if (text_to_write.EndsWith("\n") == false)	 output_textctrl->AppendText("\n");
}

void MyRefine2DPanel::OnExpertOptionsToggle( wxCommandEvent& event )
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

void MyRefine2DPanel::FillRefinementPackagesComboBox()
{
	if (RefinementPackageComboBox->FillComboBox() == false) NewRefinementPackageSelected();
}

void MyRefine2DPanel::FillInputParamsComboBox()
{
	if (RefinementPackageComboBox->GetSelection() >= 0 && refinement_package_asset_panel->all_refinement_packages.GetCount() > 0)
	{
		InputParametersComboBox->FillComboBox(RefinementPackageComboBox->GetSelection(), true);
	}
}


void MyRefine2DPanel::OnUpdateUI( wxUpdateUIEvent& event )
{
	// are there enough members in the selected group.
	if (main_frame->current_project.is_open == false)
	{
		RefinementPackageComboBox->Enable(false);
		InputParametersComboBox->Enable(false);
		RefinementRunProfileComboBox->Enable(false);
		ExpertToggleButton->Enable(false);
		StartRefinementButton->Enable(false);
		NumberRoundsSpinCtrl->Enable(false);
		NumberClassesSpinCtrl->Enable(false);

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
		if (running_job == false)
		{
			RefinementRunProfileComboBox->Enable(true);

			ExpertToggleButton->Enable(true);

			if (RefinementPackageComboBox->GetCount() > 0)
			{
				RefinementPackageComboBox->Enable(true);
				InputParametersComboBox->Enable(true);

				if (InputParametersComboBox->GetSelection() == 0) NumberClassesSpinCtrl->Enable(true);
				else NumberClassesSpinCtrl->Enable(false);

				if (PleaseCreateRefinementPackageText->IsShown())
				{
					PleaseCreateRefinementPackageText->Show(false);
					Layout();
				}

			}
			else
			{
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

			NumberRoundsSpinCtrl->Enable(true);


			if (ExpertToggleButton->GetValue() == true)
			{
				if (AutoPercentUsedRadioYes->GetValue() == true)
				{
					PercentUsedStaticText->Enable(false);
					PercentUsedTextCtrl->Enable(false);
				}
				else
				{
					PercentUsedStaticText->Enable(true);
					PercentUsedTextCtrl->Enable(true);
				}


			}

			bool estimation_button_status = false;

			if (RefinementPackageComboBox->GetCount() > 0 )
			{
				if (run_profiles_panel->run_profile_manager.ReturnTotalJobs(RefinementRunProfileComboBox->GetSelection()) > 1 )
				{
					if (RefinementPackageComboBox->GetSelection() != wxNOT_FOUND && InputParametersComboBox->GetSelection() != wxNOT_FOUND)
					{
						estimation_button_status = true;
					}

				}
			}

			StartRefinementButton->Enable(estimation_button_status);
		}
		else
		{
			ExpertToggleButton->Enable(false);
			RefinementPackageComboBox->Enable(false);
			InputParametersComboBox->Enable(false);
			NumberClassesSpinCtrl->Enable(false);
			NumberRoundsSpinCtrl->Enable(false);

			//	RunProfileComboBox->Enable(false);
			//  StartAlignmentButton->SetLabel("Stop Job");
			//  StartAlignmentButton->Enable(true);
		}

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

	}

}


void MyRefine2DPanel::FillRunProfileComboBoxes()
{
	RefinementRunProfileComboBox->FillWithRunProfiles();
}




void MyRefine2DPanel::NewRefinementPackageSelected()
{
	selected_refinement_package = RefinementPackageComboBox->GetSelection();
	FillInputParamsComboBox();
	SetDefaults();
}

void MyRefine2DPanel::SetInfo()
{

	wxLogNull *suppress_png_warnings = new wxLogNull;
	#include "icons/classification_infotext1.cpp"
	wxBitmap class_picture1_bmp = wxBITMAP_PNG_FROM_DATA(classification_infotext1);

	#include "icons/classification_infotext2.cpp"
	wxBitmap class_picture2_bmp = wxBITMAP_PNG_FROM_DATA(classification_infotext2);

	#include "icons/classification_infotext3.cpp"
	wxBitmap class_picture3_bmp = wxBITMAP_PNG_FROM_DATA(classification_infotext3);

	#include "icons/classification_infotext4.cpp"
	wxBitmap class_picture4_bmp = wxBITMAP_PNG_FROM_DATA(classification_infotext4);
	delete suppress_png_warnings;

	InfoText->GetCaret()->Hide();

	InfoText->BeginSuppressUndo();
	InfoText->BeginAlignment(wxTEXT_ALIGNMENT_CENTRE);
	InfoText->BeginBold();
	InfoText->BeginUnderline();
	InfoText->BeginFontSize(14);
	InfoText->WriteText(wxT("2D Classification"));
	InfoText->EndFontSize();
	InfoText->EndBold();
	InfoText->EndUnderline();
	InfoText->Newline();
	InfoText->Newline();
	InfoText->EndAlignment();

	InfoText->BeginAlignment(wxTEXT_ALIGNMENT_LEFT);
	InfoText->WriteText(wxT("2D classification offers a fast and robust way to assess the quality and homogeneity of a dataset. The results of classification can also be used to remove particles belonging to undesirable classes, for example classes with ill-defined features or features that suggest the presence of damaged particles or impurities."));
	InfoText->Newline();
	InfoText->Newline();
	InfoText->EndAlignment();

	InfoText->BeginAlignment(wxTEXT_ALIGNMENT_LEFT);
	InfoText->WriteText(wxT("2D classification typically starts with classes calculated as averages of randomly sampled particles from the dataset. This is shown here for a dataset of VSV polymerase, a 240 kDa protein (Liang et al. 2015):"));
	InfoText->Newline();
	InfoText->Newline();
	InfoText->EndAlignment();

	InfoText->BeginAlignment(wxTEXT_ALIGNMENT_CENTRE);
	InfoText->WriteImage(class_picture1_bmp);
	InfoText->Newline();
	InfoText->Newline();
	InfoText->EndAlignment();


	InfoText->BeginAlignment(wxTEXT_ALIGNMENT_LEFT);
	InfoText->WriteText(wxT("The VSV polymerase dataset was collected after the initial publication in 2015 and contained 84,608 particle images with a pixel size of 0.97 Å, cut out into 400 x 400 pixel boxes. The figures shows the first 24 out of 50 starting class averages. When generating starting classes, the user has to specify the number of classes, for example 50 or 100. Furthermore, the percentage of the dataset used for the calculation has to be specified. In most cases, an average of about 200 images per class is sufficient to obtain reasonable averages (also in later refinement cycles). For example, if a dataset contains 100,000 particle images and the user requests 100 class averages, the number of particle images recommended for generating initial averages would be 100 classes x 200 images/class = 20,000 images. The percentage in this example should therefore be set to 0.2. In the case of VSV polymerase, the percentage was set to 0.12 (an average of 203 images/class)."));
	InfoText->Newline();
	InfoText->Newline();
	InfoText->EndAlignment();

	InfoText->BeginAlignment(wxTEXT_ALIGNMENT_LEFT);
	InfoText->WriteText(wxT("In subsequent iteration cycles, the class averages are refined using a maximum likelihood algorithm (Sigworth 1998, Scheres et al. 2005). It is recommended to keep the percentage of the dataset used in the calculation unchanged for the first 10 cycles, or to increase it somewhat but remain below 100% if class averages appear very noisy. Also, the resolution limit should be set to 8 – 10 Å and the x,y search range should be limited in such a way that most particle displacement from the image center will be within this range. The required range will depend on the way the particles were picked. If particles are well centered after picking, the range can be set between 20 and 40 Å, otherwise a range of 100 Å or more is probably a safer option. Finally, the angular sampling rate must be set by the user. It is rarely required to choose a value below 5 degrees and in most cases 15 degrees is adequate. All four options (percentage, resolution limit, search range, angular sampling rate) can significantly speed up computation and it is therefore worth setting these carefully. For the VSV polymerase dataset, the following class averages were obtained after 9 cycles at 8 Å resolution, a search range of 40 Å and a sampling rate of 15 degrees:"));
	InfoText->Newline();
	InfoText->Newline();
	InfoText->EndAlignment();

	InfoText->BeginAlignment(wxTEXT_ALIGNMENT_CENTRE);
	InfoText->WriteImage(class_picture2_bmp);
	InfoText->Newline();
	InfoText->Newline();
	InfoText->EndAlignment();

	InfoText->BeginAlignment(wxTEXT_ALIGNMENT_LEFT);
	InfoText->WriteText(wxT("This result was obtained in 8 minutes on a single Linux workstation with two 22-core CPUs (Intel Xeon E5-2699 v4 running at 2.2 GHz). After the initial refinement with a smaller percentage of the dataset, the dataset should be refined with a higher percentage and finally including all data, initially at 8 – 10 Å resolution, and perhaps at higher resolution if desired and warranted by the data. This final refinement usually requires less than 10 cycles but users can try more cycles if they still see improvements in the class averages. For the VSV polymerase, 5 cycles with 30% of the dataset yielded:"));
	InfoText->Newline();
	InfoText->Newline();
	InfoText->EndAlignment();

	InfoText->BeginAlignment(wxTEXT_ALIGNMENT_CENTRE);
	InfoText->WriteImage(class_picture3_bmp);
	InfoText->Newline();
	InfoText->Newline();
	InfoText->EndAlignment();

	InfoText->BeginAlignment(wxTEXT_ALIGNMENT_LEFT);
	InfoText->WriteText(wxT("This took 10 minutes on the aforementioned Linux workstation. An additional 5 cycles at 7 Å resolution and using the full dataset took another 35 minutes and yielded the final class averages, many showing clear secondary structure (α-helixes):"));
	InfoText->Newline();
	InfoText->Newline();
	InfoText->EndAlignment();

	InfoText->BeginAlignment(wxTEXT_ALIGNMENT_CENTRE);
	InfoText->WriteImage(class_picture4_bmp);
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
	InfoText->WriteText(wxT("Input Starting References : "));
	InfoText->EndBold();
	InfoText->WriteText(wxT("A set of class averages from a previous run of 2D classification. If no prior class averages are available, the option “New Classification” can be selected, allowing the user to enter the number of desired classes in the next menu."));
	InfoText->Newline();
	InfoText->BeginBold();
	InfoText->WriteText(wxT("No. of Classes : "));
	InfoText->EndBold();
	InfoText->WriteText(wxT("The number of classes that should be generated. This input is only available when starting a fresh classification run."));
	InfoText->Newline();
	InfoText->BeginBold();
	InfoText->WriteText(wxT("No. of Cycles to Run : "));
	InfoText->EndBold();
	InfoText->WriteText(wxT("The number of refinement cycles to run. If the option “Auto Percent Used” is selected, 20 cycles are usually sufficient to generate good class averages. If the user decides to set parameters manually, 5 to 10 cycles are usually sufficient for a particular set of parameters. Several of these shorter runs should be used to obtain final class averages, updating parameters as needed (e.g. Percent Used, see example above)."));
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

	InfoText->BeginBold();
	InfoText->WriteText(wxT("Low/High-Resolution Limit (Å) : "));
	InfoText->EndBold();
	InfoText->WriteText(wxT("The data used for classification is usually bandpass-limited to exclude spurious low-resolution features in the particle background (set by the low-resolution limit) and high-resolution noise (set by the high-resolution limit). It is good practice to set the low-resolution limit to 2.5x the approximate particle mask radius. The high-resolution limit should be selected to remove data with low signal-to-noise ratio, and to help speed up the calculation. For example, setting the high-resolution limit to 8 Å includes signal originating from protein secondary structure that often helps generate recognizable features in the class averages (see example above)."));
	InfoText->Newline();

	InfoText->BeginBold();
	InfoText->WriteText(wxT("Mask Radius (Å) : "));
	InfoText->EndBold();
	InfoText->WriteText(wxT("The radius of the circular mask applied to the input class averages before classification starts. This mask should be sufficiently large to include the largest dimension of the particle. The mask helps remove noise outside the area of the particle."));
	InfoText->Newline();

	InfoText->BeginBold();
	InfoText->WriteText(wxT("Angular Search Step (°) : "));
	InfoText->EndBold();
	InfoText->WriteText(wxT("The angular step used to generate the search grid when marginalizing over the in-plane rotational alignment parameter. The smaller the value, the finer the search grid and the slower the search. It is often sufficient to set the step to 15° as the algorithm varies the starting point of the grid in each refinement cycle, thereby covering intermediate in-plane alignment angles. However, users can try to reduce the step to 5° (smaller is probably not helpful) to see if class averages can be improved further once no further improvement is seen at 15°."));
	InfoText->Newline();

	InfoText->BeginBold();
	InfoText->WriteText(wxT("Search Range in X/Y (Å) : "));
	InfoText->EndBold();
	InfoText->WriteText(wxT("The search can be limited in the X and Y directions (measured from the box center) to ensure that only particles close to the box center are used for classification. A smaller range, for example 20 to 40 Å, can speed up computation. However, the range should be chosen sufficiently generously to capture most particles. If the range of particle displacements from the box center is unknown, start with a larger value, e.g. 100 Å, check the results when the run finishes and reduce the range appropriately."));
	InfoText->Newline();

	InfoText->BeginBold();
	InfoText->WriteText(wxT("Smoothing Factor : "));
	InfoText->EndBold();
	InfoText->WriteText(wxT("A factor that reduces the range of likelihoods used during classification. A reduced range can help prevent the appearance of “empty” classes (no members) early in the classification. Soothing may also suppress some high-resolution noise. The user should try values between 0.1 and 1 if classification suffers from the disappearance of small classes or noisy class averages."));
	InfoText->Newline();

	InfoText->BeginBold();
	InfoText->WriteText(wxT("Exclude Blank Edges? "));
	InfoText->EndBold();
	InfoText->WriteText(wxT("Should particle boxes with blank edges be excluded from classification? Blank edges can be the result of particles selected close to the edges of micrographs. Blank edges can lead to errors in the calculation of the likelihood function, which depends on the noise statistics."));
	InfoText->Newline();

	InfoText->BeginBold();
	InfoText->WriteText(wxT("Auto Percent Used? "));
	InfoText->EndBold();
	InfoText->WriteText(wxT("Should the percent of included particles be adjusted automatically? A classification scheme using initially 300 particles/class, then 30% and then 100% is often sufficient to obtain good classes and this scheme will be used when this option is selected."));
	InfoText->Newline();

	InfoText->BeginBold();
	InfoText->WriteText(wxT("Percent Used : "));
	InfoText->EndBold();
	InfoText->WriteText(wxT("The fraction of the dataset used for classification. Especially in the beginning, classification proceeds more rapidly when only a small number of particles are used per class, e.g. 300 (see example above). Later runs that refine the class averages should use a higher percentage and the final run(s) should use all the data. This option is only available when “Auto Percent Used” is not selected."));
	InfoText->Newline();
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
	InfoText->WriteText(wxT("Sigworth, F. J.,"));
	InfoText->EndBold();
	InfoText->WriteText(wxT(" 1998. A maximum-likelihood approach to single-particle image refinement. J. Struct. Biol. 122, 328-339."));
	InfoText->BeginURL("http://dx.doi.org/10.1006/jsbi.1998.4014");
	InfoText->BeginUnderline();
	InfoText->BeginTextColour(*wxBLUE);
	InfoText->WriteText(wxT("dio:10.1006/jsbi.1998.4014"));
	InfoText->EndURL();
	InfoText->EndTextColour();
	InfoText->EndUnderline();
	InfoText->EndAlignment();
	InfoText->Newline();
	InfoText->Newline();

	InfoText->BeginAlignment(wxTEXT_ALIGNMENT_LEFT);
	InfoText->BeginBold();
	InfoText->WriteText(wxT("Scheres, S. H. W., Valle, M., Nuñez, R., Sorzano, C. O. S., Marabini, R., Herman, G. T., and Jose-Maria Carazo, J.-M.,"));
	InfoText->EndBold();
	InfoText->WriteText(wxT(" 2005. Maximum-likelihood multi-reference refinement for electron microscopy images. J. Mol. Biol. 348, 139–149."));
	InfoText->BeginURL("http://dx.doi.org/10.1016/j.jmb.2005.02.031");
	InfoText->BeginUnderline();
	InfoText->BeginTextColour(*wxBLUE);
	InfoText->WriteText(wxT("doi:10.1016/j.jmb.2005.02.031"));
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

void MyRefine2DPanel::OnInfoURL(wxTextUrlEvent& event)
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

void MyRefine2DPanel::StartClassificationClick( wxCommandEvent& event )
{
	my_classification_manager.BeginRefinementCycle();
}

void MyRefine2DPanel::SetDefaults()
{
	if (RefinementPackageComboBox->GetCount() > 0)
	{
		ExpertPanel->Freeze();
		if (InputParametersComboBox->GetSelection() > 0)
		{
			NumberClassesSpinCtrl->SetValue(refinement_package_asset_panel->ReturnPointerToShortClassificationInfoByClassificationID(refinement_package_asset_panel->all_refinement_packages.Item(RefinementPackageComboBox->GetSelection()).classification_ids.Item(InputParametersComboBox->GetSelection() - 1))->number_of_classes);
		}
		else
		{
			if (InputParametersComboBox->GetSelection() == 0)
			{
				int calculated_number_of_classes = 	refinement_package_asset_panel->all_refinement_packages.Item(RefinementPackageComboBox->GetSelection()).contained_particles.GetCount() / 300;

				if (calculated_number_of_classes > 50) calculated_number_of_classes = 50;
				else
				if (calculated_number_of_classes > 40) calculated_number_of_classes = 40;
				else
				if (calculated_number_of_classes > 30) calculated_number_of_classes = 30;
				else
				if (calculated_number_of_classes > 20) calculated_number_of_classes = 20;
				else
				if (calculated_number_of_classes > 10) calculated_number_of_classes = 10;
				else
				calculated_number_of_classes = 5;

				NumberClassesSpinCtrl->SetValue(calculated_number_of_classes);
			}
		}

		NumberRoundsSpinCtrl->SetValue(20);

		LowResolutionLimitTextCtrl->SetValue("300.00");
		HighResolutionLimitTextCtrl->SetValue("8.00");

		float local_mask_radius = refinement_package_asset_panel->all_refinement_packages.Item(RefinementPackageComboBox->GetSelection()).estimated_particle_size_in_angstroms * 0.6;
		MaskRadiusTextCtrl->SetValue(wxString::Format("%.2f", local_mask_radius));
		AngularStepTextCtrl->SetValue("15.00");

		float search_range = refinement_package_asset_panel->all_refinement_packages.Item(RefinementPackageComboBox->GetSelection()).estimated_particle_size_in_angstroms * 0.5;

		SearchRangeXTextCtrl->SetValue(wxString::Format("%.2f", search_range));
		SearchRangeYTextCtrl->SetValue(wxString::Format("%.2f", search_range));

		SmoothingFactorTextCtrl->SetValue("1.00");
		ExcludeBlankEdgesYesRadio->SetValue(true);
		AutoPercentUsedRadioYes->SetValue(true);
		PercentUsedTextCtrl->SetValue("100.00");
		ExpertPanel->Thaw();
	}
}

void MyRefine2DPanel::ResetAllDefaultsClick( wxCommandEvent& event )
{
	// TODO : should probably check that the user hasn't changed the defaults yet in the future
	SetDefaults();
}

void MyRefine2DPanel::TerminateButtonClick( wxCommandEvent& event )
{
	main_frame->job_controller.KillJob(my_job_id);

	WriteBlueText("Terminated Job");
	TimeRemainingText->SetLabel("Time Remaining : Terminated");
	CancelAlignmentButton->Show(false);
	FinishButton->Show(true);
	ProgressPanel->Layout();
}

void MyRefine2DPanel::FinishButtonClick( wxCommandEvent& event )
{
	ProgressBar->SetValue(0);
	TimeRemainingText->SetLabel("Time Remaining : ???h:??m:??s");
    CancelAlignmentButton->Show(true);
	FinishButton->Show(false);


	ProgressPanel->Show(false);
	StartPanel->Show(true);
	OutputTextPanel->Show(false);
	PlotPanel->Show(false);
	output_textctrl->Clear();
	ResultDisplayPanel->Show(false);
	InfoPanel->Show(true);
	InputParamsPanel->Show(true);

	if (ExpertToggleButton->GetValue() == true) ExpertPanel->Show(true);
	else ExpertPanel->Show(false);
	running_job = false;
	Layout();

	ResultDisplayPanel->Clear();

	if (my_classification_manager.number_of_rounds_run > 0) FillInputParamsComboBox();

	//CTFResultsPanel->CTF2DResultsPanel->should_show = false;
	//CTFResultsPanel->CTF2DResultsPanel->Refresh();

}

void MyRefine2DPanel::OnRefinementPackageComboBox( wxCommandEvent& event )
{

	NewRefinementPackageSelected();

}

void MyRefine2DPanel::OnInputParametersComboBox( wxCommandEvent& event )
{

}







/*


void MyRefine3DPanel::OnHighResLimitChange( wxCommandEvent& event )
{
	float global_angular_step = CalculateAngularStep(HighResolutionLimitTextCtrl->ReturnValue(), MaskRadiusTextCtrl->ReturnValue());
	AngularStepTextCtrl->SetValue(wxString::Format("%.2f", global_angular_step));
	ClassificationHighResLimitTextCtrl->SetValue(wxString::Format("%.2f", HighResolutionLimitTextCtrl->ReturnValue()));
}





void MyRefine3DPanel::NewRefinementPackageSelected()
{
	selected_refinement_package = RefinementPackageComboBox->GetSelection();
	FillInputParamsComboBox();
	SetDefaults();
	//wxPrintf("New Refinement Package Selection\n");

}


void MyRefine3DPanel::StartRefinementClick( wxCommandEvent& event )
{
	my_refinement_manager.BeginRefinementCycle();
}


void MyRefine3DPanel::OnJobSocketEvent(wxSocketEvent& event)
{
      SETUP_SOCKET_CODES

	  wxString s = _("OnSocketEvent: ");
	  wxSocketBase *sock = event.GetSocket();
	  sock->SetFlags(wxSOCKET_BLOCK | wxSOCKET_WAITALL);

	//  MyDebugAssertTrue(sock == main_frame->job_controller.job_list[my_job_id].socket, "Socket event from Non conduit socket??");

	  // First, print a message
	  switch(event.GetSocketEvent())
	  {
	    case wxSOCKET_INPUT : s.Append(_("wxSOCKET_INPUT\n")); break;
	    case wxSOCKET_LOST  : s.Append(_("wxSOCKET_LOST\n")); break;
	    default             : s.Append(_("Unexpected event !\n")); break;
	  }

	  //m_text->AppendText(s);

	  //MyDebugPrint(s);

	  // Now we process the event
	  switch(event.GetSocketEvent())
	  {
	    case wxSOCKET_INPUT:
	    {
	      // We disable input events, so that the test doesn't trigger
	      // wxSocketEvent again.
	      sock->SetNotify(wxSOCKET_LOST_FLAG);
	      ReadFromSocket(sock, &socket_input_buffer, SOCKET_CODE_SIZE);


	      if (memcmp(socket_input_buffer, socket_send_job_details, SOCKET_CODE_SIZE) == 0) // identification
	      {
	    	  // send the job details..

	    	  //wxPrintf("Sending Job Details...\n");
	    	  my_job_package.SendJobPackage(sock);

	      }
	      else
	      if (memcmp(socket_input_buffer, socket_i_have_an_error, SOCKET_CODE_SIZE) == 0) // identification
	      {

	    	  wxString error_message;
   			  error_message = ReceivewxStringFromSocket(sock);

   			  WriteErrorText(error_message);
    	  }
	      else
	      if (memcmp(socket_input_buffer, socket_i_have_info, SOCKET_CODE_SIZE) == 0) // identification
	      {

	    	  wxString info_message;
   			  info_message = ReceivewxStringFromSocket(sock);

   			  WriteInfoText(info_message);
    	  }
	      else
	      if (memcmp(socket_input_buffer, socket_job_finished, SOCKET_CODE_SIZE) == 0) // identification
	 	  {
	 		 // which job is finished?

	 		 int finished_job;
	 		 ReadFromSocket(sock, &finished_job, 4);

	 		 my_job_tracker.MarkJobFinished();

//	 		 if (my_job_tracker.ShouldUpdate() == true) UpdateProgressBar();
	 		 //WriteInfoText(wxString::Format("Job %i has finished!", finished_job));
	 	  }
	      else
	      if (memcmp(socket_input_buffer, socket_job_result, SOCKET_CODE_SIZE) == 0) // identification
	 	  {
	    	  JobResult temp_result;
	    	  temp_result.ReceiveFromSocket(sock);

	    	  // send the result to the

	    	  my_refinement_manager.ProcessJobResult(&temp_result);
	    	  wxPrintf("Warning: Received socket_job_result - should this happen?");

	 	  }
	      else
	      if (memcmp(socket_input_buffer, socket_job_result_queue, SOCKET_CODE_SIZE) == 0) // identification
	 	  {
	    	  ArrayofJobResults temp_queue;
	    	  ReceiveResultQueueFromSocket(sock, temp_queue);

	    	  for (int counter = 0; counter < temp_queue.GetCount(); counter++)
	    	  {
	    		  my_refinement_manager.ProcessJobResult(&temp_queue.Item(counter));
	    	  }
	 	  }
	      else
		  if (memcmp(socket_input_buffer, socket_number_of_connections, SOCKET_CODE_SIZE) == 0) // identification
		  {
			  // how many connections are there?

			  int number_of_connections;
			  ReadFromSocket(sock, &number_of_connections, 4);

              my_job_tracker.AddConnection();

    //          if (graph_is_hidden == true) ProgressBar->Pulse();

              //WriteInfoText(wxString::Format("There are now %i connections\n", number_of_connections));

              // send the info to the gui

              int total_processes = my_job_package.my_profile.ReturnTotalJobs();

		 	  if (number_of_connections == total_processes) WriteInfoText(wxString::Format("All %i processes are connected.", number_of_connections));

			  if (length_of_process_number == 6) NumberConnectedText->SetLabel(wxString::Format("%6i / %6i processes connected.", number_of_connections, total_processes));
			  else
			  if (length_of_process_number == 5) NumberConnectedText->SetLabel(wxString::Format("%5i / %5i processes connected.", number_of_connections, total_processes));
		      else
			  if (length_of_process_number == 4) NumberConnectedText->SetLabel(wxString::Format("%4i / %4i processes connected.", number_of_connections, total_processes));
			  else
			  if (length_of_process_number == 3) NumberConnectedText->SetLabel(wxString::Format("%3i / %3i processes connected.", number_of_connections, total_processes));
			  else
			  if (length_of_process_number == 2) NumberConnectedText->SetLabel(wxString::Format("%2i / %2i processes connected.", number_of_connections, total_processes));
			  else
		      NumberConnectedText->SetLabel(wxString::Format("%1i / %1i processes connected.", number_of_connections, total_processes));
		  }
	      else
		  if (memcmp(socket_input_buffer, socket_all_jobs_finished, SOCKET_CODE_SIZE) == 0) // identification
		  {
			  my_refinement_manager.ProcessAllJobsFinished();
		  }

	      // Enable input events again.

	      sock->SetNotify(wxSOCKET_LOST_FLAG | wxSOCKET_INPUT_FLAG);

	      break;
	    }


	    case wxSOCKET_LOST:
	    {

	    	//MyDebugPrint("Socket Disconnected!!\n");
	        sock->Destroy();
	        break;
	    }
	    default: ;
	  }

}

*/

void ClassificationManager::SetParent(MyRefine2DPanel *wanted_parent)
{
	my_parent = wanted_parent;
}

void ClassificationManager::BeginRefinementCycle()
{

	start_with_random = false;

	number_of_rounds_run = 0;
	number_of_rounds_to_run = my_parent->NumberRoundsSpinCtrl->GetValue();

	current_refinement_package_asset_id = refinement_package_asset_panel->all_refinement_packages.Item(my_parent->RefinementPackageComboBox->GetSelection()).asset_id;

	my_parent->PlotPanel->Clear();
	my_parent->PlotPanel->my_notebook->SetSelection(0);

	if (my_parent->InputParametersComboBox->GetSelection() == 0)
	{
		start_with_random = true;
		current_input_classification_id = -1;
		input_classification = NULL;
		output_classification = new Classification;

		RunInitialStartJob();
	}
	else
	{
		current_input_classification_id = refinement_package_asset_panel->all_refinement_packages.Item(my_parent->RefinementPackageComboBox->GetSelection()).classification_ids[my_parent->InputParametersComboBox->GetSelection() - 1];
		first_round_id = current_input_classification_id;
		wxPrintf("classification list id %i = %li\n", my_parent->InputParametersComboBox->GetSelection() - 1, current_input_classification_id);
		wxPrintf("Getting classification for classification id %li\n", current_input_classification_id);
		input_classification = main_frame->current_project.database.GetClassificationByID(current_input_classification_id);
		output_classification = new Classification;
		my_parent->ResultDisplayPanel->OpenFile(input_classification->class_average_file, wxString::Format("Class #%li (Start Ref.)", input_classification->classification_id));
		RunRefinementJob();
	}

}

void ClassificationManager::RunInitialStartJob()
{
	running_job_type = STARTUP;
	number_of_received_particle_results = 0;

	output_classification->classification_id = main_frame->current_project.database.ReturnHighestClassificationID() + 1;
	output_classification->refinement_package_asset_id = current_refinement_package_asset_id;
	output_classification->name = wxString::Format("Random Start #%li", output_classification->classification_id);
	output_classification->class_average_file = main_frame->current_project.class_average_directory.GetFullPath() + wxString::Format("/class_averages_%.4li.mrc", output_classification->classification_id);
	output_classification->classification_was_imported_or_generated = true;
	output_classification->datetime_of_run = wxDateTime::Now();
	output_classification->starting_classification_id = -1;
	output_classification->number_of_particles = refinement_package_asset_panel->all_refinement_packages.Item(my_parent->RefinementPackageComboBox->GetSelection()).contained_particles.GetCount();
	output_classification->number_of_classes = my_parent->NumberClassesSpinCtrl->GetValue();
	output_classification->low_resolution_limit = my_parent->LowResolutionLimitTextCtrl->ReturnValue();
	output_classification->high_resolution_limit = my_parent->HighResolutionLimitTextCtrl->ReturnValue();
	output_classification->mask_radius = my_parent->MaskRadiusTextCtrl->ReturnValue();
	output_classification->angular_search_step = my_parent->AngularStepTextCtrl->ReturnValue();
	output_classification->search_range_x = my_parent->SearchRangeXTextCtrl->ReturnValue();
	output_classification->search_range_y = my_parent->SearchRangeYTextCtrl->ReturnValue();
	output_classification->smoothing_factor = my_parent->SmoothingFactorTextCtrl->ReturnValue();
	output_classification->exclude_blank_edges = my_parent->ExcludeBlankEdgesYesRadio->GetValue();
	output_classification->auto_percent_used = my_parent->AutoPercentUsedRadioYes->GetValue();

	output_classification->percent_used = (float(output_classification->number_of_classes * 300) / float(output_classification->number_of_particles)) * 100.0;
	if (output_classification->percent_used > 100.0) output_classification->percent_used = 100.0;

	output_classification->SizeAndFillWithEmpty(output_classification->number_of_particles);

	for (long counter = 0; counter < output_classification->number_of_particles; counter++)
	{
		output_classification->classification_results[counter].position_in_stack = counter + 1;
	}

	wxString input_parameter_file = output_classification->WriteFrealignParameterFiles(main_frame->current_project.parameter_file_directory.GetFullPath() + "/classification_input_par", &refinement_package_asset_panel->all_refinement_packages.Item(my_parent->RefinementPackageComboBox->GetSelection()));
	my_parent->my_job_package.Reset(run_profiles_panel->run_profile_manager.run_profiles[my_parent->RefinementRunProfileComboBox->GetSelection()], "refine2d", 1);

	wxString input_particle_images =  refinement_package_asset_panel->all_refinement_packages.Item(my_parent->RefinementPackageComboBox->GetSelection()).stack_filename;
	wxString input_class_averages = "/dev/null";
	wxString output_parameter_file = "/dev/null";
	wxString output_class_averages = output_classification->class_average_file;
	int number_of_classes = output_classification->number_of_classes;
	int first_particle = 1;
	int last_particle = output_classification->number_of_particles;
	float percent_used = output_classification->percent_used / 100.00;
	float pixel_size = refinement_package_asset_panel->all_refinement_packages.Item(my_parent->RefinementPackageComboBox->GetSelection()).contained_particles[0].pixel_size;
	float voltage_kV = refinement_package_asset_panel->all_refinement_packages.Item(my_parent->RefinementPackageComboBox->GetSelection()).contained_particles[0].microscope_voltage;
	float spherical_aberration_mm = refinement_package_asset_panel->all_refinement_packages.Item(my_parent->RefinementPackageComboBox->GetSelection()).contained_particles[0].spherical_aberration;
	float amplitude_contrast = refinement_package_asset_panel->all_refinement_packages.Item(my_parent->RefinementPackageComboBox->GetSelection()).contained_particles[0].amplitude_contrast;
	float mask_radius = my_parent->MaskRadiusTextCtrl->ReturnValue();
	float low_resolution_limit =output_classification->low_resolution_limit;
	float high_resolution_limit = output_classification->high_resolution_limit;
	float angular_step = output_classification->angular_search_step;
	float max_search_x = output_classification->search_range_x;
	float max_search_y = output_classification->search_range_y;
	float smoothing_factor = output_classification->smoothing_factor;
	int padding_factor = 2;
	bool normalize_particles = true;
	bool invert_contrast = refinement_package_asset_panel->all_refinement_packages.Item(my_parent->RefinementPackageComboBox->GetSelection()).stack_has_white_protein;
	bool exclude_blank_edges = output_classification->exclude_blank_edges;
	bool dump_arrays = false;
	wxString dump_file = "/dev/null";


	my_parent->my_job_package.AddJob("tttttiiiffffffffffffibbbbt",	input_particle_images.ToUTF8().data(),
																	input_parameter_file.ToUTF8().data(),
																	input_class_averages.ToUTF8().data(),
																	output_parameter_file.ToUTF8().data(),
																	output_class_averages.ToUTF8().data(),
																	number_of_classes,
																	first_particle,
																	last_particle,
																	percent_used,
																	pixel_size,
																	voltage_kV,
																	spherical_aberration_mm,
																	amplitude_contrast,
																	mask_radius,
																	low_resolution_limit,
																	high_resolution_limit,
																	angular_step,
																	max_search_x,
																	max_search_y,
																	smoothing_factor,
																	padding_factor,
																	normalize_particles,
																	invert_contrast,
																	exclude_blank_edges,
																	dump_arrays,
																	dump_file.ToUTF8().data());


	my_parent->WriteBlueText("Creating Initial References...");
	current_job_id = main_frame->job_controller.AddJob(my_parent, run_profiles_panel->run_profile_manager.run_profiles[my_parent->RefinementRunProfileComboBox->GetSelection()].manager_command, run_profiles_panel->run_profile_manager.run_profiles[my_parent->RefinementRunProfileComboBox->GetSelection()].gui_address);
	my_parent->my_job_id = current_job_id;

	if (current_job_id != -1)
	{
		long number_of_refinement_processes;
        if (my_parent->my_job_package.number_of_jobs + 1 < my_parent->my_job_package.my_profile.ReturnTotalJobs()) number_of_refinement_processes = my_parent->my_job_package.number_of_jobs + 1;
        else number_of_refinement_processes =  my_parent->my_job_package.my_profile.ReturnTotalJobs();

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
		my_parent->InputParamsPanel->Show(false);
		my_parent->OutputTextPanel->Show(true);
		//my_parent->PlotPanel->Show(true);
 		//my_parent->AngularPlotPanel->Show(true);

		my_parent->ExpertToggleButton->Enable(false);
		my_parent->RefinementPackageComboBox->Enable(false);
		my_parent->InputParametersComboBox->Enable(false);



		my_parent->TimeRemainingText->SetLabel("Time Remaining : ???h:??m:??s");
		my_parent->Layout();
		my_parent->running_job = true;
		my_parent->my_job_tracker.StartTracking(my_parent->my_job_package.number_of_jobs);

	}

	my_parent->ProgressBar->Pulse();

}

void ClassificationManager::RunRefinementJob()
{
	long number_of_refinement_jobs;
	int number_of_refinement_processes;
	float current_particle_counter;
	long number_of_particles;
	float particles_per_job;
	int job_counter;

	running_job_type = REFINEMENT;
	number_of_received_particle_results = 0;

	output_classification->classification_id = main_frame->current_project.database.ReturnHighestClassificationID() + 1;
	output_classification->refinement_package_asset_id = current_refinement_package_asset_id;
	output_classification->name = wxString::Format("Classification #%li (Start #%li, Round %i)", output_classification->classification_id, first_round_id, number_of_rounds_run + 1);
	output_classification->class_average_file = main_frame->current_project.class_average_directory.GetFullPath() + wxString::Format("/class_averages_%.4li.mrc", output_classification->classification_id);
	output_classification->classification_was_imported_or_generated = false;
	output_classification->datetime_of_run = wxDateTime::Now();
	output_classification->starting_classification_id = input_classification->classification_id;
	output_classification->number_of_particles = refinement_package_asset_panel->all_refinement_packages.Item(my_parent->RefinementPackageComboBox->GetSelection()).contained_particles.GetCount();
	output_classification->number_of_classes = input_classification->number_of_classes;
	output_classification->low_resolution_limit = my_parent->LowResolutionLimitTextCtrl->ReturnValue();
	output_classification->high_resolution_limit = my_parent->HighResolutionLimitTextCtrl->ReturnValue();
	output_classification->mask_radius = my_parent->MaskRadiusTextCtrl->ReturnValue();
	output_classification->angular_search_step = my_parent->AngularStepTextCtrl->ReturnValue();
	output_classification->search_range_x = my_parent->SearchRangeXTextCtrl->ReturnValue();
	output_classification->search_range_y = my_parent->SearchRangeYTextCtrl->ReturnValue();
	output_classification->smoothing_factor = my_parent->SmoothingFactorTextCtrl->ReturnValue();
	output_classification->exclude_blank_edges = my_parent->ExcludeBlankEdgesYesRadio->GetValue();
	output_classification->auto_percent_used = my_parent->AutoPercentUsedRadioYes->GetValue();

	if (output_classification->auto_percent_used == true)
	{
		if (number_of_rounds_to_run < 10) output_classification->percent_used = 100.0;
		else
		if (number_of_rounds_to_run < 20)
		{
			if (number_of_rounds_run < 5)
			{
				output_classification->percent_used = (float(output_classification->number_of_classes * 300) / float(output_classification->number_of_particles)) * 100.0;
				if (output_classification->percent_used > 100.0) output_classification->percent_used = 100.0;
			}
			else
			if (number_of_rounds_run < number_of_rounds_to_run - 5)
			{
				output_classification->percent_used = (float(output_classification->number_of_classes * 300) / float(output_classification->number_of_particles)) * 100.0;
				if (output_classification->percent_used > 100.0) output_classification->percent_used = 100.0;
				else
				if (output_classification->percent_used < 30) output_classification->percent_used = 30.0;
			}
			else
			output_classification->percent_used = 100.0;
		}
		else
		if (number_of_rounds_to_run < 30)
		{
			if (number_of_rounds_run < 10)
			{
				output_classification->percent_used = (float(output_classification->number_of_classes * 300) / float(output_classification->number_of_particles)) * 100.0;
				if (output_classification->percent_used > 100.0) output_classification->percent_used = 100.0;
			}
			else
			if (number_of_rounds_run < number_of_rounds_to_run - 5)
			{
				output_classification->percent_used = (float(output_classification->number_of_classes * 300) / float(output_classification->number_of_particles)) * 100.0;
				if (output_classification->percent_used > 100.0) output_classification->percent_used = 100.0;
				else
				if (output_classification->percent_used < 30) output_classification->percent_used = 30.0;
			}
			else
			output_classification->percent_used = 100.0;
		}
		else
		{
			if (number_of_rounds_run < 15)
			{
				output_classification->percent_used = (float(output_classification->number_of_classes * 300) / float(output_classification->number_of_particles)) * 100.0;
				if (output_classification->percent_used > 100.0) output_classification->percent_used = 100.0;
			}
			else
			if (number_of_rounds_run < number_of_rounds_to_run - 5)
			{
				output_classification->percent_used = (float(output_classification->number_of_classes * 300) / float(output_classification->number_of_particles)) * 100.0;
				if (output_classification->percent_used > 100.0) output_classification->percent_used = 100.0;
				else
				if (output_classification->percent_used < 30) output_classification->percent_used = 30.0;
			}
			else
			output_classification->percent_used = 100.0;
		}
	}
	else
	{
		output_classification->percent_used = my_parent->PercentUsedTextCtrl->ReturnValue();
	}

	output_classification->SizeAndFillWithEmpty(output_classification->number_of_particles);

	for (long counter = 0; counter < output_classification->number_of_particles; counter++)
	{
		output_classification->classification_results[counter].position_in_stack = counter + 1;
	}

	wxString input_parameter_file = input_classification->WriteFrealignParameterFiles(main_frame->current_project.parameter_file_directory.GetFullPath() + "/classification_input_par", &refinement_package_asset_panel->all_refinement_packages.Item(my_parent->RefinementPackageComboBox->GetSelection()));

	number_of_refinement_processes = run_profiles_panel->run_profile_manager.run_profiles[my_parent->RefinementRunProfileComboBox->GetSelection()].ReturnTotalJobs();
	number_of_refinement_jobs = number_of_refinement_processes - 1;
	number_of_particles = output_classification->number_of_particles;

	if (number_of_particles - number_of_refinement_jobs < number_of_refinement_jobs) particles_per_job = 1.0;
	else particles_per_job = float(number_of_particles - number_of_refinement_jobs) / float(number_of_refinement_jobs);

	my_parent->my_job_package.Reset(run_profiles_panel->run_profile_manager.run_profiles[my_parent->RefinementRunProfileComboBox->GetSelection()], "refine2d", number_of_refinement_jobs);
	current_particle_counter = 1.0;

	for (job_counter = 0; job_counter < number_of_refinement_jobs; job_counter++)
	{

		wxString input_particle_images =  refinement_package_asset_panel->all_refinement_packages.Item(my_parent->RefinementPackageComboBox->GetSelection()).stack_filename;
		wxString input_class_averages = input_classification->class_average_file;

		wxString output_parameter_file = "/dev/null";
		wxString output_class_averages = main_frame->current_project.class_average_directory.GetFullPath() + wxString::Format("/class_averages_%li.mrc", output_classification->classification_id);
		int number_of_classes = 0;

		long	 first_particle							= myroundint(current_particle_counter);
		current_particle_counter += particles_per_job;
		if (current_particle_counter > number_of_particles) current_particle_counter = number_of_particles;
		long	 last_particle							= myroundint(current_particle_counter);
		current_particle_counter++;

		//output_parameter_file = wxString::Format("/tmp/out_%li_%li.par", first_particle, last_particle);
		output_parameter_file = "/dev/null";

		float percent_used = output_classification->percent_used / 100.00;
		float pixel_size = refinement_package_asset_panel->all_refinement_packages.Item(my_parent->RefinementPackageComboBox->GetSelection()).contained_particles[0].pixel_size;
		float voltage_kV = refinement_package_asset_panel->all_refinement_packages.Item(my_parent->RefinementPackageComboBox->GetSelection()).contained_particles[0].microscope_voltage;
		float spherical_aberration_mm = refinement_package_asset_panel->all_refinement_packages.Item(my_parent->RefinementPackageComboBox->GetSelection()).contained_particles[0].spherical_aberration;
		float amplitude_contrast = refinement_package_asset_panel->all_refinement_packages.Item(my_parent->RefinementPackageComboBox->GetSelection()).contained_particles[0].amplitude_contrast;
		float mask_radius = my_parent->MaskRadiusTextCtrl->ReturnValue();
		float low_resolution_limit =output_classification->low_resolution_limit;
		float high_resolution_limit = output_classification->high_resolution_limit;
		float angular_step = output_classification->angular_search_step;
		float max_search_x = output_classification->search_range_x;
		float max_search_y = output_classification->search_range_y;
		float smoothing_factor = output_classification->smoothing_factor;
		int padding_factor = 2;
		bool normalize_particles = true;
		bool invert_contrast = refinement_package_asset_panel->all_refinement_packages.Item(my_parent->RefinementPackageComboBox->GetSelection()).stack_has_white_protein;
		bool exclude_blank_edges = output_classification->exclude_blank_edges;
		bool dump_arrays = true;
		wxString dump_file = main_frame->ReturnRefine2DScratchDirectory() + wxString::Format("/class_dump_file_%li_%i.dump", output_classification->classification_id, job_counter +1);


		my_parent->my_job_package.AddJob("tttttiiiffffffffffffibbbbt",	input_particle_images.ToUTF8().data(),
																		input_parameter_file.ToUTF8().data(),
																		input_class_averages.ToUTF8().data(),
																		output_parameter_file.ToUTF8().data(),
																		output_class_averages.ToUTF8().data(),
																		number_of_classes,
																		first_particle,
																		last_particle,
																		percent_used,
																		pixel_size,
																		voltage_kV,
																		spherical_aberration_mm,
																		amplitude_contrast,
																		mask_radius,
																		low_resolution_limit,
																		high_resolution_limit,
																		angular_step,
																		max_search_x,
																		max_search_y,
																		smoothing_factor,
																		padding_factor,
																		normalize_particles,
																		invert_contrast,
																		exclude_blank_edges,
																		dump_arrays,
																		dump_file.ToUTF8().data());
	}

	my_parent->WriteBlueText(wxString::Format("Running refinement round %2i of %2i\n", number_of_rounds_run + 1, number_of_rounds_to_run));
	if (my_parent->AutoPercentUsedRadioYes->GetValue() == true)
	{
		my_parent->WriteInfoText(wxString::Format("Using %.0f %% of the particles (%i per class)", output_classification->percent_used, myroundint((float(output_classification->number_of_particles) * output_classification->percent_used * 0.01) / float(output_classification->number_of_classes))));

	}

	current_job_id = main_frame->job_controller.AddJob(my_parent, run_profiles_panel->run_profile_manager.run_profiles[my_parent->RefinementRunProfileComboBox->GetSelection()].manager_command, run_profiles_panel->run_profile_manager.run_profiles[my_parent->RefinementRunProfileComboBox->GetSelection()].gui_address);
	my_parent->my_job_id = current_job_id;

	if (current_job_id != -1)
	{
		long number_of_refinement_processes;
        if (my_parent->my_job_package.number_of_jobs + 1 < my_parent->my_job_package.my_profile.ReturnTotalJobs()) number_of_refinement_processes = my_parent->my_job_package.number_of_jobs + 1;
        else number_of_refinement_processes =  my_parent->my_job_package.my_profile.ReturnTotalJobs();


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
		my_parent->InputParamsPanel->Show(false);
		my_parent->OutputTextPanel->Show(true);
		//y_parent->PlotPanel->Show(true);
	 	//my_parent->AngularPlotPanel->Show(true);
		my_parent->ResultDisplayPanel->Show(true);

		my_parent->ExpertToggleButton->Enable(false);
		my_parent->RefinementPackageComboBox->Enable(false);
		my_parent->InputParametersComboBox->Enable(false);

		my_parent->TimeRemainingText->SetLabel("Time Remaining : ???h:??m:??s");
		my_parent->Layout();
		my_parent->running_job = true;
		my_parent->my_job_tracker.StartTracking(my_parent->my_job_package.number_of_jobs);

	}
	my_parent->ProgressBar->Pulse();

}


void MyRefine2DPanel::OnJobSocketEvent(wxSocketEvent& event)
{
      SETUP_SOCKET_CODES

	  wxString s = _("OnSocketEvent: ");
	  wxSocketBase *sock = event.GetSocket();
	  sock->SetFlags(wxSOCKET_BLOCK | wxSOCKET_WAITALL);

	//  MyDebugAssertTrue(sock == main_frame->job_controller.job_list[my_job_id].socket, "Socket event from Non conduit socket??");

	  // First, print a message
	  switch(event.GetSocketEvent())
	  {
	    case wxSOCKET_INPUT : s.Append(_("wxSOCKET_INPUT\n")); break;
	    case wxSOCKET_LOST  : s.Append(_("wxSOCKET_LOST\n")); break;
	    default             : s.Append(_("Unexpected event !\n")); break;
	  }

	  //m_text->AppendText(s);

	  //MyDebugPrint(s);

	  // Now we process the event
	  switch(event.GetSocketEvent())
	  {
	    case wxSOCKET_INPUT:
	    {
	      // We disable input events, so that the test doesn't trigger
	      // wxSocketEvent again.
	      sock->SetNotify(wxSOCKET_LOST_FLAG);
	      ReadFromSocket(sock, &socket_input_buffer, SOCKET_CODE_SIZE);


	      if (memcmp(socket_input_buffer, socket_send_job_details, SOCKET_CODE_SIZE) == 0) // identification
	      {
	    	  // send the job details..

	    	  //wxPrintf("Sending Job Details...\n");
	    	  my_job_package.SendJobPackage(sock);

	      }
	      else
	      if (memcmp(socket_input_buffer, socket_i_have_an_error, SOCKET_CODE_SIZE) == 0) // identification
	      {

	    	  wxString error_message;
   			  error_message = ReceivewxStringFromSocket(sock);

   			  WriteErrorText(error_message);
    	  }
	      else
	      if (memcmp(socket_input_buffer, socket_i_have_info, SOCKET_CODE_SIZE) == 0) // identification
	      {

	    	  wxString info_message;
   			  info_message = ReceivewxStringFromSocket(sock);

   			  WriteInfoText(info_message);
    	  }
	      else
	      if (memcmp(socket_input_buffer, socket_job_finished, SOCKET_CODE_SIZE) == 0) // identification
	 	  {
	 		 // which job is finished?

	 		 int finished_job;
	 		 ReadFromSocket(sock, &finished_job, 4);

	 		 my_job_tracker.MarkJobFinished();

//	 		 if (my_job_tracker.ShouldUpdate() == true) UpdateProgressBar();
	 		 //WriteInfoText(wxString::Format("Job %i has finished!", finished_job));
	 	  }
	      else
	      if (memcmp(socket_input_buffer, socket_job_result, SOCKET_CODE_SIZE) == 0) // identification
	 	  {
	    	  JobResult temp_result;
	    	  temp_result.ReceiveFromSocket(sock);

	    	  // send the result to the

	    	  my_classification_manager.ProcessJobResult(&temp_result);
	    	  wxPrintf("Warning: Received socket_job_result - should this happen?");

	 	  }
	      else
	      if (memcmp(socket_input_buffer, socket_job_result_queue, SOCKET_CODE_SIZE) == 0) // identification
	 	  {
	    	  ArrayofJobResults temp_queue;
	    	  ReceiveResultQueueFromSocket(sock, temp_queue);

	    	  for (int counter = 0; counter < temp_queue.GetCount(); counter++)
	    	  {
	    		  my_classification_manager.ProcessJobResult(&temp_queue.Item(counter));
	    	  }
	 	  }
	      else
		  if (memcmp(socket_input_buffer, socket_number_of_connections, SOCKET_CODE_SIZE) == 0) // identification
		  {
			  // how many connections are there?

			  int number_of_connections;
			  ReadFromSocket(sock, &number_of_connections, 4);

              my_job_tracker.AddConnection();

    //          if (graph_is_hidden == true) ProgressBar->Pulse();

              //WriteInfoText(wxString::Format("There are now %i connections\n", number_of_connections));

              // send the info to the gui

              int total_processes;
              if (my_job_package.number_of_jobs + 1 < my_job_package.my_profile.ReturnTotalJobs()) total_processes = my_job_package.number_of_jobs + 1;
              else total_processes =  my_job_package.my_profile.ReturnTotalJobs();

		 	  if (number_of_connections == total_processes) WriteInfoText(wxString::Format("All %i processes are connected.", number_of_connections));

			  if (length_of_process_number == 6) NumberConnectedText->SetLabel(wxString::Format("%6i / %6i processes connected.", number_of_connections, total_processes));
			  else
			  if (length_of_process_number == 5) NumberConnectedText->SetLabel(wxString::Format("%5i / %5i processes connected.", number_of_connections, total_processes));
		      else
			  if (length_of_process_number == 4) NumberConnectedText->SetLabel(wxString::Format("%4i / %4i processes connected.", number_of_connections, total_processes));
			  else
			  if (length_of_process_number == 3) NumberConnectedText->SetLabel(wxString::Format("%3i / %3i processes connected.", number_of_connections, total_processes));
			  else
			  if (length_of_process_number == 2) NumberConnectedText->SetLabel(wxString::Format("%2i / %2i processes connected.", number_of_connections, total_processes));
			  else
		      NumberConnectedText->SetLabel(wxString::Format("%1i / %1i processes connected.", number_of_connections, total_processes));
		  }
	      else
		  if (memcmp(socket_input_buffer, socket_all_jobs_finished, SOCKET_CODE_SIZE) == 0) // identification
		  {
			  my_classification_manager.ProcessAllJobsFinished();
		  }

	      // Enable input events again.

	      sock->SetNotify(wxSOCKET_LOST_FLAG | wxSOCKET_INPUT_FLAG);

	      break;
	    }


	    case wxSOCKET_LOST:
	    {

	    	//MyDebugPrint("Socket Disconnected!!\n");
	        sock->Destroy();
	        break;
	    }
	    default: ;
	  }

}


void ClassificationManager::RunMerge2dJob()
{

	running_job_type = MERGE;

	my_parent->my_job_package.Reset(run_profiles_panel->run_profile_manager.run_profiles[my_parent->RefinementRunProfileComboBox->GetSelection()], "merge2d", 1);

	wxString output_class_averages = output_classification->class_average_file;
	wxString dump_file_seed = main_frame->ReturnRefine2DScratchDirectory() + wxString::Format("/class_dump_file_%li_.dump", output_classification->classification_id);

	my_parent->my_job_package.AddJob("tt",	output_class_averages.ToUTF8().data(),
											dump_file_seed.ToUTF8().data());
	// start job..

	my_parent->WriteBlueText("Merging Class Averages...");

	current_job_id = main_frame->job_controller.AddJob(my_parent, run_profiles_panel->run_profile_manager.run_profiles[my_parent->RefinementRunProfileComboBox->GetSelection()].manager_command, run_profiles_panel->run_profile_manager.run_profiles[my_parent->RefinementRunProfileComboBox->GetSelection()].gui_address);
	my_parent->my_job_id = current_job_id;

	if (current_job_id != -1)
	{
		long number_of_refinement_processes;
		if (my_parent->my_job_package.number_of_jobs + 1 < my_parent->my_job_package.my_profile.ReturnTotalJobs()) number_of_refinement_processes = my_parent->my_job_package.number_of_jobs + 1;
		else number_of_refinement_processes =  my_parent->my_job_package.my_profile.ReturnTotalJobs();

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
		my_parent->InputParamsPanel->Show(false);
		my_parent->OutputTextPanel->Show(true);
		//my_parent->PlotPanel->Show(true);
			//	CTFResultsPanel->Show(true);

		my_parent->ExpertToggleButton->Enable(false);
		my_parent->RefinementPackageComboBox->Enable(false);
		my_parent->InputParametersComboBox->Enable(false);

		my_parent->TimeRemainingText->SetLabel("Time Remaining : ???h:??m:??s");
		my_parent->Layout();
		my_parent->running_job = true;
		my_parent->my_job_tracker.StartTracking(my_parent->my_job_package.number_of_jobs);

		}

		my_parent->ProgressBar->Pulse();

}




void ClassificationManager::ProcessJobResult(JobResult *result_to_process)
{

	if (running_job_type == STARTUP)
	{
		//wxPrintf("Got result %i\n", int(result_to_process->result_data[0] + 0.5));

		long current_image = long(result_to_process->result_data[0] + 0.5) - 1;
		number_of_received_particle_results++;
		long current_time = time(NULL);

		if (number_of_received_particle_results == 1)
		{
			current_job_starttime = current_time;
			time_of_last_update = 0;
		}
		else
		if (current_time != time_of_last_update)
		{
			int current_percentage = float(number_of_received_particle_results) / float(output_classification->number_of_particles) * 100.0;
			time_of_last_update = current_time;
			if (current_percentage > 100) current_percentage = 100;
			my_parent->ProgressBar->SetValue(current_percentage);

			long job_time = current_time - current_job_starttime;
			float seconds_per_job = float(job_time) / float(number_of_received_particle_results - 1);
			long seconds_remaining = float((output_classification->number_of_particles) - number_of_received_particle_results) * seconds_per_job;

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
	if (running_job_type == REFINEMENT)
	{
		long current_image = long(result_to_process->result_data[0] + 0.5) - 1;

		output_classification->classification_results[current_image].psi = result_to_process->result_data[1];
		output_classification->classification_results[current_image].xshift = result_to_process->result_data[2];
		output_classification->classification_results[current_image].yshift = result_to_process->result_data[3];
		output_classification->classification_results[current_image].best_class = int(result_to_process->result_data[4] + 0.5);
		output_classification->classification_results[current_image].sigma = result_to_process->result_data[5];
		output_classification->classification_results[current_image].logp = result_to_process->result_data[6];

		//wxPrintf("Received Result (%f,%f,%f,%f,%f,%f,%f)\n", result_to_process->result_data[0], result_to_process->result_data[1], result_to_process->result_data[2], result_to_process->result_data[3] ,result_to_process->result_data[4] ,result_to_process->result_data[5],result_to_process->result_data[6]);
		number_of_received_particle_results++;
		long current_time = time(NULL);

		if (number_of_received_particle_results == 1)
		{
			current_job_starttime = current_time;
			time_of_last_update = 0;
		}
		else
		if (current_time != time_of_last_update)
		{
			int current_percentage = float(number_of_received_particle_results) / float(output_classification->number_of_particles) * 100.0;
			time_of_last_update = current_time;
			if (current_percentage > 100) current_percentage = 100;
			my_parent->ProgressBar->SetValue(current_percentage);

			long job_time = current_time - current_job_starttime;
			float seconds_per_job = float(job_time) / float(number_of_received_particle_results - 1);
			long seconds_remaining = float((input_classification->number_of_particles) - number_of_received_particle_results) * seconds_per_job;

			TimeRemaining time_remaining;

			if (seconds_remaining > 3600) time_remaining.hours = seconds_remaining / 3600;
			else time_remaining.hours = 0;

			if (seconds_remaining > 60) time_remaining.minutes = (seconds_remaining / 60) - (time_remaining.hours * 60);
			else time_remaining.minutes = 0;

			time_remaining.seconds = seconds_remaining - ((time_remaining.hours * 60 + time_remaining.minutes) * 60);
			my_parent->TimeRemainingText->SetLabel(wxString::Format("Time Remaining : %ih:%im:%is", time_remaining.hours, time_remaining.minutes, time_remaining.seconds));
		}
	}

	/*
	if (running_job_type == REFINEMENT)
	{

		int current_class = int(result_to_process->result_data[0] + 0.5);
		long current_particle = long(result_to_process->result_data[1] + 0.5) - 1;

		MyDebugAssertTrue(current_particle != -1 && current_class != -1, "Current Particle (%li) or Current Class(%i) = -1!", current_particle, current_class);

	//	wxPrintf("Received a refinement result for class #%i, particle %li\n", current_class + 1, current_particle + 1);
		//wxPrintf("output refinement has %i classes and %li particles\n", output_refinement->number_of_classes, output_refinement->number_of_particles);


		output_refinement->class_refinement_results[current_class].particle_refinement_results[current_particle].position_in_stack = long(result_to_process->result_data[1] + 0.5);
		output_refinement->class_refinement_results[current_class].particle_refinement_results[current_particle].psi = result_to_process->result_data[2];
		output_refinement->class_refinement_results[current_class].particle_refinement_results[current_particle].theta = result_to_process->result_data[3];
		output_refinement->class_refinement_results[current_class].particle_refinement_results[current_particle].phi = result_to_process->result_data[4];
		output_refinement->class_refinement_results[current_class].particle_refinement_results[current_particle].xshift = result_to_process->result_data[5];
		output_refinement->class_refinement_results[current_class].particle_refinement_results[current_particle].yshift = result_to_process->result_data[6];
		output_refinement->class_refinement_results[current_class].particle_refinement_results[current_particle].defocus1 = result_to_process->result_data[9];
		output_refinement->class_refinement_results[current_class].particle_refinement_results[current_particle].defocus2 = result_to_process->result_data[10];
		output_refinement->class_refinement_results[current_class].particle_refinement_results[current_particle].defocus_angle = result_to_process->result_data[11];
		output_refinement->class_refinement_results[current_class].particle_refinement_results[current_particle].occupancy = result_to_process->result_data[12];
		output_refinement->class_refinement_results[current_class].particle_refinement_results[current_particle].logp = result_to_process->result_data[13];
		output_refinement->class_refinement_results[current_class].particle_refinement_results[current_particle].sigma = result_to_process->result_data[14];
		output_refinement->class_refinement_results[current_class].particle_refinement_results[current_particle].score = result_to_process->result_data[15];
		output_refinement->class_refinement_results[current_class].particle_refinement_results[current_particle].score_change = result_to_process->result_data[16];

		number_of_received_particle_results++;
		//wxPrintf("received result!\n");
		long current_time = time(NULL);

		if (number_of_received_particle_results == 1)
		{
			current_job_starttime = current_time;
			time_of_last_update = 0;
			my_parent->AngularPlotPanel->SetSymmetryAndNumber(refinement_package_asset_panel->all_refinement_packages.Item(my_parent->RefinementPackageComboBox->GetSelection()).symmetry,output_refinement->number_of_particles);
			my_parent->AngularPlotPanel->Show(true);
			my_parent->FSCResultsPanel->Show(false);
			my_parent->Layout();
		}
		else
		if (current_time != time_of_last_update)
		{
			int current_percentage = float(number_of_received_particle_results) / float(output_refinement->number_of_particles * output_refinement->number_of_classes) * 100.0;
			time_of_last_update = current_time;
			if (current_percentage > 100) current_percentage = 100;
			my_parent->ProgressBar->SetValue(current_percentage);
			long job_time = current_time - current_job_starttime;
			float seconds_per_job = float(job_time) / float(number_of_received_particle_results - 1);
			long seconds_remaining = float((input_refinement->number_of_particles * output_refinement->number_of_classes) - number_of_received_particle_results) * seconds_per_job;

			TimeRemaining time_remaining;

			if (seconds_remaining > 3600) time_remaining.hours = seconds_remaining / 3600;
			else time_remaining.hours = 0;

			if (seconds_remaining > 60) time_remaining.minutes = (seconds_remaining / 60) - (time_remaining.hours * 60);
			else time_remaining.minutes = 0;

			time_remaining.seconds = seconds_remaining - ((time_remaining.hours * 60 + time_remaining.minutes) * 60);
			my_parent->TimeRemainingText->SetLabel(wxString::Format("Time Remaining : %ih:%im:%is", time_remaining.hours, time_remaining.minutes, time_remaining.seconds));
		}


        // Add this result to the list of results to be plotted onto the angular plot
		if (current_class == 0)
		{
			my_parent->AngularPlotPanel->AddRefinementResult( &output_refinement->class_refinement_results[current_class].particle_refinement_results[current_particle]);
	         // Plot this new result onto the angular plot immediately if it's one of the first few results to come in. Otherwise, only plot at regular intervals.

	        if(my_parent->AngularPlotPanel->refinement_results_to_plot.Count() * my_parent->AngularPlotPanel->symmetry_matrices.number_of_matrices < 1500 || current_time - my_parent->time_of_last_result_update > 0)
	        {

	            my_parent->AngularPlotPanel->Refresh();
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
			int current_percentage = float(number_of_received_particle_results) / float(output_refinement->number_of_particles * output_refinement->number_of_classes) * 100.0;
			if (current_percentage > 100) current_percentage = 100;
			my_parent->ProgressBar->SetValue(current_percentage);
			long job_time = current_time - current_job_starttime;
			float seconds_per_job = float(job_time) / float(number_of_received_particle_results - 1);
			long seconds_remaining = float((input_refinement->number_of_particles * input_refinement->number_of_classes) - number_of_received_particle_results) * seconds_per_job;

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


	*/
}

void ClassificationManager::ProcessAllJobsFinished()
{

	if (running_job_type == STARTUP)
	{
		//my_parent->WriteBlueText("Done.");
		CycleRefinement();
	}
	else
	if (running_job_type == REFINEMENT)
	{
		RunMerge2dJob();
	}
	else
	if (running_job_type == MERGE)
	{
		global_delete_refine2d_scratch();
		CycleRefinement();
	}

	/*
	if (running_job_type == REFINEMENT)
	{
		//wxPrintf("Refinement has finished\n");
		main_frame->job_controller.KillJob(my_parent->my_job_id);
		//wxPrintf("Setting up reconstruction\n");
		SetupReconstructionJob();
		//wxPrintf("Running reconstruction\n");
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

		int class_counter;

		main_frame->job_controller.KillJob(my_parent->my_job_id);

		VolumeAsset temp_asset;

		temp_asset.reconstruction_job_id = current_output_refinement_id;
		temp_asset.pixel_size = output_refinement->resolution_statistics_pixel_size;
		temp_asset.x_size = output_refinement->resolution_statistics_box_size;
		temp_asset.y_size = output_refinement->resolution_statistics_box_size;
		temp_asset.z_size = output_refinement->resolution_statistics_box_size;

		// add the volumes to the database..

		output_refinement->reference_volume_ids.Clear();
		refinement_package_asset_panel->all_refinement_packages[my_parent->RefinementPackageComboBox->GetSelection()].references_for_next_refinement.Clear();
		main_frame->current_project.database.BeginVolumeAssetInsert();

		my_parent->WriteInfoText("");

		for (class_counter = 0; class_counter < output_refinement->number_of_classes; class_counter++)
		{
			my_parent->WriteInfoText(wxString::Format(wxT("    Estimated 0.143 resolution for Class %2i = %2.2f Å"), class_counter + 1, output_refinement->class_refinement_results[class_counter].class_resolution_statistics.ReturnEstimatedResolution()));
			temp_asset.asset_id = volume_asset_panel->current_asset_number;

			if (start_with_reconstruction == true)
			{
				temp_asset.asset_name = wxString::Format("Volume From Initial Params #%li - Class #%i", current_output_refinement_id, class_counter + 1);

			}
			else
			{
				if (my_parent->GlobalRefinementRadio->GetValue() == true)
				{
					temp_asset.asset_name = wxString::Format("Volume From Global Search #%li - Class #%i", current_output_refinement_id, class_counter + 1);
				}
				else temp_asset.asset_name = wxString::Format("Volume From Local Search #%li - Class #%i", current_output_refinement_id, class_counter + 1);

			}


			temp_asset.filename = main_frame->current_project.volume_asset_directory.GetFullPath() + wxString::Format("/volume_%li_%i.mrc", output_refinement->refinement_id, class_counter + 1);
			output_refinement->reference_volume_ids.Add(temp_asset.asset_id);

			refinement_package_asset_panel->all_refinement_packages[my_parent->RefinementPackageComboBox->GetSelection()].references_for_next_refinement.Add(temp_asset.asset_id);
			main_frame->current_project.database.ExecuteSQL(wxString::Format("UPDATE REFINEMENT_PACKAGE_CURRENT_REFERENCES_%li SET VOLUME_ASSET_ID=%i WHERE CLASS_NUMBER=%i", current_refinement_package_asset_id, temp_asset.asset_id, class_counter + 1 ));


			volume_asset_panel->AddAsset(&temp_asset);
			main_frame->current_project.database.AddNextVolumeAsset(temp_asset.asset_id, temp_asset.asset_name, temp_asset.filename.GetFullPath(), temp_asset.reconstruction_job_id, temp_asset.pixel_size, temp_asset.x_size, temp_asset.y_size, temp_asset.z_size);
		}

		my_parent->WriteInfoText("");

		main_frame->current_project.database.EndVolumeAssetInsert();

		// Now calculate the occupancies..
		wxPrintf("Calculating Occupancies\n");
		if (output_refinement->number_of_classes > 1)
		{
			int class_counter;
			int particle_counter;
			int point_counter;

			float sum_probabilities;
			float occupancy;
			float max_logp;
			float average_occupancies[output_refinement->number_of_classes];
			float sum_part_ssnr;
			float sum_ave_occ;
			float current_part_ssnr;


			// calculate average occupancies
			for (class_counter = 0; class_counter < output_refinement->number_of_classes; class_counter++)
			{
				average_occupancies[class_counter] = 0.0;

				for (particle_counter = 0; particle_counter < output_refinement->number_of_particles; particle_counter++)
				{
					average_occupancies[class_counter] += output_refinement->class_refinement_results[class_counter].particle_refinement_results[particle_counter].occupancy;
				}

				average_occupancies[class_counter] /= float(output_refinement->number_of_particles);
			}


			for (particle_counter = 0; particle_counter < output_refinement->number_of_particles; particle_counter++)
			{
				max_logp = -FLT_MAX;

				for (class_counter = 0; class_counter < output_refinement->number_of_classes; class_counter++)
				{
					if (output_refinement->class_refinement_results[class_counter].particle_refinement_results[particle_counter].logp > max_logp) max_logp = output_refinement->class_refinement_results[class_counter].particle_refinement_results[particle_counter].logp;
				}


				sum_probabilities = 0.0;

				for (class_counter = 0; class_counter < output_refinement->number_of_classes; class_counter++)
				{
					if (max_logp - output_refinement->class_refinement_results[class_counter].particle_refinement_results[particle_counter].logp < 10.0)
					{
						sum_probabilities += exp(output_refinement->class_refinement_results[class_counter].particle_refinement_results[particle_counter].logp  - max_logp) * average_occupancies[class_counter];
					}
				}

				output_refinement->average_sigma = 0.0;


				for (class_counter = 0; class_counter < output_refinement->number_of_classes; class_counter++)
				{
					if (max_logp -  output_refinement->class_refinement_results[class_counter].particle_refinement_results[particle_counter].logp < 10.0)
					{
						occupancy = exp(output_refinement->class_refinement_results[class_counter].particle_refinement_results[particle_counter].logp - max_logp) * average_occupancies[class_counter] / sum_probabilities *100.0;
					}
					else
					{
						occupancy = 0.0;
					}

					occupancy = 1. * (occupancy - output_refinement->class_refinement_results[class_counter].particle_refinement_results[particle_counter].occupancy) + output_refinement->class_refinement_results[class_counter].particle_refinement_results[particle_counter].occupancy;
					output_refinement->class_refinement_results[class_counter].particle_refinement_results[particle_counter].occupancy = occupancy;
					output_refinement->average_sigma +=  output_refinement->class_refinement_results[class_counter].particle_refinement_results[particle_counter].sigma * output_refinement->class_refinement_results[class_counter].particle_refinement_results[particle_counter].occupancy / 100.0;
				}

			}

			// Now work out the proper part_ssnr

			sum_ave_occ = 0.0;

			for (class_counter = 0; class_counter < output_refinement->number_of_classes; class_counter++)
			{
				sum_ave_occ += average_occupancies[class_counter];

			}

			//wxPrintf("For class %i there are %i points", class_counter, output_refinement->class_refinement_results[0].class_resolution_statistics.part_SSNR.number_of_points);

			for (point_counter = 0; point_counter < output_refinement->class_refinement_results[0].class_resolution_statistics.part_SSNR.number_of_points; point_counter++)
			{
				sum_part_ssnr = 0;
				for (class_counter = 0; class_counter < output_refinement->number_of_classes; class_counter++)
				{
					sum_part_ssnr += output_refinement->class_refinement_results[class_counter].class_resolution_statistics.part_SSNR.data_y[point_counter] * average_occupancies[class_counter];
				}

				current_part_ssnr = sum_part_ssnr / sum_ave_occ;

				for (class_counter = 0; class_counter < output_refinement->number_of_classes; class_counter++)
				{
					output_refinement->class_refinement_results[class_counter].class_resolution_statistics.part_SSNR.data_y[point_counter] = current_part_ssnr;
				}

			}



		}
		else
		{
			output_refinement->average_sigma = 0.0;
			for (long particle_counter = 0; particle_counter < output_refinement->number_of_particles; particle_counter++)
			{
				output_refinement->average_sigma += output_refinement->class_refinement_results[0].particle_refinement_results[particle_counter].sigma;
			}

			output_refinement->average_sigma /= float (output_refinement->number_of_particles);
		}



		//wxPrintf("Writing to databse\n");
		// write the refinement to the database

		if (start_with_reconstruction == true)
		{
			long point_counter;

			volume_asset_panel->is_dirty = true;
			refinement_package_asset_panel->is_dirty = true;
			my_parent->input_params_combo_is_dirty = true;
		//	my_parent->SetDefaults();
			refinement_results_panel->is_dirty = true;

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


			my_parent->FSCResultsPanel->AddRefinement(output_refinement);
		}
		else
		{
			main_frame->current_project.database.AddRefinement(output_refinement);
			ShortRefinementInfo temp_info;
			temp_info = output_refinement;
			refinement_package_asset_panel->all_refinement_short_infos.Add(temp_info);

			// add this refinment to the refinement package..

			refinement_package_asset_panel->all_refinement_packages[my_parent->RefinementPackageComboBox->GetSelection()].last_refinment_id = output_refinement->refinement_id;
			refinement_package_asset_panel->all_refinement_packages[my_parent->RefinementPackageComboBox->GetSelection()].refinement_ids.Add(output_refinement->refinement_id);

			main_frame->current_project.database.ExecuteSQL(wxString::Format("UPDATE REFINEMENT_PACKAGE_ASSETS SET LAST_REFINEMENT_ID=%li WHERE REFINEMENT_PACKAGE_ASSET_ID=%li", output_refinement->refinement_id, current_refinement_package_asset_id));
			main_frame->current_project.database.ExecuteSQL(wxString::Format("INSERT INTO REFINEMENT_PACKAGE_REFINEMENTS_LIST_%li (REFINEMENT_NUMBER, REFINEMENT_ID) VALUES (%li, %li);", current_refinement_package_asset_id, refinement_package_asset_panel->all_refinement_packages[my_parent->RefinementPackageComboBox->GetSelection()].refinement_ids.GetCount(),  output_refinement->refinement_id));

			volume_asset_panel->is_dirty = true;
			refinement_package_asset_panel->is_dirty = true;
			my_parent->input_params_combo_is_dirty = true;
	//		my_parent->SetDefaults();
			refinement_results_panel->is_dirty = true;

			my_parent->FSCResultsPanel->AddRefinement(output_refinement);



		}

		if (wxDir::Exists(main_frame->current_project.scratch_directory.GetFullPath()) == true) wxFileName::Rmdir(main_frame->current_project.scratch_directory.GetFullPath(), wxPATH_RMDIR_RECURSIVE);
		if (wxDir::Exists(main_frame->current_project.scratch_directory.GetFullPath()) == false) wxFileName::Mkdir(main_frame->current_project.scratch_directory.GetFullPath());

		my_parent->FSCResultsPanel->Show(true);
		my_parent->AngularPlotPanel->Show(false);
		my_parent->Layout();


		//wxPrintf("Calling cycle refinement\n");
		CycleRefinement();
	}
	*/

}

void ClassificationManager::CycleRefinement()
{



	// add classification
	main_frame->current_project.database.AddClassification(output_classification);

	ShortClassificationInfo temp_info;
	temp_info = output_classification;
	refinement_package_asset_panel->all_classification_short_infos.Add(temp_info);

	// add classification id to the refinement package..

	refinement_package_asset_panel->all_refinement_packages[my_parent->RefinementPackageComboBox->GetSelection()].classification_ids.Add(output_classification->classification_id);
	main_frame->current_project.database.ExecuteSQL(wxString::Format("INSERT INTO REFINEMENT_PACKAGE_CLASSIFICATIONS_LIST_%li (CLASSIFICATION_NUMBER, CLASSIFICATION_ID) VALUES (%li, %li);", current_refinement_package_asset_id, refinement_package_asset_panel->all_refinement_packages[my_parent->RefinementPackageComboBox->GetSelection()].classification_ids.GetCount(),  output_classification->classification_id));


	if (start_with_random == true)
	{
		input_classification = output_classification;
		output_classification = new Classification;
		start_with_random = false;
		first_round_id = input_classification->classification_id;
		my_parent->ResultDisplayPanel->OpenFile(input_classification->class_average_file, wxString::Format("Class. #%li (Start Ref.)", input_classification->classification_id));
		my_parent->ResultDisplayPanel->Show(true);

		RunRefinementJob();
	}
	else
	{
		number_of_rounds_run++;

		my_parent->ResultDisplayPanel->OpenFile(output_classification->class_average_file, wxString::Format("Class. #%li (Round %i)", output_classification->classification_id, number_of_rounds_run));


		// statistics..

		double average_likelihood = 0.0;
		double average_sigma = 0.0;
		long number_active = 0.0;
		long number_moved = 0.0;
		float percent_moved;

		for (long counter = 0; counter < input_classification->number_of_particles; counter++)
		{
			if (output_classification->classification_results[counter].best_class > 0)
			{

				number_active++;
				average_likelihood += output_classification->classification_results[counter].logp;
				average_sigma += output_classification->classification_results[counter].sigma;

				if (long(fabsf(output_classification->classification_results[counter].best_class)) != long(fabsf(input_classification->classification_results[counter].best_class))) number_moved++;
			}

			percent_moved = (double(number_moved) / double (number_active)) * 100.0;
		}

		average_likelihood /= double(number_active);
		average_sigma /= double(number_active);

		my_parent->PlotPanel->AddPoints(number_of_rounds_run, average_likelihood, average_sigma, percent_moved);
		my_parent->PlotPanel->Draw();

		if (number_of_rounds_run == 1)
		{
			my_parent->PlotPanel->Show(true);
			my_parent->Layout();
		}

		if (number_of_rounds_run < number_of_rounds_to_run)
		{
			current_input_classification_id = output_classification->classification_id;
			delete input_classification;
			input_classification = output_classification;
			output_classification = new Classification;

			RunRefinementJob();

		}
		else
		{
			delete input_classification;
			delete output_classification;
			my_parent->WriteBlueText("All refinement cycles are finished!");
			my_parent->CancelAlignmentButton->Show(false);
			my_parent->FinishButton->Show(true);
			my_parent->TimeRemainingText->SetLabel("Time Remaining : Finished!");
			my_parent->ProgressBar->SetValue(100);
			my_parent->ProgressPanel->Layout();
			my_parent->input_params_combo_is_dirty = true;

		}

	}

	main_frame->DirtyClassifications();
}


