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

	ResultDisplayPanel->Initialise(CAN_FFT | START_WITH_FOURIER_SCALING | KEEP_TABS_LINKED_IF_POSSIBLE);

	RefinementPackageComboBox->AssetComboBox->Bind(wxEVT_COMMAND_COMBOBOX_SELECTED, &MyRefine2DPanel::OnRefinementPackageComboBox, this);
	InputParametersComboBox->AssetComboBox->Bind(wxEVT_COMMAND_COMBOBOX_SELECTED, &MyRefine2DPanel::OnInputParametersComboBox, this);

	Bind(wxEVT_WRITECLASSIFICATIONSTARFILETHREAD_COMPLETED, &MyRefine2DPanel::OnStarFileWriteThreadComplete, this);

	input_params_combo_is_dirty = false;
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

void MyRefine2DPanel::Reset()
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

	ExpertToggleButton->SetValue(false);
	ExpertPanel->Show(false);

	RefinementPackageComboBox->Clear();
	InputParametersComboBox->Clear();
	RefinementRunProfileComboBox->Clear();


	ResultDisplayPanel->Clear();

	if (running_job == true)
	{
		main_frame->job_controller.KillJob(my_job_id);
		global_delete_refine2d_scratch();

		running_job = false;
	}

	SetDefaults();
	Layout();
}

void MyRefine2DPanel::FillRefinementPackagesComboBox()
{
	if (RefinementPackageComboBox->FillComboBox() == false) NewRefinementPackageSelected();
}

void MyRefine2DPanel::FillInputParamsComboBox()
{
	if (RefinementPackageComboBox->GetSelection() >= 0 && refinement_package_asset_panel->all_refinement_packages.GetCount() > 0)
	{
		InputParametersComboBox->FillComboBox(RefinementPackageComboBox->GetSelection(), true, true);
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

			if (RefinementPackageComboBox->GetCount() > 0 && RefinementRunProfileComboBox->GetSelection() >= 0)
			{
				if (run_profiles_panel->run_profile_manager.ReturnTotalJobs(RefinementRunProfileComboBox->GetSelection()) > 0 )
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

void MyRefine2DPanel::NewInputParametersSelected()
{
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
	InfoText->WriteText(wxT("Low-Resolution Limit (Å) : "));
	InfoText->EndBold();
	InfoText->WriteText(wxT("The data used for classification is usually bandpass-limited to exclude spurious low-resolution features in the particle background. It is therefore good practice to set the low-resolution limit to 2.5x the approximate particle mask radius."));
	InfoText->Newline();

	InfoText->BeginBold();
	InfoText->WriteText(wxT("High-Resolution Limit (start/finish) (Å) : "));
	InfoText->EndBold();
	InfoText->WriteText(wxT("The high-resolution bandpass limit should be selected to remove data with low signal-to-noise ratio, and to help speed up the calculation. Since the class averages are not well defined initially, the starting limit should be set to a low resolution, for example 40 Å. The limit used for the final iterations should be set sufficiently high, for example 8 Å, to include signal originating from protein secondary structure that often helps generate recognizable features in the class averages (see example above)."));
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

void MyRefine2DPanel::OnStarFileWriteThreadComplete( wxThreadEvent& event )
{
	if (my_classification_manager.running_job_type == STARTUP)
	{
		my_classification_manager.RunInitialStartJobPostStarFileWrite(event.GetString());
	}
	else
	if (my_classification_manager.running_job_type == REFINEMENT)
	{
		my_classification_manager.RunRefinementJobPostStarFileWrite(event.GetString());
	}
}

void MyRefine2DPanel::StartClassificationClick( wxCommandEvent& event )
{
	stopwatch.Start();
	my_classification_manager.BeginRefinementCycle();
}

void MyRefine2DPanel::SetDefaults()
{
	if (RefinementPackageComboBox->GetCount() > 0)
	{
		ExpertPanel->Freeze();

		LowResolutionLimitTextCtrl->SetValue("300.00");
		HighResolutionLimitStartTextCtrl->SetValue("40.00");
		HighResolutionLimitFinishTextCtrl->SetValue("8.00");

		if (InputParametersComboBox->GetSelection() > 0)
		{
			NumberClassesSpinCtrl->SetValue(refinement_package_asset_panel->ReturnPointerToShortClassificationInfoByClassificationID(refinement_package_asset_panel->all_refinement_packages.Item(RefinementPackageComboBox->GetSelection()).classification_ids.Item(InputParametersComboBox->GetSelection() - 1))->number_of_classes);
			HighResolutionLimitStartTextCtrl->SetValue(wxString::Format("%.2f",refinement_package_asset_panel->ReturnPointerToShortClassificationInfoByClassificationID(refinement_package_asset_panel->all_refinement_packages.Item(RefinementPackageComboBox->GetSelection()).classification_ids.Item(InputParametersComboBox->GetSelection() - 1))->high_resolution_limit));
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

		float local_mask_radius = refinement_package_asset_panel->all_refinement_packages.Item(RefinementPackageComboBox->GetSelection()).estimated_particle_size_in_angstroms * 0.6;
		MaskRadiusTextCtrl->SetValue(wxString::Format("%.2f", local_mask_radius));
		AngularStepTextCtrl->SetValue("15.00");

		float search_range = refinement_package_asset_panel->all_refinement_packages.Item(RefinementPackageComboBox->GetSelection()).estimated_particle_size_in_angstroms * 0.33;

		MaxSearchRangeTextCtrl->SetValue(wxString::Format("%.2f", search_range));

		SmoothingFactorTextCtrl->SetValue("1.00");
		ExcludeBlankEdgesNoRadio->SetValue(true);
		AutoPercentUsedRadioYes->SetValue(true);
		PercentUsedTextCtrl->SetValue("100.00");
		AutoMaskRadioNo->SetValue(true);
		AutoCentreRadioYes->SetValue(true);
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
	my_classification_manager.running_job_type = NOJOB;
	WriteBlueText("Terminated Job");
	TimeRemainingText->SetLabel("Time Remaining : Terminated");
	CancelAlignmentButton->Show(false);
	global_delete_refine2d_scratch();
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

	if (my_classification_manager.number_of_rounds_run > 0)
	{
		FillInputParamsComboBox();
		if (RefinementPackageComboBox->GetSelection() >= 0 && InputParametersComboBox->GetSelection() > 0) HighResolutionLimitStartTextCtrl->SetValue(wxString::Format("%.2f",refinement_package_asset_panel->ReturnPointerToShortClassificationInfoByClassificationID(refinement_package_asset_panel->all_refinement_packages.Item(RefinementPackageComboBox->GetSelection()).classification_ids.Item(InputParametersComboBox->GetSelection() - 1))->high_resolution_limit));
	}

	//CTFResultsPanel->CTF2DResultsPanel->should_show = false;
	//CTFResultsPanel->CTF2DResultsPanel->Refresh();

}

void MyRefine2DPanel::OnRefinementPackageComboBox( wxCommandEvent& event )
{

	NewRefinementPackageSelected();

}


void MyRefine2DPanel::OnInputParametersComboBox( wxCommandEvent& event )
{
	ExpertPanel->Freeze();
	if (InputParametersComboBox->GetSelection() > 0)
	{
		HighResolutionLimitStartTextCtrl->SetValue(wxString::Format("%.2f",refinement_package_asset_panel->ReturnPointerToShortClassificationInfoByClassificationID(refinement_package_asset_panel->all_refinement_packages.Item(RefinementPackageComboBox->GetSelection()).classification_ids.Item(InputParametersComboBox->GetSelection() - 1))->high_resolution_limit));
	}
	else
	{
		HighResolutionLimitStartTextCtrl->SetValue("40.00");
	}
	ExpertPanel->Thaw();
}


void ClassificationManager::SetParent(MyRefine2DPanel *wanted_parent)
{
	my_parent = wanted_parent;
}

void ClassificationManager::BeginRefinementCycle()
{

	start_with_random = false;

	number_of_rounds_run = 0;
	number_of_rounds_to_run = my_parent->NumberRoundsSpinCtrl->GetValue();

	active_refinement_package = &refinement_package_asset_panel->all_refinement_packages.Item(my_parent->RefinementPackageComboBox->GetSelection());
	current_refinement_package_asset_id = active_refinement_package->asset_id;
	active_run_profile = run_profiles_panel->run_profile_manager.run_profiles[my_parent->RefinementRunProfileComboBox->GetSelection()];

	my_parent->PlotPanel->Clear();
	my_parent->PlotPanel->my_notebook->SetSelection(0);

	active_number_of_classes = my_parent->NumberClassesSpinCtrl->GetValue();
	active_low_resolution_limit = my_parent->LowResolutionLimitTextCtrl->ReturnValue();
	active_start_high_resolution_limit = my_parent->HighResolutionLimitStartTextCtrl->ReturnValue();
	active_finish_high_resolution_limit = my_parent->HighResolutionLimitFinishTextCtrl->ReturnValue();
	active_mask_radius = my_parent->MaskRadiusTextCtrl->ReturnValue();
	active_angular_search_step = my_parent->AngularStepTextCtrl->ReturnValue();
	active_max_search_range = my_parent->MaxSearchRangeTextCtrl->ReturnValue();
	active_smoothing_factor = my_parent->SmoothingFactorTextCtrl->ReturnValue();
	active_exclude_blank_edges = my_parent->ExcludeBlankEdgesYesRadio->GetValue();
	active_auto_percent_used = my_parent->AutoPercentUsedRadioYes->GetValue();
	active_percent_used = my_parent->PercentUsedTextCtrl->ReturnValue();


	if (my_parent->InputParametersComboBox->GetSelection() == 0)
	{
		start_with_random = true;
		current_input_classification_id = -1;
		input_classification = NULL;
		output_classification = new Classification;
		min_percent_used = 0.0f;

		RunInitialStartJob();
	}
	else
	{
		current_input_classification_id = active_refinement_package->classification_ids[my_parent->InputParametersComboBox->GetSelection() - 1];
		first_round_id = current_input_classification_id;
		wxPrintf("classification list id %i = %li\n", my_parent->InputParametersComboBox->GetSelection() - 1, current_input_classification_id);
		wxPrintf("Getting classification for classification id %li\n", current_input_classification_id);
		input_classification = main_frame->current_project.database.GetClassificationByID(current_input_classification_id);
		output_classification = new Classification;
		my_parent->ResultDisplayPanel->OpenFile(input_classification->class_average_file, wxString::Format("Class #%li (Start Ref.)", input_classification->classification_id));
		min_percent_used = input_classification->percent_used;
		RunRefinementJob();
	}

}


// This now just changes gui launches a thread to write the star file, this is to stop gui lockups.
// After the thread finishes, RunInitialStartJobbPostStarFileWrite will be called.

void ClassificationManager::RunInitialStartJob()
{
	running_job_type = STARTUP;
	my_parent->current_job_package.Reset(active_run_profile, "refine2d", 1);

	my_parent->WriteBlueText("Creating Initial References...");
	my_parent->WriteBlueText("Preparing Data Files...\n");

	my_parent->StartPanel->Show(false);
	my_parent->ProgressPanel->Show(true);

	my_parent->ExpertPanel->Show(false);
	my_parent->InfoPanel->Show(false);
	my_parent->InputParamsPanel->Show(false);
	my_parent->OutputTextPanel->Show(true);

	my_parent->ExpertToggleButton->Enable(false);
	my_parent->RefinementPackageComboBox->Enable(false);
	my_parent->InputParametersComboBox->Enable(false);

	my_parent->SetNumberConnectedTextToZeroAndStartTracking();

	WriteClassificationStarFileThread *star_file_writer_thread;
	star_file_writer_thread = new WriteClassificationStarFileThread(my_parent, main_frame->current_project.parameter_file_directory.GetFullPath() + "/classification_input_star", output_classification, active_refinement_package);

	if ( star_file_writer_thread->Run() != wxTHREAD_NO_ERROR )
	{
		my_parent->WriteErrorText("Error: Cannot start star file writer thread, things are going to break...");
		delete star_file_writer_thread;
	}
}



void ClassificationManager::RunInitialStartJobPostStarFileWrite(wxString input_star_file)
{
	number_of_received_particle_results = 0;

	output_classification->classification_id = main_frame->current_project.database.ReturnHighestClassificationID() + 1;
	output_classification->refinement_package_asset_id = current_refinement_package_asset_id;
	output_classification->name = wxString::Format("Random Start #%li", output_classification->classification_id);
	output_classification->class_average_file = main_frame->current_project.class_average_directory.GetFullPath() + wxString::Format("/class_averages_%.4li.mrc", output_classification->classification_id);
	output_classification->classification_was_imported_or_generated = true;
	output_classification->datetime_of_run = wxDateTime::Now();
	output_classification->starting_classification_id = -1;
	output_classification->number_of_particles = active_refinement_package->contained_particles.GetCount();
	output_classification->number_of_classes = active_number_of_classes;
	output_classification->low_resolution_limit = active_low_resolution_limit;
	output_classification->high_resolution_limit = active_start_high_resolution_limit;
	output_classification->mask_radius = active_mask_radius;
	output_classification->angular_search_step = active_angular_search_step;
	output_classification->search_range_x = active_max_search_range;
	output_classification->search_range_y = active_max_search_range;
	output_classification->smoothing_factor = active_smoothing_factor;
	output_classification->exclude_blank_edges = active_exclude_blank_edges;
	output_classification->auto_percent_used = active_auto_percent_used;

	output_classification->percent_used = (float(output_classification->number_of_classes * 300) / float(output_classification->number_of_particles)) * 100.0;
	if (output_classification->percent_used > 100.0) output_classification->percent_used = 100.0;

	output_classification->SizeAndFillWithEmpty(output_classification->number_of_particles);

	for (long counter = 1; counter <= output_classification->number_of_particles; counter++)
	{
		output_classification->classification_results[counter - 1].position_in_stack = counter;
		output_classification->classification_results[counter - 1].amplitude_contrast = active_refinement_package->ReturnParticleInfoByPositionInStack( counter ).amplitude_contrast;
		output_classification->classification_results[counter - 1].pixel_size = active_refinement_package->ReturnParticleInfoByPositionInStack( counter ).pixel_size;
		output_classification->classification_results[counter - 1].microscope_voltage_kv = active_refinement_package->ReturnParticleInfoByPositionInStack( counter ).microscope_voltage;
		output_classification->classification_results[counter - 1].microscope_spherical_aberration_mm = active_refinement_package->ReturnParticleInfoByPositionInStack( counter ).spherical_aberration;
		output_classification->classification_results[counter - 1].beam_tilt_x = 0.0f;
		output_classification->classification_results[counter - 1].beam_tilt_y = 0.0f;
		output_classification->classification_results[counter - 1].image_shift_x = 0.0f;
		output_classification->classification_results[counter - 1].image_shift_y = 0.0f;
		output_classification->classification_results[counter - 1].defocus_1 = active_refinement_package->ReturnParticleInfoByPositionInStack( counter ).defocus_1;
		output_classification->classification_results[counter - 1].defocus_2 = active_refinement_package->ReturnParticleInfoByPositionInStack( counter ).defocus_2;
		output_classification->classification_results[counter - 1].defocus_angle = active_refinement_package->ReturnParticleInfoByPositionInStack( counter ).defocus_angle;
		output_classification->classification_results[counter - 1].phase_shift = active_refinement_package->ReturnParticleInfoByPositionInStack( counter ).phase_shift;
	}

	wxString input_particle_images =  active_refinement_package->stack_filename;
	wxString input_class_averages = "/dev/null";
	wxString output_star_file = "/dev/null";
	wxString output_class_averages = output_classification->class_average_file;
	int number_of_classes = output_classification->number_of_classes;
	int first_particle = 1;
	int last_particle = output_classification->number_of_particles;
	float percent_used = output_classification->percent_used / 100.00;
	float output_pixel_size = active_refinement_package->output_pixel_size;
	float mask_radius = active_mask_radius;
	float low_resolution_limit =output_classification->low_resolution_limit;
	float high_resolution_limit = output_classification->high_resolution_limit;
	float angular_step = output_classification->angular_search_step;
	float max_search_x = output_classification->search_range_x;
	float max_search_y = output_classification->search_range_y;
	float smoothing_factor = output_classification->smoothing_factor;
	int padding_factor = 2;
	bool normalize_particles = true;
	bool invert_contrast = active_refinement_package->stack_has_white_protein;
	bool exclude_blank_edges = output_classification->exclude_blank_edges;
	bool dump_arrays = false;
	wxString dump_file = "/dev/null";

	bool auto_mask = false;
	bool auto_centre = false;
	int max_threads = 1;

	my_parent->current_job_package.AddJob("tttttiiiffffffffibbbbtbbi",	input_particle_images.ToUTF8().data(),
																	input_star_file.ToUTF8().data(),
																	input_class_averages.ToUTF8().data(),
																	output_star_file.ToUTF8().data(),
																	output_class_averages.ToUTF8().data(),
																	number_of_classes,
																	first_particle,
																	last_particle,
																	percent_used,
																	output_pixel_size,
																	mask_radius,
																	low_resolution_limit,
																	high_resolution_limit,
																	angular_step,
																	active_max_search_range,
																	smoothing_factor,
																	padding_factor,
																	normalize_particles,
																	invert_contrast,
																	exclude_blank_edges,
																	dump_arrays,
																	dump_file.ToUTF8().data(),
																	auto_mask,
																	auto_centre,
																	max_threads);

	current_job_id = main_frame->job_controller.AddJob(my_parent, active_run_profile.manager_command, active_run_profile.gui_address);
	my_parent->my_job_id = current_job_id;

	my_parent->ProgressBar->Pulse();

}

// This now just changes gui anlaunches a thread to write the star file, this is to stop gui lockups.
// After the thread finishes, RunRefinementJobPostStarFileWrite will be called.

void ClassificationManager::RunRefinementJob()
{
	running_job_type = REFINEMENT;
	my_parent->current_job_package.Reset(active_run_profile, "refine2d", std::min(long(active_run_profile.ReturnTotalJobs()), input_classification->number_of_particles));
	my_parent->WriteBlueText(wxString::Format("Running refinement round %2i of %2i \n", number_of_rounds_run + 1, number_of_rounds_to_run));
	my_parent->WriteBlueText("Preparing Data Files...\n");

	my_parent->StartPanel->Show(false);
	my_parent->ProgressPanel->Show(true);
	my_parent->ExpertPanel->Show(false);
	my_parent->InfoPanel->Show(false);
	my_parent->InputParamsPanel->Show(false);
	my_parent->OutputTextPanel->Show(true);
	my_parent->ResultDisplayPanel->Show(true);

	my_parent->ExpertToggleButton->Enable(false);
	my_parent->RefinementPackageComboBox->Enable(false);
	my_parent->InputParametersComboBox->Enable(false);

	my_parent->SetNumberConnectedTextToZeroAndStartTracking();

	WriteClassificationStarFileThread *star_file_writer_thread;
	star_file_writer_thread = new WriteClassificationStarFileThread(my_parent, main_frame->current_project.parameter_file_directory.GetFullPath() + "/classification_input_star", input_classification, active_refinement_package);

	if ( star_file_writer_thread->Run() != wxTHREAD_NO_ERROR )
	{
		my_parent->WriteErrorText("Error: Cannot start star file writer thread, things are going to break...");
		delete star_file_writer_thread;
	}
}

void ClassificationManager::RunRefinementJobPostStarFileWrite(wxString input_star_file)
{
	long number_of_refinement_jobs;
	int number_of_refinement_processes;
	long number_of_particles;
	int job_counter;
	int reach_max_high_res_at_cycle;
	long first_particle;
	long last_particle;

	number_of_received_particle_results = 0;

	output_classification->classification_id = main_frame->current_project.database.ReturnHighestClassificationID() + 1;
	output_classification->refinement_package_asset_id = current_refinement_package_asset_id;
	output_classification->name = wxString::Format("Classification #%li (Start #%li, Round %i)", output_classification->classification_id, first_round_id, number_of_rounds_run + 1);
	output_classification->class_average_file = main_frame->current_project.class_average_directory.GetFullPath() + wxString::Format("/class_averages_%.4li.mrc", output_classification->classification_id);
	output_classification->classification_was_imported_or_generated = false;
	output_classification->datetime_of_run = wxDateTime::Now();
	output_classification->starting_classification_id = input_classification->classification_id;
	output_classification->number_of_particles = active_refinement_package->contained_particles.GetCount();
	output_classification->number_of_classes = input_classification->number_of_classes;
	output_classification->low_resolution_limit = active_low_resolution_limit;
	//output_classification->high_resolution_limit = my_parent->HighResolutionLimitTextCtrl->ReturnValue();
	output_classification->mask_radius = active_mask_radius;
	output_classification->angular_search_step = active_angular_search_step;
	output_classification->search_range_x = active_max_search_range;
	output_classification->search_range_y = active_max_search_range;
	output_classification->smoothing_factor = active_smoothing_factor;
	output_classification->exclude_blank_edges = active_exclude_blank_edges;
	output_classification->auto_percent_used = active_auto_percent_used;

	// Ramp up the high resolution limit
	if (number_of_rounds_to_run > 1)
	{
		if (number_of_rounds_to_run >= 4)
		{
			reach_max_high_res_at_cycle = number_of_rounds_to_run * 3 / 4;
		}
		else
		{
			reach_max_high_res_at_cycle = number_of_rounds_to_run;
		}
		if (number_of_rounds_run >= reach_max_high_res_at_cycle)
		{
			output_classification->high_resolution_limit = active_finish_high_resolution_limit;
		}
		else
		{
			output_classification->high_resolution_limit = active_start_high_resolution_limit + float(number_of_rounds_run) / float(reach_max_high_res_at_cycle - 1) * (active_finish_high_resolution_limit - active_start_high_resolution_limit);
		}
	}
	else
	{
		output_classification->high_resolution_limit = active_finish_high_resolution_limit;
	}

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
		output_classification->percent_used = active_percent_used;
	}

	if (output_classification->percent_used < min_percent_used) output_classification->percent_used = min_percent_used;

	output_classification->SizeAndFillWithEmpty(output_classification->number_of_particles);

	for (long counter = 0; counter < output_classification->number_of_particles; counter++)
	{
		output_classification->classification_results[counter].position_in_stack = counter + 1;
	}

	number_of_particles = output_classification->number_of_particles;
	number_of_refinement_processes = std::min(long(active_run_profile.ReturnTotalJobs()),number_of_particles);
	number_of_refinement_jobs = number_of_refinement_processes;

	for (job_counter = 0; job_counter < number_of_refinement_jobs; job_counter++)
	{

		wxString input_particle_images =  active_refinement_package->stack_filename;
		wxString input_class_averages = input_classification->class_average_file;

		wxString output_star_file = "/dev/null";
		wxString output_class_averages = main_frame->current_project.class_average_directory.GetFullPath() + wxString::Format("/class_averages_%li.mrc", output_classification->classification_id);
		int number_of_classes = 0;

		FirstLastParticleForJob(first_particle,last_particle,number_of_particles,job_counter+1,number_of_refinement_jobs);

		//output_parameter_file = wxString::Format("/tmp/out_%li_%li.par", first_particle, last_particle);

		float percent_used = output_classification->percent_used / 100.00;
		float output_pixel_size = active_refinement_package->output_pixel_size;
		float mask_radius = active_mask_radius;
		float low_resolution_limit =output_classification->low_resolution_limit;
		float high_resolution_limit = output_classification->high_resolution_limit;
		float angular_step = output_classification->angular_search_step;
		float max_search_range = active_max_search_range;
		float smoothing_factor = output_classification->smoothing_factor;
		int padding_factor = 2;
		bool normalize_particles = true;
		bool invert_contrast = active_refinement_package->stack_has_white_protein;
		bool exclude_blank_edges = output_classification->exclude_blank_edges;
		bool dump_arrays = true;
		wxString dump_file = main_frame->ReturnRefine2DScratchDirectory() + wxString::Format("/class_dump_file_%li_%i.dump", output_classification->classification_id, job_counter +1);
		bool auto_mask = my_parent->AutoMaskRadioYes->GetValue();
		bool auto_centre = my_parent->AutoCentreRadioYes->GetValue();
		int max_threads = 1;

		if (job_counter == 0) my_parent->WriteInfoText("Will automask reference class averages");

		my_parent->current_job_package.AddJob("tttttiiiffffffffibbbbtbbi",	input_particle_images.ToUTF8().data(),
																		input_star_file.ToUTF8().data(),
																		input_class_averages.ToUTF8().data(),
																		output_star_file.ToUTF8().data(),
																		output_class_averages.ToUTF8().data(),
																		number_of_classes,
																		first_particle,
																		last_particle,
																		percent_used,
																		output_pixel_size,
																		mask_radius,
																		low_resolution_limit,
																		high_resolution_limit,
																		angular_step,
																		max_search_range,
																		smoothing_factor,
																		padding_factor,
																		normalize_particles,
																		invert_contrast,
																		exclude_blank_edges,
																		dump_arrays,
																		dump_file.ToUTF8().data(),
																		auto_mask,
																		auto_centre,
																		max_threads);
	}

	my_parent->WriteInfoText(wxString::Format("High resolution limit: %.1f A",output_classification->high_resolution_limit));
	if (my_parent->AutoPercentUsedRadioYes->GetValue() == true)
	{
		my_parent->WriteInfoText(wxString::Format("Using %.0f %% of the particles (%i per class)", output_classification->percent_used, myroundint((float(output_classification->number_of_particles) * output_classification->percent_used * 0.01) / float(output_classification->number_of_classes))));

	}


	current_job_id = main_frame->job_controller.AddJob(my_parent, active_run_profile.manager_command, active_run_profile.gui_address);
	my_parent->my_job_id = current_job_id;
	my_parent->ProgressBar->Pulse();

}

/*
 * After merging is done, the temporary dump files can be removed from the scratch directory
 *
 * (could have done this with wxDir::GetAllFiles to get a list of all the files, but I'm weary of relying on the filesystem
 * in case there are NFS-style caching issues)
 */
void ClassificationManager::RemoveFilesFromScratch()
{
	int number_of_refinement_jobs = active_run_profile.ReturnTotalJobs() - 1;
	wxString dump_file;
	for (int job_counter = 0; job_counter < number_of_refinement_jobs; job_counter++)
	{
		dump_file = main_frame->ReturnRefine2DScratchDirectory() + wxString::Format("/class_dump_file_%li_%i.dump", output_classification->classification_id, job_counter +1);
		wxRemoveFile(dump_file);
	}
}

void MyRefine2DPanel::OnSocketJobResultMsg(JobResult &received_result)
{
	my_classification_manager.ProcessJobResult(&received_result);
}

void MyRefine2DPanel::OnSocketJobResultQueueMsg(ArrayofJobResults &received_queue)
{
	for (int counter = 0; counter < received_queue.GetCount(); counter++)
	{
		my_classification_manager.ProcessJobResult(&received_queue.Item(counter));
	}
}

void MyRefine2DPanel::SetNumberConnectedText(wxString wanted_text)
{
	NumberConnectedText->SetLabel(wanted_text);
}

void MyRefine2DPanel::SetTimeRemainingText(wxString wanted_text)
{
	TimeRemainingText->SetLabel(wanted_text);
}

void MyRefine2DPanel::OnSocketAllJobsFinished()
{
	my_classification_manager.ProcessAllJobsFinished();
}


void ClassificationManager::RunMerge2dJob()
{

	long number_of_particles = output_classification->number_of_particles;
	int number_of_refinement_jobs = std::min(my_parent->current_job_package.my_profile.ReturnTotalJobs() - 1,number_of_particles-1);

	running_job_type = MERGE;

	my_parent->current_job_package.Reset(active_run_profile, "merge2d", 1);

	wxString output_class_averages = output_classification->class_average_file;
	wxString dump_file_seed = main_frame->ReturnRefine2DScratchDirectory() + wxString::Format("/class_dump_file_%li_.dump", output_classification->classification_id);

	my_parent->current_job_package.AddJob("tti",	output_class_averages.ToUTF8().data(),
											dump_file_seed.ToUTF8().data(),
											number_of_refinement_jobs);
	// start job..

	my_parent->WriteBlueText("Merging Class Averages...");

	current_job_id = main_frame->job_controller.AddJob(my_parent, active_run_profile.manager_command, active_run_profile.gui_address);
	my_parent->my_job_id = current_job_id;

	if (current_job_id != -1)
	{

		my_parent->StartPanel->Show(false);
		my_parent->ProgressPanel->Show(true);

		my_parent->ExpertPanel->Show(false);
		my_parent->InfoPanel->Show(false);
		my_parent->InputParamsPanel->Show(false);
		my_parent->OutputTextPanel->Show(true);

		my_parent->ExpertToggleButton->Enable(false);
		my_parent->RefinementPackageComboBox->Enable(false);
		my_parent->InputParametersComboBox->Enable(false);

		my_parent->SetNumberConnectedTextToZeroAndStartTracking();
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
			int current_percentage = float(number_of_received_particle_results) / float(output_classification->number_of_particles * output_classification->percent_used * 0.01) * 100.0;
			time_of_last_update = current_time;
			if (current_percentage > 100) current_percentage = 100;
			else if (current_percentage < 0) current_percentage = 0;

			my_parent->ProgressBar->SetValue(current_percentage);

			long job_time = current_time - current_job_starttime;
			float seconds_per_job = float(job_time) / float(number_of_received_particle_results - 1);
			long seconds_remaining = float((output_classification->number_of_particles * output_classification->percent_used * 0.01) - number_of_received_particle_results) * seconds_per_job;

			wxTimeSpan time_remaining = wxTimeSpan(0,0,seconds_remaining);

			my_parent->TimeRemainingText->SetLabel(time_remaining.Format("Time Remaining : %Hh:%Mm:%Ss"));
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
		output_classification->classification_results[current_image].amplitude_contrast = result_to_process->result_data[7];
		output_classification->classification_results[current_image].pixel_size = result_to_process->result_data[8];
		output_classification->classification_results[current_image].microscope_voltage_kv = result_to_process->result_data[9];
		output_classification->classification_results[current_image].microscope_spherical_aberration_mm = result_to_process->result_data[10];
		output_classification->classification_results[current_image].beam_tilt_x = result_to_process->result_data[11];
		output_classification->classification_results[current_image].beam_tilt_y = result_to_process->result_data[12];
		output_classification->classification_results[current_image].image_shift_x = result_to_process->result_data[13];
		output_classification->classification_results[current_image].image_shift_y = result_to_process->result_data[14];
		output_classification->classification_results[current_image].defocus_1 = result_to_process->result_data[15];
		output_classification->classification_results[current_image].defocus_2 = result_to_process->result_data[16];
		output_classification->classification_results[current_image].defocus_angle = result_to_process->result_data[17];
		output_classification->classification_results[current_image].phase_shift = result_to_process->result_data[18];

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
			else if (current_percentage < 0) current_percentage = 0;

			my_parent->ProgressBar->SetValue(current_percentage);

			long job_time = current_time - current_job_starttime;
			float seconds_per_job = float(job_time) / float(number_of_received_particle_results - 1);
			long seconds_remaining = float((input_classification->number_of_particles) - number_of_received_particle_results) * seconds_per_job;

			wxTimeSpan time_remaining = wxTimeSpan(0,0,seconds_remaining);
			my_parent->TimeRemainingText->SetLabel(time_remaining.Format("Time Remaining : %Hh:%Mm:%Ss"));		}
	}
}

void ClassificationManager::ProcessAllJobsFinished()
{

	// Update the GUI with project timings
	extern MyOverviewPanel *overview_panel;
	overview_panel->SetProjectInfo();

	if (running_job_type == STARTUP)
	{
		main_frame->job_controller.KillJob(my_parent->my_job_id);
		//my_parent->WriteBlueText("Done.");
		CycleRefinement();
	}
	else
	if (running_job_type == REFINEMENT)
	{
		main_frame->job_controller.KillJob(my_parent->my_job_id);
		RunMerge2dJob();
	}
	else
	if (running_job_type == MERGE)
	{
		main_frame->job_controller.KillJob(my_parent->my_job_id);
		RemoveFilesFromScratch();
		//global_delete_refine2d_scratch();
		CycleRefinement();
	}
}

void ClassificationManager::CycleRefinement()
{



	// add classification
	main_frame->current_project.database.AddClassification(output_classification);

	ShortClassificationInfo temp_info;
	temp_info = output_classification;
	refinement_package_asset_panel->all_classification_short_infos.Add(temp_info);

	// add classification id to the refinement package..

	active_refinement_package->classification_ids.Add(output_classification->classification_id);
	main_frame->current_project.database.ExecuteSQL(wxString::Format("INSERT INTO REFINEMENT_PACKAGE_CLASSIFICATIONS_LIST_%li (CLASSIFICATION_NUMBER, CLASSIFICATION_ID) VALUES (%li, %li);", current_refinement_package_asset_id, active_refinement_package->classification_ids.GetCount(),  output_classification->classification_id));


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

		my_parent->ResultDisplayPanel->OpenFile(output_classification->class_average_file, wxString::Format("Class. #%li (Round %i)", output_classification->classification_id, number_of_rounds_run), NULL, true);


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

				if (long(abs(output_classification->classification_results[counter].best_class)) != long(abs(input_classification->classification_results[counter].best_class))) number_moved++;
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

			running_job_type = NOJOB;
			my_parent->WriteBlueText("All refinement cycles are finished!");
			my_parent->CancelAlignmentButton->Show(false);
			global_delete_refine2d_scratch();
			my_parent->FinishButton->Show(true);
			my_parent->TimeRemainingText->SetLabel(wxString::Format("All Done! (%s)", wxTimeSpan::Milliseconds(my_parent->stopwatch.Time()).Format(wxT("%Hh:%Mm:%Ss"))));
			my_parent->ProgressBar->SetValue(100);
			my_parent->ProgressPanel->Layout();
			my_parent->input_params_combo_is_dirty = true;

		}

	}

	main_frame->DirtyClassifications();
}


