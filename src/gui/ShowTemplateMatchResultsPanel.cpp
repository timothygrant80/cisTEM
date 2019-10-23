//#include "../core/core_headers.h"
#include "../core/gui_core_headers.h"
extern MyImageAssetPanel *image_asset_panel;
extern MyVolumeAssetPanel *volume_asset_panel;

ShowTemplateMatchResultsPanel::ShowTemplateMatchResultsPanel(wxWindow* parent, wxWindowID id, const wxPoint& pos, const wxSize& size, long style)
: ShowTemplateMatchResultsParentPanel(parent, id, pos, size, style)
{
	HistogramPlotPanel->Initialise(wxT("Correlation Value"), "", false, false, 20, 50, 60, 20, true, false, true);
	ImageDisplayPanel->Initialise( FIRST_LOCATION_ONLY | START_WITH_AUTO_CONTRAST | START_WITH_FOURIER_SCALING | DO_NOT_SHOW_STATUS_BAR | SKIP_LEFTCLICK_TO_PARENT);

	PeakListCtrl->Bind(wxEVT_LIST_ITEM_SELECTED, &ShowTemplateMatchResultsPanel::OnPeakListSelectionChange, this);
	PeakListCtrl->Bind(wxEVT_LIST_ITEM_DESELECTED, &ShowTemplateMatchResultsPanel::OnPeakListSelectionChange, this);

	ChangesListCtrl->Bind(wxEVT_LIST_ITEM_SELECTED, &ShowTemplateMatchResultsPanel::OnChangeListSelectionChange, this);
	ChangesListCtrl->Bind(wxEVT_LIST_ITEM_DESELECTED, &ShowTemplateMatchResultsPanel::OnChangeListSelectionChange, this);

	ImageDisplayPanel->Bind(wxEVT_LEFT_DOWN, &ShowTemplateMatchResultsPanel::OnImageLeftClick, this);

	wxLogNull *suppress_png_warnings = new wxLogNull;
	#include "icons/small_save_icon.cpp"
	wxBitmap save_bmp = wxBITMAP_PNG_FROM_DATA(small_save_icon);

	SaveButton->SetBitmap(save_bmp);


}

void ShowTemplateMatchResultsPanel::OnSavePeaksClick( wxCommandEvent& event )
{
	if (current_result.found_peaks.GetCount() > 0)
	{
		ProperOverwriteCheckSaveDialog *saveFileDialog;
		saveFileDialog = new ProperOverwriteCheckSaveDialog(this, _("Save peak .txt file"), "TXT files (*.txt)|*.txt", ".txt");

		if (saveFileDialog->ShowModal() == wxID_CANCEL)
		{
			saveFileDialog->Destroy();
			return;
		}

		// save the file then..

		float coordinates[8];

		NumericTextFile coordinate_file(saveFileDialog->ReturnProperPath(), OPEN_TO_WRITE, 8);
		wxDateTime wxdatetime_of_run;
		wxdatetime_of_run.SetFromDOS((unsigned long) current_result.datetime_of_run);

		coordinate_file.WriteCommentLine(wxString::Format("Template Match Result #%li (%s - %s, %s)", current_result.job_name, wxdatetime_of_run.FormatISODate(), wxdatetime_of_run.FormatISOTime()).ToUTF8().data());
		coordinate_file.WriteCommentLine(wxString::Format("Searched Image               : %s", image_asset_panel->ReturnAssetLongFilename(image_asset_panel->ReturnArrayPositionFromAssetID(current_result.image_asset_id))).ToUTF8().data());
		coordinate_file.WriteCommentLine(wxString::Format("Reference Volume File        : %s", volume_asset_panel->ReturnAssetLongFilename(volume_asset_panel->ReturnArrayPositionFromAssetID(current_result.ref_volume_asset_id))).ToUTF8().data());
		coordinate_file.WriteCommentLine(wxString::Format("Used Result Threshold        : %.2f", current_result.used_threshold).ToUTF8().data());
		coordinate_file.WriteCommentLine(wxString::Format("MIP File                     : %s", current_result.mip_filename).ToUTF8().data());
		coordinate_file.WriteCommentLine(wxString::Format("Scaled MIP File              : %s", current_result.scaled_mip_filename).ToUTF8().data());
		coordinate_file.WriteCommentLine(wxString::Format("Best Psi File                : %s", current_result.psi_filename).ToUTF8().data());
		coordinate_file.WriteCommentLine(wxString::Format("Best Theta File              : %s", current_result.theta_filename).ToUTF8().data());
		coordinate_file.WriteCommentLine(wxString::Format("Best Phi File                : %s", current_result.phi_filename).ToUTF8().data());
		coordinate_file.WriteCommentLine(wxString::Format("Best Defocus File            : %s", current_result.defocus_filename).ToUTF8().data());
		coordinate_file.WriteCommentLine(wxString::Format("Best Pixel Size File         : %s", current_result.pixel_size_filename).ToUTF8().data());
		coordinate_file.WriteCommentLine(wxString::Format("Histogram File               : %s", current_result.histogram_filename).ToUTF8().data());
		coordinate_file.WriteCommentLine(wxString::Format("2D Projected result File     : %s", current_result.projection_result_filename).ToUTF8().data());
		coordinate_file.WriteCommentLine(wxString::Format("Used Symmetry                : %s", current_result.symmetry).ToUTF8().data());
		coordinate_file.WriteCommentLine(wxString::Format("Used Pixel Size              : %.4f", current_result.pixel_size).ToUTF8().data());
		coordinate_file.WriteCommentLine(wxString::Format("Used Spherical Aberration    : %.2f", current_result.spherical_aberration).ToUTF8().data());
		coordinate_file.WriteCommentLine(wxString::Format("Used Amplitude Contrast      : %.2f", current_result.amplitude_contrast).ToUTF8().data());
		coordinate_file.WriteCommentLine(wxString::Format("Used Defocus 1               : %.2f", current_result.defocus1).ToUTF8().data());
		coordinate_file.WriteCommentLine(wxString::Format("Used Defocus 2               : %.2f", current_result.defocus2).ToUTF8().data());
		coordinate_file.WriteCommentLine(wxString::Format("Used Defocus Angle           : %.2f", current_result.defocus_angle).ToUTF8().data());
		coordinate_file.WriteCommentLine(wxString::Format("Used Phase Shift             : %.2f", current_result.phase_shift).ToUTF8().data());
		coordinate_file.WriteCommentLine(wxString::Format("Used Low-Res Limit           : %.2f", current_result.low_res_limit).ToUTF8().data());
		coordinate_file.WriteCommentLine(wxString::Format("Used High-Res Limit          : %.2f", current_result.high_res_limit).ToUTF8().data());
		coordinate_file.WriteCommentLine(wxString::Format("Used Out of Plane Step       : %.2f", current_result.out_of_plane_step).ToUTF8().data());
		coordinate_file.WriteCommentLine(wxString::Format("Used In Plane Step           : %.2f", current_result.in_plane_step).ToUTF8().data());
		coordinate_file.WriteCommentLine(wxString::Format("Used Defocus Search Range    : %.2f", current_result.defocus_search_range).ToUTF8().data());
		coordinate_file.WriteCommentLine(wxString::Format("Used Defocus Step Size       : %.2f", current_result.defocus_step).ToUTF8().data());
		coordinate_file.WriteCommentLine(wxString::Format("Used Pixel Size Search Range : %.2f", current_result.pixel_size_search_range).ToUTF8().data());
		coordinate_file.WriteCommentLine(wxString::Format("Used Pixel Size Step Size    : %.2f", current_result.pixel_size_step).ToUTF8().data());
		coordinate_file.WriteCommentLine("");
		coordinate_file.WriteCommentLine("         Psi          Theta            Phi              X              Y              Z      PixelSize           Peak");

		for (int counter = 0; counter < current_result.found_peaks.GetCount(); counter++)
		{
			coordinates[0] = current_result.found_peaks[counter].psi;
			coordinates[1] = current_result.found_peaks[counter].theta;
			coordinates[2] = current_result.found_peaks[counter].phi;
			coordinates[3] = current_result.found_peaks[counter].x_pos;
			coordinates[4] = current_result.found_peaks[counter].y_pos;
			coordinates[5] = current_result.found_peaks[counter].defocus;
			coordinates[6] = current_result.found_peaks[counter].pixel_size;
			coordinates[7] = current_result.found_peaks[counter].peak_height;

			coordinate_file.WriteLine(coordinates);
		}

		saveFileDialog->Destroy();
	}
}


ShowTemplateMatchResultsPanel::~ShowTemplateMatchResultsPanel()
{
	PeakListCtrl->Unbind(wxEVT_LIST_ITEM_SELECTED, &ShowTemplateMatchResultsPanel::OnPeakListSelectionChange, this);
	PeakListCtrl->Unbind(wxEVT_LIST_ITEM_DESELECTED, &ShowTemplateMatchResultsPanel::OnPeakListSelectionChange, this);
	ImageDisplayPanel->Unbind(wxEVT_LEFT_DOWN, &ShowTemplateMatchResultsPanel::OnImageLeftClick, this);
}

void ShowTemplateMatchResultsPanel::OnPeakListSelectionChange(wxListEvent& event)
{
	if (PeakListCtrl->GetSelectedItemCount() == 0) ImageDisplayPanel->ClearActiveTemplateMatchMarker();
	else
	{
		int selected_peak  = PeakListCtrl->GetNextItem(-1, wxLIST_NEXT_ALL, wxLIST_STATE_SELECTED);
		ImageDisplayPanel->SetActiveTemplateMatchMarkerPostion(current_result.found_peaks[selected_peak].x_pos / current_result.pixel_size, current_result.found_peaks[selected_peak].y_pos / current_result.pixel_size, (current_result.reference_box_size_in_angstroms / current_result.pixel_size) * 0.5f);

		// if this is a refinement, also select the appropriate peak

		if (current_result.job_type == TEMPLATE_MATCH_REFINEMENT)
		{
			int selected_change;
			if (ChangesListCtrl->GetSelectedItemCount()> 0) selected_change = ChangesListCtrl->GetNextItem(-1, wxLIST_NEXT_ALL, wxLIST_STATE_SELECTED);
			else selected_change = -1;

			if (selected_change != selected_peak)
			{
				ChangesListCtrl->SetItemState(selected_peak, wxLIST_STATE_SELECTED, wxLIST_STATE_SELECTED);
				ChangesListCtrl->EnsureVisible(selected_peak);
			}
		}
	}
}

void ShowTemplateMatchResultsPanel::OnChangeListSelectionChange(wxListEvent& event)
{
	if (ChangesListCtrl->GetSelectedItemCount()> 0 && current_result.job_type == TEMPLATE_MATCH_REFINEMENT)
	{
		int selected_change = ChangesListCtrl->GetNextItem(-1, wxLIST_NEXT_ALL, wxLIST_STATE_SELECTED);

		if (current_result.peak_changes[selected_change].new_peak_number != -1)
		{
			int selected_peak;
			if (PeakListCtrl->GetSelectedItemCount() == 0) selected_peak = PeakListCtrl->GetNextItem(-1, wxLIST_NEXT_ALL, wxLIST_STATE_SELECTED);
			else selected_peak = -1;

			if (selected_peak != current_result.peak_changes[selected_change].new_peak_number - 1)
			{
				PeakListCtrl->SetItemState(current_result.peak_changes[selected_change].new_peak_number - 1, wxLIST_STATE_SELECTED, wxLIST_STATE_SELECTED);
				PeakListCtrl->EnsureVisible(current_result.peak_changes[selected_change].new_peak_number - 1);
			}
		}
	}
}

void ShowTemplateMatchResultsPanel::SetActiveResult(TemplateMatchJobResults &result_to_show)
{
	current_result = result_to_show;

	Freeze();
	BottomPanel->Freeze();

	wxFileName input_file = image_asset_panel->ReturnAssetLongFilename(image_asset_panel->ReturnArrayPositionFromAssetID(current_result.image_asset_id));

	if (current_result.job_type == TEMPLATE_MATCH_FULL_SEARCH)
	{
		SetHistogramLabelText(wxString::Format("Survival Histogram (%s)", current_result.job_name));
		DrawHistogram(current_result.histogram_filename);
		HistogramPlotPanel->Show(true);
		PeakChangesPanel->Show(false);
	}
	else
	{
		SetHistogramLabelText(wxString::Format("Refined Peak Changes (%s - Threshold : %.2f) ",  current_result.job_name, current_result.refinement_threshold));
		HistogramPlotPanel->Show(false);
		PeakChangesPanel->Show(true);
	}

	FillPeakInfoTable(current_result.used_threshold);
	BottomPanel->Layout();
	Thaw();
	BottomPanel->Thaw();

	wxYield();

	if (ImageDisplayPanel->my_notebook->GetPageCount() == 0)
	{
		ImageDisplayPanel->OpenFile(input_file.GetFullPath(), input_file.GetName());
		ImageDisplayPanel->OpenFile(current_result.scaled_mip_filename, "Scaled MIP", NULL, false, true);
		DisplayNotebookPanel *current_panel = reinterpret_cast <DisplayNotebookPanel *> (ImageDisplayPanel->my_notebook->GetPage(1));
		current_panel->use_unscaled_image_for_popup = true;
		ImageDisplayPanel->OpenFile(current_result.projection_result_filename, "Plotted Result",  NULL, false, true);
	}
	else
	{

	/*	ImageDisplayPanel->ClearActiveTemplateMatchMarker();
		int current_selection = ImageDisplayPanel->my_notebook->GetSelection();
		ImageDisplayPanel->my_notebook->ChangeSelection(0);
		ImageDisplayPanel->ChangeFile(input_file.GetFullPath(), input_file.GetName());
		ImageDisplayPanel->my_notebook->ChangeSelection(1);
		ImageDisplayPanel->ChangeFile(current_result.scaled_mip_filename, "Scaled MIP");
		ImageDisplayPanel->my_notebook->ChangeSelection(2);
		ImageDisplayPanel->ChangeFile(current_result.projection_result_filename, "Plotted Result");
		ImageDisplayPanel->my_notebook->ChangeSelection(current_selection);
		*/

		ImageDisplayPanel->ClearActiveTemplateMatchMarker();
		ImageDisplayPanel->ChangeFileForTabNumber(0, input_file.GetFullPath(), input_file.GetName());
		ImageDisplayPanel->ChangeFileForTabNumber(1, current_result.scaled_mip_filename, "Scaled MIP");
		ImageDisplayPanel->ChangeFileForTabNumber(2, current_result.projection_result_filename, "Plotted Result");
	}
}

void ShowTemplateMatchResultsPanel::ClearPeakList()
{
	PeakListCtrl->Freeze();
	PeakListCtrl->ClearAll();
	PeakListCtrl->InsertColumn(0, wxT("Peak No."), wxLIST_FORMAT_CENTRE,  wxLIST_AUTOSIZE_USEHEADER );
	PeakListCtrl->InsertColumn(1, wxT("X-Pos."), wxLIST_FORMAT_CENTRE,  wxLIST_AUTOSIZE_USEHEADER );
	PeakListCtrl->InsertColumn(2, wxT("Y-Pos."), wxLIST_FORMAT_CENTRE,  wxLIST_AUTOSIZE_USEHEADER );
	PeakListCtrl->InsertColumn(3, wxT("Ψ"), wxLIST_FORMAT_CENTRE,  wxLIST_AUTOSIZE_USEHEADER );
	PeakListCtrl->InsertColumn(4, wxT("Θ"), wxLIST_FORMAT_CENTRE,  wxLIST_AUTOSIZE_USEHEADER );
	PeakListCtrl->InsertColumn(5, wxT("Φ"), wxLIST_FORMAT_CENTRE,  wxLIST_AUTOSIZE_USEHEADER );
	PeakListCtrl->InsertColumn(6, wxT("Defocus"), wxLIST_FORMAT_CENTRE,  wxLIST_AUTOSIZE_USEHEADER );
	PeakListCtrl->InsertColumn(7, wxT("Pixel Size"), wxLIST_FORMAT_CENTRE,  wxLIST_AUTOSIZE_USEHEADER );
	PeakListCtrl->InsertColumn(8, wxT("Peak Height"), wxLIST_FORMAT_CENTRE,  wxLIST_AUTOSIZE_USEHEADER );
	PeakListCtrl->Thaw();

	ChangesListCtrl->Freeze();
	ChangesListCtrl->ClearAll();
	ChangesListCtrl->InsertColumn(0, wxT("Peak No."), wxLIST_FORMAT_CENTRE,  wxLIST_AUTOSIZE_USEHEADER );
	ChangesListCtrl->InsertColumn(1, wxT("Input Peak"), wxLIST_FORMAT_CENTRE,  wxLIST_AUTOSIZE_USEHEADER );
	ChangesListCtrl->InsertColumn(2, wxT("ΔX"), wxLIST_FORMAT_CENTRE,  wxLIST_AUTOSIZE_USEHEADER );
	ChangesListCtrl->InsertColumn(3, wxT("ΔY"), wxLIST_FORMAT_CENTRE,  wxLIST_AUTOSIZE_USEHEADER );
	ChangesListCtrl->InsertColumn(4, wxT("ΔΨ"), wxLIST_FORMAT_CENTRE,  wxLIST_AUTOSIZE_USEHEADER );
	ChangesListCtrl->InsertColumn(5, wxT("ΔΘ"), wxLIST_FORMAT_CENTRE,  wxLIST_AUTOSIZE_USEHEADER );
	ChangesListCtrl->InsertColumn(6, wxT("ΔΦ"), wxLIST_FORMAT_CENTRE,  wxLIST_AUTOSIZE_USEHEADER );
	ChangesListCtrl->InsertColumn(7, wxT("ΔDefocus"), wxLIST_FORMAT_CENTRE,  wxLIST_AUTOSIZE_USEHEADER );
	ChangesListCtrl->InsertColumn(8, wxT("ΔPx. Size"), wxLIST_FORMAT_CENTRE,  wxLIST_AUTOSIZE_USEHEADER );
	ChangesListCtrl->InsertColumn(9, wxT("ΔPeak Height"), wxLIST_FORMAT_CENTRE,  wxLIST_AUTOSIZE_USEHEADER );
	ChangesListCtrl->Thaw();
}

void ShowTemplateMatchResultsPanel::Clear(bool show_peak_change_window)
{
	Freeze();
	SetPeakTableLabelText("Peaks Above Threshold");
	HistogramPlotPanel->Clear();
	ImageDisplayPanel->Clear();
	current_result.found_peaks.Clear();
	current_result.peak_changes.Clear();

	if (show_peak_change_window == false)
	{
		SetHistogramLabelText("Survival Histogram");
		HistogramPlotPanel->Show(true);
		PeakChangesPanel->Show(false);
	}
	else
	{
		SetHistogramLabelText("Refined Peak Changes");
		HistogramPlotPanel->Show(false);
		PeakChangesPanel->Show(true);
	}

	ClearPeakList();
	BottomPanel->Layout();
	Refresh();
	Thaw();
}

void ShowTemplateMatchResultsPanel::OnImageLeftClick( wxMouseEvent& event )
{
	long x_pos;
	long y_pos;
	int counter;

	event.GetPosition(&x_pos, &y_pos);

	int peak_x_pos;
	int peak_y_pos;

	DisplayNotebookPanel *current_panel = ImageDisplayPanel->ReturnCurrentPanel();

	peak_x_pos = myroundint(x_pos / current_panel->actual_scale_factor);
	peak_y_pos = myroundint((current_panel->current_y_size - y_pos + 1) / current_panel->actual_scale_factor);

	int closest_peak = -1;
	int closest_distance = INT_MAX;
	int current_distance;

	for (counter = 0; counter < current_result.found_peaks.GetCount(); counter++)
	{
		current_distance = abs(current_result.found_peaks[counter].x_pos - peak_x_pos) + abs(current_result.found_peaks[counter].y_pos - peak_y_pos);

		if (current_distance < closest_distance)
		{
			closest_distance = current_distance;
			closest_peak = counter;
		}
	}

	if (closest_peak != -1 && closest_distance < 500)
	{
		PeakListCtrl->SetItemState(closest_peak, wxLIST_STATE_SELECTED, wxLIST_STATE_SELECTED);
		PeakListCtrl->EnsureVisible(closest_peak);
	}
}

void ShowTemplateMatchResultsPanel::FillPeakInfoTable( float threshold_used)
{
	PeakListCtrl->DeleteAllItems();

	long item_index;
	int counter;

	for (counter = 0; counter <  current_result.found_peaks.GetCount(); counter++)
	{
		PeakListCtrl->InsertItem(counter, wxString::Format("%i", counter + 1));
		PeakListCtrl->SetItem(counter, 1, wxString::Format("%.2f", current_result.found_peaks[counter].x_pos));
		PeakListCtrl->SetItem(counter, 2, wxString::Format("%.2f", current_result.found_peaks[counter].y_pos));
		PeakListCtrl->SetItem(counter, 3, wxString::Format("%.2f", current_result.found_peaks[counter].psi));
		PeakListCtrl->SetItem(counter, 4, wxString::Format("%.2f", current_result.found_peaks[counter].theta));
		PeakListCtrl->SetItem(counter, 5, wxString::Format("%.2f", current_result.found_peaks[counter].phi));
		PeakListCtrl->SetItem(counter, 6, wxString::Format("%.2f", current_result.found_peaks[counter].defocus));
		PeakListCtrl->SetItem(counter, 7, wxString::Format("%.2f", current_result.found_peaks[counter].pixel_size));
		PeakListCtrl->SetItem(counter, 8, wxString::Format("%.2f", current_result.found_peaks[counter].peak_height));
	}

	if (current_result.found_peaks.GetCount() > 0)
	{
		for (counter = 1; counter < 6; counter++)
		{
			PeakListCtrl->SetColumnWidth(counter, wxLIST_AUTOSIZE);
		}
	}

	SetPeakTableLabelText(wxString::Format("Peaks Above Threshold (%li found - Threshold : %.2f)", current_result.found_peaks.GetCount(), threshold_used));

	// if it's a refinement fill the changes table..

	if (current_result.job_type == TEMPLATE_MATCH_REFINEMENT)
	{
		ChangesListCtrl->DeleteAllItems();

		for (counter = 0; counter <  current_result.peak_changes.GetCount(); counter++)
		{

			if (current_result.peak_changes[counter].new_peak_number == -1)	ChangesListCtrl->InsertItem(counter, "NA");
			else ChangesListCtrl->InsertItem(counter, wxString::Format("%i", current_result.peak_changes[counter].new_peak_number));


			if (current_result.peak_changes[counter].original_peak_number == -1) ChangesListCtrl->SetItem(counter, 1, "NA");
			else ChangesListCtrl->SetItem(counter, 1, wxString::Format("%i", current_result.peak_changes[counter].original_peak_number));

			ChangesListCtrl->SetItem(counter, 2, wxString::Format("%.2f", current_result.peak_changes[counter].x_pos));
			ChangesListCtrl->SetItem(counter, 3, wxString::Format("%.2f", current_result.peak_changes[counter].y_pos));
			ChangesListCtrl->SetItem(counter, 4, wxString::Format("%.2f", current_result.peak_changes[counter].psi));
			ChangesListCtrl->SetItem(counter, 5, wxString::Format("%.2f", current_result.peak_changes[counter].theta));
			ChangesListCtrl->SetItem(counter, 6, wxString::Format("%.2f", current_result.peak_changes[counter].phi));
			ChangesListCtrl->SetItem(counter, 7, wxString::Format("%.2f", current_result.peak_changes[counter].defocus));
			ChangesListCtrl->SetItem(counter, 8, wxString::Format("%.2f", current_result.peak_changes[counter].pixel_size));
			ChangesListCtrl->SetItem(counter, 9, wxString::Format("%.2f", current_result.peak_changes[counter].peak_height));


		}

		if (current_result.peak_changes.GetCount() > 0)
		{
			for (counter = 2; counter < 6; counter++)
			{
				ChangesListCtrl->SetColumnWidth(counter, wxLIST_AUTOSIZE);
			}
		}
	}
}


void ShowTemplateMatchResultsPanel::DrawHistogram(wxString histogram_filename)
{
	HistogramPlotPanel->Clear();

	// fill the curves..
	if (DoesFileExist(histogram_filename) == true)
	{

		NumericTextFile histogram_file(histogram_filename, OPEN_TO_READ);
		float values[20];
		float log_survival_histogram;
		float log_expected_survival_histogram;

		Curve survival_histogram;
		Curve expected_survival_histogram;

		for (int counter = 0; counter < histogram_file.number_of_lines; counter++)
		{
			histogram_file.ReadLine(values);

			if (values[0] > 0)
			{
				if (values[2] > 0) log_survival_histogram = logf(values[2]);
				else log_survival_histogram = 0.0f;

				if (values[3] > 0) log_expected_survival_histogram = logf(values[3]);
				else log_expected_survival_histogram = 0.0f;

				if (log_expected_survival_histogram == 0.0f && log_survival_histogram == 0.0f) break;

				survival_histogram.AddPoint(values[0], log_survival_histogram);
				expected_survival_histogram.AddPoint(values[0], log_expected_survival_histogram);

			}
		}

		HistogramPlotPanel->AddCurve(survival_histogram, *wxBLUE);
		HistogramPlotPanel->AddCurve(expected_survival_histogram, *wxRED, "", 1, wxPENSTYLE_DOT );

		float x_min;
		float x_max;
		float y_min;
		float y_max;

		expected_survival_histogram.GetXMinMax(x_min, x_max);
		expected_survival_histogram.GetYMinMax(y_min, y_max);

		HistogramPlotPanel->Draw(0, x_max, 1, y_max);
	}

}
