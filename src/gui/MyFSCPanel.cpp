#include "../core/gui_core_headers.h"

MyFSCPanel::MyFSCPanel( wxWindow* parent, wxWindowID id, const wxPoint& pos, const wxSize& size, long style)
:
FSCPanel( parent, id, pos, size, style )
{
	wxLogNull *suppress_png_warnings = new wxLogNull;
	#include "icons/show_text.cpp"
	#include "icons/small_save_icon.cpp"
	//#include "icons/notepad.cpp"
	wxBitmap notepad_bmp = wxBITMAP_PNG_FROM_DATA(show_text);
	wxBitmap save_bmp = wxBITMAP_PNG_FROM_DATA(small_save_icon);

	FSCDetailsButton->SetBitmap(notepad_bmp);
	SaveButton->SetBitmap(save_bmp);

	highlighted_class = -1;


}

void MyFSCPanel::HighlightClass(int wanted_class)
{
	if (highlighted_class != wanted_class) PlotPanel->HighlightClass(wanted_class);
}

void MyFSCPanel::PopupTextClick( wxCommandEvent& event )
{
	PopupTextDialog *text_dialog = new PopupTextDialog(this, wxID_ANY, "FSC Info");

	int class_counter;
	int dash_counter;
	int bin_counter;

	for (class_counter = 0; class_counter < my_refinement->number_of_classes; class_counter++)
	{
		wxString title_string = wxString::Format(wxT(" Class %i - Estimated Res = %.2f Å (Refinement Limit = %.2f Å)\n"), class_counter + 1, my_refinement->class_refinement_results[class_counter].class_resolution_statistics.ReturnEstimatedResolution(), my_refinement->class_refinement_results[class_counter].high_resolution_limit);
		text_dialog->OutputTextCtrl->AppendText(title_string);
		text_dialog->OutputTextCtrl->AppendText(" ");
		for (dash_counter = 0; dash_counter < title_string.Length()-1; dash_counter++)
		{
			text_dialog->OutputTextCtrl->AppendText("-");
		}

		text_dialog->OutputTextCtrl->AppendText("\n\n");
		text_dialog->OutputTextCtrl->AppendText(wxT(" Shell | Res.(Å) | Radius |  FSC  | Part. FSC | √Part. SSNR | √Rec. SSNR\n"));
		text_dialog->OutputTextCtrl->AppendText(wxT(" -----   -------   ------    ---   ----------   -----------   ----------\n"));

		for (int i = 1; i < my_refinement->class_refinement_results[class_counter].class_resolution_statistics.number_of_bins; i++)
		{
			text_dialog->OutputTextCtrl->AppendText(wxString::Format("%4i    %8.2f   %6.4f   %6.3f    %6.3f      %8.3f      %8.3f\n", i+1, my_refinement->class_refinement_results[class_counter].class_resolution_statistics.FSC.data_x[i],  my_refinement->resolution_statistics_pixel_size / my_refinement->class_refinement_results[class_counter].class_resolution_statistics.FSC.data_x[i], my_refinement->class_refinement_results[class_counter].class_resolution_statistics.FSC.data_y[i], my_refinement->class_refinement_results[class_counter].class_resolution_statistics.part_FSC.data_y[i], sqrtf(my_refinement->class_refinement_results[class_counter].class_resolution_statistics.part_SSNR.data_y[i]), sqrtf(my_refinement->class_refinement_results[class_counter].class_resolution_statistics.rec_SSNR.data_y[i])));
		}

		text_dialog->OutputTextCtrl->AppendText("\n");
	}

	text_dialog->OutputTextCtrl->ShowPosition(0);
	text_dialog->ShowModal();
	text_dialog->Destroy();
}

void MyFSCPanel::SaveImageClick( wxCommandEvent& event )
{
	ProperOverwriteCheckSaveDialog *saveFileDialog;
	saveFileDialog = new ProperOverwriteCheckSaveDialog(this, _("Save png image"), "PNG files (*.png)|*.png", ".png");
	if (saveFileDialog->ShowModal() == wxID_CANCEL)
	{
		saveFileDialog->Destroy();
		return;
	}

	// save the file then..

	PlotPanel->current_plot_window->SaveScreenshot(saveFileDialog->ReturnProperPath(), wxBITMAP_TYPE_PNG,wxSize(800,400));
	saveFileDialog->Destroy();
}

void MyFSCPanel::AddRefinement(Refinement *refinement_to_plot)
{
	PlotPanel->Clear();
	my_refinement = refinement_to_plot;
	float smallest_resolution_limit = FLT_MAX;
	float highest_resolution = FLT_MAX;

	for (int class_counter = 0; class_counter < my_refinement->number_of_classes; class_counter++)
	{
		PlotPanel->AddPartFSC(&refinement_to_plot->class_refinement_results[class_counter].class_resolution_statistics, refinement_to_plot->resolution_statistics_pixel_size * 2.0);
		smallest_resolution_limit = std::min(my_refinement->class_refinement_results[class_counter].high_resolution_limit, smallest_resolution_limit);
		highest_resolution = std::min(my_refinement->class_refinement_results[class_counter].class_resolution_statistics.ReturnEstimatedResolution(), highest_resolution);
	}

	PlotPanel->SetupBaseLayers();
	EstimatedResolutionText->SetLabel(wxString::Format(wxT("%.2f Å"), highest_resolution));
	TitleSizer->Layout();
	PlotPanel->current_refinement_resolution_limit = 1.0 /  smallest_resolution_limit;
	PlotPanel->Draw(my_refinement->resolution_statistics_pixel_size * 2.0);
}

void MyFSCPanel::Clear()
{
	PlotPanel->Clear(true);
	EstimatedResolutionText->SetLabel("");
}
