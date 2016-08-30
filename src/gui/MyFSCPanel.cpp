#include "../core/gui_core_headers.h"

MyFSCPanel::MyFSCPanel( wxWindow* parent, wxWindowID id, const wxPoint& pos, const wxSize& size, long style)
:
FSCPanel( parent, id, pos, size, style )
{

}

void MyFSCPanel::OnClassComboBoxChange( wxCommandEvent& event )
{
	PlotCurrentClass();
	event.Skip();
}

void MyFSCPanel::AddRefinement(Refinement *refinement_to_plot)
{
	my_refinement = refinement_to_plot;

	ClassComboBox->Freeze();
	ClassComboBox->Clear();
	ClassComboBox->ChangeValue("");

	for (int class_counter = 0; class_counter < my_refinement->number_of_classes; class_counter++)
	{
		ClassComboBox->Append(wxString::Format("Class #%2i", class_counter + 1));
	}

	ClassComboBox->SetSelection(0);
	ClassComboBox->Thaw();

	PlotCurrentClass();
}

void MyFSCPanel::PlotCurrentClass()
{
	int current_class = ClassComboBox->GetSelection();
	PlotPanel->Clear();
	//wxPrintf("Here\n");
	if (current_class >= 0)
	{
		EstimatedResolutionText->SetLabel(wxString::Format(wxT("%.2f Å"), my_refinement->class_refinement_results[current_class].class_resolution_statistics.ReturnEstimatedResolution()));
		TitleSizer->Layout();
		for (int point_counter = 1; point_counter < my_refinement->class_refinement_results[current_class].class_resolution_statistics.part_FSC.number_of_points; point_counter++)
		{
			PlotPanel->AddPoint(1.0 / my_refinement->class_refinement_results[current_class].class_resolution_statistics.part_FSC.data_x[point_counter], my_refinement->class_refinement_results[current_class].class_resolution_statistics.part_FSC.data_y[point_counter]);
		}

		PlotPanel->Draw(my_refinement->resolution_statistics_pixel_size * 2.0);


	}

}

void MyFSCPanel::Clear()
{
	PlotPanel->Clear();
	EstimatedResolutionText->SetLabel("");
	ClassComboBox->Freeze();
	ClassComboBox->Clear();
	ClassComboBox->ChangeValue("");
	ClassComboBox->Thaw();
}
