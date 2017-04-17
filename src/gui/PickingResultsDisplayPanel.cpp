//#include "../core/core_headers.h"
#include "../core/gui_core_headers.h"

PickingResultsDisplayPanel::PickingResultsDisplayPanel(wxWindow* parent, wxWindowID id, const wxPoint& pos, const wxSize& size, long style)
: PickingResultsDisplayParentPanel(parent, id, pos, size, style)
{
	//Bind(wxEVT_COMBOBOX, &ShowPickingResultsPanel::OnFitTypeRadioButton, this);
	//Bind(wxEVT_COMBOBOX, &ShowPickingResultsPanel::OnFitTypeRadioButton, this);

	//CTF2DResultsPanel->font_size_multiplier = 1.5;

	LowResFilterTextCtrl->SetMinMaxValue(0.1, FLT_MAX);

}

PickingResultsDisplayPanel::~PickingResultsDisplayPanel()
{
	//Unbind(wxEVT_COMBOBOX, &ShowCTFResultsPanel::OnFitTypeRadioButton, this);
	//Unbind(wxEVT_COMBOBOX, &ShowCTFResultsPanel::OnFitTypeRadioButton, this);

}

void PickingResultsDisplayPanel::Clear()
{
	PickingResultsImagePanel->should_show = false;
	Refresh();
}

void PickingResultsDisplayPanel::Draw(const wxString &image_filename, ArrayOfParticlePositionAssets &array_of_assets, const float particle_radius_in_angstroms, const float pixel_size_in_angstroms)
{


	// Don't do this - it deallocates the images
	//PickingResultsImagePanel->Clear();

	PickingResultsImagePanel->SetImageFilename(image_filename,pixel_size_in_angstroms);

	PickingResultsImagePanel->SetParticleCoordinatesAndRadius(array_of_assets, particle_radius_in_angstroms);

	PickingResultsImagePanel->UpdateScalingAndDimensions();

	PickingResultsImagePanel->UpdateImageInBitmap();

	PickingResultsImagePanel->should_show = true;

	PickingResultsImagePanel->Refresh();


}

void PickingResultsDisplayPanel::OnCirclesAroundParticlesCheckBox(wxCommandEvent & event)
{
	if (CirclesAroundParticlesCheckBox->IsChecked())
	{
		if (! PickingResultsImagePanel->draw_circles_around_particles)
		{
			PickingResultsImagePanel->draw_circles_around_particles = true;
			PickingResultsImagePanel->Refresh();
		}
	}
	else
	{
		if (PickingResultsImagePanel->draw_circles_around_particles)
		{
			PickingResultsImagePanel->draw_circles_around_particles = false;
			PickingResultsImagePanel->Refresh();
		}
	}
}

void PickingResultsDisplayPanel::OnScaleBarCheckBox(wxCommandEvent & event)
{
	if (ScaleBarCheckBox->IsChecked())
	{
		if (! PickingResultsImagePanel->draw_scale_bar)
		{
			PickingResultsImagePanel->draw_scale_bar = true;
			PickingResultsImagePanel->Refresh();
		}
	}
	else
	{
		if (PickingResultsImagePanel->draw_scale_bar)
		{
			PickingResultsImagePanel->draw_scale_bar = false;
			PickingResultsImagePanel->Refresh();
		}
	}
}

void PickingResultsDisplayPanel::OnLowPassEnter(wxCommandEvent& event)
{
	if (PickingResultsImagePanel->should_low_pass == true)
	{
		PickingResultsImagePanel->low_res_filter_value = LowResFilterTextCtrl->ReturnValue();
		PickingResultsImagePanel->UpdateImageInBitmap(true);
		PickingResultsImagePanel->Refresh();
	}
	event.Skip();


}

void PickingResultsDisplayPanel::OnLowPassKillFocus(wxFocusEvent& event)
{
	if (PickingResultsImagePanel->should_low_pass == true)
	{
		PickingResultsImagePanel->low_res_filter_value = LowResFilterTextCtrl->ReturnValue();
		PickingResultsImagePanel->UpdateImageInBitmap(true);
		PickingResultsImagePanel->Refresh();
	}
	event.Skip();
}

void PickingResultsDisplayPanel::OnHighPassFilterCheckBox(wxCommandEvent & event)
{
	if (HighPassFilterCheckBox->IsChecked())
	{
		if (! PickingResultsImagePanel->should_high_pass)
		{
			PickingResultsImagePanel->should_high_pass = true;
			PickingResultsImagePanel->UpdateImageInBitmap(true);
			PickingResultsImagePanel->Refresh();
		}
	}
	else
	{
		if (PickingResultsImagePanel->should_high_pass)
		{
			PickingResultsImagePanel->should_high_pass = false;
			PickingResultsImagePanel->UpdateImageInBitmap(true);
			PickingResultsImagePanel->Refresh();
		}
	}
}

void PickingResultsDisplayPanel::OnLowPassFilterCheckBox(wxCommandEvent & event)
{
	if (LowPassFilterCheckBox->IsChecked())
	{
		LowResFilterTextCtrl->Enable(true);
		LowAngstromStatic->Enable(true);

		PickingResultsImagePanel->low_res_filter_value = LowResFilterTextCtrl->ReturnValue();

		PickingResultsImagePanel->should_low_pass = true;
		PickingResultsImagePanel->UpdateImageInBitmap(true);
		PickingResultsImagePanel->Refresh();

	}
	else
	{
		PickingResultsImagePanel->low_res_filter_value = -1.0;
		PickingResultsImagePanel->should_low_pass = false;
		PickingResultsImagePanel->UpdateImageInBitmap(true);
		PickingResultsImagePanel->Refresh();
	}
}

void PickingResultsDisplayPanel::OnUndoButtonClick(wxCommandEvent& event)
{
	PickingResultsImagePanel->StepBackwardInHistoryOfParticleCoordinates();
}

void PickingResultsDisplayPanel::OnRedoButtonClick(wxCommandEvent& event)
{
	PickingResultsImagePanel->StepForwardInHistoryOfParticleCoordinates();
}
