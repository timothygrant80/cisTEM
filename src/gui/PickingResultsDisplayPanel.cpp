//#include "../core/core_headers.h"
#include "../core/gui_core_headers.h"

PickingResultsDisplayPanel::PickingResultsDisplayPanel(wxWindow* parent, wxWindowID id, const wxPoint& pos, const wxSize& size, long style)
: PickingResultsDisplayParentPanel(parent, id, pos, size, style)
{
	//Bind(wxEVT_COMBOBOX, &ShowPickingResultsPanel::OnFitTypeRadioButton, this);
	//Bind(wxEVT_COMBOBOX, &ShowPickingResultsPanel::OnFitTypeRadioButton, this);

	//CTF2DResultsPanel->font_size_multiplier = 1.5;

	PickingResultsImagePanel->UnsetToolTip();

	LowResFilterTextCtrl->SetMinMaxValue(0.1, FLT_MAX);
	LowResFilterTextCtrl->SetPrecision(0);

}

PickingResultsDisplayPanel::~PickingResultsDisplayPanel()
{
	//Unbind(wxEVT_COMBOBOX, &ShowCTFResultsPanel::OnFitTypeRadioButton, this);
	//Unbind(wxEVT_COMBOBOX, &ShowCTFResultsPanel::OnFitTypeRadioButton, this);
	PickingResultsImagePanel->UnsetToolTip();

}

void PickingResultsDisplayPanel::Clear()
{
	PickingResultsImagePanel->should_show = false;
	PickingResultsImagePanel->UnsetToolTip();
	Refresh();
}

void PickingResultsDisplayPanel::Draw(const wxString &image_filename, ArrayOfParticlePositionAssets &array_of_assets, const float particle_radius_in_angstroms, const float pixel_size_in_angstroms, CTF micrograph_ctf, int image_asset_id, float iciness)
{


	// Don't do this - it deallocates the images
	//PickingResultsImagePanel->Clear();

	PickingResultsImagePanel->SetImageFilename(image_filename,pixel_size_in_angstroms,micrograph_ctf);
	PickingResultsImagePanel->SetParticleCoordinatesAndRadius(array_of_assets, particle_radius_in_angstroms);
	PickingResultsImagePanel->UpdateScalingAndDimensions();
	PickingResultsImagePanel->UpdateImageInBitmap();
	PickingResultsImagePanel->should_show = true;
	PickingResultsImagePanel->SetToolTip(wxString::Format(wxT("%i coordinates picked"),int(array_of_assets.GetCount())));
	SetNumberOfPickedCoordinates(int(array_of_assets.Count()));
	SetImageAssetID(image_asset_id);
	SetDefocus(0.5*(micrograph_ctf.GetDefocus1()+micrograph_ctf.GetDefocus2())*pixel_size_in_angstroms);
	SetIciness(iciness);

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

void PickingResultsDisplayPanel::OnWienerFilterCheckBox(wxCommandEvent & event)
{
	if (WienerFilterCheckBox->IsChecked())
	{
		if (! PickingResultsImagePanel->should_wiener_filter)
		{
			PickingResultsImagePanel->should_wiener_filter = true;
			PickingResultsImagePanel->UpdateImageInBitmap(true);
			PickingResultsImagePanel->Refresh();
		}
	}
	else
	{
		if (PickingResultsImagePanel->should_wiener_filter)
		{
			PickingResultsImagePanel->should_wiener_filter = false;
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

void PickingResultsDisplayPanel::SetNumberOfPickedCoordinates(int number_of_coordinates)
{
	NumberOfPicksStaticText->SetLabel(wxString::Format(wxT("%i picked coordinates"),number_of_coordinates));
}

void PickingResultsDisplayPanel::SetImageAssetID(int image_asset_id)
{
	ImageIDStaticText->SetLabel(wxString::Format(wxT("Image ID: %i"),image_asset_id));
}

void PickingResultsDisplayPanel::SetIciness(float iciness)
{
	IcinessStaticText->SetLabel(wxString::Format(wxT("Iciness: %.2f"),iciness));
}

void PickingResultsDisplayPanel::SetDefocus(float defocus_in_angstroms)
{
	DefocusStaticText->SetLabel(wxString::Format(wxT("Defocus: %.2f μm"),defocus_in_angstroms/10000.0));
}
