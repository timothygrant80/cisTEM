#include "../core/gui_core_headers.h"

extern MyMainFrame *main_frame;
extern MyImageAssetPanel *image_asset_panel;
extern MyFindParticlesPanel *findparticles_panel;


MyRelionExportDialog::MyRelionExportDialog( wxWindow * parent )
:
RelionExportDialog( parent )
{
	ExportButton->Enable(false);
	FillGroupComboBoxWorker( GroupComboBox , true );
	WarningText->Show(findparticles_panel->ReturnNumberOfJobsCurrentlyRunning() > 0);
}

void MyRelionExportDialog::OnCancelButtonClick( wxCommandEvent & event )
{
	Close();
}

void MyRelionExportDialog::OnNormalizeCheckBox( wxCommandEvent & event )
{
	particleRadiusStaticText->Enable(NormalizeCheckBox->IsChecked());
	particleRadiusTextCtrl->Enable(NormalizeCheckBox->IsChecked());
}

/*
 *
 * We export to Relion, which means creating one stack of particles per micrograph,
 * and a star file with the following header:
 *
 *
 *
data_

loop_
_rlnMicrographName #1
_rlnCoordinateX #2
_rlnCoordinateY #3
_rlnImageName #4
_rlnDefocusU #5
_rlnDefocusV #6
_rlnDefocusAngle #7
_rlnVoltage #8
_rlnSphericalAberration #9
_rlnAmplitudeContrast #10
_rlnMagnification #11
_rlnDetectorPixelSize #12
_rlnCtfFigureOfMerit #13
 *
 * Here is an example of a line from the star file:
 *
 movie_sums/r_001_aligned_sum.mrc   523.000000   558.000000 000001@Particles/movie_sums/r_001_aligned_sum_particles.mrcs 26792.400391 26373.519531   -27.580000   200.000000     2.000000     0.100000 47361.000000    14.000000     0.075220
 *
 *
 */
void MyRelionExportDialog::OnExportButtonClick( wxCommandEvent & event )
{
	ArrayOfParticlePositionAssets current_array_of_assets;

	int number_of_images_in_group = image_asset_panel->ReturnGroupSize(GroupComboBox->GetSelection());
	ImageAsset *current_image_asset;
	wxFileName output_stack_filename = OutputImageStackPicker->GetFileName();
	MRCFile *output_stack;
	wxFileName relion_star_filename = output_stack_filename;
	relion_star_filename.SetExt("star");
	//NumericTextFile *frealign_txt_file = new NumericTextFile(frealign_txt_filename.GetFullPath(),OPEN_TO_WRITE,4);
	wxTextFile *relion_star_file = new wxTextFile(relion_star_filename.GetFullPath());
	Image micrograph;
	float temp_float[4];
	double acceleration_voltage;
	double spherical_aberration;
	double amplitude_contrast;
	double defocus_1;
	double defocus_2;
	double astigmatism_angle;
	double additional_phase_shift;
	double iciness;
	float micrograph_mean;
	int box_at_x;
	int box_at_y;
	Image box;
	Image box_large;
	box.Allocate(BoxSizeSpinCtrl->GetValue(),BoxSizeSpinCtrl->GetValue(),1);
	if (DownsamplingFactorSpinCtrl->GetValue() > 1) box_large.Allocate(BoxSizeSpinCtrl->GetValue() * DownsamplingFactorSpinCtrl->GetValue(),BoxSizeSpinCtrl->GetValue() * DownsamplingFactorSpinCtrl->GetValue(),1);
	long number_of_boxes = 0;

	//wxPrintf("output files: %s %s\n",output_stack_filename.GetFullPath(), frealign_txt_filename.GetFullPath());

	// Check whether the file already exists. If does, open and clear. If not, create
	if (relion_star_file->Exists())
	{
		relion_star_file->Open();
		relion_star_file->Clear();
	}
	else
	{
		relion_star_file->Create();
	}

	// Write headers
	relion_star_file->AddLine(wxString(" "));
	relion_star_file->AddLine(wxString("data_"));
	relion_star_file->AddLine(wxString(" "));
	relion_star_file->AddLine(wxString("loop_"));
	relion_star_file->AddLine(wxString("_rlnMicrographName #1"));
	relion_star_file->AddLine(wxString("_rlnCoordinateX #2"));
	relion_star_file->AddLine(wxString("_rlnCoordinateY #3"));
	relion_star_file->AddLine(wxString("_rlnImageName #4"));
	relion_star_file->AddLine(wxString("_rlnDefocusU #5"));
	relion_star_file->AddLine(wxString("_rlnDefocusV #6"));
	relion_star_file->AddLine(wxString("_rlnDefocusAngle #7"));
	relion_star_file->AddLine(wxString("_rlnPhaseShift #8"));
	relion_star_file->AddLine(wxString("_rlnVoltage #9"));
	relion_star_file->AddLine(wxString("_rlnSphericalAberration #10"));
	relion_star_file->AddLine(wxString("_rlnAmplitudeContrast #11"));
	relion_star_file->AddLine(wxString("_rlnMagnification #12"));
	relion_star_file->AddLine(wxString("_rlnDetectorPixelSize #13"));
	relion_star_file->AddLine(wxString("_rlnCtfFigureOfMerit #14"));

	OneSecondProgressDialog *my_dialog = new OneSecondProgressDialog ("Exporting stack and star file", "Exporting...", number_of_images_in_group, this);

	for (int image_counter = 0; image_counter < number_of_images_in_group; image_counter ++ )
	{

		my_dialog->Update(image_counter+1, wxString::Format("Image %i of %i",image_counter+1,number_of_images_in_group));

		current_image_asset = 	image_asset_panel->ReturnAssetPointer(image_asset_panel->ReturnGroupMember(GroupComboBox->GetSelection(),image_counter));

		current_array_of_assets = main_frame->current_project.database.ReturnArrayOfParticlePositionAssetsFromAssetsTable(current_image_asset->asset_id);

		main_frame->current_project.database.GetCTFParameters(current_image_asset->ctf_estimation_id,acceleration_voltage,spherical_aberration,amplitude_contrast,defocus_1,defocus_2,astigmatism_angle,additional_phase_shift,iciness);

		temp_float[0] = image_counter + 1;
		temp_float[1] = defocus_1;
		temp_float[2] = defocus_2;
		temp_float[3] = astigmatism_angle;


		micrograph.QuickAndDirtyReadSlice(current_image_asset->filename.GetFullPath().ToStdString(),1);

		if (FlipCTFCheckBox->IsChecked())
		{
			CTF my_ctf(acceleration_voltage,spherical_aberration,amplitude_contrast,defocus_1, defocus_2, astigmatism_angle,current_image_asset->pixel_size, additional_phase_shift);
			micrograph.ForwardFFT();
			micrograph.ApplyCTFPhaseFlip(my_ctf);
			micrograph.BackwardFFT();
		}
		else
		{
			micrograph.InvertRealValues();
		}

		micrograph_mean = micrograph.ReturnAverageOfRealValues();
		//if (NormalizeCheckBox->IsChecked()) micrograph.ZeroFloatAndNormalize();

		if (image_counter == 0)
		{
			output_stack = new MRCFile(output_stack_filename.GetFullPath().ToStdString(),true);
			output_stack->SetPixelSize(current_image_asset->pixel_size);
		}

		for (size_t particle_counter = 0; particle_counter < current_array_of_assets.GetCount(); particle_counter ++ )
		{

			number_of_boxes ++;

			box_at_x = current_array_of_assets.Item(particle_counter).x_position / current_image_asset->pixel_size - micrograph.physical_address_of_box_center_x + 1.0;
			box_at_y = current_array_of_assets.Item(particle_counter).y_position / current_image_asset->pixel_size - micrograph.physical_address_of_box_center_y + 1.0;



			if (DownsamplingFactorSpinCtrl->GetValue() == 1)
			{
				micrograph.ClipInto(&box,micrograph_mean,false,1.0,box_at_x,box_at_y,0);
			}
			else
			{
				micrograph.ClipInto(&box_large,micrograph_mean,false,1.0,box_at_x,box_at_y,0);
				box_large.ForwardFFT();
				box_large.ClipInto(&box);
				box.BackwardFFT();
			}

			double particle_radius;
			particleRadiusTextCtrl->GetValue().ToDouble(&particle_radius);
			if (NormalizeCheckBox->IsChecked()) box.ZeroFloatAndNormalize(1.0,particle_radius / (float(DownsamplingFactorSpinCtrl->GetValue()) * current_image_asset->pixel_size),true); // this is for Relion, which likes background values to be normalized

			box.WriteSlice(output_stack,number_of_boxes);
			//frealign_txt_file->WriteLine(temp_float);
			//movie_sums/r_001_aligned_sum.mrc   523.000000   558.000000 000001@Particles/movie_sums/r_001_aligned_sum_particles.mrcs 26792.400391 26373.519531   -27.580000   200.000000     2.000000     0.100000 47361.000000    14.000000     0.075220
			relion_star_file->AddLine(wxString::Format("%s %f %f %06li@%s %f %f %f %f %f %f %f %f %f %f",	current_image_asset->filename.GetFullPath(),
																										current_array_of_assets.Item(particle_counter).x_position / current_image_asset->pixel_size,
																										current_array_of_assets.Item(particle_counter).y_position / current_image_asset->pixel_size,
																										number_of_boxes,
																										output_stack_filename.GetFullPath(),
																										defocus_1,
																										defocus_2,
																										astigmatism_angle,
																										additional_phase_shift,
																										acceleration_voltage,
																										spherical_aberration,
																										amplitude_contrast,
																										10000.0f,
																										(float(DownsamplingFactorSpinCtrl->GetValue()) * current_image_asset->pixel_size),
																										0.50f));


		}

	}

	relion_star_file->Write();
	relion_star_file->Close();

	output_stack->CloseFile();
	delete relion_star_file	;

	my_dialog->Destroy();

	Close();
}

void MyRelionExportDialog::OnOutputImageStackFileChanged( wxFileDirPickerEvent & event )
{
	FileNameStaticText->SetLabelText(OutputImageStackPicker->GetFileName().GetFullName());
	ExportButton->Enable(true);
}

