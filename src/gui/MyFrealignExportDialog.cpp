#include "../core/gui_core_headers.h"

extern MyMainFrame *main_frame;
extern MyImageAssetPanel *image_asset_panel;
extern MyFindParticlesPanel *findparticles_panel;


MyFrealignExportDialog::MyFrealignExportDialog( wxWindow * parent )
:
FrealignExportDialog( parent )
{
	ExportButton->Enable(false);
	FillGroupComboBoxSlave( GroupComboBox , true );
	WarningText->Show(findparticles_panel->ReturnNumberOfJobsCurrentlyRunning() > 0);
}

void MyFrealignExportDialog::OnCancelButtonClick( wxCommandEvent & event )
{
	Close();
}

void MyFrealignExportDialog::OnExportButtonClick( wxCommandEvent & event )
{
	ArrayOfParticlePositionAssets current_array_of_assets;

	int number_of_images_in_group = image_asset_panel->ReturnGroupSize(GroupComboBox->GetSelection());
	ImageAsset *current_image_asset;
	wxFileName output_stack_filename = OutputImageStackPicker->GetFileName();
	MRCFile *output_stack;
	wxFileName frealign_txt_filename = output_stack_filename;
	frealign_txt_filename.SetExt("txt");
	NumericTextFile *frealign_txt_file = new NumericTextFile(frealign_txt_filename.GetFullPath(),OPEN_TO_WRITE,4);
	Image micrograph;
	float temp_float[4];
	double acceleration_voltage;
	double spherical_aberration;
	double amplitude_contrast;
	double defocus_1;
	double defocus_2;
	double astigmatism_angle;
	double additional_phase_shift;
	float micrograph_mean;
	int box_at_x;
	int box_at_y;
	Image box;
	Image box_large;
	box.Allocate(BoxSizeSpinCtrl->GetValue(),BoxSizeSpinCtrl->GetValue(),1);
	if (DownsamplingFactorSpinCtrl->GetValue() > 1) box_large.Allocate(BoxSizeSpinCtrl->GetValue() * DownsamplingFactorSpinCtrl->GetValue(),BoxSizeSpinCtrl->GetValue() * DownsamplingFactorSpinCtrl->GetValue(),1);
	long number_of_boxes = 0;

	//wxPrintf("output files: %s %s\n",output_stack_filename.GetFullPath(), frealign_txt_filename.GetFullPath());

	for (int image_counter = 0; image_counter < number_of_images_in_group; image_counter ++ )
	{
		current_image_asset = image_asset_panel->ReturnAssetPointer(image_asset_panel->ReturnGroupMember(GroupComboBox->GetSelection(),image_counter));

		current_array_of_assets = main_frame->current_project.database.ReturnArrayOfParticlePositionAssetsFromAssetsTable(current_image_asset->asset_id);

		main_frame->current_project.database.GetCTFParameters(current_image_asset->ctf_estimation_id,acceleration_voltage,spherical_aberration,amplitude_contrast,defocus_1,defocus_2,astigmatism_angle,additional_phase_shift);

		temp_float[0] = image_counter + 1;
		temp_float[1] = defocus_1;
		temp_float[2] = defocus_2;
		temp_float[3] = astigmatism_angle;


		micrograph.QuickAndDirtyReadSlice(current_image_asset->filename.GetFullPath().ToStdString(),1);
		micrograph_mean = micrograph.ReturnAverageOfRealValues();

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

			if (NormalizeCheckBox->IsChecked()) box.ZeroFloatAndNormalize();

			box.WriteSlice(output_stack,number_of_boxes);
			frealign_txt_file->WriteLine(temp_float);

		}

	}

	output_stack->CloseFile();
	delete frealign_txt_file;

	Close();
}

void MyFrealignExportDialog::OnOutputImageStackFileChanged( wxFileDirPickerEvent & event )
{
	ExportButton->Enable(true);
}

