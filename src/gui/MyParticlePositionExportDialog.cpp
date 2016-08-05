#include "../core/gui_core_headers.h"

extern MyMainFrame *main_frame;
extern MyImageAssetPanel *image_asset_panel;
extern MyFindParticlesPanel *findparticles_panel;


MyParticlePositionExportDialog::MyParticlePositionExportDialog( wxWindow * parent )
:
ParticlePositionExportDialog( parent )
{
	ExportButton->Enable(false);
	FillGroupComboBoxSlave( GroupComboBox , true );
	WarningText->Show(findparticles_panel->ReturnNumberOfJobsCurrentlyRunning() > 0);
}

void MyParticlePositionExportDialog::OnCancelButtonClick( wxCommandEvent & event )
{
	Close();
}

void MyParticlePositionExportDialog::OnExportButtonClick( wxCommandEvent & event )
{
	ArrayOfParticlePositionAssets current_array_of_assets;

	int number_of_images_in_group = image_asset_panel->ReturnGroupSize(GroupComboBox->GetSelection());
	ImageAsset *current_image_asset;
	wxFileName plt_filename;
	NumericTextFile *output_plt_file;
	wxFileName output_directory = DestinationDirectoryPickerCtrl->GetDirName();
	float temp_float[3];
	temp_float[2] = 1.0;

	for (int image_counter = 0; image_counter < number_of_images_in_group; image_counter ++ )
	{
		current_image_asset = image_asset_panel->ReturnAssetPointer(image_asset_panel->ReturnGroupMember(GroupComboBox->GetSelection(),image_counter));

		current_array_of_assets = main_frame->current_project.database.ReturnArrayOfParticlePositionAssetsFromAssetsTable(current_image_asset->asset_id);

		plt_filename.Assign(output_directory.GetPath(),current_image_asset->filename.GetName(),"plt");

		//wxPrintf("Writing coordinates to %s\n",plt_filename.GetFullPath());

		if (image_counter == 0)
		{
			output_plt_file = new NumericTextFile(plt_filename.GetFullPath(),OPEN_TO_WRITE,3);
		}
		else
		{
			output_plt_file->Open(plt_filename.GetFullPath(),OPEN_TO_WRITE,3);
		}

		for (size_t particle_counter = 0; particle_counter < current_array_of_assets.GetCount(); particle_counter ++ )
		{
			temp_float[0] = current_array_of_assets.Item(particle_counter).y_position / current_image_asset->pixel_size + 1.0;
			temp_float[1] = current_image_asset->x_size - current_array_of_assets.Item(particle_counter).x_position / current_image_asset->pixel_size + 1.0;

			output_plt_file->WriteLine(temp_float);
		}

		output_plt_file->Close();
		if (image_counter == number_of_images_in_group - 1) delete output_plt_file;
	}

	Close();
}

void MyParticlePositionExportDialog::OnDirChanged( wxFileDirPickerEvent & event )
{
	//TODO: check whether the selected group has at least one particle position asset picked from among its images
	ExportButton->Enable(true);
}

