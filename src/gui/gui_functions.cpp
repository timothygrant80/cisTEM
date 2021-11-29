//#include "../core/core_headers.h"
#include "../core/gui_core_headers.h"

extern MyMainFrame *main_frame;


void ConvertImageToBitmap(Image *input_image, wxBitmap *output_bitmap, bool auto_contrast)
{
	MyDebugAssertTrue(input_image->logical_z_dimension == 1, "Only 2D images can be used");
	MyDebugAssertTrue(output_bitmap->GetDepth() == 24, "bitmap should be 24 bit");

	float image_min_value;
	float image_max_value;
	float range;
	float inverse_range;

	int current_grey_value;
	int i, j;

	long mirror_line_address;
	long address = 0;


	if (input_image->logical_x_dimension != output_bitmap->GetWidth() || input_image->logical_y_dimension != output_bitmap->GetHeight())
	{
		output_bitmap->Create(input_image->logical_x_dimension, input_image->logical_y_dimension, 24);
	}


	if (auto_contrast == false)
	{

		input_image->GetMinMax(image_min_value, image_max_value);
	}
	else
	{
		float average_value = input_image->ReturnAverageOfRealValues();
		float variance = input_image->ReturnVarianceOfRealValues();
		float stdev = sqrt(variance);

		image_min_value = average_value - (stdev * 2.5);
		image_max_value = average_value + (stdev * 2.5);
	}

	range = image_max_value - image_min_value;
	inverse_range = 1. / range;
	inverse_range *= 256.;

	wxNativePixelData pixel_data(*output_bitmap);

	if ( !pixel_data )
	{
	   MyPrintWithDetails("Can't access bitmap data");
	   DEBUG_ABORT;
	}



	wxNativePixelData::Iterator p(pixel_data);
	p.Reset(pixel_data);


	// we have to mirror the lines as wxwidgets using 0,0 at top left
	for (j = 0; j < input_image->logical_y_dimension; j++)
	{
		mirror_line_address = (input_image->logical_y_dimension - 1 - j) * (input_image->logical_x_dimension + input_image->padding_jump_value);
		p.MoveTo(pixel_data,0,j);

		for (i = 0; i < input_image->logical_x_dimension; i++)
		{
			//mirror_line_address = input_image->ReturnReal1DAddressFromPhysicalCoord(i,j,0);
			current_grey_value = myroundint((input_image->real_values[mirror_line_address] - image_min_value) * inverse_range);

			if (current_grey_value < 0) current_grey_value = 0;
			else
			if (current_grey_value > 255) current_grey_value = 255;

			p.Red() = current_grey_value;
			p.Green() = current_grey_value;
			p.Blue() = current_grey_value;

			p++;
			mirror_line_address++;
		}
	}


}


void GetMultilineTextExtent	(wxDC *wanted_dc, const wxString & string, int &width, int &height)
{
	wxStringTokenizer tokens(string, "\n");
	wxString current_token;
	wxSize line_size;
	int number_of_lines = tokens.CountTokens();
	int current_line;

	width = 0;
	height = 0;

	for (current_line = 0; current_line < number_of_lines; current_line++)
	{
		current_token = tokens.GetNextToken();
		line_size = wanted_dc->GetTextExtent(current_token);
		if (line_size.GetWidth() > width) width = line_size.GetWidth();
		height += line_size.GetHeight();
	}
}

wxArrayString GetRecentProjectsFromSettings()
{
	wxArrayString temp;

	if (wxConfig::Get()->HasEntry("RecentProject1") == true) temp.Add(wxConfig::Get()->Read("RecentProject1"));
	if (wxConfig::Get()->HasEntry("RecentProject2") == true) temp.Add(wxConfig::Get()->Read("RecentProject2"));
	if (wxConfig::Get()->HasEntry("RecentProject3") == true) temp.Add(wxConfig::Get()->Read("RecentProject3"));
	if (wxConfig::Get()->HasEntry("RecentProject4") == true) temp.Add(wxConfig::Get()->Read("RecentProject4"));
	if (wxConfig::Get()->HasEntry("RecentProject5") == true) temp.Add(wxConfig::Get()->Read("RecentProject5"));

	return temp;
}

void AddProjectToRecentProjects(wxString project_to_add)
{
	int counter;
	wxArrayString recent_projects = GetRecentProjectsFromSettings();

	recent_projects.Insert(project_to_add, 0);
	for (counter = recent_projects.GetCount() - 1; counter > 0; counter--)
	{
		if (recent_projects.Item(counter) == project_to_add) recent_projects.RemoveAt(counter);
	}

	if (recent_projects.GetCount() > 5)
	{
		for (counter = 5; counter < recent_projects.GetCount();counter++)
		{
			recent_projects.RemoveAt(counter);
		}
	}

	// set the recent projects in settings..

	for (counter = 0; counter < recent_projects.GetCount(); counter++)
	{
		wxConfig::Get()->Write(wxString::Format("RecentProject%i", counter + 1), recent_projects.Item(counter));
	}

	wxConfig::Get()->Flush();

}


void FillGroupComboBoxWorker( wxComboBox *GroupComboBox, bool include_all_images_group )
{
	extern MyImageAssetPanel *image_asset_panel;
	GroupComboBox->Freeze();
	GroupComboBox->ChangeValue("");
	GroupComboBox->Clear();

	long first_group_to_include = 0;
	if (!include_all_images_group) first_group_to_include = 1;

	for (long counter = first_group_to_include; counter < image_asset_panel->ReturnNumberOfGroups(); counter++)
	{
		GroupComboBox->Append(image_asset_panel->ReturnGroupName(counter) +  " (" + wxString::Format(wxT("%li"), image_asset_panel->ReturnGroupSize(counter)) + ")");

	}

	if (GroupComboBox->GetCount() > 0) GroupComboBox->SetSelection(0);

	GroupComboBox->Thaw();
}

void FillParticlePositionsGroupComboBox(wxComboBox *GroupComboBox, bool include_all_particle_positions_group)
{
	extern MyParticlePositionAssetPanel *particle_position_asset_panel;
	GroupComboBox->Freeze();
	GroupComboBox->ChangeValue("");
	GroupComboBox->Clear();

	long first_group_to_include = 0;
	if (!include_all_particle_positions_group) first_group_to_include = 1;

	for (long counter = first_group_to_include; counter < particle_position_asset_panel->ReturnNumberOfGroups(); counter++)
	{
		GroupComboBox->Append(particle_position_asset_panel->ReturnGroupName(counter) +  " (" + wxString::Format(wxT("%li"), particle_position_asset_panel->ReturnGroupSize(counter)) + ")");
	}

	if (GroupComboBox->GetCount() > 0) GroupComboBox->SetSelection(0);

	GroupComboBox->Thaw();

}

void AppendVolumeAssetsToComboBox(wxComboBox *ComboBox)
{
	extern MyVolumeAssetPanel *volume_asset_panel;
	ComboBox->Freeze();

	for (unsigned long counter = 0; counter < volume_asset_panel->ReturnNumberOfAssets(); counter++)
		{
			ComboBox->Append(volume_asset_panel->ReturnAssetName(counter));
		}

		ComboBox->Thaw();
}

#ifdef EXPERIMENTAL
void AppendAtomicCoordinatesAssetsToComboBox(wxComboBox *ComboBox)
{
	extern AtomicCoordinatesAssetPanel *atomic_coordinates_asset_panel;
	ComboBox->Freeze();

	for (unsigned long counter = 0; counter < atomic_coordinates_asset_panel->ReturnNumberOfAssets(); counter++)
		{
			ComboBox->Append(atomic_coordinates_asset_panel->ReturnAssetName(counter));
		}

		ComboBox->Thaw();
}
#endif

void AppendRefinementPackagesToComboBox(wxComboBox *ComboBox)
{
	extern MyRefinementPackageAssetPanel *refinement_package_asset_panel;
	ComboBox->Freeze();

	for (unsigned long counter = 0; counter < refinement_package_asset_panel->all_refinement_packages.GetCount(); counter++)
	{
			ComboBox->Append(refinement_package_asset_panel->all_refinement_packages.Item(counter).name);
	}

		ComboBox->Thaw();
}

void RunSimpleFunctionInAnotherThread(wxWindow *parent_window, void (*function_to_run)(void))
{
	RunSimpleFunctionThread *function_thread = new RunSimpleFunctionThread(parent_window, function_to_run);

	if ( function_thread->Run() != wxTHREAD_NO_ERROR )
	{
		MyPrintWithDetails("Error Running Thread!");
	}

}

wxThread::ExitCode RunSimpleFunctionThread::Entry()
{
	function_to_run();
	return (wxThread::ExitCode)0;
}

void SetupDefaultColorMap()
{
	wxColor color1(0, 0, 255);
	default_colormap.Add(color1);
	wxColor color2(128, 0, 0);
	default_colormap.Add(color2);
	wxColor color3(139, 182, 182);
	default_colormap.Add(color3);
	wxColor color4(0, 0, 128);
	default_colormap.Add(color4);
	wxColor color5(255, 0, 255);
	default_colormap.Add(color5);
	wxColor color6(255,255, 0);
	default_colormap.Add(color6);
	wxColor color7(115, 229, 229);
	default_colormap.Add(color7);
	wxColor color8(128, 0, 128);
	default_colormap.Add(color8);
	wxColor color9(156, 206, 0);
	default_colormap.Add(color9);
	wxColor color10(255, 206, 0);
	default_colormap.Add(color10);
	wxColor color11(206, 156, 255);
	default_colormap.Add(color11);
	wxColor color12(0, 206, 255);
	default_colormap.Add(color12);
	wxColor color13(255, 156, 206);
	default_colormap.Add(color13);
	wxColor color14(206, 255, 255);
	default_colormap.Add(color14);
	wxColor color15(206, 255, 206);
	default_colormap.Add(color15);
	wxColor color16(255, 255, 156);
	default_colormap.Add(color16);
	wxColor color17(224, 208, 191);
	default_colormap.Add(color17);
	wxColor color18(49, 99, 255);
	default_colormap.Add(color18);
	wxColor color19(52, 207, 207);
	default_colormap.Add(color19);
	wxColor color20(0, 0, 0);
	default_colormap.Add(color20);
}

void SetupDefaultColorBar()
{
	wxColour buffer_colour;
	unsigned char current_red = 0;
	unsigned char current_blue = 128;
	unsigned current_green = 0;
	int counter;

	for (counter = 0; counter < 127; counter++)
	{
			buffer_colour.Set(current_red, current_green, current_blue);
			default_colorbar.Add(buffer_colour);
			current_blue++;
	}

	//wxPrintf("r=%u, g=%u, b=%u\n", current_red, current_green, current_blue);

	for (counter = 0; counter < 255; counter++)
	{
		buffer_colour.Set(current_red, current_green, current_blue);
		default_colorbar.Add(buffer_colour);
		current_green+=1;
		//current_blue-=1;
	}

	//wxPrintf("r=%u, g=%u, b=%u\n", current_red, current_green, current_blue);

	for (counter = 0; counter < 127; counter++)
	{
		buffer_colour.Set(current_red, current_green, current_blue);
		default_colorbar.Add(buffer_colour);
		//current_green+=1;
		//current_blue-=1;
		current_red+=1;
		//current_green+=1;
	}

	//wxPrintf("r=%u, g=%u, b=%u\n", current_red, current_green, current_blue);

	for (counter = 0; counter < 128; counter++)
	{
		buffer_colour.Set(current_red, current_green, current_blue);
		default_colorbar.Add(buffer_colour);
		current_red+=1;
		current_blue-=1;
	}

	//wxPrintf("r=%u, g=%u, b=%u\n", current_red, current_green, current_blue);

	for (counter = 0; counter < 127; counter++)
	{
		buffer_colour.Set(current_red, current_green, current_blue);
		default_colorbar.Add(buffer_colour);
		current_blue--;
	}

	//wxPrintf("r=%u, g=%u, b=%u\n", current_red, current_green, current_blue);

	for (counter = 0; counter < 255; counter++)
	{
		buffer_colour.Set(current_red, current_green, current_blue);
		default_colorbar.Add(buffer_colour);
		//current_red+=1;
		current_green-=1;
	}
	//wxPrintf("r=%u, g=%u, b=%u\n", current_red, current_green, current_blue);

	buffer_colour.Set(current_red, current_green, current_blue);
	default_colorbar.Add(buffer_colour);

	/*
	for (counter = 0; counter < 127; counter++)
	{
			buffer_colour.Set(current_red, current_green, current_blue);
			default_colorbar.Add(buffer_colour);
			current_blue++;
	}

//	wxPrintf("r=%u, g=%u, b=%u\n", current_red, current_green, current_blue);

	for (counter = 0; counter < 51; counter++)
	{
		buffer_colour.Set(current_red, current_green, current_blue);
		default_colorbar.Add(buffer_colour);
		current_green+=5;
	}

	//wxPrintf("r=%u, g=%u, b=%u\n", current_red, current_green, current_blue);

	for (counter = 0; counter < 51; counter++)
	{
		buffer_colour.Set(current_red, current_green, current_blue);
		default_colorbar.Add(buffer_colour);
		current_blue-=5;
	}

	//wxPrintf("r=%u, g=%u, b=%u\n", current_red, current_green, current_blue);

	for (counter = 0; counter < 51; counter++)
	{
		buffer_colour.Set(current_red, current_green, current_blue);
		default_colorbar.Add(buffer_colour);
		current_red+=5;
	}

	//("r=%u, g=%u, b=%u\n", current_red, current_green, current_blue);

	for (counter = 0; counter < 255; counter++)
	{
		buffer_colour.Set(current_red, current_green, current_blue);
		default_colorbar.Add(buffer_colour);
		current_green--;
	}

//	wxPrintf("r=%u, g=%u, b=%u\n", current_red, current_green, current_blue);
*/

}


void global_delete_scratch()
{
	main_frame->ClearScratchDirectory();
}

void global_delete_refine2d_scratch()
{
	main_frame->ClearRefine2DScratch();
}

void global_delete_refine3d_scratch()
{
	main_frame->ClearRefine3DScratch();
}

void global_delete_startup_scratch()
{
	main_frame->ClearStartupScratch();
}

void global_delete_autorefine3d_scratch()
{
	main_frame->ClearAutoRefine3DScratch();
}

void global_delete_generate3d_scratch()
{
	main_frame->ClearGenerate3DScratch();
}

void global_delete_refinectf_scratch()
{
	main_frame->ClearRefineCTFScratch();
}


