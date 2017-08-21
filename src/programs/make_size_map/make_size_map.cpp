#include "../../core/core_headers.h"

class
MakeSizeMap : public MyApp
{

	public:

	bool DoCalculation();
	void DoInteractiveUserInput();

	private:
};


IMPLEMENT_APP(MakeSizeMap)

// override the DoInteractiveUserInput

void MakeSizeMap::DoInteractiveUserInput()
{

	UserInput *my_input = new UserInput("MakeSizeMap", 1.0);

	wxString input_volume	= my_input->GetFilenameFromUser("Input Volume file name", "Name of input image volume", "input.mrc", true );
	wxString output_image	= my_input->GetFilenameFromUser("Output Size Map file name", "Name of output size map volume ", "my_size_map.mrc", false );
	float    binarise_value = my_input->GetFloatFromUser("Binarisation threshold?", "The volume will first be binarised at this threshold", "0.01");

	delete my_input;

	my_current_job.Reset(3);
	my_current_job.ManualSetArguments("ttf", input_volume.ToUTF8().data(), output_image.ToUTF8().data(), binarise_value);
}

// override the do calculation method which will be what is actually run..

bool MakeSizeMap::DoCalculation()
{

	wxString input_volume	= my_current_job.arguments[0].ReturnStringArgument();
	wxString output_image	= my_current_job.arguments[1].ReturnStringArgument();
	float    binarise_value = my_current_job.arguments[2].ReturnFloatArgument();

	MRCFile input3d_file(input_volume.ToStdString(), false);
	MRCFile output_file(output_image.ToStdString(), true);

	Image my_input_volume;
	Image my_size_map;




	my_input_volume.ReadSlices(&input3d_file, 1, input3d_file.ReturnNumberOfSlices());
	my_size_map.Allocate(my_input_volume.logical_x_dimension, my_input_volume.logical_y_dimension, my_input_volume.logical_z_dimension, true);
	my_size_map.SetToConstant(0.0);
	wxPrintf("\nBinarising...\n");
	my_input_volume.Binarise(binarise_value);
	//my_input_volume.QuickAndDirtyWriteSlices("/tmp/bin.mrc", 1, my_input_volume.logical_z_dimension);
	wxPrintf("Run Length Encoding...\n");
	rle3d my_rle3d(my_input_volume);
	//my_rle3d.Write("/tmp/rle.txt");
	wxPrintf("Making Size Map...\n");
	my_rle3d.ConnectedSizeDecodeTo(my_size_map);

	my_size_map.WriteSlices(&output_file, 1, my_size_map.logical_z_dimension);


	return true;
}
