#include "../../core/core_headers.h"

class
NikoTestApp : public MyApp
{

	public:

	bool DoCalculation();
	void DoInteractiveUserInput();

	private:
};


IMPLEMENT_APP(NikoTestApp)

// override the DoInteractiveUserInput

void NikoTestApp::DoInteractiveUserInput()
{

}

// override the do calculation method which will be what is actually run..

bool NikoTestApp::DoCalculation()
{
	int i;
	float temp_float;
	MRCFile input_file("input.mrc", false);
	MRCFile output_file("output.mrc", true);
	Image input_image;
	Image filtered_image;
	Image kernel;

	input_image.Allocate(input_file.ReturnXSize(), input_file.ReturnYSize(), input_file.ReturnZSize(), true);
	filtered_image.Allocate(input_file.ReturnXSize(), input_file.ReturnYSize(), input_file.ReturnZSize(), true);
	kernel.Allocate(input_file.ReturnXSize(), input_file.ReturnYSize(), input_file.ReturnZSize(), true);
	input_image.ReadSlices(&input_file,1,input_image.logical_z_dimension);

/*	kernel.SetToConstant(1.0);
	kernel.CosineMask(8.0, 8.0, false, true, 0.0);
//	kernel.real_values[0] = 1.0;
	temp_float = kernel.ReturnAverageOfRealValues() * kernel.number_of_real_space_pixels;
//	wxPrintf("average = %g\n", temp_float);
//	kernel.WriteSlices(&output_file,1,input_image.logical_z_dimension);
	kernel.ForwardFFT();
	kernel.SwapRealSpaceQuadrants();
	kernel.MultiplyByConstant(float(kernel.number_of_real_space_pixels) / temp_float);
//	kernel.CosineMask(0.03, 0.03, true);

	input_image.SetMinimumValue(0.0);
	filtered_image.CopyFrom(&input_image);
	filtered_image.ForwardFFT();
	filtered_image.MultiplyPixelWise(kernel);
//	filtered_image.CosineMask(0.01, 0.02);
	filtered_image.BackwardFFT();
//	filtered_image.MultiplyByConstant(0.3);
	input_image.SubtractImage(&filtered_image);
*/
	for (i = 0; i < input_image.real_memory_allocated; i++) if (input_image.real_values[i] < 0.0) input_image.real_values[i] = -log(-input_image.real_values[i] + 1.0);
	input_image.WriteSlices(&output_file,1,input_image.logical_z_dimension);
//	filtered_image.WriteSlices(&output_file,1,input_image.logical_z_dimension);

	return true;
}
