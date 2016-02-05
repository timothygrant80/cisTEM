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

	/*
	Image input_image;
	Image output_3d;
	float padding = 1.0;
	float temp_double[50];

	MRCFile input_file("stack.mrc", false);
	NumericTextFile my_par_file("input", OPEN_TO_READ);
	AnglesAndShifts my_parameters;
	CTF my_ctf;

//	MRCFile output_2d("test2d.mrc", true);

	//Reconstruct3d my_reconstruction(input_file.ReturnXSize() * padding, input_file.ReturnYSize() * padding, input_file.ReturnXSize() * padding);

	for (int current_image = 1; current_image <= input_file.ReturnNumberOfSlices(); current_image++)
//	for (int current_image = 1; current_image <= 100; current_image++)
	{
		input_image.ReadSlice(&input_file, current_image);

//		input_image.SetToConstant(0.0);
//		input_image.real_values[input_image.ReturnReal1DAddressFromPhysicalCoord(64,64,0)] = 1.0;

		my_par_file.ReadLine(temp_double);
		if (padding != 1.0)
		{
			input_image.Resize(input_image.logical_x_dimension * padding, input_image.logical_y_dimension * padding, input_image.logical_z_dimension, input_image.ReturnAverageOfRealValuesOnEdges());
		}
//		input_image.WriteSlice(&output_2d,current_image);
		my_parameters.Init(temp_double[3], temp_double[2], temp_double[1], temp_double[4], temp_double[5]);
		my_ctf.Init(300.0, 2.7, 0.07, temp_double[8], temp_double[9], temp_double[10], 0.0, 0.0, 0.0, 3.28, 0.0);

		input_image.ForwardFFT();
//		input_image.ApplyCTF(my_ctf);
		input_image.PhaseShift(-temp_double[4] / 3.28, -temp_double[5] / 3.28);
		input_image.SwapRealSpaceQuadrants();

		if (current_image % 100 == 0 || current_image == 1)
		{
			wxPrintf("Working on image %i, %f, %f, %f, %f, %f, %f, %f, %f\n", current_image, temp_double[1], temp_double[2], temp_double[3], temp_double[4], temp_double[5], temp_double[8], temp_double[9], temp_double[10]);
		}
		//my_reconstruction.InsertSlice(input_image, my_ctf, my_parameters);
//		my_reconstruction.InsertSlice(input_image, my_parameters);
	}

//	input_image.QuickAndDirtyReadSlice("test.mrc", 1);


	//my_reconstruction.FinalizeSimple(output_3d);
//	output_3d = my_reconstruction.image_reconstruction;
//	output_3d = my_reconstruction.ctf_reconstruction;
	output_3d.SwapRealSpaceQuadrants();
	output_3d.BackwardFFT();
	if (padding != 1.0)
	{
		output_3d.Resize(output_3d.logical_x_dimension / padding, output_3d.logical_y_dimension / padding, output_3d.logical_z_dimension / padding);
	}
	output_3d.CosineMask(170.0/3.28, 5.0);
	MRCFile output_file("test3d.mrc", true);
	output_3d.WriteSlices(&output_file,1,output_3d.logical_z_dimension);

	*/

	return true;
}





