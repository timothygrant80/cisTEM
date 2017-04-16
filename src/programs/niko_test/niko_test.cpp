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
	MRCFile input_stack("input.mrc", false);
	MRCFile output_stack("output.mrc", true);
	Image input_image;

	input_image.Allocate(input_stack.ReturnXSize(), input_stack.ReturnYSize(), true);


	wxDateTime before;
	wxDateTime after;
	wxLongLong average = 0;

	for (i = 1; i <= input_stack.ReturnZSize(); i++)
	{
		input_image.ReadSlice(&input_stack, i);
		input_image.RealSpaceIntegerShift(8,6,0);
		input_image.WriteSlice(&output_stack, i);
	}

	return true;
}
