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

	Image test_image;

	wxDateTime before;
	wxDateTime after;
	wxLongLong average = 0;

	test_image.QuickAndDirtyReadSlice("/tmp/image.mrc", 1);

	for (long counter = 0; counter < 100; counter++)
	{
		before = wxDateTime::UNow();
		test_image.ForwardFFT(false);
		after = wxDateTime::UNow();

		average += after.Subtract(before).GetMilliseconds();


	}

	wxPrintf("average time = %li ms\n", average / 100);



	return true;
}
