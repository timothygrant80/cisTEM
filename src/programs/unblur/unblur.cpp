#include "../../core/core_headers.h"

class
UnBlurApp : public MyApp
{

	public:

	virtual void DoCalculation();

	private:

};

IMPLEMENT_APP(UnBlurApp)

// overide the do calculation method which will be what is actually run..

void UnBlurApp::DoCalculation()
{
	MyDebugPrint("Doing a calculation..");

}
