#include "../../core/core_headers.h"

class
UnBlurApp : public MyApp
{

	public:

	bool DoCalculation();

	private:

};

IMPLEMENT_APP(UnBlurApp)

// overide the do calculation method which will be what is actually run..

bool UnBlurApp::DoCalculation()
{
	MyDebugPrint("Doing a calculation..");
	MyDebugPrint("There are %i arguments", my_current_job.number_of_arguments);

	MyDebugPrint("First Argument = %i", my_current_job.arguments[0].type_of_argument);

	for (long counter = 0; counter < my_current_job.number_of_arguments; counter++)
	{
		if (my_current_job.arguments[counter].type_of_argument == TEXT)
		{
			MyDebugPrint("Argument %li = %s", counter, my_current_job.arguments[counter].ReturnStringArgument())
		}
		else
		if (my_current_job.arguments[counter].type_of_argument == INTEGER)
		{
			MyDebugPrint("Argument %li = %i", counter, my_current_job.arguments[counter].ReturnIntegerArgument())
		}
		else
		if (my_current_job.arguments[counter].type_of_argument == FLOAT)
		{
			MyDebugPrint("Argument %li = %f", counter, my_current_job.arguments[counter].ReturnFloatArgument())
		}
		else
		if (my_current_job.arguments[counter].type_of_argument == BOOL)
		{
			if (my_current_job.arguments[counter].ReturnBoolArgument() == TRUE)
			{
				MyDebugPrint("Argument %li = YES", counter);
			}
			else
			{
				MyDebugPrint("Argument %li = NO", counter);
			}

		}
		else
		{
			//MyDebugPrint("Unknown Argument!!");
		}

	}

	//SendError(wxString::Format("Error: (Not Really) I am aligning job #%i (%s)", my_current_job.job_number, my_current_job.arguments[0].ReturnStringArgument()));


	//MyDebugPrint("That is all arguments")

	return true;

}
