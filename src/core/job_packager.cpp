#include "core_headers.h"

bool JobPackage::SendJobPackage(wxSocketBase *socket) // package the whole object into a single char stream which can be decoded at the other end..
{
	SETUP_SOCKET_CODES

	long counter;
	long job_counter;
	long argument_counter;
	long byte_counter;
	long command_counter;

	int length_of_string;
	int number_of_arguments;
	int temp_int;
	int temp_job_number;
	float temp_float;
	std::string temp_string;

	unsigned char *char_pointer;

	// first work out how long the character array needs to be..

	long transfer_size = ReturnEncodedByteTransferSize();

	// allocate a character array for the jobs..

	unsigned char *transfer_buffer = new unsigned char[transfer_size];

	// fill the buffer, first number of jobs

	char_pointer = (unsigned char *)&number_of_jobs;

	transfer_buffer[0] = char_pointer[0];
	transfer_buffer[1] = char_pointer[1];
	transfer_buffer[2] = char_pointer[2];
	transfer_buffer[3] = char_pointer[3];

	///////////////////////////////////////

	// RUN PROFILE

	//////////////////////////////////////

	// number_of_run_commands

	char_pointer = (unsigned char *)&my_profile.number_of_run_commands;

	transfer_buffer[4] = char_pointer[0];
	transfer_buffer[5] = char_pointer[1];
	transfer_buffer[6] = char_pointer[2];
	transfer_buffer[7] = char_pointer[3];

	// executable name

	temp_int = my_profile.executable_name.Length();
	char_pointer = (unsigned char *)&temp_int;

	transfer_buffer[8] = char_pointer[0];
	transfer_buffer[9] = char_pointer[1];
	transfer_buffer[10] = char_pointer[2];
	transfer_buffer[11] = char_pointer[3];

	byte_counter = 12;

	for (counter = 0; counter < my_profile.executable_name.Length(); counter++)
	{
		transfer_buffer[byte_counter] = my_profile.executable_name.GetChar(counter);

		byte_counter++;
	}

	// gui_address

	temp_int = my_profile.gui_address.Length();
	char_pointer = (unsigned char *)&temp_int;

	transfer_buffer[byte_counter] = char_pointer[0];
	byte_counter++;
	transfer_buffer[byte_counter] = char_pointer[1];
	byte_counter++;
	transfer_buffer[byte_counter] = char_pointer[2];
	byte_counter++;
	transfer_buffer[byte_counter] = char_pointer[3];
	byte_counter++;

	for (counter = 0; counter < my_profile.gui_address.Length(); counter++)
	{
		transfer_buffer[byte_counter] = my_profile.gui_address.GetChar(counter);
		byte_counter++;
	}

	// controller_address

	temp_int = my_profile.controller_address.Length();
	char_pointer = (unsigned char *)&temp_int;

	transfer_buffer[byte_counter] = char_pointer[0];
	byte_counter++;
	transfer_buffer[byte_counter] = char_pointer[1];
	byte_counter++;
	transfer_buffer[byte_counter] = char_pointer[2];
	byte_counter++;
	transfer_buffer[byte_counter] = char_pointer[3];
	byte_counter++;

	for (counter = 0; counter < my_profile.controller_address.Length(); counter++)
	{
		transfer_buffer[byte_counter] = my_profile.controller_address.GetChar(counter);
		byte_counter++;
	}

	// now add each run_command

	for (command_counter = 0; command_counter < my_profile.number_of_run_commands; command_counter++)
	{
		length_of_string = my_profile.run_commands[command_counter].command_to_run.Length();

		char_pointer = (unsigned char *)&length_of_string;

		transfer_buffer[byte_counter] = char_pointer[0];
		byte_counter++;
		transfer_buffer[byte_counter] = char_pointer[1];
		byte_counter++;
		transfer_buffer[byte_counter] = char_pointer[2];
		byte_counter++;
		transfer_buffer[byte_counter] = char_pointer[3];
		byte_counter++;

		// now the contents of the command..

		for (counter = 0; counter < length_of_string; counter++)
		{
			transfer_buffer[byte_counter] = static_cast<char>(my_profile.run_commands[command_counter].command_to_run[counter]);
			byte_counter++;
		}

		// number of copies..

		char_pointer = (unsigned char *)&my_profile.run_commands[command_counter].number_of_copies;

		transfer_buffer[byte_counter] = char_pointer[0];
		byte_counter++;
		transfer_buffer[byte_counter] = char_pointer[1];
		byte_counter++;
		transfer_buffer[byte_counter] = char_pointer[2];
		byte_counter++;
		transfer_buffer[byte_counter] = char_pointer[3];
		byte_counter++;

		// number of threads..

		char_pointer = (unsigned char *)&my_profile.run_commands[command_counter].number_of_threads_per_copy;

		transfer_buffer[byte_counter] = char_pointer[0];
		byte_counter++;
		transfer_buffer[byte_counter] = char_pointer[1];
		byte_counter++;
		transfer_buffer[byte_counter] = char_pointer[2];
		byte_counter++;
		transfer_buffer[byte_counter] = char_pointer[3];
		byte_counter++;

		// should_override..

		int should_override = int(my_profile.run_commands[command_counter].override_total_copies);
		char_pointer = (unsigned char *)&should_override;

		transfer_buffer[byte_counter] = char_pointer[0];
		byte_counter++;
		transfer_buffer[byte_counter] = char_pointer[1];
		byte_counter++;
		transfer_buffer[byte_counter] = char_pointer[2];
		byte_counter++;
		transfer_buffer[byte_counter] = char_pointer[3];
		byte_counter++;

		// overriden number of copies..

		char_pointer = (unsigned char *)&my_profile.run_commands[command_counter].overriden_number_of_copies;

		transfer_buffer[byte_counter] = char_pointer[0];
		byte_counter++;
		transfer_buffer[byte_counter] = char_pointer[1];
		byte_counter++;
		transfer_buffer[byte_counter] = char_pointer[2];
		byte_counter++;
		transfer_buffer[byte_counter] = char_pointer[3];
		byte_counter++;

		// delay time in ms..

		char_pointer = (unsigned char *)&my_profile.run_commands[command_counter].delay_time_in_ms;

		transfer_buffer[byte_counter] = char_pointer[0];
		byte_counter++;
		transfer_buffer[byte_counter] = char_pointer[1];
		byte_counter++;
		transfer_buffer[byte_counter] = char_pointer[2];
		byte_counter++;
		transfer_buffer[byte_counter] = char_pointer[3];
		byte_counter++;
	}

	// now append all the job info..

	///////////////////

	// END RUN PROFILE

	//////////////////////

	for (job_counter = 0; job_counter < number_of_jobs; job_counter++)
	{
		// job_number for this job

		char_pointer = (unsigned char *)&jobs[job_counter].job_number;

		transfer_buffer[byte_counter] = char_pointer[0];
		byte_counter++;
		transfer_buffer[byte_counter] = char_pointer[1];
		byte_counter++;
		transfer_buffer[byte_counter] = char_pointer[2];
		byte_counter++;
		transfer_buffer[byte_counter] = char_pointer[3];
		byte_counter++;

		// add number of arguments for this job

		char_pointer = (unsigned char *)&jobs[job_counter].number_of_arguments;

		transfer_buffer[byte_counter] = char_pointer[0];
		byte_counter++;
		transfer_buffer[byte_counter] = char_pointer[1];
		byte_counter++;
		transfer_buffer[byte_counter] = char_pointer[2];
		byte_counter++;
		transfer_buffer[byte_counter] = char_pointer[3];
		byte_counter++;

		// for this job we loop over all arguments

		for (argument_counter = 0; argument_counter < jobs[job_counter].number_of_arguments; argument_counter++)
		{
			// ok, what is this argument..

			if (jobs[job_counter].arguments[argument_counter].type_of_argument == INTEGER)
			{
				// set the descriptor byte

				transfer_buffer[byte_counter] = INTEGER;
				byte_counter++;

				// set the value of the integer..

				temp_int = jobs[job_counter].arguments[argument_counter].ReturnIntegerArgument();
				char_pointer = (unsigned char *)&temp_int;

				transfer_buffer[byte_counter] = char_pointer[0];
				byte_counter++;
				transfer_buffer[byte_counter] = char_pointer[1];
				byte_counter++;
				transfer_buffer[byte_counter] = char_pointer[2];
				byte_counter++;
				transfer_buffer[byte_counter] = char_pointer[3];
				byte_counter++;
			}
			else if (jobs[job_counter].arguments[argument_counter].type_of_argument == FLOAT)
			{
				// set the descriptor byte

				transfer_buffer[byte_counter] = FLOAT;
				byte_counter++;

				// set the value of the float..

				temp_float = jobs[job_counter].arguments[argument_counter].ReturnFloatArgument();
				char_pointer = (unsigned char *)&temp_float;

				transfer_buffer[byte_counter] = char_pointer[0];
				byte_counter++;
				transfer_buffer[byte_counter] = char_pointer[1];
				byte_counter++;
				transfer_buffer[byte_counter] = char_pointer[2];
				byte_counter++;
				transfer_buffer[byte_counter] = char_pointer[3];
				byte_counter++;
			}
			else if (jobs[job_counter].arguments[argument_counter].type_of_argument == BOOL)
			{
				// set the descriptor byte

				transfer_buffer[byte_counter] = BOOL;
				byte_counter++;

				// set the value of the bool..

				transfer_buffer[byte_counter] = (unsigned char)jobs[job_counter].arguments[argument_counter].ReturnBoolArgument();
				byte_counter++;
			}
			else if (jobs[job_counter].arguments[argument_counter].type_of_argument == TEXT)
			{
				// set the descriptor byte

				transfer_buffer[byte_counter] = TEXT;
				byte_counter++;

				// add the length of the string..

				temp_string = jobs[job_counter].arguments[argument_counter].ReturnStringArgument();
				length_of_string = temp_string.length();

				char_pointer = (unsigned char *)&length_of_string;

				transfer_buffer[byte_counter] = char_pointer[0];
				byte_counter++;
				transfer_buffer[byte_counter] = char_pointer[1];
				byte_counter++;
				transfer_buffer[byte_counter] = char_pointer[2];
				byte_counter++;
				transfer_buffer[byte_counter] = char_pointer[3];
				byte_counter++;

				// now add the contents of the string..

				for (counter = 0; counter < length_of_string; counter++)
				{
					transfer_buffer[byte_counter] = temp_string[counter];
					byte_counter++;
				}
			}
			else
			{
				MyPrintWithDetails("Unknown Argument Type!");
				DEBUG_ABORT;
			}
		}
	}

	// now we should everything encoded, so send the information to the socket..
	// disable events on the socket..
	//socket->SetNotify(wxSOCKET_LOST_FLAG);
	// inform what we want to do..
	//	 if (WriteToSocket(socket, socket_sending_job_package, SOCKET_CODE_SIZE, true, "SendSocketJobType", FUNCTION_DETAILS_AS_WXSTRING) == false)
	//	 {
	//		 delete [] transfer_buffer;
	//		 return false;
	//	 }

	// first - send how many bytes it is..

	if (WriteToSocket(socket, &transfer_size, sizeof(long), true, "SendTransferSize", FUNCTION_DETAILS_AS_WXSTRING) == false)
	{
		delete[] transfer_buffer;
		return false;
	}

	// now send the whole buffer..

	if (WriteToSocket(socket, transfer_buffer, transfer_size, true, "SendTransferBuffer", FUNCTION_DETAILS_AS_WXSTRING) == false)
	{
		delete[] transfer_buffer;
		return false;
	}

	delete[] transfer_buffer;
	return true;
}

bool JobPackage::ReceiveJobPackage(wxSocketBase *socket)
{
	SETUP_SOCKET_CODES

	long counter;
	long command_counter;
	long job_counter;
	long argument_counter;
	long byte_counter;
	long transfer_size;

	int wanted_number_of_jobs;
	int wanted_number_of_processes;
	int number_of_run_commands;
	int number_of_copies;
	int number_of_threads_per_copy;
	int overriden_total_number_of_copies;
	int override_total_number_of_copies;
	int delay_in_ms;
	std::string wanted_command_to_run;

	int length_of_string;
	int number_of_arguments;
	int temp_job_number;
	int temp_int;
	float temp_float;
	std::string temp_string;

	unsigned char *char_pointer;

	RunProfile temp_run_profile;
	wxString temp_wxstring;
	wxString executable_name;
	wxString gui_address;
	wxString controller_address;

	// disable events on the socket..
	//	socket->SetNotify(wxSOCKET_LOST_FLAG);
	//	socket->SetFlags(wxSOCKET_BLOCK);
	// Send a message saying we are ready to receive the package

	// receive how many bytes we need for the buffer..

	if (ReadFromSocket(socket, &transfer_size, sizeof(long), true, "SendTransferSize", FUNCTION_DETAILS_AS_WXSTRING) == false)
		return false;

	//MyDebugPrint("Package is %li bytes long", transfer_size);

	// allocate an array..

	unsigned char *transfer_buffer = new unsigned char[transfer_size];

	// now receive the package..

	if (ReadFromSocket(socket, transfer_buffer, transfer_size, true, "SendTransferBuffer", FUNCTION_DETAILS_AS_WXSTRING) == false)
		return false;
	//wxPrintf("We read %u bytes\n", socket->LastReadCount());

	//MyDebugPrint("Received package, decoding job...");

	// restore socket events..
	//    socket->SetNotify(wxSOCKET_LOST_FLAG | wxSOCKET_INPUT_FLAG);

	// now we need to decode the buffer

	// number of jobs
	char_pointer = (unsigned char *)&wanted_number_of_jobs;
	char_pointer[0] = transfer_buffer[0];
	char_pointer[1] = transfer_buffer[1];
	char_pointer[2] = transfer_buffer[2];
	char_pointer[3] = transfer_buffer[3];

	//MyDebugPrint("There are %i jobs", wanted_number_of_jobs);

	// number of run commands
	char_pointer = (unsigned char *)&number_of_run_commands;
	char_pointer[0] = transfer_buffer[4];
	char_pointer[1] = transfer_buffer[5];
	char_pointer[2] = transfer_buffer[6];
	char_pointer[3] = transfer_buffer[7];

	//MyDebugPrint("There are %i commands", number_of_run_commands);

	// executable name

	char_pointer = (unsigned char *)&temp_int;
	char_pointer[0] = transfer_buffer[8];
	char_pointer[1] = transfer_buffer[9];
	char_pointer[2] = transfer_buffer[10];
	char_pointer[3] = transfer_buffer[11];

	byte_counter = 12;

	executable_name = "";

	for (counter = 0; counter < temp_int; counter++)
	{
		executable_name += transfer_buffer[byte_counter];
		byte_counter++;
	}

	// gui address

	char_pointer = (unsigned char *)&temp_int;
	char_pointer[0] = transfer_buffer[byte_counter];
	byte_counter++;
	char_pointer[1] = transfer_buffer[byte_counter];
	byte_counter++;
	char_pointer[2] = transfer_buffer[byte_counter];
	byte_counter++;
	char_pointer[3] = transfer_buffer[byte_counter];
	byte_counter++;

	gui_address = "";

	for (counter = 0; counter < temp_int; counter++)
	{
		gui_address += transfer_buffer[byte_counter];
		byte_counter++;
	}

	// controller address

	char_pointer = (unsigned char *)&temp_int;
	char_pointer[0] = transfer_buffer[byte_counter];
	byte_counter++;
	char_pointer[1] = transfer_buffer[byte_counter];
	byte_counter++;
	char_pointer[2] = transfer_buffer[byte_counter];
	byte_counter++;
	char_pointer[3] = transfer_buffer[byte_counter];
	byte_counter++;

	controller_address = "";

	for (counter = 0; counter < temp_int; counter++)
	{
		controller_address += transfer_buffer[byte_counter];
		byte_counter++;
	}

	// each run command

	for (command_counter = 0; command_counter < number_of_run_commands; command_counter++)
	{
		char_pointer = (unsigned char *)&length_of_string;

		char_pointer[0] = transfer_buffer[byte_counter];
		byte_counter++;
		char_pointer[1] = transfer_buffer[byte_counter];
		byte_counter++;
		char_pointer[2] = transfer_buffer[byte_counter];
		byte_counter++;
		char_pointer[3] = transfer_buffer[byte_counter];
		byte_counter++;

		temp_wxstring = "";

		// now the contents of the command..

		for (counter = 0; counter < length_of_string; counter++)
		{
			temp_wxstring += transfer_buffer[byte_counter];
			byte_counter++;
		}

		// number of copies..

		char_pointer = (unsigned char *)&number_of_copies;

		char_pointer[0] = transfer_buffer[byte_counter];
		byte_counter++;
		char_pointer[1] = transfer_buffer[byte_counter];
		byte_counter++;
		char_pointer[2] = transfer_buffer[byte_counter];
		byte_counter++;
		char_pointer[3] = transfer_buffer[byte_counter];
		byte_counter++;

		// number of threads..

		char_pointer = (unsigned char *)&number_of_threads_per_copy;

		char_pointer[0] = transfer_buffer[byte_counter];
		byte_counter++;
		char_pointer[1] = transfer_buffer[byte_counter];
		byte_counter++;
		char_pointer[2] = transfer_buffer[byte_counter];
		byte_counter++;
		char_pointer[3] = transfer_buffer[byte_counter];
		byte_counter++;

		// should_override..

		char_pointer = (unsigned char *)&override_total_number_of_copies;

		char_pointer[0] = transfer_buffer[byte_counter];
		byte_counter++;
		char_pointer[1] = transfer_buffer[byte_counter];
		byte_counter++;
		char_pointer[2] = transfer_buffer[byte_counter];
		byte_counter++;
		char_pointer[3] = transfer_buffer[byte_counter];
		byte_counter++;

		// override_number..

		char_pointer = (unsigned char *)&overriden_total_number_of_copies;

		char_pointer[0] = transfer_buffer[byte_counter];
		byte_counter++;
		char_pointer[1] = transfer_buffer[byte_counter];
		byte_counter++;
		char_pointer[2] = transfer_buffer[byte_counter];
		byte_counter++;
		char_pointer[3] = transfer_buffer[byte_counter];
		byte_counter++;

		// delay_in_ms..

		char_pointer = (unsigned char *)&delay_in_ms;

		char_pointer[0] = transfer_buffer[byte_counter];
		byte_counter++;
		char_pointer[1] = transfer_buffer[byte_counter];
		byte_counter++;
		char_pointer[2] = transfer_buffer[byte_counter];
		byte_counter++;
		char_pointer[3] = transfer_buffer[byte_counter];
		byte_counter++;

		// add the command.

		temp_run_profile.AddCommand(temp_wxstring, number_of_copies, number_of_threads_per_copy, bool(override_total_number_of_copies), overriden_total_number_of_copies, delay_in_ms);
	}

	// now we need to loop over all the jobs..

	temp_run_profile.gui_address = gui_address;
	temp_run_profile.controller_address = controller_address;

	Reset(temp_run_profile, executable_name, wanted_number_of_jobs);

	for (job_counter = 0; job_counter < number_of_jobs; job_counter++)
	{
		char_pointer = (unsigned char *)&temp_job_number;

		char_pointer[0] = transfer_buffer[byte_counter];
		byte_counter++;
		char_pointer[1] = transfer_buffer[byte_counter];
		byte_counter++;
		char_pointer[2] = transfer_buffer[byte_counter];
		byte_counter++;
		char_pointer[3] = transfer_buffer[byte_counter];
		byte_counter++;

		// How many arguments are there for this job

		char_pointer = (unsigned char *)&temp_int;

		char_pointer[0] = transfer_buffer[byte_counter];
		byte_counter++;
		char_pointer[1] = transfer_buffer[byte_counter];
		byte_counter++;
		char_pointer[2] = transfer_buffer[byte_counter];
		byte_counter++;
		char_pointer[3] = transfer_buffer[byte_counter];
		byte_counter++;

		// reset the job..
		jobs[job_counter].Reset(temp_int);
		jobs[job_counter].job_number = temp_job_number;

		// for this job we loop over all arguments

		for (argument_counter = 0; argument_counter < jobs[job_counter].number_of_arguments; argument_counter++)
		{
			// ok, what is this argument..

			jobs[job_counter].arguments[argument_counter].type_of_argument = int(transfer_buffer[byte_counter]);

			byte_counter++;

			if (jobs[job_counter].arguments[argument_counter].type_of_argument == INTEGER)
			{
				// read the value of the integer..

				char_pointer = (unsigned char *)&temp_int;

				char_pointer[0] = transfer_buffer[byte_counter];
				byte_counter++;
				char_pointer[1] = transfer_buffer[byte_counter];
				byte_counter++;
				char_pointer[2] = transfer_buffer[byte_counter];
				byte_counter++;
				char_pointer[3] = transfer_buffer[byte_counter];
				byte_counter++;

				jobs[job_counter].arguments[argument_counter].SetIntArgument(temp_int);
			}
			else if (jobs[job_counter].arguments[argument_counter].type_of_argument == FLOAT)
			{
				// read the value of the float..

				char_pointer = (unsigned char *)&temp_float;

				char_pointer[0] = transfer_buffer[byte_counter];
				byte_counter++;
				char_pointer[1] = transfer_buffer[byte_counter];
				byte_counter++;
				char_pointer[2] = transfer_buffer[byte_counter];
				byte_counter++;
				char_pointer[3] = transfer_buffer[byte_counter];
				byte_counter++;

				jobs[job_counter].arguments[argument_counter].SetFloatArgument(temp_float);
			}
			else if (jobs[job_counter].arguments[argument_counter].type_of_argument == BOOL)
			{
				jobs[job_counter].arguments[argument_counter].SetBoolArgument(bool(transfer_buffer[byte_counter]));
				byte_counter++;
			}
			else if (jobs[job_counter].arguments[argument_counter].type_of_argument == TEXT)
			{
				// read length of command string

				char_pointer = (unsigned char *)&length_of_string;

				char_pointer[0] = transfer_buffer[byte_counter];
				byte_counter++;
				char_pointer[1] = transfer_buffer[byte_counter];
				byte_counter++;
				char_pointer[2] = transfer_buffer[byte_counter];
				byte_counter++;
				char_pointer[3] = transfer_buffer[byte_counter];
				byte_counter++;

				// allocate memory

				jobs[job_counter].arguments[argument_counter].string_argument = new std::string;
				jobs[job_counter].arguments[argument_counter].string_argument[0].clear();
				jobs[job_counter].arguments[argument_counter].is_allocated = true;

				// fill the string..

				for (counter = 0; counter < length_of_string; counter++)
				{
					jobs[job_counter].arguments[argument_counter].string_argument[0] += transfer_buffer[byte_counter];
					byte_counter++;
				}
			}
			else
			{
				MyPrintWithDetails("Unknown Argument Type!");
				DEBUG_ABORT;
			}
		}
	}

	MyDebugPrint("Job Decoded");

	// delete the buffer
	delete[] transfer_buffer;

	return true;
}

long JobPackage::ReturnEncodedByteTransferSize()
{
	long byte_size = 0;
	long counter;

	// size of the profile..

	byte_size += 8; // number_of_jobs and number_of_run_commands

	// executable name

	byte_size += 4; // length of string
	byte_size += my_profile.executable_name.Length();

	// gui_address

	byte_size += 4;
	byte_size += my_profile.gui_address.Length();

	// controller address

	byte_size += 4;
	byte_size += my_profile.controller_address.Length();

	for (counter = 0; counter < my_profile.number_of_run_commands; counter++)
	{
		byte_size += 4;														   // length_of_current_command;
		byte_size += my_profile.run_commands[counter].command_to_run.Length(); // actual text;
		byte_size += 4;														   // number_of_copies;
		byte_size += 4;														   // number_of_threads
		byte_size += 4;														   // override_total_commands as int;
		byte_size += 4;														   // overriden_total_commannds;
		byte_size += 4;														   // delay time in ms
	}

	for (counter = 0; counter < number_of_jobs; counter++)
	{
		byte_size += jobs[counter].ReturnEncodedByteTransferSize();
	}

	return byte_size;
}

JobPackage::JobPackage(RunProfile wanted_profile, wxString wanted_executable_name, int wanted_number_of_jobs)
{
	Reset(wanted_profile, wanted_executable_name, wanted_number_of_jobs);
}

JobPackage::JobPackage()
{
	number_of_jobs = 0;
	number_of_added_jobs = 0;

	// memory allocation..

	if (number_of_jobs > 0)
	{
		if (number_of_jobs == 1)
			jobs = new RunJob;
		else
			jobs = new RunJob[number_of_jobs];
	}
}

JobPackage::~JobPackage()
{
	if (number_of_jobs > 0)
	{
		if (number_of_jobs == 1)
			delete jobs;
		else
			delete[] jobs;
	}
}

void JobPackage::Reset(RunProfile wanted_profile, wxString wanted_executable_name, int wanted_number_of_jobs)
{
	if (number_of_jobs > 0)
	{
		if (number_of_jobs == 1)
			delete jobs;
		else
			delete[] jobs;
	}

	number_of_jobs = wanted_number_of_jobs;
	number_of_added_jobs = 0;

	// memory allocation..

	if (number_of_jobs == 1)
		jobs = new RunJob;
	else
		jobs = new RunJob[number_of_jobs];

	my_profile = wanted_profile;
	my_profile.executable_name = wanted_executable_name;

	//my_profile.SubstituteExecutableName(wanted_executable_name);
}

void JobPackage::AddJob(const char *format, ...)
{
	//wxPrintf("Number of Jobs: %i, Number of Added Jobs: %i \n", number_of_jobs, number_of_added_jobs);
	MyDebugAssertTrue(number_of_added_jobs < number_of_jobs, "number of jobs exceeded!");

	va_list args;
	va_start(args, format);

	jobs[number_of_added_jobs].job_number = number_of_added_jobs;
	jobs[number_of_added_jobs].SetArguments(format, args);

	va_end(args);

	number_of_added_jobs++;
}

int JobPackage::ReturnNumberOfJobsRemaining()
{
	int number_remaining = 0;

	for (long counter = 0; counter < number_of_jobs; counter++)
	{
		if (jobs[counter].has_been_run == false)
			number_remaining++;
	}

	return number_remaining;
}

JobPackage &JobPackage::operator=(const JobPackage *other_package)
{
	// Check for self assignment
	if (this != other_package)
	{
		if (number_of_jobs > 0)
		{
			if (number_of_jobs == 1)
				delete jobs;
			else
				delete[] jobs;
		}

		number_of_jobs = other_package->number_of_jobs;
		number_of_added_jobs = other_package->number_of_added_jobs;

		my_profile = other_package->my_profile;

		if (number_of_jobs > 0)
		{
			if (number_of_jobs == 1)
				jobs = new RunJob;
			else
				jobs = new RunJob[number_of_jobs];
		}

		for (int counter = 0; counter < number_of_jobs; counter++)
		{
			jobs[counter] = other_package->jobs[counter];
		}
	}

	return *this;
}

JobPackage &JobPackage::operator=(const JobPackage &other_package)
{
	*this = &other_package;
	return *this;
}

RunJob::RunJob()
{
	job_number = -1;
	has_been_run = false;
	number_of_arguments = 0;
	arguments = NULL;
}

RunJob::~RunJob()
{
	Deallocate();
}

bool RunJob::SendJob(wxSocketBase *socket)
{
	SETUP_SOCKET_CODES

	long counter;
	long argument_counter;
	long byte_counter = 0;

	int length_of_string;
	int temp_int;
	float temp_float;
	std::string temp_string;

	unsigned char *char_pointer;

	long transfer_size = ReturnEncodedByteTransferSize();

	// allocate a character array for the jobs..

	unsigned char *transfer_buffer = new unsigned char[transfer_size];

	// job_number

	char_pointer = (unsigned char *)&job_number;

	transfer_buffer[0] = char_pointer[0];
	transfer_buffer[1] = char_pointer[1];
	transfer_buffer[2] = char_pointer[2];
	transfer_buffer[3] = char_pointer[3];

	// number of arguments..

	char_pointer = (unsigned char *)&number_of_arguments;

	transfer_buffer[4] = char_pointer[0];
	transfer_buffer[5] = char_pointer[1];
	transfer_buffer[6] = char_pointer[2];
	transfer_buffer[7] = char_pointer[3];

	byte_counter = 8;

	for (argument_counter = 0; argument_counter < number_of_arguments; argument_counter++)
	{
		// ok, what is this argument..

		if (arguments[argument_counter].type_of_argument == INTEGER)
		{
			// set the descriptor byte

			transfer_buffer[byte_counter] = INTEGER;
			byte_counter++;

			// set the value of the integer..

			temp_int = arguments[argument_counter].ReturnIntegerArgument();
			char_pointer = (unsigned char *)&temp_int;

			transfer_buffer[byte_counter] = char_pointer[0];
			byte_counter++;
			transfer_buffer[byte_counter] = char_pointer[1];
			byte_counter++;
			transfer_buffer[byte_counter] = char_pointer[2];
			byte_counter++;
			transfer_buffer[byte_counter] = char_pointer[3];
			byte_counter++;
		}
		else if (arguments[argument_counter].type_of_argument == FLOAT)
		{
			// set the descriptor byte

			transfer_buffer[byte_counter] = FLOAT;
			byte_counter++;

			// set the value of the float..

			temp_float = arguments[argument_counter].ReturnFloatArgument();
			char_pointer = (unsigned char *)&temp_float;

			transfer_buffer[byte_counter] = char_pointer[0];
			byte_counter++;
			transfer_buffer[byte_counter] = char_pointer[1];
			byte_counter++;
			transfer_buffer[byte_counter] = char_pointer[2];
			byte_counter++;
			transfer_buffer[byte_counter] = char_pointer[3];
			byte_counter++;
		}
		else if (arguments[argument_counter].type_of_argument == BOOL)
		{
			// set the descriptor byte

			transfer_buffer[byte_counter] = BOOL;
			byte_counter++;

			// set the value of the bool..

			transfer_buffer[byte_counter] = (unsigned char)arguments[argument_counter].ReturnBoolArgument();
			byte_counter++;
		}
		else if (arguments[argument_counter].type_of_argument == TEXT)
		{
			// set the descriptor byte

			transfer_buffer[byte_counter] = TEXT;
			byte_counter++;

			// add the length of the string..

			temp_string = arguments[argument_counter].ReturnStringArgument();
			length_of_string = temp_string.length();

			char_pointer = (unsigned char *)&length_of_string;

			transfer_buffer[byte_counter] = char_pointer[0];
			byte_counter++;
			transfer_buffer[byte_counter] = char_pointer[1];
			byte_counter++;
			transfer_buffer[byte_counter] = char_pointer[2];
			byte_counter++;
			transfer_buffer[byte_counter] = char_pointer[3];
			byte_counter++;

			// now add the contents of the string..

			for (counter = 0; counter < length_of_string; counter++)
			{
				transfer_buffer[byte_counter] = temp_string[counter];
				byte_counter++;
			}
		}
		else
		{
			MyPrintWithDetails("Unrecognized argument!");
			DEBUG_ABORT;
		}
	}

	// now we should everything encoded, so send the information to the socket..
	// disable events on the socket..

	//	 socket->SetNotify(wxSOCKET_LOST_FLAG);

	// inform what we want to do..
	if (WriteToSocket(socket, socket_ready_to_send_single_job, SOCKET_CODE_SIZE, true, "SendSocketJobType", FUNCTION_DETAILS_AS_WXSTRING) == false)
	{
		delete[] transfer_buffer;
		return false;
	}
	// we should get a message saying the socket is ready to receive the data..
	// ReadFromSocket(socket, &socket_input_buffer, SOCKET_CODE_SIZE);

	// check it is ok..

	// first - send how many bytes it is..

	if (WriteToSocket(socket, &transfer_size, sizeof(long), true, "SendTransferSize", FUNCTION_DETAILS_AS_WXSTRING) == false)
	{
		delete[] transfer_buffer;
		return false;
	}

	// now send the whole buffer..

	if (WriteToSocket(socket, transfer_buffer, transfer_size, true, "SendTransferBuffer", FUNCTION_DETAILS_AS_WXSTRING) == false)
	{
		delete[] transfer_buffer;
		return false;
	}

	//	 }
	//	 else
	//	 {
	//		 MyPrintWithDetails("Oops, didn't understand the reply!");
	//		 DEBUG_ABORT;
	//	 }

	// restore socket events..

	//    socket->SetNotify(wxSOCKET_LOST_FLAG | wxSOCKET_INPUT_FLAG);
	delete[] transfer_buffer;
	return true;
}

bool RunJob::RecieveJob(wxSocketBase *socket)
{
	SETUP_SOCKET_CODES

	long counter;
	long argument_counter;
	long byte_counter;
	long transfer_size;

	int length_of_string;
	int number_of_arguments;
	int temp_int;
	int temp_job_number;
	float temp_float;
	std::string temp_string;

	unsigned char *char_pointer;

	// disable events on the socket..
	//	socket->SetNotify(wxSOCKET_LOST_FLAG);

	// Send a message saying we are ready to receive the package

	// receive how many bytes we need for the buffer..

	if (ReadFromSocket(socket, &transfer_size, sizeof(long), true, "SendTransferSize", FUNCTION_DETAILS_AS_WXSTRING) == false)
		return false;

	// allocate an array..

	unsigned char *transfer_buffer = new unsigned char[transfer_size];

	// now receive the package..

	if (ReadFromSocket(socket, transfer_buffer, transfer_size, true, "SendTransferBuffer", FUNCTION_DETAILS_AS_WXSTRING) == false)
	{
		delete[] transfer_buffer;
		return false;
	}

	// restore socket events..
	//    socket->SetNotify(wxSOCKET_LOST_FLAG | wxSOCKET_INPUT_FLAG);

	// now we need to decode the buffer

	byte_counter = 0;

	// job id

	char_pointer = (unsigned char *)&temp_job_number;
	char_pointer[0] = transfer_buffer[byte_counter];
	byte_counter++;
	char_pointer[1] = transfer_buffer[byte_counter];
	byte_counter++;
	char_pointer[2] = transfer_buffer[byte_counter];
	byte_counter++;
	char_pointer[3] = transfer_buffer[byte_counter];
	byte_counter++;

	// How many arguments are there for this job

	char_pointer = (unsigned char *)&number_of_arguments;
	char_pointer[0] = transfer_buffer[byte_counter];
	byte_counter++;
	char_pointer[1] = transfer_buffer[byte_counter];
	byte_counter++;
	char_pointer[2] = transfer_buffer[byte_counter];
	byte_counter++;
	char_pointer[3] = transfer_buffer[byte_counter];
	byte_counter++;

	// reset the job..
	Reset(number_of_arguments);
	job_number = temp_job_number;

	// for this job we loop over all arguments

	for (argument_counter = 0; argument_counter < number_of_arguments; argument_counter++)
	{
		// ok, what is this argument..

		arguments[argument_counter].type_of_argument = int(transfer_buffer[byte_counter]);

		byte_counter++;

		if (arguments[argument_counter].type_of_argument == INTEGER)
		{
			// read the value of the integer..
			char_pointer = (unsigned char *)&temp_int;

			char_pointer[0] = transfer_buffer[byte_counter];
			byte_counter++;
			char_pointer[1] = transfer_buffer[byte_counter];
			byte_counter++;
			char_pointer[2] = transfer_buffer[byte_counter];
			byte_counter++;
			char_pointer[3] = transfer_buffer[byte_counter];
			byte_counter++;

			arguments[argument_counter].SetIntArgument(temp_int);
		}
		else if (arguments[argument_counter].type_of_argument == FLOAT)
		{
			// read the value of the float..
			char_pointer = (unsigned char *)&temp_float;

			char_pointer[0] = transfer_buffer[byte_counter];
			byte_counter++;
			char_pointer[1] = transfer_buffer[byte_counter];
			byte_counter++;
			char_pointer[2] = transfer_buffer[byte_counter];
			byte_counter++;
			char_pointer[3] = transfer_buffer[byte_counter];
			byte_counter++;

			arguments[argument_counter].SetFloatArgument(temp_float);
		}
		else if (arguments[argument_counter].type_of_argument == BOOL)
		{

			arguments[argument_counter].SetBoolArgument(bool(transfer_buffer[byte_counter]));
			byte_counter++;
		}
		else if (arguments[argument_counter].type_of_argument == TEXT)
		{
			// read length of command string

			char_pointer = (unsigned char *)&length_of_string;

			char_pointer[0] = transfer_buffer[byte_counter];
			byte_counter++;
			char_pointer[1] = transfer_buffer[byte_counter];
			byte_counter++;
			char_pointer[2] = transfer_buffer[byte_counter];
			byte_counter++;
			char_pointer[3] = transfer_buffer[byte_counter];
			byte_counter++;

			// allocate memory

			arguments[argument_counter].string_argument = new std::string;
			arguments[argument_counter].string_argument[0].clear();
			arguments[argument_counter].is_allocated = true;

			// fill the string..

			for (counter = 0; counter < length_of_string; counter++)
			{
				arguments[argument_counter].string_argument[0] += transfer_buffer[byte_counter];
				byte_counter++;
			}
		}
		else
		{
			MyDebugPrint("Unknown Argument!!");
			DEBUG_ABORT;
		}
	}

	// delete the buffer
	delete[] transfer_buffer;
	return true;
}

void RunJob::Deallocate()
{
	if (number_of_arguments > 0)
	{
		if (number_of_arguments == 1)
			delete arguments;
		else
			delete[] arguments;
	}
}

void RunJob::ManualSetArguments(const char *format, ...)
{
	va_list args;
	va_start(args, format);

	SetArguments(format, args);

	va_end(args);
}

void RunJob::SetArguments(const char *format, va_list args)
{
	Deallocate();

	// work out the number of arguments

	number_of_arguments = strlen(format);

	// allocate space

	if (number_of_arguments == 1)
		arguments = new RunArgument;
	else
		arguments = new RunArgument[number_of_arguments];

	// fill the arguments..

	long counter = 0;

	while (*format != '\0')
	{
		if (*format == 't' || *format == 's') // argument is text..
		{
			arguments[counter].SetStringArgument(va_arg(args, const char *));
		}
		else if (*format == 'f') // float
		{
			arguments[counter].SetFloatArgument(va_arg(args, double));
		}
		else if (*format == 'i') // integer
		{
			arguments[counter].SetIntArgument(va_arg(args, int));
		}
		else if (*format == 'b') // bool
		{
			arguments[counter].SetBoolArgument(va_arg(args, int));
		}
		else
		{
			MyPrintWithDetails("Error: Unknown format character!\n");
		}

		counter++;
		++format;
	}
}

void RunJob::Reset(int wanted_number_of_arguments)
{
	Deallocate();
	number_of_arguments = wanted_number_of_arguments;

	if (number_of_arguments == 1)
		arguments = new RunArgument;
	else
		arguments = new RunArgument[number_of_arguments];

	job_number = -1;
	has_been_run = false;
}

wxString RunJob::PrintAllArgumentsTowxString()
{
	wxString string_to_return = "\n";

	for (int counter = 0; counter < number_of_arguments; counter++)
	{
		if (arguments[counter].type_of_argument == TEXT)
		{
			string_to_return += wxString::Format("Argument %3i is a string   : %s\n", counter, arguments[counter].ReturnStringArgument());
		}
		else if (arguments[counter].type_of_argument == INTEGER)
		{
			string_to_return += wxString::Format("Argument %3i is an integer : %i\n", counter, arguments[counter].ReturnIntegerArgument());
		}
		else if (arguments[counter].type_of_argument == FLOAT)
		{
			string_to_return += wxString::Format("Argument %3i is a float    : %f\n", counter, arguments[counter].ReturnFloatArgument());
		}
		else if (arguments[counter].type_of_argument == BOOL)
		{
			string_to_return += wxString::Format("Argument %3i is a bool     : ", counter);

			if (arguments[counter].ReturnBoolArgument() == true)
				string_to_return += wxString::Format("TRUE\n");
			else
				string_to_return += wxString::Format("FALSE\n");
		}
	}

	return string_to_return;
}

void RunJob::PrintAllArguments()
{
	wxPrintf(PrintAllArgumentsTowxString());
}

long RunJob::ReturnEncodedByteTransferSize()
{
	long byte_size = 0;
	long counter;

	for (counter = 0; counter < number_of_arguments; counter++)
	{
		byte_size += arguments[counter].ReturnEncodedByteTransferSize();
	}

	return byte_size + 8; // argument bytes, + 4 bytes for the number of arguments + 4 bytes for job_number
}

RunJob &RunJob::operator=(const RunJob *other_job)
{
	// Check for self assignment
	if (this != other_job)
	{
		Deallocate();
		number_of_arguments = other_job->number_of_arguments;

		if (number_of_arguments == 1)
			arguments = new RunArgument;
		else
			arguments = new RunArgument[number_of_arguments];

		job_number = other_job->job_number;
		has_been_run = other_job->has_been_run;

		for (int counter = 0; counter < number_of_arguments; counter++)
		{
			if (other_job->arguments[counter].type_of_argument == TEXT)
			{
				arguments[counter].SetStringArgument(other_job->arguments[counter].ReturnStringArgument().c_str());
			}
			else if (other_job->arguments[counter].type_of_argument == INTEGER)
			{
				arguments[counter].SetIntArgument(other_job->arguments[counter].ReturnIntegerArgument());
			}
			else if (other_job->arguments[counter].type_of_argument == FLOAT)
			{
				arguments[counter].SetFloatArgument(other_job->arguments[counter].ReturnFloatArgument());
			}
			else if (other_job->arguments[counter].type_of_argument == BOOL)
			{
				arguments[counter].SetBoolArgument(other_job->arguments[counter].ReturnBoolArgument());
			}
		}
	}

	return *this;
}

RunJob &RunJob::operator=(const RunJob &other_job)
{
	*this = &other_job;
	return *this;
}

RunArgument::RunArgument()
{
	is_allocated = false;
	type_of_argument = NONE;
	string_argument = NULL;
	integer_argument = NULL;
	float_argument = NULL;
	bool_argument = NULL;
}

RunArgument::~RunArgument()
{
	if (is_allocated == true)
		Deallocate();
}

void RunArgument::Deallocate()
{
	if (type_of_argument == TEXT)
		delete string_argument;
	else if (type_of_argument == INTEGER)
		delete integer_argument;
	else if (type_of_argument == FLOAT)
		delete float_argument;
	else if (type_of_argument == BOOL)
		delete bool_argument;

	is_allocated = false;
}

long RunArgument::ReturnEncodedByteTransferSize()
{
	MyDebugAssertTrue(type_of_argument != NONE, "Can't calculate size of a nothing argument!!");

	if (type_of_argument == TEXT)
		return string_argument->length() + 4 + 1; // descriptor byte + length of string (4 bytes) + 1 byte per character
	else if (type_of_argument == BOOL)
		return 2; // descriptor bytes + bool bytes
	else
		return 5; // descriptor byte + 4 data bytes
}

void RunArgument::SetStringArgument(const char *wanted_text)
{
	if (is_allocated == true)
		Deallocate();

	type_of_argument = TEXT;
	string_argument = new std::string;
	string_argument[0] = wanted_text;
	is_allocated = true;
}

void RunArgument::SetIntArgument(int wanted_argument)
{
	if (is_allocated == true)
		Deallocate();

	type_of_argument = INTEGER;
	integer_argument = new int;
	integer_argument[0] = wanted_argument;
	is_allocated = true;
}

void RunArgument::SetFloatArgument(float wanted_argument)
{
	if (is_allocated == true)
		Deallocate();

	type_of_argument = FLOAT;
	float_argument = new float;
	float_argument[0] = wanted_argument;
	is_allocated = true;
}

void RunArgument::SetBoolArgument(bool wanted_argument)
{
	if (is_allocated == true)
		Deallocate();

	type_of_argument = BOOL;
	bool_argument = new bool;
	bool_argument[0] = wanted_argument;
	is_allocated = true;
}

#include <wx/arrimpl.cpp> // this is a magic incantation which must be done!
WX_DEFINE_OBJARRAY(ArrayofJobResults);
WX_DEFINE_OBJARRAY(ArrayofJobPackages);

JobResult::JobResult()
{
	job_number = -1;
	result_size = 0;
	result_data = NULL;
}

JobResult::JobResult(int wanted_result_size, float *wanted_result_data)
{
	SetResult(wanted_result_size, wanted_result_data);
}

JobResult::~JobResult()
{
	if (result_size != 0 && result_data != NULL)
	{
		delete[] result_data;
	}
}

JobResult::JobResult(const JobResult &obj) // copy contructor
{
	result_size = 0;
	result_data = NULL;

	job_number = obj.job_number;
	SetResult(obj.result_size, obj.result_data);
}

JobResult &JobResult::operator=(const JobResult *other_result)
{
	// Check for self assignment
	if (this != other_result)
	{
		job_number = other_result->job_number;
		SetResult(other_result->result_size, other_result->result_data);
	}

	return *this;
}

JobResult &JobResult::operator=(const JobResult &other_result)
{
	*this = &other_result;
	return *this;
}

void JobResult::SetResult(int wanted_result_size, float *wanted_result_data)
{
	if (result_size != 0 || result_data != NULL)
	{
		delete[] result_data;
	}

	result_size = wanted_result_size;

	if (wanted_result_size > 0)
	{

		result_data = new float[result_size];

		for (int counter = 0; counter < result_size; counter++)
		{
			result_data[counter] = wanted_result_data[counter];
		}
	}
	else
	{
		result_data = NULL;
	}
}

bool JobResult::SendToSocket(wxSocketBase *wanted_socket)
{
	char job_number_and_result_size[8];
	unsigned char *byte_pointer;

	byte_pointer = (unsigned char *)&job_number;

	job_number_and_result_size[0] = byte_pointer[0];
	job_number_and_result_size[1] = byte_pointer[1];
	job_number_and_result_size[2] = byte_pointer[2];
	job_number_and_result_size[3] = byte_pointer[3];

	byte_pointer = (unsigned char *)&result_size;

	job_number_and_result_size[4] = byte_pointer[0];
	job_number_and_result_size[5] = byte_pointer[1];
	job_number_and_result_size[6] = byte_pointer[2];
	job_number_and_result_size[7] = byte_pointer[3];

	if (WriteToSocket(wanted_socket, &job_number_and_result_size, 8, true, "SendJobNumberAndResultSize", FUNCTION_DETAILS_AS_WXSTRING) == false)
		return false;

	if (result_size > 0)
	{
		if (WriteToSocket(wanted_socket, result_data, result_size * 4, true, "SendResultData", FUNCTION_DETAILS_AS_WXSTRING) == false)
			return false;
	}

	return true;
}

bool JobResult::ReceiveFromSocket(wxSocketBase *wanted_socket)
{

	char job_number_and_result_size[8];
	int new_result_size;
	unsigned char *byte_pointer;

	if (ReadFromSocket(wanted_socket, job_number_and_result_size, 8, true, "SendJobNumberAndResultSize", FUNCTION_DETAILS_AS_WXSTRING) == false)
		return false;

	byte_pointer = (unsigned char *)&job_number;
	byte_pointer[0] = job_number_and_result_size[0];
	byte_pointer[1] = job_number_and_result_size[1];
	byte_pointer[2] = job_number_and_result_size[2];
	byte_pointer[3] = job_number_and_result_size[3];

	byte_pointer = (unsigned char *)&new_result_size;

	byte_pointer[0] = job_number_and_result_size[4];
	byte_pointer[1] = job_number_and_result_size[5];
	byte_pointer[2] = job_number_and_result_size[6];
	byte_pointer[3] = job_number_and_result_size[7];

	if (new_result_size != result_size)
	{
		if (result_size != 0 && result_data != NULL)
		{
			delete[] result_data;
		}

		result_size = new_result_size;

		if (result_size == 0)
		{
			result_data = NULL;
		}
		else
			result_data = new float[new_result_size];
	}

	if (result_size > 0)
	{
		if (ReadFromSocket(wanted_socket, result_data, result_size * 4, true, "SendResultData", FUNCTION_DETAILS_AS_WXSTRING) == false)
			return false;
	}

	return true;
}

bool ReceiveResultQueueFromSocket(wxSocketBase *socket, ArrayofJobResults &my_array)
{
	int total_number_of_bytes;
	int number_of_jobs;
	int job_counter;
	int job_number;
	int result_size;
	int byte_counter;
	unsigned char *byte_pointer;
	int result_byte_counter;
	JobResult temp_result;

	// clear the array
	my_array.Clear();

	// recieve the total number of bytes..

	if (ReadFromSocket(socket, &total_number_of_bytes, sizeof(int), true, "SendResultQueueTotalBytes", FUNCTION_DETAILS_AS_WXSTRING) == false)
		return false;
	//wxPrintf("(Recieve) Total Size is %i bytes\n", total_number_of_bytes);
	// make the array..

	unsigned char *buffer_array = new unsigned char[total_number_of_bytes];

	// receieve

	if (ReadFromSocket(socket, buffer_array, total_number_of_bytes, true, "SendResultQueueData", FUNCTION_DETAILS_AS_WXSTRING) == false)
	{
		delete[] buffer_array;
		return false;
	}

	byte_pointer = (unsigned char *)&number_of_jobs;

	byte_pointer[0] = buffer_array[0];
	byte_pointer[1] = buffer_array[1];
	byte_pointer[2] = buffer_array[2];
	byte_pointer[3] = buffer_array[3];

	byte_counter = 4;

	for (job_counter = 0; job_counter < number_of_jobs; job_counter++)
	{
		byte_pointer = (unsigned char *)&job_number;
		byte_pointer[0] = buffer_array[byte_counter];
		byte_counter++;
		byte_pointer[1] = buffer_array[byte_counter];
		byte_counter++;
		byte_pointer[2] = buffer_array[byte_counter];
		byte_counter++;
		byte_pointer[3] = buffer_array[byte_counter];
		byte_counter++;

		byte_pointer = (unsigned char *)&result_size;
		byte_pointer[0] = buffer_array[byte_counter];
		byte_counter++;
		byte_pointer[1] = buffer_array[byte_counter];
		byte_counter++;
		byte_pointer[2] = buffer_array[byte_counter];
		byte_counter++;
		byte_pointer[3] = buffer_array[byte_counter];
		byte_counter++;

		temp_result.job_number = job_number;

		if (temp_result.result_size != result_size)
		{
			if (temp_result.result_size != 0 && temp_result.result_data != NULL)
			{
				delete[] temp_result.result_data;
			}

			temp_result.result_size = result_size;

			if (result_size == 0)
			{
				temp_result.result_data = NULL;
			}
			else
				temp_result.result_data = new float[result_size];
		}

		for (result_byte_counter = 0; result_byte_counter < result_size; result_byte_counter++)
		{
			byte_pointer = (unsigned char *)&temp_result.result_data[result_byte_counter];

			if (byte_counter >= total_number_of_bytes)
				wxPrintf("byte_counter = %i/%i\n", byte_counter, total_number_of_bytes);
			byte_pointer[0] = buffer_array[byte_counter];
			byte_counter++;
			byte_pointer[1] = buffer_array[byte_counter];
			byte_counter++;
			byte_pointer[2] = buffer_array[byte_counter];
			byte_counter++;
			byte_pointer[3] = buffer_array[byte_counter];
			byte_counter++;
		}

		// add it to the array.

		my_array.Add(temp_result);
	}

	delete[] buffer_array;

	return true;
}

bool SendResultQueueToSocket(wxSocketBase *socket, ArrayofJobResults &my_array)
{
	int total_number_of_bytes = 4; // number of results

	int number_of_jobs = my_array.GetCount();

	unsigned char *byte_pointer;
	int byte_counter = 0;
	int job_counter;
	int result_byte_counter;

	//wxPrintf("there are %i jobs\n", number_of_jobs);

	for (job_counter = 0; job_counter < number_of_jobs; job_counter++)
	{
		total_number_of_bytes += 8;											 // job_number, result_size
		total_number_of_bytes += my_array.Item(job_counter).result_size * 4; // actual result
																			 //	wxPrintf("result size for job %i = %i\n", job_counter, my_array.Item(job_counter).result_size);
	}

	//wxPrintf("(Write) Total Size is %i bytes\n", total_number_of_bytes);

	unsigned char *buffer_array = new unsigned char[total_number_of_bytes];

	byte_pointer = (unsigned char *)&number_of_jobs;

	buffer_array[0] = byte_pointer[0];
	buffer_array[1] = byte_pointer[1];
	buffer_array[2] = byte_pointer[2];
	buffer_array[3] = byte_pointer[3];

	byte_counter = 4;

	for (job_counter = 0; job_counter < number_of_jobs; job_counter++)
	{
		byte_pointer = (unsigned char *)&my_array.Item(job_counter).job_number;
		buffer_array[byte_counter] = byte_pointer[0];
		byte_counter++;
		buffer_array[byte_counter] = byte_pointer[1];
		byte_counter++;
		buffer_array[byte_counter] = byte_pointer[2];
		byte_counter++;
		buffer_array[byte_counter] = byte_pointer[3];
		byte_counter++;

		byte_pointer = (unsigned char *)&my_array.Item(job_counter).result_size;
		buffer_array[byte_counter] = byte_pointer[0];
		byte_counter++;
		buffer_array[byte_counter] = byte_pointer[1];
		byte_counter++;
		buffer_array[byte_counter] = byte_pointer[2];
		byte_counter++;
		buffer_array[byte_counter] = byte_pointer[3];
		byte_counter++;

		for (result_byte_counter = 0; result_byte_counter < my_array.Item(job_counter).result_size; result_byte_counter++)
		{
			byte_pointer = (unsigned char *)&my_array.Item(job_counter).result_data[result_byte_counter];
			//	wxPrintf("byte_counter = %i\n", byte_counter);
			buffer_array[byte_counter] = byte_pointer[0];
			byte_counter++;
			buffer_array[byte_counter] = byte_pointer[1];
			byte_counter++;
			buffer_array[byte_counter] = byte_pointer[2];
			byte_counter++;
			buffer_array[byte_counter] = byte_pointer[3];
			byte_counter++;
		}
	}

	// send the number of bytes
	if (WriteToSocket(socket, &total_number_of_bytes, sizeof(int), true, "SendResultQueueTotalBytes", FUNCTION_DETAILS_AS_WXSTRING) == false)
	{
		delete[] buffer_array;
		return false;
	}
	// send the array..

	if (WriteToSocket(socket, buffer_array, total_number_of_bytes, true, "SendResultQueueData", FUNCTION_DETAILS_AS_WXSTRING) == false)
	{
		delete[] buffer_array;
		return false;
	}

	delete[] buffer_array;
	return true;
}
