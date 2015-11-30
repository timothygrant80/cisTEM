#include "core_headers.h"

void JobPackage::SendJobPackage(wxSocketBase *socket) // package the whole object into a single char stream which can be decoded at the other end..
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

	unsigned char* char_pointer;

	// first work out how long the character array needs to be..

	long transfer_size = ReturnEncodedByteTransferSize();

	// allocate a character array for the jobs..

	unsigned char *transfer_buffer = new unsigned char[transfer_size];

	// fill the buffer, first number of jobs

	 char_pointer = (unsigned char*)&number_of_jobs;

	 transfer_buffer[0] = char_pointer[0];
	 transfer_buffer[1] = char_pointer[1];
	 transfer_buffer[2] = char_pointer[2];
	 transfer_buffer[3] = char_pointer[3];

	 // number_of_run_commands

	 char_pointer = (unsigned char*)& my_profile.number_of_run_commands;

	 transfer_buffer[4] = char_pointer[0];
	 transfer_buffer[5] = char_pointer[1];
	 transfer_buffer[6] = char_pointer[2];
	 transfer_buffer[7] = char_pointer[3];

	 // executable name

	 temp_int = my_profile.executable_name.Length();
	 char_pointer = (unsigned char*)& temp_int;

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
	 // now add each run_command

	 for (command_counter = 0; command_counter < my_profile.number_of_run_commands; command_counter++)
	 {
		 length_of_string = my_profile.run_commands[command_counter].command_to_run.Length();

		 char_pointer = (unsigned char*)&length_of_string;

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
			 transfer_buffer[byte_counter] = static_cast <char> (my_profile.run_commands[command_counter].command_to_run[counter]);
			 byte_counter++;
		 }

		 // number of copies..

		 char_pointer = (unsigned char*) &my_profile.run_commands[command_counter].number_of_copies;

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


	 for (job_counter = 0; job_counter < number_of_jobs; job_counter++)
	 {
		 // job_number for this job

		 char_pointer = (unsigned char*)&jobs[job_counter].job_number;

		 transfer_buffer[byte_counter] = char_pointer[0];
		 byte_counter++;
		 transfer_buffer[byte_counter] = char_pointer[1];
		 byte_counter++;
		 transfer_buffer[byte_counter] = char_pointer[2];
		 byte_counter++;
		 transfer_buffer[byte_counter] = char_pointer[3];
		 byte_counter++;


		 // add number of arguments for this job

		 char_pointer = (unsigned char*)&jobs[job_counter].number_of_arguments;

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
				 char_pointer = (unsigned char*)&temp_int;

				 transfer_buffer[byte_counter] = char_pointer[0];
				 byte_counter++;
				 transfer_buffer[byte_counter] = char_pointer[1];
				 byte_counter++;
				 transfer_buffer[byte_counter] = char_pointer[2];
				 byte_counter++;
				 transfer_buffer[byte_counter] = char_pointer[3];
				 byte_counter++;
			 }
			 else
			 if (jobs[job_counter].arguments[argument_counter].type_of_argument == FLOAT)
			 {
				 // set the descriptor byte

				 transfer_buffer[byte_counter] = FLOAT;
				 byte_counter++;

				 // set the value of the float..

				 temp_float = jobs[job_counter].arguments[argument_counter].ReturnFloatArgument();
				 char_pointer = (unsigned char*)&temp_float;

				 transfer_buffer[byte_counter] = char_pointer[0];
				 byte_counter++;
				 transfer_buffer[byte_counter] = char_pointer[1];
				 byte_counter++;
				 transfer_buffer[byte_counter] = char_pointer[2];
				 byte_counter++;
				 transfer_buffer[byte_counter] = char_pointer[3];
				 byte_counter++;
			 }
			 else
			 if (jobs[job_counter].arguments[argument_counter].type_of_argument == BOOL)
			 {
				 // set the descriptor byte

				 transfer_buffer[byte_counter] = BOOL;
				 byte_counter++;

				 // set the value of the bool..

				 transfer_buffer[byte_counter] = (unsigned char) jobs[job_counter].arguments[argument_counter].ReturnBoolArgument();
				 byte_counter++;
			 }
			 else
			 if (jobs[job_counter].arguments[argument_counter].type_of_argument == TEXT)
			 {
				 // set the descriptor byte

				 transfer_buffer[byte_counter] = TEXT;
 				 byte_counter++;

 				 // add the length of the string..

 				 temp_string = jobs[job_counter].arguments[argument_counter].ReturnStringArgument();
 				 length_of_string = temp_string.length();

				 char_pointer = (unsigned char*)&length_of_string;

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
				 abort();
			 }

		 }

	 }

	 // now we should everything encoded, so send the information to the socket..
	 // disable events on the socket..
	 socket->SetNotify(wxSOCKET_LOST_FLAG);

	 // inform what we want to do..
	 socket->WriteMsg(socket_ready_to_send_job_package, SOCKET_CODE_SIZE);
	 socket->WaitForRead(5);

     if (socket->IsData() == true)
     {

    	 // we should get a message saying the socket is ready to receive the data..

    	 socket->ReadMsg(&socket_input_buffer, SOCKET_CODE_SIZE);

    	 // check it is ok..

    	 if (memcmp(socket_input_buffer, socket_send_job_package, SOCKET_CODE_SIZE) == 0) // send it..
    	 {
    		 // first - send how many bytes it is..

    		 char_pointer = (unsigned char*)&transfer_size;
    		 socket->WriteMsg(char_pointer, 8);

    		 // now send the whole buffer..

    		 MyDebugPrint("doing socket write");
    		 socket->WriteMsg(transfer_buffer, transfer_size);


    	 }
    	 else
    	 {
    		MyPrintWithDetails("Oops, didn't understand the reply!");
    		abort();
    	 }
     }
     else
     {
    		MyPrintWithDetails("Oops, unexpected timeout!");
         		abort();
     }

     // restore socket events..

     socket->SetNotify(wxSOCKET_LOST_FLAG | wxSOCKET_INPUT_FLAG);
	 delete [] transfer_buffer;

}

void JobPackage::ReceiveJobPackage(wxSocketBase *socket)
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
	std::string wanted_command_to_run;

	int length_of_string;
	int number_of_arguments;
	int temp_job_number;
	int temp_int;
	float temp_float;
	std::string temp_string;

	unsigned char* char_pointer;

	RunProfile temp_run_profile;
	wxString temp_wxstring;
	wxString executable_name;



	// disable events on the socket..
	socket->SetNotify(wxSOCKET_LOST_FLAG);

	// Send a message saying we are ready to receive the package

	socket->WriteMsg(socket_send_job_package, SOCKET_CODE_SIZE);

	char_pointer = (unsigned char*)&transfer_size;

	// receive how many bytes we need for the buffer..

	socket->ReadMsg(char_pointer, 8);

	MyDebugPrint("Package is %li bytes long", transfer_size);

	// allocate an array..

	unsigned char *transfer_buffer = new unsigned char[transfer_size];

	// now receive the package..

	socket->ReadMsg(transfer_buffer, transfer_size);

	MyDebugPrint("Received package, decoding job...");

    // restore socket events..
    socket->SetNotify(wxSOCKET_LOST_FLAG | wxSOCKET_INPUT_FLAG);


	// now we need to decode the buffer


    // number of jobs
    char_pointer = (unsigned char*)&wanted_number_of_jobs;
    char_pointer[0] = transfer_buffer[0];
    char_pointer[1] = transfer_buffer[1];
    char_pointer[2] = transfer_buffer[2];
    char_pointer[3] = transfer_buffer[3];

    MyDebugPrint("There are %i jobs", wanted_number_of_jobs);

    // number of run commands
    char_pointer = (unsigned char*)&number_of_run_commands;
    char_pointer[0] = transfer_buffer[4];
    char_pointer[1] = transfer_buffer[5];
    char_pointer[2] = transfer_buffer[6];
    char_pointer[3] = transfer_buffer[7];

    MyDebugPrint("There are %i commands", number_of_run_commands);

    // executable name

   	 char_pointer = (unsigned char*)& temp_int;
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

     // each run command

	for (command_counter = 0; command_counter < number_of_run_commands; command_counter++)
	{
		 char_pointer = (unsigned char*)&length_of_string;

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

		 char_pointer = (unsigned char*)&number_of_copies;

 		 char_pointer[0] = transfer_buffer[byte_counter];
 		 byte_counter++;
 		 char_pointer[1] = transfer_buffer[byte_counter];
 		 byte_counter++;
 		 char_pointer[2] = transfer_buffer[byte_counter];
 		 byte_counter++;
		 char_pointer[3] = transfer_buffer[byte_counter];
 		 byte_counter++;

 		 // add the command.

 		 temp_run_profile.AddCommand(temp_wxstring, number_of_copies);
	 }

	// now we need to loop over all the jobs..

	Reset(temp_run_profile, executable_name, wanted_number_of_jobs);

	for (job_counter = 0; job_counter < number_of_jobs; job_counter++)
	{
		 char_pointer = (unsigned char*)&temp_job_number;

		 char_pointer[0] = transfer_buffer[byte_counter];
		 byte_counter++;
		 char_pointer[1] = transfer_buffer[byte_counter];
		 byte_counter++;
		 char_pointer[2] = transfer_buffer[byte_counter];
		 byte_counter++;
		 char_pointer[3] = transfer_buffer[byte_counter];
		 byte_counter++;

		 // How many arguments are there for this job

		 char_pointer = (unsigned char*)&temp_int;

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

				char_pointer = (unsigned char*)&temp_int;

				char_pointer[0] = transfer_buffer[byte_counter];
				byte_counter++;
				char_pointer[1] = transfer_buffer[byte_counter];
				byte_counter++;
				char_pointer[2] = transfer_buffer[byte_counter];
				byte_counter++;
				char_pointer[3] = transfer_buffer[byte_counter];
				byte_counter++;

				// allocate memory

				jobs[job_counter].arguments[argument_counter].integer_argument = new int;
				jobs[job_counter].arguments[argument_counter].integer_argument[0] = temp_int;
			 }
			 else
			 if (jobs[job_counter].arguments[argument_counter].type_of_argument == FLOAT)
			 {
				 // read the value of the float..

				 char_pointer = (unsigned char*)&temp_float;

				 char_pointer[0] = transfer_buffer[byte_counter];
				 byte_counter++;
				 char_pointer[1] = transfer_buffer[byte_counter];
				 byte_counter++;
				 char_pointer[2] = transfer_buffer[byte_counter];
				 byte_counter++;
				 char_pointer[3] = transfer_buffer[byte_counter];
				 byte_counter++;

				 // allocate memory

				 jobs[job_counter].arguments[argument_counter].float_argument = new float;
				 jobs[job_counter].arguments[argument_counter].float_argument[0] = temp_float;
			 }
			 else
			 if (jobs[job_counter].arguments[argument_counter].type_of_argument == BOOL)
			 {
				 // read the value of the bool..

				 // allocate memory

				 jobs[job_counter].arguments[argument_counter].bool_argument = new bool;

				 jobs[job_counter].arguments[argument_counter].bool_argument[0] = bool(transfer_buffer[byte_counter]);
				 byte_counter++;
			 }
			 else
			 if (jobs[job_counter].arguments[argument_counter].type_of_argument == TEXT)
			 {
				 // read length of command string

				 char_pointer = (unsigned char*)&length_of_string;

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
				 abort();
			 }

			}

		}

	MyDebugPrint("Job Decoded");

	// delete the buffer
	delete [] transfer_buffer;
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

	for (counter = 0; counter < my_profile.number_of_run_commands; counter++)
	{
		byte_size += 4; // length_of_current_command;
		byte_size += my_profile.run_commands[counter].command_to_run.Length(); // actual text;
		byte_size += 4; // number_of_copies;
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
		if (number_of_jobs == 1) jobs = new RunJob;
		else
		jobs = new RunJob[number_of_jobs];
	}

}




JobPackage::~JobPackage()
{
	if (number_of_jobs > 0)
	{
		if (number_of_jobs == 1) delete jobs;
		else delete [] jobs;
	}

}

void JobPackage::Reset(RunProfile wanted_profile, wxString wanted_executable_name,int wanted_number_of_jobs)
{
	if (number_of_jobs > 0)
	{
		if (number_of_jobs == 1) delete jobs;
		else delete [] jobs;
	}

	number_of_jobs = wanted_number_of_jobs;
	number_of_added_jobs = 0;

	// memory allocation..

	if (number_of_jobs == 1) jobs = new RunJob;
	else
	jobs = new RunJob[number_of_jobs];

	my_profile = wanted_profile;
	my_profile.executable_name = wanted_executable_name;

	//my_profile.SubstituteExecutableName(wanted_executable_name);

}

void JobPackage::AddJob(const char *format, ...)
{
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
		if (jobs[counter].has_been_run == false) number_remaining++;
	}

	return number_remaining;

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

void RunJob::SendJob(wxSocketBase *socket)
{
	SETUP_SOCKET_CODES

	long counter;
	long argument_counter;
	long byte_counter = 0;

	int length_of_string;
	int temp_int;
	float temp_float;
	std::string temp_string;

	unsigned char* char_pointer;

	long transfer_size = ReturnEncodedByteTransferSize();

	// allocate a character array for the jobs..

	unsigned char *transfer_buffer = new unsigned char[transfer_size];

	// job_number

	 char_pointer = (unsigned char*)&job_number;

	 transfer_buffer[0] = char_pointer[0];
	 transfer_buffer[1] = char_pointer[1];
	 transfer_buffer[2] = char_pointer[2];
	 transfer_buffer[3] = char_pointer[3];


	// number of arguments..

	 char_pointer = (unsigned char*)&number_of_arguments;

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
			 char_pointer = (unsigned char*)&temp_int;

			 transfer_buffer[byte_counter] = char_pointer[0];
			 byte_counter++;
			 transfer_buffer[byte_counter] = char_pointer[1];
			 byte_counter++;
			 transfer_buffer[byte_counter] = char_pointer[2];
			 byte_counter++;
			 transfer_buffer[byte_counter] = char_pointer[3];
			 byte_counter++;
		 }
		 else
		 if (arguments[argument_counter].type_of_argument == FLOAT)
		 {
			 // set the descriptor byte

			 transfer_buffer[byte_counter] = FLOAT;
			 byte_counter++;

			 // set the value of the float..

			 temp_float = arguments[argument_counter].ReturnFloatArgument();
			 char_pointer = (unsigned char*)&temp_float;

			 transfer_buffer[byte_counter] = char_pointer[0];
			 byte_counter++;
			 transfer_buffer[byte_counter] = char_pointer[1];
			 byte_counter++;
			 transfer_buffer[byte_counter] = char_pointer[2];
			 byte_counter++;
			 transfer_buffer[byte_counter] = char_pointer[3];
			 byte_counter++;
		 }
		 else
		 if (arguments[argument_counter].type_of_argument == BOOL)
		 {
			 // set the descriptor byte

			 transfer_buffer[byte_counter] = BOOL;
			 byte_counter++;

			 // set the value of the bool..

			 transfer_buffer[byte_counter] = (unsigned char) arguments[argument_counter].ReturnBoolArgument();
			 byte_counter++;
		 }
		 else
		 if (arguments[argument_counter].type_of_argument == TEXT)
		 {
			 // set the descriptor byte

			 transfer_buffer[byte_counter] = TEXT;
			 byte_counter++;

			 // add the length of the string..

			 temp_string = arguments[argument_counter].ReturnStringArgument();
			 length_of_string = temp_string.length();

			 char_pointer = (unsigned char*)&length_of_string;

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
			 abort();
		 }

	 }

	 // now we should everything encoded, so send the information to the socket..
	 // disable events on the socket..

	 socket->SetNotify(wxSOCKET_LOST_FLAG);

	 // inform what we want to do..
	 socket->WriteMsg(socket_ready_to_send_single_job, SOCKET_CODE_SIZE);
	 socket->WaitForRead(5);

    if (socket->IsData() == true)
    {

    	// we should get a message saying the socket is ready to receive the data..

    	socket->ReadMsg(&socket_input_buffer, SOCKET_CODE_SIZE);

    	// check it is ok..

    	if (memcmp(socket_input_buffer, socket_send_single_job, SOCKET_CODE_SIZE) == 0) // send it..
    	{
    		// first - send how many bytes it is..

    		char_pointer = (unsigned char*)&transfer_size;
    		socket->WriteMsg(char_pointer, 8);

    		// now send the whole buffer..

    		socket->WriteMsg(transfer_buffer, transfer_size);

    	}
    	else
    	{
    		MyPrintWithDetails("Oops, didn't understand the reply!");
    		abort();
    	}
    }
    else
    {
    		MyPrintWithDetails("Oops, unexpected timeout!");
        	abort();
    }

    // restore socket events..

    socket->SetNotify(wxSOCKET_LOST_FLAG | wxSOCKET_INPUT_FLAG);
	delete [] transfer_buffer;



}

void RunJob::RecieveJob(wxSocketBase *socket)
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

	unsigned char* char_pointer;

	// disable events on the socket..
	socket->SetNotify(wxSOCKET_LOST_FLAG);

	// Send a message saying we are ready to receive the package

	socket->WriteMsg(socket_send_single_job, SOCKET_CODE_SIZE);

	char_pointer = (unsigned char*)&transfer_size;

	// receive how many bytes we need for the buffer..

	socket->ReadMsg(char_pointer, 8);

	// allocate an array..

	unsigned char *transfer_buffer = new unsigned char[transfer_size];

	// now receive the package..

	socket->ReadMsg(transfer_buffer, transfer_size);

    // restore socket events..
    socket->SetNotify(wxSOCKET_LOST_FLAG | wxSOCKET_INPUT_FLAG);

	// now we need to decode the buffer

    byte_counter = 0;

    // job id

	char_pointer = (unsigned char*)&temp_job_number;
	char_pointer[0] = transfer_buffer[byte_counter];
	byte_counter++;
	char_pointer[1] = transfer_buffer[byte_counter];
	byte_counter++;
	char_pointer[2] = transfer_buffer[byte_counter];
	byte_counter++;
	char_pointer[3] = transfer_buffer[byte_counter];
	byte_counter++;


    // How many arguments are there for this job

	char_pointer = (unsigned char*)&number_of_arguments;
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
			char_pointer = (unsigned char*)&temp_int;

			char_pointer[0] = transfer_buffer[byte_counter];
			byte_counter++;
			char_pointer[1] = transfer_buffer[byte_counter];
			byte_counter++;
			char_pointer[2] = transfer_buffer[byte_counter];
			byte_counter++;
			char_pointer[3] = transfer_buffer[byte_counter];
			byte_counter++;

			// allocate memory

			arguments[argument_counter].integer_argument = new int;
			arguments[argument_counter].integer_argument[0] = temp_int;
		 }
		 else
		 if (arguments[argument_counter].type_of_argument == FLOAT)
		 {
			 // read the value of the float..
			 char_pointer = (unsigned char*)&temp_float;

			 char_pointer[0] = transfer_buffer[byte_counter];
			 byte_counter++;
			 char_pointer[1] = transfer_buffer[byte_counter];
			 byte_counter++;
			 char_pointer[2] = transfer_buffer[byte_counter];
			 byte_counter++;
			 char_pointer[3] = transfer_buffer[byte_counter];
			 byte_counter++;

			 // allocate memory

			 arguments[argument_counter].float_argument = new float;
			 arguments[argument_counter].float_argument[0] = temp_float;
		 }
		 else
		 if (arguments[argument_counter].type_of_argument == BOOL)
		 {
			 // read the value of the bool..

			 // allocate memory

			 arguments[argument_counter].bool_argument = new bool;
			 arguments[argument_counter].bool_argument[0] = bool(transfer_buffer[byte_counter]);
			 byte_counter++;
		 }
		 else
		 if (arguments[argument_counter].type_of_argument == TEXT)
		 {
			 // read length of command string

			 char_pointer = (unsigned char*)&length_of_string;

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
			 abort();
		 }
	}

	// delete the buffer
	delete [] transfer_buffer;

}

void RunJob::Deallocate()
{
	if (number_of_arguments > 0)
	{
		if (number_of_arguments == 1) delete arguments;
		else
		delete [] arguments;
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

	if (number_of_arguments == 1) arguments = new RunArgument;
	else arguments = new RunArgument[number_of_arguments];

	// fill the arguments..

	long counter = 0;


    while (*format!= '\0')
    {
        if (*format == 't' || *format == 's') // argument is text..
        {
            arguments[counter].SetStringArgument(va_arg(args, const char *));
        }
        else if (*format == 'f') // float
        {
        	arguments[counter].SetFloatArgument(va_arg(args, double));
        }
        else
        if (*format == 'i') // integer
        {
        	arguments[counter].SetIntArgument(va_arg(args, int));
        }
        else
        if (*format == 'b') // bool
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

	if (number_of_arguments == 1) arguments = new RunArgument;
	else arguments = new RunArgument[number_of_arguments];

	job_number = -1;
	has_been_run = false;


}

void RunJob::PrintAllArguments()
{
	wxPrintf("\n\n");

	for (int counter = 0; counter < number_of_arguments; counter++)
	{
		if (arguments[counter].type_of_argument == TEXT)
		{
			wxPrintf("Argument %3i is a string   : %s\n", counter, arguments[counter].ReturnStringArgument());
		}
		else
		if (arguments[counter].type_of_argument == INTEGER)
		{
			wxPrintf("Argument %3i is an integer : %i\n", counter, arguments[counter].ReturnIntegerArgument());
		}
		else
		if (arguments[counter].type_of_argument == FLOAT)
		{
			wxPrintf("Argument %3i is a float    : %f\n", counter, arguments[counter].ReturnFloatArgument());
		}
		else
		if (arguments[counter].type_of_argument == BOOL)
		{
			wxPrintf("Argument %3i is a bool     : ", counter);

			if (arguments[counter].ReturnBoolArgument() == true) wxPrintf("TRUE\n");
			else wxPrintf("FALSE\n");
		}
	}
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
	if (is_allocated == true) Deallocate();
}

void RunArgument::Deallocate()
{
	if (type_of_argument == TEXT) delete string_argument;
	else
	if (type_of_argument == INTEGER) delete integer_argument;
	else
	if (type_of_argument == FLOAT) delete float_argument;
	else
	if (type_of_argument == BOOL) delete bool_argument;

	is_allocated = false;
}

long RunArgument::ReturnEncodedByteTransferSize()
{
	MyDebugAssertTrue(type_of_argument != NONE, "Can't calculate size of a nothing argument!!");

	if (type_of_argument == TEXT) return string_argument->length() + 4 + 1; // descriptor byte + length of string (4 bytes) + 1 byte per character
	else
	if (type_of_argument == BOOL) return 2; // descriptor bytes + bool bytes
	else
	return 5; // descriptor byte + 4 data bytes
}

void RunArgument::SetStringArgument(const char *wanted_text)
{
	if (is_allocated == true) Deallocate();

	type_of_argument = TEXT;
	string_argument = new std::string;
	string_argument[0] = wanted_text;
	is_allocated = true;
}

void RunArgument::SetIntArgument(int wanted_argument)
{
	if (is_allocated == true) Deallocate();

	type_of_argument = INTEGER;
	integer_argument = new int;
	integer_argument[0] = wanted_argument;
	is_allocated = true;
}

void RunArgument::SetFloatArgument(float wanted_argument)
{
	if (is_allocated == true) Deallocate();

	type_of_argument = FLOAT;
	float_argument = new float;
	float_argument[0] = wanted_argument;
	is_allocated = true;
}

void RunArgument::SetBoolArgument(bool wanted_argument)
{
	if (is_allocated == true) Deallocate();

	type_of_argument = BOOL;
	bool_argument = new bool;
	bool_argument[0] = wanted_argument;
	is_allocated = true;
}





