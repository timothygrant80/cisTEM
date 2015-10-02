#include "core_headers.h"

RunCommand::RunCommand()
{
	command_to_run = "";
	number_of_copies = 0;
}

RunCommand::~RunCommand()
{

}

void RunCommand::SetCommand(wxString wanted_command, int wanted_number_of_copies)
{
	command_to_run = wanted_command;
	number_of_copies = wanted_number_of_copies;
}

RunProfile::RunProfile()
{
	id = 1;
	number_of_run_commands = 0;

	manager_command = "$command";
	run_commands = new RunCommand[5];
	number_allocated = 5;

}

RunProfile::RunProfile( const RunProfile &obj) // copy contructor
{
	id = obj.id;
	number_of_run_commands = obj.number_allocated;
	manager_command = obj.manager_command;
	number_allocated = obj.number_allocated;

	run_commands = new RunCommand[number_allocated];

	for (long counter = 0; counter < number_of_run_commands; counter++)
	{
	   run_commands[counter] = obj.run_commands[counter];
	}
}

RunProfile::~RunProfile()
{
	delete [] run_commands;
}

void RunProfile::AddCommand(RunCommand wanted_command)
{
	// check we have enough memory

	RunCommand *buffer;

	if (number_of_run_commands >= number_allocated)
	{
		// reallocate..

		if (number_allocated < 100) number_allocated *= 2;
		else number_allocated += 100;

		buffer = new RunCommand[number_allocated];

		for (long counter = 0; counter < number_of_run_commands; counter++)
		{
			buffer[counter] = run_commands[counter];
		}

		delete [] run_commands;
		run_commands = buffer;
	}

	// Should be fine for memory, so just add one.

	run_commands[number_of_run_commands] = wanted_command;
	number_of_run_commands++;
}

void RunProfile::AddCommand(wxString wanted_command, int wanted_number_of_copies)
{
	// check we have enough memory

	RunCommand *buffer;

	if (number_of_run_commands >= number_allocated)
	{
		// reallocate..

		if (number_allocated < 100) number_allocated *= 2;
		else number_allocated += 100;

		buffer = new RunCommand[number_allocated];

		for (long counter = 0; counter < number_of_run_commands; counter++)
		{
			buffer[counter] = run_commands[counter];
		}

		delete [] run_commands;
		run_commands = buffer;
	}

	// Should be fine for memory, so just add one.

	run_commands[number_of_run_commands].command_to_run = wanted_command;
	run_commands[number_of_run_commands].number_of_copies = wanted_number_of_copies;
	number_of_run_commands++;
}

void RunProfile::RemoveCommand(int number_to_remove)
{
	MyDebugAssertTrue(number_to_remove >= 0 && number_to_remove < number_of_run_commands, "Error: Trying to remove a command that doesnt't exist");

	for (long counter = number_to_remove; counter < number_of_run_commands - 1; counter++)
	{
		run_commands[counter] = run_commands[counter + 1];
	}

	number_of_run_commands--;
}

RunProfile & RunProfile::operator = (const RunProfile &t)
{
   // Check for self assignment
   if(this != &t)
   {
	   if(this->number_allocated != t.number_allocated)
	   {
		   delete [] this->run_commands;

		   this->run_commands = new RunCommand[t.number_allocated];
		   this->number_allocated = t.number_allocated;
	   }

	   this->id = t.id;
	   this->name = t.name;
	   this->number_of_run_commands = t.number_of_run_commands;
	   this->manager_command = t.manager_command;

	   for (long counter = 0; counter < t.number_of_run_commands; counter++)
	   {
		   this->run_commands[counter] = t.run_commands[counter];
	   }
   }

   return *this;
}

RunProfile & RunProfile::operator = (const RunProfile *t)
{
   // Check for self assignment
   if(this != t)
   {
	   if(this->number_allocated != t->number_allocated)
	   {
		   delete [] this->run_commands;

		   this->run_commands = new RunCommand[t->number_allocated];
		   this->number_allocated = t->number_allocated;
	   }

	   this->id = t->id;
	   this->name = t->name;
	   this->number_of_run_commands = t->number_of_run_commands;
	   this->manager_command = t->manager_command;

	   for (long counter = 0; counter < t->number_of_run_commands; counter++)
	   {
		   this->run_commands[counter] = t->run_commands[counter];
	   }
   }

   return *this;
}


RunProfileManager::RunProfileManager()
{
	current_id_number = 0;
	number_of_run_profiles = 0;
	run_profiles = new RunProfile[5];
	number_allocated = 5;
}

RunProfileManager::~RunProfileManager()
{
	delete [] run_profiles;
}

void RunProfileManager::AddProfile(RunProfile *profile_to_add)
{
	// check we have enough memory

	RunProfile *buffer;

	if (number_of_run_profiles >= number_allocated)
	{
		// reallocate..

		if (number_allocated < 100) number_allocated *= 2;
		else number_allocated += 100;

		buffer = new RunProfile[number_allocated];

		for (long counter = 0; counter < number_of_run_profiles; counter++)
		{
			buffer[counter] = run_profiles[counter];
		}

		delete [] run_profiles;
		run_profiles = buffer;
	}

	// Should be fine for memory, so just add one.

	run_profiles[number_of_run_profiles] = profile_to_add;
	number_of_run_profiles++;

	if (profile_to_add->id > current_id_number) current_id_number = profile_to_add->id;
}

void RunProfileManager::AddBlankProfile()
{
	// check we have enough memory

	RunProfile *buffer;

	if (number_of_run_profiles >= number_allocated)
	{
		// reallocate..

		if (number_allocated < 100) number_allocated *= 2;
		else number_allocated += 100;

		buffer = new RunProfile[number_allocated];

		for (long counter = 0; counter < number_of_run_profiles; counter++)
		{
			buffer[counter] = run_profiles[counter];
		}

		delete [] run_profiles;
		run_profiles = buffer;
	}

	current_id_number++;
	run_profiles[number_of_run_profiles].id = current_id_number;
	run_profiles[number_of_run_profiles].name = "New Profile";
	run_profiles[number_of_run_profiles].number_of_run_commands = 0;
	run_profiles[number_of_run_profiles].manager_command = "$command";

	number_of_run_profiles++;

}

RunProfile * RunProfileManager::ReturnLastProfilePointer()
{
	return &run_profiles[number_of_run_profiles - 1];
}


RunProfile * RunProfileManager::ReturnProfilePointer(int wanted_profile)
{
	return &run_profiles[wanted_profile];
}

void RunProfileManager::RemoveProfile(int number_to_remove)
{
	MyDebugAssertTrue(number_to_remove >= 0 && number_to_remove < number_of_run_profiles, "Error: Trying to remove a profile that doesnt't exist");

	for (long counter = number_to_remove; counter < number_of_run_profiles - 1; counter++)
	{
		run_profiles[counter] = run_profiles[counter + 1];
	}

	number_of_run_profiles--;

}

void RunProfileManager::RemoveAllProfiles()
{
	number_of_run_profiles = 0;

	if (number_allocated > 100)
	{
		delete [] run_profiles;
		number_allocated = 100;
		run_profiles = new RunProfile[number_allocated];
	}

}


