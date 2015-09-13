#include "core_headers.h"

Project::Project()
{


	is_open = false;
	total_cpu_hours = 0;
	total_jobs_run = 0;

	project_name = "";
	project_directory = "";
}

Project::~Project()
{

}

bool Project::CreateNewProject(wxFileName wanted_database_file, wxString wanted_project_directory, wxString wanted_project_name)
{
	int return_code;


	// is project already open?

	if (is_open == true)
	{
		MyPrintWithDetails("Attempting to create a new project, but there is already an open project");
		return false;
	}


	database.CreateNewDatabase(wanted_database_file);
	database.CreateAllTables();

	if (wanted_project_name.IsEmpty() == true)
	{
		MyDebugPrintWithDetails("Attempting to create a new project, but the project name is blank");
		return false;
	}

	project_name = wanted_project_name;

	if (wanted_project_directory.IsEmpty() == true)
	{
		MyDebugPrintWithDetails("Attempting to create a new project, but the project dir is blank");
		return false;
	}

	project_directory = wanted_project_directory;

	total_cpu_hours = 0;
	total_jobs_run = 0;

	// set master settings..

	if (database.InsertOrReplace("MASTER_SETTINGS", "ittiri", "NUMBER", "PROJECT_DIRECTORY", "PROJECT_NAME", "CURRENT_VERSION", "TOTAL_CPU_HOURS", "TOTAL_JOBS_RUN", 1, project_directory.GetFullPath().ToUTF8().data(), project_name.ToUTF8().data(), INTEGER_DATABASE_VERSION, total_cpu_hours, total_jobs_run) == false) return false;

	is_open = true;

	return true;
}

bool Project::OpenProjectFromFile(wxFileName file_to_open)
{
	bool success;

	// is project already open?

	if (is_open == true)
	{
		MyPrintWithDetails("Attempting to create a new project, but there is already an open project");
		return false;
	}

	success = database.Open(file_to_open);
	success = ReadMasterSettings();

	is_open = true;

	return success;

}

bool Project::ReadMasterSettings()
{
	bool success;

	int imported_integer_version;

	//MyDebugAssertTrue(is_open == true, "Project not open!");

	success = database.GetMasterSettings(project_directory, project_name, imported_integer_version, total_cpu_hours, total_jobs_run);

	if (success == true)
	{
		MyDebugAssertTrue(imported_integer_version == INTEGER_DATABASE_VERSION, "Database version numbers are different!");
	}

	return success;
}

void Project::Close()
{
	database.Close();

	is_open = false;
	total_cpu_hours = 0;
	total_jobs_run = 0;

	project_name = "";
	project_directory = "";

}


