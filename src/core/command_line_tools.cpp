#include "core_headers.h"

// AsyncProcess::AsyncProcess()
// {
//     finished = false;
// }

// void AsyncProcess::OnTerminate(int pid, int status)
// {
//     finished = true;
// }

// void AsyncProcess::SetReturnString(wxString wanted_return_string)
// {
//     return_string = wanted_return_string;
// }

CommandLineTools::CommandLineTools()
{
    args = "";
    bin_dir = "";
    executable = "";
    exit_code = 0;
    return_string = "";
}

void CommandLineTools::Init(wxString wanted_bin_dir, wxString wanted_executable)
{
    bin_dir = wanted_bin_dir;
    executable = wanted_executable;
}

CommandLineTools::~CommandLineTools() {}

void CommandLineTools::AddArgument(wxString wanted_argument)
{
    args = wxString::Format("%s %s", args, wanted_argument);
}

wxString CommandLineTools::GetCommand()
{
    return wxString::Format("%s/%s %s", bin_dir, executable, args);
}

wxString CommandLineTools::RunSync()
{
    long exit_code = wxExecute(wxString::Format("%s/%s %s", bin_dir, executable, args), wxEXEC_SYNC, NULL);
	if ( exit_code == 0)
	{
        return_string = wxString::Format("Execution of %s/%s %s completed successfully.\n", bin_dir, executable, args);
	}
	else
	{
		return_string = wxString::Format("Execution of %s/%s %s failed with return code %li.\n", bin_dir, executable, args, exit_code);
	}
    return return_string;
}

wxString CommandLineTools::RunAsync(int wanted_process_id)
{
    wxProcess *process = new wxProcess();
    long exit_code = wxExecute(wxString::Format("%s/%s %s", bin_dir, executable, args), wxEXEC_ASYNC, process);
    // FIXME: this process cannot be killed :(
    while ( wxProcess::Exists(process->GetPid()) )
    {
        wxSleep(10);
    }
    return_string = wxString::Format("Execution of %s/%s %s completed with return code %li.\n", bin_dir, executable, args, exit_code);
    return return_string;
}

// wxString CommandLineTools::RunAsync()
// {
//     long exit_code = wxExecute(wxString::Format("%s/%s %s", bin_dir, executable, args), wxEXEC_ASYNC, process);
//     // TODO: subclass wxProcess and customize the OnTerminate method to do what follows
//     if ( exit_code != 0 )
// 	{
// 		return_string = wxString::Format("Execution of %s/%s %s failed to launch with return code %li.\n", bin_dir, executable, args, exit_code);
// 	}
//     else {
//         while ( process->finished != true )
//         {
//             wxSleep(10);
//         }
//         return_string = wxString::Format("Execution of %s/%s %s completed.\n", bin_dir, executable, args);
//         return return_string;
//     }
// }