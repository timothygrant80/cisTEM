#include "core_headers.h"

CommandLineTools::CommandLineTools()
{
    args = "";
    bin_dir = "";
    executable = "";
    exit_code = 0;
    return_string = "";
    error_string = "";
    output_string = "";
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

wxArrayString CommandLineTools::RunAsync()
{
    wxProcess *process = new wxProcess();
    process->Redirect();
    long exit_code = wxExecute(wxString::Format("%s/%s %s", bin_dir, executable, args), wxEXEC_ASYNC, process);

    wxInputStream* output_stream = process->GetInputStream();
    wxInputStream* error_stream = process->GetErrorStream();
    wxTextInputStream text_output_stream { *output_stream };
    wxTextInputStream text_error_stream { *error_stream };

    wxArrayString output_strings;
    wxArrayString error_strings;
    for ( int i=0; i<10; i++ )
    {
        output_strings.Add("");
    }

    // FIXME: this process cannot be killed :(
    while ( wxProcess::Exists(process->GetPid()) )
    {
        if ( output_stream->CanRead() )
        {
            output_strings.Add(text_output_stream.ReadLine());
        }
        if (error_stream->CanRead() )
        {
            error_strings.Add(text_error_stream.ReadLine());
        }
    }
    for ( int i=0; i<output_strings.GetCount(); i++ )
    {
        output_string = output_string + wxString("\n") + output_strings.Item(i);
    }
    for ( int i=0; i<error_strings.GetCount(); i++ )
    {
        error_string = error_string + wxString("\n") + error_strings.Item(i);
    }
    // wxString output_tail = "";
    // for ( int i=0; i<10; i++ )
    // {
    //     output_tail = output_tail + wxString("\n") + output_strings.Item(output_strings.GetCount() - 10 + i);
    // }
    // return_string = wxString::Format("Output ends with:\n%s", output_tail);
    return_array_string.Add(output_string);
    return_array_string.Add(error_string);
    return return_array_string;
}
