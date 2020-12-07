#include "core_headers.h"
#include <wx/arrimpl.cpp> // this is a magic incantation which must be done!
WX_DEFINE_OBJARRAY(ArrayOfLongs);

CommandLineTools::CommandLineTools()
{
    args = "";
    bin_dir = "";
    executable = "";
    exit_code = 0;
    outfile = "";
    errfile = "";
    return_string = "";
    error_string = "";
    output_string = "";
}

void CommandLineTools::Init(wxString wanted_bin_dir, wxString wanted_executable, wxString wanted_outfile="", wxString wanted_errfile="")
{
    bin_dir = wanted_bin_dir;
    executable = wanted_executable;
    outfile = wanted_outfile;
    errfile = wanted_errfile;
}

CommandLineTools::~CommandLineTools() {}

void CommandLineTools::AddArgument(wxString wanted_argument)
{
    args = wxString::Format("%s %s", args, wanted_argument);
}

wxString CommandLineTools::GetCommand()
{
    if ( bin_dir == wxString("") )
    {
        return wxString::Format("%s %s", executable, args);
    }
    else
    {
        return wxString::Format("%s/%s %s", bin_dir, executable, args);
    }
}

wxString CommandLineTools::RunSync()
{
    long exit_code = wxExecute(GetCommand(), wxEXEC_SYNC, NULL);
	if ( exit_code == 0)
	{
        return_string = wxString::Format("Execution of %s completed successfully.\n", GetCommand());
	}
	else
	{
		return_string = wxString::Format("Execution of %s failed with return code %li.\n", GetCommand(), exit_code);
	}
    return return_string;
}

wxArrayString CommandLineTools::RunAsync()
{
    wxProcess *process = new wxProcess();
    process->Redirect();
    long exit_code = wxExecute(GetCommand(), wxEXEC_ASYNC, process);
    // return return_array_string; // this saves us from commenting out everything involving redirected output below when commenting out the redirect cmd above

    wxInputStream* output_stream = process->GetInputStream();
    wxInputStream* error_stream = process->GetErrorStream();
    wxTextInputStream text_output_stream { *output_stream };
    wxTextInputStream text_error_stream { *error_stream };
    wxTextFile outfile_obj;
    wxTextFile errfile_obj;

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

    outfile_obj.Create(outfile);
    errfile_obj.Create(errfile);

    for ( int i=0; i<output_strings.GetCount(); i++ )
    {
        output_string = output_string + wxString("\n") + output_strings.Item(i);
        outfile_obj.AddLine(output_strings.Item(i));
    }
    for ( int i=0; i<error_strings.GetCount(); i++ )
    {
        error_string = error_string + wxString("\n") + error_strings.Item(i);
        errfile_obj.AddLine(error_strings.Item(i));
    }

    outfile_obj.Write();
    errfile_obj.Write();
    outfile_obj.Close();
    errfile_obj.Close();

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

WriteAndRunScript::WriteAndRunScript()
{
    shell = wxString("");
    script = wxString("");
    filename = wxString("");
    output = wxString("");
    error = wxString("");
}

WriteAndRunScript::~WriteAndRunScript() {}

void WriteAndRunScript::Init(wxString wanted_shell, wxString wanted_script, wxString wanted_filename)
{
    shell = wanted_shell;
    script = wanted_script;
    filename = wanted_filename;
}

bool WriteAndRunScript::Write()
{
    bool exit_code;
    wxTextFile *script_file = new wxTextFile(filename.GetFullPath());
    script_file->AddLine(wxString::Format("#! %s\n\n", shell));
    script_file->AddLine(script);
    exit_code = script_file->Write();
    exit_code = exit_code & script_file->Close();
    filename.SetPermissions(wxPOSIX_USER_READ | wxPOSIX_USER_WRITE | wxPOSIX_USER_EXECUTE | wxPOSIX_GROUP_READ | wxPOSIX_OTHERS_READ);
    delete script_file;
    return exit_code;
}

bool WriteAndRunScript::Run()
{
    bool exit_code = Write();
    CommandLineTools job;
    if ( exit_code != true )
    {
        error = wxString("Preparing script failed.");
        return false;
    }
    else
    {
        job.Init(wxString(""), wxString::Format("./%s", filename.GetFullPath()), wxString("test.out"), wxString("test.err")); //FIXME hard-coded output and error filenames
        // wxString cmd = job.GetCommand();
        // // wxPrintf(cmd);
        // const char *cmd_char = cmd.c_str();
        // system(cmd_char);
        // return true;
        wxPrintf("run async is about to hang\n");
        // wxString results = job.RunSync();
        wxArrayString results = job.RunAsync();
        error = results[0];
        output = results[1];
        wxPrintf(wxString::Format("error: %s", error));
        wxPrintf(wxString::Format("output: %s", output));
        return true;
    }
}

//     long exit_code_num = wxExecute(wxString::Format("%s %s", shell, filename.GetFullPath()), wxEXEC_ASYNC, NULL);
//     exit_code = exit_code & (exit_code_num > -1);
//     return exit_code;
// }

DenmodJob::DenmodJob()
{
   original_map_filename = wxString("");
   half_map_1_filename = wxString("");
   half_map_2_filename = wxString("");
   working_dir = wxString("");
   denmod_map_filename = wxString("");
}

DenmodJob::~DenmodJob() {}

void DenmodJob::Init(wxString wanted_bin_dir, wxString wanted_executable, wxString wanted_outfile="", wxString wanted_errfile="", wxString wanted_original_map_filename, wxString wanted_half_map_1_filename, wxString wanted_half_map_2_filename, wxString wanted_working_dir, wxString wanted_denmod_map_filename)
{
    bin_dir = wanted_bin_dir;
    executable = wanted_executable;
    outfile = wanted_outfile;
    errfile = wanted_errfile;
    original_map_filename = wanted_original_map_filename;
    half_map_1_filename = wanted_half_map_1_filename;
    half_map_2_filename = wanted_half_map_2_filename;
    working_dir = wanted_working_dir;
    denmod_map_filename = wanted_denmod_map_filename;
    AddArgument(wxString::Format("map_file_name=%s", original_map_filename));
    AddArgument(wxString::Format("half_map_file_name_1=%s", half_map_1_filename));
    AddArgument(wxString::Format("half_map_file_name_2=%s", half_map_2_filename));
    AddArgument(wxString::Format("temp_dir=%s", working_dir));
    AddArgument(wxString::Format("output_directory=%s", working_dir));
    AddArgument(wxString::Format("output_files.denmod_map_file_name=%s", denmod_map_filename));
}

DenmodJobWrapup::DenmodJobWrapup()
{
    output_path = wxString("");
    reconstructed_volume_path = wxString("");
    denmod_volume_path = wxString("");
}

DenmodJobWrapup::~DenmodJobWrapup() {}

void DenmodJobWrapup::Init(wxString wanted_output_path, wxString wanted_reconstructed_volume_path, wxString wanted_denmod_volume_path)
{
    output_path = wanted_output_path;
    reconstructed_volume_path = wanted_reconstructed_volume_path;
    denmod_volume_path = wanted_denmod_volume_path;
    grep_script = wxString::Format("grep \"Lower bounds of cut out box\" %s | sed \"s/,//g; s/\\[//g; s/\\]//g\" | awk '{ print $7,$8,$9,$12,$13,$14 }'", output_path);
}

ArrayOfLongs DenmodJobWrapup::GetBounds()
{
    WriteAndRunScript grep_run;
    grep_run.Init(wxString("/usr/bin/bash"), grep_script, wxString("grep_bounds.sh"));
    grep_run.Run();
    wxPrintf("DEBUG: SUCCESS");
    wxPrintf(grep_run.output);
    wxArrayString output_bounds_as_strings = wxSplit(grep_run.output, ' ');
    ArrayOfLongs bounds;
    long bound_long;
    for ( int i=0; i<6; i++ )
    {
        wxPrintf(output_bounds_as_strings[i]);
        output_bounds_as_strings[i].ToLong(&bound_long);
        bounds.Add(bound_long);
    }
    return bounds;
}

void DenmodJobWrapup::Run()
{
    // read in the images and prepare denmod file as empty MRCFile for writing back out
    ImageFile reconstructed_volume_file;
    Image reconstructed_volume;
    reconstructed_volume_file.OpenFile(reconstructed_volume_path.ToStdString(), false);
    reconstructed_volume.ReadSlices(&reconstructed_volume_file, 1, reconstructed_volume_file.ReturnNumberOfSlices());
    // MyDebugAssertTrue(reconstructed_volume.IsCubic());
    long full_box_radius = reconstructed_volume_file.ReturnNumberOfSlices() / 2;
    reconstructed_volume_file.CloseFile();

    ImageFile denmod_volume_file;
    Image denmod_volume;
    denmod_volume_file.OpenFile(denmod_volume_path.ToStdString(), false);
    denmod_volume.ReadSlices(&denmod_volume_file, 1, denmod_volume_file.ReturnNumberOfSlices());
    // MyDebugAssertTrue(denmod_volume.IsCubic());
    denmod_volume_file.CloseFile();
    // wxRemoveFile(denmod_volume_path);

    MRCFile denmod_with_background_volume_file;
    denmod_with_background_volume_file.OpenFile(denmod_volume_path.ToStdString(), true);

    // use the image method to replace the background region of the image produced with the original image (scaled)
    ArrayOfLongs bounds = GetBounds(); //FIXME initializing this wrong somehow
    long xmin, ymin, zmin, xmax, ymax, zmax = bounds[0,1,2,3,4,5];
    long shortest_distance_from_center = std::min({(full_box_radius - xmin),
                                                   (xmax - full_box_radius),
                                                   (full_box_radius - ymin),
                                                   (ymax - full_box_radius),
                                                   (full_box_radius - zmin),
                                                   (zmax - full_box_radius)});
    long radius = shortest_distance_from_center + 25; // adding half the box_cushion value to put us comfortably beyond the particle radius
    denmod_volume.ReplaceBorderOfOneImageWithOtherImage(&reconstructed_volume, xmin, ymin, zmin, xmax, ymax, zmax, radius);

    // write out the modified image
    denmod_volume.WriteSlices(&denmod_with_background_volume_file, 1, denmod_volume.logical_z_dimension);
    denmod_with_background_volume_file.CloseFile();

    // Replace existing final volume with density modified volume
    wxCopyFile(denmod_volume_path, reconstructed_volume_path, true);
}
