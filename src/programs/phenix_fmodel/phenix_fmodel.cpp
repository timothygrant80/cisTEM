#include "../../core/core_headers.h"

class
PhenixFmodelApp : public MyApp
{

	public:

	bool DoCalculation();
	void DoInteractiveUserInput();


	private:

};

IMPLEMENT_APP(PhenixFmodelApp)

// override the DoInteractiveUserInput

void PhenixFmodelApp::DoInteractiveUserInput()
{
	std::string phenix_installation;
	std::string pdb_path;
	std::string output_path;
	float resolution;

	UserInput *my_input = new UserInput("phenix.fmodel", 1.0);

	phenix_installation = my_input->GetDirnameFromUser("Installation of Phenix", "Location of a Phenix installation", "", true);
	pdb_path = my_input->GetFilenameFromUser("Input pdb", "The input pdb absolute path", "input.pdb", true);
	output_path = my_input->GetFilenameFromUser("Output ccp4", "The output map absolute path", "output.ccp4", false);
	resolution = my_input->GetFloatFromUser("Desired resolution(A)", "Resolution limit of the calculated structure factors", "", 0.0);

	delete my_input;

	my_current_job.Reset(10);
	my_current_job.ManualSetArguments("tttf",       phenix_installation.c_str(),
													pdb_path.c_str(),
													output_path.c_str(),
													resolution
													);
}

// override the do calculation method which will be what is actually run..

bool PhenixFmodelApp::DoCalculation()
{
	// get the arguments for this job..

	std::string		phenix_installation					= my_current_job.arguments[0].ReturnStringArgument();
	std::string		pdb_path 							= my_current_job.arguments[1].ReturnStringArgument();
	std::string		output_path 						= my_current_job.arguments[2].ReturnStringArgument();
	float      		resolution					        = my_current_job.arguments[3].ReturnFloatArgument();

	wxString		current_dir;
	wxString		working_dir;
	wxString		output_basename;
	wxString		output_ext;
	wxTextFile		pdb_file;
	wxTextFile		intermed_file;
	wxString		record;
	wxString		sym_match;
	std::string		record_tmp;
	std::string		intermed_basename;
	std::string		intermed_mtz;
	std::string		fmodel_args;
	std::string		fmodel_main;
	std::string		fmodel_echo;
	int				fmodel_code;
	std::string		mtz2map_args;
	std::string		mtz2map_main;
	std::string		mtz2map_echo;
	int				mtz2map_code;

	current_dir = wxGetCwd();
	wxFileName::SplitPath(output_path, &working_dir, &output_basename, &output_ext, wxPATH_NATIVE);
	intermed_basename = output_basename.ToStdString();
	intermed_mtz = intermed_basename + ".mtz";
	if (working_dir != "")
	{
		wxFileName::SetCwd(wxString(working_dir));
	};

	// construct the command to run phenix.fmodel and execute with wxwidgets

	// workaround: Phenix will misinterpret this argument as reflection file data if the file exists
	if (wxFileName::FileExists(intermed_mtz) == true)
	{
		wxRemoveFile(intermed_mtz);
	}
	fmodel_args = pdb_path + " scattering_table=electron generate_fake_p1_symmetry=True high_resolution=" + std::to_string(resolution) + " output.file_name=" + intermed_mtz;
	wxPrintf("\nLaunching phenix.fmodel calculation...\n");
	fmodel_main = phenix_installation + "/build/bin/phenix.fmodel " + fmodel_args;
	fmodel_echo = "\nExecuting command \"" + fmodel_main + "\"\n";
	wxPrintf(wxString(fmodel_echo));
	fmodel_code = wxExecute(wxString(fmodel_main), wxEXEC_SYNC, NULL);
	if ( fmodel_code != 0 )
	{
		wxLogError(wxString("Execution failed; exiting.\n\n"));
		return false;
	}

	// convert to a ccp4 map with phenix.mtz2map
	mtz2map_main = phenix_installation + "/build/bin/phenix.mtz2map " + intermed_mtz + " pdb_file=" + pdb_path + " include_fmodel=True output.prefix=" + intermed_basename + " extension=ccp4";
	mtz2map_echo = "\nExecuting command \"" + mtz2map_main + "\"\n";
	wxPrintf(wxString(mtz2map_echo));
	mtz2map_code = wxExecute(wxString(mtz2map_main), wxEXEC_SYNC, NULL);
	if ( mtz2map_code != 0 )
	{
		wxLogError(wxString("Execution failed; exiting.\n\n"));
		return false;
	}
	wxRenameFile(intermed_basename + "_fmodel.ccp4", output_basename + ".ccp4");

	// print a success message
	std::string success_message = "\nWrote " + output_path + "\n";
	wxPrintf(wxString(success_message));
	wxFileName::SetCwd(current_dir);

	return true;
}




