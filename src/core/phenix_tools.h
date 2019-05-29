// Use one of several Phenix command line tools


class CommandLineTools {

public:

	CommandLineTools();
	~CommandLineTools();
	void SetupLauncher(wxString wanted_program_dir, wxString wanted_working_dir);
	bool Launch();

protected:

	wxString		program_dir;
	wxString		working_dir;

private:

	wxString		args;

};

class PhenixTools : public CommandLineTools {

public:

	PhenixTools();
	~PhenixTools();
	bool Launch();

protected:

	wxString		program_dir;
	wxString		working_dir;

private:

	wxString		args;

};

class FmodelRegrid : public CommandLineTools {

public:

	FmodelRegrid();
	~FmodelRegrid();
	void SetAllUserParameters(wxString wanted_model_path, wxString wanted_mrc_path, wxString wanted_output_path, float wanted_resolution);
	bool RunFmodelRegrid();

protected:

	wxString		model_path;
	wxString		mrc_path;
	wxString		output_path;
	float			resolution;
	wxString		model_basename;
	wxString		model_ext;
	wxString		output_basename;
	wxString		output_ext;
	wxString		fmodel_regrid_args;
	wxString		fmodel_regrid_main;
	wxString		fmodel_regrid_echo;
	int				fmodel_regrid_code;
	wxString		success_message;
	wxString		fmodel_regrid_script;
	wxString		fmodel_regrid_script_path;

};
