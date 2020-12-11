// class AsyncProcess : public wxProcess {

// public :
//     bool finished;
//     // void SetReturnString(wxString wanted_return_string);

//     // constructors & deconstructors
//     AsyncProcess();
//     ~AsyncProcess();

//     // methods
//     virtual void OnTerminate(int pid, int status);
// };

class CommandLineTools {

public :
    wxString bin_dir;
    wxString executable;
    wxString outfile;
    wxString errfile;
    wxString args;
    // int process_id;
    // AsyncProcess *process;
    long exit_code;
    wxArrayString return_array_string;
    wxString return_string;
    wxString error_string;
    wxString output_string;

    // constructors & deconstructors
    CommandLineTools();
    ~CommandLineTools();

    // methods
    void Init(wxString wanted_bin_dir, wxString wanted_executable, wxString wanted_outfile, wxString wanted_errfile);
    void AddArgument(wxString wanted_argument);
    wxString GetCommand();
    // wxString Run();
    wxString RunSync();
    wxArrayString RunAsync();
    wxString RedirectedSystemCall(wxString command);
    wxString RedirectedSystemCallWithError(wxString command);
    wxString ReadFile(wxString path);
    void WriteFile(wxString path, wxString contents);
    void WriteFile(wxString path, wxArrayString contents);
    // wxArrayString SystemCallWithFiles(wxString command);
};

class WriteAndRunScript : public CommandLineTools {

public:
    wxString shell;
    wxString script;
    wxFileName filename;
    wxString output;
    wxString error;

    WriteAndRunScript();
    ~WriteAndRunScript();
    void Init(wxString wanted_shell, wxString wanted_script, wxString wanted_filename);
    bool Write();
    bool Run();
    wxString GetOutput();
    wxString GetError();
};

WX_DECLARE_OBJARRAY(long, ArrayOfLongs);

class DenmodJob : public CommandLineTools {

public:
 wxString original_map_filename;
 wxString half_map_1_filename;
 wxString half_map_2_filename;
 wxString working_dir;
 wxString denmod_map_filename;

    DenmodJob();
    ~DenmodJob();
    void Init(wxString wanted_bin_dir, wxString wanted_executable, wxString wanted_outfile, wxString wanted_errfile, wxString wanted_original_map_filename="", wxString wanted_half_map_1_filename="", wxString wanted_half_map_2_filename="", wxString wanted_working_dir="", wxString wanted_denmod_map_filename="");

};

class DenmodJobWrapup : public CommandLineTools {

public:
    wxString output_path;
    wxString reconstructed_volume_path;
    wxString denmod_volume_path;
    wxString grep_script;

    DenmodJobWrapup();
    ~DenmodJobWrapup();
    void Init(wxString wanted_output_path, wxString wanted_reconstructed_volume_path, wxString wanted_denmod_volume_path);
    ArrayOfLongs GetBounds();
    void Run();
};