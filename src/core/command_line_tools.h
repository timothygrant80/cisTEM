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

};