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
    wxString args;
    // int process_id;
    // AsyncProcess *process;
    long exit_code;
    wxString return_string;

    // constructors & deconstructors
    CommandLineTools();
    ~CommandLineTools();

    // methods
    void Init(wxString wanted_bin_dir, wxString wanted_executable);
    void AddArgument(wxString wanted_argument);
    wxString GetCommand();
    // wxString Run();
    wxString RunSync();
    wxString RunAsync(int wanted_process_id);

};