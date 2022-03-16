

// we need a header to describe how to decode the stream. - which i'm about to make up off the top of my head..  SENDER AND RECEIVER MUST HAVE THE SAME ENDIANESS

// 4 bytes = number_of_jobs (int)
// 4 bytes = number_of_processes (int)
// 4 bytes = length_of_command_string (int)
// n bytes = command_string (chars)
// then we loop over each job.. of which there are (1st 4 bytes)
// 4 bytes = number_of_arguments;
// 1 byte = type of argument;
// (on of following depending on type of argument :-)
// 4 bytes - int
// 4 bytes - float
// 4 bytes - length of text (int) followed by n bytes where n is length of string
// 1 byte - bool (0 or 1)

class RunArgument {

  public:
    bool is_allocated;
    int  type_of_argument;

    std::string* string_argument;
    int*         integer_argument;
    float*       float_argument;
    bool*        bool_argument;

    RunArgument( );
    ~RunArgument( );

    void Deallocate( );

    void SetStringArgument(const char* wanted_text);
    void SetIntArgument(int wanted_argument);
    void SetFloatArgument(float wanted_argument);
    void SetBoolArgument(bool wanted_argument);

    inline std::string ReturnStringArgument( ) {
        MyDebugAssertTrue(type_of_argument == TEXT, "Returning wrong type!");
        return string_argument[0];
    }

    inline int ReturnIntegerArgument( ) {
        MyDebugAssertTrue(type_of_argument == INTEGER, "Returning wrong type!");
        return integer_argument[0];
    }

    inline float ReturnFloatArgument( ) {
        MyDebugAssertTrue(type_of_argument == FLOAT, "Returning wrong type!");
        return float_argument[0];
    }

    inline bool ReturnBoolArgument( ) {
        MyDebugAssertTrue(type_of_argument == BOOL, "Returning wrong type!");
        return bool_argument[0];
    }

    long ReturnEncodedByteTransferSize( );
};

class RunJob {

  public:
    int job_number;
    int number_of_arguments;

    bool         has_been_run;
    RunArgument* arguments;

    RunJob( );
    ~RunJob( );

    void     Reset(int wanted_number_of_arguments);
    void     Deallocate( );
    void     SetArguments(const char* format, va_list args);
    void     ManualSetArguments(const char* format, ...);
    long     ReturnEncodedByteTransferSize( );
    bool     SendJob(wxSocketBase* socket);
    bool     RecieveJob(wxSocketBase* socket);
    void     PrintAllArguments( );
    wxString PrintAllArgumentsTowxString( );

    RunJob& operator=(const RunJob& other_job);
    RunJob& operator=(const RunJob* other_job);
};

class JobPackage {

  public:
    int number_of_jobs;
    int number_of_added_jobs;

    RunProfile my_profile;
    RunJob*    jobs;

    JobPackage(RunProfile wanted_profile, wxString wanted_executable_name, int wanted_number_of_jobs);
    JobPackage( );
    ~JobPackage( );

    void Reset(RunProfile wanted_profile, wxString wanted_executable_name, int wanted_number_of_jobs);
    void AddJob(const char* format, ...);
    bool SendJobPackage(wxSocketBase* socket);
    bool ReceiveJobPackage(wxSocketBase* socket);

    long ReturnEncodedByteTransferSize( );
    int  ReturnNumberOfJobsRemaining( );

    JobPackage& operator=(const JobPackage& other_package);
    JobPackage& operator=(const JobPackage* other_package);
};

WX_DECLARE_OBJARRAY(JobPackage, ArrayofJobPackages);

class JobResult {

  public:
    int    job_number;
    int    result_size;
    float* result_data;

    JobResult( );
    JobResult(int wanted_result_size, float* wanted_result_data);
    JobResult(const JobResult& obj); // copy contructor

    ~JobResult( );

    JobResult& operator=(const JobResult& other_result);
    JobResult& operator=(const JobResult* other_result);

    void SetResult(int wanted_result_size, float* wanted_result_data);
    bool SendToSocket(wxSocketBase* wanted_socket);
    bool ReceiveFromSocket(wxSocketBase* wanted_socket);
};

WX_DECLARE_OBJARRAY(JobResult, ArrayofJobResults);

bool ReceiveResultQueueFromSocket(wxSocketBase* socket, ArrayofJobResults& my_array);
bool SendResultQueueToSocket(wxSocketBase* socket, ArrayofJobResults& my_array);
