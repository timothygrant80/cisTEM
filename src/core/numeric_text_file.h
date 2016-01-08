#define OPEN_TO_READ 0
#define OPEN_TO_WRITE 1
#define OPEN_TO_APPEND 2

class NumericTextFile {

        private:

                void Init();
                FILE *text_file;
                char text_filename[410];
                long access_type;

	public:


		// Constructors
		NumericTextFile();
		NumericTextFile(wxString Filename, long wanted_access_type, long wanted_records_per_line = 1);
		~NumericTextFile();

		// data

		long number_of_lines;
		long records_per_line;

		// Methods

        void Open(wxString Filename, long wanted_access_type);
        void Close();
        void Rewind();
        void Flush();
        std::string ReturnFilename();

		long ReadLine(double *data_array);
        void WriteLine(double *data_array);
        void WriteCommentLine(const char * format, ...);

};
