#define OPEN_TO_READ 0
#define OPEN_TO_WRITE 1
#define OPEN_TO_APPEND 2

class NumericTextFile {

        private:

                void Init();
                wxString text_filename;
                long access_type;
                wxFileInputStream *input_file_stream;
                wxTextInputStream *input_text_stream;
                wxFileOutputStream *output_file_stream;
                wxTextOutputStream *output_text_stream;

	public:


		// Constructors
		NumericTextFile();
		NumericTextFile(wxString Filename, long wanted_access_type, long wanted_records_per_line = 1);
		~NumericTextFile();

		// data

		int number_of_lines;
		int records_per_line;

		// Methods

        void Open(wxString Filename, long wanted_access_type, long wanted_records_per_line = 1);
        void Close();
        void Rewind();
        void Flush();
        wxString ReturnFilename();

		void ReadLine(float *data_array);
        void WriteLine(float *data_array);
        void WriteLine(double *data_array);
        void WriteCommentLine(const char * format, ...);

};
