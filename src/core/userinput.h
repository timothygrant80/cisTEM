// User Input Class :-

class UserInput {

        private:

                void Init(const char *program_name, wxString program_version);

                FILE *defaults_file;
                FILE *new_defaults_file;

                char defaults_filename[1000];
                char new_defaults_filename[1000];
                char memory_string[10000];

                void GetDefault(const char *my_text, const char *default_default_value, char *default_value);
                void CarryOverDefaults();
                void DoGotValidAnswer(const char *question_text, const char *new_default);
                void DoGotInvalidAnswer();

                bool output_is_a_tty;
                bool input_is_a_tty;

	public:


		// Constructors
		UserInput();
		UserInput(const char *program_name, float program_version);
		UserInput(const char *program_name, wxString program_version);
		~UserInput();

		// Methods
				void AskQuestion(const char *question_text, const char *help_text, const char *default_value, char *received_input);

                float GetFloatFromUser(const char * my_text, const char * help_text, const char * wanted_default_value = 0, float min_value = -FLT_MAX, float max_value = FLT_MAX);
                int GetIntFromUser(const char * my_text, const char * help_text, const char * wanted_default_value = 0, int min_value = INT_MIN, int max_value = INT_MAX);
                std::string GetFilenameFromUser(const char * my_question_text, const char * help_text, const char * wanted_default_value = 0, bool must_exist = false);
                std::string GetStringFromUser(const char * my_question_text, const char * help_text, const char * wanted_default_value = 0);
                std::string GetSymmetryFromUser(const char * my_question_text, const char * help_text, const char * wanted_default_value = 0);
                bool GetYesNoFromUser(const char * my_test, const char * help_text, const char * wanted_default_value = 0);

                //void GetTextFromUser(char *text_buffer, const char * my_text, const char * help_text, const char * wanted_default_value = 0);
                //void GetTXTFilenameFromUser(char *Filename, const char * my_text, const char * help_text, long status, const char * wanted_default_value = 0);
                //void GetPLTFilenameFromUser(char *Filename, const char * my_text, const char * help_text, long status, const char * wanted_default_value = 0);
                // void GetCLSFilenameFromUser(char *Filename, const char * my_text, const char * help_text, long status, const char * wanted_default_value = 0);
                //void GetPlainFilenameFromUser(char *Filename, const char * my_text, const char * help_text, long status, const char * wanted_default_value);
                //void GetGeneralFilenameFromUser(char *Filename, const char * my_text, const char * help_text, long status, const char * extension, const char * wanted_default_value);
                //void GetSymmetryFromUser(char *symmetry_type, long *symmetry_number, const char * my_text, const char * help_text, const char * wanted_default_value = 0);
                //long GetOptionFromUser(char *option_list, long number_of_options, const char * my_text, const char * help_text, const char * wanted_default_value = 0);

};
