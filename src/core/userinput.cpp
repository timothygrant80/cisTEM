#include "core_headers.h"

UserInput::UserInput( ) {
    std::cout << "Warning, UserInput Class declared with no program name!" << std::endl
              << std::endl;
    Init("default", "0.0");
}

UserInput::UserInput(const char* program_name, float program_version) {
    Init(program_name, Format("%0.2f", program_version));
}

UserInput::UserInput(const char* program_name, std::string program_version) {
    Init(program_name, program_version);
}

UserInput::~UserInput( ) {

    if ( input_is_a_tty == true ) {

        // we need to carry over any unused defaults!
        if ( defaults_file != 0 && new_defaults_file != 0 )
            CarryOverDefaults( );

        if ( defaults_file != 0 )
            fclose(defaults_file);

        if ( new_defaults_file != 0 ) {
            fclose(new_defaults_file);
            remove(defaults_filename);
            rename(new_defaults_filename, defaults_filename);
            remove(new_defaults_filename);
        }
    }
}

void UserInput::AskQuestion(const char* question_text, const char* help_text, const char* default_value, char* received_input) {
    int  current_length;
    char temp_string[1000];

    for ( int i = 0; i < 1000; i++ ) {
        temp_string[i] = 0;
    }

    strcpy(temp_string, question_text);

    if ( input_is_a_tty == true && default_value[0] != 0 ) {
        strcat(temp_string, " [");
        strcat(temp_string, default_value);
        strcat(temp_string, "]");

        current_length = 50 - strlen(temp_string);

        if ( current_length >= 0 ) {
            Printf("%s ", question_text);
            MyPrintfCyan("[%s]", default_value);
        }
        else {
            Printf("%s\n", question_text);
            MyPrintfCyan("[%s]", default_value);

            current_length = 48 - strlen(default_value);
        }
    }
    else {
        current_length = 50 - strlen(temp_string);
        Printf("%s", question_text);
    }

    for ( long temp_counter = 0; temp_counter < current_length; temp_counter++ ) {
        Printf(" ");
    }

    Printf(" : ");

    // get some input..

    std::cin.getline(received_input, 1000);

    if ( received_input[0] == '*' )
        exit(0); // user request exit
    else if ( received_input[0] == 0 ) {
        if ( input_is_a_tty == false ) {
            Printf("\n Error: Blank answer in scripted mode, exiting...\n\n");
            exit(-1);
        }
        else
            strcpy(received_input, default_value);
    }
    else if ( received_input[0] == '?' ) {
        Printf("\n%s\n\n", help_text);
    }
}

void UserInput::Init(const char* program_name, std::string program_version) {
    int counter;

    defaults_file     = NULL;
    new_defaults_file = NULL;

    for ( counter = 0; counter < 1000; counter++ ) {
        defaults_filename[counter]     = 0;
        new_defaults_filename[counter] = 0;
    }

    for ( counter = 0; counter < 10000; counter++ ) {
        memory_string[counter] = 0;
    }

    input_is_a_tty  = InputIsATerminal( );
    output_is_a_tty = OutputIsAtTerminal( );

    if ( input_is_a_tty ) {
        for ( counter = 0; counter < 1000; counter++ ) {
            defaults_filename[counter]     = 0;
            new_defaults_filename[counter] = 0;
        }

        defaults_filename[0] = '.';
        strcat(defaults_filename, program_name);
        strcat(defaults_filename, ".dff");

        new_defaults_filename[0] = '.';
        strcat(new_defaults_filename, "current");
        strcat(new_defaults_filename, program_name);
        strcat(new_defaults_filename, ".dff");

        defaults_file     = fopen(defaults_filename, "r");
        new_defaults_file = fopen(new_defaults_filename, "w+");

        if ( new_defaults_file == 0 )
            Printf("\n\nError Can't open defaults file! for writing\n\n");
    }

    Printf("\n\n        **   Welcome to %s   **\n\n", program_name);
    for ( counter = 0; counter < strlen(program_name) / 2; counter++ )
        Printf(" ");
    Printf("         Version : %s\n", program_version);
    for ( counter = 0; counter < strlen(program_name) / 2; counter++ )
        Printf(" ");
    Printf("        Compiled : %s\n", __DATE__);
    for ( counter = 0; counter < strlen(program_name) / 2; counter++ )
        Printf(" ");
    Printf(" Library Version : %s\n", CISTEM_VERSION_TEXT);
    for ( counter = 0; counter < strlen(program_name) / 2; counter++ )
        Printf(" ");
    Printf("     From Branch : %s\n", CISTEM_CURRENT_BRANCH);
    for ( counter = 0; counter < strlen(program_name) / 2; counter++ )
        Printf(" ");
    Printf("            Mode : ");

    if ( input_is_a_tty == true )
        Printf("Interactive\n\n");
    else
        Printf("Scripted\n\n");
}

bool UserInput::GetYesNoFromUser(const char* my_text, const char* help_text, const char* wanted_default_value) {

    char input[1000];
    char default_value[1000];

    for ( int i = 0; i < 1000; i++ ) {
        input[i]         = 0;
        default_value[i] = 0;
    }

    // Sort out the defaults..
    GetDefault(my_text, wanted_default_value, default_value);

    while ( 1 == 1 ) {
        AskQuestion(my_text, help_text, default_value, input);

        if ( input[0] == 'y' || input[0] == 'Y' ) {
            DoGotValidAnswer(my_text, input);
            return true;
        }
        else if ( input[0] == 'n' || input[0] == 'N' ) {
            DoGotValidAnswer(my_text, input);
            return false;
        }

        DoGotInvalidAnswer( );
    }
}

float UserInput::GetFloatFromUser(const char* my_text, const char* help_text, const char* wanted_default_value, float min_value, float max_value) {
    float my_float;
    char  default_value[1000];
    char  input[1000];

    for ( int i = 0; i < 1000; i++ ) {
        default_value[i] = 0;
        input[i]         = 0;
    }

    // Sort out the defaults..

    GetDefault(my_text, wanted_default_value, default_value);

    while ( 1 == 1 ) {
        AskQuestion(my_text, help_text, default_value, input);

        if ( input[0] == 45 || input[0] == 46 ) {
            if ( input[1] >= 48 && input[1] <= 57 ) {
                my_float = atof(input);
                if ( my_float >= min_value && my_float <= max_value ) {
                    DoGotValidAnswer(my_text, input);
                    return my_float;
                }
                else {
                    if ( output_is_a_tty == true ) {
                        MyPrintfRed("\nError: Number outside of acceptable range!\n\n");
                    }
                    else {
                        Printf("\nError: Number outside of acceptable range!\n\n");
                    }
                }
            }
        }
        else if ( input[0] >= 48 && input[0] <= 57 ) {
            my_float = atof(input);
            if ( my_float >= min_value && my_float <= max_value ) {
                DoGotValidAnswer(my_text, input);
                return my_float;
            }
            else {
                if ( output_is_a_tty == true ) {
                    MyPrintfRed("\nError: Number outside of acceptable range!\n\n");
                }
                else {
                    Printf("\nError: Number outside of acceptable range!\n\n");
                }
            }
        }

        DoGotInvalidAnswer( );
    }
}

int UserInput::GetIntFromUser(const char* my_text, const char* help_text, const char* wanted_default_value, int min_value, int max_value) {
    int  my_int;
    char input[1000];
    char default_value[1000];

    for ( int i = 0; i < 1000; i++ ) {
        input[i]         = 0;
        default_value[i] = 0;
    }

    GetDefault(my_text, wanted_default_value, default_value);

    while ( 1 == 1 ) {
        AskQuestion(my_text, help_text, default_value, input);

        if ( input[0] == 45 ) {
            if ( input[1] >= 48 && input[1] <= 57 ) {
                my_int = atoi(input);
                if ( my_int >= min_value && my_int <= max_value ) {
                    DoGotValidAnswer(my_text, input);
                    return my_int;
                }
                else {
                    if ( output_is_a_tty == true ) {
                        MyPrintfRed("\nError: Number outside of acceptable range!\n\n");
                    }
                    else {
                        Printf("\nError: Number outside of acceptable range!\n\n");
                    }
                }
            }
        }
        else if ( input[0] >= 48 && input[0] <= 57 ) {
            my_int = atoi(input);
            if ( my_int >= min_value && my_int <= max_value ) {
                DoGotValidAnswer(my_text, input);
                return my_int;
            }
            else {
                if ( output_is_a_tty == true ) {
                    MyPrintfRed("\nError: Number outside of acceptable range!\n\n");
                }
                else {
                    Printf("\nError: Number outside of acceptable range!\n\n");
                }
            }
        }

        DoGotInvalidAnswer( );
    }
}

std::string UserInput::GetStringFromUser(const char* my_text, const char* help_text, const char* wanted_default_value) {

    char input[1000];
    char default_value[1000];

    for ( int i = 0; i < 1000; i++ ) {
        input[i]         = 0;
        default_value[i] = 0;
    }

    GetDefault(my_text, wanted_default_value, default_value);

    while ( 1 == 1 ) {
        AskQuestion(my_text, help_text, default_value, input);

        if ( input[0] != 0 && input[0] != '?' ) {
            DoGotValidAnswer(my_text, input);
            std::string my_string(input);
            return my_string;
        }

        DoGotInvalidAnswer( );
    }
}

std::string UserInput::GetSymmetryFromUser(const char* my_text, const char* help_text, const char* wanted_default_value) {
    char input[1000];
    char default_value[1000];

    for ( int i = 0; i < 1000; i++ ) {
        input[i]         = 0;
        default_value[i] = 0;
    }

    GetDefault(my_text, wanted_default_value, default_value);

    while ( 1 == 1 ) {
        AskQuestion(my_text, help_text, default_value, input);

        if ( input[0] != 0 && input[0] != '?' ) {
            std::string supplied_input = input;
            char   symmetry_type;
            long     symmetry_number;

            if ( supplied_input.length() < 1 ) {
                if ( output_is_a_tty == true ) {
                    MyPrintfRed("\nError: Must specify symmetry symbol!\n\n");
                }
                else {
                    Printf("\nError: Must specify symmetry symbol!\n\n");
                }
            }
            else {
                symmetry_type = toupper(static_cast<unsigned char>(supplied_input[0]));

                if ( supplied_input.length() == 1 ) {
                    symmetry_number = 0;
                }
                else {
                    if ( ! StringToLong(supplied_input.substr(1), symmetry_number) ) {
                        symmetry_number = -1;
                        if ( output_is_a_tty == true ) {
                            MyPrintfRed("\nError: Unrecognized symmetry number!\n\n");
                        }
                        else {
                            Printf("\nError: Unrecognized symmetry number!\n\n");
                        }
                    }
                }

                if ( symmetry_number >= 0 ) {
                    if ( (symmetry_type == 'C' || symmetry_type == 'D') && symmetry_number > 0 ) {
                        DoGotValidAnswer(my_text, input);
                        std::string my_string(input);
                        return my_string;
                    }

                    if ( symmetry_type == 'T' && (symmetry_number < 3) ) {
                        DoGotValidAnswer(my_text, input);
                        std::string my_string(input);
                        return my_string;
                    }

                    if ( symmetry_type == 'O' && (symmetry_number == 0) ) {
                        DoGotValidAnswer(my_text, input);
                        std::string my_string(input);
                        return my_string;
                    }

                    if ( symmetry_type == 'I' && (symmetry_number < 3) ) {
                        DoGotValidAnswer(my_text, input);
                        std::string my_string(input);
                        return my_string;
                    }

                    if ( output_is_a_tty == true ) {
                        MyPrintfRed("\nError: Unrecognized symmetry!\n\n");
                    }
                    else {
                        Printf("\nError: Unrecognized symmetry!\n\n");
                    }
                }
            }
        }
    }

    DoGotInvalidAnswer( );
}

std::string UserInput::GetFilenameFromUser(const char* my_text, const char* help_text, const char* wanted_default_value, bool must_exist) {

    char input[1000];
    char default_value[1000];
    for ( int i = 0; i < 1000; i++ ) {
        input[i]         = 0;
        default_value[i] = 0;
    }

    GetDefault(my_text, wanted_default_value, default_value);

    while ( 1 == 1 ) {
        AskQuestion(my_text, help_text, default_value, input);

        if ( input[0] != 0 && input[0] != '?' ) {

            if ( must_exist == false ) {
                DoGotValidAnswer(my_text, input);
                std::string my_string(input);
                return my_string;
            }
            else {
                // check the file exits..

                if ( FileExists(input) == true ) {
                    DoGotValidAnswer(my_text, input);
                    std::string my_string(input);
                    return my_string;
                }
                else {
                    if ( output_is_a_tty == true ) {
                        MyPrintfRed("\nError: File does not exist, please provide an existing file!\n\n");
                    }
                    else {
                        Printf("\nError: File does not exist, please provide an existing file!\n\n");
                    }
                }
            }
        }

        DoGotInvalidAnswer( );
    }
}

void UserInput::GetDefault(const char* my_text, const char* default_default_value, char* default_value) {
    default_value[0] = 0;

    if ( defaults_file != 0 && input_is_a_tty == true ) {
        int  file_check = 1;
        char label_temp[1000];
        char current_label[1000];

        default_value[0] = 0;
        for ( int i = 0; i < 1000; i++ ) {
            current_label[i] = 0;
            label_temp[i]    = 0;
        }

        rewind(defaults_file);

        while ( file_check != EOF ) {
            while ( file_check != EOF ) {
                file_check = fscanf(defaults_file, "%s", label_temp);
                strcat(current_label, label_temp);

                if ( current_label[int(strlen(current_label)) - 1] == ':' && current_label[int(strlen(current_label)) - 2] == ':' ) {
                    current_label[int(strlen(current_label)) - 2] = 0;
                    break;
                }
                else
                    strcat(current_label, " ");
            }

            file_check = fscanf(defaults_file, "%s", default_value);

            if ( strcmp(current_label, my_text) == 0 ) {
                // we have found a matching default so finish!
                break;
            }
            else {
                default_value[0] = 0;
                current_label[0] = 0;
            }
        }
    }

    if ( default_default_value[0] != 0 && default_value[0] == 0 && input_is_a_tty == true )
        strcpy(default_value, default_default_value);
}

void UserInput::DoGotValidAnswer(const char* question_text, const char* new_default) {
    if ( new_defaults_file != 0 && input_is_a_tty == true )
        fprintf(new_defaults_file, "%s:: %s\n", question_text, new_default);
    if ( input_is_a_tty == false )
        Printf("%s\n", new_default);
}

void UserInput::DoGotInvalidAnswer( ) {
    // if we got here and we are in scripted mode, something went wrong.

    if ( input_is_a_tty == false ) {
        Printf("\n Error: Running as script, and answer is not recognized...\n\n");
        exit(-1);
    }
}

void UserInput::CarryOverDefaults( ) {
    if ( input_is_a_tty == true ) {

        int  file_check         = 1;
        int  file_check_newfile = 1;
        char current_label[1000];
        char current_label_newfile[1000];
        char current_default[1000];
        char current_default_newfile[1000];
        char label_temp[1000];
        char label_temp_newfile[1000];
        bool is_present;

        long file_pos;

        for ( int i = 0; i < 1000; i++ ) {
            current_label[i]           = 0;
            current_label_newfile[i]   = 0;
            current_default[i]         = 0;
            current_default_newfile[i] = 0;
            label_temp[i]              = 0;
            label_temp_newfile[i]      = 0;
        }

        rewind(defaults_file);
        file_pos                 = ftell(new_defaults_file);
        current_label_newfile[0] = 0;

        while ( file_check != EOF ) {
            current_label[0] = 0;
            while ( file_check != EOF ) {
                file_check = fscanf(defaults_file, "%s", label_temp);
                strcat(current_label, label_temp);

                if ( current_label[int(strlen(current_label)) - 1] == ':' && current_label[int(strlen(current_label)) - 2] == ':' ) {
                    current_label[int(strlen(current_label)) - 2] = 0;
                    file_check                                    = fscanf(defaults_file, "%s", current_default);
                    break;
                }
                else
                    strcat(current_label, " ");
            }

            if ( file_check == EOF )
                break;

            // search for that in the defaults file we are writing...

            rewind(new_defaults_file);
            file_check_newfile = 1;
            is_present         = false;

            while ( file_check_newfile != EOF ) {
                file_check_newfile = fscanf(new_defaults_file, "%s", label_temp_newfile);
                strcat(current_label_newfile, label_temp_newfile);

                if ( current_label_newfile[int(strlen(current_label_newfile)) - 1] == ':' && current_label_newfile[int(strlen(current_label_newfile)) - 2] == ':' ) {
                    current_label_newfile[int(strlen(current_label_newfile)) - 2] = 0;
                    file_check_newfile                                            = fscanf(new_defaults_file, "%s", current_default_newfile);

                    if ( strcmp(current_label, current_label_newfile) == 0 ) {
                        is_present = true;
                        break;
                    }
                    else
                        current_label_newfile[0] = 0;
                }
                else
                    strcat(current_label_newfile, " ");
            }

            if ( is_present == false ) {
                // it doesn't exist so write it at the end
                fseek(new_defaults_file, file_pos, SEEK_SET);
                fprintf(new_defaults_file, "%s:: %s\n", current_label, current_default);
                file_pos = ftell(new_defaults_file);
            }
        }
    }
}

/*
void UserInput::GetTextFromUser(char *text_buffer, const char * my_text, const char * help_text, const char * wanted_default_value = 0)
{
   while(1==1)
   {
    std::cout << my_text << " : ";
    std::cin.getline(text_buffer, 100);

    if (text_buffer[0] == '?') std::cout << std::endl << help_text << std::endl << std::endl;
    else
    if (text_buffer[0] != 0) break;

   }

}
*/

/*
void UserInput::GetPLTFilenameFromUser(char *Filename, const char * my_text, const char * help_text, long status, const char * wanted_default_value)
{
  // status of 0 means i don't care
  // status of 1 means must exist
  // status of 2 means wipe it, error if can't

  FILE *input;
  char PLTFilename[1010];
  char default_value[1000];
  long current_length = 0;
  default_value[0] = 0;

  // Sort out the defaults..

  if (defaults_file != 0)
  {
  	// look for a default matching the input text..
  	GetDefault(my_text, default_value);
  }

  if (wanted_default_value != 0 && default_value[0] == 0) strcpy(default_value, wanted_default_value);

  while (1==1)
  {
    char temp_string[1000];
    strcpy (temp_string, my_text);

    if (default_value[0] != 0)
    {
      strcat (temp_string, " [");
      strcat (temp_string, default_value);
      strcat (temp_string, "]");
    }

    current_length = 50 - strlen(temp_string);

    std::cout << temp_string;
    for (long temp_counter = 0; temp_counter < current_length; temp_counter++)
    {
      std::cout << " ";
    }
    std::cout << " : ";

    std::cin.getline(Filename, 500);

    if (Filename[0] == '*') exit(0);

    if (Filename[0] == 0) strcpy(Filename, default_value);

    strcpy(PLTFilename, Filename);
    strcat(PLTFilename, ".plt");

    if (Filename[0] == '?') std::cout << std::endl << help_text << std::endl << std::endl;
    else
    if (Filename[0] != 0)
    {
      if (status == 0) {fprintf (new_defaults_file, "%s:: %s\n", my_text, Filename); break;}
      else
      if (status == 1)
      {

        input = fopen(PLTFilename, "rb");
        if (input==NULL)
        {
          perror ("This error has occurred (ReadPLT) : ");
        }
        else
        {
          fclose(input);
          fprintf (new_defaults_file, "%s:: %s\n", my_text, Filename);
          break;
        }
      }
      else
      if (status == 2)
      {
    	  input = fopen(PLTFilename, "r+b");

    	  if (input == 0)
    	  {
    		   input = fopen(PLTFilename, "w+b");
    	  }

        if (input == NULL)
        {
          std::cout << "\n\nI can't open the output file for some reason.. quitting...\n\n";
          exit(-1);
        }
        else
        {
          fclose(input);
          remove(PLTFilename);
          fprintf (new_defaults_file, "%s:: %s\n", my_text, Filename);
	  break;
        }

      }
    }


  }
}

void UserInput::GetTXTFilenameFromUser(char *Filename, const char * my_text, const char * help_text, long status, const char * wanted_default_value)
{
  // status of 0 means i don't care
  // status of 1 means must exist
  // status of 2 means wipe it, error if can't

  FILE *input;
  char TXTFilename[1010];
  char default_value[1000];
  long current_length = 0;
  default_value[0] = 0;

  // Sort out the defaults..

  if (defaults_file != 0)
  {
  	// look for a default matching the input text..
  	GetDefault(my_text, default_value);
  }

  if (wanted_default_value != 0 && default_value[0] == 0) strcpy(default_value, wanted_default_value);

  while (1==1)
  {
    char temp_string[1000];
    strcpy (temp_string, my_text);

    if (default_value[0] != 0)
    {
      strcat (temp_string, " [");
      strcat (temp_string, default_value);
      strcat (temp_string, "]");
    }

    current_length = 50 - strlen(temp_string);

    std::cout << temp_string;
    for (long temp_counter = 0; temp_counter < current_length; temp_counter++)
    {
      std::cout << " ";
    }
    std::cout << " : ";

    std::cin.getline(Filename, 500);

    if (Filename[0] == '*') exit(0);

    if (Filename[0] == 0) strcpy(Filename, default_value);

    strcpy(TXTFilename, Filename);
    strcat(TXTFilename, ".txt");

    if (Filename[0] == '?') std::cout << std::endl << help_text << std::endl << std::endl;
    else
    if (Filename[0] != 0)
    {
      if (status == 0) {fprintf (new_defaults_file, "%s:: %s\n", my_text, Filename); break;}
      else
      if (status == 1)
      {

        input = fopen(TXTFilename, "rb");
        if (input==NULL)
        {
          perror ("This error has occurred (ReadTXT) : ");
        }
        else
        {
          fclose(input);
          fprintf (new_defaults_file, "%s:: %s\n", my_text, Filename);
          break;
        }
      }
      else
      if (status == 2)
      {
       	  input = fopen(TXTFilename, "r+b");

        	  if (input == 0)
        	  {
        		   input = fopen(TXTFilename, "w+b");
        	  }

        if (input == NULL)
        {
          std::cout << "\n\nI can't open the output file for some reason.. quitting...\n\n";
          exit(-1);
        }
        else
        {
          fclose(input);
          remove(TXTFilename);
          fprintf (new_defaults_file, "%s:: %s\n", my_text, Filename);
	  break;
        }

      }
    }


  }
}

void UserInput::GetGeneralFilenameFromUser(char *Filename, const char * my_text, const char * help_text, long status, const char *extension, const char * wanted_default_value)
{
  // status of 0 means i don't care
  // status of 1 means must exist
  // status of 2 means wipe it, error if can't

  FILE *input;
  char GeneralFilename[1010];
  char default_value[1000];
  long current_length = 0;
  default_value[0] = 0;

  // Sort out the defaults..

  if (defaults_file != 0)
  {
  	// look for a default matching the input text..
  	GetDefault(my_text, default_value);
  }

  if (wanted_default_value != 0 && default_value[0] == 0) strcpy(default_value, wanted_default_value);

  while (1==1)
  {
    char temp_string[1000];
    strcpy (temp_string, my_text);

    if (default_value[0] != 0)
    {
      strcat (temp_string, " [");
      strcat (temp_string, default_value);
      strcat (temp_string, "]");
    }

    current_length = 50 - strlen(temp_string);

    std::cout << temp_string;
    for (long temp_counter = 0; temp_counter < current_length; temp_counter++)
    {
      std::cout << " ";
    }
    std::cout << " : ";

    std::cin.getline(Filename, 500);

    if (Filename[0] == '*') exit(0);

    if (Filename[0] == 0) strcpy(Filename, default_value);

    strcpy(GeneralFilename, Filename);
    strcat(GeneralFilename, ".");
    strcat(GeneralFilename, extension);

    if (Filename[0] == '?') std::cout << std::endl << help_text << std::endl << std::endl;
    else
    if (Filename[0] != 0)
    {
      if (status == 0) {fprintf (new_defaults_file, "%s:: %s\n", my_text, Filename); break;}
      else
      if (status == 1)
      {

        input = fopen(GeneralFilename, "rb");
        if (input==NULL)
        {
          perror ("This error has occurred (ReadGeneral) : ");
        }
        else
        {
          fclose(input);
          fprintf (new_defaults_file, "%s:: %s\n", my_text, Filename);
          break;
        }
      }
      else
      if (status == 2)
      {
       	  input = fopen(GeneralFilename, "r+b");

        	  if (input == 0)
        	  {
        		   input = fopen(GeneralFilename, "w+b");
        	  }
        if (input == NULL)
        {
          std::cout << "\n\nI can't open the output file for some reason.. quitting...\n\n";
          exit(-1);
        }
        else
        {
          fclose(input);
          remove(GeneralFilename);
          fprintf (new_defaults_file, "%s:: %s\n", my_text, Filename);
	  break;
        }

      }
    }


  }
}

void UserInput::GetPlainFilenameFromUser(char *Filename, const char * my_text, const char * help_text, long status,  const char * wanted_default_value)
{
  // status of 0 means i don't care
  // status of 1 means must exist
  // status of 2 means wipe it, error if can't

  FILE *input;
  char GeneralFilename[1010];
  char default_value[1000];
  long current_length = 0;
  default_value[0] = 0;

  // Sort out the defaults..

  if (defaults_file != 0)
  {
  	// look for a default matching the input text..
  	GetDefault(my_text, default_value);
  }

  if (wanted_default_value != 0 && default_value[0] == 0) strcpy(default_value, wanted_default_value);

  while (1==1)
  {
    char temp_string[1000];
    strcpy (temp_string, my_text);

    if (default_value[0] != 0)
    {
      strcat (temp_string, " [");
      strcat (temp_string, default_value);
      strcat (temp_string, "]");
    }

    current_length = 50 - strlen(temp_string);

    std::cout << temp_string;
    for (long temp_counter = 0; temp_counter < current_length; temp_counter++)
    {
      std::cout << " ";
    }
    std::cout << " : ";

    std::cin.getline(Filename, 500);

    if (Filename[0] == '*') exit(0);

    if (Filename[0] == 0) strcpy(Filename, default_value);


    if (Filename[0] == '?') std::cout << std::endl << help_text << std::endl << std::endl;
    else
    if (Filename[0] != 0)
    {
      if (status == 0) {fprintf (new_defaults_file, "%s:: %s\n", my_text, Filename); break;}
      else
      if (status == 1)
      {

        input = fopen(Filename, "rb");
        if (input==NULL)
        {
          perror ("This error has occurred (ReadGeneral) : ");
        }
        else
        {
          fclose(input);
          fprintf (new_defaults_file, "%s:: %s\n", my_text, Filename);
          break;
        }
      }
      else
      if (status == 2)
      {
       	  input = fopen(Filename, "r+b");

        	  if (input == 0)
        	  {
        		   input = fopen(Filename, "w+b");
        	  }
        if (input == NULL)
        {
          std::cout << "\n\nI can't open the output file for some reason.. quitting...\n\n";
          exit(-1);
        }
        else
        {
          fclose(input);
      //    remove(GeneralFilename);
          fprintf (new_defaults_file, "%s:: %s\n", my_text, Filename);
	  break;
        }

      }
    }


  }
}

void UserInput::GetCLSFilenameFromUser(char *Filename, const char * my_text, const char * help_text, long status, const char * wanted_default_value)
{
  // status of 0 means i don't care
  // status of 1 means must exist
  // status of 2 means wipe it, error if can't

  FILE *input;
  char CLSFilename[1010];
  char default_value[1000];
  long current_length = 0;
  default_value[0] = 0;

  // Sort out the defaults..

  if (defaults_file != 0)
  {
  	// look for a default matching the input text..
  	GetDefault(my_text, default_value);
  }

  if (wanted_default_value != 0 && default_value[0] == 0) strcpy(default_value, wanted_default_value);

  while (1==1)
  {
    char temp_string[1000];
    strcpy (temp_string, my_text);

    if (default_value[0] != 0)
    {
      strcat (temp_string, " [");
      strcat (temp_string, default_value);
      strcat (temp_string, "]");
    }

    current_length = 50 - strlen(temp_string);

    std::cout << temp_string;
    for (long temp_counter = 0; temp_counter < current_length; temp_counter++)
    {
      std::cout << " ";
    }
    std::cout << " : ";

    std::cin.getline(Filename, 500);

    if (Filename[0] == '*') exit(0);

    if (Filename[0] == 0) strcpy(Filename, default_value);

    strcpy(CLSFilename, Filename);
    strcat(CLSFilename, ".cls");

    if (Filename[0] == '?') std::cout << std::endl << help_text << std::endl << std::endl;
    else
    if (Filename[0] != 0)
    {
      if (status == 0) {fprintf (new_defaults_file, "%s:: %s\n", my_text, Filename); break;}
      else
      if (status == 1)
      {

        input = fopen(CLSFilename, "rb");
        if (input==NULL)
        {
          perror ("This error has occurred (ReadCLS) : ");
        }
        else
        {
          fclose(input);
          fprintf (new_defaults_file, "%s:: %s\n", my_text, Filename);
          break;
        }
      }
      else
      if (status == 2)
      {
       	  input = fopen(CLSFilename, "r+b");

        	  if (input == 0)
        	  {
        		   input = fopen(CLSFilename, "w+b");
        	  }
        if (input == NULL)
        {
          std::cout << "\n\nI can't open the output file for some reason.. quitting...\n\n";
          exit(-1);
        }
        else
        {
          fclose(input);
          remove(CLSFilename);
          fprintf (new_defaults_file, "%s:: %s\n", my_text, Filename);
	  break;
        }

      }
    }


  }
}


void UserInput::GetSymmetryFromUser(char *symmetry_type, long *symmetry_number, const char * my_text, const char * help_text, const char * wanted_default_value)
{
  long current_length = 0;
  char default_value[1000];
  default_value[0] = 0;
  char Option[1000];

  // Sort out the defaults..

  if (defaults_file != 0)
  {
  	// look for a default matching the input text..
  	GetDefault(my_text, default_value);
  }

  if (wanted_default_value != 0 && default_value[0] == 0) strcpy(default_value, wanted_default_value);

  while (1==1)
  {
    char temp_string[1000];
    strcpy (temp_string, my_text);

    if (default_value[0] != 0)
    {
      strcat (temp_string, " [");
      strcat (temp_string, default_value);
      strcat (temp_string, "]");
    }

    current_length = 50 - strlen(temp_string);

    std::cout << temp_string;
    for (long temp_counter = 0; temp_counter < current_length; temp_counter++)
    {
      std::cout << " ";
    }
    std::cout << " : ";

    std::cin.getline(Option, 500);

    if (Option[0] == '*') exit(0);

    if (Option[0] == '?') std::cout << std::endl << help_text << std::endl << std::endl;

    if (Option[0] == 0) strcpy(Option, default_value);

    if (Option[0] == 'C' || Option[0] == 'c')
    {
      symmetry_type[0] = 'C';
      Option[0] = '0';
      symmetry_number[0] = long(atoi(Option));
      if (symmetry_number[0] != 0)
      {
        Option[0] = 'C';
        fprintf (new_defaults_file, "%s:: %s\n", my_text, Option);
        break;
      }
    }
    else
    if (Option[0] == 'D' || Option[0] == 'd')
    {
      symmetry_type[0] = 'D';
      Option[0] = '0';
      symmetry_number[0] = long(atoi(Option));
      if (symmetry_number[0] != 0)
      {
        Option[0] = 'D';
        fprintf (new_defaults_file, "%s:: %s\n", my_text, Option);
        break;
      }

    }
    else
    if (Option[0] == 'I' || Option[0] == 'i')
    {
      symmetry_type[0] = 'I';
      symmetry_number[0] = 1;
      if (symmetry_number[0] != 0)
      {
        Option[0] = 'I';
        fprintf (new_defaults_file, "%s:: %s\n", my_text, Option);
        break;
      }
    }
    else
    if (Option[0] == 'O' || Option[0] == 'o')
    {
      symmetry_type[0] = 'O';
      symmetry_number[0] = 1;
      if (symmetry_number[0] != 0)
      {
        Option[0] = 'O';
        fprintf (new_defaults_file, "%s:: %s\n", my_text, Option);
        break;
      }
    }
    else
    if (Option[0] == 'T' || Option[0] == 't')
    {
      symmetry_type[0] = 'T';
      symmetry_number[0] = 1;

      if (symmetry_number[0] != 0)
      {
         Option[0] = 'T';
         fprintf (new_defaults_file, "%s:: %s\n", my_text, Option);
         break;
      }
    }
  }
}

long UserInput::GetOptionFromUser(char *option_list, long number_of_options, const char * my_text, const char * help_text, const char * wanted_default_value)
{
  long current_length = 0;
  char default_value[1000];
  default_value[0] = 0;
  char Option[1000];
  long counter;

  // Sort out the defaults..

  if (defaults_file != 0)
  {
  	// look for a default matching the input text..
  	GetDefault(my_text, default_value);
  }

  if (wanted_default_value != 0 && default_value[0] == 0) strcpy(default_value, wanted_default_value);

  while (1==1)
  {
    char temp_string[1000];
    strcpy (temp_string, my_text);

    if (default_value[0] != 0)
    {
      strcat (temp_string, " [");
      strcat (temp_string, default_value);
      strcat (temp_string, "]");
    }

    current_length = 50 - strlen(temp_string);

    std::cout << temp_string;
    for (long temp_counter = 0; temp_counter < current_length; temp_counter++)
    {
      std::cout << " ";
    }
    std::cout << " : ";

    std::cin.getline(Option, 500);

    if (Option[0] == '*') exit(0);

    if (Option[0] == 0) strcpy(Option, default_value);

    for (counter = 0; counter < number_of_options; counter++)
    {
      if (Option[0] == option_list[counter])
      {
        fprintf (new_defaults_file, "%s:: %s\n", my_text, Option);
        return counter;
      }
    }
  }
}*/
