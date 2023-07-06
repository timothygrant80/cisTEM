#ifndef _src_programs_refine_template_dev_refine_template_dev_h
#define _src_programs_refine_template_dev_refine_template_dev_h

class RefineTemplateArguments {
  public:
    std::string input_starfile;

    std::string output_starfile;

    std::string input_template;

    int start_position;

    int end_position;

    int num_threads;

    RefineTemplateArguments( ) {

        input_starfile = "input.star";

        output_starfile = "output.star";

        input_template = "input.mrc";

        start_position = 0;

        end_position = -1;

        num_threads = 1;
    };

    void recieve(RunArgument* arguments) {

        input_starfile = arguments[0].ReturnStringArgument( );

        output_starfile = arguments[1].ReturnStringArgument( );

        input_template = arguments[2].ReturnStringArgument( );

        start_position = arguments[3].ReturnIntegerArgument( );

        end_position = arguments[4].ReturnIntegerArgument( );

        num_threads = arguments[5].ReturnIntegerArgument( );
    };

    void userinput( ) {

        UserInput* my_input = new UserInput("RefineTemplateArguments", 1.00);

        input_starfile = my_input->GetFilenameFromUser("The starfile describing the matches that should be defined", "The starfile describing the matches that should be defined", "input.star", false);

        output_starfile = my_input->GetFilenameFromUser("The output strafile", "The output starfile", "output.star", false);

        input_template = my_input->GetFilenameFromUser("The template to be refined", "The template to be refined", "input.mrc", false);

        start_position = my_input->GetIntFromUser("The position in the starfile to start refining from", "The position in the starfile to start refining from", "0");

        end_position = my_input->GetIntFromUser("The position in the starfile to end refining at", "The position in the starfile to end refining at", "-1");

        num_threads = my_input->GetIntFromUser("The number of threads to use", "The number of threads to use", "1");

        delete my_input;
    };

    void setargument(RunJob& my_current_job) {

        my_current_job.ManualSetArguments("tttiii", input_starfile.c_str( ), output_starfile.c_str( ), input_template.c_str( ), start_position, end_position, num_threads);
    };
};
#endif // _src_programs_refine_template_dev_refine_template_dev_h