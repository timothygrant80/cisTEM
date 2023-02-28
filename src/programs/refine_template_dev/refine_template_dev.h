#ifndef _src_programs_refine_template_dev_refine_template_dev_h
#define _src_programs_refine_template_dev_refine_template_dev_h

class RefineTemplateArguments {
  public:
    std::string input_starfile;

    std::string output_starfile;

    std::string input_template;

    RefineTemplateArguments( ) {

        input_starfile = "input.star";

        output_starfile = "output.star";

        input_template = "input.mrc";
    };

    void recieve(RunArgument* arguments) {

        input_starfile = arguments[0].ReturnStringArgument( );

        output_starfile = arguments[1].ReturnStringArgument( );

        input_template = arguments[2].ReturnStringArgument( );
    };

    void userinput( ) {

        UserInput* my_input = new UserInput("RefineTemplateArguments", 1.00);

        input_starfile = my_input->GetFilenameFromUser("The starfile describing the matches that should be defined", "The starfile describing the matches that should be defined", "input.star", false);

        output_starfile = my_input->GetFilenameFromUser("The starfile describing the matfgfches that should be defined", "The starfile defgfscribing the matches that should be defined", "output.star", false);

        input_template = my_input->GetFilenameFromUser("The template to be refined", "The template to be refined", "input.mrc", false);

        delete my_input;
    };

    void setargument(RunJob& my_current_job) {

        my_current_job.ManualSetArguments("ttt", input_starfile.c_str( ), output_starfile.c_str( ), input_template.c_str( ));
    };
};
#endif // _src_programs_refine_template_dev_refine_template_dev_h