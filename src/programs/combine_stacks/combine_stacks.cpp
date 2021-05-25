#include "../../core/core_headers.h"
#include <dirent.h>
#include <regex>

class
CombineStack : public MyApp
{

	public:

    void DoInteractiveUserInput();
	bool DoCalculation();
	
	private:
};

IMPLEMENT_APP(CombineStack)

// override the DoInteractiveUserInput

void CombineStack::DoInteractiveUserInput()
{
	UserInput *my_input = new UserInput("CombineStack", 1.0);

    std::string input_star_file	=		my_input->GetFilenameFromUser("Input Star File", "required star file", "starFile.star", true);  
	std::string output_filename	=		my_input->GetFilenameFromUser("Output image file name", "the combined result", "output.mrc", false );

	delete my_input;

	my_current_job.Reset(2);
	my_current_job.ManualSetArguments("tt", input_star_file.c_str(), output_filename.c_str());

}

bool CombineStack::DoCalculation()
{
    std::string input_star_file = my_current_job.arguments[0].ReturnStringArgument();
	std::string	output_filename = my_current_job.arguments[1].ReturnStringArgument();
    
    MRCFile output_file(output_filename,true);

    float input_pixel_size;
    int input_pixel_size_set = 0;

    BasicStarFileReader input_star_file_reader;
	wxString star_error_text;

	if (input_star_file_reader.ReadFile(input_star_file, &star_error_text) == false)
	{
        //print error
		SendErrorAndCrash(star_error_text);
	}

    int sumOfStacks = 0;
    int stack_x_size = 0;
    int stack_y_size = 0;
    int stack_number_of_images = 0;
    
    //start looping over star file 
    std::string delim1 = "@";
    long num_slices = 0;
    int numLines = input_star_file_reader.returnNumLines();

    for (int particle_counter = 0; particle_counter < numLines; particle_counter++) {
        
        std::string current_mrc_name = std::string((input_star_file_reader.ReturnImageName(particle_counter)).ToStdString());

        //parse image name 
        std::string slice = current_mrc_name.substr(0, current_mrc_name.find(delim1));
        current_mrc_name.erase(0, current_mrc_name.find(delim1) + delim1.length());
        std::string current_image_name = current_mrc_name.substr(0, current_mrc_name.length());

        //open relavent file, add relavent slice to the file
        ImageFile current_input_file(current_image_name,false);
        Image current_image;
	    input_pixel_size = current_input_file.ReturnPixelSize();
        input_pixel_size_set = 1;

        //slice to string 
        long slice_num = std::stol(slice, nullptr);
        current_image.ReadSlice(&current_input_file, slice_num);
        num_slices += 1;
		current_image.WriteSlice(&output_file, num_slices);
    }
    
    output_file.SetPixelSize(input_pixel_size);
	output_file.WriteHeader();
	
	return true;
}

