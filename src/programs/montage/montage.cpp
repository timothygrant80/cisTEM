#include "../../core/core_headers.h"

class
        MontageApp : public MyApp {

  public:
    bool DoCalculation( );
    void DoInteractiveUserInput( );

  private:
};

IMPLEMENT_APP(MontageApp)

void MontageApp::DoInteractiveUserInput( ) {

    UserInput* my_input = new UserInput("Montage", 1.0);

    std::string input_stack_filename    = my_input->GetFilenameFromUser("Input stack file name", "Filename of input stack.", "input_stack.mrc", true);
    std::string output_montage_filename = my_input->GetFilenameFromUser("Output montage file name", "Filename of output montage.", "montage.mrc", false);
    int         overlap                 = my_input->GetIntFromUser("Overlap", "Overlap in pixels between images in montage", "5", 0);
    int         num_pieces_x            = my_input->GetIntFromUser("Number of pieces in X", "Number of images to be stitched together in the X dimension", "5", 1);
    int         num_pieces_y            = my_input->GetIntFromUser("Number of pieces in Y", "Number of images to be stitched together in the Y dimension", "5", 1);

    delete my_input;

    my_current_job.Reset(5);
    my_current_job.ManualSetArguments("ttiii", input_stack_filename.c_str( ), output_montage_filename.c_str( ), overlap, num_pieces_x, num_pieces_y);
}

bool MontageApp::DoCalculation( ) {
    //const wxString input_stack_filename = "solvents.mrc";
    //const wxString output_montage_filename = "montage.mrc";
    //const int overlap = 5;
    //const int montage_num_pieces_x = 5;
    //const int montage_num_pieces_y = 5;

    std::string input_stack_filename    = my_current_job.arguments[0].ReturnStringArgument( );
    std::string output_montage_filename = my_current_job.arguments[1].ReturnStringArgument( );
    int         overlap                 = my_current_job.arguments[2].ReturnIntegerArgument( );
    int         montage_num_pieces_x    = my_current_job.arguments[3].ReturnIntegerArgument( );
    int         montage_num_pieces_y    = my_current_job.arguments[4].ReturnIntegerArgument( );

    Image montage;
    int   montage_dim_x;
    int   montage_dim_y;

    Image input_image;

    MRCFile input_stack(input_stack_filename);
    MRCFile output_montage(output_montage_filename, true);
    wxPrintf("Input stack has %i images of %i by %i pixels\n", input_stack.ReturnNumberOfSlices( ), input_stack.ReturnXSize( ), input_stack.ReturnYSize( ));

    MyDebugAssertTrue(montage_num_pieces_x * montage_num_pieces_y == input_stack.ReturnNumberOfSlices( ), "Number of input images incompatible with %i x %i montage\n", montage_num_pieces_x, montage_num_pieces_y);

    montage_dim_x = montage_num_pieces_x * input_stack.ReturnXSize( ) - ((montage_num_pieces_x - 1) * overlap);
    montage_dim_y = montage_num_pieces_y * input_stack.ReturnYSize( ) - ((montage_num_pieces_y - 1) * overlap);

    wxPrintf("Output montage will be %i x %i pixels\n", montage_dim_x, montage_dim_y);

    montage.Allocate(montage_dim_x, montage_dim_y, true);
    montage.SetToConstant(0.0);

    //
    int   image_counter_x;
    int   image_counter_y;
    int   j_input;
    int   i_input;
    long  counter_input;
    long  counter_output;
    int   i_start_output;
    int   j_start_output;
    float attenuation_x;
    float attenuation_y;

    int image_counter = 0;
    for ( image_counter_y = 0; image_counter_y < montage_num_pieces_y; image_counter_y++ ) {

        j_start_output = image_counter_y * (input_stack.ReturnYSize( ) - overlap);

        for ( image_counter_x = 0; image_counter_x < montage_num_pieces_x; image_counter_x++ ) {

            i_start_output = image_counter_x * (input_stack.ReturnXSize( ) - overlap);

            image_counter++;
            input_image.ReadSlice(&input_stack, image_counter);

            // Loop over the input image
            counter_input = 0;
            for ( j_input = 0; j_input < input_image.logical_y_dimension; j_input++ ) {
                attenuation_y = 1.0;

                if ( j_input < overlap ) {
                    if ( image_counter_y > 0 )
                        attenuation_y = 1.0 / float(overlap + 1) * (j_input + 1);
                }
                else if ( j_input >= input_image.logical_y_dimension - overlap ) {
                    if ( image_counter_y < montage_num_pieces_y - 1 )
                        attenuation_y = 1.0 / float(overlap + 1) * (input_image.logical_y_dimension - j_input);
                }

                for ( i_input = 0; i_input < input_image.logical_x_dimension; i_input++ ) {
                    attenuation_x = 1.0;

                    if ( i_input < overlap ) {
                        if ( image_counter_x > 0 )
                            attenuation_x = 1.0 / float(overlap + 1) * (i_input + 1);
                    }
                    else if ( i_input >= input_image.logical_x_dimension - overlap ) {
                        if ( image_counter_x < montage_num_pieces_x - 1 )
                            attenuation_x = 1.0 / float(overlap + 1) * (input_image.logical_x_dimension - i_input);
                    }

                    montage.real_values[montage.ReturnReal1DAddressFromPhysicalCoord(i_start_output + i_input, j_start_output + j_input, 0)] += input_image.real_values[counter_input] * attenuation_x * attenuation_y;
                    counter_input++;
                }
                counter_input += input_image.padding_jump_value;
            }
        }
    }

    montage.WriteSlice(&output_montage, 1);

    return true;
}
