#include "../../core/core_headers.h"

// TODO : Switch to new parameter file format (star) properly and remove hacks
class
        SplitClassAveragesApp : public MyApp {
  public:
    bool DoCalculation( );
    void DoInteractiveUserInput( );

  private:
};

IMPLEMENT_APP(SplitClassAveragesApp)

// override the DoInteractiveUserInput

void SplitClassAveragesApp::DoInteractiveUserInput( ) {
    wxString input_particle_images;
    wxString input_parameter_file;
    wxString ouput_class_averages;
    int      number_of_classes = 10;
    int      images_per_class  = 10;
    int      wanted_class_number;
    float    pixel_size;
    float    microscope_voltage;
    float    microscope_cs;
    float    amplitude_contrast;

    UserInput* my_input = new UserInput("Refine2D", 1.02);

    input_particle_images = my_input->GetFilenameFromUser("Input particle images", "The input image stack, containing the experimental particle images", "my_image_stack.mrc", true);
    input_parameter_file  = my_input->GetFilenameFromUser("Input Frealign parameter filename", "The input parameter file, containing your particle alignment parameters", "my_parameters.par", true);
    ouput_class_averages  = my_input->GetFilenameFromUser("Output class averages", "The refined 2D class averages", "my_refined_classes.mrc", false);
    wanted_class_number   = my_input->GetIntFromUser("Original Class number wanted", "", "1", 1);
    number_of_classes     = my_input->GetIntFromUser("Number of classes wanted", "", "100", 1);
    images_per_class      = my_input->GetIntFromUser("Number of images per class wanted", "", "25", 1);
    pixel_size            = my_input->GetFloatFromUser("Pixel size", "", "1.0");
    microscope_voltage    = my_input->GetFloatFromUser("Microscope voltage (kV)", "", "300.0");
    microscope_cs         = my_input->GetFloatFromUser("Microscope Cs (mm)", "", "2.7");
    amplitude_contrast    = my_input->GetFloatFromUser("Amplitude Contrast", "", "0.07");

    delete my_input;

    int current_class = 0;
    my_current_job.Reset(10);
    my_current_job.ManualSetArguments("tttiiiffff", input_particle_images.ToUTF8( ).data( ),
                                      input_parameter_file.ToUTF8( ).data( ),
                                      ouput_class_averages.ToUTF8( ).data( ),
                                      wanted_class_number,
                                      number_of_classes,
                                      images_per_class,
                                      pixel_size,
                                      microscope_voltage,
                                      microscope_cs,
                                      amplitude_contrast);
}

// override the do calculation method which will be what is actually run..

bool SplitClassAveragesApp::DoCalculation( ) {
    wxString input_particle_images = my_current_job.arguments[0].ReturnStringArgument( );
    wxString input_parameter_file  = my_current_job.arguments[1].ReturnStringArgument( );
    wxString ouput_class_averages  = my_current_job.arguments[2].ReturnStringArgument( );
    int      wanted_class_number   = my_current_job.arguments[3].ReturnIntegerArgument( );
    int      number_of_classes     = my_current_job.arguments[4].ReturnIntegerArgument( );
    int      images_per_class      = my_current_job.arguments[5].ReturnIntegerArgument( );
    float    pixel_size            = my_current_job.arguments[6].ReturnFloatArgument( );
    float    microscope_voltage    = my_current_job.arguments[7].ReturnFloatArgument( );
    float    microscope_cs         = my_current_job.arguments[8].ReturnFloatArgument( );
    float    amplitude_contrast    = my_current_job.arguments[9].ReturnFloatArgument( );

    int line_counter;
    int class_counter;
    int image_counter;
    int random_image;
    int pixel_counter;

    float rotated_x;
    float rotated_y;

    float input_parameters[17];

    ZeroFloatArray(input_parameters, 17);

    if ( ! DoesFileExist(input_parameter_file) ) {
        SendError(wxString::Format("Error: Input parameter file %s not found\n", input_parameter_file));
        exit(-1);
    }
    if ( ! DoesFileExist(input_particle_images) ) {
        SendError(wxString::Format("Error: Input particle stack %s not found\n", input_particle_images));
        exit(-1);
    }
    MRCFile               input_stack(input_particle_images.ToStdString( ), false);
    FrealignParameterFile input_par_file(input_parameter_file, OPEN_TO_READ);

    Image input_image;
    Image sum_image;
    Image ctf_sum_image;
    Image ctf_input_image;
    Image rotated_image;

    AnglesAndShifts rotation_angle;
    CTF             current_ctf;

    ArrayOfcisTEMParameterLines class_members;
    cisTEMParameterLine         temp_line;

    input_image.Allocate(input_stack.ReturnXSize( ), input_stack.ReturnYSize( ), 1);
    sum_image.Allocate(input_stack.ReturnXSize( ), input_stack.ReturnYSize( ), 1);
    ctf_sum_image.Allocate(input_stack.ReturnXSize( ), input_stack.ReturnYSize( ), 1);
    ctf_input_image.Allocate(input_stack.ReturnXSize( ), input_stack.ReturnYSize( ), 1);
    rotated_image.Allocate(input_stack.ReturnXSize( ), input_stack.ReturnYSize( ), 1);

    // get all the members of the selected class

    input_par_file.ReadFile( );

    for ( line_counter = 0; line_counter <= input_par_file.number_of_lines; line_counter++ ) {
        input_par_file.ReadLine(input_parameters);

        if ( int(input_parameters[7]) == wanted_class_number ) {
            temp_line.position_in_stack = input_parameters[0];
            temp_line.psi               = input_parameters[1];
            temp_line.x_shift           = input_parameters[4];
            temp_line.y_shift           = input_parameters[5];
            temp_line.defocus_1         = input_parameters[8];
            temp_line.defocus_2         = input_parameters[9];
            temp_line.defocus_angle     = input_parameters[10];
            temp_line.phase_shift       = input_parameters[11];

            class_members.Add(temp_line);
        }
    }

    wxPrintf("\nClass %i has %li members\n\n", wanted_class_number, class_members.GetCount( ));

    for ( class_counter = 0; class_counter < number_of_classes; class_counter++ ) {

        sum_image.SetToConstant(0.0f);
        sum_image.is_in_real_space     = false;
        rotated_image.is_in_real_space = false;

        ctf_sum_image.SetToConstant(0.0f);
        ctf_sum_image.is_in_real_space = false;

        for ( image_counter = 0; image_counter < images_per_class; image_counter++ ) {
            random_image = myroundint(fabsf(global_random_number_generator.GetUniformRandom( ) * (class_members.GetCount( ) - 1)));
            //random_image = image_counter;// + 1;
            //wxPrintf("random = %i\n", random_image);
            input_image.ReadSlice(&input_stack, class_members[random_image].position_in_stack);
            current_ctf.Init(microscope_voltage, microscope_cs, amplitude_contrast, class_members[random_image].defocus_1, class_members[random_image].defocus_2, class_members[random_image].defocus_angle, pixel_size, class_members[random_image].phase_shift);
            ctf_input_image.CalculateCTFImage(current_ctf);

            input_image.PhaseShift(class_members[random_image].x_shift / pixel_size, class_members[random_image].y_shift / pixel_size);
            rotation_angle.Init(0.0, 0.0, -class_members[random_image].psi, 0, 0);

            input_image.ForwardFFT( );
            input_image.SwapRealSpaceQuadrants( );
            input_image.MultiplyPixelWiseReal(ctf_input_image);

            input_image.RotateFourier2D(rotated_image, rotation_angle);

            sum_image.AddImage(&rotated_image);

            ctf_input_image.MultiplyPixelWiseReal(ctf_input_image);
            ctf_input_image.object_is_centred_in_box = false;
            ctf_input_image.RotateFourier2D(rotated_image, rotation_angle);
            ctf_sum_image.AddImage(&rotated_image);
        }

        ctf_sum_image.QuickAndDirtyWriteSlice("/tmp/ctf_sum.mrc", 1);
        for ( pixel_counter = 0; pixel_counter < sum_image.real_memory_allocated / 2; pixel_counter++ ) {
            if ( abs(ctf_sum_image.complex_values[pixel_counter]) != 0.0f )
                sum_image.complex_values[pixel_counter] /= (abs(ctf_sum_image.complex_values[pixel_counter]) + images_per_class / 2);
        }

        sum_image.SwapRealSpaceQuadrants( );
        sum_image.CosineMask(0.45, 0.1);
        sum_image.BackwardFFT( );
        sum_image.QuickAndDirtyWriteSlice(ouput_class_averages.ToStdString( ), class_counter + 1);
    }

    return true;
}
