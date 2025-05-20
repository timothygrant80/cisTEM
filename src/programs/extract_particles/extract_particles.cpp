#include "../../core/core_headers.h"

class
        ExtractParticlesApp : public MyApp {

  public:
    bool DoCalculation( );
    void DoInteractiveUserInput( );

  private:
};

IMPLEMENT_APP(ExtractParticlesApp)

void ExtractParticlesApp::DoInteractiveUserInput( ) {

    UserInput* my_input = new UserInput("ExtractParticles", 0.0);

    wxString micrograph_filename   = my_input->GetFilenameFromUser("Input micrograph filename", "The input micrograph, in which we will look for particles", "micrograph.mrc", true);
    wxString coordinates_filename  = my_input->GetFilenameFromUser("Coordinates (PLT) filename", "The input particle coordinates, in Imagic-style PLT forlmat", "coos.plt", true);
    wxString output_stack_filename = my_input->GetFilenameFromUser("Filename for output stack of particles.", "A stack of particles will be written to disk", "particles.mrc", false);
    int      output_stack_box_size = my_input->GetIntFromUser("Box size for output candidate particle images (pixels)", "In pixels. Give 0 to skip writing particle images to disk.", "256", 0);

    delete my_input;

    my_current_job.Reset(4);
    my_current_job.ManualSetArguments("ttti", micrograph_filename.ToStdString( ).c_str( ),
                                      coordinates_filename.ToStdString( ).c_str( ),
                                      output_stack_filename.ToStdString( ).c_str( ),
                                      output_stack_box_size);
}

// override the do calculation method which will be what is actually run..

bool ExtractParticlesApp::DoCalculation( ) {

    ProgressBar*                  my_progress_bar;
    EmpiricalDistribution<double> my_dist;

    // Get the arguments for this job..
    wxString micrograph_filename   = my_current_job.arguments[0].ReturnStringArgument( );
    wxString coordinates_filename  = my_current_job.arguments[1].ReturnStringArgument( );
    wxString output_stack_filename = my_current_job.arguments[2].ReturnStringArgument( );
    int      output_stack_box_size = my_current_job.arguments[3].ReturnIntegerArgument( );

    // Open input files so we know dimensions
    MRCFile micrograph_file(micrograph_filename.ToStdString( ), false);
    MyDebugAssertTrue(micrograph_file.ReturnNumberOfSlices( ) == 1, "Input micrograph file should only contain one image for now");

    // Let's box particles out
    Image micrograph;
    Image box;
    box.Allocate(output_stack_box_size, output_stack_box_size, 1, true);
    micrograph.ReadSlice(&micrograph_file, 1);
    float            micrograph_mean = micrograph.ReturnAverageOfRealValues( );
    NumericTextFile* input_coos_file;
    input_coos_file             = new NumericTextFile(coordinates_filename, OPEN_TO_READ, 3);
    int     number_of_particles = input_coos_file->number_of_lines;
    MRCFile output_stack;
    output_stack.OpenFile(output_stack_filename.ToStdString( ), true);
    my_progress_bar = new ProgressBar(number_of_particles);
    float plt_x, plt_y;
    float my_x, my_y;
    float temp_array[3];
    for ( int counter = 0; counter < number_of_particles; counter++ ) {
        input_coos_file->ReadLine(temp_array);
        plt_x = temp_array[0];
        plt_y = temp_array[1];
        /*
		 * plt_x = (my_y + phys_addr_box_center_y) + 1.0;
		 * plt_y = (logical_x_dim - (phys_addr_box_center_x + my_x)) + 1.0;
		 *
		 * my_x = -1 * ((plt_y - 1.0)  - logical_x_dim + phys_addr_box_center_x)
		 * my_y = (plt_x - 1.0)  - phys_addr_box_center_y;
		 */
        my_x = (-1.0) * ((plt_y - 1.0) - micrograph.logical_x_dimension + micrograph.physical_address_of_box_center_x);
        my_y = (plt_x - 1.0) - micrograph.physical_address_of_box_center_y;
        micrograph.ClipInto(&box, micrograph_mean, false, 1.0, -int(my_x), -int(my_y), 0);
        box.WriteSlice(&output_stack, counter + 1);

        //
        my_progress_bar->Update(counter + 1);
    }
    delete my_progress_bar;
    wxPrintf("\nExtracted %i particles\n", number_of_particles);
    delete input_coos_file;

    return true;
}
