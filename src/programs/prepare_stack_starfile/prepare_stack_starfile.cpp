#include "../../core/core_headers.h"

class
        MakeParticleStack : public MyApp {
  public:
    bool DoCalculation( );
    void DoInteractiveUserInput( );

  private:
};

IMPLEMENT_APP(MakeParticleStack)

// override the DoInteractiveUserInput

void MakeParticleStack::DoInteractiveUserInput( ) {

    wxString input_starfile_filename;
    wxString output_starfile_filename;
    wxString output_mrc_filename;
    int      box_size;
    int      start_position = -1;
    int      end_position   = -1;

    UserInput* my_input      = new UserInput("MakeParticleStack", 1.0);
    input_starfile_filename  = my_input->GetFilenameFromUser("Input starfile", "The starfile containing particle information", "input.star", true);
    box_size                 = my_input->GetIntFromUser("Box size", "Size of the box to cut out", "128", 1);
    output_starfile_filename = my_input->GetFilenameFromUser("Output starfile", "The starfile containing particle information", "output.star", false);
    output_mrc_filename      = my_input->GetFilenameFromUser("Output stack", "The stack containing the particles", "output.mrc", false);
    delete my_input;

    //	my_current_job.Reset(14);
    my_current_job.ManualSetArguments("tittii",
                                      input_starfile_filename.ToUTF8( ).data( ),
                                      box_size,
                                      output_starfile_filename.ToUTF8( ).data( ),
                                      output_mrc_filename.ToUTF8( ).data( ),
                                      start_position,
                                      end_position);
}

// override the do calculation method which will be what is actually run..

bool MakeParticleStack::DoCalculation( ) {

    wxDateTime start_time = wxDateTime::Now( );

    wxString input_starfile_filename  = my_current_job.arguments[0].ReturnStringArgument( );
    int      box_size                 = my_current_job.arguments[1].ReturnIntegerArgument( );
    wxString output_starfile_filename = my_current_job.arguments[2].ReturnStringArgument( );
    wxString output_mrc_filename      = my_current_job.arguments[3].ReturnStringArgument( );
    int      start_position           = my_current_job.arguments[4].ReturnIntegerArgument( );
    int      end_position             = my_current_job.arguments[5].ReturnIntegerArgument( );

    cisTEMParameterLine parameters;
    cisTEMParameters    star_file;
    Image               cut_particle;
    wxFileName          output_stack_filename = output_mrc_filename;
    Image               current_image;
    ImageFile           current_image_file;
    std::string         current_image_filename;
    int                 current_x_pos;
    int                 current_y_pos;
    float               average_value_at_edges;

    MRCFile output_stack(output_stack_filename.GetFullPath( ).ToStdString( ), true);

    star_file.ReadFromcisTEMStarFile(input_starfile_filename);

    cut_particle.Allocate(box_size, box_size, 1);

    if ( start_position < 1 )
        start_position = 1;
    if ( end_position > star_file.all_parameters.GetCount( ) or end_position < 1 )
        end_position = star_file.all_parameters.GetCount( );

    long position_in_stack = 1;
    for ( long match_id = start_position - 1; match_id < end_position; match_id++ ) {
        if ( current_image_filename.compare(star_file.all_parameters[match_id].original_image_filename.ToStdString( )) != 0 ) {
            current_image_file.OpenFile(star_file.all_parameters[match_id].original_image_filename.ToStdString( ), false);
            current_image_filename = star_file.all_parameters[match_id].original_image_filename.ToStdString( );
            current_image.ReadSlice(&current_image_file, 1);
            current_image.ReplaceOutliersWithMean(6);
            average_value_at_edges = current_image.ReturnAverageOfRealValuesOnEdges( );
        }
        current_x_pos = myround(star_file.all_parameters[match_id].original_x_position / star_file.all_parameters[match_id].pixel_size) - current_image.physical_address_of_box_center_x;
        current_y_pos = myround(star_file.all_parameters[match_id].original_y_position / star_file.all_parameters[match_id].pixel_size) - current_image.physical_address_of_box_center_y;

        current_image.ClipInto(&cut_particle, average_value_at_edges, false, 1.0, current_x_pos, current_y_pos, 0);
        cut_particle.ZeroFloatAndNormalize( );

        cut_particle.WriteSlice(&output_stack, position_in_stack);
        star_file.all_parameters[match_id].position_in_stack = position_in_stack;
        star_file.all_parameters[match_id].logp              = 0.0;
        star_file.all_parameters[match_id].occupancy         = 1.0f;
        star_file.all_parameters[match_id].sigma             = 10.0f;
        star_file.all_parameters[match_id].logp              = 5000.0f;
        star_file.all_parameters[match_id].score             = 50.0f;
        star_file.all_parameters[match_id].image_is_active   = 1;
        star_file.all_parameters[match_id].stack_filename    = output_stack_filename.GetFullPath( );
        position_in_stack++;
    }

    star_file.parameters_to_write.SetAllToTrue( );
    star_file.WriteTocisTEMStarFile(output_starfile_filename);
    return true;
}
