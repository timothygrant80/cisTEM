

void ConvertImageToBitmap(Image* input_image, wxBitmap* output_bitmap, bool auto_contrast = false, int client_x_size = INT_MAX, int client_y_size = INT_MAX, int img_x_start = 0, int img_y_start = 0, float scaling_factor = 0.0);
void GetMultilineTextExtent(wxDC* wanted_dc, const wxString& string, int& width, int& height);
void FillGroupComboBoxWorker(wxComboBox* GroupComboBox, bool include_all_images_group = true);
void FillParticlePositionsGroupComboBox(wxComboBox* GroupComboBox, bool include_all_particle_positions_group = true);

void          AppendVolumeAssetsToComboBox(wxComboBox* GroupComboBox);
void          AppendRefinementPackagesToComboBox(wxComboBox* GroupComboBox);
wxArrayString GetRecentProjectsFromSettings( );
void          AddProjectToRecentProjects(wxString project_to_add);

void RunSimpleFunctionInAnotherThread(wxWindow* parent_window, void (*function_to_run)(void));

class RunSimpleFunctionThread : public wxThread {
  public:
    RunSimpleFunctionThread(wxWindow* parent, void (*wanted_function_to_run)(void)) : wxThread(wxTHREAD_DETACHED) {
        main_thread_pointer = parent;
        function_to_run     = wanted_function_to_run;
    }

  protected:
    wxWindow* main_thread_pointer;
    void (*function_to_run)(void);
    virtual ExitCode Entry( );
};

void SetupDefaultColorMap( );
void SetupDefaultColorBar( );

void global_delete_scratch( );
void global_delete_refine2d_scratch( );
void global_delete_refine3d_scratch( );
void global_delete_startup_scratch( );
void global_delete_autorefine3d_scratch( );
void global_delete_generate3d_scratch( );
void global_delete_refinectf_scratch( );

inline wxColour GetColourBarValue(float current_value, float min_value, float max_value) {
    if ( current_value <= min_value ) {
        return default_colorbar[0];
    }
    else if ( current_value >= max_value ) {
        return default_colorbar[default_colorbar.GetCount( ) - 1];
    }
    else {

        float range        = (max_value - min_value) / float(default_colorbar.GetCount( ));
        int   bar_position = myroundint((current_value - min_value) / range);

        if ( bar_position < 0 )
            bar_position = 0;
        else if ( bar_position > default_colorbar.GetCount( ) - 1 )
            bar_position = default_colorbar.GetCount( );

        return default_colorbar[bar_position];
    }
}
