

void ConvertImageToBitmap(Image *input_image, wxBitmap *output_bitmap, bool auto_contrast = false);
void GetMultilineTextExtent	(wxDC *wanted_dc, const wxString & string, int &width, int &height);
void FillGroupComboBoxSlave( wxComboBox *GroupComboBox, bool include_all_images_group = true );
void FillParticlePositionsGroupComboBox(wxComboBox *GroupComboBox, bool include_all_particle_positions_group = true);

void AppendVolumeAssetsToComboBox(wxComboBox *GroupComboBox);
void AppendRefinementPackagesToComboBox(wxComboBox *GroupComboBox);
wxArrayString GetRecentProjectsFromSettings();
void AddProjectToRecentProjects(wxString project_to_add);

void RunSimpleFunctionInAnotherThread(wxWindow *parent_window, void (*function_to_run)(void));


class RunSimpleFunctionThread : public wxThread
{
	public:
	RunSimpleFunctionThread(wxWindow *parent, void (*wanted_function_to_run)(void)) : wxThread(wxTHREAD_DETACHED)
	{
		main_thread_pointer = parent;
		function_to_run = wanted_function_to_run;
	}

	protected:

	wxWindow *main_thread_pointer;
	void (*function_to_run)(void);
	 virtual ExitCode Entry();
};

void SetupDefaultColorMap();

void global_delete_scratch();
void global_delete_refine2d_scratch();
void global_delete_refine3d_scratch();
void global_delete_startup_scratch();
void global_delete_autorefine3d_scratch();
void global_delete_generate3d_scratch();

