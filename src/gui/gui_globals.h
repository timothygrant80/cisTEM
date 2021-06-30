class ReturnProcessedImageEvent;
class ReturnSharpeningResultsEvent;
wxDECLARE_EVENT(RETURN_PROCESSED_IMAGE_EVT, ReturnProcessedImageEvent);
wxDECLARE_EVENT(RETURN_SHARPENING_RESULTS_EVT, ReturnSharpeningResultsEvent);
wxDECLARE_EVENT(wxEVT_GENERATEMASKTHREAD_COMPLETED, wxThreadEvent);
wxDECLARE_EVENT(wxEVT_AUTOMASKERTHREAD_COMPLETED, wxThreadEvent);
wxDECLARE_EVENT(wxEVT_MULTIPLY3DMASKTHREAD_COMPLETED, wxThreadEvent);
wxDECLARE_EVENT(wxEVT_WRITECLASSIFICATIONSTARFILETHREAD_COMPLETED, wxThreadEvent);

class ReturnProcessedImageEvent : public wxCommandEvent
{
public:
	ReturnProcessedImageEvent(wxEventType commandType = RETURN_PROCESSED_IMAGE_EVT, int id = 0)
		: wxCommandEvent(commandType, id) {}

	// You *must* copy here the data to be transported
	ReturnProcessedImageEvent(const ReturnProcessedImageEvent &event)
		: wxCommandEvent(event) { this->SetImage(event.GetImage()); }

	// Required for sending with wxPostEvent()
	wxEvent *Clone() const { return new ReturnProcessedImageEvent(*this); }

	Image *GetImage() const { return m_image; }
	void SetImage(Image *image) { m_image = image; }

private:
	Image *m_image;
};

typedef void (wxEvtHandler::*ReturnProcessedImageEventFunction)(ReturnProcessedImageEvent &);
#define ReturnProcessedImageEventHandler(func) wxEVENT_HANDLER_CAST(ReturnProcessedImageEventFunction, func)

class ReturnSharpeningResultsEvent : public wxCommandEvent
{
public:
	ReturnSharpeningResultsEvent(wxEventType commandType = RETURN_SHARPENING_RESULTS_EVT, int id = 0) : wxCommandEvent(commandType, id)
	{
		m_sharpened_image = NULL;
		m_original_orth_image = NULL;
		m_sharpened_orth_image = NULL;
		m_original_curve = NULL;
		m_sharpened_curve = NULL;
	}

	// You *must* copy here the data to be transported
	ReturnSharpeningResultsEvent(const ReturnSharpeningResultsEvent &event) : wxCommandEvent(event)
	{
		this->SetSharpenedImage(event.GetSharpenedImage());
		this->SetOriginalOrthImage(event.GetOriginalOrthImage());
		this->SetOriginalCurve(event.GetOriginalCurve());
		this->SetSharpenedCurve(event.GetSharpenedCurve());
	}

	// Required for sending with wxPostEvent()
	wxEvent *Clone() const { return new ReturnSharpeningResultsEvent(*this); }

	Image *GetSharpenedImage() const { return m_sharpened_image; }
	Image *GetOriginalOrthImage() const { return m_original_orth_image; }
	Image *GetSharpenedOrthImage() const { return m_sharpened_orth_image; }
	Curve *GetOriginalCurve() const { return m_original_curve; }
	Curve *GetSharpenedCurve() const { return m_sharpened_curve; }

	void SetSharpenedImage(Image *image) { m_sharpened_image = image; }
	void SetOriginalOrthImage(Image *image) { m_original_orth_image = image; }
	void SetSharpenedOrthImage(Image *image) { m_sharpened_orth_image = image; }
	void SetOriginalCurve(Curve *curve) { m_original_curve = curve; }
	void SetSharpenedCurve(Curve *curve) { m_sharpened_curve = curve; }

private:
	Image *m_sharpened_image;
	Image *m_original_orth_image;
	Image *m_sharpened_orth_image;
	Curve *m_original_curve;
	Curve *m_sharpened_curve;
};

typedef void (wxEvtHandler::*ReturnSharpeningResultsEventFunction)(ReturnSharpeningResultsEvent &);
#define ReturnSharpeningResultsEventHandler(func) wxEVENT_HANDLER_CAST(ReturnSharpeningResultsEventFunction, func)

class OrthDrawerThread : public wxThread
{
public:
	OrthDrawerThread(wxWindow *parent, wxArrayString wanted_filenames_of_volumes, wxString wanted_tab_name, float wanted_scale_factor = 1.0f, float wanted_mask_radius_in_pixels = 0.0f, int wanted_thread_id = -1) : wxThread(wxTHREAD_DETACHED)
	{
		main_thread_pointer = parent;
		filenames_of_volumes = wanted_filenames_of_volumes;
		tab_name = wanted_tab_name;
		scale_factor = wanted_scale_factor;
		mask_radius_in_pixels = wanted_mask_radius_in_pixels;
		thread_id = wanted_thread_id;
	}

protected:
	wxWindow *main_thread_pointer;
	wxArrayString filenames_of_volumes;
	wxString tab_name;
	float scale_factor;
	float mask_radius_in_pixels;
	int thread_id;

	virtual ExitCode Entry();
};

//WTW find where the mask thread does its work and then change it to accomidate user supplied mask
class GenerateMaskThread : public wxThread
{
public:
	GenerateMaskThread(wxWindow *parent, wxString half_map_1, wxString half_map_2, wxString mask_image_name, float wanted_pixel_size, float wanted_outer_mask_radius, bool active_should_mask, wxString active_mask_filename, int wanted_thread_id = -1) : wxThread(wxTHREAD_DETACHED)
	{
		main_thread_pointer = parent;
		half_map_1 = half_map_1;
		half_map_2 = half_map_2;
		mask_image_name = mask_image_name;
		pixel_size = wanted_pixel_size;
		outer_mask_radius = wanted_outer_mask_radius;
		thread_id = wanted_thread_id;
		active_should_mask = active_should_mask;
		active_mask_filename = active_mask_filename;
	}

protected:
	wxWindow *main_thread_pointer;
	wxString half_map_1;
	wxString half_map_2;
	wxString mask_image_name;
	float pixel_size;
	float outer_mask_radius;
	int thread_id;
	bool acitve_should_mask;
	wxString active_mask_filename;

	virtual ExitCode Entry();
};

class AutoMaskerThread : public wxThread
{
public:
	AutoMaskerThread(wxWindow *parent, wxArrayString wanted_input_files, wxArrayString wanted_output_files, float wanted_pixel_size, float wanted_mask_radius, int wanted_thread_id = -1, float wanted_max_resolution = -1) : wxThread(wxTHREAD_DETACHED)
	{
		main_thread_pointer = parent;
		input_files = wanted_input_files;
		output_files = wanted_output_files;
		pixel_size = wanted_pixel_size;
		mask_radius = wanted_mask_radius;
		thread_id = wanted_thread_id;
		max_resolution = wanted_max_resolution;
		if (max_resolution < pixel_size * 2.0f)
			max_resolution = pixel_size * 2.0f;
	}

protected:
	wxWindow *main_thread_pointer;
	wxArrayString input_files;
	wxArrayString output_files;
	float pixel_size;
	float mask_radius;
	int thread_id;
	float max_resolution;

	virtual ExitCode Entry();
};

class Multiply3DMaskerThread : public wxThread
{
public:
	Multiply3DMaskerThread(wxWindow *parent, wxArrayString wanted_input_files, wxArrayString wanted_output_files, wxString wanted_mask_filename, float wanted_cosine_edge_width, float wanted_weight_outside_mask, float wanted_low_pass_filter_radius, float wanted_pixel_size, int wanted_thread_id = -1) : wxThread(wxTHREAD_DETACHED)
	{
		main_thread_pointer = parent;
		input_files = wanted_input_files;
		output_files = wanted_output_files;
		mask_filename = wanted_mask_filename;
		cosine_edge_width = wanted_cosine_edge_width;
		weight_outside_mask = wanted_weight_outside_mask;
		low_pass_filter_radius = wanted_low_pass_filter_radius;
		pixel_size = wanted_pixel_size;
		thread_id = wanted_thread_id;
	}

protected:
	wxWindow *main_thread_pointer;
	wxArrayString input_files;
	wxArrayString output_files;
	wxString mask_filename;
	int thread_id;

	float cosine_edge_width;
	float weight_outside_mask;
	float low_pass_filter_radius;
	float pixel_size;

	virtual ExitCode Entry();
};

class WriteClassificationStarFileThread : public wxThread
{
public:
	WriteClassificationStarFileThread(wxWindow *parent, wxString wanted_base_filename, Classification *wanted_classification, RefinementPackage *wanted_active_refinement_package) : wxThread(wxTHREAD_DETACHED)
	{
		main_thread_pointer = parent;
		current_classification = wanted_classification;
		active_refinement_package = wanted_active_refinement_package;
		base_filename = wanted_base_filename;
	}

protected:
	Classification *current_classification;
	RefinementPackage *active_refinement_package;
	wxWindow *main_thread_pointer;
	wxString base_filename;
	virtual ExitCode Entry();
};

WX_DECLARE_OBJARRAY(wxColor, ArrayofColors);

extern ArrayofColors default_colormap;
extern ArrayofColors default_colorbar;
