
class MyOrthDrawEvent;
wxDECLARE_EVENT(MY_ORTH_DRAW_EVENT, MyOrthDrawEvent);
wxDECLARE_EVENT(wxEVT_AUTOMASKERTHREAD_COMPLETED, wxThreadEvent);

class MyOrthDrawEvent: public wxCommandEvent
{
public:
	MyOrthDrawEvent(wxEventType commandType = MY_ORTH_DRAW_EVENT, int id = 0)
        		:  wxCommandEvent(commandType, id) { }

	// You *must* copy here the data to be transported
	MyOrthDrawEvent(const MyOrthDrawEvent& event)
        		:  wxCommandEvent(event) { this->SetImage(event.GetImage()); }

	// Required for sending with wxPostEvent()
	wxEvent* Clone() const { return new MyOrthDrawEvent(*this); }

	Image* GetImage() const { return m_image; }
	void SetImage( Image *image) { m_image = image; }

private:
	Image *m_image;
};

class OrthDrawerThread : public wxThread
{
	public:
	OrthDrawerThread(wxWindow *parent, wxArrayString wanted_filenames_of_volumes, wxString wanted_tab_name) : wxThread(wxTHREAD_DETACHED)
	{
		main_thread_pointer = parent;
		filenames_of_volumes = wanted_filenames_of_volumes;
		tab_name = wanted_tab_name;
	}

	protected:

	wxWindow *main_thread_pointer;
	wxArrayString filenames_of_volumes;
	wxString tab_name;

    virtual ExitCode Entry();
};

class AutoMaskerThread : public wxThread
{
	public:
	AutoMaskerThread(wxWindow *parent, wxArrayString wanted_input_files, wxArrayString wanted_output_files, float wanted_pixel_size, float wanted_mask_radius) : wxThread(wxTHREAD_DETACHED)
	{
		main_thread_pointer = parent;
		input_files = wanted_input_files;
		output_files = wanted_output_files;
		pixel_size = wanted_pixel_size;
		mask_radius = wanted_mask_radius;
	}

	protected:

	wxWindow *main_thread_pointer;
	wxArrayString input_files;
	wxArrayString output_files;
	float pixel_size;
	float mask_radius;

    virtual ExitCode Entry();
};


typedef void (wxEvtHandler::*MyOrthDrawEventFunction)(MyOrthDrawEvent &);
#define MyOrthDrawEventHandler(func) wxEVENT_HANDLER_CAST(MyOrthDrawEventFunction, func)

WX_DECLARE_OBJARRAY(wxColor, ArrayofColors);
extern ArrayofColors default_colormap;
