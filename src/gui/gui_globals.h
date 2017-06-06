
class MyOrthDrawEvent;
wxDECLARE_EVENT(MY_ORTH_DRAW_EVENT, MyOrthDrawEvent);

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


typedef void (wxEvtHandler::*MyOrthDrawEventFunction)(MyOrthDrawEvent &);
#define MyOrthDrawEventHandler(func) wxEVENT_HANDLER_CAST(MyOrthDrawEventFunction, func)

WX_DECLARE_OBJARRAY(wxColor, ArrayofColors);
extern ArrayofColors default_colormap;
