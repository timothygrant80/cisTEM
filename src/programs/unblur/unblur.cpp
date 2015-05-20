#include <wx/wx.h>
#include <wx/app.h>
#include <wx/cmdline.h>
#include <cstdio>
#include "wx/socket.h"

#include "../../core/core_headers.h"

#define SERVER_ID 100
#define SOCKET_ID 101

wxDEFINE_EVENT(wxEVT_COMMAND_MYTHREAD_COMPLETED, wxThreadEvent);
wxDEFINE_EVENT(wxEVT_COMMAND_MYTHREAD_UPDATE, wxThreadEvent);


class MyApp; // So CalculateThread class knows about it

// The workhorse / calculation thread

class CalculateThread : public wxThread
{
	public:
    	CalculateThread(MyApp *handler) : wxThread(wxTHREAD_DETACHED) { main_thread_pointer = handler; }
    	~CalculateThread();

	protected:

    	virtual ExitCode Entry();
    	MyApp *main_thread_pointer;
};



// The console APP class.. should just deal with events..

class
MyApp : public wxAppConsole
{

	public:
		virtual bool OnInit();

		wxSocketClient *m_sock;
		bool            m_busy;
		bool 			is_connected;

		void OnSocketEvent(wxSocketEvent& event);

	private:

		 CalculateThread *work_thread;

		 void OnThreadUpdate(wxThreadEvent&);
		 void OnThreadComplete(wxThreadEvent&);



};


IMPLEMENT_APP(MyApp)


bool MyApp::OnInit()
{
	// Bind the thread events

	Bind(wxEVT_COMMAND_MYTHREAD_UPDATE, &MyApp::OnThreadUpdate, this);
	Bind(wxEVT_COMMAND_MYTHREAD_COMPLETED, &MyApp::OnThreadComplete, this);


	// Socket stuff?


	// Run the processing thread

	work_thread = new CalculateThread(this);

	if ( work_thread->Run() != wxTHREAD_NO_ERROR )
	{
		printf("\n\nError: Can't create the workhorse thread!!\n\n");

		// Send the appropriate error msg back to the master socket..

		abort();
	}

	// should be done.

	/*
	m_sock = new wxSocketClient();
	wxIPV4address addr;

	// Setup the event handler and subscribe to most events
	m_sock->SetEventHandler(*this, SOCKET_ID);
	m_sock->SetNotify(wxSOCKET_CONNECTION_FLAG |wxSOCKET_INPUT_FLAG |wxSOCKET_LOST_FLAG);
	m_sock->Notify(true);
	m_busy = false;

	is_connected = false;

	this->Connect(SOCKET_ID, wxEVT_SOCKET, wxSocketEventHandler( MyApp::OnSocketEvent) );

	 addr.Hostname("localhost");
	 addr.Service(3000);

	 wxPrintf(_("\nTrying to connect (timeout = 10 sec) ...\n"));
	 m_sock->Connect(addr, false);
	 m_sock->WaitOnConnect(10);

	 if (m_sock->IsConnected())
	  {
	    wxPrintf(_("Succeeded - Connection established!\n\n"));
		is_connected = true;
		return true;
	  }
	  else
	  {
		m_sock->Close();
		wxPrintf(_("Failed ! Unable to connect\n"));
		return false;
	  }*/

	return true;
}

void MyApp::OnThreadUpdate(wxThreadEvent&)
{

}


void MyApp::OnThreadComplete(wxThreadEvent&)
{
	printf("compute thread finished!\n");

	// The compute thread is finished.. so we can now finish

	ExitMainLoop();

}


// Main execution in this thread..

wxThread::ExitCode CalculateThread::Entry()
{
    //while (!TestDestroy())
    //{
     //   // ... do a bit of work...
      //  wxQueueEvent(main_thread_pointer, new wxThreadEvent(wxEVT_COMMAND_MYTHREAD_UPDATE));
    //}

	printf ("I am a thread, I will !\n");








    // Finished.

    // signal the event handler that this thread is going to be destroyed
    // NOTE: here we assume that using the m_pHandler pointer is safe,
    //       (in this case this is assured by the MyFrame destructor)

	wxQueueEvent(main_thread_pointer, new wxThreadEvent(wxEVT_COMMAND_MYTHREAD_COMPLETED));

    return (wxThread::ExitCode)0;     // success
}

CalculateThread::~CalculateThread()
{
    //wxCriticalSectionLocker enter(m_pHandler->m_pThreadCS);
    // the thread is being destroyed; make sure not to leave dangling pointers around
    main_thread_pointer = NULL;
}

/*
void MyApp::OnSocketEvent(wxSocketEvent& event)
{
  wxString s = _("OnSocketEvent: ");

  char side[10];
  char system_command[500];

  if (event.GetSocketEvent() == wxSOCKET_INPUT)
  {
		// we need to package the bet to send into a stream of bytes..

	 	// byte 1 = home (0), draw(1) or away(2);
		// byte 2,3,4,5 = pounds to bet
		// byte 6 = pence to bet
		// byte 7,8,9,10 = expected_nominator
		// byte 11,12,13,14 = expected_denominator
	    // byte 15 = bet365 (0), Bwin (1), VC (2);

	  	int side_to_bet;
		int pounds_to_bet;
		int pence_to_bet;
		int nominator;
		int denominator;
		int site_to_bet;

		int temp_int;
		char *char_pointer = (char*) &temp_int;

		float money_to_bet;

		wxDateTime now = wxDateTime::Now();

	  	unsigned char c[15];
	  	m_sock->Read(&c, 15);

	  	side_to_bet = int(c[0]);

	  	char_pointer[0] = c[1];
	  	char_pointer[1] = c[2];
	  	char_pointer[2] = c[3];
	  	char_pointer[3] = c[4];

	  	pounds_to_bet = temp_int;

	  	pence_to_bet = int(c[5]);

	  	char_pointer[0] = c[6];
	  	char_pointer[1] = c[7];
	  	char_pointer[2] = c[8];
	  	char_pointer[3] = c[9];

	  	nominator = temp_int;

	  	char_pointer[0] = c[10];
	  	char_pointer[1] = c[11];
	  	char_pointer[2] = c[12];
	  	char_pointer[3] = c[13];

	  	denominator = temp_int;

	  	site_to_bet = int(c[14]);

	  	wxPrintf(wxT("  **   Bet received at "));
	  	wxPrintf(wxString::Format(wxT("%s   **\n"), now.Format().c_str()));

	  	wxPrintf(wxString::Format(_("Betting %c%d.%d on "), 0xA3, pounds_to_bet, pence_to_bet));

	  	money_to_bet = float(pounds_to_bet) + (float(pence_to_bet) / 100);

	  	if (side_to_bet == 0)
	  	{
	  		wxPrintf(wxT("HOME\n"));
	  		sprintf(side, "home");
	  	}
	  	else
	  	if (side_to_bet == 1)
	  	{
	  		wxPrintf(wxT("DRAW\n"));
	  		sprintf(side, "draw");
	  	}
	  	else
	  	if (side_to_bet == 2)
	  	{
	  		wxPrintf(wxT("AWAY\n"));
	  		sprintf(side, "away");
	  	}

	  	wxPrintf(wxString::Format((wxT("Expected Odds Are %d/%d\n")), nominator, denominator));

	  	// now call Andy's script appropriately...

	  	if (site_to_bet == 0)
	  	{
	  		wxPrintf(wxT("Betting at BET365\n\n"));
	  		sprintf(system_command, "\"Z:\\VM Scripts\\bet365.ahk\" %s %.2f", side, money_to_bet);
	  	}
	  	else
	  	if (site_to_bet == 1)
	  	{
	  		wxPrintf(wxT("Betting at BWIN\n\n"));
	  		sprintf(system_command, "\"Z:\\VM Scripts\\bwin.ahk\" %s %.2f", side, money_to_bet);
	  	}
	  	else
	  	if (site_to_bet == 2)
	  	{
	  		wxPrintf(wxT("Betting at VC\n\n"));
	  		sprintf(system_command, "\"Z:\\VM Scripts\\vc.ahk\" %s %.2f", side, money_to_bet);
	  	}

	  	system(system_command);

	  	//wxExecute(wxT(system_command));


	  	wxPrintf(wxT("Running the following script :-\n\n"));

  }
  else
  if (event.GetSocketEvent() == wxSOCKET_LOST)
  {
	  is_connected = false;
	  wxPrintf(wxT("Connection Lost!!\n\n"));
	  ExitMainLoop();

  }
}

*/

/*
class
MyApp : public wxAppConsole
{

	public:
		virtual bool OnInit();

		wxSocketClient *m_sock;
		bool            m_busy;
		bool 			is_connected;

		void OnSocketEvent(wxSocketEvent& event);


};


IMPLEMENT_APP(MyApp)


bool MyApp::OnInit()
{

	m_sock = new wxSocketClient(wxSOCKET_BLOCK);
	wxIPV4address addr;

	// Setup the event handler and subscribe to most events
	//m_sock->SetEventHandler(*this, SOCKET_ID);
	//m_sock->SetNotify(wxSOCKET_CONNECTION_FLAG |wxSOCKET_INPUT_FLAG |wxSOCKET_LOST_FLAG);
	//m_sock->Notify(true);
	//m_busy = false;

	is_connected = false;

	//this->Connect(SOCKET_ID, wxEVT_SOCKET, wxSocketEventHandler( MyApp::OnSocketEvent) );

	 addr.Hostname("localhost");
	 addr.Service(3000);

	 wxPrintf(_("\nTrying to connect (timeout = 10 sec) ...\n"));
	 m_sock->Connect(addr, false);
	 m_sock->WaitOnConnect(10);

	 if (m_sock->IsConnected())
	  {
	    wxPrintf(_("Succeeded - Connection established!\n\n"));
		is_connected = true;
		return true;
	  }
	  else
	  {
		m_sock->Close();
		wxPrintf(_("Failed ! Unable to connect\n"));
		return false;
	  }
}*/

