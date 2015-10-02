#include "../core/core_headers.h"
#include "../core/gui_core_headers.h"
#include "../core/socket_codes.h"

#define SERVER_ID 100
#define SOCKET_ID 101

extern MyMovieAssetPanel *movie_asset_panel;
extern MyAlignMoviesPanel *align_movies_panel;
extern MyRunProfilesPanel *run_profiles_panel;

MyMainFrame::MyMainFrame( wxWindow* parent )
:
MainFrame( parent )
{
	tree_root = AssetTree->AddRoot("Assets");

	// Add Movies..
	movie_branch = AssetTree->AppendItem(tree_root, wxString("Movies (0)"));

	socket_server = NULL;

	SetupServer();


}

MyMainFrame::~MyMainFrame()
{
	if (socket_server != NULL)
	{
		socket_server->Destroy();
	}
}
void MyMainFrame::SetupServer()
{
	wxIPV4address my_address;
	wxIPV4address buffer_address;

	for (short int current_port = START_PORT; current_port <= END_PORT; current_port++)
	{

		if (current_port == END_PORT)
		{
			wxPrintf("JOB CONTROL : Could not find a valid port !\n\n");
			abort();
		}

		my_port = current_port;
		my_address.Service(my_port);

		socket_server = new wxSocketServer(my_address);

		if (	socket_server->Ok())
		{
			  // setup events for the socket server..

		   	  socket_server->SetEventHandler(*this, SERVER_ID);
		   	  socket_server->SetNotify(wxSOCKET_CONNECTION_FLAG);
		  	  socket_server->Notify(true);

		  	  this->Connect(SERVER_ID, wxEVT_SOCKET, wxSocketEventHandler( MyMainFrame::OnServerEvent) );

			  buffer_address.Hostname(wxGetFullHostName()); // hopefully get my ip
			  my_ip_address = buffer_address.IPAddress();
			  my_port_string = wxString::Format("%hi", my_port);


			  break;
		}
		else socket_server->Destroy();
	}

}


void MyMainFrame::RecalculateAssetBrowser(void)
{

	wxTreeItemId current_group_branch;

	unsigned long group_counter;
	unsigned long asset_counter;

	long current_group_size;

	bool movies_expanded;

	// need to know the old layout..

	movies_expanded = AssetTree->IsExpanded(movie_branch);


	AssetTree->Freeze();
	AssetTree->DeleteAllItems();

	tree_root = AssetTree->AddRoot("Assets");

	// Add Movies..
	movie_branch = AssetTree->AppendItem(tree_root, wxString("Movies (") + wxString::Format(wxT("%li"), movie_asset_panel->ReturnNumberOfAssets()) + wxString(")"));

		// Movie Groups

		for (group_counter = 0; group_counter < movie_asset_panel->ReturnNumberOfGroups(); group_counter++)
		{
			current_group_size = movie_asset_panel->ReturnGroupSize(group_counter);

			if (current_group_size > 1)
			{
				current_group_branch = AssetTree->AppendItem (movie_branch, movie_asset_panel->ReturnGroupName(group_counter) + wxString(" (")+ wxString::Format(wxT("%li"), current_group_size) + wxString(")"));

				// add the movies..

				for (asset_counter = 0; asset_counter < current_group_size; asset_counter++)
				{
					AssetTree->AppendItem(current_group_branch, movie_asset_panel->ReturnAssetShortFilename(movie_asset_panel->ReturnGroupMember(group_counter, asset_counter)));
				}

			}
		}

	//Images_Branch = main_frame->AssetTree->AppendItem(Tree_Root, "Images (0)");

	// If they were expanded, expand them

	AssetTree->SetItemBold(movie_branch);

	if (movies_expanded == true) AssetTree->Expand(movie_branch);

	AssetTree->Thaw();

}


void MyMainFrame::OnCollapseAll( wxCommandEvent& event )
{
	AssetTree->CollapseAll();
}

void MyMainFrame::OnMenuBookChange( wxListbookEvent& event )
{
	// redo groups..

	align_movies_panel->FillGroupComboBox();

}

// SOCKETS

void MyMainFrame::OnSocketEvent(wxSocketEvent& event)
{
	MyDebugPrint("MyMainFrame::OnSocketEvent - This should not happen!");
}

void MyMainFrame::OnServerEvent(wxSocketEvent& event)
{
	  SETUP_SOCKET_CODES

	  long current_job;
	  wxString s = _("OnServerEvent: ");
	  wxSocketBase *sock = NULL;

	  switch(event.GetSocketEvent())
	  {
	    case wxSOCKET_CONNECTION : s.Append(_("wxSOCKET_CONNECTION\n")); break;
	    default                  : s.Append(_("Unexpected event !\n")); break;
	  }

	  //MyDebugPrint(s);

      // Accept new connection if there is one in the pending
      // connections queue, else exit. We use Accept(false) for
      // non-blocking accept (although if we got here, there
      // should ALWAYS be a pending connection).

      sock = socket_server->Accept(false);
	  sock->SetFlags(wxSOCKET_WAITALL);//|wxSOCKET_BLOCK);

	  // request identification..
	  //MyDebugPrint(" Requesting identification...");
   	  sock->WriteMsg(socket_please_identify, SOCKET_CODE_SIZE);
   	  //MyDebugPrint(" Waiting for reply...");
  	  sock->WaitForRead(5);

      if (sock->IsData() == true)
      {

    	  sock->ReadMsg(&socket_input_buffer, SOCKET_CODE_SIZE);

    	  // does this correspond to one of our jobs?

    	  current_job = job_controller.ReturnJobNumberFromJobCode(socket_input_buffer);

  	      if (current_job == -1)
  	      {
  	    	  MyDebugPrint(" Unknown JOB ID - Closing Connection\n");

  	    	  // incorrect identification - close the connection..
	    	  sock->Destroy();
	    	  sock = NULL;
	      }
	      else
	      {
	    	  MyDebugPrint("Connection from Job #%li", current_job);

	    	  job_controller.job_list[current_job].socket = sock;
	    	  job_controller.job_list[current_job].parent_panel->Bind(wxEVT_SOCKET, &JobPanel::OnJobSocketEvent, job_controller.job_list[current_job].parent_panel, SOCKET_ID);

	    	  sock->SetEventHandler(*job_controller.job_list[current_job].parent_panel, SOCKET_ID);
	    	  //sock->SetEventHandler(*this, SOCKET_ID);
	    	  sock->SetNotify(wxSOCKET_INPUT_FLAG | wxSOCKET_LOST_FLAG);
	    	  sock->Notify(true);

	    	  // Tell the socket it is connected

	    	  sock->WriteMsg(socket_you_are_connected, SOCKET_CODE_SIZE);
	      }
      }
      else
   	  {
	  	   	   MyDebugPrint(" ...Read Timeout \n\n");
	  	   	   // time out - close the connection
	    	   sock->Destroy();
	    	   sock = NULL;
	  }
}

void MyMainFrame::OnFileNewProject( wxCommandEvent& event )
{
	MyNewProjectWizard *my_wizard = new MyNewProjectWizard(this);
	my_wizard->GetPageAreaSizer()->Add(my_wizard->m_pages.Item(0));
	my_wizard->RunWizard(my_wizard->m_pages.Item(0));

	if (current_project.is_open == true)
	{
		SetTitle("ProjectX - [" + current_project.project_name + "]");
	}

}

void MyMainFrame::OnFileOpenProject( wxCommandEvent& event )
{
	// find a DB file..

	if (current_project.is_open)
	{
	    if (wxMessageBox("The current project must be closed before opening a new project.\n\nClose it now?", "Please confirm", wxICON_QUESTION | wxYES_NO, this) == wxNO ) return;

	    current_project.Close();
		SetTitle("ProjectX");

	}

	wxFileDialog openFileDialog(this, _("Open db file"), "", "", "DB files (*.db)|*.db", wxFD_OPEN|wxFD_FILE_MUST_EXIST);

	if (openFileDialog.ShowModal() == wxID_CANCEL) return;     // the user changed idea...

	if (current_project.OpenProjectFromFile(openFileDialog.GetPath()) == true)
	{
		SetTitle("ProjectX - [" + current_project.project_name + "]");

		movie_asset_panel->ImportAllFromDatabase();
		run_profiles_panel->ImportAllFromDatabase();

	}
	else
	{
		MyPrintWithDetails("An error occured opening the database file..");
	}



}

void MyMainFrame::OnFileExit( wxCommandEvent& event )
{
	Close(true);

}

void MyMainFrame::OnFileCloseProject( wxCommandEvent& event )
{
	current_project.Close();

	movie_asset_panel->Reset();
	RecalculateAssetBrowser();
	run_profiles_panel->Reset();


	SetTitle("ProjectX");

}
void MyMainFrame::OnFileMenuUpdate( wxUpdateUIEvent& event )
{
	if (current_project.is_open == true)
	{
		FileMenu->FindItem(FileMenu->FindItem("Close Project"))->Enable(true);
	}
	else FileMenu->FindItem(FileMenu->FindItem("Close Project"))->Enable(false);
}
