///////////////////////////////////////////////////////////////////////////
// C++ code generated with wxFormBuilder (version 3.10.1-0-g8feb16b)
// http://www.wxformbuilder.org/
//
// PLEASE DO *NOT* EDIT THIS FILE!
///////////////////////////////////////////////////////////////////////////

#include "../core/gui_core_headers.h"

#include "ProjectX_gui_main.h"

///////////////////////////////////////////////////////////////////////////

MainFrame::MainFrame( wxWindow* parent, wxWindowID id, const wxString& title, const wxPoint& pos, const wxSize& size, long style ) : wxFrame( parent, id, title, pos, size, style )
{
	this->SetSizeHints( wxSize( 1366,768 ), wxDefaultSize );

	wxBoxSizer* bSizer2;
	bSizer2 = new wxBoxSizer( wxVERTICAL );

	LeftPanel = new wxPanel( this, wxID_ANY, wxDefaultPosition, wxDefaultSize, wxTAB_TRAVERSAL|wxBORDER_RAISED );
	wxBoxSizer* bSizer8;
	bSizer8 = new wxBoxSizer( wxVERTICAL );

	MenuBook = new wxListbook( LeftPanel, wxID_ANY, wxDefaultPosition, wxDefaultSize, wxLB_LEFT );
	#ifdef __WXGTK__ // Small icon style not supported in GTK
	wxListView* MenuBookListView = MenuBook->GetListView();
	long MenuBookFlags = MenuBookListView->GetWindowStyleFlag();
	if( MenuBookFlags & wxLC_SMALL_ICON )
	{
		MenuBookFlags = ( MenuBookFlags & ~wxLC_SMALL_ICON ) | wxLC_ICON;
	}
	MenuBookListView->SetWindowStyleFlag( MenuBookFlags );
	#endif

	bSizer8->Add( MenuBook, 1, wxEXPAND | wxALL, 5 );


	LeftPanel->SetSizer( bSizer8 );
	LeftPanel->Layout();
	bSizer8->Fit( LeftPanel );
	bSizer2->Add( LeftPanel, 1, wxEXPAND | wxALL, 5 );


	this->SetSizer( bSizer2 );
	this->Layout();
	m_menubar1 = new wxMenuBar( 0 );
	FileMenu = new wxMenu();
	wxMenuItem* FileNewProject;
	FileNewProject = new wxMenuItem( FileMenu, wxID_ANY, wxString( wxT("New Project") ) , wxEmptyString, wxITEM_NORMAL );
	FileMenu->Append( FileNewProject );

	wxMenuItem* FileOpenProject;
	FileOpenProject = new wxMenuItem( FileMenu, wxID_ANY, wxString( wxT("Open Project") ) + wxT('\t') + wxT("Ctrl-O"), wxEmptyString, wxITEM_NORMAL );
	FileMenu->Append( FileOpenProject );

	wxMenuItem* FileCloseProject;
	FileCloseProject = new wxMenuItem( FileMenu, wxID_ANY, wxString( wxT("Close Project") ) , wxEmptyString, wxITEM_NORMAL );
	FileMenu->Append( FileCloseProject );

	FileMenu->AppendSeparator();

	wxMenuItem* FileExit;
	FileExit = new wxMenuItem( FileMenu, wxID_ANY, wxString( wxT("Exit") ) , wxEmptyString, wxITEM_NORMAL );
	FileMenu->Append( FileExit );

	m_menubar1->Append( FileMenu, wxT("Project") );

	WorkflowMenu = new wxMenu();
	WorkflowSingleParticle = new wxMenuItem( WorkflowMenu, wxID_ANY, wxString( wxT("Single Particle ") ) , wxEmptyString, wxITEM_RADIO );
	WorkflowMenu->Append( WorkflowSingleParticle );
	WorkflowSingleParticle->Check( true );

	WorkflowTemplateMatching = new wxMenuItem( WorkflowMenu, wxID_ANY, wxString( wxT("Template Matching") ) , wxEmptyString, wxITEM_RADIO );
	WorkflowMenu->Append( WorkflowTemplateMatching );

	WorkflowMenu->AppendSeparator();

	m_menubar1->Append( WorkflowMenu, wxT("Workflow") );

	HelpMenu = new wxMenu();
	wxMenuItem* OnlineHelpLaunch;
	OnlineHelpLaunch = new wxMenuItem( HelpMenu, wxID_ANY, wxString( wxT("Online Help") ) , wxEmptyString, wxITEM_NORMAL );
	HelpMenu->Append( OnlineHelpLaunch );

	wxMenuItem* AboutLaunch;
	AboutLaunch = new wxMenuItem( HelpMenu, wxID_ANY, wxString( wxT("About") ) , wxEmptyString, wxITEM_NORMAL );
	HelpMenu->Append( AboutLaunch );

	m_menubar1->Append( HelpMenu, wxT("Help") );

	this->SetMenuBar( m_menubar1 );


	this->Centre( wxBOTH );

	// Connect Events
	MenuBook->Connect( wxEVT_COMMAND_LISTBOOK_PAGE_CHANGED, wxListbookEventHandler( MainFrame::OnMenuBookChange ), NULL, this );
	m_menubar1->Connect( wxEVT_UPDATE_UI, wxUpdateUIEventHandler( MainFrame::OnFileMenuUpdate ), NULL, this );
	FileMenu->Bind(wxEVT_COMMAND_MENU_SELECTED, wxCommandEventHandler( MainFrame::OnFileNewProject ), this, FileNewProject->GetId());
	FileMenu->Bind(wxEVT_COMMAND_MENU_SELECTED, wxCommandEventHandler( MainFrame::OnFileOpenProject ), this, FileOpenProject->GetId());
	FileMenu->Bind(wxEVT_COMMAND_MENU_SELECTED, wxCommandEventHandler( MainFrame::OnFileCloseProject ), this, FileCloseProject->GetId());
	FileMenu->Bind(wxEVT_COMMAND_MENU_SELECTED, wxCommandEventHandler( MainFrame::OnFileExit ), this, FileExit->GetId());
	WorkflowMenu->Bind(wxEVT_COMMAND_MENU_SELECTED, wxCommandEventHandler( MainFrame::OnSingleParticleWorkflow ), this, WorkflowSingleParticle->GetId());
	WorkflowMenu->Bind(wxEVT_COMMAND_MENU_SELECTED, wxCommandEventHandler( MainFrame::OnTemplateMatchingWorkflow ), this, WorkflowTemplateMatching->GetId());
	HelpMenu->Bind(wxEVT_COMMAND_MENU_SELECTED, wxCommandEventHandler( MainFrame::OnHelpLaunch ), this, OnlineHelpLaunch->GetId());
	HelpMenu->Bind(wxEVT_COMMAND_MENU_SELECTED, wxCommandEventHandler( MainFrame::OnAboutLaunch ), this, AboutLaunch->GetId());
}

MainFrame::~MainFrame()
{
	// Disconnect Events
	MenuBook->Disconnect( wxEVT_COMMAND_LISTBOOK_PAGE_CHANGED, wxListbookEventHandler( MainFrame::OnMenuBookChange ), NULL, this );
	m_menubar1->Disconnect( wxEVT_UPDATE_UI, wxUpdateUIEventHandler( MainFrame::OnFileMenuUpdate ), NULL, this );

}

ActionsPanelParent::ActionsPanelParent( wxWindow* parent, wxWindowID id, const wxPoint& pos, const wxSize& size, long style, const wxString& name ) : wxPanel( parent, id, pos, size, style, name )
{
	wxBoxSizer* bSizer12;
	bSizer12 = new wxBoxSizer( wxHORIZONTAL );

	m_staticline3 = new wxStaticLine( this, wxID_ANY, wxDefaultPosition, wxDefaultSize, wxLI_VERTICAL );
	bSizer12->Add( m_staticline3, 0, wxEXPAND | wxALL, 5 );

	ActionsBook = new wxListbook( this, wxID_ANY, wxDefaultPosition, wxDefaultSize, wxLB_TOP );
	#ifdef __WXGTK__ // Small icon style not supported in GTK
	wxListView* ActionsBookListView = ActionsBook->GetListView();
	long ActionsBookFlags = ActionsBookListView->GetWindowStyleFlag();
	if( ActionsBookFlags & wxLC_SMALL_ICON )
	{
		ActionsBookFlags = ( ActionsBookFlags & ~wxLC_SMALL_ICON ) | wxLC_ICON;
	}
	ActionsBookListView->SetWindowStyleFlag( ActionsBookFlags );
	#endif

	bSizer12->Add( ActionsBook, 1, wxEXPAND | wxALL, 5 );


	this->SetSizer( bSizer12 );
	this->Layout();

	// Connect Events
	ActionsBook->Connect( wxEVT_COMMAND_LISTBOOK_PAGE_CHANGED, wxListbookEventHandler( ActionsPanelParent::OnActionsBookPageChanged ), NULL, this );
}

ActionsPanelParent::~ActionsPanelParent()
{
	// Disconnect Events
	ActionsBook->Disconnect( wxEVT_COMMAND_LISTBOOK_PAGE_CHANGED, wxListbookEventHandler( ActionsPanelParent::OnActionsBookPageChanged ), NULL, this );

}

SettingsPanel::SettingsPanel( wxWindow* parent, wxWindowID id, const wxPoint& pos, const wxSize& size, long style, const wxString& name ) : wxPanel( parent, id, pos, size, style, name )
{
	wxBoxSizer* bSizer12;
	bSizer12 = new wxBoxSizer( wxHORIZONTAL );

	m_staticline3 = new wxStaticLine( this, wxID_ANY, wxDefaultPosition, wxDefaultSize, wxLI_VERTICAL );
	bSizer12->Add( m_staticline3, 0, wxEXPAND | wxALL, 5 );

	SettingsBook = new wxListbook( this, wxID_ANY, wxDefaultPosition, wxDefaultSize, wxLB_TOP );
	#ifdef __WXGTK__ // Small icon style not supported in GTK
	wxListView* SettingsBookListView = SettingsBook->GetListView();
	long SettingsBookFlags = SettingsBookListView->GetWindowStyleFlag();
	if( SettingsBookFlags & wxLC_SMALL_ICON )
	{
		SettingsBookFlags = ( SettingsBookFlags & ~wxLC_SMALL_ICON ) | wxLC_ICON;
	}
	SettingsBookListView->SetWindowStyleFlag( SettingsBookFlags );
	#endif

	bSizer12->Add( SettingsBook, 1, wxEXPAND | wxALL, 5 );


	this->SetSizer( bSizer12 );
	this->Layout();

	// Connect Events
	SettingsBook->Connect( wxEVT_COMMAND_LISTBOOK_PAGE_CHANGED, wxListbookEventHandler( SettingsPanel::OnSettingsBookPageChanged ), NULL, this );
}

SettingsPanel::~SettingsPanel()
{
	// Disconnect Events
	SettingsBook->Disconnect( wxEVT_COMMAND_LISTBOOK_PAGE_CHANGED, wxListbookEventHandler( SettingsPanel::OnSettingsBookPageChanged ), NULL, this );

}

ResultsPanel::ResultsPanel( wxWindow* parent, wxWindowID id, const wxPoint& pos, const wxSize& size, long style, const wxString& name ) : wxPanel( parent, id, pos, size, style, name )
{
	wxBoxSizer* bSizer12;
	bSizer12 = new wxBoxSizer( wxHORIZONTAL );

	m_staticline3 = new wxStaticLine( this, wxID_ANY, wxDefaultPosition, wxDefaultSize, wxLI_VERTICAL );
	bSizer12->Add( m_staticline3, 0, wxEXPAND | wxALL, 5 );

	ResultsBook = new wxListbook( this, wxID_ANY, wxDefaultPosition, wxDefaultSize, wxLB_TOP );
	#ifdef __WXGTK__ // Small icon style not supported in GTK
	wxListView* ResultsBookListView = ResultsBook->GetListView();
	long ResultsBookFlags = ResultsBookListView->GetWindowStyleFlag();
	if( ResultsBookFlags & wxLC_SMALL_ICON )
	{
		ResultsBookFlags = ( ResultsBookFlags & ~wxLC_SMALL_ICON ) | wxLC_ICON;
	}
	ResultsBookListView->SetWindowStyleFlag( ResultsBookFlags );
	#endif

	bSizer12->Add( ResultsBook, 1, wxEXPAND | wxALL, 5 );


	this->SetSizer( bSizer12 );
	this->Layout();

	// Connect Events
	ResultsBook->Connect( wxEVT_COMMAND_LISTBOOK_PAGE_CHANGED, wxListbookEventHandler( ResultsPanel::OnResultsBookPageChanged ), NULL, this );
}

ResultsPanel::~ResultsPanel()
{
	// Disconnect Events
	ResultsBook->Disconnect( wxEVT_COMMAND_LISTBOOK_PAGE_CHANGED, wxListbookEventHandler( ResultsPanel::OnResultsBookPageChanged ), NULL, this );

}

AssetsPanel::AssetsPanel( wxWindow* parent, wxWindowID id, const wxPoint& pos, const wxSize& size, long style, const wxString& name ) : wxPanel( parent, id, pos, size, style, name )
{
	wxBoxSizer* bSizer76;
	bSizer76 = new wxBoxSizer( wxHORIZONTAL );

	m_staticline68 = new wxStaticLine( this, wxID_ANY, wxDefaultPosition, wxDefaultSize, wxLI_VERTICAL );
	bSizer76->Add( m_staticline68, 0, wxEXPAND | wxALL, 5 );

	AssetsBook = new wxListbook( this, wxID_ANY, wxDefaultPosition, wxDefaultSize, wxLB_TOP );
	#ifdef __WXGTK__ // Small icon style not supported in GTK
	wxListView* AssetsBookListView = AssetsBook->GetListView();
	long AssetsBookFlags = AssetsBookListView->GetWindowStyleFlag();
	if( AssetsBookFlags & wxLC_SMALL_ICON )
	{
		AssetsBookFlags = ( AssetsBookFlags & ~wxLC_SMALL_ICON ) | wxLC_ICON;
	}
	AssetsBookListView->SetWindowStyleFlag( AssetsBookFlags );
	#endif

	bSizer76->Add( AssetsBook, 1, wxEXPAND | wxALL, 5 );


	this->SetSizer( bSizer76 );
	this->Layout();

	// Connect Events
	AssetsBook->Connect( wxEVT_COMMAND_LISTBOOK_PAGE_CHANGED, wxListbookEventHandler( AssetsPanel::OnAssetsBookPageChanged ), NULL, this );
}

AssetsPanel::~AssetsPanel()
{
	// Disconnect Events
	AssetsBook->Disconnect( wxEVT_COMMAND_LISTBOOK_PAGE_CHANGED, wxListbookEventHandler( AssetsPanel::OnAssetsBookPageChanged ), NULL, this );

}

ExperimentalPanel::ExperimentalPanel( wxWindow* parent, wxWindowID id, const wxPoint& pos, const wxSize& size, long style, const wxString& name ) : wxPanel( parent, id, pos, size, style, name )
{
	wxBoxSizer* bSizer76;
	bSizer76 = new wxBoxSizer( wxHORIZONTAL );

	m_staticline68 = new wxStaticLine( this, wxID_ANY, wxDefaultPosition, wxDefaultSize, wxLI_VERTICAL );
	bSizer76->Add( m_staticline68, 0, wxEXPAND | wxALL, 5 );

	ExperimentalBook = new wxListbook( this, wxID_ANY, wxDefaultPosition, wxDefaultSize, wxLB_TOP );
	#ifdef __WXGTK__ // Small icon style not supported in GTK
	wxListView* ExperimentalBookListView = ExperimentalBook->GetListView();
	long ExperimentalBookFlags = ExperimentalBookListView->GetWindowStyleFlag();
	if( ExperimentalBookFlags & wxLC_SMALL_ICON )
	{
		ExperimentalBookFlags = ( ExperimentalBookFlags & ~wxLC_SMALL_ICON ) | wxLC_ICON;
	}
	ExperimentalBookListView->SetWindowStyleFlag( ExperimentalBookFlags );
	#endif

	bSizer76->Add( ExperimentalBook, 1, wxEXPAND | wxALL, 5 );


	this->SetSizer( bSizer76 );
	this->Layout();

	// Connect Events
	ExperimentalBook->Connect( wxEVT_COMMAND_LISTBOOK_PAGE_CHANGED, wxListbookEventHandler( ExperimentalPanel::OnExperimentalBookPageChanged ), NULL, this );
}

ExperimentalPanel::~ExperimentalPanel()
{
	// Disconnect Events
	ExperimentalBook->Disconnect( wxEVT_COMMAND_LISTBOOK_PAGE_CHANGED, wxListbookEventHandler( ExperimentalPanel::OnExperimentalBookPageChanged ), NULL, this );

}

OverviewPanel::OverviewPanel( wxWindow* parent, wxWindowID id, const wxPoint& pos, const wxSize& size, long style, const wxString& name ) : wxPanel( parent, id, pos, size, style, name )
{
	wxBoxSizer* bSizer10;
	bSizer10 = new wxBoxSizer( wxHORIZONTAL );

	m_staticline2 = new wxStaticLine( this, wxID_ANY, wxDefaultPosition, wxDefaultSize, wxLI_VERTICAL );
	bSizer10->Add( m_staticline2, 0, wxEXPAND | wxALL, 5 );

	wxBoxSizer* bSizer198;
	bSizer198 = new wxBoxSizer( wxVERTICAL );

	WelcomePanel = new wxPanel( this, wxID_ANY, wxDefaultPosition, wxDefaultSize, wxTAB_TRAVERSAL );
	wxBoxSizer* bSizer281;
	bSizer281 = new wxBoxSizer( wxVERTICAL );

	InfoText = new wxRichTextCtrl( WelcomePanel, wxID_ANY, wxEmptyString, wxDefaultPosition, wxDefaultSize, wxTE_READONLY|wxHSCROLL|wxVSCROLL|wxWANTS_CHARS );
	bSizer281->Add( InfoText, 1, wxEXPAND | wxALL, 5 );


	WelcomePanel->SetSizer( bSizer281 );
	WelcomePanel->Layout();
	bSizer281->Fit( WelcomePanel );
	bSizer198->Add( WelcomePanel, 1, wxEXPAND | wxALL, 5 );


	bSizer10->Add( bSizer198, 1, wxALIGN_CENTER|wxEXPAND, 5 );


	this->SetSizer( bSizer10 );
	this->Layout();

	// Connect Events
	InfoText->Connect( wxEVT_COMMAND_TEXT_URL, wxTextUrlEventHandler( OverviewPanel::OnInfoURL ), NULL, this );
}

OverviewPanel::~OverviewPanel()
{
	// Disconnect Events
	InfoText->Disconnect( wxEVT_COMMAND_TEXT_URL, wxTextUrlEventHandler( OverviewPanel::OnInfoURL ), NULL, this );

}

AboutDialog::AboutDialog( wxWindow* parent, wxWindowID id, const wxString& title, const wxPoint& pos, const wxSize& size, long style ) : wxDialog( parent, id, title, pos, size, style )
{
	this->SetSizeHints( wxSize( 350,-1 ), wxDefaultSize );

	wxBoxSizer* bSizer445;
	bSizer445 = new wxBoxSizer( wxVERTICAL );

	m_staticline131 = new wxStaticLine( this, wxID_ANY, wxDefaultPosition, wxDefaultSize, wxLI_HORIZONTAL );
	bSizer445->Add( m_staticline131, 0, wxEXPAND | wxALL, 5 );

	LogoBitmap = new wxStaticBitmap( this, wxID_ANY, wxNullBitmap, wxDefaultPosition, wxDefaultSize, 0 );
	bSizer445->Add( LogoBitmap, 0, wxALIGN_CENTER_HORIZONTAL|wxALL, 5 );

	m_staticline130 = new wxStaticLine( this, wxID_ANY, wxDefaultPosition, wxDefaultSize, wxLI_HORIZONTAL );
	bSizer445->Add( m_staticline130, 0, wxEXPAND | wxALL, 5 );

	VersionStaticText = new wxStaticText( this, wxID_ANY, wxT("cisTEM version 1.0beta-RC1"), wxDefaultPosition, wxDefaultSize, 0 );
	VersionStaticText->Wrap( -1 );
	bSizer445->Add( VersionStaticText, 0, wxALIGN_CENTER_HORIZONTAL|wxALL, 5 );

	BuildDateText = new wxStaticText( this, wxID_ANY, wxT("Built : Nov 20 2017"), wxDefaultPosition, wxDefaultSize, 0 );
	BuildDateText->Wrap( -1 );
	bSizer445->Add( BuildDateText, 0, wxALIGN_CENTER_HORIZONTAL|wxALL, 5 );

	m_staticText611 = new wxStaticText( this, wxID_ANY, wxEmptyString, wxDefaultPosition, wxDefaultSize, 0 );
	m_staticText611->Wrap( -1 );
	bSizer445->Add( m_staticText611, 0, wxALL, 5 );

	wxFlexGridSizer* fgSizer30;
	fgSizer30 = new wxFlexGridSizer( 0, 2, 0, 0 );
	fgSizer30->SetFlexibleDirection( wxBOTH );
	fgSizer30->SetNonFlexibleGrowMode( wxFLEX_GROWMODE_SPECIFIED );

	m_staticText605 = new wxStaticText( this, wxID_ANY, wxT("Developed By :"), wxDefaultPosition, wxDefaultSize, 0 );
	m_staticText605->Wrap( -1 );
	fgSizer30->Add( m_staticText605, 0, wxALIGN_RIGHT|wxALL, 5 );

	m_staticText606 = new wxStaticText( this, wxID_ANY, wxT("Tim Grant"), wxDefaultPosition, wxDefaultSize, 0 );
	m_staticText606->Wrap( -1 );
	fgSizer30->Add( m_staticText606, 0, wxALL, 5 );

	m_staticText607 = new wxStaticText( this, wxID_ANY, wxEmptyString, wxDefaultPosition, wxDefaultSize, 0 );
	m_staticText607->Wrap( -1 );
	fgSizer30->Add( m_staticText607, 0, wxALL, 5 );

	m_staticText608 = new wxStaticText( this, wxID_ANY, wxT("Alexis Rohou"), wxDefaultPosition, wxDefaultSize, 0 );
	m_staticText608->Wrap( -1 );
	fgSizer30->Add( m_staticText608, 0, wxALL, 5 );

	m_staticText609 = new wxStaticText( this, wxID_ANY, wxEmptyString, wxDefaultPosition, wxDefaultSize, 0 );
	m_staticText609->Wrap( -1 );
	fgSizer30->Add( m_staticText609, 0, wxALL, 5 );

	m_staticText610 = new wxStaticText( this, wxID_ANY, wxT("Nikolaus Grigorieff"), wxDefaultPosition, wxDefaultSize, 0 );
	m_staticText610->Wrap( -1 );
	fgSizer30->Add( m_staticText610, 0, wxALL, 5 );

	m_staticText613 = new wxStaticText( this, wxID_ANY, wxEmptyString, wxDefaultPosition, wxDefaultSize, 0 );
	m_staticText613->Wrap( -1 );
	fgSizer30->Add( m_staticText613, 0, wxALL, 5 );

	m_staticText614 = new wxStaticText( this, wxID_ANY, wxEmptyString, wxDefaultPosition, wxDefaultSize, 0 );
	m_staticText614->Wrap( -1 );
	fgSizer30->Add( m_staticText614, 0, wxALL, 5 );

	m_staticText615 = new wxStaticText( this, wxID_ANY, wxT("Web Page :"), wxDefaultPosition, wxDefaultSize, 0 );
	m_staticText615->Wrap( -1 );
	fgSizer30->Add( m_staticText615, 0, wxALIGN_CENTER_VERTICAL|wxALIGN_RIGHT|wxALL, 5 );

	m_hyperlink1 = new wxHyperlinkCtrl( this, wxID_ANY, wxT("cistem.org"), wxT("http://www.cistem.org"), wxDefaultPosition, wxDefaultSize, wxHL_DEFAULT_STYLE );
	fgSizer30->Add( m_hyperlink1, 0, wxALL, 5 );

	m_staticText617 = new wxStaticText( this, wxID_ANY, wxT("License :"), wxDefaultPosition, wxDefaultSize, 0 );
	m_staticText617->Wrap( -1 );
	fgSizer30->Add( m_staticText617, 0, wxALIGN_CENTER_VERTICAL|wxALIGN_RIGHT|wxALL, 5 );

	m_hyperlink2 = new wxHyperlinkCtrl( this, wxID_ANY, wxT("Janelia License"), wxT("http://license.janelia.org/license/"), wxDefaultPosition, wxDefaultSize, wxHL_DEFAULT_STYLE );
	fgSizer30->Add( m_hyperlink2, 0, wxALL, 5 );


	bSizer445->Add( fgSizer30, 5, wxALIGN_CENTER_HORIZONTAL, 20 );

	m_staticline129 = new wxStaticLine( this, wxID_ANY, wxDefaultPosition, wxDefaultSize, wxLI_HORIZONTAL );
	bSizer445->Add( m_staticline129, 0, wxEXPAND | wxALL, 5 );

	m_button141 = new wxButton( this, wxID_CANCEL, wxT("OK"), wxDefaultPosition, wxDefaultSize, 0 );
	bSizer445->Add( m_button141, 0, wxALIGN_CENTER_HORIZONTAL|wxALL, 5 );


	this->SetSizer( bSizer445 );
	this->Layout();
	bSizer445->Fit( this );

	this->Centre( wxBOTH );
}

AboutDialog::~AboutDialog()
{
}

DatabaseUpdateDialogParent::DatabaseUpdateDialogParent( wxWindow* parent, wxWindowID id, const wxString& title, const wxPoint& pos, const wxSize& size, long style ) : wxDialog( parent, id, title, pos, size, style )
{
	this->SetSizeHints( wxDefaultSize, wxDefaultSize );

	wxBoxSizer* bSizer12;
	bSizer12 = new wxBoxSizer( wxVERTICAL );

	wxBoxSizer* bSizer13;
	bSizer13 = new wxBoxSizer( wxVERTICAL );

	UpdateText = new wxStaticText( this, wxID_ANY, wxT("cisTEM can try to update the database schema format, which for databases with many classifications and refinements from earlier versions of cisTEM can take a long time. \n\nIt is wise to make a backup before trying this, and to ensure there is enough space remaining on your disk to do so.\n\nAttempt to update the project?"), wxDefaultPosition, wxDefaultSize, 0 );
	UpdateText->Wrap( 620 );
	bSizer13->Add( UpdateText, 0, wxALL|wxEXPAND, 5 );


	bSizer12->Add( bSizer13, 1, wxALIGN_CENTER_HORIZONTAL, 5 );

	wxBoxSizer* bSizer17;
	bSizer17 = new wxBoxSizer( wxHORIZONTAL );

	SchemaChangesTextCtrl = new wxTextCtrl( this, wxID_ANY, wxEmptyString, wxDefaultPosition, wxDefaultSize, wxTE_DONTWRAP|wxTE_LEFT|wxTE_MULTILINE|wxTE_READONLY );
	bSizer17->Add( SchemaChangesTextCtrl, 1, wxALL|wxEXPAND, 5 );


	bSizer12->Add( bSizer17, 1, wxALIGN_TOP|wxEXPAND, 5 );

	m_staticline10 = new wxStaticLine( this, wxID_ANY, wxDefaultPosition, wxDefaultSize, wxLI_HORIZONTAL );
	bSizer12->Add( m_staticline10, 0, wxEXPAND | wxALL, 5 );

	wxBoxSizer* bSizer20;
	bSizer20 = new wxBoxSizer( wxVERTICAL );

	wxBoxSizer* bSizer14;
	bSizer14 = new wxBoxSizer( wxHORIZONTAL );

	CancelButton = new wxButton( this, wxID_CANCEL, wxT("Cancel"), wxDefaultPosition, wxDefaultSize, 0 );
	bSizer14->Add( CancelButton, 0, wxALL|wxALIGN_BOTTOM, 5 );

	UpdateButton = new wxButton( this, wxID_UPDATE_ONLY, wxT("Update"), wxDefaultPosition, wxDefaultSize, 0 );
	bSizer14->Add( UpdateButton, 0, wxALL|wxALIGN_BOTTOM, 5 );

	BackupUpdateButton = new wxButton( this, wxID_BACKUP_AND_UPDATE, wxT("Backup And Update"), wxDefaultPosition, wxDefaultSize, 0 );
	bSizer14->Add( BackupUpdateButton, 0, wxALIGN_BOTTOM|wxALL, 5 );


	bSizer20->Add( bSizer14, 1, wxALIGN_RIGHT, 5 );


	bSizer12->Add( bSizer20, 0, wxALIGN_RIGHT, 5 );


	this->SetSizer( bSizer12 );
	this->Layout();
	bSizer12->Fit( this );

	this->Centre( wxBOTH );

	// Connect Events
	CancelButton->Connect( wxEVT_COMMAND_BUTTON_CLICKED, wxCommandEventHandler( DatabaseUpdateDialogParent::OnButtonClicked ), NULL, this );
	UpdateButton->Connect( wxEVT_COMMAND_BUTTON_CLICKED, wxCommandEventHandler( DatabaseUpdateDialogParent::OnButtonClicked ), NULL, this );
	BackupUpdateButton->Connect( wxEVT_COMMAND_BUTTON_CLICKED, wxCommandEventHandler( DatabaseUpdateDialogParent::OnButtonClicked ), NULL, this );
}

DatabaseUpdateDialogParent::~DatabaseUpdateDialogParent()
{
	// Disconnect Events
	CancelButton->Disconnect( wxEVT_COMMAND_BUTTON_CLICKED, wxCommandEventHandler( DatabaseUpdateDialogParent::OnButtonClicked ), NULL, this );
	UpdateButton->Disconnect( wxEVT_COMMAND_BUTTON_CLICKED, wxCommandEventHandler( DatabaseUpdateDialogParent::OnButtonClicked ), NULL, this );
	BackupUpdateButton->Disconnect( wxEVT_COMMAND_BUTTON_CLICKED, wxCommandEventHandler( DatabaseUpdateDialogParent::OnButtonClicked ), NULL, this );

}
