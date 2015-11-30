///////////////////////////////////////////////////////////////////////////
// C++ code generated with wxFormBuilder (version Feb 20 2015)
// http://www.wxformbuilder.org/
//
// PLEASE DO "NOT" EDIT THIS FILE!
///////////////////////////////////////////////////////////////////////////

#include "ProjectX_gui.h"

///////////////////////////////////////////////////////////////////////////

MainFrame::MainFrame( wxWindow* parent, wxWindowID id, const wxString& title, const wxPoint& pos, const wxSize& size, long style ) : wxFrame( parent, id, title, pos, size, style )
{
	this->SetSizeHints( wxSize( 1200,800 ), wxDefaultSize );
	
	wxBoxSizer* bSizer2;
	bSizer2 = new wxBoxSizer( wxVERTICAL );
	
	MainSplitter = new wxSplitterWindow( this, wxID_ANY, wxDefaultPosition, wxDefaultSize, wxSP_3D );
	MainSplitter->SetSashGravity( 1 );
	MainSplitter->Connect( wxEVT_IDLE, wxIdleEventHandler( MainFrame::MainSplitterOnIdle ), NULL, this );
	
	LeftPanel = new wxPanel( MainSplitter, wxID_ANY, wxDefaultPosition, wxDefaultSize, wxRAISED_BORDER|wxTAB_TRAVERSAL );
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
	RightPanel = new wxPanel( MainSplitter, wxID_ANY, wxDefaultPosition, wxDefaultSize, wxTAB_TRAVERSAL );
	wxStaticBoxSizer* sbSizer4;
	sbSizer4 = new wxStaticBoxSizer( new wxStaticBox( RightPanel, wxID_ANY, wxT("Asset Browser") ), wxVERTICAL );
	
	AssetTree = new wxTreeCtrl( RightPanel, wxID_ANY, wxDefaultPosition, wxDefaultSize, wxTR_DEFAULT_STYLE|wxTR_HIDE_ROOT );
	sbSizer4->Add( AssetTree, 100, wxALL|wxEXPAND, 5 );
	
	m_button12 = new wxButton( RightPanel, wxID_ANY, wxT("Collapse All"), wxDefaultPosition, wxDefaultSize, 0 );
	sbSizer4->Add( m_button12, 0, wxALL|wxEXPAND, 5 );
	
	
	RightPanel->SetSizer( sbSizer4 );
	RightPanel->Layout();
	sbSizer4->Fit( RightPanel );
	MainSplitter->SplitVertically( LeftPanel, RightPanel, 1000 );
	bSizer2->Add( MainSplitter, 1, wxEXPAND, 5 );
	
	
	this->SetSizer( bSizer2 );
	this->Layout();
	m_menubar1 = new wxMenuBar( 0 );
	FileMenu = new wxMenu();
	wxMenuItem* FileNewProject;
	FileNewProject = new wxMenuItem( FileMenu, wxID_ANY, wxString( wxT("New Project") ) , wxEmptyString, wxITEM_NORMAL );
	FileMenu->Append( FileNewProject );
	
	wxMenuItem* FileOpenProject;
	FileOpenProject = new wxMenuItem( FileMenu, wxID_ANY, wxString( wxT("Open Project") ) , wxEmptyString, wxITEM_NORMAL );
	FileMenu->Append( FileOpenProject );
	
	wxMenuItem* FileCloseProject;
	FileCloseProject = new wxMenuItem( FileMenu, wxID_ANY, wxString( wxT("Close Project") ) , wxEmptyString, wxITEM_NORMAL );
	FileMenu->Append( FileCloseProject );
	
	FileMenu->AppendSeparator();
	
	wxMenuItem* FileExit;
	FileExit = new wxMenuItem( FileMenu, wxID_ANY, wxString( wxT("Exit") ) , wxEmptyString, wxITEM_NORMAL );
	FileMenu->Append( FileExit );
	
	m_menubar1->Append( FileMenu, wxT("Project") ); 
	
	this->SetMenuBar( m_menubar1 );
	
	
	this->Centre( wxBOTH );
	
	// Connect Events
	MenuBook->Connect( wxEVT_COMMAND_LISTBOOK_PAGE_CHANGED, wxListbookEventHandler( MainFrame::OnMenuBookChange ), NULL, this );
	m_button12->Connect( wxEVT_COMMAND_BUTTON_CLICKED, wxCommandEventHandler( MainFrame::OnCollapseAll ), NULL, this );
	m_menubar1->Connect( wxEVT_UPDATE_UI, wxUpdateUIEventHandler( MainFrame::OnFileMenuUpdate ), NULL, this );
	this->Connect( FileNewProject->GetId(), wxEVT_COMMAND_MENU_SELECTED, wxCommandEventHandler( MainFrame::OnFileNewProject ) );
	this->Connect( FileOpenProject->GetId(), wxEVT_COMMAND_MENU_SELECTED, wxCommandEventHandler( MainFrame::OnFileOpenProject ) );
	this->Connect( FileCloseProject->GetId(), wxEVT_COMMAND_MENU_SELECTED, wxCommandEventHandler( MainFrame::OnFileCloseProject ) );
	this->Connect( FileExit->GetId(), wxEVT_COMMAND_MENU_SELECTED, wxCommandEventHandler( MainFrame::OnFileExit ) );
}

MainFrame::~MainFrame()
{
	// Disconnect Events
	MenuBook->Disconnect( wxEVT_COMMAND_LISTBOOK_PAGE_CHANGED, wxListbookEventHandler( MainFrame::OnMenuBookChange ), NULL, this );
	m_button12->Disconnect( wxEVT_COMMAND_BUTTON_CLICKED, wxCommandEventHandler( MainFrame::OnCollapseAll ), NULL, this );
	m_menubar1->Disconnect( wxEVT_UPDATE_UI, wxUpdateUIEventHandler( MainFrame::OnFileMenuUpdate ), NULL, this );
	this->Disconnect( wxID_ANY, wxEVT_COMMAND_MENU_SELECTED, wxCommandEventHandler( MainFrame::OnFileNewProject ) );
	this->Disconnect( wxID_ANY, wxEVT_COMMAND_MENU_SELECTED, wxCommandEventHandler( MainFrame::OnFileOpenProject ) );
	this->Disconnect( wxID_ANY, wxEVT_COMMAND_MENU_SELECTED, wxCommandEventHandler( MainFrame::OnFileCloseProject ) );
	this->Disconnect( wxID_ANY, wxEVT_COMMAND_MENU_SELECTED, wxCommandEventHandler( MainFrame::OnFileExit ) );
	
}

ActionsPanel::ActionsPanel( wxWindow* parent, wxWindowID id, const wxPoint& pos, const wxSize& size, long style ) : wxPanel( parent, id, pos, size, style )
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
}

ActionsPanel::~ActionsPanel()
{
}

SettingsPanel::SettingsPanel( wxWindow* parent, wxWindowID id, const wxPoint& pos, const wxSize& size, long style ) : wxPanel( parent, id, pos, size, style )
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
}

SettingsPanel::~SettingsPanel()
{
}

AssetsPanel::AssetsPanel( wxWindow* parent, wxWindowID id, const wxPoint& pos, const wxSize& size, long style ) : wxPanel( parent, id, pos, size, style )
{
	wxBoxSizer* bSizer13;
	bSizer13 = new wxBoxSizer( wxHORIZONTAL );
	
	m_staticline4 = new wxStaticLine( this, wxID_ANY, wxDefaultPosition, wxDefaultSize, wxLI_VERTICAL );
	bSizer13->Add( m_staticline4, 0, wxEXPAND | wxALL, 5 );
	
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
	
	bSizer13->Add( AssetsBook, 1, wxEXPAND | wxALL, 5 );
	
	
	this->SetSizer( bSizer13 );
	this->Layout();
}

AssetsPanel::~AssetsPanel()
{
}

OverviewPanel::OverviewPanel( wxWindow* parent, wxWindowID id, const wxPoint& pos, const wxSize& size, long style ) : wxPanel( parent, id, pos, size, style )
{
	wxBoxSizer* bSizer10;
	bSizer10 = new wxBoxSizer( wxHORIZONTAL );
	
	m_staticline2 = new wxStaticLine( this, wxID_ANY, wxDefaultPosition, wxDefaultSize, wxLI_VERTICAL );
	bSizer10->Add( m_staticline2, 0, wxEXPAND | wxALL, 5 );
	
	m_staticText1 = new wxStaticText( this, wxID_ANY, wxT("This is an overview,\nthat will display many\ninteresting things!"), wxDefaultPosition, wxDefaultSize, wxALIGN_CENTRE );
	m_staticText1->Wrap( -1 );
	m_staticText1->SetFont( wxFont( 24, 70, 90, 90, false, wxEmptyString ) );
	
	bSizer10->Add( m_staticText1, 0, wxALIGN_CENTER_HORIZONTAL|wxALIGN_CENTER_VERTICAL|wxALL, 5 );
	
	
	this->SetSizer( bSizer10 );
	this->Layout();
}

OverviewPanel::~OverviewPanel()
{
}

MovieImportDialog::MovieImportDialog( wxWindow* parent, wxWindowID id, const wxString& title, const wxPoint& pos, const wxSize& size, long style ) : wxDialog( parent, id, title, pos, size, style )
{
	this->SetSizeHints( wxDefaultSize, wxDefaultSize );
	
	wxBoxSizer* bSizer26;
	bSizer26 = new wxBoxSizer( wxVERTICAL );
	
	PathListCtrl = new wxListCtrl( this, wxID_ANY, wxDefaultPosition, wxDefaultSize, wxLC_LIST|wxLC_NO_HEADER|wxLC_REPORT|wxVSCROLL );
	bSizer26->Add( PathListCtrl, 100, wxALL|wxEXPAND, 5 );
	
	wxBoxSizer* bSizer27;
	bSizer27 = new wxBoxSizer( wxHORIZONTAL );
	
	m_button10 = new wxButton( this, wxID_ANY, wxT("Add Files"), wxDefaultPosition, wxDefaultSize, 0 );
	bSizer27->Add( m_button10, 33, wxALL, 5 );
	
	m_button11 = new wxButton( this, wxID_ANY, wxT("Add Directory"), wxDefaultPosition, wxDefaultSize, 0 );
	bSizer27->Add( m_button11, 33, wxALL, 5 );
	
	ClearButton = new wxButton( this, wxID_ANY, wxT("Clear"), wxDefaultPosition, wxDefaultSize, 0 );
	bSizer27->Add( ClearButton, 33, wxALL, 5 );
	
	
	bSizer26->Add( bSizer27, 0, wxEXPAND, 5 );
	
	m_staticline7 = new wxStaticLine( this, wxID_ANY, wxDefaultPosition, wxDefaultSize, wxLI_HORIZONTAL );
	bSizer26->Add( m_staticline7, 0, wxEXPAND | wxALL, 5 );
	
	wxBoxSizer* bSizer28;
	bSizer28 = new wxBoxSizer( wxHORIZONTAL );
	
	m_staticText19 = new wxStaticText( this, wxID_ANY, wxT("Voltage (kV) :"), wxDefaultPosition, wxDefaultSize, 0 );
	m_staticText19->Wrap( -1 );
	bSizer28->Add( m_staticText19, 50, wxALIGN_CENTER_VERTICAL|wxALL, 5 );
	
	VoltageCombo = new wxComboBox( this, wxID_ANY, wxEmptyString, wxDefaultPosition, wxDefaultSize, 0, NULL, 0 );
	VoltageCombo->Append( wxT("300") );
	VoltageCombo->Append( wxT("200") );
	VoltageCombo->Append( wxT("120") );
	bSizer28->Add( VoltageCombo, 50, wxALIGN_CENTER_VERTICAL|wxALL, 5 );
	
	
	bSizer26->Add( bSizer28, 0, wxEXPAND, 5 );
	
	wxBoxSizer* bSizer29;
	bSizer29 = new wxBoxSizer( wxHORIZONTAL );
	
	m_staticText20 = new wxStaticText( this, wxID_ANY, wxT("Pixel Size (Å) :"), wxDefaultPosition, wxDefaultSize, 0 );
	m_staticText20->Wrap( -1 );
	bSizer29->Add( m_staticText20, 50, wxALIGN_CENTER_VERTICAL|wxALL, 5 );
	
	PixelSizeText = new wxTextCtrl( this, wxID_ANY, wxEmptyString, wxDefaultPosition, wxDefaultSize, 0 );
	bSizer29->Add( PixelSizeText, 50, wxALIGN_CENTER_VERTICAL|wxALL, 5 );
	
	
	bSizer26->Add( bSizer29, 0, wxEXPAND, 5 );
	
	wxBoxSizer* bSizer30;
	bSizer30 = new wxBoxSizer( wxHORIZONTAL );
	
	m_staticText21 = new wxStaticText( this, wxID_ANY, wxT("Spherical Aberration (mm) :"), wxDefaultPosition, wxDefaultSize, 0 );
	m_staticText21->Wrap( -1 );
	bSizer30->Add( m_staticText21, 50, wxALIGN_CENTER_VERTICAL|wxALL, 5 );
	
	CsText = new wxTextCtrl( this, wxID_ANY, wxEmptyString, wxDefaultPosition, wxDefaultSize, 0 );
	bSizer30->Add( CsText, 50, wxALIGN_CENTER_VERTICAL|wxALL, 5 );
	
	
	bSizer26->Add( bSizer30, 0, wxEXPAND, 5 );
	
	wxBoxSizer* bSizer32;
	bSizer32 = new wxBoxSizer( wxHORIZONTAL );
	
	m_staticText22 = new wxStaticText( this, wxID_ANY, wxT("Dose per frame (e¯/Å²) :"), wxDefaultPosition, wxDefaultSize, wxALIGN_RIGHT );
	m_staticText22->Wrap( -1 );
	bSizer32->Add( m_staticText22, 50, wxALIGN_CENTER_VERTICAL|wxALL, 5 );
	
	DoseText = new wxTextCtrl( this, wxID_ANY, wxEmptyString, wxDefaultPosition, wxDefaultSize, 0 );
	bSizer32->Add( DoseText, 50, wxALIGN_CENTER_VERTICAL|wxALL, 5 );
	
	
	bSizer26->Add( bSizer32, 0, wxEXPAND, 5 );
	
	m_staticline8 = new wxStaticLine( this, wxID_ANY, wxDefaultPosition, wxDefaultSize, wxLI_HORIZONTAL );
	bSizer26->Add( m_staticline8, 0, wxEXPAND | wxALL, 5 );
	
	wxBoxSizer* bSizer33;
	bSizer33 = new wxBoxSizer( wxHORIZONTAL );
	
	m_button13 = new wxButton( this, wxID_ANY, wxT("Cancel"), wxDefaultPosition, wxDefaultSize, 0 );
	bSizer33->Add( m_button13, 0, wxALL, 5 );
	
	
	bSizer33->Add( 0, 0, 1, wxEXPAND, 5 );
	
	
	bSizer33->Add( 0, 0, 1, wxEXPAND, 5 );
	
	ImportButton = new wxButton( this, wxID_ANY, wxT("Import"), wxDefaultPosition, wxDefaultSize, 0 );
	ImportButton->Enable( false );
	
	bSizer33->Add( ImportButton, 0, wxALL, 5 );
	
	
	bSizer26->Add( bSizer33, 1, wxEXPAND, 5 );
	
	
	this->SetSizer( bSizer26 );
	this->Layout();
	
	this->Centre( wxBOTH );
	
	// Connect Events
	m_button10->Connect( wxEVT_COMMAND_BUTTON_CLICKED, wxCommandEventHandler( MovieImportDialog::AddFilesClick ), NULL, this );
	m_button11->Connect( wxEVT_COMMAND_BUTTON_CLICKED, wxCommandEventHandler( MovieImportDialog::AddDirectoryClick ), NULL, this );
	ClearButton->Connect( wxEVT_COMMAND_BUTTON_CLICKED, wxCommandEventHandler( MovieImportDialog::ClearClick ), NULL, this );
	VoltageCombo->Connect( wxEVT_CHAR, wxKeyEventHandler( MovieImportDialog::OnTextKeyPress ), NULL, this );
	VoltageCombo->Connect( wxEVT_COMMAND_TEXT_UPDATED, wxCommandEventHandler( MovieImportDialog::TextChanged ), NULL, this );
	PixelSizeText->Connect( wxEVT_CHAR, wxKeyEventHandler( MovieImportDialog::OnTextKeyPress ), NULL, this );
	PixelSizeText->Connect( wxEVT_COMMAND_TEXT_UPDATED, wxCommandEventHandler( MovieImportDialog::TextChanged ), NULL, this );
	CsText->Connect( wxEVT_CHAR, wxKeyEventHandler( MovieImportDialog::OnTextKeyPress ), NULL, this );
	CsText->Connect( wxEVT_COMMAND_TEXT_UPDATED, wxCommandEventHandler( MovieImportDialog::TextChanged ), NULL, this );
	DoseText->Connect( wxEVT_CHAR, wxKeyEventHandler( MovieImportDialog::OnTextKeyPress ), NULL, this );
	DoseText->Connect( wxEVT_COMMAND_TEXT_UPDATED, wxCommandEventHandler( MovieImportDialog::TextChanged ), NULL, this );
	m_button13->Connect( wxEVT_COMMAND_BUTTON_CLICKED, wxCommandEventHandler( MovieImportDialog::CancelClick ), NULL, this );
	ImportButton->Connect( wxEVT_COMMAND_BUTTON_CLICKED, wxCommandEventHandler( MovieImportDialog::ImportClick ), NULL, this );
}

MovieImportDialog::~MovieImportDialog()
{
	// Disconnect Events
	m_button10->Disconnect( wxEVT_COMMAND_BUTTON_CLICKED, wxCommandEventHandler( MovieImportDialog::AddFilesClick ), NULL, this );
	m_button11->Disconnect( wxEVT_COMMAND_BUTTON_CLICKED, wxCommandEventHandler( MovieImportDialog::AddDirectoryClick ), NULL, this );
	ClearButton->Disconnect( wxEVT_COMMAND_BUTTON_CLICKED, wxCommandEventHandler( MovieImportDialog::ClearClick ), NULL, this );
	VoltageCombo->Disconnect( wxEVT_CHAR, wxKeyEventHandler( MovieImportDialog::OnTextKeyPress ), NULL, this );
	VoltageCombo->Disconnect( wxEVT_COMMAND_TEXT_UPDATED, wxCommandEventHandler( MovieImportDialog::TextChanged ), NULL, this );
	PixelSizeText->Disconnect( wxEVT_CHAR, wxKeyEventHandler( MovieImportDialog::OnTextKeyPress ), NULL, this );
	PixelSizeText->Disconnect( wxEVT_COMMAND_TEXT_UPDATED, wxCommandEventHandler( MovieImportDialog::TextChanged ), NULL, this );
	CsText->Disconnect( wxEVT_CHAR, wxKeyEventHandler( MovieImportDialog::OnTextKeyPress ), NULL, this );
	CsText->Disconnect( wxEVT_COMMAND_TEXT_UPDATED, wxCommandEventHandler( MovieImportDialog::TextChanged ), NULL, this );
	DoseText->Disconnect( wxEVT_CHAR, wxKeyEventHandler( MovieImportDialog::OnTextKeyPress ), NULL, this );
	DoseText->Disconnect( wxEVT_COMMAND_TEXT_UPDATED, wxCommandEventHandler( MovieImportDialog::TextChanged ), NULL, this );
	m_button13->Disconnect( wxEVT_COMMAND_BUTTON_CLICKED, wxCommandEventHandler( MovieImportDialog::CancelClick ), NULL, this );
	ImportButton->Disconnect( wxEVT_COMMAND_BUTTON_CLICKED, wxCommandEventHandler( MovieImportDialog::ImportClick ), NULL, this );
	
}

ImageImportDialog::ImageImportDialog( wxWindow* parent, wxWindowID id, const wxString& title, const wxPoint& pos, const wxSize& size, long style ) : wxDialog( parent, id, title, pos, size, style )
{
	this->SetSizeHints( wxDefaultSize, wxDefaultSize );
	
	wxBoxSizer* bSizer26;
	bSizer26 = new wxBoxSizer( wxVERTICAL );
	
	PathListCtrl = new wxListCtrl( this, wxID_ANY, wxDefaultPosition, wxDefaultSize, wxLC_LIST|wxLC_NO_HEADER|wxLC_REPORT|wxVSCROLL );
	bSizer26->Add( PathListCtrl, 100, wxALL|wxEXPAND, 5 );
	
	wxBoxSizer* bSizer27;
	bSizer27 = new wxBoxSizer( wxHORIZONTAL );
	
	m_button10 = new wxButton( this, wxID_ANY, wxT("Add Files"), wxDefaultPosition, wxDefaultSize, 0 );
	bSizer27->Add( m_button10, 33, wxALL, 5 );
	
	m_button11 = new wxButton( this, wxID_ANY, wxT("Add Directory"), wxDefaultPosition, wxDefaultSize, 0 );
	bSizer27->Add( m_button11, 33, wxALL, 5 );
	
	ClearButton = new wxButton( this, wxID_ANY, wxT("Clear"), wxDefaultPosition, wxDefaultSize, 0 );
	bSizer27->Add( ClearButton, 33, wxALL, 5 );
	
	
	bSizer26->Add( bSizer27, 0, wxEXPAND, 5 );
	
	m_staticline7 = new wxStaticLine( this, wxID_ANY, wxDefaultPosition, wxDefaultSize, wxLI_HORIZONTAL );
	bSizer26->Add( m_staticline7, 0, wxEXPAND | wxALL, 5 );
	
	wxBoxSizer* bSizer28;
	bSizer28 = new wxBoxSizer( wxHORIZONTAL );
	
	m_staticText19 = new wxStaticText( this, wxID_ANY, wxT("Voltage (kV) :"), wxDefaultPosition, wxDefaultSize, 0 );
	m_staticText19->Wrap( -1 );
	bSizer28->Add( m_staticText19, 50, wxALIGN_CENTER_VERTICAL|wxALL, 5 );
	
	VoltageCombo = new wxComboBox( this, wxID_ANY, wxEmptyString, wxDefaultPosition, wxDefaultSize, 0, NULL, 0 );
	VoltageCombo->Append( wxT("300") );
	VoltageCombo->Append( wxT("200") );
	VoltageCombo->Append( wxT("120") );
	bSizer28->Add( VoltageCombo, 50, wxALIGN_CENTER_VERTICAL|wxALL, 5 );
	
	
	bSizer26->Add( bSizer28, 0, wxEXPAND, 5 );
	
	wxBoxSizer* bSizer29;
	bSizer29 = new wxBoxSizer( wxHORIZONTAL );
	
	m_staticText20 = new wxStaticText( this, wxID_ANY, wxT("Pixel Size (Å) :"), wxDefaultPosition, wxDefaultSize, 0 );
	m_staticText20->Wrap( -1 );
	bSizer29->Add( m_staticText20, 50, wxALIGN_CENTER_VERTICAL|wxALL, 5 );
	
	PixelSizeText = new wxTextCtrl( this, wxID_ANY, wxEmptyString, wxDefaultPosition, wxDefaultSize, 0 );
	bSizer29->Add( PixelSizeText, 50, wxALIGN_CENTER_VERTICAL|wxALL, 5 );
	
	
	bSizer26->Add( bSizer29, 0, wxEXPAND, 5 );
	
	wxBoxSizer* bSizer30;
	bSizer30 = new wxBoxSizer( wxHORIZONTAL );
	
	m_staticText21 = new wxStaticText( this, wxID_ANY, wxT("Spherical Aberration (mm) :"), wxDefaultPosition, wxDefaultSize, 0 );
	m_staticText21->Wrap( -1 );
	bSizer30->Add( m_staticText21, 50, wxALIGN_CENTER_VERTICAL|wxALL, 5 );
	
	CsText = new wxTextCtrl( this, wxID_ANY, wxEmptyString, wxDefaultPosition, wxDefaultSize, 0 );
	bSizer30->Add( CsText, 50, wxALIGN_CENTER_VERTICAL|wxALL, 5 );
	
	
	bSizer26->Add( bSizer30, 0, wxEXPAND, 5 );
	
	m_staticline8 = new wxStaticLine( this, wxID_ANY, wxDefaultPosition, wxDefaultSize, wxLI_HORIZONTAL );
	bSizer26->Add( m_staticline8, 0, wxEXPAND | wxALL, 5 );
	
	wxBoxSizer* bSizer33;
	bSizer33 = new wxBoxSizer( wxHORIZONTAL );
	
	m_button13 = new wxButton( this, wxID_ANY, wxT("Cancel"), wxDefaultPosition, wxDefaultSize, 0 );
	bSizer33->Add( m_button13, 0, wxALL, 5 );
	
	
	bSizer33->Add( 0, 0, 1, wxEXPAND, 5 );
	
	
	bSizer33->Add( 0, 0, 1, wxEXPAND, 5 );
	
	ImportButton = new wxButton( this, wxID_ANY, wxT("Import"), wxDefaultPosition, wxDefaultSize, 0 );
	ImportButton->Enable( false );
	
	bSizer33->Add( ImportButton, 0, wxALL, 5 );
	
	
	bSizer26->Add( bSizer33, 1, wxEXPAND, 5 );
	
	
	this->SetSizer( bSizer26 );
	this->Layout();
	
	this->Centre( wxBOTH );
	
	// Connect Events
	m_button10->Connect( wxEVT_COMMAND_BUTTON_CLICKED, wxCommandEventHandler( ImageImportDialog::AddFilesClick ), NULL, this );
	m_button11->Connect( wxEVT_COMMAND_BUTTON_CLICKED, wxCommandEventHandler( ImageImportDialog::AddDirectoryClick ), NULL, this );
	ClearButton->Connect( wxEVT_COMMAND_BUTTON_CLICKED, wxCommandEventHandler( ImageImportDialog::ClearClick ), NULL, this );
	VoltageCombo->Connect( wxEVT_CHAR, wxKeyEventHandler( ImageImportDialog::OnTextKeyPress ), NULL, this );
	VoltageCombo->Connect( wxEVT_COMMAND_TEXT_UPDATED, wxCommandEventHandler( ImageImportDialog::TextChanged ), NULL, this );
	PixelSizeText->Connect( wxEVT_CHAR, wxKeyEventHandler( ImageImportDialog::OnTextKeyPress ), NULL, this );
	PixelSizeText->Connect( wxEVT_COMMAND_TEXT_UPDATED, wxCommandEventHandler( ImageImportDialog::TextChanged ), NULL, this );
	CsText->Connect( wxEVT_CHAR, wxKeyEventHandler( ImageImportDialog::OnTextKeyPress ), NULL, this );
	CsText->Connect( wxEVT_COMMAND_TEXT_UPDATED, wxCommandEventHandler( ImageImportDialog::TextChanged ), NULL, this );
	m_button13->Connect( wxEVT_COMMAND_BUTTON_CLICKED, wxCommandEventHandler( ImageImportDialog::CancelClick ), NULL, this );
	ImportButton->Connect( wxEVT_COMMAND_BUTTON_CLICKED, wxCommandEventHandler( ImageImportDialog::ImportClick ), NULL, this );
}

ImageImportDialog::~ImageImportDialog()
{
	// Disconnect Events
	m_button10->Disconnect( wxEVT_COMMAND_BUTTON_CLICKED, wxCommandEventHandler( ImageImportDialog::AddFilesClick ), NULL, this );
	m_button11->Disconnect( wxEVT_COMMAND_BUTTON_CLICKED, wxCommandEventHandler( ImageImportDialog::AddDirectoryClick ), NULL, this );
	ClearButton->Disconnect( wxEVT_COMMAND_BUTTON_CLICKED, wxCommandEventHandler( ImageImportDialog::ClearClick ), NULL, this );
	VoltageCombo->Disconnect( wxEVT_CHAR, wxKeyEventHandler( ImageImportDialog::OnTextKeyPress ), NULL, this );
	VoltageCombo->Disconnect( wxEVT_COMMAND_TEXT_UPDATED, wxCommandEventHandler( ImageImportDialog::TextChanged ), NULL, this );
	PixelSizeText->Disconnect( wxEVT_CHAR, wxKeyEventHandler( ImageImportDialog::OnTextKeyPress ), NULL, this );
	PixelSizeText->Disconnect( wxEVT_COMMAND_TEXT_UPDATED, wxCommandEventHandler( ImageImportDialog::TextChanged ), NULL, this );
	CsText->Disconnect( wxEVT_CHAR, wxKeyEventHandler( ImageImportDialog::OnTextKeyPress ), NULL, this );
	CsText->Disconnect( wxEVT_COMMAND_TEXT_UPDATED, wxCommandEventHandler( ImageImportDialog::TextChanged ), NULL, this );
	m_button13->Disconnect( wxEVT_COMMAND_BUTTON_CLICKED, wxCommandEventHandler( ImageImportDialog::CancelClick ), NULL, this );
	ImportButton->Disconnect( wxEVT_COMMAND_BUTTON_CLICKED, wxCommandEventHandler( ImageImportDialog::ImportClick ), NULL, this );
	
}

AssetParentPanel::AssetParentPanel( wxWindow* parent, wxWindowID id, const wxPoint& pos, const wxSize& size, long style ) : wxPanel( parent, id, pos, size, style )
{
	this->SetMinSize( wxSize( 680,400 ) );
	
	wxBoxSizer* bSizer15;
	bSizer15 = new wxBoxSizer( wxVERTICAL );
	
	m_staticline5 = new wxStaticLine( this, wxID_ANY, wxDefaultPosition, wxDefaultSize, wxLI_HORIZONTAL );
	bSizer15->Add( m_staticline5, 0, wxEXPAND | wxALL, 5 );
	
	wxBoxSizer* bSizer20;
	bSizer20 = new wxBoxSizer( wxHORIZONTAL );
	
	m_splitter2 = new wxSplitterWindow( this, wxID_ANY, wxDefaultPosition, wxDefaultSize, wxSP_3D );
	m_splitter2->SetSashGravity( 0.2 );
	m_splitter2->Connect( wxEVT_IDLE, wxIdleEventHandler( AssetParentPanel::m_splitter2OnIdle ), NULL, this );
	
	m_panel4 = new wxPanel( m_splitter2, wxID_ANY, wxDefaultPosition, wxDefaultSize, wxTAB_TRAVERSAL );
	wxBoxSizer* bSizer27;
	bSizer27 = new wxBoxSizer( wxVERTICAL );
	
	m_staticText18 = new wxStaticText( m_panel4, wxID_ANY, wxT("Groups:"), wxDefaultPosition, wxDefaultSize, 0 );
	m_staticText18->Wrap( -1 );
	bSizer27->Add( m_staticText18, 0, wxALL, 5 );
	
	wxBoxSizer* bSizer221;
	bSizer221 = new wxBoxSizer( wxHORIZONTAL );
	
	GroupListBox = new wxListCtrl( m_panel4, wxID_ANY, wxDefaultPosition, wxDefaultSize, wxLC_EDIT_LABELS|wxLC_NO_HEADER|wxLC_REPORT|wxLC_SINGLE_SEL );
	bSizer221->Add( GroupListBox, 1, wxALL|wxEXPAND, 5 );
	
	
	bSizer27->Add( bSizer221, 1, wxEXPAND, 5 );
	
	wxBoxSizer* bSizer49;
	bSizer49 = new wxBoxSizer( wxHORIZONTAL );
	
	AddGroupButton = new wxButton( m_panel4, wxID_ANY, wxT("Add"), wxDefaultPosition, wxDefaultSize, 0 );
	bSizer49->Add( AddGroupButton, 0, wxALL, 5 );
	
	RenameGroupButton = new wxButton( m_panel4, wxID_ANY, wxT("Rename"), wxDefaultPosition, wxDefaultSize, 0 );
	bSizer49->Add( RenameGroupButton, 0, wxALL, 5 );
	
	RemoveGroupButton = new wxButton( m_panel4, wxID_ANY, wxT("Remove"), wxDefaultPosition, wxDefaultSize, 0 );
	bSizer49->Add( RemoveGroupButton, 0, wxALL, 5 );
	
	
	bSizer27->Add( bSizer49, 0, wxEXPAND, 5 );
	
	
	m_panel4->SetSizer( bSizer27 );
	m_panel4->Layout();
	bSizer27->Fit( m_panel4 );
	m_panel3 = new wxPanel( m_splitter2, wxID_ANY, wxDefaultPosition, wxDefaultSize, wxTAB_TRAVERSAL );
	wxBoxSizer* bSizer30;
	bSizer30 = new wxBoxSizer( wxVERTICAL );
	
	AssetTypeText = new wxStaticText( m_panel3, wxID_ANY, wxT("Asset Type :"), wxDefaultPosition, wxDefaultSize, 0 );
	AssetTypeText->Wrap( -1 );
	bSizer30->Add( AssetTypeText, 0, wxALL, 5 );
	
	wxBoxSizer* bSizer25;
	bSizer25 = new wxBoxSizer( wxVERTICAL );
	
	ContentsListBox = new wxListCtrl( m_panel3, wxID_ANY, wxDefaultPosition, wxDefaultSize, wxLC_ALIGN_LEFT|wxLC_NO_SORT_HEADER|wxLC_REPORT|wxLC_VRULES );
	bSizer25->Add( ContentsListBox, 100, wxALL|wxEXPAND, 5 );
	
	wxBoxSizer* bSizer28;
	bSizer28 = new wxBoxSizer( wxHORIZONTAL );
	
	ImportAsset = new wxButton( m_panel3, wxID_ANY, wxT("Import"), wxDefaultPosition, wxDefaultSize, 0 );
	bSizer28->Add( ImportAsset, 0, wxALL|wxEXPAND, 5 );
	
	RemoveSelectedAssetButton = new wxButton( m_panel3, wxID_ANY, wxT("Remove"), wxDefaultPosition, wxDefaultSize, 0 );
	RemoveSelectedAssetButton->Enable( false );
	
	bSizer28->Add( RemoveSelectedAssetButton, 0, wxALL|wxEXPAND, 5 );
	
	RemoveAllAssetsButton = new wxButton( m_panel3, wxID_ANY, wxT("Remove All"), wxDefaultPosition, wxDefaultSize, 0 );
	RemoveAllAssetsButton->Enable( false );
	
	bSizer28->Add( RemoveAllAssetsButton, 0, wxALL|wxEXPAND, 5 );
	
	AddSelectedAssetButton = new wxButton( m_panel3, wxID_ANY, wxT("Add To Group"), wxDefaultPosition, wxDefaultSize, 0 );
	AddSelectedAssetButton->Enable( false );
	
	bSizer28->Add( AddSelectedAssetButton, 0, wxALL|wxEXPAND, 5 );
	
	
	bSizer25->Add( bSizer28, 0, wxEXPAND, 5 );
	
	
	bSizer30->Add( bSizer25, 100, wxEXPAND, 5 );
	
	
	m_panel3->SetSizer( bSizer30 );
	m_panel3->Layout();
	bSizer30->Fit( m_panel3 );
	m_splitter2->SplitVertically( m_panel4, m_panel3, 300 );
	bSizer20->Add( m_splitter2, 100, wxEXPAND, 5 );
	
	
	bSizer15->Add( bSizer20, 1, wxEXPAND, 5 );
	
	m_staticline6 = new wxStaticLine( this, wxID_ANY, wxDefaultPosition, wxDefaultSize, wxLI_HORIZONTAL );
	bSizer15->Add( m_staticline6, 0, wxEXPAND | wxALL, 5 );
	
	wxBoxSizer* bSizer34;
	bSizer34 = new wxBoxSizer( wxHORIZONTAL );
	
	wxGridSizer* gSizer1;
	gSizer1 = new wxGridSizer( 4, 6, 0, 0 );
	
	Label0Title = new wxStaticText( this, wxID_ANY, wxT("Label 0 :"), wxDefaultPosition, wxDefaultSize, 0 );
	Label0Title->Wrap( -1 );
	Label0Title->SetFont( wxFont( wxNORMAL_FONT->GetPointSize(), 70, 90, 92, false, wxEmptyString ) );
	
	gSizer1->Add( Label0Title, 0, wxALIGN_RIGHT|wxALL, 5 );
	
	Label0Text = new wxStaticText( this, wxID_ANY, wxT("-"), wxDefaultPosition, wxDefaultSize, 0 );
	Label0Text->Wrap( -1 );
	gSizer1->Add( Label0Text, 0, wxALIGN_LEFT|wxALL, 5 );
	
	
	gSizer1->Add( 0, 0, 1, wxEXPAND, 5 );
	
	
	gSizer1->Add( 0, 0, 1, wxEXPAND, 5 );
	
	
	gSizer1->Add( 0, 0, 1, wxEXPAND, 5 );
	
	
	gSizer1->Add( 0, 0, 1, wxEXPAND, 5 );
	
	Label1Title = new wxStaticText( this, wxID_ANY, wxT("Label 1 :"), wxDefaultPosition, wxDefaultSize, 0 );
	Label1Title->Wrap( -1 );
	Label1Title->SetFont( wxFont( wxNORMAL_FONT->GetPointSize(), 70, 90, 92, false, wxEmptyString ) );
	
	gSizer1->Add( Label1Title, 0, wxALIGN_RIGHT|wxALL, 5 );
	
	Label1Text = new wxStaticText( this, wxID_ANY, wxT("-"), wxDefaultPosition, wxDefaultSize, 0 );
	Label1Text->Wrap( -1 );
	gSizer1->Add( Label1Text, 0, wxALL, 5 );
	
	Label2Title = new wxStaticText( this, wxID_ANY, wxT("Label 2 :"), wxDefaultPosition, wxDefaultSize, 0 );
	Label2Title->Wrap( -1 );
	Label2Title->SetFont( wxFont( wxNORMAL_FONT->GetPointSize(), 70, 90, 92, false, wxEmptyString ) );
	
	gSizer1->Add( Label2Title, 0, wxALIGN_RIGHT|wxALL, 5 );
	
	Label2Text = new wxStaticText( this, wxID_ANY, wxT("-"), wxDefaultPosition, wxDefaultSize, 0 );
	Label2Text->Wrap( -1 );
	gSizer1->Add( Label2Text, 0, wxALIGN_LEFT|wxALL, 5 );
	
	Label3Title = new wxStaticText( this, wxID_ANY, wxT("Label 3 :"), wxDefaultPosition, wxDefaultSize, 0 );
	Label3Title->Wrap( -1 );
	Label3Title->SetFont( wxFont( wxNORMAL_FONT->GetPointSize(), 70, 90, 92, false, wxEmptyString ) );
	
	gSizer1->Add( Label3Title, 0, wxALIGN_RIGHT|wxALL, 5 );
	
	Label3Text = new wxStaticText( this, wxID_ANY, wxT("-"), wxDefaultPosition, wxDefaultSize, 0 );
	Label3Text->Wrap( -1 );
	gSizer1->Add( Label3Text, 0, wxALIGN_LEFT|wxALL, 5 );
	
	Label4Title = new wxStaticText( this, wxID_ANY, wxT("Label 4 :"), wxDefaultPosition, wxDefaultSize, 0 );
	Label4Title->Wrap( -1 );
	Label4Title->SetFont( wxFont( wxNORMAL_FONT->GetPointSize(), 70, 90, 92, false, wxEmptyString ) );
	
	gSizer1->Add( Label4Title, 0, wxALIGN_RIGHT|wxALL, 5 );
	
	Label4Text = new wxStaticText( this, wxID_ANY, wxT("-"), wxDefaultPosition, wxDefaultSize, 0 );
	Label4Text->Wrap( -1 );
	gSizer1->Add( Label4Text, 0, wxALIGN_LEFT|wxALL, 5 );
	
	Label5Title = new wxStaticText( this, wxID_ANY, wxT("Label 5 :"), wxDefaultPosition, wxDefaultSize, 0 );
	Label5Title->Wrap( -1 );
	Label5Title->SetFont( wxFont( wxNORMAL_FONT->GetPointSize(), 70, 90, 92, false, wxEmptyString ) );
	
	gSizer1->Add( Label5Title, 0, wxALIGN_RIGHT|wxALL, 5 );
	
	Label5Text = new wxStaticText( this, wxID_ANY, wxT("-"), wxDefaultPosition, wxDefaultSize, 0 );
	Label5Text->Wrap( -1 );
	gSizer1->Add( Label5Text, 0, wxALL, 5 );
	
	Label6Title = new wxStaticText( this, wxID_ANY, wxT("Label 6 :"), wxDefaultPosition, wxDefaultSize, 0 );
	Label6Title->Wrap( -1 );
	Label6Title->SetFont( wxFont( wxNORMAL_FONT->GetPointSize(), 70, 90, 92, false, wxEmptyString ) );
	
	gSizer1->Add( Label6Title, 0, wxALIGN_RIGHT|wxALL, 5 );
	
	Label6Text = new wxStaticText( this, wxID_ANY, wxT("-"), wxDefaultPosition, wxDefaultSize, 0 );
	Label6Text->Wrap( -1 );
	gSizer1->Add( Label6Text, 0, wxALIGN_LEFT|wxALL, 5 );
	
	Label7Title = new wxStaticText( this, wxID_ANY, wxT("Label 7 :"), wxDefaultPosition, wxDefaultSize, 0 );
	Label7Title->Wrap( -1 );
	Label7Title->SetFont( wxFont( wxNORMAL_FONT->GetPointSize(), 70, 90, 92, false, wxEmptyString ) );
	
	gSizer1->Add( Label7Title, 0, wxALIGN_RIGHT|wxALL, 5 );
	
	Label7Text = new wxStaticText( this, wxID_ANY, wxT("-"), wxDefaultPosition, wxDefaultSize, 0 );
	Label7Text->Wrap( -1 );
	gSizer1->Add( Label7Text, 0, wxALIGN_LEFT|wxALL, 5 );
	
	Label8Title = new wxStaticText( this, wxID_ANY, wxT("Label 8 :"), wxDefaultPosition, wxDefaultSize, 0 );
	Label8Title->Wrap( -1 );
	Label8Title->SetFont( wxFont( wxNORMAL_FONT->GetPointSize(), 70, 90, 92, false, wxEmptyString ) );
	
	gSizer1->Add( Label8Title, 0, wxALIGN_RIGHT|wxALL, 5 );
	
	Label8Text = new wxStaticText( this, wxID_ANY, wxT("-"), wxDefaultPosition, wxDefaultSize, 0 );
	Label8Text->Wrap( -1 );
	gSizer1->Add( Label8Text, 0, wxALL, 5 );
	
	Label9Title = new wxStaticText( this, wxID_ANY, wxT("Label 9 :"), wxDefaultPosition, wxDefaultSize, 0 );
	Label9Title->Wrap( -1 );
	Label9Title->SetFont( wxFont( wxNORMAL_FONT->GetPointSize(), 70, 90, 92, false, wxEmptyString ) );
	
	gSizer1->Add( Label9Title, 0, wxALIGN_RIGHT|wxALL, 5 );
	
	Label9Text = new wxStaticText( this, wxID_ANY, wxT("-"), wxDefaultPosition, wxDefaultSize, 0 );
	Label9Text->Wrap( -1 );
	gSizer1->Add( Label9Text, 0, wxALL, 5 );
	
	
	bSizer34->Add( gSizer1, 90, wxEXPAND, 5 );
	
	
	bSizer15->Add( bSizer34, 0, wxEXPAND, 5 );
	
	
	this->SetSizer( bSizer15 );
	this->Layout();
	
	// Connect Events
	this->Connect( wxEVT_UPDATE_UI, wxUpdateUIEventHandler( AssetParentPanel::OnUpdateUI ) );
	GroupListBox->Connect( wxEVT_LEFT_DCLICK, wxMouseEventHandler( AssetParentPanel::MouseCheckGroupsVeto ), NULL, this );
	GroupListBox->Connect( wxEVT_LEFT_DOWN, wxMouseEventHandler( AssetParentPanel::MouseCheckGroupsVeto ), NULL, this );
	GroupListBox->Connect( wxEVT_LEFT_UP, wxMouseEventHandler( AssetParentPanel::MouseVeto ), NULL, this );
	GroupListBox->Connect( wxEVT_COMMAND_LIST_BEGIN_LABEL_EDIT, wxListEventHandler( AssetParentPanel::OnBeginEdit ), NULL, this );
	GroupListBox->Connect( wxEVT_COMMAND_LIST_END_LABEL_EDIT, wxListEventHandler( AssetParentPanel::OnEndEdit ), NULL, this );
	GroupListBox->Connect( wxEVT_COMMAND_LIST_ITEM_ACTIVATED, wxListEventHandler( AssetParentPanel::OnGroupActivated ), NULL, this );
	GroupListBox->Connect( wxEVT_COMMAND_LIST_ITEM_FOCUSED, wxListEventHandler( AssetParentPanel::OnGroupFocusChange ), NULL, this );
	GroupListBox->Connect( wxEVT_MIDDLE_DCLICK, wxMouseEventHandler( AssetParentPanel::MouseVeto ), NULL, this );
	GroupListBox->Connect( wxEVT_MIDDLE_DOWN, wxMouseEventHandler( AssetParentPanel::MouseVeto ), NULL, this );
	GroupListBox->Connect( wxEVT_MIDDLE_UP, wxMouseEventHandler( AssetParentPanel::MouseVeto ), NULL, this );
	GroupListBox->Connect( wxEVT_MOTION, wxMouseEventHandler( AssetParentPanel::MouseVeto ), NULL, this );
	GroupListBox->Connect( wxEVT_RIGHT_DCLICK, wxMouseEventHandler( AssetParentPanel::MouseVeto ), NULL, this );
	GroupListBox->Connect( wxEVT_RIGHT_DOWN, wxMouseEventHandler( AssetParentPanel::MouseVeto ), NULL, this );
	GroupListBox->Connect( wxEVT_RIGHT_UP, wxMouseEventHandler( AssetParentPanel::MouseVeto ), NULL, this );
	AddGroupButton->Connect( wxEVT_COMMAND_BUTTON_CLICKED, wxCommandEventHandler( AssetParentPanel::NewGroupClick ), NULL, this );
	RenameGroupButton->Connect( wxEVT_COMMAND_BUTTON_CLICKED, wxCommandEventHandler( AssetParentPanel::RenameGroupClick ), NULL, this );
	RemoveGroupButton->Connect( wxEVT_COMMAND_BUTTON_CLICKED, wxCommandEventHandler( AssetParentPanel::RemoveGroupClick ), NULL, this );
	ContentsListBox->Connect( wxEVT_LEFT_DCLICK, wxMouseEventHandler( AssetParentPanel::MouseCheckContentsVeto ), NULL, this );
	ContentsListBox->Connect( wxEVT_LEFT_DOWN, wxMouseEventHandler( AssetParentPanel::MouseCheckContentsVeto ), NULL, this );
	ContentsListBox->Connect( wxEVT_LEFT_UP, wxMouseEventHandler( AssetParentPanel::MouseVeto ), NULL, this );
	ContentsListBox->Connect( wxEVT_COMMAND_LIST_BEGIN_DRAG, wxListEventHandler( AssetParentPanel::OnBeginContentsDrag ), NULL, this );
	ContentsListBox->Connect( wxEVT_COMMAND_LIST_ITEM_SELECTED, wxListEventHandler( AssetParentPanel::OnContentsSelected ), NULL, this );
	ContentsListBox->Connect( wxEVT_MIDDLE_DCLICK, wxMouseEventHandler( AssetParentPanel::MouseVeto ), NULL, this );
	ContentsListBox->Connect( wxEVT_MIDDLE_DOWN, wxMouseEventHandler( AssetParentPanel::MouseVeto ), NULL, this );
	ContentsListBox->Connect( wxEVT_MIDDLE_UP, wxMouseEventHandler( AssetParentPanel::MouseVeto ), NULL, this );
	ContentsListBox->Connect( wxEVT_MOTION, wxMouseEventHandler( AssetParentPanel::OnMotion ), NULL, this );
	ContentsListBox->Connect( wxEVT_RIGHT_DCLICK, wxMouseEventHandler( AssetParentPanel::MouseVeto ), NULL, this );
	ContentsListBox->Connect( wxEVT_RIGHT_DOWN, wxMouseEventHandler( AssetParentPanel::MouseVeto ), NULL, this );
	ContentsListBox->Connect( wxEVT_RIGHT_UP, wxMouseEventHandler( AssetParentPanel::MouseVeto ), NULL, this );
	ImportAsset->Connect( wxEVT_COMMAND_BUTTON_CLICKED, wxCommandEventHandler( AssetParentPanel::ImportAssetClick ), NULL, this );
	RemoveSelectedAssetButton->Connect( wxEVT_COMMAND_BUTTON_CLICKED, wxCommandEventHandler( AssetParentPanel::RemoveAssetClick ), NULL, this );
	RemoveAllAssetsButton->Connect( wxEVT_COMMAND_BUTTON_CLICKED, wxCommandEventHandler( AssetParentPanel::RemoveAllAssetsClick ), NULL, this );
	AddSelectedAssetButton->Connect( wxEVT_COMMAND_BUTTON_CLICKED, wxCommandEventHandler( AssetParentPanel::AddSelectedAssetClick ), NULL, this );
}

AssetParentPanel::~AssetParentPanel()
{
	// Disconnect Events
	this->Disconnect( wxEVT_UPDATE_UI, wxUpdateUIEventHandler( AssetParentPanel::OnUpdateUI ) );
	GroupListBox->Disconnect( wxEVT_LEFT_DCLICK, wxMouseEventHandler( AssetParentPanel::MouseCheckGroupsVeto ), NULL, this );
	GroupListBox->Disconnect( wxEVT_LEFT_DOWN, wxMouseEventHandler( AssetParentPanel::MouseCheckGroupsVeto ), NULL, this );
	GroupListBox->Disconnect( wxEVT_LEFT_UP, wxMouseEventHandler( AssetParentPanel::MouseVeto ), NULL, this );
	GroupListBox->Disconnect( wxEVT_COMMAND_LIST_BEGIN_LABEL_EDIT, wxListEventHandler( AssetParentPanel::OnBeginEdit ), NULL, this );
	GroupListBox->Disconnect( wxEVT_COMMAND_LIST_END_LABEL_EDIT, wxListEventHandler( AssetParentPanel::OnEndEdit ), NULL, this );
	GroupListBox->Disconnect( wxEVT_COMMAND_LIST_ITEM_ACTIVATED, wxListEventHandler( AssetParentPanel::OnGroupActivated ), NULL, this );
	GroupListBox->Disconnect( wxEVT_COMMAND_LIST_ITEM_FOCUSED, wxListEventHandler( AssetParentPanel::OnGroupFocusChange ), NULL, this );
	GroupListBox->Disconnect( wxEVT_MIDDLE_DCLICK, wxMouseEventHandler( AssetParentPanel::MouseVeto ), NULL, this );
	GroupListBox->Disconnect( wxEVT_MIDDLE_DOWN, wxMouseEventHandler( AssetParentPanel::MouseVeto ), NULL, this );
	GroupListBox->Disconnect( wxEVT_MIDDLE_UP, wxMouseEventHandler( AssetParentPanel::MouseVeto ), NULL, this );
	GroupListBox->Disconnect( wxEVT_MOTION, wxMouseEventHandler( AssetParentPanel::MouseVeto ), NULL, this );
	GroupListBox->Disconnect( wxEVT_RIGHT_DCLICK, wxMouseEventHandler( AssetParentPanel::MouseVeto ), NULL, this );
	GroupListBox->Disconnect( wxEVT_RIGHT_DOWN, wxMouseEventHandler( AssetParentPanel::MouseVeto ), NULL, this );
	GroupListBox->Disconnect( wxEVT_RIGHT_UP, wxMouseEventHandler( AssetParentPanel::MouseVeto ), NULL, this );
	AddGroupButton->Disconnect( wxEVT_COMMAND_BUTTON_CLICKED, wxCommandEventHandler( AssetParentPanel::NewGroupClick ), NULL, this );
	RenameGroupButton->Disconnect( wxEVT_COMMAND_BUTTON_CLICKED, wxCommandEventHandler( AssetParentPanel::RenameGroupClick ), NULL, this );
	RemoveGroupButton->Disconnect( wxEVT_COMMAND_BUTTON_CLICKED, wxCommandEventHandler( AssetParentPanel::RemoveGroupClick ), NULL, this );
	ContentsListBox->Disconnect( wxEVT_LEFT_DCLICK, wxMouseEventHandler( AssetParentPanel::MouseCheckContentsVeto ), NULL, this );
	ContentsListBox->Disconnect( wxEVT_LEFT_DOWN, wxMouseEventHandler( AssetParentPanel::MouseCheckContentsVeto ), NULL, this );
	ContentsListBox->Disconnect( wxEVT_LEFT_UP, wxMouseEventHandler( AssetParentPanel::MouseVeto ), NULL, this );
	ContentsListBox->Disconnect( wxEVT_COMMAND_LIST_BEGIN_DRAG, wxListEventHandler( AssetParentPanel::OnBeginContentsDrag ), NULL, this );
	ContentsListBox->Disconnect( wxEVT_COMMAND_LIST_ITEM_SELECTED, wxListEventHandler( AssetParentPanel::OnContentsSelected ), NULL, this );
	ContentsListBox->Disconnect( wxEVT_MIDDLE_DCLICK, wxMouseEventHandler( AssetParentPanel::MouseVeto ), NULL, this );
	ContentsListBox->Disconnect( wxEVT_MIDDLE_DOWN, wxMouseEventHandler( AssetParentPanel::MouseVeto ), NULL, this );
	ContentsListBox->Disconnect( wxEVT_MIDDLE_UP, wxMouseEventHandler( AssetParentPanel::MouseVeto ), NULL, this );
	ContentsListBox->Disconnect( wxEVT_MOTION, wxMouseEventHandler( AssetParentPanel::OnMotion ), NULL, this );
	ContentsListBox->Disconnect( wxEVT_RIGHT_DCLICK, wxMouseEventHandler( AssetParentPanel::MouseVeto ), NULL, this );
	ContentsListBox->Disconnect( wxEVT_RIGHT_DOWN, wxMouseEventHandler( AssetParentPanel::MouseVeto ), NULL, this );
	ContentsListBox->Disconnect( wxEVT_RIGHT_UP, wxMouseEventHandler( AssetParentPanel::MouseVeto ), NULL, this );
	ImportAsset->Disconnect( wxEVT_COMMAND_BUTTON_CLICKED, wxCommandEventHandler( AssetParentPanel::ImportAssetClick ), NULL, this );
	RemoveSelectedAssetButton->Disconnect( wxEVT_COMMAND_BUTTON_CLICKED, wxCommandEventHandler( AssetParentPanel::RemoveAssetClick ), NULL, this );
	RemoveAllAssetsButton->Disconnect( wxEVT_COMMAND_BUTTON_CLICKED, wxCommandEventHandler( AssetParentPanel::RemoveAllAssetsClick ), NULL, this );
	AddSelectedAssetButton->Disconnect( wxEVT_COMMAND_BUTTON_CLICKED, wxCommandEventHandler( AssetParentPanel::AddSelectedAssetClick ), NULL, this );
	
}

RunProfilesPanel::RunProfilesPanel( wxWindow* parent, wxWindowID id, const wxPoint& pos, const wxSize& size, long style ) : wxPanel( parent, id, pos, size, style )
{
	this->SetMinSize( wxSize( 680,400 ) );
	
	wxBoxSizer* bSizer56;
	bSizer56 = new wxBoxSizer( wxVERTICAL );
	
	m_staticline12 = new wxStaticLine( this, wxID_ANY, wxDefaultPosition, wxDefaultSize, wxLI_HORIZONTAL );
	bSizer56->Add( m_staticline12, 0, wxEXPAND | wxALL, 5 );
	
	m_splitter5 = new wxSplitterWindow( this, wxID_ANY, wxDefaultPosition, wxDefaultSize, wxSP_3D );
	m_splitter5->Connect( wxEVT_IDLE, wxIdleEventHandler( RunProfilesPanel::m_splitter5OnIdle ), NULL, this );
	
	ProfilesPanel = new wxPanel( m_splitter5, wxID_ANY, wxDefaultPosition, wxDefaultSize, wxTAB_TRAVERSAL );
	wxBoxSizer* bSizer57;
	bSizer57 = new wxBoxSizer( wxVERTICAL );
	
	ProfilesListBox = new wxListCtrl( ProfilesPanel, wxID_ANY, wxDefaultPosition, wxDefaultSize, wxLC_EDIT_LABELS|wxLC_NO_HEADER|wxLC_REPORT|wxLC_SINGLE_SEL );
	bSizer57->Add( ProfilesListBox, 1, wxALL|wxEXPAND, 5 );
	
	wxBoxSizer* bSizer431;
	bSizer431 = new wxBoxSizer( wxHORIZONTAL );
	
	AddProfileButton = new wxButton( ProfilesPanel, wxID_ANY, wxT("Add"), wxDefaultPosition, wxDefaultSize, 0 );
	bSizer431->Add( AddProfileButton, 1, wxALL, 5 );
	
	RenameProfileButton = new wxButton( ProfilesPanel, wxID_ANY, wxT("Rename"), wxDefaultPosition, wxDefaultSize, 0 );
	bSizer431->Add( RenameProfileButton, 0, wxALL, 5 );
	
	RemoveProfileButton = new wxButton( ProfilesPanel, wxID_ANY, wxT("Remove"), wxDefaultPosition, wxDefaultSize, 0 );
	bSizer431->Add( RemoveProfileButton, 1, wxALL, 5 );
	
	
	bSizer57->Add( bSizer431, 0, wxALIGN_LEFT, 5 );
	
	
	ProfilesPanel->SetSizer( bSizer57 );
	ProfilesPanel->Layout();
	bSizer57->Fit( ProfilesPanel );
	CommandsPanel = new wxPanel( m_splitter5, wxID_ANY, wxDefaultPosition, wxDefaultSize, wxTAB_TRAVERSAL );
	CommandsPanel->Enable( false );
	
	wxBoxSizer* bSizer44;
	bSizer44 = new wxBoxSizer( wxHORIZONTAL );
	
	m_staticline15 = new wxStaticLine( CommandsPanel, wxID_ANY, wxDefaultPosition, wxDefaultSize, wxLI_HORIZONTAL|wxLI_VERTICAL );
	bSizer44->Add( m_staticline15, 0, wxEXPAND | wxALL, 5 );
	
	wxBoxSizer* bSizer37;
	bSizer37 = new wxBoxSizer( wxVERTICAL );
	
	wxBoxSizer* bSizer40;
	bSizer40 = new wxBoxSizer( wxHORIZONTAL );
	
	m_staticText34 = new wxStaticText( CommandsPanel, wxID_ANY, wxT("Total Number of Processes : "), wxDefaultPosition, wxDefaultSize, 0 );
	m_staticText34->Wrap( -1 );
	bSizer40->Add( m_staticText34, 0, wxALIGN_CENTER_VERTICAL|wxALL, 5 );
	
	NumberProcessesStaticText = new wxStaticText( CommandsPanel, wxID_ANY, wxEmptyString, wxDefaultPosition, wxDefaultSize, 0 );
	NumberProcessesStaticText->Wrap( -1 );
	bSizer40->Add( NumberProcessesStaticText, 1, wxALIGN_CENTER_VERTICAL|wxALL, 5 );
	
	
	bSizer37->Add( bSizer40, 0, wxEXPAND, 5 );
	
	wxBoxSizer* bSizer38;
	bSizer38 = new wxBoxSizer( wxVERTICAL );
	
	m_staticText36 = new wxStaticText( CommandsPanel, wxID_ANY, wxT("Manager Command :-"), wxDefaultPosition, wxDefaultSize, 0 );
	m_staticText36->Wrap( -1 );
	bSizer38->Add( m_staticText36, 50, wxALIGN_CENTER_VERTICAL|wxALL, 5 );
	
	ManagerTextCtrl = new wxTextCtrl( CommandsPanel, wxID_ANY, wxEmptyString, wxDefaultPosition, wxDefaultSize, wxHSCROLL|wxTE_WORDWRAP|wxALWAYS_SHOW_SB|wxHSCROLL );
	bSizer38->Add( ManagerTextCtrl, 0, wxALL|wxEXPAND, 5 );
	
	CommandErrorStaticText = new wxStaticText( CommandsPanel, wxID_ANY, wxEmptyString, wxDefaultPosition, wxDefaultSize, wxALIGN_CENTRE );
	CommandErrorStaticText->Wrap( -1 );
	CommandErrorStaticText->SetForegroundColour( wxColour( 180, 0, 0 ) );
	
	bSizer38->Add( CommandErrorStaticText, 0, wxALL|wxEXPAND, 5 );
	
	
	bSizer37->Add( bSizer38, 0, wxEXPAND, 5 );
	
	wxBoxSizer* bSizer43;
	bSizer43 = new wxBoxSizer( wxVERTICAL );
	
	
	bSizer37->Add( bSizer43, 0, wxEXPAND, 5 );
	
	wxBoxSizer* bSizer41;
	bSizer41 = new wxBoxSizer( wxVERTICAL );
	
	CommandsListBox = new wxListCtrl( CommandsPanel, wxID_ANY, wxDefaultPosition, wxDefaultSize, wxLC_NO_SORT_HEADER|wxLC_REPORT|wxLC_SINGLE_SEL );
	bSizer41->Add( CommandsListBox, 1, wxALL|wxEXPAND, 5 );
	
	wxGridSizer* gSizer2;
	gSizer2 = new wxGridSizer( 0, 2, 0, 0 );
	
	wxBoxSizer* bSizer42;
	bSizer42 = new wxBoxSizer( wxHORIZONTAL );
	
	AddCommandButton = new wxButton( CommandsPanel, wxID_ANY, wxT("Add"), wxDefaultPosition, wxDefaultSize, 0 );
	bSizer42->Add( AddCommandButton, 1, wxALL, 5 );
	
	EditCommandButton = new wxButton( CommandsPanel, wxID_ANY, wxT("Edit"), wxDefaultPosition, wxDefaultSize, 0 );
	bSizer42->Add( EditCommandButton, 1, wxALL, 5 );
	
	RemoveCommandButton = new wxButton( CommandsPanel, wxID_ANY, wxT("Remove"), wxDefaultPosition, wxDefaultSize, 0 );
	bSizer42->Add( RemoveCommandButton, 1, wxALL, 5 );
	
	
	gSizer2->Add( bSizer42, 0, wxALIGN_LEFT, 5 );
	
	CommandsSaveButton = new wxButton( CommandsPanel, wxID_ANY, wxT("Save"), wxDefaultPosition, wxDefaultSize, 0 );
	gSizer2->Add( CommandsSaveButton, 0, wxALIGN_RIGHT|wxALL, 5 );
	
	
	bSizer41->Add( gSizer2, 0, wxEXPAND, 5 );
	
	
	bSizer37->Add( bSizer41, 1, wxEXPAND, 5 );
	
	
	bSizer44->Add( bSizer37, 1, wxEXPAND, 5 );
	
	
	CommandsPanel->SetSizer( bSizer44 );
	CommandsPanel->Layout();
	bSizer44->Fit( CommandsPanel );
	m_splitter5->SplitVertically( ProfilesPanel, CommandsPanel, 300 );
	bSizer56->Add( m_splitter5, 1, wxEXPAND, 5 );
	
	
	this->SetSizer( bSizer56 );
	this->Layout();
	
	// Connect Events
	this->Connect( wxEVT_UPDATE_UI, wxUpdateUIEventHandler( RunProfilesPanel::OnUpdateIU ) );
	ProfilesListBox->Connect( wxEVT_LEFT_DCLICK, wxMouseEventHandler( RunProfilesPanel::OnProfileDClick ), NULL, this );
	ProfilesListBox->Connect( wxEVT_LEFT_DOWN, wxMouseEventHandler( RunProfilesPanel::OnProfileLeftDown ), NULL, this );
	ProfilesListBox->Connect( wxEVT_LEFT_UP, wxMouseEventHandler( RunProfilesPanel::MouseVeto ), NULL, this );
	ProfilesListBox->Connect( wxEVT_COMMAND_LIST_END_LABEL_EDIT, wxListEventHandler( RunProfilesPanel::OnEndProfileEdit ), NULL, this );
	ProfilesListBox->Connect( wxEVT_COMMAND_LIST_ITEM_ACTIVATED, wxListEventHandler( RunProfilesPanel::OnProfilesListItemActivated ), NULL, this );
	ProfilesListBox->Connect( wxEVT_COMMAND_LIST_ITEM_FOCUSED, wxListEventHandler( RunProfilesPanel::OnProfilesFocusChange ), NULL, this );
	ProfilesListBox->Connect( wxEVT_MIDDLE_DCLICK, wxMouseEventHandler( RunProfilesPanel::MouseVeto ), NULL, this );
	ProfilesListBox->Connect( wxEVT_MIDDLE_DOWN, wxMouseEventHandler( RunProfilesPanel::MouseVeto ), NULL, this );
	ProfilesListBox->Connect( wxEVT_MIDDLE_UP, wxMouseEventHandler( RunProfilesPanel::MouseVeto ), NULL, this );
	ProfilesListBox->Connect( wxEVT_MOTION, wxMouseEventHandler( RunProfilesPanel::MouseVeto ), NULL, this );
	ProfilesListBox->Connect( wxEVT_RIGHT_DCLICK, wxMouseEventHandler( RunProfilesPanel::MouseVeto ), NULL, this );
	ProfilesListBox->Connect( wxEVT_RIGHT_DOWN, wxMouseEventHandler( RunProfilesPanel::MouseVeto ), NULL, this );
	ProfilesListBox->Connect( wxEVT_RIGHT_UP, wxMouseEventHandler( RunProfilesPanel::MouseVeto ), NULL, this );
	AddProfileButton->Connect( wxEVT_COMMAND_BUTTON_CLICKED, wxCommandEventHandler( RunProfilesPanel::OnAddProfileClick ), NULL, this );
	RenameProfileButton->Connect( wxEVT_COMMAND_BUTTON_CLICKED, wxCommandEventHandler( RunProfilesPanel::OnRenameProfileClick ), NULL, this );
	RemoveProfileButton->Connect( wxEVT_COMMAND_BUTTON_CLICKED, wxCommandEventHandler( RunProfilesPanel::OnRemoveProfileClick ), NULL, this );
	ManagerTextCtrl->Connect( wxEVT_COMMAND_TEXT_UPDATED, wxCommandEventHandler( RunProfilesPanel::ManagerTextChanged ), NULL, this );
	CommandsListBox->Connect( wxEVT_LEFT_DCLICK, wxMouseEventHandler( RunProfilesPanel::OnCommandDClick ), NULL, this );
	CommandsListBox->Connect( wxEVT_LEFT_DOWN, wxMouseEventHandler( RunProfilesPanel::OnCommandLeftDown ), NULL, this );
	CommandsListBox->Connect( wxEVT_LEFT_UP, wxMouseEventHandler( RunProfilesPanel::MouseVeto ), NULL, this );
	CommandsListBox->Connect( wxEVT_COMMAND_LIST_ITEM_ACTIVATED, wxListEventHandler( RunProfilesPanel::OnCommandsActivated ), NULL, this );
	CommandsListBox->Connect( wxEVT_COMMAND_LIST_ITEM_FOCUSED, wxListEventHandler( RunProfilesPanel::OnCommandsFocusChange ), NULL, this );
	CommandsListBox->Connect( wxEVT_MIDDLE_DCLICK, wxMouseEventHandler( RunProfilesPanel::MouseVeto ), NULL, this );
	CommandsListBox->Connect( wxEVT_MIDDLE_DOWN, wxMouseEventHandler( RunProfilesPanel::MouseVeto ), NULL, this );
	CommandsListBox->Connect( wxEVT_MIDDLE_UP, wxMouseEventHandler( RunProfilesPanel::MouseVeto ), NULL, this );
	CommandsListBox->Connect( wxEVT_MOTION, wxMouseEventHandler( RunProfilesPanel::MouseVeto ), NULL, this );
	CommandsListBox->Connect( wxEVT_RIGHT_DCLICK, wxMouseEventHandler( RunProfilesPanel::MouseVeto ), NULL, this );
	CommandsListBox->Connect( wxEVT_RIGHT_DOWN, wxMouseEventHandler( RunProfilesPanel::MouseVeto ), NULL, this );
	CommandsListBox->Connect( wxEVT_RIGHT_UP, wxMouseEventHandler( RunProfilesPanel::MouseVeto ), NULL, this );
	AddCommandButton->Connect( wxEVT_COMMAND_BUTTON_CLICKED, wxCommandEventHandler( RunProfilesPanel::AddCommandButtonClick ), NULL, this );
	EditCommandButton->Connect( wxEVT_COMMAND_BUTTON_CLICKED, wxCommandEventHandler( RunProfilesPanel::EditCommandButtonClick ), NULL, this );
	RemoveCommandButton->Connect( wxEVT_COMMAND_BUTTON_CLICKED, wxCommandEventHandler( RunProfilesPanel::RemoveCommandButtonClick ), NULL, this );
	CommandsSaveButton->Connect( wxEVT_COMMAND_BUTTON_CLICKED, wxCommandEventHandler( RunProfilesPanel::CommandsSaveButtonClick ), NULL, this );
}

RunProfilesPanel::~RunProfilesPanel()
{
	// Disconnect Events
	this->Disconnect( wxEVT_UPDATE_UI, wxUpdateUIEventHandler( RunProfilesPanel::OnUpdateIU ) );
	ProfilesListBox->Disconnect( wxEVT_LEFT_DCLICK, wxMouseEventHandler( RunProfilesPanel::OnProfileDClick ), NULL, this );
	ProfilesListBox->Disconnect( wxEVT_LEFT_DOWN, wxMouseEventHandler( RunProfilesPanel::OnProfileLeftDown ), NULL, this );
	ProfilesListBox->Disconnect( wxEVT_LEFT_UP, wxMouseEventHandler( RunProfilesPanel::MouseVeto ), NULL, this );
	ProfilesListBox->Disconnect( wxEVT_COMMAND_LIST_END_LABEL_EDIT, wxListEventHandler( RunProfilesPanel::OnEndProfileEdit ), NULL, this );
	ProfilesListBox->Disconnect( wxEVT_COMMAND_LIST_ITEM_ACTIVATED, wxListEventHandler( RunProfilesPanel::OnProfilesListItemActivated ), NULL, this );
	ProfilesListBox->Disconnect( wxEVT_COMMAND_LIST_ITEM_FOCUSED, wxListEventHandler( RunProfilesPanel::OnProfilesFocusChange ), NULL, this );
	ProfilesListBox->Disconnect( wxEVT_MIDDLE_DCLICK, wxMouseEventHandler( RunProfilesPanel::MouseVeto ), NULL, this );
	ProfilesListBox->Disconnect( wxEVT_MIDDLE_DOWN, wxMouseEventHandler( RunProfilesPanel::MouseVeto ), NULL, this );
	ProfilesListBox->Disconnect( wxEVT_MIDDLE_UP, wxMouseEventHandler( RunProfilesPanel::MouseVeto ), NULL, this );
	ProfilesListBox->Disconnect( wxEVT_MOTION, wxMouseEventHandler( RunProfilesPanel::MouseVeto ), NULL, this );
	ProfilesListBox->Disconnect( wxEVT_RIGHT_DCLICK, wxMouseEventHandler( RunProfilesPanel::MouseVeto ), NULL, this );
	ProfilesListBox->Disconnect( wxEVT_RIGHT_DOWN, wxMouseEventHandler( RunProfilesPanel::MouseVeto ), NULL, this );
	ProfilesListBox->Disconnect( wxEVT_RIGHT_UP, wxMouseEventHandler( RunProfilesPanel::MouseVeto ), NULL, this );
	AddProfileButton->Disconnect( wxEVT_COMMAND_BUTTON_CLICKED, wxCommandEventHandler( RunProfilesPanel::OnAddProfileClick ), NULL, this );
	RenameProfileButton->Disconnect( wxEVT_COMMAND_BUTTON_CLICKED, wxCommandEventHandler( RunProfilesPanel::OnRenameProfileClick ), NULL, this );
	RemoveProfileButton->Disconnect( wxEVT_COMMAND_BUTTON_CLICKED, wxCommandEventHandler( RunProfilesPanel::OnRemoveProfileClick ), NULL, this );
	ManagerTextCtrl->Disconnect( wxEVT_COMMAND_TEXT_UPDATED, wxCommandEventHandler( RunProfilesPanel::ManagerTextChanged ), NULL, this );
	CommandsListBox->Disconnect( wxEVT_LEFT_DCLICK, wxMouseEventHandler( RunProfilesPanel::OnCommandDClick ), NULL, this );
	CommandsListBox->Disconnect( wxEVT_LEFT_DOWN, wxMouseEventHandler( RunProfilesPanel::OnCommandLeftDown ), NULL, this );
	CommandsListBox->Disconnect( wxEVT_LEFT_UP, wxMouseEventHandler( RunProfilesPanel::MouseVeto ), NULL, this );
	CommandsListBox->Disconnect( wxEVT_COMMAND_LIST_ITEM_ACTIVATED, wxListEventHandler( RunProfilesPanel::OnCommandsActivated ), NULL, this );
	CommandsListBox->Disconnect( wxEVT_COMMAND_LIST_ITEM_FOCUSED, wxListEventHandler( RunProfilesPanel::OnCommandsFocusChange ), NULL, this );
	CommandsListBox->Disconnect( wxEVT_MIDDLE_DCLICK, wxMouseEventHandler( RunProfilesPanel::MouseVeto ), NULL, this );
	CommandsListBox->Disconnect( wxEVT_MIDDLE_DOWN, wxMouseEventHandler( RunProfilesPanel::MouseVeto ), NULL, this );
	CommandsListBox->Disconnect( wxEVT_MIDDLE_UP, wxMouseEventHandler( RunProfilesPanel::MouseVeto ), NULL, this );
	CommandsListBox->Disconnect( wxEVT_MOTION, wxMouseEventHandler( RunProfilesPanel::MouseVeto ), NULL, this );
	CommandsListBox->Disconnect( wxEVT_RIGHT_DCLICK, wxMouseEventHandler( RunProfilesPanel::MouseVeto ), NULL, this );
	CommandsListBox->Disconnect( wxEVT_RIGHT_DOWN, wxMouseEventHandler( RunProfilesPanel::MouseVeto ), NULL, this );
	CommandsListBox->Disconnect( wxEVT_RIGHT_UP, wxMouseEventHandler( RunProfilesPanel::MouseVeto ), NULL, this );
	AddCommandButton->Disconnect( wxEVT_COMMAND_BUTTON_CLICKED, wxCommandEventHandler( RunProfilesPanel::AddCommandButtonClick ), NULL, this );
	EditCommandButton->Disconnect( wxEVT_COMMAND_BUTTON_CLICKED, wxCommandEventHandler( RunProfilesPanel::EditCommandButtonClick ), NULL, this );
	RemoveCommandButton->Disconnect( wxEVT_COMMAND_BUTTON_CLICKED, wxCommandEventHandler( RunProfilesPanel::RemoveCommandButtonClick ), NULL, this );
	CommandsSaveButton->Disconnect( wxEVT_COMMAND_BUTTON_CLICKED, wxCommandEventHandler( RunProfilesPanel::CommandsSaveButtonClick ), NULL, this );
	
}

ErrorDialog::ErrorDialog( wxWindow* parent, wxWindowID id, const wxString& title, const wxPoint& pos, const wxSize& size, long style ) : wxDialog( parent, id, title, pos, size, style )
{
	this->SetSizeHints( wxSize( -1,500 ), wxDefaultSize );
	
	wxBoxSizer* bSizer35;
	bSizer35 = new wxBoxSizer( wxVERTICAL );
	
	wxBoxSizer* bSizer36;
	bSizer36 = new wxBoxSizer( wxHORIZONTAL );
	
	m_bitmap1 = new wxStaticBitmap( this, wxID_ANY, wxArtProvider::GetBitmap( wxART_ERROR, wxART_OTHER ), wxDefaultPosition, wxDefaultSize, 0 );
	bSizer36->Add( m_bitmap1, 0, wxALL, 5 );
	
	m_staticText25 = new wxStaticText( this, wxID_ANY, wxT(": ERROR"), wxDefaultPosition, wxDefaultSize, 0 );
	m_staticText25->Wrap( -1 );
	bSizer36->Add( m_staticText25, 0, wxALIGN_CENTER_VERTICAL|wxALL, 5 );
	
	
	bSizer35->Add( bSizer36, 0, wxEXPAND, 5 );
	
	ErrorText = new wxTextCtrl( this, wxID_ANY, wxEmptyString, wxDefaultPosition, wxDefaultSize, wxTE_MULTILINE|wxTE_READONLY );
	bSizer35->Add( ErrorText, 100, wxALL|wxEXPAND, 5 );
	
	m_button23 = new wxButton( this, wxID_ANY, wxT("OK"), wxDefaultPosition, wxDefaultSize, 0 );
	bSizer35->Add( m_button23, 0, wxALIGN_RIGHT|wxALL, 5 );
	
	
	this->SetSizer( bSizer35 );
	this->Layout();
	
	this->Centre( wxBOTH );
	
	// Connect Events
	m_button23->Connect( wxEVT_COMMAND_BUTTON_CLICKED, wxCommandEventHandler( ErrorDialog::OnClickOK ), NULL, this );
}

ErrorDialog::~ErrorDialog()
{
	// Disconnect Events
	m_button23->Disconnect( wxEVT_COMMAND_BUTTON_CLICKED, wxCommandEventHandler( ErrorDialog::OnClickOK ), NULL, this );
	
}

AlignMoviesPanel::AlignMoviesPanel( wxWindow* parent, wxWindowID id, const wxPoint& pos, const wxSize& size, long style ) : JobPanel( parent, id, pos, size, style )
{
	wxBoxSizer* bSizer43;
	bSizer43 = new wxBoxSizer( wxVERTICAL );
	
	m_staticline12 = new wxStaticLine( this, wxID_ANY, wxDefaultPosition, wxDefaultSize, wxLI_HORIZONTAL );
	bSizer43->Add( m_staticline12, 0, wxEXPAND | wxALL, 5 );
	
	wxBoxSizer* bSizer45;
	bSizer45 = new wxBoxSizer( wxHORIZONTAL );
	
	wxBoxSizer* bSizer44;
	bSizer44 = new wxBoxSizer( wxHORIZONTAL );
	
	m_staticText21 = new wxStaticText( this, wxID_ANY, wxT("Input Group :"), wxDefaultPosition, wxDefaultSize, 0 );
	m_staticText21->Wrap( -1 );
	bSizer44->Add( m_staticText21, 0, wxALIGN_CENTER_VERTICAL|wxALL, 5 );
	
	GroupComboBox = new wxComboBox( this, wxID_ANY, wxEmptyString, wxDefaultPosition, wxDefaultSize, 0, NULL, wxCB_READONLY ); 
	bSizer44->Add( GroupComboBox, 40, wxALL, 5 );
	
	
	bSizer44->Add( 0, 0, 60, wxEXPAND, 5 );
	
	
	bSizer45->Add( bSizer44, 1, wxEXPAND, 5 );
	
	ExpertToggleButton = new wxToggleButton( this, wxID_ANY, wxT("Show Expert Options"), wxDefaultPosition, wxDefaultSize, 0 );
	bSizer45->Add( ExpertToggleButton, 0, wxALL, 5 );
	
	
	bSizer43->Add( bSizer45, 0, wxEXPAND, 5 );
	
	m_staticline10 = new wxStaticLine( this, wxID_ANY, wxDefaultPosition, wxDefaultSize, wxLI_HORIZONTAL );
	bSizer43->Add( m_staticline10, 0, wxEXPAND | wxALL, 5 );
	
	wxBoxSizer* bSizer46;
	bSizer46 = new wxBoxSizer( wxHORIZONTAL );
	
	ExpertPanel = new wxPanel( this, wxID_ANY, wxDefaultPosition, wxDefaultSize, wxTAB_TRAVERSAL );
	ExpertPanel->Hide();
	
	wxBoxSizer* bSizer64;
	bSizer64 = new wxBoxSizer( wxVERTICAL );
	
	wxFlexGridSizer* fgSizer1;
	fgSizer1 = new wxFlexGridSizer( 13, 2, 0, 0 );
	fgSizer1->SetFlexibleDirection( wxBOTH );
	fgSizer1->SetNonFlexibleGrowMode( wxFLEX_GROWMODE_SPECIFIED );
	
	m_staticText43 = new wxStaticText( ExpertPanel, wxID_ANY, wxT("Shifts"), wxDefaultPosition, wxDefaultSize, 0 );
	m_staticText43->Wrap( -1 );
	m_staticText43->SetFont( wxFont( wxNORMAL_FONT->GetPointSize(), 70, 90, 92, true, wxEmptyString ) );
	
	fgSizer1->Add( m_staticText43, 0, wxALL, 5 );
	
	
	fgSizer1->Add( 0, 0, 1, wxEXPAND, 5 );
	
	m_staticText24 = new wxStaticText( ExpertPanel, wxID_ANY, wxT("Minimum Shift (Å) : "), wxDefaultPosition, wxDefaultSize, 0 );
	m_staticText24->Wrap( -1 );
	fgSizer1->Add( m_staticText24, 0, wxALIGN_CENTER_VERTICAL|wxALL, 5 );
	
	minimum_shift_text = new wxTextCtrl( ExpertPanel, wxID_ANY, wxT("2"), wxDefaultPosition, wxDefaultSize, 0 );
	minimum_shift_text->SetToolTip( wxT("Minimum shift for first alignment round") );
	
	fgSizer1->Add( minimum_shift_text, 0, wxALL|wxEXPAND, 5 );
	
	m_staticText40 = new wxStaticText( ExpertPanel, wxID_ANY, wxT("Maximum Shift (Å) :"), wxDefaultPosition, wxDefaultSize, 0 );
	m_staticText40->Wrap( -1 );
	fgSizer1->Add( m_staticText40, 0, wxALIGN_CENTER_VERTICAL|wxALL, 5 );
	
	maximum_shift_text = new wxTextCtrl( ExpertPanel, wxID_ANY, wxT("80"), wxDefaultPosition, wxDefaultSize, 0 );
	maximum_shift_text->SetToolTip( wxT("Maximum shift for each alignment round") );
	
	fgSizer1->Add( maximum_shift_text, 0, wxALL|wxEXPAND, 5 );
	
	m_staticText44 = new wxStaticText( ExpertPanel, wxID_ANY, wxT("Dose Filter"), wxDefaultPosition, wxDefaultSize, 0 );
	m_staticText44->Wrap( -1 );
	m_staticText44->SetFont( wxFont( 10, 74, 90, 92, true, wxT("Sans") ) );
	
	fgSizer1->Add( m_staticText44, 0, wxALL, 5 );
	
	
	fgSizer1->Add( 0, 0, 1, wxEXPAND, 5 );
	
	dose_filter_checkbox = new wxCheckBox( ExpertPanel, wxID_ANY, wxT("Dose Filter Sums?"), wxDefaultPosition, wxDefaultSize, 0 );
	dose_filter_checkbox->SetValue(true); 
	dose_filter_checkbox->SetToolTip( wxT("Make a dose weighted sum") );
	
	fgSizer1->Add( dose_filter_checkbox, 1, wxALIGN_LEFT|wxALL, 5 );
	
	restore_power_checkbox = new wxCheckBox( ExpertPanel, wxID_ANY, wxT("Restore Power?"), wxDefaultPosition, wxDefaultSize, 0 );
	restore_power_checkbox->SetValue(true); 
	fgSizer1->Add( restore_power_checkbox, 1, wxALIGN_RIGHT|wxALL, 5 );
	
	m_staticText45 = new wxStaticText( ExpertPanel, wxID_ANY, wxT("Convergence"), wxDefaultPosition, wxDefaultSize, 0 );
	m_staticText45->Wrap( -1 );
	m_staticText45->SetFont( wxFont( wxNORMAL_FONT->GetPointSize(), 70, 90, 92, true, wxEmptyString ) );
	
	fgSizer1->Add( m_staticText45, 0, wxALL, 5 );
	
	
	fgSizer1->Add( 0, 0, 1, wxEXPAND, 5 );
	
	m_staticText46 = new wxStaticText( ExpertPanel, wxID_ANY, wxT("Termination Threshold (Å) : "), wxDefaultPosition, wxDefaultSize, 0 );
	m_staticText46->Wrap( -1 );
	fgSizer1->Add( m_staticText46, 0, wxALL, 5 );
	
	termination_threshold_text = new wxTextCtrl( ExpertPanel, wxID_ANY, wxT("1"), wxDefaultPosition, wxDefaultSize, 0 );
	fgSizer1->Add( termination_threshold_text, 0, wxALL|wxEXPAND, 5 );
	
	m_staticText47 = new wxStaticText( ExpertPanel, wxID_ANY, wxT("Max Iterations :"), wxDefaultPosition, wxDefaultSize, 0 );
	m_staticText47->Wrap( -1 );
	fgSizer1->Add( m_staticText47, 0, wxALL, 5 );
	
	max_iterations_spinctrl = new wxSpinCtrl( ExpertPanel, wxID_ANY, wxEmptyString, wxDefaultPosition, wxDefaultSize, wxSP_ARROW_KEYS, 1, 50, 20 );
	fgSizer1->Add( max_iterations_spinctrl, 0, wxALL|wxEXPAND, 5 );
	
	m_staticText48 = new wxStaticText( ExpertPanel, wxID_ANY, wxT("Filter"), wxDefaultPosition, wxDefaultSize, 0 );
	m_staticText48->Wrap( -1 );
	m_staticText48->SetFont( wxFont( wxNORMAL_FONT->GetPointSize(), 70, 90, 92, true, wxEmptyString ) );
	
	fgSizer1->Add( m_staticText48, 0, wxALL, 5 );
	
	
	fgSizer1->Add( 0, 0, 1, wxEXPAND, 5 );
	
	m_staticText49 = new wxStaticText( ExpertPanel, wxID_ANY, wxT("B-Factor (Å²) :"), wxDefaultPosition, wxDefaultSize, 0 );
	m_staticText49->Wrap( -1 );
	fgSizer1->Add( m_staticText49, 0, wxALL, 5 );
	
	bfactor_spinctrl = new wxSpinCtrl( ExpertPanel, wxID_ANY, wxEmptyString, wxDefaultPosition, wxDefaultSize, wxSP_ARROW_KEYS, 0, 5000, 1500 );
	fgSizer1->Add( bfactor_spinctrl, 0, wxALL|wxEXPAND, 5 );
	
	mask_central_cross_checkbox = new wxCheckBox( ExpertPanel, wxID_ANY, wxT("Mask Central Cross?"), wxDefaultPosition, wxDefaultSize, 0 );
	mask_central_cross_checkbox->SetValue(true); 
	fgSizer1->Add( mask_central_cross_checkbox, 0, wxALL, 5 );
	
	
	fgSizer1->Add( 0, 0, 1, wxEXPAND, 5 );
	
	m_staticText50 = new wxStaticText( ExpertPanel, wxID_ANY, wxT("Horizontal Mask (Pixels) :"), wxDefaultPosition, wxDefaultSize, 0 );
	m_staticText50->Wrap( -1 );
	fgSizer1->Add( m_staticText50, 0, wxALL, 5 );
	
	horizontal_mask_spinctrl = new wxSpinCtrl( ExpertPanel, wxID_ANY, wxEmptyString, wxDefaultPosition, wxDefaultSize, wxSP_ARROW_KEYS, 0, 10, 1 );
	fgSizer1->Add( horizontal_mask_spinctrl, 0, wxALL|wxEXPAND, 5 );
	
	m_staticText51 = new wxStaticText( ExpertPanel, wxID_ANY, wxT("Vertical Mask (Pixels) :"), wxDefaultPosition, wxDefaultSize, 0 );
	m_staticText51->Wrap( -1 );
	fgSizer1->Add( m_staticText51, 0, wxALL, 5 );
	
	vertical_mask_spinctrl = new wxSpinCtrl( ExpertPanel, wxID_ANY, wxEmptyString, wxDefaultPosition, wxDefaultSize, wxSP_ARROW_KEYS, 0, 10, 1 );
	fgSizer1->Add( vertical_mask_spinctrl, 0, wxALL|wxEXPAND, 5 );
	
	
	bSizer64->Add( fgSizer1, 0, wxEXPAND, 5 );
	
	
	ExpertPanel->SetSizer( bSizer64 );
	ExpertPanel->Layout();
	bSizer64->Fit( ExpertPanel );
	bSizer46->Add( ExpertPanel, 0, wxEXPAND | wxALL, 5 );
	
	OutputTextPanel = new wxPanel( this, wxID_ANY, wxDefaultPosition, wxDefaultSize, wxTAB_TRAVERSAL );
	OutputTextPanel->Hide();
	
	wxBoxSizer* bSizer56;
	bSizer56 = new wxBoxSizer( wxVERTICAL );
	
	output_textctrl = new wxTextCtrl( OutputTextPanel, wxID_ANY, wxEmptyString, wxDefaultPosition, wxDefaultSize, wxTE_MULTILINE|wxTE_READONLY );
	bSizer56->Add( output_textctrl, 1, wxALL|wxEXPAND, 5 );
	
	
	OutputTextPanel->SetSizer( bSizer56 );
	OutputTextPanel->Layout();
	bSizer56->Fit( OutputTextPanel );
	bSizer46->Add( OutputTextPanel, 30, wxEXPAND | wxALL, 5 );
	
	GraphPanel = new wxPanel( this, wxID_ANY, wxDefaultPosition, wxDefaultSize, wxTAB_TRAVERSAL );
	GraphPanel->Hide();
	
	GraphSizer = new wxBoxSizer( wxVERTICAL );
	
	
	GraphPanel->SetSizer( GraphSizer );
	GraphPanel->Layout();
	GraphSizer->Fit( GraphPanel );
	bSizer46->Add( GraphPanel, 50, wxEXPAND | wxALL, 5 );
	
	InfoPanel = new wxPanel( this, wxID_ANY, wxDefaultPosition, wxDefaultSize, wxTAB_TRAVERSAL );
	wxBoxSizer* bSizer61;
	bSizer61 = new wxBoxSizer( wxVERTICAL );
	
	InfoText = new wxRichTextCtrl( InfoPanel, wxID_ANY, wxEmptyString, wxDefaultPosition, wxDefaultSize, wxTE_READONLY|wxHSCROLL|wxVSCROLL );
	bSizer61->Add( InfoText, 1, wxEXPAND | wxALL, 5 );
	
	
	InfoPanel->SetSizer( bSizer61 );
	InfoPanel->Layout();
	bSizer61->Fit( InfoPanel );
	bSizer46->Add( InfoPanel, 1, wxEXPAND | wxALL, 5 );
	
	
	bSizer43->Add( bSizer46, 1, wxEXPAND, 5 );
	
	m_staticline11 = new wxStaticLine( this, wxID_ANY, wxDefaultPosition, wxDefaultSize, wxLI_HORIZONTAL );
	bSizer43->Add( m_staticline11, 0, wxEXPAND | wxALL, 5 );
	
	wxBoxSizer* bSizer48;
	bSizer48 = new wxBoxSizer( wxHORIZONTAL );
	
	wxBoxSizer* bSizer70;
	bSizer70 = new wxBoxSizer( wxHORIZONTAL );
	
	wxBoxSizer* bSizer71;
	bSizer71 = new wxBoxSizer( wxVERTICAL );
	
	wxBoxSizer* bSizer55;
	bSizer55 = new wxBoxSizer( wxHORIZONTAL );
	
	
	bSizer71->Add( bSizer55, 0, wxALIGN_CENTER_HORIZONTAL, 5 );
	
	
	bSizer70->Add( bSizer71, 1, wxALIGN_CENTER_VERTICAL, 5 );
	
	wxBoxSizer* bSizer74;
	bSizer74 = new wxBoxSizer( wxVERTICAL );
	
	
	bSizer70->Add( bSizer74, 0, wxALIGN_CENTER_VERTICAL, 5 );
	
	
	bSizer48->Add( bSizer70, 1, wxEXPAND, 5 );
	
	ProgressPanel = new wxPanel( this, wxID_ANY, wxDefaultPosition, wxDefaultSize, wxTAB_TRAVERSAL );
	ProgressPanel->Hide();
	
	wxBoxSizer* bSizer57;
	bSizer57 = new wxBoxSizer( wxHORIZONTAL );
	
	wxBoxSizer* bSizer59;
	bSizer59 = new wxBoxSizer( wxVERTICAL );
	
	ProgressBar = new wxGauge( ProgressPanel, wxID_ANY, 100, wxDefaultPosition, wxDefaultSize, wxGA_HORIZONTAL );
	ProgressBar->SetValue( 0 ); 
	bSizer59->Add( ProgressBar, 0, wxALIGN_CENTER|wxALIGN_CENTER_HORIZONTAL|wxALIGN_CENTER_VERTICAL|wxALL|wxEXPAND, 5 );
	
	wxBoxSizer* bSizer62;
	bSizer62 = new wxBoxSizer( wxHORIZONTAL );
	
	NumberConnectedText = new wxStaticText( ProgressPanel, wxID_ANY, wxT("1000 / 1000 processes connected."), wxDefaultPosition, wxDefaultSize, 0 );
	NumberConnectedText->Wrap( -1 );
	bSizer62->Add( NumberConnectedText, 1, wxALIGN_CENTER|wxALL|wxEXPAND, 5 );
	
	TimeRemainingText = new wxStaticText( ProgressPanel, wxID_ANY, wxT("Remaining Time : ???h:??m:??s"), wxDefaultPosition, wxDefaultSize, wxALIGN_CENTRE );
	TimeRemainingText->Wrap( -1 );
	bSizer62->Add( TimeRemainingText, 0, wxALIGN_RIGHT|wxALL|wxEXPAND, 5 );
	
	
	bSizer59->Add( bSizer62, 1, wxEXPAND, 5 );
	
	CancelAlignmentButton = new wxButton( ProgressPanel, wxID_ANY, wxT("Terminate Running Job"), wxDefaultPosition, wxDefaultSize, 0 );
	bSizer59->Add( CancelAlignmentButton, 0, wxALIGN_CENTER_HORIZONTAL|wxALL, 5 );
	
	FinishButton = new wxButton( ProgressPanel, wxID_ANY, wxT("Finish"), wxDefaultPosition, wxDefaultSize, 0 );
	FinishButton->Hide();
	
	bSizer59->Add( FinishButton, 0, wxALIGN_CENTER_HORIZONTAL|wxALIGN_CENTER_VERTICAL|wxALL, 5 );
	
	
	bSizer57->Add( bSizer59, 1, wxEXPAND, 5 );
	
	
	ProgressPanel->SetSizer( bSizer57 );
	ProgressPanel->Layout();
	bSizer57->Fit( ProgressPanel );
	bSizer48->Add( ProgressPanel, 1, wxEXPAND | wxALL, 5 );
	
	StartPanel = new wxPanel( this, wxID_ANY, wxDefaultPosition, wxDefaultSize, wxTAB_TRAVERSAL );
	wxBoxSizer* bSizer58;
	bSizer58 = new wxBoxSizer( wxHORIZONTAL );
	
	RunProfileText = new wxStaticText( StartPanel, wxID_ANY, wxT("Run Profile :"), wxDefaultPosition, wxDefaultSize, 0 );
	RunProfileText->Wrap( -1 );
	bSizer58->Add( RunProfileText, 0, wxALIGN_CENTER_VERTICAL|wxALL, 5 );
	
	RunProfileComboBox = new wxComboBox( StartPanel, wxID_ANY, wxEmptyString, wxDefaultPosition, wxDefaultSize, 0, NULL, wxCB_READONLY ); 
	bSizer58->Add( RunProfileComboBox, 50, wxALIGN_CENTER_VERTICAL|wxALL|wxEXPAND, 5 );
	
	wxBoxSizer* bSizer60;
	bSizer60 = new wxBoxSizer( wxVERTICAL );
	
	StartAlignmentButton = new wxButton( StartPanel, wxID_ANY, wxT("Start Alignment"), wxDefaultPosition, wxDefaultSize, 0 );
	bSizer60->Add( StartAlignmentButton, 0, wxALL, 5 );
	
	
	bSizer58->Add( bSizer60, 50, wxEXPAND, 5 );
	
	
	StartPanel->SetSizer( bSizer58 );
	StartPanel->Layout();
	bSizer58->Fit( StartPanel );
	bSizer48->Add( StartPanel, 1, wxEXPAND | wxALL, 5 );
	
	
	bSizer43->Add( bSizer48, 0, wxEXPAND, 5 );
	
	
	this->SetSizer( bSizer43 );
	this->Layout();
	
	// Connect Events
	this->Connect( wxEVT_UPDATE_UI, wxUpdateUIEventHandler( AlignMoviesPanel::OnUpdateUI ) );
	ExpertToggleButton->Connect( wxEVT_COMMAND_TOGGLEBUTTON_CLICKED, wxCommandEventHandler( AlignMoviesPanel::OnExpertOptionsToggle ), NULL, this );
	InfoText->Connect( wxEVT_COMMAND_TEXT_URL, wxTextUrlEventHandler( AlignMoviesPanel::OnInfoURL ), NULL, this );
	CancelAlignmentButton->Connect( wxEVT_COMMAND_BUTTON_CLICKED, wxCommandEventHandler( AlignMoviesPanel::TerminateButtonClick ), NULL, this );
	FinishButton->Connect( wxEVT_COMMAND_BUTTON_CLICKED, wxCommandEventHandler( AlignMoviesPanel::FinishButtonClick ), NULL, this );
	StartAlignmentButton->Connect( wxEVT_COMMAND_BUTTON_CLICKED, wxCommandEventHandler( AlignMoviesPanel::StartAlignmentClick ), NULL, this );
}

AlignMoviesPanel::~AlignMoviesPanel()
{
	// Disconnect Events
	this->Disconnect( wxEVT_UPDATE_UI, wxUpdateUIEventHandler( AlignMoviesPanel::OnUpdateUI ) );
	ExpertToggleButton->Disconnect( wxEVT_COMMAND_TOGGLEBUTTON_CLICKED, wxCommandEventHandler( AlignMoviesPanel::OnExpertOptionsToggle ), NULL, this );
	InfoText->Disconnect( wxEVT_COMMAND_TEXT_URL, wxTextUrlEventHandler( AlignMoviesPanel::OnInfoURL ), NULL, this );
	CancelAlignmentButton->Disconnect( wxEVT_COMMAND_BUTTON_CLICKED, wxCommandEventHandler( AlignMoviesPanel::TerminateButtonClick ), NULL, this );
	FinishButton->Disconnect( wxEVT_COMMAND_BUTTON_CLICKED, wxCommandEventHandler( AlignMoviesPanel::FinishButtonClick ), NULL, this );
	StartAlignmentButton->Disconnect( wxEVT_COMMAND_BUTTON_CLICKED, wxCommandEventHandler( AlignMoviesPanel::StartAlignmentClick ), NULL, this );
	
}

AddRunCommandDialog::AddRunCommandDialog( wxWindow* parent, wxWindowID id, const wxString& title, const wxPoint& pos, const wxSize& size, long style ) : wxDialog( parent, id, title, pos, size, style )
{
	this->SetSizeHints( wxDefaultSize, wxDefaultSize );
	
	wxBoxSizer* bSizer45;
	bSizer45 = new wxBoxSizer( wxVERTICAL );
	
	wxFlexGridSizer* fgSizer2;
	fgSizer2 = new wxFlexGridSizer( 2, 2, 0, 0 );
	fgSizer2->AddGrowableCol( 1 );
	fgSizer2->SetFlexibleDirection( wxBOTH );
	fgSizer2->SetNonFlexibleGrowMode( wxFLEX_GROWMODE_SPECIFIED );
	
	m_staticText45 = new wxStaticText( this, wxID_ANY, wxT("Command :"), wxDefaultPosition, wxDefaultSize, 0 );
	m_staticText45->Wrap( -1 );
	fgSizer2->Add( m_staticText45, 0, wxALIGN_CENTER_VERTICAL|wxALL, 5 );
	
	CommandTextCtrl = new wxTextCtrl( this, wxID_ANY, wxT("$command"), wxDefaultPosition, wxDefaultSize, wxTE_PROCESS_ENTER );
	CommandTextCtrl->SetMinSize( wxSize( 300,-1 ) );
	
	fgSizer2->Add( CommandTextCtrl, 1, wxALL|wxEXPAND, 5 );
	
	m_staticText46 = new wxStaticText( this, wxID_ANY, wxT("No. Copies :"), wxDefaultPosition, wxDefaultSize, 0 );
	m_staticText46->Wrap( -1 );
	fgSizer2->Add( m_staticText46, 0, wxALIGN_CENTER_VERTICAL|wxALL, 5 );
	
	NumberCopiesSpinCtrl = new wxSpinCtrl( this, wxID_ANY, wxEmptyString, wxDefaultPosition, wxDefaultSize, wxSP_ARROW_KEYS, 1, 999999, 1 );
	fgSizer2->Add( NumberCopiesSpinCtrl, 1, wxALL|wxEXPAND, 5 );
	
	
	bSizer45->Add( fgSizer2, 0, wxEXPAND, 5 );
	
	ErrorStaticText = new wxStaticText( this, wxID_ANY, wxT("Oops! - Command must contain \"$command\""), wxDefaultPosition, wxDefaultSize, 0 );
	ErrorStaticText->Wrap( -1 );
	ErrorStaticText->SetForegroundColour( wxColour( 180, 0, 0 ) );
	ErrorStaticText->Hide();
	
	bSizer45->Add( ErrorStaticText, 0, wxALIGN_CENTER|wxALL, 5 );
	
	m_staticline14 = new wxStaticLine( this, wxID_ANY, wxDefaultPosition, wxDefaultSize, wxLI_HORIZONTAL );
	bSizer45->Add( m_staticline14, 0, wxEXPAND | wxALL, 5 );
	
	wxBoxSizer* bSizer47;
	bSizer47 = new wxBoxSizer( wxHORIZONTAL );
	
	OKButton = new wxButton( this, wxID_ANY, wxT("OK"), wxDefaultPosition, wxDefaultSize, 0 );
	bSizer47->Add( OKButton, 0, wxALL, 5 );
	
	CancelButton = new wxButton( this, wxID_ANY, wxT("Cancel"), wxDefaultPosition, wxDefaultSize, 0 );
	bSizer47->Add( CancelButton, 0, wxALL, 5 );
	
	
	bSizer45->Add( bSizer47, 0, wxALIGN_CENTER, 5 );
	
	
	this->SetSizer( bSizer45 );
	this->Layout();
	bSizer45->Fit( this );
	
	this->Centre( wxBOTH );
	
	// Connect Events
	CommandTextCtrl->Connect( wxEVT_COMMAND_TEXT_ENTER, wxCommandEventHandler( AddRunCommandDialog::OnEnter ), NULL, this );
	OKButton->Connect( wxEVT_COMMAND_BUTTON_CLICKED, wxCommandEventHandler( AddRunCommandDialog::OnOKClick ), NULL, this );
	CancelButton->Connect( wxEVT_COMMAND_BUTTON_CLICKED, wxCommandEventHandler( AddRunCommandDialog::OnCancelClick ), NULL, this );
}

AddRunCommandDialog::~AddRunCommandDialog()
{
	// Disconnect Events
	CommandTextCtrl->Disconnect( wxEVT_COMMAND_TEXT_ENTER, wxCommandEventHandler( AddRunCommandDialog::OnEnter ), NULL, this );
	OKButton->Disconnect( wxEVT_COMMAND_BUTTON_CLICKED, wxCommandEventHandler( AddRunCommandDialog::OnOKClick ), NULL, this );
	CancelButton->Disconnect( wxEVT_COMMAND_BUTTON_CLICKED, wxCommandEventHandler( AddRunCommandDialog::OnCancelClick ), NULL, this );
	
}

NewProjectWizard::NewProjectWizard( wxWindow* parent, wxWindowID id, const wxString& title, const wxBitmap& bitmap, const wxPoint& pos, long style ) 
{
	this->Create( parent, id, title, bitmap, pos, style );
	this->SetSizeHints( wxSize( 500,350 ), wxDefaultSize );
	
	wxWizardPageSimple* m_wizPage1 = new wxWizardPageSimple( this );
	m_pages.Add( m_wizPage1 );
	
	wxBoxSizer* bSizer47;
	bSizer47 = new wxBoxSizer( wxVERTICAL );
	
	m_staticText41 = new wxStaticText( m_wizPage1, wxID_ANY, wxT("Project Name :"), wxDefaultPosition, wxDefaultSize, 0 );
	m_staticText41->Wrap( -1 );
	bSizer47->Add( m_staticText41, 0, wxALIGN_CENTER_VERTICAL|wxALL, 5 );
	
	ProjectNameTextCtrl = new wxTextCtrl( m_wizPage1, wxID_ANY, wxT("New_Project"), wxDefaultPosition, wxSize( -1,-1 ), 0 );
	ProjectNameTextCtrl->SetMinSize( wxSize( 500,-1 ) );
	
	bSizer47->Add( ProjectNameTextCtrl, 0, wxALL|wxEXPAND, 5 );
	
	m_staticText42 = new wxStaticText( m_wizPage1, wxID_ANY, wxT("Project Parent Directory :"), wxDefaultPosition, wxDefaultSize, 0 );
	m_staticText42->Wrap( -1 );
	bSizer47->Add( m_staticText42, 0, wxALIGN_CENTER_VERTICAL|wxALL, 5 );
	
	wxBoxSizer* bSizer50;
	bSizer50 = new wxBoxSizer( wxHORIZONTAL );
	
	ParentDirTextCtrl = new wxTextCtrl( m_wizPage1, wxID_ANY, wxEmptyString, wxDefaultPosition, wxDefaultSize, 0 );
	bSizer50->Add( ParentDirTextCtrl, 1, wxALL, 5 );
	
	m_button24 = new wxButton( m_wizPage1, wxID_ANY, wxT("Browse..."), wxDefaultPosition, wxDefaultSize, 0 );
	bSizer50->Add( m_button24, 0, wxALL, 5 );
	
	
	bSizer47->Add( bSizer50, 0, wxEXPAND, 5 );
	
	m_staticText45 = new wxStaticText( m_wizPage1, wxID_ANY, wxT("Resulting project path :"), wxDefaultPosition, wxDefaultSize, 0 );
	m_staticText45->Wrap( -1 );
	bSizer47->Add( m_staticText45, 0, wxALL, 5 );
	
	ProjectPathTextCtrl = new wxTextCtrl( m_wizPage1, wxID_ANY, wxT("/tmp/New_Project"), wxDefaultPosition, wxDefaultSize, 0 );
	bSizer47->Add( ProjectPathTextCtrl, 0, wxALL|wxEXPAND, 5 );
	
	ErrorText = new wxStaticText( m_wizPage1, wxID_ANY, wxEmptyString, wxDefaultPosition, wxDefaultSize, 0 );
	ErrorText->Wrap( -1 );
	ErrorText->SetForegroundColour( wxColour( 255, 0, 0 ) );
	
	bSizer47->Add( ErrorText, 0, wxALL, 5 );
	
	
	m_wizPage1->SetSizer( bSizer47 );
	m_wizPage1->Layout();
	bSizer47->Fit( m_wizPage1 );
	
	this->Centre( wxBOTH );
	
	for ( unsigned int i = 1; i < m_pages.GetCount(); i++ )
	{
		m_pages.Item( i )->SetPrev( m_pages.Item( i - 1 ) );
		m_pages.Item( i - 1 )->SetNext( m_pages.Item( i ) );
	}
	
	// Connect Events
	this->Connect( wxID_ANY, wxEVT_WIZARD_FINISHED, wxWizardEventHandler( NewProjectWizard::OnFinished ) );
	ProjectNameTextCtrl->Connect( wxEVT_COMMAND_TEXT_UPDATED, wxCommandEventHandler( NewProjectWizard::OnProjectTextChange ), NULL, this );
	ParentDirTextCtrl->Connect( wxEVT_COMMAND_TEXT_UPDATED, wxCommandEventHandler( NewProjectWizard::OnParentDirChange ), NULL, this );
	m_button24->Connect( wxEVT_COMMAND_BUTTON_CLICKED, wxCommandEventHandler( NewProjectWizard::OnBrowseButtonClick ), NULL, this );
	ProjectPathTextCtrl->Connect( wxEVT_COMMAND_TEXT_UPDATED, wxCommandEventHandler( NewProjectWizard::OnProjectPathChange ), NULL, this );
}

NewProjectWizard::~NewProjectWizard()
{
	// Disconnect Events
	this->Disconnect( wxID_ANY, wxEVT_WIZARD_FINISHED, wxWizardEventHandler( NewProjectWizard::OnFinished ) );
	ProjectNameTextCtrl->Disconnect( wxEVT_COMMAND_TEXT_UPDATED, wxCommandEventHandler( NewProjectWizard::OnProjectTextChange ), NULL, this );
	ParentDirTextCtrl->Disconnect( wxEVT_COMMAND_TEXT_UPDATED, wxCommandEventHandler( NewProjectWizard::OnParentDirChange ), NULL, this );
	m_button24->Disconnect( wxEVT_COMMAND_BUTTON_CLICKED, wxCommandEventHandler( NewProjectWizard::OnBrowseButtonClick ), NULL, this );
	ProjectPathTextCtrl->Disconnect( wxEVT_COMMAND_TEXT_UPDATED, wxCommandEventHandler( NewProjectWizard::OnProjectPathChange ), NULL, this );
	
	m_pages.Clear();
}
