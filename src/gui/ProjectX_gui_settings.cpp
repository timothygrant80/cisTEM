///////////////////////////////////////////////////////////////////////////
// C++ code generated with wxFormBuilder (version 3.10.0-4761b0c)
// http://www.wxformbuilder.org/
//
// PLEASE DO *NOT* EDIT THIS FILE!
///////////////////////////////////////////////////////////////////////////

#include "ProjectX_gui_settings.h"

///////////////////////////////////////////////////////////////////////////

RunProfilesPanel::RunProfilesPanel( wxWindow* parent, wxWindowID id, const wxPoint& pos, const wxSize& size, long style, const wxString& name ) : wxPanel( parent, id, pos, size, style, name )
{
	this->SetMinSize( wxSize( 680,400 ) );

	wxBoxSizer* bSizer56;
	bSizer56 = new wxBoxSizer( wxVERTICAL );

	m_staticline12 = new wxStaticLine( this, wxID_ANY, wxDefaultPosition, wxDefaultSize, wxLI_HORIZONTAL );
	bSizer56->Add( m_staticline12, 0, wxEXPAND | wxALL, 5 );

	m_splitter5 = new wxSplitterWindow( this, wxID_ANY, wxDefaultPosition, wxDefaultSize, wxSP_3D );
	m_splitter5->Connect( wxEVT_IDLE, wxIdleEventHandler( RunProfilesPanel::m_splitter5OnIdle ), NULL, this );
	m_splitter5->SetMinimumPaneSize( 200 );

	ProfilesPanel = new wxPanel( m_splitter5, wxID_ANY, wxDefaultPosition, wxDefaultSize, wxTAB_TRAVERSAL );
	wxBoxSizer* bSizer57;
	bSizer57 = new wxBoxSizer( wxHORIZONTAL );

	ProfilesListBox = new wxListCtrl( ProfilesPanel, wxID_ANY, wxDefaultPosition, wxDefaultSize, wxLC_EDIT_LABELS|wxLC_NO_HEADER|wxLC_REPORT|wxLC_SINGLE_SEL );
	bSizer57->Add( ProfilesListBox, 1, wxALL|wxEXPAND, 5 );

	wxBoxSizer* bSizer431;
	bSizer431 = new wxBoxSizer( wxVERTICAL );

	AddProfileButton = new wxButton( ProfilesPanel, wxID_ANY, wxT("Add"), wxDefaultPosition, wxDefaultSize, 0 );
	bSizer431->Add( AddProfileButton, 1, wxALL, 5 );

	RenameProfileButton = new wxButton( ProfilesPanel, wxID_ANY, wxT("Rename"), wxDefaultPosition, wxDefaultSize, 0 );
	bSizer431->Add( RenameProfileButton, 0, wxALL, 5 );

	RemoveProfileButton = new wxButton( ProfilesPanel, wxID_ANY, wxT("Remove"), wxDefaultPosition, wxDefaultSize, 0 );
	bSizer431->Add( RemoveProfileButton, 1, wxALL, 5 );

	DuplicateProfileButton = new wxButton( ProfilesPanel, wxID_ANY, wxT("Duplicate"), wxDefaultPosition, wxDefaultSize, 0 );
	bSizer431->Add( DuplicateProfileButton, 0, wxALL, 5 );

	m_staticline26 = new wxStaticLine( ProfilesPanel, wxID_ANY, wxDefaultPosition, wxDefaultSize, wxLI_HORIZONTAL );
	bSizer431->Add( m_staticline26, 0, wxEXPAND | wxALL, 5 );

	ImportButton = new wxButton( ProfilesPanel, wxID_ANY, wxT("Import"), wxDefaultPosition, wxDefaultSize, 0 );
	bSizer431->Add( ImportButton, 0, wxALL, 5 );

	ExportButton = new wxButton( ProfilesPanel, wxID_ANY, wxT("Export"), wxDefaultPosition, wxDefaultSize, 0 );
	bSizer431->Add( ExportButton, 0, wxALL, 5 );


	bSizer57->Add( bSizer431, 0, wxALIGN_LEFT, 5 );


	ProfilesPanel->SetSizer( bSizer57 );
	ProfilesPanel->Layout();
	bSizer57->Fit( ProfilesPanel );
	CommandsPanel = new wxPanel( m_splitter5, wxID_ANY, wxDefaultPosition, wxDefaultSize, wxTAB_TRAVERSAL );
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

	CommandErrorStaticText = new wxStaticText( CommandsPanel, wxID_ANY, wxEmptyString, wxDefaultPosition, wxDefaultSize, wxALIGN_CENTER_HORIZONTAL );
	CommandErrorStaticText->Wrap( -1 );
	CommandErrorStaticText->SetForegroundColour( wxColour( 180, 0, 0 ) );

	bSizer38->Add( CommandErrorStaticText, 0, wxALL|wxEXPAND, 5 );

	wxFlexGridSizer* fgSizer3;
	fgSizer3 = new wxFlexGridSizer( 0, 5, 0, 0 );
	fgSizer3->AddGrowableCol( 2 );
	fgSizer3->SetFlexibleDirection( wxBOTH );
	fgSizer3->SetNonFlexibleGrowMode( wxFLEX_GROWMODE_SPECIFIED );


	fgSizer3->Add( 0, 0, 100, wxEXPAND, 5 );

	m_staticText65 = new wxStaticText( CommandsPanel, wxID_ANY, wxT("Gui Address :"), wxDefaultPosition, wxDefaultSize, 0 );
	m_staticText65->Wrap( -1 );
	fgSizer3->Add( m_staticText65, 0, wxALIGN_CENTER_VERTICAL|wxALIGN_RIGHT|wxALL, 5 );

	GuiAddressStaticText = new wxStaticText( CommandsPanel, wxID_ANY, wxEmptyString, wxDefaultPosition, wxDefaultSize, 0 );
	GuiAddressStaticText->Wrap( -1 );
	fgSizer3->Add( GuiAddressStaticText, 0, wxALIGN_CENTER_VERTICAL|wxALIGN_LEFT|wxALL, 5 );

	GuiAutoButton = new wxButton( CommandsPanel, wxID_ANY, wxT("Auto"), wxDefaultPosition, wxDefaultSize, 0 );
	fgSizer3->Add( GuiAutoButton, 0, wxALL, 5 );

	ControllerSpecifyButton = new wxButton( CommandsPanel, wxID_ANY, wxT("Specify"), wxDefaultPosition, wxDefaultSize, 0 );
	fgSizer3->Add( ControllerSpecifyButton, 0, wxALL, 5 );


	fgSizer3->Add( 0, 0, 100, wxEXPAND, 5 );

	m_staticText67 = new wxStaticText( CommandsPanel, wxID_ANY, wxT("Controller Address :"), wxDefaultPosition, wxDefaultSize, 0 );
	m_staticText67->Wrap( -1 );
	fgSizer3->Add( m_staticText67, 0, wxALIGN_CENTER_VERTICAL|wxALIGN_RIGHT|wxALL, 5 );

	ControllerAddressStaticText = new wxStaticText( CommandsPanel, wxID_ANY, wxEmptyString, wxDefaultPosition, wxDefaultSize, 0 );
	ControllerAddressStaticText->Wrap( -1 );
	fgSizer3->Add( ControllerAddressStaticText, 0, wxALIGN_CENTER_VERTICAL|wxALIGN_LEFT|wxALL, 5 );

	ControllerAutoButton = new wxButton( CommandsPanel, wxID_ANY, wxT("Auto"), wxDefaultPosition, wxDefaultSize, 0 );
	fgSizer3->Add( ControllerAutoButton, 0, wxALL, 5 );

	m_button38 = new wxButton( CommandsPanel, wxID_ANY, wxT("Specify"), wxDefaultPosition, wxDefaultSize, 0 );
	fgSizer3->Add( m_button38, 0, wxALL, 5 );


	bSizer38->Add( fgSizer3, 0, wxEXPAND, 5 );

	m_staticText70 = new wxStaticText( CommandsPanel, wxID_ANY, wxEmptyString, wxDefaultPosition, wxDefaultSize, 0 );
	m_staticText70->Wrap( -1 );
	bSizer38->Add( m_staticText70, 0, wxALL, 5 );


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
	m_splitter5->SplitVertically( ProfilesPanel, CommandsPanel, 349 );
	bSizer56->Add( m_splitter5, 1, wxEXPAND, 5 );


	this->SetSizer( bSizer56 );
	this->Layout();

	// Connect Events
	this->Connect( wxEVT_UPDATE_UI, wxUpdateUIEventHandler( RunProfilesPanel::OnUpdateUI ) );
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
	DuplicateProfileButton->Connect( wxEVT_COMMAND_BUTTON_CLICKED, wxCommandEventHandler( RunProfilesPanel::OnDuplicateProfileClick ), NULL, this );
	ImportButton->Connect( wxEVT_COMMAND_BUTTON_CLICKED, wxCommandEventHandler( RunProfilesPanel::OnImportButtonClick ), NULL, this );
	ExportButton->Connect( wxEVT_COMMAND_BUTTON_CLICKED, wxCommandEventHandler( RunProfilesPanel::OnExportButtonClick ), NULL, this );
	ManagerTextCtrl->Connect( wxEVT_COMMAND_TEXT_UPDATED, wxCommandEventHandler( RunProfilesPanel::ManagerTextChanged ), NULL, this );
	GuiAutoButton->Connect( wxEVT_COMMAND_BUTTON_CLICKED, wxCommandEventHandler( RunProfilesPanel::GuiAddressAutoClick ), NULL, this );
	ControllerSpecifyButton->Connect( wxEVT_COMMAND_BUTTON_CLICKED, wxCommandEventHandler( RunProfilesPanel::GuiAddressSpecifyClick ), NULL, this );
	ControllerAutoButton->Connect( wxEVT_COMMAND_BUTTON_CLICKED, wxCommandEventHandler( RunProfilesPanel::ControllerAddressAutoClick ), NULL, this );
	m_button38->Connect( wxEVT_COMMAND_BUTTON_CLICKED, wxCommandEventHandler( RunProfilesPanel::ControllerAddressSpecifyClick ), NULL, this );
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
	this->Disconnect( wxEVT_UPDATE_UI, wxUpdateUIEventHandler( RunProfilesPanel::OnUpdateUI ) );
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
	DuplicateProfileButton->Disconnect( wxEVT_COMMAND_BUTTON_CLICKED, wxCommandEventHandler( RunProfilesPanel::OnDuplicateProfileClick ), NULL, this );
	ImportButton->Disconnect( wxEVT_COMMAND_BUTTON_CLICKED, wxCommandEventHandler( RunProfilesPanel::OnImportButtonClick ), NULL, this );
	ExportButton->Disconnect( wxEVT_COMMAND_BUTTON_CLICKED, wxCommandEventHandler( RunProfilesPanel::OnExportButtonClick ), NULL, this );
	ManagerTextCtrl->Disconnect( wxEVT_COMMAND_TEXT_UPDATED, wxCommandEventHandler( RunProfilesPanel::ManagerTextChanged ), NULL, this );
	GuiAutoButton->Disconnect( wxEVT_COMMAND_BUTTON_CLICKED, wxCommandEventHandler( RunProfilesPanel::GuiAddressAutoClick ), NULL, this );
	ControllerSpecifyButton->Disconnect( wxEVT_COMMAND_BUTTON_CLICKED, wxCommandEventHandler( RunProfilesPanel::GuiAddressSpecifyClick ), NULL, this );
	ControllerAutoButton->Disconnect( wxEVT_COMMAND_BUTTON_CLICKED, wxCommandEventHandler( RunProfilesPanel::ControllerAddressAutoClick ), NULL, this );
	m_button38->Disconnect( wxEVT_COMMAND_BUTTON_CLICKED, wxCommandEventHandler( RunProfilesPanel::ControllerAddressSpecifyClick ), NULL, this );
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

AddRunCommandDialog::AddRunCommandDialog( wxWindow* parent, wxWindowID id, const wxString& title, const wxPoint& pos, const wxSize& size, long style ) : wxDialog( parent, id, title, pos, size, style )
{
	this->SetSizeHints( wxDefaultSize, wxDefaultSize );

	wxBoxSizer* bSizer45;
	bSizer45 = new wxBoxSizer( wxVERTICAL );

	wxFlexGridSizer* fgSizer2;
	fgSizer2 = new wxFlexGridSizer( 0, 2, 0, 0 );
	fgSizer2->AddGrowableCol( 1 );
	fgSizer2->SetFlexibleDirection( wxBOTH );
	fgSizer2->SetNonFlexibleGrowMode( wxFLEX_GROWMODE_SPECIFIED );

	m_staticText45 = new wxStaticText( this, wxID_ANY, wxT("Command To Run :"), wxDefaultPosition, wxDefaultSize, 0 );
	m_staticText45->Wrap( -1 );
	fgSizer2->Add( m_staticText45, 0, wxALIGN_CENTER_VERTICAL|wxALIGN_RIGHT|wxALL, 5 );

	CommandTextCtrl = new wxTextCtrl( this, wxID_ANY, wxT("$command"), wxDefaultPosition, wxDefaultSize, wxTE_PROCESS_ENTER );
	CommandTextCtrl->SetMinSize( wxSize( 300,-1 ) );

	fgSizer2->Add( CommandTextCtrl, 1, wxALL|wxEXPAND, 5 );

	m_staticText46 = new wxStaticText( this, wxID_ANY, wxT("No. Copies To Run :"), wxDefaultPosition, wxDefaultSize, 0 );
	m_staticText46->Wrap( -1 );
	fgSizer2->Add( m_staticText46, 0, wxALIGN_CENTER_VERTICAL|wxALIGN_RIGHT|wxALL, 5 );

	NumberCopiesSpinCtrl = new wxSpinCtrl( this, wxID_ANY, wxEmptyString, wxDefaultPosition, wxDefaultSize, wxSP_ARROW_KEYS, 1, 999999, 1 );
	fgSizer2->Add( NumberCopiesSpinCtrl, 1, wxALL|wxEXPAND, 5 );

	ThreadsPerCopySpinCtrl = new wxStaticText( this, wxID_ANY, wxT("No. Threads Per Copy :"), wxDefaultPosition, wxDefaultSize, 0 );
	ThreadsPerCopySpinCtrl->Wrap( -1 );
	fgSizer2->Add( ThreadsPerCopySpinCtrl, 0, wxALIGN_CENTER_VERTICAL|wxALL, 5 );

	NumberThreadsSpinCtrl = new wxSpinCtrl( this, wxID_ANY, wxEmptyString, wxDefaultPosition, wxDefaultSize, wxSP_ARROW_KEYS, 1, 999999, 1 );
	fgSizer2->Add( NumberThreadsSpinCtrl, 0, wxALL|wxEXPAND, 5 );

	m_staticText58 = new wxStaticText( this, wxID_ANY, wxT("Launch Delay (ms) :"), wxDefaultPosition, wxDefaultSize, 0 );
	m_staticText58->Wrap( -1 );
	fgSizer2->Add( m_staticText58, 0, wxALIGN_CENTER_VERTICAL|wxALIGN_RIGHT|wxALL, 5 );

	DelayTimeSpinCtrl = new wxSpinCtrl( this, wxID_ANY, wxEmptyString, wxDefaultPosition, wxDefaultSize, wxSP_ARROW_KEYS, 0, 10000, 100 );
	fgSizer2->Add( DelayTimeSpinCtrl, 0, wxALL|wxEXPAND, 5 );


	bSizer45->Add( fgSizer2, 0, wxEXPAND, 5 );

	m_staticline166 = new wxStaticLine( this, wxID_ANY, wxDefaultPosition, wxDefaultSize, wxLI_HORIZONTAL );
	bSizer45->Add( m_staticline166, 0, wxEXPAND | wxALL, 5 );

	ErrorStaticText = new wxStaticText( this, wxID_ANY, wxT("Oops! - Command must contain \"$command\""), wxDefaultPosition, wxDefaultSize, 0 );
	ErrorStaticText->Wrap( -1 );
	ErrorStaticText->SetForegroundColour( wxColour( 180, 0, 0 ) );
	ErrorStaticText->Hide();

	bSizer45->Add( ErrorStaticText, 0, wxALIGN_CENTER|wxALL, 5 );

	wxBoxSizer* bSizer590;
	bSizer590 = new wxBoxSizer( wxHORIZONTAL );

	OverrideCheckBox = new wxCheckBox( this, wxID_ANY, wxT("Override Apparent No. Copies :"), wxDefaultPosition, wxDefaultSize, 0 );
	bSizer590->Add( OverrideCheckBox, 0, wxALL, 5 );

	OverridenNoCopiesSpinCtrl = new wxSpinCtrl( this, wxID_ANY, wxEmptyString, wxDefaultPosition, wxDefaultSize, wxSP_ARROW_KEYS, 0, 99999999, 0 );
	OverridenNoCopiesSpinCtrl->Enable( false );

	bSizer590->Add( OverridenNoCopiesSpinCtrl, 1, wxALL|wxEXPAND, 5 );


	bSizer45->Add( bSizer590, 0, wxALIGN_CENTER, 5 );

	m_staticline14 = new wxStaticLine( this, wxID_ANY, wxDefaultPosition, wxDefaultSize, wxLI_HORIZONTAL );
	bSizer45->Add( m_staticline14, 0, wxEXPAND | wxALL, 5 );

	wxBoxSizer* bSizer47;
	bSizer47 = new wxBoxSizer( wxHORIZONTAL );

	OKButton = new wxButton( this, wxID_ANY, wxT("OK"), wxDefaultPosition, wxDefaultSize, 0 );
	bSizer47->Add( OKButton, 0, wxALL|wxEXPAND, 5 );

	CancelButton = new wxButton( this, wxID_ANY, wxT("Cancel"), wxDefaultPosition, wxDefaultSize, 0 );
	bSizer47->Add( CancelButton, 0, wxALL|wxEXPAND, 5 );


	bSizer45->Add( bSizer47, 0, wxALIGN_CENTER, 5 );


	this->SetSizer( bSizer45 );
	this->Layout();
	bSizer45->Fit( this );

	this->Centre( wxBOTH );

	// Connect Events
	CommandTextCtrl->Connect( wxEVT_COMMAND_TEXT_ENTER, wxCommandEventHandler( AddRunCommandDialog::OnEnter ), NULL, this );
	OverrideCheckBox->Connect( wxEVT_COMMAND_CHECKBOX_CLICKED, wxCommandEventHandler( AddRunCommandDialog::OnOverrideCheckbox ), NULL, this );
	OKButton->Connect( wxEVT_COMMAND_BUTTON_CLICKED, wxCommandEventHandler( AddRunCommandDialog::OnOKClick ), NULL, this );
	CancelButton->Connect( wxEVT_COMMAND_BUTTON_CLICKED, wxCommandEventHandler( AddRunCommandDialog::OnCancelClick ), NULL, this );
}

AddRunCommandDialog::~AddRunCommandDialog()
{
	// Disconnect Events
	CommandTextCtrl->Disconnect( wxEVT_COMMAND_TEXT_ENTER, wxCommandEventHandler( AddRunCommandDialog::OnEnter ), NULL, this );
	OverrideCheckBox->Disconnect( wxEVT_COMMAND_CHECKBOX_CLICKED, wxCommandEventHandler( AddRunCommandDialog::OnOverrideCheckbox ), NULL, this );
	OKButton->Disconnect( wxEVT_COMMAND_BUTTON_CLICKED, wxCommandEventHandler( AddRunCommandDialog::OnOKClick ), NULL, this );
	CancelButton->Disconnect( wxEVT_COMMAND_BUTTON_CLICKED, wxCommandEventHandler( AddRunCommandDialog::OnCancelClick ), NULL, this );

}
