///////////////////////////////////////////////////////////////////////////
// C++ code generated with wxFormBuilder (version 3.10.0-4761b0c)
// http://www.wxformbuilder.org/
//
// PLEASE DO *NOT* EDIT THIS FILE!
///////////////////////////////////////////////////////////////////////////

#include "../../core/gui_core_headers.h"

#include "display_gui.h"

///////////////////////////////////////////////////////////////////////////

DisplayPanelParent::DisplayPanelParent( wxWindow* parent, wxWindowID id, const wxPoint& pos, const wxSize& size, long style, const wxString& name ) : wxPanel( parent, id, pos, size, style, name )
{
	MainSizer = new wxBoxSizer( wxVERTICAL );

	Toolbar = new wxToolBar( this, wxID_ANY, wxDefaultPosition, wxDefaultSize, wxTB_FLAT|wxTB_HORIZONTAL );
	Toolbar->Realize();

	MainSizer->Add( Toolbar, 0, wxEXPAND, 5 );


	this->SetSizer( MainSizer );
	this->Layout();

	// Connect Events
	this->Connect( wxEVT_MIDDLE_UP, wxMouseEventHandler( DisplayPanelParent::OnMiddleUp ) );
}

DisplayPanelParent::~DisplayPanelParent()
{
	// Disconnect Events
	this->Disconnect( wxEVT_MIDDLE_UP, wxMouseEventHandler( DisplayPanelParent::OnMiddleUp ) );

}

DisplayFrameParent::DisplayFrameParent( wxWindow* parent, wxWindowID id, const wxString& title, const wxPoint& pos, const wxSize& size, long style ) : wxFrame( parent, id, title, pos, size, style )
{
	this->SetSizeHints( wxDefaultSize, wxDefaultSize );

	m_menubar2 = new wxMenuBar( 0 );
	DisplayFileMenu = new wxMenu();
	DisplayFileOpen = new wxMenuItem( DisplayFileMenu, wxID_ANY, wxString( wxT("Open Image") ) , wxEmptyString, wxITEM_NORMAL );
	DisplayFileMenu->Append( DisplayFileOpen );

	DisplayCloseTab = new wxMenuItem( DisplayFileMenu, wxID_ANY, wxString( wxT("Close tab") ) , wxEmptyString, wxITEM_NORMAL );
	DisplayFileMenu->Append( DisplayCloseTab );
	DisplayCloseTab->Enable( false );

	DisplayFileMenu->AppendSeparator();

	SaveDisplayedImages = new wxMenuItem( DisplayFileMenu, wxID_ANY, wxString( wxT("Save Displayed Image(s) As PNG") ) , wxEmptyString, wxITEM_NORMAL );
	DisplayFileMenu->Append( SaveDisplayedImages );
	SaveDisplayedImages->Enable( false );

	SaveDisplayedImagesWithLegend = new wxMenuItem( DisplayFileMenu, wxID_ANY, wxString( wxT("Save Displayed Image(s) As PNG with Legend") ) , wxEmptyString, wxITEM_NORMAL );
	DisplayFileMenu->Append( SaveDisplayedImagesWithLegend );
	SaveDisplayedImagesWithLegend->Enable( false );

	DisplayFileMenu->AppendSeparator();

	SelectOpenTxt = new wxMenuItem( DisplayFileMenu, wxID_ANY, wxString( wxT("Open Text File") ) , wxEmptyString, wxITEM_NORMAL );
	DisplayFileMenu->Append( SelectOpenTxt );
	SelectOpenTxt->Enable( false );

	SelectSaveTxt = new wxMenuItem( DisplayFileMenu, wxID_ANY, wxString( wxT("Save Text File") ) , wxEmptyString, wxITEM_NORMAL );
	DisplayFileMenu->Append( SelectSaveTxt );
	SelectSaveTxt->Enable( false );

	SelectSaveTxtAs = new wxMenuItem( DisplayFileMenu, wxID_ANY, wxString( wxT("Save Text File As") ) , wxEmptyString, wxITEM_NORMAL );
	DisplayFileMenu->Append( SelectSaveTxtAs );
	SelectSaveTxtAs->Enable( false );

	DisplayFileMenu->AppendSeparator();

	DisplayExit = new wxMenuItem( DisplayFileMenu, wxID_ANY, wxString( wxT("Exit") ) , wxEmptyString, wxITEM_NORMAL );
	DisplayFileMenu->Append( DisplayExit );

	m_menubar2->Append( DisplayFileMenu, wxT("File") );

	DisplayLabelMenu = new wxMenu();
	LabelLocationNumber = new wxMenuItem( DisplayLabelMenu, wxID_ANY, wxString( wxT("Location Number") ) , wxEmptyString, wxITEM_CHECK );
	DisplayLabelMenu->Append( LabelLocationNumber );
	LabelLocationNumber->Enable( false );
	LabelLocationNumber->Check( true );

	m_menubar2->Append( DisplayLabelMenu, wxT("Label") );

	DisplaySelectMenu = new wxMenu();
	SelectImageSelectionMode = new wxMenuItem( DisplaySelectMenu, wxID_ANY, wxString( wxT("Image Selection Mode") ) , wxEmptyString, wxITEM_RADIO );
	DisplaySelectMenu->Append( SelectImageSelectionMode );
	SelectImageSelectionMode->Enable( false );
	SelectImageSelectionMode->Check( true );

	SelectCoordsSelectionMode = new wxMenuItem( DisplaySelectMenu, wxID_ANY, wxString( wxT("Coords Selection Mode") ) , wxEmptyString, wxITEM_RADIO );
	DisplaySelectMenu->Append( SelectCoordsSelectionMode );
	SelectCoordsSelectionMode->Enable( false );

	DisplaySelectMenu->AppendSeparator();

	SelectInvertSelection = new wxMenuItem( DisplaySelectMenu, wxID_ANY, wxString( wxT("Invert Selection") ) , wxEmptyString, wxITEM_NORMAL );
	DisplaySelectMenu->Append( SelectInvertSelection );
	SelectInvertSelection->Enable( false );

	SelectClearSelection = new wxMenuItem( DisplaySelectMenu, wxID_ANY, wxString( wxT("Clear Selection") ) , wxEmptyString, wxITEM_NORMAL );
	DisplaySelectMenu->Append( SelectClearSelection );
	SelectClearSelection->Enable( false );

	m_menubar2->Append( DisplaySelectMenu, wxT("Select") );

	DisplayOptionsMenu = new wxMenu();
	OptionsSetPointSize = new wxMenu();
	wxMenuItem* OptionsSetPointSizeItem = new wxMenuItem( DisplayOptionsMenu, wxID_ANY, wxT("Set Point Size"), wxEmptyString, wxITEM_NORMAL, OptionsSetPointSize );
	CoordSize3 = new wxMenuItem( OptionsSetPointSize, wxID_ANY, wxString( wxT("3") ) , wxEmptyString, wxITEM_RADIO );
	OptionsSetPointSize->Append( CoordSize3 );

	CoordSize5 = new wxMenuItem( OptionsSetPointSize, wxID_ANY, wxString( wxT("5") ) , wxEmptyString, wxITEM_RADIO );
	OptionsSetPointSize->Append( CoordSize5 );

	CoordSize7 = new wxMenuItem( OptionsSetPointSize, wxID_ANY, wxString( wxT("7") ) , wxEmptyString, wxITEM_RADIO );
	OptionsSetPointSize->Append( CoordSize7 );

	CoordSize10 = new wxMenuItem( OptionsSetPointSize, wxID_ANY, wxString( wxT("10") ) , wxEmptyString, wxITEM_RADIO );
	OptionsSetPointSize->Append( CoordSize10 );

	DisplayOptionsMenu->Append( OptionsSetPointSizeItem );

	OptionsSingleImageMode = new wxMenuItem( DisplayOptionsMenu, wxID_ANY, wxString( wxT("Single Image Mode") ) , wxEmptyString, wxITEM_CHECK );
	DisplayOptionsMenu->Append( OptionsSingleImageMode );
	OptionsSingleImageMode->Enable( false );

	OptionsShowSelectionDistances = new wxMenuItem( DisplayOptionsMenu, wxID_ANY, wxString( wxT("Show Selection Distances") ) , wxEmptyString, wxITEM_CHECK );
	DisplayOptionsMenu->Append( OptionsShowSelectionDistances );
	OptionsShowSelectionDistances->Enable( false );

	DisplayOptionsMenu->AppendSeparator();

	OptionsShowResolution = new wxMenuItem( DisplayOptionsMenu, wxID_ANY, wxString( wxT("Show Resolution Instead of Radius") ) , wxEmptyString, wxITEM_CHECK );
	DisplayOptionsMenu->Append( OptionsShowResolution );
	OptionsShowResolution->Enable( false );

	m_menubar2->Append( DisplayOptionsMenu, wxT("Options") );

	DisplayHelpMenu = new wxMenu();
	HelpDisplayControls = new wxMenuItem( DisplayHelpMenu, wxID_ANY, wxString( wxT("Display Controls") ) , wxT("User Manual for cisTEM Display"), wxITEM_NORMAL );
	DisplayHelpMenu->Append( HelpDisplayControls );

	HelpAbout = new wxMenuItem( DisplayHelpMenu, wxID_ANY, wxString( wxT("Documentation") ) , wxEmptyString, wxITEM_NORMAL );
	DisplayHelpMenu->Append( HelpAbout );

	m_menubar2->Append( DisplayHelpMenu, wxT("Help") );

	this->SetMenuBar( m_menubar2 );

	bSizer631 = new wxBoxSizer( wxVERTICAL );

	cisTEMDisplayPanel = new DisplayPanel( this, wxID_ANY, wxDefaultPosition, wxDefaultSize, wxTAB_TRAVERSAL );
	bSizer631->Add( cisTEMDisplayPanel, 1, wxEXPAND | wxALL, 5 );


	this->SetSizer( bSizer631 );
	this->Layout();

	this->Centre( wxBOTH );

	// Connect Events
	this->Connect( wxEVT_UPDATE_UI, wxUpdateUIEventHandler( DisplayFrameParent::OnUpdateUI ) );
	DisplayFileMenu->Bind(wxEVT_COMMAND_MENU_SELECTED, wxCommandEventHandler( DisplayFrameParent::OnFileOpenClick ), this, DisplayFileOpen->GetId());
	DisplayFileMenu->Bind(wxEVT_COMMAND_MENU_SELECTED, wxCommandEventHandler( DisplayFrameParent::OnCloseTabClick ), this, DisplayCloseTab->GetId());
	DisplayFileMenu->Bind(wxEVT_COMMAND_MENU_SELECTED, wxCommandEventHandler( DisplayFrameParent::OnSaveDisplayedImagesClick ), this, SaveDisplayedImages->GetId());
	DisplayFileMenu->Bind(wxEVT_COMMAND_MENU_SELECTED, wxCommandEventHandler( DisplayFrameParent::OnSaveDisplayedImagesWithLegendClick ), this, SaveDisplayedImagesWithLegend->GetId());
	DisplayFileMenu->Bind(wxEVT_COMMAND_MENU_SELECTED, wxCommandEventHandler( DisplayFrameParent::OnOpenTxtClick ), this, SelectOpenTxt->GetId());
	DisplayFileMenu->Bind(wxEVT_COMMAND_MENU_SELECTED, wxCommandEventHandler( DisplayFrameParent::OnSaveTxtClick ), this, SelectSaveTxt->GetId());
	DisplayFileMenu->Bind(wxEVT_COMMAND_MENU_SELECTED, wxCommandEventHandler( DisplayFrameParent::OnSaveTxtAsClick ), this, SelectSaveTxtAs->GetId());
	DisplayFileMenu->Bind(wxEVT_COMMAND_MENU_SELECTED, wxCommandEventHandler( DisplayFrameParent::OnExitClick ), this, DisplayExit->GetId());
	DisplayLabelMenu->Bind(wxEVT_COMMAND_MENU_SELECTED, wxCommandEventHandler( DisplayFrameParent::OnLocationNumberClick ), this, LabelLocationNumber->GetId());
	DisplaySelectMenu->Bind(wxEVT_COMMAND_MENU_SELECTED, wxCommandEventHandler( DisplayFrameParent::OnImageSelectionModeClick ), this, SelectImageSelectionMode->GetId());
	DisplaySelectMenu->Bind(wxEVT_COMMAND_MENU_SELECTED, wxCommandEventHandler( DisplayFrameParent::OnCoordsSelectionModeClick ), this, SelectCoordsSelectionMode->GetId());
	DisplaySelectMenu->Bind(wxEVT_COMMAND_MENU_SELECTED, wxCommandEventHandler( DisplayFrameParent::OnInvertSelectionClick ), this, SelectInvertSelection->GetId());
	DisplaySelectMenu->Bind(wxEVT_COMMAND_MENU_SELECTED, wxCommandEventHandler( DisplayFrameParent::OnClearSelectionClick ), this, SelectClearSelection->GetId());
	OptionsSetPointSize->Bind(wxEVT_COMMAND_MENU_SELECTED, wxCommandEventHandler( DisplayFrameParent::OnSize3 ), this, CoordSize3->GetId());
	OptionsSetPointSize->Bind(wxEVT_COMMAND_MENU_SELECTED, wxCommandEventHandler( DisplayFrameParent::OnSize5 ), this, CoordSize5->GetId());
	OptionsSetPointSize->Bind(wxEVT_COMMAND_MENU_SELECTED, wxCommandEventHandler( DisplayFrameParent::OnSize7 ), this, CoordSize7->GetId());
	OptionsSetPointSize->Bind(wxEVT_COMMAND_MENU_SELECTED, wxCommandEventHandler( DisplayFrameParent::OnSize10 ), this, CoordSize10->GetId());
	DisplayOptionsMenu->Bind(wxEVT_COMMAND_MENU_SELECTED, wxCommandEventHandler( DisplayFrameParent::OnSingleImageModeClick ), this, OptionsSingleImageMode->GetId());
	DisplayOptionsMenu->Bind(wxEVT_COMMAND_MENU_SELECTED, wxCommandEventHandler( DisplayFrameParent::OnShowSelectionDistancesClick ), this, OptionsShowSelectionDistances->GetId());
	DisplayOptionsMenu->Bind(wxEVT_COMMAND_MENU_SELECTED, wxCommandEventHandler( DisplayFrameParent::OnShowResolution ), this, OptionsShowResolution->GetId());
	DisplayHelpMenu->Bind(wxEVT_COMMAND_MENU_SELECTED, wxCommandEventHandler( DisplayFrameParent::OnDisplayControlsClick ), this, HelpDisplayControls->GetId());
	DisplayHelpMenu->Bind(wxEVT_COMMAND_MENU_SELECTED, wxCommandEventHandler( DisplayFrameParent::OnDocumentationClick ), this, HelpAbout->GetId());
}

DisplayFrameParent::~DisplayFrameParent()
{
	// Disconnect Events
	this->Disconnect( wxEVT_UPDATE_UI, wxUpdateUIEventHandler( DisplayFrameParent::OnUpdateUI ) );

}

DisplayManualDialogParent::DisplayManualDialogParent( wxWindow* parent, wxWindowID id, const wxString& title, const wxPoint& pos, const wxSize& size, long style ) : wxDialog( parent, id, title, pos, size, style )
{
	this->SetSizeHints( wxDefaultSize, wxDefaultSize );

	MainSizer = new wxBoxSizer( wxVERTICAL );


	MainSizer->Add( 400, 200, 0, wxFIXED_MINSIZE, 5 );

	m_staticline58 = new wxStaticLine( this, wxID_ANY, wxDefaultPosition, wxDefaultSize, wxLI_HORIZONTAL );
	MainSizer->Add( m_staticline58, 0, wxEXPAND | wxALL, 5 );

	wxBoxSizer* bSizer262;
	bSizer262 = new wxBoxSizer( wxHORIZONTAL );


	bSizer262->Add( 0, 0, 1, wxEXPAND, 5 );

	m_staticText315 = new wxStaticText( this, wxID_ANY, wxT("Min : "), wxDefaultPosition, wxDefaultSize, 0 );
	m_staticText315->Wrap( -1 );
	bSizer262->Add( m_staticText315, 0, wxALIGN_CENTER_VERTICAL|wxALL, 5 );

	minimum_text_ctrl = new wxTextCtrl( this, wxID_ANY, wxEmptyString, wxDefaultPosition, wxDefaultSize, wxTE_PROCESS_ENTER|wxTE_CENTER );
	bSizer262->Add( minimum_text_ctrl, 0, wxALL, 5 );

	m_staticText316 = new wxStaticText( this, wxID_ANY, wxT("/"), wxDefaultPosition, wxDefaultSize, 0 );
	m_staticText316->Wrap( -1 );
	bSizer262->Add( m_staticText316, 0, wxALIGN_CENTER_VERTICAL|wxALL, 5 );

	maximum_text_ctrl = new wxTextCtrl( this, wxID_ANY, wxEmptyString, wxDefaultPosition, wxDefaultSize, wxTE_PROCESS_ENTER|wxTE_CENTER );
	bSizer262->Add( maximum_text_ctrl, 0, wxALL, 5 );

	m_staticText317 = new wxStaticText( this, wxID_ANY, wxT(": Max"), wxDefaultPosition, wxDefaultSize, 0 );
	m_staticText317->Wrap( -1 );
	bSizer262->Add( m_staticText317, 0, wxALIGN_CENTER_VERTICAL|wxALL, 5 );


	bSizer262->Add( 0, 0, 1, wxEXPAND, 5 );


	MainSizer->Add( bSizer262, 0, wxEXPAND, 5 );

	wxBoxSizer* bSizer265;
	bSizer265 = new wxBoxSizer( wxHORIZONTAL );


	bSizer265->Add( 0, 0, 1, wxEXPAND, 5 );

	Toolbar = new wxToolBar( this, wxID_ANY, wxDefaultPosition, wxDefaultSize, wxTB_HORIZONTAL );
	Toolbar->Realize();

	bSizer265->Add( Toolbar, 0, 0, 5 );


	bSizer265->Add( 0, 0, 1, wxEXPAND, 5 );


	MainSizer->Add( bSizer265, 0, wxEXPAND, 5 );

	m_staticline61 = new wxStaticLine( this, wxID_ANY, wxDefaultPosition, wxDefaultSize, wxLI_HORIZONTAL );
	MainSizer->Add( m_staticline61, 0, wxEXPAND | wxALL, 5 );

	wxGridSizer* gSizer13;
	gSizer13 = new wxGridSizer( 0, 2, 0, 0 );

	histogram_checkbox = new wxCheckBox( this, wxID_ANY, wxT("Use Entire File For Histogram"), wxDefaultPosition, wxDefaultSize, 0 );
	gSizer13->Add( histogram_checkbox, 0, wxALIGN_CENTER_HORIZONTAL|wxALL, 5 );

	live_checkbox = new wxCheckBox( this, wxID_ANY, wxT("Live Update of Display"), wxDefaultPosition, wxDefaultSize, 0 );
	live_checkbox->SetValue(true);
	gSizer13->Add( live_checkbox, 0, wxALIGN_CENTER_HORIZONTAL|wxALL, 5 );


	MainSizer->Add( gSizer13, 0, wxEXPAND, 5 );

	m_staticline63 = new wxStaticLine( this, wxID_ANY, wxDefaultPosition, wxDefaultSize, wxLI_HORIZONTAL );
	MainSizer->Add( m_staticline63, 0, wxEXPAND | wxALL, 5 );

	wxBoxSizer* bSizer264;
	bSizer264 = new wxBoxSizer( wxHORIZONTAL );


	bSizer264->Add( 0, 0, 1, wxEXPAND, 5 );

	m_button94 = new wxButton( this, wxID_ANY, wxT("OK"), wxDefaultPosition, wxDefaultSize, 0 );
	bSizer264->Add( m_button94, 0, wxALL, 5 );

	m_button95 = new wxButton( this, wxID_ANY, wxT("Cancel"), wxDefaultPosition, wxDefaultSize, 0 );
	bSizer264->Add( m_button95, 0, wxALL, 5 );


	bSizer264->Add( 0, 0, 1, wxEXPAND, 5 );


	MainSizer->Add( bSizer264, 0, wxEXPAND, 5 );


	this->SetSizer( MainSizer );
	this->Layout();
	MainSizer->Fit( this );

	this->Centre( wxBOTH );

	// Connect Events
	this->Connect( wxEVT_CLOSE_WINDOW, wxCloseEventHandler( DisplayManualDialogParent::OnClose ) );
	this->Connect( wxEVT_LEFT_DOWN, wxMouseEventHandler( DisplayManualDialogParent::OnLeftDown ) );
	this->Connect( wxEVT_MOTION, wxMouseEventHandler( DisplayManualDialogParent::OnMotion ) );
	this->Connect( wxEVT_PAINT, wxPaintEventHandler( DisplayManualDialogParent::OnPaint ) );
	this->Connect( wxEVT_RIGHT_DOWN, wxMouseEventHandler( DisplayManualDialogParent::OnRightDown ) );
	minimum_text_ctrl->Connect( wxEVT_COMMAND_TEXT_ENTER, wxCommandEventHandler( DisplayManualDialogParent::OnLowChange ), NULL, this );
	maximum_text_ctrl->Connect( wxEVT_COMMAND_TEXT_ENTER, wxCommandEventHandler( DisplayManualDialogParent::OnHighChange ), NULL, this );
	histogram_checkbox->Connect( wxEVT_COMMAND_CHECKBOX_CLICKED, wxCommandEventHandler( DisplayManualDialogParent::OnHistogramCheck ), NULL, this );
	live_checkbox->Connect( wxEVT_COMMAND_CHECKBOX_CLICKED, wxCommandEventHandler( DisplayManualDialogParent::OnRealtimeCheck ), NULL, this );
	m_button94->Connect( wxEVT_COMMAND_BUTTON_CLICKED, wxCommandEventHandler( DisplayManualDialogParent::OnButtonOK ), NULL, this );
	m_button95->Connect( wxEVT_COMMAND_BUTTON_CLICKED, wxCommandEventHandler( DisplayManualDialogParent::OnButtonCancel ), NULL, this );
}

DisplayManualDialogParent::~DisplayManualDialogParent()
{
	// Disconnect Events
	this->Disconnect( wxEVT_CLOSE_WINDOW, wxCloseEventHandler( DisplayManualDialogParent::OnClose ) );
	this->Disconnect( wxEVT_LEFT_DOWN, wxMouseEventHandler( DisplayManualDialogParent::OnLeftDown ) );
	this->Disconnect( wxEVT_MOTION, wxMouseEventHandler( DisplayManualDialogParent::OnMotion ) );
	this->Disconnect( wxEVT_PAINT, wxPaintEventHandler( DisplayManualDialogParent::OnPaint ) );
	this->Disconnect( wxEVT_RIGHT_DOWN, wxMouseEventHandler( DisplayManualDialogParent::OnRightDown ) );
	minimum_text_ctrl->Disconnect( wxEVT_COMMAND_TEXT_ENTER, wxCommandEventHandler( DisplayManualDialogParent::OnLowChange ), NULL, this );
	maximum_text_ctrl->Disconnect( wxEVT_COMMAND_TEXT_ENTER, wxCommandEventHandler( DisplayManualDialogParent::OnHighChange ), NULL, this );
	histogram_checkbox->Disconnect( wxEVT_COMMAND_CHECKBOX_CLICKED, wxCommandEventHandler( DisplayManualDialogParent::OnHistogramCheck ), NULL, this );
	live_checkbox->Disconnect( wxEVT_COMMAND_CHECKBOX_CLICKED, wxCommandEventHandler( DisplayManualDialogParent::OnRealtimeCheck ), NULL, this );
	m_button94->Disconnect( wxEVT_COMMAND_BUTTON_CLICKED, wxCommandEventHandler( DisplayManualDialogParent::OnButtonOK ), NULL, this );
	m_button95->Disconnect( wxEVT_COMMAND_BUTTON_CLICKED, wxCommandEventHandler( DisplayManualDialogParent::OnButtonCancel ), NULL, this );

}
