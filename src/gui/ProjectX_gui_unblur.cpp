///////////////////////////////////////////////////////////////////////////
// C++ code generated with wxFormBuilder (version 3.10.0-4761b0c)
// http://www.wxformbuilder.org/
//
// PLEASE DO *NOT* EDIT THIS FILE!
///////////////////////////////////////////////////////////////////////////

#include "../core/gui_core_headers.h"

#include "AssetPickerComboPanel.h"
#include "BitmapPanel.h"
#include "DisplayPanel.h"
#include "ResultsDataViewListCtrl.h"
#include "UnblurResultsPanel.h"
#include "my_controls.h"

#include "ProjectX_gui_unblur.h"

///////////////////////////////////////////////////////////////////////////

MovieAlignResultsPanel::MovieAlignResultsPanel( wxWindow* parent, wxWindowID id, const wxPoint& pos, const wxSize& size, long style, const wxString& name ) : wxPanel( parent, id, pos, size, style, name )
{
	wxBoxSizer* bSizer63;
	bSizer63 = new wxBoxSizer( wxVERTICAL );

	m_staticline25 = new wxStaticLine( this, wxID_ANY, wxDefaultPosition, wxDefaultSize, wxLI_HORIZONTAL );
	bSizer63->Add( m_staticline25, 0, wxEXPAND | wxALL, 5 );

	m_splitter4 = new wxSplitterWindow( this, wxID_ANY, wxDefaultPosition, wxDefaultSize, wxSP_3D );
	m_splitter4->SetSashGravity( 0.5 );
	m_splitter4->Connect( wxEVT_IDLE, wxIdleEventHandler( MovieAlignResultsPanel::m_splitter4OnIdle ), NULL, this );

	m_panel13 = new wxPanel( m_splitter4, wxID_ANY, wxDefaultPosition, wxDefaultSize, wxTAB_TRAVERSAL );
	wxBoxSizer* bSizer66;
	bSizer66 = new wxBoxSizer( wxVERTICAL );

	wxBoxSizer* bSizer64;
	bSizer64 = new wxBoxSizer( wxHORIZONTAL );

	AllMoviesButton = new wxRadioButton( m_panel13, wxID_ANY, wxT("All Movies"), wxDefaultPosition, wxDefaultSize, 0 );
	bSizer64->Add( AllMoviesButton, 0, wxALIGN_CENTER_VERTICAL|wxALL, 5 );

	ByFilterButton = new wxRadioButton( m_panel13, wxID_ANY, wxT("By Filter"), wxDefaultPosition, wxDefaultSize, 0 );
	bSizer64->Add( ByFilterButton, 0, wxALIGN_CENTER_VERTICAL|wxALL, 5 );

	FilterButton = new wxButton( m_panel13, wxID_ANY, wxT("Define Filter"), wxDefaultPosition, wxDefaultSize, 0 );
	FilterButton->Enable( false );

	bSizer64->Add( FilterButton, 0, wxALIGN_CENTER_VERTICAL|wxALL, 5 );


	bSizer64->Add( 0, 0, 1, wxEXPAND, 5 );

	m_staticline76 = new wxStaticLine( m_panel13, wxID_ANY, wxDefaultPosition, wxDefaultSize, wxLI_HORIZONTAL|wxLI_VERTICAL );
	bSizer64->Add( m_staticline76, 0, wxEXPAND | wxALL, 5 );

	JobDetailsToggleButton = new wxToggleButton( m_panel13, wxID_ANY, wxT("Show Job Details"), wxDefaultPosition, wxDefaultSize, 0 );
	bSizer64->Add( JobDetailsToggleButton, 0, wxALIGN_CENTER|wxALIGN_RIGHT|wxALL|wxEXPAND, 5 );


	bSizer66->Add( bSizer64, 0, wxEXPAND, 5 );

	ResultDataView = new ResultsDataViewListCtrl( m_panel13, wxID_ANY, wxDefaultPosition, wxDefaultSize, wxDV_VERT_RULES );
	bSizer66->Add( ResultDataView, 1, wxALL|wxEXPAND, 5 );

	wxBoxSizer* bSizer68;
	bSizer68 = new wxBoxSizer( wxHORIZONTAL );

	PreviousButton = new wxButton( m_panel13, wxID_ANY, wxT("&Previous"), wxDefaultPosition, wxDefaultSize, 0 );
	bSizer68->Add( PreviousButton, 0, wxALL, 5 );


	bSizer68->Add( 0, 0, 1, wxEXPAND, 5 );

	AddAllToGroupButton = new wxButton( m_panel13, wxID_ANY, wxT("Add All To Group"), wxDefaultPosition, wxDefaultSize, 0 );
	bSizer68->Add( AddAllToGroupButton, 0, wxALL, 5 );


	bSizer68->Add( 0, 0, 1, 0, 5 );

	NextButton = new wxButton( m_panel13, wxID_ANY, wxT("&Next"), wxDefaultPosition, wxDefaultSize, 0 );
	bSizer68->Add( NextButton, 0, wxALL, 5 );


	bSizer66->Add( bSizer68, 0, wxEXPAND, 5 );


	m_panel13->SetSizer( bSizer66 );
	m_panel13->Layout();
	bSizer66->Fit( m_panel13 );
	RightPanel = new wxPanel( m_splitter4, wxID_ANY, wxDefaultPosition, wxDefaultSize, wxTAB_TRAVERSAL );
	wxBoxSizer* bSizer681;
	bSizer681 = new wxBoxSizer( wxVERTICAL );

	wxBoxSizer* bSizer73;
	bSizer73 = new wxBoxSizer( wxVERTICAL );

	JobDetailsPanel = new wxPanel( RightPanel, wxID_ANY, wxDefaultPosition, wxDefaultSize, wxTAB_TRAVERSAL );
	JobDetailsPanel->Hide();

	InfoSizer = new wxFlexGridSizer( 0, 6, 0, 0 );
	InfoSizer->SetFlexibleDirection( wxBOTH );
	InfoSizer->SetNonFlexibleGrowMode( wxFLEX_GROWMODE_SPECIFIED );

	m_staticText72 = new wxStaticText( JobDetailsPanel, wxID_ANY, wxT("Alignment ID :"), wxDefaultPosition, wxDefaultSize, 0 );
	m_staticText72->Wrap( -1 );
	m_staticText72->SetFont( wxFont( 10, wxFONTFAMILY_SWISS, wxFONTSTYLE_NORMAL, wxFONTWEIGHT_BOLD, false, wxT("Sans") ) );

	InfoSizer->Add( m_staticText72, 0, wxALIGN_RIGHT|wxALL, 5 );

	AlignmentIDStaticText = new wxStaticText( JobDetailsPanel, wxID_ANY, wxEmptyString, wxDefaultPosition, wxDefaultSize, 0 );
	AlignmentIDStaticText->Wrap( -1 );
	InfoSizer->Add( AlignmentIDStaticText, 0, wxALIGN_LEFT|wxALL, 5 );

	m_staticText74 = new wxStaticText( JobDetailsPanel, wxID_ANY, wxT("Date of Run :"), wxDefaultPosition, wxDefaultSize, 0 );
	m_staticText74->Wrap( -1 );
	m_staticText74->SetFont( wxFont( 10, wxFONTFAMILY_SWISS, wxFONTSTYLE_NORMAL, wxFONTWEIGHT_BOLD, false, wxT("Sans") ) );

	InfoSizer->Add( m_staticText74, 0, wxALIGN_RIGHT|wxALL, 5 );

	DateOfRunStaticText = new wxStaticText( JobDetailsPanel, wxID_ANY, wxEmptyString, wxDefaultPosition, wxDefaultSize, 0 );
	DateOfRunStaticText->Wrap( -1 );
	InfoSizer->Add( DateOfRunStaticText, 0, wxALL, 5 );

	m_staticText93 = new wxStaticText( JobDetailsPanel, wxID_ANY, wxT("Time Of Run :"), wxDefaultPosition, wxDefaultSize, 0 );
	m_staticText93->Wrap( -1 );
	m_staticText93->SetFont( wxFont( 10, wxFONTFAMILY_SWISS, wxFONTSTYLE_NORMAL, wxFONTWEIGHT_BOLD, false, wxT("Sans") ) );

	InfoSizer->Add( m_staticText93, 0, wxALIGN_RIGHT|wxALL, 5 );

	TimeOfRunStaticText = new wxStaticText( JobDetailsPanel, wxID_ANY, wxEmptyString, wxDefaultPosition, wxDefaultSize, 0 );
	TimeOfRunStaticText->Wrap( -1 );
	InfoSizer->Add( TimeOfRunStaticText, 0, wxALIGN_LEFT|wxALL, 5 );

	m_staticText83 = new wxStaticText( JobDetailsPanel, wxID_ANY, wxT("Voltage :"), wxDefaultPosition, wxDefaultSize, 0 );
	m_staticText83->Wrap( -1 );
	m_staticText83->SetFont( wxFont( 10, wxFONTFAMILY_SWISS, wxFONTSTYLE_NORMAL, wxFONTWEIGHT_BOLD, false, wxT("Sans") ) );

	InfoSizer->Add( m_staticText83, 0, wxALIGN_RIGHT|wxALL, 5 );

	VoltageStaticText = new wxStaticText( JobDetailsPanel, wxID_ANY, wxEmptyString, wxDefaultPosition, wxDefaultSize, 0 );
	VoltageStaticText->Wrap( -1 );
	InfoSizer->Add( VoltageStaticText, 0, wxALIGN_LEFT|wxALL, 5 );

	m_staticText78 = new wxStaticText( JobDetailsPanel, wxID_ANY, wxT("Pixel Size :"), wxDefaultPosition, wxDefaultSize, 0 );
	m_staticText78->Wrap( -1 );
	m_staticText78->SetFont( wxFont( 10, wxFONTFAMILY_SWISS, wxFONTSTYLE_NORMAL, wxFONTWEIGHT_BOLD, false, wxT("Sans") ) );

	InfoSizer->Add( m_staticText78, 0, wxALIGN_RIGHT|wxALL, 5 );

	PixelSizeStaticText = new wxStaticText( JobDetailsPanel, wxID_ANY, wxEmptyString, wxDefaultPosition, wxDefaultSize, 0 );
	PixelSizeStaticText->Wrap( -1 );
	InfoSizer->Add( PixelSizeStaticText, 0, wxALIGN_LEFT|wxALL, 5 );

	m_staticText82 = new wxStaticText( JobDetailsPanel, wxID_ANY, wxT("Exp. per Frame :"), wxDefaultPosition, wxDefaultSize, 0 );
	m_staticText82->Wrap( -1 );
	m_staticText82->SetFont( wxFont( 10, wxFONTFAMILY_SWISS, wxFONTSTYLE_NORMAL, wxFONTWEIGHT_BOLD, false, wxT("Sans") ) );

	InfoSizer->Add( m_staticText82, 0, wxALIGN_RIGHT|wxALL, 5 );

	ExposureStaticText = new wxStaticText( JobDetailsPanel, wxID_ANY, wxEmptyString, wxDefaultPosition, wxDefaultSize, 0 );
	ExposureStaticText->Wrap( -1 );
	InfoSizer->Add( ExposureStaticText, 0, wxALIGN_LEFT|wxALL, 5 );

	m_staticText96 = new wxStaticText( JobDetailsPanel, wxID_ANY, wxT("Pre Exp. :"), wxDefaultPosition, wxDefaultSize, 0 );
	m_staticText96->Wrap( -1 );
	m_staticText96->SetFont( wxFont( 10, wxFONTFAMILY_SWISS, wxFONTSTYLE_NORMAL, wxFONTWEIGHT_BOLD, false, wxT("Sans") ) );

	InfoSizer->Add( m_staticText96, 0, wxALIGN_RIGHT|wxALL, 5 );

	PreExposureStaticText = new wxStaticText( JobDetailsPanel, wxID_ANY, wxEmptyString, wxDefaultPosition, wxDefaultSize, 0 );
	PreExposureStaticText->Wrap( -1 );
	InfoSizer->Add( PreExposureStaticText, 0, wxALIGN_LEFT|wxALL, 5 );

	m_staticText85 = new wxStaticText( JobDetailsPanel, wxID_ANY, wxT("Min. Shift :"), wxDefaultPosition, wxDefaultSize, 0 );
	m_staticText85->Wrap( -1 );
	m_staticText85->SetFont( wxFont( 10, wxFONTFAMILY_SWISS, wxFONTSTYLE_NORMAL, wxFONTWEIGHT_BOLD, false, wxT("Sans") ) );

	InfoSizer->Add( m_staticText85, 0, wxALIGN_RIGHT|wxALL, 5 );

	MinShiftStaticText = new wxStaticText( JobDetailsPanel, wxID_ANY, wxEmptyString, wxDefaultPosition, wxDefaultSize, 0 );
	MinShiftStaticText->Wrap( -1 );
	InfoSizer->Add( MinShiftStaticText, 0, wxALIGN_LEFT|wxALL, 5 );

	m_staticText87 = new wxStaticText( JobDetailsPanel, wxID_ANY, wxT("Max. Shift :"), wxDefaultPosition, wxDefaultSize, 0 );
	m_staticText87->Wrap( -1 );
	m_staticText87->SetFont( wxFont( 10, wxFONTFAMILY_SWISS, wxFONTSTYLE_NORMAL, wxFONTWEIGHT_BOLD, false, wxT("Sans") ) );

	InfoSizer->Add( m_staticText87, 0, wxALIGN_RIGHT|wxALL, 5 );

	MaxShiftStaticText = new wxStaticText( JobDetailsPanel, wxID_ANY, wxEmptyString, wxDefaultPosition, wxDefaultSize, 0 );
	MaxShiftStaticText->Wrap( -1 );
	InfoSizer->Add( MaxShiftStaticText, 0, wxALIGN_LEFT|wxALL, 5 );

	m_staticText89 = new wxStaticText( JobDetailsPanel, wxID_ANY, wxT("Term. Threshold :"), wxDefaultPosition, wxDefaultSize, 0 );
	m_staticText89->Wrap( -1 );
	m_staticText89->SetFont( wxFont( 10, wxFONTFAMILY_SWISS, wxFONTSTYLE_NORMAL, wxFONTWEIGHT_BOLD, false, wxT("Sans") ) );

	InfoSizer->Add( m_staticText89, 0, wxALIGN_RIGHT|wxALL, 5 );

	TerminationThresholdStaticText = new wxStaticText( JobDetailsPanel, wxID_ANY, wxEmptyString, wxDefaultPosition, wxDefaultSize, 0 );
	TerminationThresholdStaticText->Wrap( -1 );
	InfoSizer->Add( TerminationThresholdStaticText, 0, wxALIGN_LEFT|wxALL, 5 );

	m_staticText91 = new wxStaticText( JobDetailsPanel, wxID_ANY, wxT("Max. Iterations :"), wxDefaultPosition, wxDefaultSize, 0 );
	m_staticText91->Wrap( -1 );
	m_staticText91->SetFont( wxFont( 10, wxFONTFAMILY_SWISS, wxFONTSTYLE_NORMAL, wxFONTWEIGHT_BOLD, false, wxT("Sans") ) );

	InfoSizer->Add( m_staticText91, 0, wxALIGN_RIGHT|wxALL, 5 );

	MaxIterationsStaticText = new wxStaticText( JobDetailsPanel, wxID_ANY, wxEmptyString, wxDefaultPosition, wxDefaultSize, 0 );
	MaxIterationsStaticText->Wrap( -1 );
	InfoSizer->Add( MaxIterationsStaticText, 0, wxALIGN_LEFT|wxALL, 5 );

	m_staticText79 = new wxStaticText( JobDetailsPanel, wxID_ANY, wxT("b-factor :"), wxDefaultPosition, wxDefaultSize, 0 );
	m_staticText79->Wrap( -1 );
	m_staticText79->SetFont( wxFont( 10, wxFONTFAMILY_SWISS, wxFONTSTYLE_NORMAL, wxFONTWEIGHT_BOLD, false, wxT("Sans") ) );

	InfoSizer->Add( m_staticText79, 0, wxALIGN_RIGHT|wxALL, 5 );

	BfactorStaticText = new wxStaticText( JobDetailsPanel, wxID_ANY, wxEmptyString, wxDefaultPosition, wxDefaultSize, 0 );
	BfactorStaticText->Wrap( -1 );
	InfoSizer->Add( BfactorStaticText, 0, wxALIGN_LEFT|wxALL, 5 );

	m_staticText95 = new wxStaticText( JobDetailsPanel, wxID_ANY, wxT("Exp. Filter :"), wxDefaultPosition, wxDefaultSize, 0 );
	m_staticText95->Wrap( -1 );
	m_staticText95->SetFont( wxFont( 10, wxFONTFAMILY_SWISS, wxFONTSTYLE_NORMAL, wxFONTWEIGHT_BOLD, false, wxT("Sans") ) );

	InfoSizer->Add( m_staticText95, 0, wxALIGN_RIGHT|wxALL, 5 );

	ExposureFilterStaticText = new wxStaticText( JobDetailsPanel, wxID_ANY, wxEmptyString, wxDefaultPosition, wxDefaultSize, 0 );
	ExposureFilterStaticText->Wrap( -1 );
	InfoSizer->Add( ExposureFilterStaticText, 0, wxALIGN_LEFT|wxALL, 5 );

	m_staticText99 = new wxStaticText( JobDetailsPanel, wxID_ANY, wxT("Restore Power :"), wxDefaultPosition, wxDefaultSize, 0 );
	m_staticText99->Wrap( -1 );
	m_staticText99->SetFont( wxFont( 10, wxFONTFAMILY_SWISS, wxFONTSTYLE_NORMAL, wxFONTWEIGHT_BOLD, false, wxT("Sans") ) );

	InfoSizer->Add( m_staticText99, 0, wxALIGN_RIGHT|wxALL, 5 );

	RestorePowerStaticText = new wxStaticText( JobDetailsPanel, wxID_ANY, wxEmptyString, wxDefaultPosition, wxDefaultSize, 0 );
	RestorePowerStaticText->Wrap( -1 );
	InfoSizer->Add( RestorePowerStaticText, 0, wxALIGN_LEFT|wxALL, 5 );

	m_staticText101 = new wxStaticText( JobDetailsPanel, wxID_ANY, wxT("Mask Cross :"), wxDefaultPosition, wxDefaultSize, 0 );
	m_staticText101->Wrap( -1 );
	m_staticText101->SetFont( wxFont( 10, wxFONTFAMILY_SWISS, wxFONTSTYLE_NORMAL, wxFONTWEIGHT_BOLD, false, wxT("Sans") ) );

	InfoSizer->Add( m_staticText101, 0, wxALIGN_RIGHT|wxALL, 5 );

	MaskCrossStaticText = new wxStaticText( JobDetailsPanel, wxID_ANY, wxEmptyString, wxDefaultPosition, wxDefaultSize, 0 );
	MaskCrossStaticText->Wrap( -1 );
	InfoSizer->Add( MaskCrossStaticText, 0, wxALIGN_LEFT|wxALL, 5 );

	m_staticText103 = new wxStaticText( JobDetailsPanel, wxID_ANY, wxT("Horiz. Mask :"), wxDefaultPosition, wxDefaultSize, 0 );
	m_staticText103->Wrap( -1 );
	m_staticText103->SetFont( wxFont( 10, wxFONTFAMILY_SWISS, wxFONTSTYLE_NORMAL, wxFONTWEIGHT_BOLD, false, wxT("Sans") ) );

	InfoSizer->Add( m_staticText103, 0, wxALIGN_RIGHT|wxALL, 5 );

	HorizontalMaskStaticText = new wxStaticText( JobDetailsPanel, wxID_ANY, wxEmptyString, wxDefaultPosition, wxDefaultSize, 0 );
	HorizontalMaskStaticText->Wrap( -1 );
	InfoSizer->Add( HorizontalMaskStaticText, 0, wxALIGN_LEFT|wxALL, 5 );

	m_staticText105 = new wxStaticText( JobDetailsPanel, wxID_ANY, wxT("Vert. Mask :"), wxDefaultPosition, wxDefaultSize, 0 );
	m_staticText105->Wrap( -1 );
	m_staticText105->SetFont( wxFont( 10, wxFONTFAMILY_SWISS, wxFONTSTYLE_NORMAL, wxFONTWEIGHT_BOLD, false, wxT("Sans") ) );

	InfoSizer->Add( m_staticText105, 0, wxALIGN_RIGHT|wxALL, 5 );

	VerticalMaskStaticText = new wxStaticText( JobDetailsPanel, wxID_ANY, wxEmptyString, wxDefaultPosition, wxDefaultSize, 0 );
	VerticalMaskStaticText->Wrap( -1 );
	InfoSizer->Add( VerticalMaskStaticText, 0, wxALIGN_LEFT|wxALL, 5 );

	m_staticText1051 = new wxStaticText( JobDetailsPanel, wxID_ANY, wxT("Sum all Frames :"), wxDefaultPosition, wxDefaultSize, 0 );
	m_staticText1051->Wrap( -1 );
	m_staticText1051->SetFont( wxFont( 10, wxFONTFAMILY_SWISS, wxFONTSTYLE_NORMAL, wxFONTWEIGHT_BOLD, false, wxT("Sans") ) );

	InfoSizer->Add( m_staticText1051, 0, wxALIGN_RIGHT|wxALL, 5 );

	IncludeAllFramesStaticText = new wxStaticText( JobDetailsPanel, wxID_ANY, wxEmptyString, wxDefaultPosition, wxDefaultSize, 0 );
	IncludeAllFramesStaticText->Wrap( -1 );
	InfoSizer->Add( IncludeAllFramesStaticText, 0, wxALIGN_LEFT|wxALL, 5 );

	m_staticText1052 = new wxStaticText( JobDetailsPanel, wxID_ANY, wxT("First Frame :"), wxDefaultPosition, wxDefaultSize, 0 );
	m_staticText1052->Wrap( -1 );
	m_staticText1052->SetFont( wxFont( 10, wxFONTFAMILY_SWISS, wxFONTSTYLE_NORMAL, wxFONTWEIGHT_BOLD, false, wxT("Sans") ) );

	InfoSizer->Add( m_staticText1052, 0, wxALIGN_RIGHT|wxALL, 5 );

	FirstFrameStaticText = new wxStaticText( JobDetailsPanel, wxID_ANY, wxEmptyString, wxDefaultPosition, wxDefaultSize, 0 );
	FirstFrameStaticText->Wrap( -1 );
	InfoSizer->Add( FirstFrameStaticText, 0, wxALIGN_LEFT|wxALL, 5 );

	m_staticText1053 = new wxStaticText( JobDetailsPanel, wxID_ANY, wxT("Last Frame :"), wxDefaultPosition, wxDefaultSize, 0 );
	m_staticText1053->Wrap( -1 );
	m_staticText1053->SetFont( wxFont( 10, wxFONTFAMILY_SWISS, wxFONTSTYLE_NORMAL, wxFONTWEIGHT_BOLD, false, wxT("Sans") ) );

	InfoSizer->Add( m_staticText1053, 0, wxALIGN_RIGHT|wxALL|wxRIGHT, 5 );

	LastFrameStaticText = new wxStaticText( JobDetailsPanel, wxID_ANY, wxEmptyString, wxDefaultPosition, wxDefaultSize, 0 );
	LastFrameStaticText->Wrap( -1 );
	InfoSizer->Add( LastFrameStaticText, 0, wxALL|wxLEFT, 5 );


	JobDetailsPanel->SetSizer( InfoSizer );
	JobDetailsPanel->Layout();
	InfoSizer->Fit( JobDetailsPanel );
	bSizer73->Add( JobDetailsPanel, 1, wxALL|wxEXPAND, 5 );


	bSizer681->Add( bSizer73, 0, wxEXPAND, 5 );

	wxGridSizer* gSizer5;
	gSizer5 = new wxGridSizer( 0, 6, 0, 0 );


	bSizer681->Add( gSizer5, 0, wxEXPAND, 5 );

	ResultPanel = new UnblurResultsPanel( RightPanel, wxID_ANY, wxDefaultPosition, wxDefaultSize, wxTAB_TRAVERSAL );
	bSizer681->Add( ResultPanel, 1, wxEXPAND | wxALL, 5 );

	wxBoxSizer* bSizer69;
	bSizer69 = new wxBoxSizer( wxHORIZONTAL );

	DeleteFromGroupButton = new wxButton( RightPanel, wxID_ANY, wxT("Delete Movie From Group"), wxDefaultPosition, wxDefaultSize, 0 );
	bSizer69->Add( DeleteFromGroupButton, 0, wxALL, 5 );

	AddToGroupButton = new wxButton( RightPanel, wxID_ANY, wxT("Add Movie To Group"), wxDefaultPosition, wxDefaultSize, 0 );
	bSizer69->Add( AddToGroupButton, 0, wxALIGN_CENTER_VERTICAL|wxALL, 5 );

	GroupComboBox = new MemoryComboBox( RightPanel, wxID_ANY, wxT("Combo!"), wxDefaultPosition, wxDefaultSize, 0, NULL, wxCB_READONLY );
	GroupComboBox->SetMinSize( wxSize( 200,-1 ) );

	bSizer69->Add( GroupComboBox, 0, wxALIGN_CENTER_VERTICAL|wxALL, 5 );


	bSizer681->Add( bSizer69, 0, wxALIGN_RIGHT, 5 );


	RightPanel->SetSizer( bSizer681 );
	RightPanel->Layout();
	bSizer681->Fit( RightPanel );
	m_splitter4->SplitVertically( m_panel13, RightPanel, 450 );
	bSizer63->Add( m_splitter4, 1, wxEXPAND, 5 );


	this->SetSizer( bSizer63 );
	this->Layout();

	// Connect Events
	this->Connect( wxEVT_UPDATE_UI, wxUpdateUIEventHandler( MovieAlignResultsPanel::OnUpdateUI ) );
	AllMoviesButton->Connect( wxEVT_COMMAND_RADIOBUTTON_SELECTED, wxCommandEventHandler( MovieAlignResultsPanel::OnAllMoviesSelect ), NULL, this );
	ByFilterButton->Connect( wxEVT_COMMAND_RADIOBUTTON_SELECTED, wxCommandEventHandler( MovieAlignResultsPanel::OnByFilterSelect ), NULL, this );
	FilterButton->Connect( wxEVT_COMMAND_BUTTON_CLICKED, wxCommandEventHandler( MovieAlignResultsPanel::OnDefineFilterClick ), NULL, this );
	JobDetailsToggleButton->Connect( wxEVT_COMMAND_TOGGLEBUTTON_CLICKED, wxCommandEventHandler( MovieAlignResultsPanel::OnJobDetailsToggle ), NULL, this );
	PreviousButton->Connect( wxEVT_COMMAND_BUTTON_CLICKED, wxCommandEventHandler( MovieAlignResultsPanel::OnPreviousButtonClick ), NULL, this );
	AddAllToGroupButton->Connect( wxEVT_COMMAND_BUTTON_CLICKED, wxCommandEventHandler( MovieAlignResultsPanel::OnAddAllToGroupClick ), NULL, this );
	NextButton->Connect( wxEVT_COMMAND_BUTTON_CLICKED, wxCommandEventHandler( MovieAlignResultsPanel::OnNextButtonClick ), NULL, this );
	DeleteFromGroupButton->Connect( wxEVT_COMMAND_BUTTON_CLICKED, wxCommandEventHandler( MovieAlignResultsPanel::OnRemoveFromGroupClick ), NULL, this );
	AddToGroupButton->Connect( wxEVT_COMMAND_BUTTON_CLICKED, wxCommandEventHandler( MovieAlignResultsPanel::OnAddToGroupClick ), NULL, this );
}

MovieAlignResultsPanel::~MovieAlignResultsPanel()
{
	// Disconnect Events
	this->Disconnect( wxEVT_UPDATE_UI, wxUpdateUIEventHandler( MovieAlignResultsPanel::OnUpdateUI ) );
	AllMoviesButton->Disconnect( wxEVT_COMMAND_RADIOBUTTON_SELECTED, wxCommandEventHandler( MovieAlignResultsPanel::OnAllMoviesSelect ), NULL, this );
	ByFilterButton->Disconnect( wxEVT_COMMAND_RADIOBUTTON_SELECTED, wxCommandEventHandler( MovieAlignResultsPanel::OnByFilterSelect ), NULL, this );
	FilterButton->Disconnect( wxEVT_COMMAND_BUTTON_CLICKED, wxCommandEventHandler( MovieAlignResultsPanel::OnDefineFilterClick ), NULL, this );
	JobDetailsToggleButton->Disconnect( wxEVT_COMMAND_TOGGLEBUTTON_CLICKED, wxCommandEventHandler( MovieAlignResultsPanel::OnJobDetailsToggle ), NULL, this );
	PreviousButton->Disconnect( wxEVT_COMMAND_BUTTON_CLICKED, wxCommandEventHandler( MovieAlignResultsPanel::OnPreviousButtonClick ), NULL, this );
	AddAllToGroupButton->Disconnect( wxEVT_COMMAND_BUTTON_CLICKED, wxCommandEventHandler( MovieAlignResultsPanel::OnAddAllToGroupClick ), NULL, this );
	NextButton->Disconnect( wxEVT_COMMAND_BUTTON_CLICKED, wxCommandEventHandler( MovieAlignResultsPanel::OnNextButtonClick ), NULL, this );
	DeleteFromGroupButton->Disconnect( wxEVT_COMMAND_BUTTON_CLICKED, wxCommandEventHandler( MovieAlignResultsPanel::OnRemoveFromGroupClick ), NULL, this );
	AddToGroupButton->Disconnect( wxEVT_COMMAND_BUTTON_CLICKED, wxCommandEventHandler( MovieAlignResultsPanel::OnAddToGroupClick ), NULL, this );

}

UnblurResultsPanelParent::UnblurResultsPanelParent( wxWindow* parent, wxWindowID id, const wxPoint& pos, const wxSize& size, long style, const wxString& name ) : wxPanel( parent, id, pos, size, style, name )
{
	MainSizer = new wxBoxSizer( wxHORIZONTAL );

	m_splitter13 = new wxSplitterWindow( this, wxID_ANY, wxDefaultPosition, wxDefaultSize, wxSP_3D );
	m_splitter13->SetSashGravity( 0.5 );
	m_splitter13->Connect( wxEVT_IDLE, wxIdleEventHandler( UnblurResultsPanelParent::m_splitter13OnIdle ), NULL, this );

	m_panel80 = new wxPanel( m_splitter13, wxID_ANY, wxDefaultPosition, wxDefaultSize, wxTAB_TRAVERSAL );
	SplitSizer = new wxBoxSizer( wxVERTICAL );

	m_splitter14 = new wxSplitterWindow( m_panel80, wxID_ANY, wxDefaultPosition, wxDefaultSize, wxSP_3D );
	m_splitter14->SetSashGravity( 0.5 );
	m_splitter14->Connect( wxEVT_IDLE, wxIdleEventHandler( UnblurResultsPanelParent::m_splitter14OnIdle ), NULL, this );

	m_panel82 = new wxPanel( m_splitter14, wxID_ANY, wxDefaultPosition, wxDefaultSize, wxTAB_TRAVERSAL );
	wxBoxSizer* bSizer287;
	bSizer287 = new wxBoxSizer( wxVERTICAL );

	wxBoxSizer* bSizer299;
	bSizer299 = new wxBoxSizer( wxHORIZONTAL );

	m_staticText372 = new wxStaticText( m_panel82, wxID_ANY, wxT("Aligned Sum Spectra (Nyquist :"), wxDefaultPosition, wxDefaultSize, 0 );
	m_staticText372->Wrap( -1 );
	m_staticText372->SetFont( wxFont( 10, wxFONTFAMILY_SWISS, wxFONTSTYLE_NORMAL, wxFONTWEIGHT_BOLD, false, wxT("Sans") ) );

	bSizer299->Add( m_staticText372, 0, wxALL, 5 );

	SpectraNyquistStaticText = new wxStaticText( m_panel82, wxID_ANY, wxT("2.8 Å)"), wxDefaultPosition, wxDefaultSize, 0 );
	SpectraNyquistStaticText->Wrap( -1 );
	SpectraNyquistStaticText->SetFont( wxFont( 10, wxFONTFAMILY_SWISS, wxFONTSTYLE_NORMAL, wxFONTWEIGHT_BOLD, false, wxT("Sans") ) );

	bSizer299->Add( SpectraNyquistStaticText, 0, wxALL, 5 );


	bSizer287->Add( bSizer299, 1, wxEXPAND, 5 );

	m_staticline73 = new wxStaticLine( m_panel82, wxID_ANY, wxDefaultPosition, wxDefaultSize, wxLI_HORIZONTAL );
	bSizer287->Add( m_staticline73, 0, wxEXPAND | wxALL, 5 );

	SpectraPanel = new BitmapPanel( m_panel82, wxID_ANY, wxDefaultPosition, wxDefaultSize, wxTAB_TRAVERSAL );
	bSizer287->Add( SpectraPanel, 50, wxEXPAND | wxALL, 5 );


	m_panel82->SetSizer( bSizer287 );
	m_panel82->Layout();
	bSizer287->Fit( m_panel82 );
	m_panel83 = new wxPanel( m_splitter14, wxID_ANY, wxDefaultPosition, wxDefaultSize, wxTAB_TRAVERSAL );
	wxBoxSizer* bSizer288;
	bSizer288 = new wxBoxSizer( wxVERTICAL );

	m_staticText373 = new wxStaticText( m_panel83, wxID_ANY, wxT("Plot of Shifts"), wxDefaultPosition, wxDefaultSize, 0 );
	m_staticText373->Wrap( -1 );
	m_staticText373->SetFont( wxFont( 10, wxFONTFAMILY_SWISS, wxFONTSTYLE_NORMAL, wxFONTWEIGHT_BOLD, false, wxT("Sans") ) );

	bSizer288->Add( m_staticText373, 0, wxALIGN_BOTTOM|wxALL, 5 );

	m_staticline74 = new wxStaticLine( m_panel83, wxID_ANY, wxDefaultPosition, wxDefaultSize, wxLI_HORIZONTAL );
	bSizer288->Add( m_staticline74, 0, wxEXPAND | wxALL, 5 );

	PlotPanel = new wxPanel( m_panel83, wxID_ANY, wxDefaultPosition, wxDefaultSize, wxTAB_TRAVERSAL );
	GraphSizer = new wxBoxSizer( wxVERTICAL );


	PlotPanel->SetSizer( GraphSizer );
	PlotPanel->Layout();
	GraphSizer->Fit( PlotPanel );
	bSizer288->Add( PlotPanel, 50, wxEXPAND | wxALL, 5 );


	m_panel83->SetSizer( bSizer288 );
	m_panel83->Layout();
	bSizer288->Fit( m_panel83 );
	m_splitter14->SplitHorizontally( m_panel82, m_panel83, 0 );
	SplitSizer->Add( m_splitter14, 1, wxEXPAND, 5 );


	m_panel80->SetSizer( SplitSizer );
	m_panel80->Layout();
	SplitSizer->Fit( m_panel80 );
	m_panel81 = new wxPanel( m_splitter13, wxID_ANY, wxDefaultPosition, wxDefaultSize, wxTAB_TRAVERSAL );
	wxBoxSizer* bSizer285;
	bSizer285 = new wxBoxSizer( wxVERTICAL );

	wxBoxSizer* bSizer300;
	bSizer300 = new wxBoxSizer( wxHORIZONTAL );

	m_staticText371 = new wxStaticText( m_panel81, wxID_ANY, wxT("Aligned Movie Sum"), wxDefaultPosition, wxDefaultSize, 0 );
	m_staticText371->Wrap( -1 );
	m_staticText371->SetFont( wxFont( 10, wxFONTFAMILY_SWISS, wxFONTSTYLE_NORMAL, wxFONTWEIGHT_BOLD, false, wxT("Sans") ) );

	bSizer300->Add( m_staticText371, 0, wxALL, 5 );

	FilenameStaticText = new wxStaticText( m_panel81, wxID_ANY, wxEmptyString, wxDefaultPosition, wxDefaultSize, 0 );
	FilenameStaticText->Wrap( -1 );
	FilenameStaticText->SetFont( wxFont( 10, wxFONTFAMILY_SWISS, wxFONTSTYLE_NORMAL, wxFONTWEIGHT_BOLD, false, wxT("Sans") ) );

	bSizer300->Add( FilenameStaticText, 0, wxALL, 5 );


	bSizer285->Add( bSizer300, 1, wxEXPAND, 5 );

	m_staticline72 = new wxStaticLine( m_panel81, wxID_ANY, wxDefaultPosition, wxDefaultSize, wxLI_HORIZONTAL );
	bSizer285->Add( m_staticline72, 0, wxEXPAND | wxALL, 5 );

	ImageDisplayPanel = new DisplayPanel( m_panel81, wxID_ANY, wxDefaultPosition, wxDefaultSize, wxTAB_TRAVERSAL );
	bSizer285->Add( ImageDisplayPanel, 60, wxEXPAND | wxALL, 5 );


	m_panel81->SetSizer( bSizer285 );
	m_panel81->Layout();
	bSizer285->Fit( m_panel81 );
	m_splitter13->SplitVertically( m_panel80, m_panel81, 542 );
	MainSizer->Add( m_splitter13, 1, wxEXPAND, 5 );


	this->SetSizer( MainSizer );
	this->Layout();
}

UnblurResultsPanelParent::~UnblurResultsPanelParent()
{
}

AlignMoviesPanel::AlignMoviesPanel( wxWindow* parent, wxWindowID id, const wxPoint& pos, const wxSize& size, long style, const wxString& name ) : JobPanel( parent, id, pos, size, style, name )
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

	GroupComboBox = new MovieGroupPickerComboPanel( this, wxID_ANY, wxDefaultPosition, wxDefaultSize, wxTAB_TRAVERSAL );
	GroupComboBox->SetMinSize( wxSize( 350,-1 ) );
	GroupComboBox->SetMaxSize( wxSize( 350,-1 ) );

	bSizer44->Add( GroupComboBox, 0, wxALIGN_CENTER_VERTICAL|wxALL, 5 );


	bSizer44->Add( 0, 0, 60, wxEXPAND, 5 );


	bSizer45->Add( bSizer44, 1, wxEXPAND, 5 );

	ExpertToggleButton = new wxToggleButton( this, wxID_ANY, wxT("Show Expert Options"), wxDefaultPosition, wxDefaultSize, 0 );
	bSizer45->Add( ExpertToggleButton, 0, wxALL, 5 );


	bSizer43->Add( bSizer45, 0, wxEXPAND, 5 );

	m_staticline10 = new wxStaticLine( this, wxID_ANY, wxDefaultPosition, wxDefaultSize, wxLI_HORIZONTAL );
	bSizer43->Add( m_staticline10, 0, wxEXPAND | wxALL, 5 );

	wxBoxSizer* bSizer46;
	bSizer46 = new wxBoxSizer( wxHORIZONTAL );

	ExpertPanel = new wxScrolledWindow( this, wxID_ANY, wxDefaultPosition, wxSize( -1,-1 ), wxVSCROLL );
	ExpertPanel->SetScrollRate( 5, 5 );
	ExpertPanel->Hide();

	InputSizer = new wxBoxSizer( wxVERTICAL );

	wxFlexGridSizer* fgSizer1;
	fgSizer1 = new wxFlexGridSizer( 0, 2, 0, 0 );
	fgSizer1->SetFlexibleDirection( wxBOTH );
	fgSizer1->SetNonFlexibleGrowMode( wxFLEX_GROWMODE_SPECIFIED );

	m_staticText43 = new wxStaticText( ExpertPanel, wxID_ANY, wxT("Shifts"), wxDefaultPosition, wxDefaultSize, 0 );
	m_staticText43->Wrap( -1 );
	m_staticText43->SetFont( wxFont( wxNORMAL_FONT->GetPointSize(), wxFONTFAMILY_DEFAULT, wxFONTSTYLE_NORMAL, wxFONTWEIGHT_BOLD, true, wxEmptyString ) );

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

	maximum_shift_text = new wxTextCtrl( ExpertPanel, wxID_ANY, wxT("40"), wxDefaultPosition, wxDefaultSize, 0 );
	maximum_shift_text->SetToolTip( wxT("Maximum shift for each alignment round") );

	fgSizer1->Add( maximum_shift_text, 0, wxALL|wxEXPAND, 5 );

	m_staticText44 = new wxStaticText( ExpertPanel, wxID_ANY, wxT("Exposure Filter"), wxDefaultPosition, wxDefaultSize, 0 );
	m_staticText44->Wrap( -1 );
	m_staticText44->SetFont( wxFont( 10, wxFONTFAMILY_SWISS, wxFONTSTYLE_NORMAL, wxFONTWEIGHT_BOLD, true, wxT("Sans") ) );

	fgSizer1->Add( m_staticText44, 0, wxALL, 5 );


	fgSizer1->Add( 0, 0, 1, wxEXPAND, 5 );

	dose_filter_checkbox = new wxCheckBox( ExpertPanel, wxID_ANY, wxT("Exposure Filter Sums?"), wxDefaultPosition, wxDefaultSize, 0 );
	dose_filter_checkbox->SetValue(true);
	dose_filter_checkbox->SetToolTip( wxT("Make a dose weighted sum") );

	fgSizer1->Add( dose_filter_checkbox, 1, wxALIGN_LEFT|wxALL, 5 );

	restore_power_checkbox = new wxCheckBox( ExpertPanel, wxID_ANY, wxT("Restore Power?"), wxDefaultPosition, wxDefaultSize, 0 );
	restore_power_checkbox->SetValue(true);
	fgSizer1->Add( restore_power_checkbox, 1, wxALIGN_RIGHT|wxALL, 5 );

	m_staticText45 = new wxStaticText( ExpertPanel, wxID_ANY, wxT("Convergence"), wxDefaultPosition, wxDefaultSize, 0 );
	m_staticText45->Wrap( -1 );
	m_staticText45->SetFont( wxFont( wxNORMAL_FONT->GetPointSize(), wxFONTFAMILY_DEFAULT, wxFONTSTYLE_NORMAL, wxFONTWEIGHT_BOLD, true, wxEmptyString ) );

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

	max_iterations_spinctrl = new wxSpinCtrl( ExpertPanel, wxID_ANY, wxEmptyString, wxDefaultPosition, wxDefaultSize, wxSP_ARROW_KEYS, 1, 50, 10 );
	fgSizer1->Add( max_iterations_spinctrl, 0, wxALL|wxEXPAND, 5 );

	m_staticText48 = new wxStaticText( ExpertPanel, wxID_ANY, wxT("Filter"), wxDefaultPosition, wxDefaultSize, 0 );
	m_staticText48->Wrap( -1 );
	m_staticText48->SetFont( wxFont( wxNORMAL_FONT->GetPointSize(), wxFONTFAMILY_DEFAULT, wxFONTSTYLE_NORMAL, wxFONTWEIGHT_BOLD, true, wxEmptyString ) );

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

	horizontal_mask_static_text = new wxStaticText( ExpertPanel, wxID_ANY, wxT("\tHoriz. Mask (Pixels) :"), wxDefaultPosition, wxDefaultSize, 0 );
	horizontal_mask_static_text->Wrap( -1 );
	fgSizer1->Add( horizontal_mask_static_text, 0, wxALL, 5 );

	horizontal_mask_spinctrl = new wxSpinCtrl( ExpertPanel, wxID_ANY, wxEmptyString, wxDefaultPosition, wxDefaultSize, wxSP_ARROW_KEYS, 0, 10, 1 );
	fgSizer1->Add( horizontal_mask_spinctrl, 0, wxALL|wxEXPAND, 5 );

	vertical_mask_static_text = new wxStaticText( ExpertPanel, wxID_ANY, wxT("\tVert. Mask (Pixels) :"), wxDefaultPosition, wxDefaultSize, 0 );
	vertical_mask_static_text->Wrap( -1 );
	fgSizer1->Add( vertical_mask_static_text, 0, wxALL, 5 );

	vertical_mask_spinctrl = new wxSpinCtrl( ExpertPanel, wxID_ANY, wxEmptyString, wxDefaultPosition, wxDefaultSize, wxSP_ARROW_KEYS, 0, 10, 1 );
	fgSizer1->Add( vertical_mask_spinctrl, 0, wxALL|wxEXPAND, 5 );

	m_staticText481 = new wxStaticText( ExpertPanel, wxID_ANY, wxT("Final Sum"), wxDefaultPosition, wxDefaultSize, 0 );
	m_staticText481->Wrap( -1 );
	m_staticText481->SetFont( wxFont( wxNORMAL_FONT->GetPointSize(), wxFONTFAMILY_DEFAULT, wxFONTSTYLE_NORMAL, wxFONTWEIGHT_BOLD, true, wxEmptyString ) );

	fgSizer1->Add( m_staticText481, 0, wxALL, 5 );


	fgSizer1->Add( 0, 0, 1, wxEXPAND, 5 );

	include_all_frames_checkbox = new wxCheckBox( ExpertPanel, wxID_ANY, wxT("Include All Frames in Sum?"), wxDefaultPosition, wxDefaultSize, 0 );
	include_all_frames_checkbox->SetValue(true);
	fgSizer1->Add( include_all_frames_checkbox, 0, wxALIGN_CENTER_VERTICAL|wxALL, 5 );


	fgSizer1->Add( 0, 0, 1, wxEXPAND, 5 );

	first_frame_static_text = new wxStaticText( ExpertPanel, wxID_ANY, wxT("\tFirst Frame to Sum :"), wxDefaultPosition, wxDefaultSize, 0 );
	first_frame_static_text->Wrap( -1 );
	fgSizer1->Add( first_frame_static_text, 0, wxALL, 5 );

	first_frame_spin_ctrl = new wxSpinCtrl( ExpertPanel, wxID_ANY, wxEmptyString, wxDefaultPosition, wxDefaultSize, wxSP_ARROW_KEYS, 1, 10000, 1 );
	fgSizer1->Add( first_frame_spin_ctrl, 0, wxALIGN_CENTER_VERTICAL|wxALL, 5 );

	last_frame_static_text = new wxStaticText( ExpertPanel, wxID_ANY, wxT("\tLast Frame to Sum :"), wxDefaultPosition, wxDefaultSize, 0 );
	last_frame_static_text->Wrap( -1 );
	fgSizer1->Add( last_frame_static_text, 0, wxALIGN_CENTER_VERTICAL|wxALL, 5 );

	last_frame_spin_ctrl = new wxSpinCtrl( ExpertPanel, wxID_ANY, wxEmptyString, wxDefaultPosition, wxDefaultSize, wxSP_ARROW_KEYS, 0, 10000, 0 );
	fgSizer1->Add( last_frame_spin_ctrl, 0, wxALIGN_CENTER_VERTICAL|wxALL, 5 );

	SaveScaledSumCheckbox = new wxCheckBox( ExpertPanel, wxID_ANY, wxT("Also Save Scaled Sum?"), wxDefaultPosition, wxDefaultSize, 0 );
	SaveScaledSumCheckbox->SetValue(true);
	fgSizer1->Add( SaveScaledSumCheckbox, 0, wxALIGN_CENTER_VERTICAL|wxALL, 5 );


	fgSizer1->Add( 0, 0, 1, wxEXPAND, 5 );


	InputSizer->Add( fgSizer1, 0, wxEXPAND, 5 );


	ExpertPanel->SetSizer( InputSizer );
	ExpertPanel->Layout();
	InputSizer->Fit( ExpertPanel );
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
	bSizer46->Add( OutputTextPanel, 20, wxEXPAND | wxALL, 5 );

	GraphPanel = new UnblurResultsPanel( this, wxID_ANY, wxDefaultPosition, wxDefaultSize, wxTAB_TRAVERSAL );
	GraphPanel->Hide();

	bSizer46->Add( GraphPanel, 80, wxEXPAND | wxALL, 5 );

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

	ProgressPanel = new wxPanel( this, wxID_ANY, wxDefaultPosition, wxDefaultSize, wxTAB_TRAVERSAL );
	ProgressPanel->Hide();

	wxBoxSizer* bSizer57;
	bSizer57 = new wxBoxSizer( wxHORIZONTAL );

	wxBoxSizer* bSizer59;
	bSizer59 = new wxBoxSizer( wxHORIZONTAL );

	NumberConnectedText = new wxStaticText( ProgressPanel, wxID_ANY, wxT("o"), wxDefaultPosition, wxDefaultSize, 0 );
	NumberConnectedText->Wrap( -1 );
	bSizer59->Add( NumberConnectedText, 0, wxALIGN_CENTER|wxALIGN_CENTER_VERTICAL|wxALL, 5 );

	ProgressBar = new wxGauge( ProgressPanel, wxID_ANY, 100, wxDefaultPosition, wxDefaultSize, wxGA_HORIZONTAL );
	ProgressBar->SetValue( 0 );
	bSizer59->Add( ProgressBar, 100, wxALIGN_CENTER|wxALIGN_CENTER_HORIZONTAL|wxALIGN_CENTER_VERTICAL|wxALL|wxEXPAND, 5 );

	TimeRemainingText = new wxStaticText( ProgressPanel, wxID_ANY, wxT("Time Remaining : ???h:??m:??s   "), wxDefaultPosition, wxDefaultSize, wxALIGN_CENTER_HORIZONTAL );
	TimeRemainingText->Wrap( -1 );
	bSizer59->Add( TimeRemainingText, 0, wxALIGN_CENTER_VERTICAL|wxALIGN_RIGHT|wxALL, 5 );

	m_staticline60 = new wxStaticLine( ProgressPanel, wxID_ANY, wxDefaultPosition, wxDefaultSize, wxLI_HORIZONTAL|wxLI_VERTICAL );
	bSizer59->Add( m_staticline60, 0, wxEXPAND | wxALL, 5 );

	FinishButton = new wxButton( ProgressPanel, wxID_ANY, wxT("Finish"), wxDefaultPosition, wxDefaultSize, 0 );
	FinishButton->Hide();

	bSizer59->Add( FinishButton, 0, wxALIGN_CENTER_HORIZONTAL|wxALIGN_CENTER_VERTICAL|wxALL, 5 );

	CancelAlignmentButton = new wxButton( ProgressPanel, wxID_ANY, wxT("Terminate Job"), wxDefaultPosition, wxDefaultSize, 0 );
	bSizer59->Add( CancelAlignmentButton, 0, wxALIGN_CENTER_HORIZONTAL|wxALIGN_CENTER_VERTICAL|wxALL, 5 );


	bSizer57->Add( bSizer59, 1, wxEXPAND, 5 );


	ProgressPanel->SetSizer( bSizer57 );
	ProgressPanel->Layout();
	bSizer57->Fit( ProgressPanel );
	bSizer70->Add( ProgressPanel, 1, wxEXPAND | wxALL, 5 );


	bSizer48->Add( bSizer70, 1, wxEXPAND, 5 );

	StartPanel = new wxPanel( this, wxID_ANY, wxDefaultPosition, wxDefaultSize, wxTAB_TRAVERSAL );
	wxBoxSizer* bSizer58;
	bSizer58 = new wxBoxSizer( wxHORIZONTAL );

	RunProfileText = new wxStaticText( StartPanel, wxID_ANY, wxT("Run Profile :"), wxDefaultPosition, wxDefaultSize, 0 );
	RunProfileText->Wrap( -1 );
	bSizer58->Add( RunProfileText, 0, wxALIGN_CENTER_VERTICAL|wxALL, 5 );

	RunProfileComboBox = new MemoryComboBox( StartPanel, wxID_ANY, wxEmptyString, wxDefaultPosition, wxDefaultSize, 0, NULL, wxCB_READONLY );
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
	FinishButton->Connect( wxEVT_COMMAND_BUTTON_CLICKED, wxCommandEventHandler( AlignMoviesPanel::FinishButtonClick ), NULL, this );
	CancelAlignmentButton->Connect( wxEVT_COMMAND_BUTTON_CLICKED, wxCommandEventHandler( AlignMoviesPanel::TerminateButtonClick ), NULL, this );
	StartAlignmentButton->Connect( wxEVT_COMMAND_BUTTON_CLICKED, wxCommandEventHandler( AlignMoviesPanel::StartAlignmentClick ), NULL, this );
}

AlignMoviesPanel::~AlignMoviesPanel()
{
	// Disconnect Events
	this->Disconnect( wxEVT_UPDATE_UI, wxUpdateUIEventHandler( AlignMoviesPanel::OnUpdateUI ) );
	ExpertToggleButton->Disconnect( wxEVT_COMMAND_TOGGLEBUTTON_CLICKED, wxCommandEventHandler( AlignMoviesPanel::OnExpertOptionsToggle ), NULL, this );
	InfoText->Disconnect( wxEVT_COMMAND_TEXT_URL, wxTextUrlEventHandler( AlignMoviesPanel::OnInfoURL ), NULL, this );
	FinishButton->Disconnect( wxEVT_COMMAND_BUTTON_CLICKED, wxCommandEventHandler( AlignMoviesPanel::FinishButtonClick ), NULL, this );
	CancelAlignmentButton->Disconnect( wxEVT_COMMAND_BUTTON_CLICKED, wxCommandEventHandler( AlignMoviesPanel::TerminateButtonClick ), NULL, this );
	StartAlignmentButton->Disconnect( wxEVT_COMMAND_BUTTON_CLICKED, wxCommandEventHandler( AlignMoviesPanel::StartAlignmentClick ), NULL, this );

}
