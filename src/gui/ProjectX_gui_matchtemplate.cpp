///////////////////////////////////////////////////////////////////////////
// C++ code generated with wxFormBuilder (version 3.10.0-4761b0c)
// http://www.wxformbuilder.org/
//
// PLEASE DO *NOT* EDIT THIS FILE!
///////////////////////////////////////////////////////////////////////////

#include "../core/gui_core_headers.h"

#include "AssetPickerComboPanel.h"
#include "DisplayPanel.h"
#include "PlotCurvePanel.h"
#include "ResultsDataViewListCtrl.h"
#include "ShowTemplateMatchResultsPanel.h"
#include "my_controls.h"

#include "ProjectX_gui_matchtemplate.h"

// clang-format off
///////////////////////////////////////////////////////////////////////////

ShowTemplateMatchResultsPanelParent::ShowTemplateMatchResultsPanelParent( wxWindow* parent, wxWindowID id, const wxPoint& pos, const wxSize& size, long style, const wxString& name ) : wxPanel( parent, id, pos, size, style, name )
{
	wxBoxSizer* bSizer92;
	bSizer92 = new wxBoxSizer( wxVERTICAL );

	m_splitter16 = new wxSplitterWindow( this, wxID_ANY, wxDefaultPosition, wxDefaultSize, wxSP_3D );
	m_splitter16->SetSashGravity( 0.5 );
	m_splitter16->Connect( wxEVT_IDLE, wxIdleEventHandler( ShowTemplateMatchResultsPanelParent::m_splitter16OnIdle ), NULL, this );

	m_panel87 = new wxPanel( m_splitter16, wxID_ANY, wxDefaultPosition, wxDefaultSize, wxTAB_TRAVERSAL );
	wxBoxSizer* bSizer301;
	bSizer301 = new wxBoxSizer( wxVERTICAL );

	m_splitter15 = new wxSplitterWindow( m_panel87, wxID_ANY, wxDefaultPosition, wxDefaultSize, wxSP_3D );
	m_splitter15->SetSashGravity( 0.5 );
	m_splitter15->Connect( wxEVT_IDLE, wxIdleEventHandler( ShowTemplateMatchResultsPanelParent::m_splitter15OnIdle ), NULL, this );

	m_panel89 = new wxPanel( m_splitter15, wxID_ANY, wxDefaultPosition, wxDefaultSize, wxTAB_TRAVERSAL );
	wxBoxSizer* bSizer303;
	bSizer303 = new wxBoxSizer( wxVERTICAL );

	wxBoxSizer* bSizer538;
	bSizer538 = new wxBoxSizer( wxHORIZONTAL );

	PeakTableStaticText = new wxStaticText( m_panel89, wxID_ANY, wxT("Table of Peaks"), wxDefaultPosition, wxDefaultSize, 0 );
	PeakTableStaticText->Wrap( -1 );
	PeakTableStaticText->SetFont( wxFont( 10, wxFONTFAMILY_SWISS, wxFONTSTYLE_NORMAL, wxFONTWEIGHT_BOLD, false, wxT("Sans") ) );

	bSizer538->Add( PeakTableStaticText, 0, wxALIGN_BOTTOM|wxALL, 5 );


	bSizer538->Add( 0, 0, 1, wxEXPAND, 5 );

	m_staticline148 = new wxStaticLine( m_panel89, wxID_ANY, wxDefaultPosition, wxDefaultSize, wxLI_HORIZONTAL|wxLI_VERTICAL );
	bSizer538->Add( m_staticline148, 0, wxEXPAND | wxALL, 5 );

	SaveButton = new NoFocusBitmapButton( m_panel89, wxID_ANY, wxNullBitmap, wxDefaultPosition, wxDefaultSize, wxBU_AUTODRAW|0 );

	SaveButton->SetDefault();
	bSizer538->Add( SaveButton, 0, wxALL, 5 );


	bSizer303->Add( bSizer538, 0, wxEXPAND, 5 );

	m_staticline82 = new wxStaticLine( m_panel89, wxID_ANY, wxDefaultPosition, wxDefaultSize, wxLI_HORIZONTAL );
	bSizer303->Add( m_staticline82, 0, wxEXPAND | wxALL, 5 );

	PeakListCtrl = new wxListCtrl( m_panel89, wxID_ANY, wxDefaultPosition, wxDefaultSize, wxLC_REPORT|wxLC_SINGLE_SEL );
	bSizer303->Add( PeakListCtrl, 1, wxALL|wxEXPAND, 5 );


	m_panel89->SetSizer( bSizer303 );
	m_panel89->Layout();
	bSizer303->Fit( m_panel89 );
	BottomPanel = new wxPanel( m_splitter15, wxID_ANY, wxDefaultPosition, wxDefaultSize, wxTAB_TRAVERSAL );
	wxBoxSizer* bSizer302;
	bSizer302 = new wxBoxSizer( wxVERTICAL );

	SurvivalHistogramText = new wxStaticText( BottomPanel, wxID_ANY, wxT("Survival Histogram"), wxDefaultPosition, wxDefaultSize, 0 );
	SurvivalHistogramText->Wrap( -1 );
	SurvivalHistogramText->SetFont( wxFont( 10, wxFONTFAMILY_SWISS, wxFONTSTYLE_NORMAL, wxFONTWEIGHT_BOLD, false, wxT("Sans") ) );

	bSizer302->Add( SurvivalHistogramText, 0, wxALL, 5 );

	m_staticline81 = new wxStaticLine( BottomPanel, wxID_ANY, wxDefaultPosition, wxDefaultSize, wxLI_HORIZONTAL );
	bSizer302->Add( m_staticline81, 0, wxEXPAND | wxALL, 5 );

	HistogramPlotPanel = new PlotCurvePanel( BottomPanel, wxID_ANY, wxDefaultPosition, wxDefaultSize, wxTAB_TRAVERSAL );
	bSizer302->Add( HistogramPlotPanel, 1, wxEXPAND | wxALL, 5 );

	PeakChangesPanel = new wxPanel( BottomPanel, wxID_ANY, wxDefaultPosition, wxDefaultSize, wxTAB_TRAVERSAL );
	wxBoxSizer* bSizer594;
	bSizer594 = new wxBoxSizer( wxVERTICAL );

	ChangesListCtrl = new wxListCtrl( PeakChangesPanel, wxID_ANY, wxDefaultPosition, wxDefaultSize, wxLC_REPORT|wxLC_SINGLE_SEL );
	bSizer594->Add( ChangesListCtrl, 1, wxALL|wxEXPAND, 5 );


	PeakChangesPanel->SetSizer( bSizer594 );
	PeakChangesPanel->Layout();
	bSizer594->Fit( PeakChangesPanel );
	bSizer302->Add( PeakChangesPanel, 1, wxEXPAND | wxALL, 5 );


	BottomPanel->SetSizer( bSizer302 );
	BottomPanel->Layout();
	bSizer302->Fit( BottomPanel );
	m_splitter15->SplitHorizontally( m_panel89, BottomPanel, 0 );
	bSizer301->Add( m_splitter15, 1, wxEXPAND, 5 );


	m_panel87->SetSizer( bSizer301 );
	m_panel87->Layout();
	bSizer301->Fit( m_panel87 );
	m_panel86 = new wxPanel( m_splitter16, wxID_ANY, wxDefaultPosition, wxDefaultSize, wxTAB_TRAVERSAL );
	wxBoxSizer* bSizer304;
	bSizer304 = new wxBoxSizer( wxVERTICAL );

	wxBoxSizer* bSizer305;
	bSizer305 = new wxBoxSizer( wxHORIZONTAL );


	bSizer304->Add( bSizer305, 0, wxEXPAND, 5 );

	wxBoxSizer* bSizer306;
	bSizer306 = new wxBoxSizer( wxHORIZONTAL );

	m_staticText394 = new wxStaticText( m_panel86, wxID_ANY, wxT("Image / MIP / Found Templates"), wxDefaultPosition, wxDefaultSize, 0 );
	m_staticText394->Wrap( -1 );
	m_staticText394->SetFont( wxFont( 10, wxFONTFAMILY_SWISS, wxFONTSTYLE_NORMAL, wxFONTWEIGHT_BOLD, false, wxT("Sans") ) );

	bSizer306->Add( m_staticText394, 0, wxALL, 5 );

	ImageFileText = new wxStaticText( m_panel86, wxID_ANY, wxEmptyString, wxDefaultPosition, wxDefaultSize, 0 );
	ImageFileText->Wrap( -1 );
	ImageFileText->SetFont( wxFont( 10, wxFONTFAMILY_SWISS, wxFONTSTYLE_NORMAL, wxFONTWEIGHT_BOLD, false, wxT("Sans") ) );

	bSizer306->Add( ImageFileText, 0, wxALL, 5 );


	bSizer304->Add( bSizer306, 0, wxEXPAND, 5 );

	m_staticline86 = new wxStaticLine( m_panel86, wxID_ANY, wxDefaultPosition, wxDefaultSize, wxLI_HORIZONTAL );
	bSizer304->Add( m_staticline86, 0, wxEXPAND | wxALL, 5 );

	ImageDisplayPanel = new DisplayPanel( m_panel86, wxID_ANY, wxDefaultPosition, wxDefaultSize, wxTAB_TRAVERSAL );
	bSizer304->Add( ImageDisplayPanel, 1, wxEXPAND | wxALL, 5 );


	m_panel86->SetSizer( bSizer304 );
	m_panel86->Layout();
	bSizer304->Fit( m_panel86 );
	m_splitter16->SplitVertically( m_panel87, m_panel86, 700 );
	bSizer92->Add( m_splitter16, 1, wxEXPAND, 5 );


	this->SetSizer( bSizer92 );
	this->Layout();

	// Connect Events
	SaveButton->Connect( wxEVT_COMMAND_BUTTON_CLICKED, wxCommandEventHandler( ShowTemplateMatchResultsPanelParent::OnSavePeaksClick ), NULL, this );
}

ShowTemplateMatchResultsPanelParent::~ShowTemplateMatchResultsPanelParent()
{
	// Disconnect Events
	SaveButton->Disconnect( wxEVT_COMMAND_BUTTON_CLICKED, wxCommandEventHandler( ShowTemplateMatchResultsPanelParent::OnSavePeaksClick ), NULL, this );

}

MatchTemplateResultsPanelParent::MatchTemplateResultsPanelParent( wxWindow* parent, wxWindowID id, const wxPoint& pos, const wxSize& size, long style, const wxString& name ) : wxPanel( parent, id, pos, size, style, name )
{
	wxBoxSizer* bSizer63;
	bSizer63 = new wxBoxSizer( wxVERTICAL );

	m_staticline25 = new wxStaticLine( this, wxID_ANY, wxDefaultPosition, wxDefaultSize, wxLI_HORIZONTAL );
	bSizer63->Add( m_staticline25, 0, wxEXPAND | wxALL, 5 );

	m_splitter4 = new wxSplitterWindow( this, wxID_ANY, wxDefaultPosition, wxDefaultSize, wxSP_3D );
	m_splitter4->SetSashGravity( 0.5 );
	m_splitter4->Connect( wxEVT_IDLE, wxIdleEventHandler( MatchTemplateResultsPanelParent::m_splitter4OnIdle ), NULL, this );

	m_panel13 = new wxPanel( m_splitter4, wxID_ANY, wxDefaultPosition, wxDefaultSize, wxTAB_TRAVERSAL );
	wxBoxSizer* bSizer66;
	bSizer66 = new wxBoxSizer( wxVERTICAL );

	wxBoxSizer* bSizer64;
	bSizer64 = new wxBoxSizer( wxHORIZONTAL );

	AllImagesButton = new wxRadioButton( m_panel13, wxID_ANY, wxT("All Images"), wxDefaultPosition, wxDefaultSize, 0 );
	AllImagesButton->Hide();

	bSizer64->Add( AllImagesButton, 0, wxALIGN_CENTER_VERTICAL|wxALL, 5 );

	ByFilterButton = new wxRadioButton( m_panel13, wxID_ANY, wxT("By Filter"), wxDefaultPosition, wxDefaultSize, 0 );
	ByFilterButton->Hide();

	bSizer64->Add( ByFilterButton, 0, wxALIGN_CENTER_VERTICAL|wxALL, 5 );

	FilterButton = new wxButton( m_panel13, wxID_ANY, wxT("Define Filter"), wxDefaultPosition, wxDefaultSize, 0 );
	FilterButton->Enable( false );
	FilterButton->Hide();

	bSizer64->Add( FilterButton, 0, wxALIGN_CENTER_VERTICAL|wxALL, 5 );


	bSizer64->Add( 0, 0, 1, wxEXPAND, 5 );

	m_staticline77 = new wxStaticLine( m_panel13, wxID_ANY, wxDefaultPosition, wxDefaultSize, wxLI_HORIZONTAL|wxLI_VERTICAL );
	bSizer64->Add( m_staticline77, 0, wxEXPAND | wxALL, 5 );

	JobDetailsToggleButton = new wxToggleButton( m_panel13, wxID_ANY, wxT("Show Job Details"), wxDefaultPosition, wxDefaultSize, 0 );
	bSizer64->Add( JobDetailsToggleButton, 0, wxALIGN_CENTER_VERTICAL|wxALIGN_RIGHT|wxALL, 5 );


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
	wxBoxSizer* bSizer101;
	bSizer101 = new wxBoxSizer( wxVERTICAL );

	JobTitleStaticText = new wxStaticText( JobDetailsPanel, wxID_ANY, wxT("MyLabel"), wxDefaultPosition, wxDefaultSize, 0 );
	JobTitleStaticText->Wrap( -1 );
	JobTitleStaticText->SetFont( wxFont( 10, wxFONTFAMILY_SWISS, wxFONTSTYLE_NORMAL, wxFONTWEIGHT_BOLD, true, wxT("Sans") ) );

	bSizer101->Add( JobTitleStaticText, 0, wxALL, 5 );

	InfoSizer = new wxFlexGridSizer( 0, 8, 0, 0 );
	InfoSizer->SetFlexibleDirection( wxBOTH );
	InfoSizer->SetNonFlexibleGrowMode( wxFLEX_GROWMODE_SPECIFIED );

	m_staticText72 = new wxStaticText( JobDetailsPanel, wxID_ANY, wxT("Job ID :"), wxDefaultPosition, wxDefaultSize, 0 );
	m_staticText72->Wrap( -1 );
	m_staticText72->SetFont( wxFont( 10, wxFONTFAMILY_SWISS, wxFONTSTYLE_NORMAL, wxFONTWEIGHT_BOLD, false, wxT("Sans") ) );

	InfoSizer->Add( m_staticText72, 0, wxALIGN_RIGHT|wxALL, 5 );

	JobIDStaticText = new wxStaticText( JobDetailsPanel, wxID_ANY, wxEmptyString, wxDefaultPosition, wxDefaultSize, 0 );
	JobIDStaticText->Wrap( -1 );
	InfoSizer->Add( JobIDStaticText, 0, wxALIGN_LEFT|wxALL, 5 );

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

	m_staticText788 = new wxStaticText( JobDetailsPanel, wxID_ANY, wxT("Ref. Volume ID :"), wxDefaultPosition, wxDefaultSize, 0 );
	m_staticText788->Wrap( -1 );
	m_staticText788->SetFont( wxFont( 10, wxFONTFAMILY_SWISS, wxFONTSTYLE_NORMAL, wxFONTWEIGHT_BOLD, false, wxT("Sans") ) );

	InfoSizer->Add( m_staticText788, 0, wxALIGN_RIGHT|wxALL, 5 );

	RefVolumeIDStaticText = new wxStaticText( JobDetailsPanel, wxID_ANY, wxEmptyString, wxDefaultPosition, wxDefaultSize, 0 );
	RefVolumeIDStaticText->Wrap( -1 );
	InfoSizer->Add( RefVolumeIDStaticText, 0, wxALL, 5 );

	m_staticText790 = new wxStaticText( JobDetailsPanel, wxID_ANY, wxT("Used Symmetry :"), wxDefaultPosition, wxDefaultSize, 0 );
	m_staticText790->Wrap( -1 );
	m_staticText790->SetFont( wxFont( 10, wxFONTFAMILY_SWISS, wxFONTSTYLE_NORMAL, wxFONTWEIGHT_BOLD, false, wxT("Sans") ) );

	InfoSizer->Add( m_staticText790, 0, wxALIGN_RIGHT|wxALL, 5 );

	SymmetryStaticText = new wxStaticText( JobDetailsPanel, wxID_ANY, wxEmptyString, wxDefaultPosition, wxDefaultSize, 0 );
	SymmetryStaticText->Wrap( -1 );
	InfoSizer->Add( SymmetryStaticText, 0, wxALL, 5 );

	m_staticText78 = new wxStaticText( JobDetailsPanel, wxID_ANY, wxT("Pixel Size :"), wxDefaultPosition, wxDefaultSize, 0 );
	m_staticText78->Wrap( -1 );
	m_staticText78->SetFont( wxFont( 10, wxFONTFAMILY_SWISS, wxFONTSTYLE_NORMAL, wxFONTWEIGHT_BOLD, false, wxT("Sans") ) );

	InfoSizer->Add( m_staticText78, 0, wxALIGN_RIGHT|wxALL, 5 );

	PixelSizeStaticText = new wxStaticText( JobDetailsPanel, wxID_ANY, wxEmptyString, wxDefaultPosition, wxDefaultSize, 0 );
	PixelSizeStaticText->Wrap( -1 );
	InfoSizer->Add( PixelSizeStaticText, 0, wxALIGN_LEFT|wxALL, 5 );

	m_staticText83 = new wxStaticText( JobDetailsPanel, wxID_ANY, wxT("Voltage :"), wxDefaultPosition, wxDefaultSize, 0 );
	m_staticText83->Wrap( -1 );
	m_staticText83->SetFont( wxFont( 10, wxFONTFAMILY_SWISS, wxFONTSTYLE_NORMAL, wxFONTWEIGHT_BOLD, false, wxT("Sans") ) );

	InfoSizer->Add( m_staticText83, 0, wxALIGN_RIGHT|wxALL, 5 );

	VoltageStaticText = new wxStaticText( JobDetailsPanel, wxID_ANY, wxEmptyString, wxDefaultPosition, wxDefaultSize, 0 );
	VoltageStaticText->Wrap( -1 );
	InfoSizer->Add( VoltageStaticText, 0, wxALIGN_LEFT|wxALL, 5 );

	m_staticText82 = new wxStaticText( JobDetailsPanel, wxID_ANY, wxT("Cs :"), wxDefaultPosition, wxDefaultSize, 0 );
	m_staticText82->Wrap( -1 );
	m_staticText82->SetFont( wxFont( 10, wxFONTFAMILY_SWISS, wxFONTSTYLE_NORMAL, wxFONTWEIGHT_BOLD, false, wxT("Sans") ) );

	InfoSizer->Add( m_staticText82, 0, wxALIGN_RIGHT|wxALL, 5 );

	CsStaticText = new wxStaticText( JobDetailsPanel, wxID_ANY, wxEmptyString, wxDefaultPosition, wxDefaultSize, 0 );
	CsStaticText->Wrap( -1 );
	InfoSizer->Add( CsStaticText, 0, wxALIGN_LEFT|wxALL, 5 );

	m_staticText96 = new wxStaticText( JobDetailsPanel, wxID_ANY, wxT("Amp. Contrast :"), wxDefaultPosition, wxDefaultSize, 0 );
	m_staticText96->Wrap( -1 );
	m_staticText96->SetFont( wxFont( 10, wxFONTFAMILY_SWISS, wxFONTSTYLE_NORMAL, wxFONTWEIGHT_BOLD, false, wxT("Sans") ) );

	InfoSizer->Add( m_staticText96, 0, wxALIGN_RIGHT|wxALL, 5 );

	AmplitudeContrastStaticText = new wxStaticText( JobDetailsPanel, wxID_ANY, wxEmptyString, wxDefaultPosition, wxDefaultSize, 0 );
	AmplitudeContrastStaticText->Wrap( -1 );
	InfoSizer->Add( AmplitudeContrastStaticText, 0, wxALIGN_LEFT|wxALL, 5 );

	m_staticText85 = new wxStaticText( JobDetailsPanel, wxID_ANY, wxT("Defocus 1 :"), wxDefaultPosition, wxDefaultSize, 0 );
	m_staticText85->Wrap( -1 );
	m_staticText85->SetFont( wxFont( 10, wxFONTFAMILY_SWISS, wxFONTSTYLE_NORMAL, wxFONTWEIGHT_BOLD, false, wxT("Sans") ) );

	InfoSizer->Add( m_staticText85, 0, wxALIGN_RIGHT|wxALL, 5 );

	Defocus1StaticText = new wxStaticText( JobDetailsPanel, wxID_ANY, wxEmptyString, wxDefaultPosition, wxDefaultSize, 0 );
	Defocus1StaticText->Wrap( -1 );
	InfoSizer->Add( Defocus1StaticText, 0, wxALIGN_LEFT|wxALL, 5 );

	m_staticText792 = new wxStaticText( JobDetailsPanel, wxID_ANY, wxT("Defocus 2 :"), wxDefaultPosition, wxDefaultSize, 0 );
	m_staticText792->Wrap( -1 );
	m_staticText792->SetFont( wxFont( 10, wxFONTFAMILY_SWISS, wxFONTSTYLE_NORMAL, wxFONTWEIGHT_BOLD, false, wxT("Sans") ) );

	InfoSizer->Add( m_staticText792, 0, wxALIGN_RIGHT|wxALL, 5 );

	Defocus2StaticText = new wxStaticText( JobDetailsPanel, wxID_ANY, wxEmptyString, wxDefaultPosition, wxDefaultSize, 0 );
	Defocus2StaticText->Wrap( -1 );
	InfoSizer->Add( Defocus2StaticText, 0, wxALL, 5 );

	m_staticText794 = new wxStaticText( JobDetailsPanel, wxID_ANY, wxT("Defocus Angle :"), wxDefaultPosition, wxDefaultSize, 0 );
	m_staticText794->Wrap( -1 );
	m_staticText794->SetFont( wxFont( 10, wxFONTFAMILY_SWISS, wxFONTSTYLE_NORMAL, wxFONTWEIGHT_BOLD, false, wxT("Sans") ) );

	InfoSizer->Add( m_staticText794, 0, wxALIGN_RIGHT|wxALL, 5 );

	DefocusAngleStaticText = new wxStaticText( JobDetailsPanel, wxID_ANY, wxEmptyString, wxDefaultPosition, wxDefaultSize, 0 );
	DefocusAngleStaticText->Wrap( -1 );
	InfoSizer->Add( DefocusAngleStaticText, 0, wxALL, 5 );

	m_staticText796 = new wxStaticText( JobDetailsPanel, wxID_ANY, wxT("Phase Shift :"), wxDefaultPosition, wxDefaultSize, 0 );
	m_staticText796->Wrap( -1 );
	m_staticText796->SetFont( wxFont( 10, wxFONTFAMILY_SWISS, wxFONTSTYLE_NORMAL, wxFONTWEIGHT_BOLD, false, wxT("Sans") ) );

	InfoSizer->Add( m_staticText796, 0, wxALIGN_RIGHT|wxALL, 5 );

	PhaseShiftStaticText = new wxStaticText( JobDetailsPanel, wxID_ANY, wxEmptyString, wxDefaultPosition, wxDefaultSize, 0 );
	PhaseShiftStaticText->Wrap( -1 );
	InfoSizer->Add( PhaseShiftStaticText, 0, wxALL, 5 );

	m_staticText87 = new wxStaticText( JobDetailsPanel, wxID_ANY, wxT("Low Res. Limit :"), wxDefaultPosition, wxDefaultSize, 0 );
	m_staticText87->Wrap( -1 );
	m_staticText87->SetFont( wxFont( 10, wxFONTFAMILY_SWISS, wxFONTSTYLE_NORMAL, wxFONTWEIGHT_BOLD, false, wxT("Sans") ) );

	InfoSizer->Add( m_staticText87, 0, wxALIGN_RIGHT|wxALL, 5 );

	LowResLimitStaticText = new wxStaticText( JobDetailsPanel, wxID_ANY, wxEmptyString, wxDefaultPosition, wxDefaultSize, 0 );
	LowResLimitStaticText->Wrap( -1 );
	InfoSizer->Add( LowResLimitStaticText, 0, wxALIGN_LEFT|wxALL, 5 );

	m_staticText89 = new wxStaticText( JobDetailsPanel, wxID_ANY, wxT("High Res. Limit :"), wxDefaultPosition, wxDefaultSize, 0 );
	m_staticText89->Wrap( -1 );
	m_staticText89->SetFont( wxFont( 10, wxFONTFAMILY_SWISS, wxFONTSTYLE_NORMAL, wxFONTWEIGHT_BOLD, false, wxT("Sans") ) );

	InfoSizer->Add( m_staticText89, 0, wxALIGN_RIGHT|wxALL, 5 );

	HighResLimitStaticText = new wxStaticText( JobDetailsPanel, wxID_ANY, wxEmptyString, wxDefaultPosition, wxDefaultSize, 0 );
	HighResLimitStaticText->Wrap( -1 );
	InfoSizer->Add( HighResLimitStaticText, 0, wxALIGN_LEFT|wxALL, 5 );

	m_staticText91 = new wxStaticText( JobDetailsPanel, wxID_ANY, wxT("OOP Angluar Step :"), wxDefaultPosition, wxDefaultSize, 0 );
	m_staticText91->Wrap( -1 );
	m_staticText91->SetFont( wxFont( 10, wxFONTFAMILY_SWISS, wxFONTSTYLE_NORMAL, wxFONTWEIGHT_BOLD, false, wxT("Sans") ) );

	InfoSizer->Add( m_staticText91, 0, wxALIGN_RIGHT|wxALL, 5 );

	OOPAngluarStepStaticText = new wxStaticText( JobDetailsPanel, wxID_ANY, wxEmptyString, wxDefaultPosition, wxDefaultSize, 0 );
	OOPAngluarStepStaticText->Wrap( -1 );
	InfoSizer->Add( OOPAngluarStepStaticText, 0, wxALIGN_LEFT|wxALL, 5 );

	m_staticText79 = new wxStaticText( JobDetailsPanel, wxID_ANY, wxT("IP Angular Step :"), wxDefaultPosition, wxDefaultSize, 0 );
	m_staticText79->Wrap( -1 );
	m_staticText79->SetFont( wxFont( 10, wxFONTFAMILY_SWISS, wxFONTSTYLE_NORMAL, wxFONTWEIGHT_BOLD, false, wxT("Sans") ) );

	InfoSizer->Add( m_staticText79, 0, wxALIGN_RIGHT|wxALL, 5 );

	IPAngluarStepStaticText = new wxStaticText( JobDetailsPanel, wxID_ANY, wxEmptyString, wxDefaultPosition, wxDefaultSize, 0 );
	IPAngluarStepStaticText->Wrap( -1 );
	InfoSizer->Add( IPAngluarStepStaticText, 0, wxALIGN_LEFT|wxALL, 5 );

	m_staticText798 = new wxStaticText( JobDetailsPanel, wxID_ANY, wxT("Defocus Range :"), wxDefaultPosition, wxDefaultSize, 0 );
	m_staticText798->Wrap( -1 );
	m_staticText798->SetFont( wxFont( 10, wxFONTFAMILY_SWISS, wxFONTSTYLE_NORMAL, wxFONTWEIGHT_BOLD, false, wxT("Sans") ) );

	InfoSizer->Add( m_staticText798, 0, wxALIGN_RIGHT|wxALL, 5 );

	DefocusRangeStaticText = new wxStaticText( JobDetailsPanel, wxID_ANY, wxEmptyString, wxDefaultPosition, wxDefaultSize, 0 );
	DefocusRangeStaticText->Wrap( -1 );
	InfoSizer->Add( DefocusRangeStaticText, 0, wxALL, 5 );

	m_staticText95 = new wxStaticText( JobDetailsPanel, wxID_ANY, wxT("Defocus Step :"), wxDefaultPosition, wxDefaultSize, 0 );
	m_staticText95->Wrap( -1 );
	m_staticText95->SetFont( wxFont( 10, wxFONTFAMILY_SWISS, wxFONTSTYLE_NORMAL, wxFONTWEIGHT_BOLD, false, wxT("Sans") ) );

	InfoSizer->Add( m_staticText95, 0, wxALIGN_RIGHT|wxALL, 5 );

	DefocusStepStaticText = new wxStaticText( JobDetailsPanel, wxID_ANY, wxEmptyString, wxDefaultPosition, wxDefaultSize, 0 );
	DefocusStepStaticText->Wrap( -1 );
	InfoSizer->Add( DefocusStepStaticText, 0, wxALIGN_LEFT|wxALL, 5 );

	LargeAstigExpectedLabel = new wxStaticText( JobDetailsPanel, wxID_ANY, wxT("Pixel Size Range :"), wxDefaultPosition, wxDefaultSize, 0 );
	LargeAstigExpectedLabel->Wrap( -1 );
	LargeAstigExpectedLabel->SetFont( wxFont( 10, wxFONTFAMILY_SWISS, wxFONTSTYLE_NORMAL, wxFONTWEIGHT_BOLD, false, wxT("Sans") ) );

	InfoSizer->Add( LargeAstigExpectedLabel, 0, wxALIGN_RIGHT|wxALL, 5 );

	PixelSizeRangeStaticText = new wxStaticText( JobDetailsPanel, wxID_ANY, wxEmptyString, wxDefaultPosition, wxDefaultSize, 0 );
	PixelSizeRangeStaticText->Wrap( -1 );
	InfoSizer->Add( PixelSizeRangeStaticText, 0, wxALL, 5 );

	m_staticText99 = new wxStaticText( JobDetailsPanel, wxID_ANY, wxT("Pixel Size Step :"), wxDefaultPosition, wxDefaultSize, 0 );
	m_staticText99->Wrap( -1 );
	m_staticText99->SetFont( wxFont( 10, wxFONTFAMILY_SWISS, wxFONTSTYLE_NORMAL, wxFONTWEIGHT_BOLD, false, wxT("Sans") ) );

	InfoSizer->Add( m_staticText99, 0, wxALIGN_RIGHT|wxALL, 5 );

	PixelSizeStepStaticText = new wxStaticText( JobDetailsPanel, wxID_ANY, wxEmptyString, wxDefaultPosition, wxDefaultSize, 0 );
	PixelSizeStepStaticText->Wrap( -1 );
	InfoSizer->Add( PixelSizeStepStaticText, 0, wxALIGN_LEFT|wxALL, 5 );

	m_staticText872 = new wxStaticText( JobDetailsPanel, wxID_ANY, wxT("Min. Peak Radius :"), wxDefaultPosition, wxDefaultSize, 0 );
	m_staticText872->Wrap( -1 );
	m_staticText872->SetFont( wxFont( 10, wxFONTFAMILY_SWISS, wxFONTSTYLE_NORMAL, wxFONTWEIGHT_BOLD, false, wxT("Sans") ) );

	InfoSizer->Add( m_staticText872, 0, wxALIGN_RIGHT|wxALL, 5 );

	MinPeakRadiusStaticText = new wxStaticText( JobDetailsPanel, wxID_ANY, wxEmptyString, wxDefaultPosition, wxDefaultSize, 0 );
	MinPeakRadiusStaticText->Wrap( -1 );
	InfoSizer->Add( MinPeakRadiusStaticText, 0, wxALL, 5 );

	m_staticText874 = new wxStaticText( JobDetailsPanel, wxID_ANY, wxT("Shift Threshold :"), wxDefaultPosition, wxDefaultSize, 0 );
	m_staticText874->Wrap( -1 );
	m_staticText874->SetFont( wxFont( 10, wxFONTFAMILY_SWISS, wxFONTSTYLE_NORMAL, wxFONTWEIGHT_BOLD, false, wxT("Sans") ) );

	InfoSizer->Add( m_staticText874, 0, wxALIGN_RIGHT|wxALL, 5 );

	ShiftThresholdStaticText = new wxStaticText( JobDetailsPanel, wxID_ANY, wxEmptyString, wxDefaultPosition, wxDefaultSize, 0 );
	ShiftThresholdStaticText->Wrap( -1 );
	InfoSizer->Add( ShiftThresholdStaticText, 0, wxALL, 5 );

	m_staticText876 = new wxStaticText( JobDetailsPanel, wxID_ANY, wxT("Ignore Shifted Peaks :"), wxDefaultPosition, wxDefaultSize, 0 );
	m_staticText876->Wrap( -1 );
	m_staticText876->SetFont( wxFont( 10, wxFONTFAMILY_SWISS, wxFONTSTYLE_NORMAL, wxFONTWEIGHT_BOLD, false, wxT("Sans") ) );

	InfoSizer->Add( m_staticText876, 0, wxALIGN_RIGHT|wxALL, 5 );

	IgnoreShiftedPeaksStaticText = new wxStaticText( JobDetailsPanel, wxID_ANY, wxEmptyString, wxDefaultPosition, wxDefaultSize, 0 );
	IgnoreShiftedPeaksStaticText->Wrap( -1 );
	InfoSizer->Add( IgnoreShiftedPeaksStaticText, 0, wxALL, 5 );


	bSizer101->Add( InfoSizer, 1, wxEXPAND, 5 );

	m_staticline30 = new wxStaticLine( JobDetailsPanel, wxID_ANY, wxDefaultPosition, wxDefaultSize, wxLI_HORIZONTAL );
	bSizer101->Add( m_staticline30, 0, wxEXPAND | wxALL, 5 );


	JobDetailsPanel->SetSizer( bSizer101 );
	JobDetailsPanel->Layout();
	bSizer101->Fit( JobDetailsPanel );
	bSizer73->Add( JobDetailsPanel, 1, wxALL|wxEXPAND, 5 );


	bSizer681->Add( bSizer73, 0, wxEXPAND, 5 );

	wxGridSizer* gSizer5;
	gSizer5 = new wxGridSizer( 0, 6, 0, 0 );


	bSizer681->Add( gSizer5, 0, wxEXPAND, 5 );

	ResultPanel = new ShowTemplateMatchResultsPanel( RightPanel, wxID_ANY, wxDefaultPosition, wxDefaultSize, wxTAB_TRAVERSAL );
	bSizer681->Add( ResultPanel, 1, wxEXPAND | wxALL, 5 );

	wxBoxSizer* bSizer69;
	bSizer69 = new wxBoxSizer( wxHORIZONTAL );

	DeleteFromGroupButton = new wxButton( RightPanel, wxID_ANY, wxT("Delete Image From Group"), wxDefaultPosition, wxDefaultSize, 0 );
	bSizer69->Add( DeleteFromGroupButton, 0, wxALL, 5 );

	AddToGroupButton = new wxButton( RightPanel, wxID_ANY, wxT("Add Image To Group"), wxDefaultPosition, wxDefaultSize, 0 );
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
	this->Connect( wxEVT_UPDATE_UI, wxUpdateUIEventHandler( MatchTemplateResultsPanelParent::OnUpdateUI ) );
	AllImagesButton->Connect( wxEVT_COMMAND_RADIOBUTTON_SELECTED, wxCommandEventHandler( MatchTemplateResultsPanelParent::OnAllImagesSelect ), NULL, this );
	ByFilterButton->Connect( wxEVT_COMMAND_RADIOBUTTON_SELECTED, wxCommandEventHandler( MatchTemplateResultsPanelParent::OnByFilterSelect ), NULL, this );
	FilterButton->Connect( wxEVT_COMMAND_BUTTON_CLICKED, wxCommandEventHandler( MatchTemplateResultsPanelParent::OnDefineFilterClick ), NULL, this );
	JobDetailsToggleButton->Connect( wxEVT_COMMAND_TOGGLEBUTTON_CLICKED, wxCommandEventHandler( MatchTemplateResultsPanelParent::OnJobDetailsToggle ), NULL, this );
	PreviousButton->Connect( wxEVT_COMMAND_BUTTON_CLICKED, wxCommandEventHandler( MatchTemplateResultsPanelParent::OnPreviousButtonClick ), NULL, this );
	AddAllToGroupButton->Connect( wxEVT_COMMAND_BUTTON_CLICKED, wxCommandEventHandler( MatchTemplateResultsPanelParent::OnAddAllToGroupClick ), NULL, this );
	NextButton->Connect( wxEVT_COMMAND_BUTTON_CLICKED, wxCommandEventHandler( MatchTemplateResultsPanelParent::OnNextButtonClick ), NULL, this );
	DeleteFromGroupButton->Connect( wxEVT_COMMAND_BUTTON_CLICKED, wxCommandEventHandler( MatchTemplateResultsPanelParent::OnRemoveFromGroupClick ), NULL, this );
	AddToGroupButton->Connect( wxEVT_COMMAND_BUTTON_CLICKED, wxCommandEventHandler( MatchTemplateResultsPanelParent::OnAddToGroupClick ), NULL, this );
}

MatchTemplateResultsPanelParent::~MatchTemplateResultsPanelParent()
{
	// Disconnect Events
	this->Disconnect( wxEVT_UPDATE_UI, wxUpdateUIEventHandler( MatchTemplateResultsPanelParent::OnUpdateUI ) );
	AllImagesButton->Disconnect( wxEVT_COMMAND_RADIOBUTTON_SELECTED, wxCommandEventHandler( MatchTemplateResultsPanelParent::OnAllImagesSelect ), NULL, this );
	ByFilterButton->Disconnect( wxEVT_COMMAND_RADIOBUTTON_SELECTED, wxCommandEventHandler( MatchTemplateResultsPanelParent::OnByFilterSelect ), NULL, this );
	FilterButton->Disconnect( wxEVT_COMMAND_BUTTON_CLICKED, wxCommandEventHandler( MatchTemplateResultsPanelParent::OnDefineFilterClick ), NULL, this );
	JobDetailsToggleButton->Disconnect( wxEVT_COMMAND_TOGGLEBUTTON_CLICKED, wxCommandEventHandler( MatchTemplateResultsPanelParent::OnJobDetailsToggle ), NULL, this );
	PreviousButton->Disconnect( wxEVT_COMMAND_BUTTON_CLICKED, wxCommandEventHandler( MatchTemplateResultsPanelParent::OnPreviousButtonClick ), NULL, this );
	AddAllToGroupButton->Disconnect( wxEVT_COMMAND_BUTTON_CLICKED, wxCommandEventHandler( MatchTemplateResultsPanelParent::OnAddAllToGroupClick ), NULL, this );
	NextButton->Disconnect( wxEVT_COMMAND_BUTTON_CLICKED, wxCommandEventHandler( MatchTemplateResultsPanelParent::OnNextButtonClick ), NULL, this );
	DeleteFromGroupButton->Disconnect( wxEVT_COMMAND_BUTTON_CLICKED, wxCommandEventHandler( MatchTemplateResultsPanelParent::OnRemoveFromGroupClick ), NULL, this );
	AddToGroupButton->Disconnect( wxEVT_COMMAND_BUTTON_CLICKED, wxCommandEventHandler( MatchTemplateResultsPanelParent::OnAddToGroupClick ), NULL, this );

}

MatchTemplatePanelParent::MatchTemplatePanelParent( wxWindow* parent, wxWindowID id, const wxPoint& pos, const wxSize& size, long style, const wxString& name ) : JobPanel( parent, id, pos, size, style, name )
{
	wxBoxSizer* bSizer43;
	bSizer43 = new wxBoxSizer( wxVERTICAL );

	m_staticline149 = new wxStaticLine( this, wxID_ANY, wxDefaultPosition, wxDefaultSize, wxLI_HORIZONTAL );
	bSizer43->Add( m_staticline149, 0, wxEXPAND | wxALL, 5 );

	InputPanel = new wxPanel( this, wxID_ANY, wxDefaultPosition, wxDefaultSize, wxTAB_TRAVERSAL );
	wxBoxSizer* bSizer534;
	bSizer534 = new wxBoxSizer( wxVERTICAL );

	wxBoxSizer* bSizer45;
	bSizer45 = new wxBoxSizer( wxHORIZONTAL );

	wxBoxSizer* bSizer557;
	bSizer557 = new wxBoxSizer( wxHORIZONTAL );

	wxFlexGridSizer* fgSizer15;
	fgSizer15 = new wxFlexGridSizer( 0, 2, 0, 0 );
	fgSizer15->SetFlexibleDirection( wxHORIZONTAL );
	fgSizer15->SetNonFlexibleGrowMode( wxFLEX_GROWMODE_SPECIFIED );

	m_staticText262 = new wxStaticText( InputPanel, wxID_ANY, wxT("Input Image Group :"), wxDefaultPosition, wxSize( -1,-1 ), 0 );
	m_staticText262->Wrap( -1 );
	fgSizer15->Add( m_staticText262, 0, wxALIGN_CENTER_VERTICAL|wxALIGN_RIGHT|wxALL, 5 );

	GroupComboBox = new ImageGroupPickerComboPanel( InputPanel, wxID_ANY, wxDefaultPosition, wxDefaultSize, wxTAB_TRAVERSAL );
	GroupComboBox->SetMinSize( wxSize( 350,-1 ) );
	GroupComboBox->SetMaxSize( wxSize( 350,-1 ) );

	fgSizer15->Add( GroupComboBox, 1, wxEXPAND | wxALL, 5 );

	m_staticText478 = new wxStaticText( InputPanel, wxID_ANY, wxT("Reference Volume :"), wxDefaultPosition, wxDefaultSize, 0 );
	m_staticText478->Wrap( -1 );
	fgSizer15->Add( m_staticText478, 0, wxALIGN_CENTER_VERTICAL|wxALIGN_RIGHT|wxALL, 5 );

	ReferenceSelectPanel = new VolumeAssetPickerComboPanel( InputPanel, wxID_ANY, wxDefaultPosition, wxDefaultSize, wxTAB_TRAVERSAL );
	ReferenceSelectPanel->SetMinSize( wxSize( 350,-1 ) );
	ReferenceSelectPanel->SetMaxSize( wxSize( 350,-1 ) );

	fgSizer15->Add( ReferenceSelectPanel, 100, wxALIGN_CENTER_VERTICAL|wxALL, 5 );


	bSizer557->Add( fgSizer15, 0, wxEXPAND, 5 );


	bSizer45->Add( bSizer557, 1, wxEXPAND, 5 );


	bSizer534->Add( bSizer45, 1, wxEXPAND, 5 );

	PleaseEstimateCTFStaticText = new wxStaticText( InputPanel, wxID_ANY, wxT("Please run CTF estimation on this group before picking particles"), wxDefaultPosition, wxDefaultSize, 0 );
	PleaseEstimateCTFStaticText->Wrap( -1 );
	PleaseEstimateCTFStaticText->SetFont( wxFont( wxNORMAL_FONT->GetPointSize(), wxFONTFAMILY_DEFAULT, wxFONTSTYLE_NORMAL, wxFONTWEIGHT_BOLD, false, wxEmptyString ) );
	PleaseEstimateCTFStaticText->SetForegroundColour( wxColour( 180, 0, 0 ) );
	PleaseEstimateCTFStaticText->Hide();

	bSizer534->Add( PleaseEstimateCTFStaticText, 0, wxALL, 5 );

	m_staticline151 = new wxStaticLine( InputPanel, wxID_ANY, wxDefaultPosition, wxDefaultSize, wxLI_HORIZONTAL );
	bSizer534->Add( m_staticline151, 0, wxEXPAND | wxALL, 5 );


	InputPanel->SetSizer( bSizer534 );
	InputPanel->Layout();
	bSizer534->Fit( InputPanel );
	bSizer43->Add( InputPanel, 0, wxEXPAND | wxALL, 5 );

	wxBoxSizer* bSizer46;
	bSizer46 = new wxBoxSizer( wxHORIZONTAL );

	ExpertPanel = new wxScrolledWindow( this, wxID_ANY, wxDefaultPosition, wxSize( -1,-1 ), wxVSCROLL );
	ExpertPanel->SetScrollRate( 5, 5 );
	InputSizer = new wxBoxSizer( wxVERTICAL );

	wxFlexGridSizer* fgSizer1;
	fgSizer1 = new wxFlexGridSizer( 0, 2, 0, 0 );
	fgSizer1->SetFlexibleDirection( wxBOTH );
	fgSizer1->SetNonFlexibleGrowMode( wxFLEX_GROWMODE_SPECIFIED );

	m_staticText201 = new wxStaticText( ExpertPanel, wxID_ANY, wxT("Search Limits"), wxDefaultPosition, wxDefaultSize, 0 );
	m_staticText201->Wrap( -1 );
	m_staticText201->SetFont( wxFont( wxNORMAL_FONT->GetPointSize(), wxFONTFAMILY_DEFAULT, wxFONTSTYLE_NORMAL, wxFONTWEIGHT_BOLD, true, wxEmptyString ) );

	fgSizer1->Add( m_staticText201, 0, wxALIGN_BOTTOM|wxALL, 5 );

	ResetAllDefaultsButton = new wxButton( ExpertPanel, wxID_ANY, wxT("Reset All Defaults"), wxDefaultPosition, wxDefaultSize, 0 );
	fgSizer1->Add( ResetAllDefaultsButton, 0, wxALIGN_RIGHT|wxALL, 5 );

	m_staticText189 = new wxStaticText( ExpertPanel, wxID_ANY, wxT("Out of Plane Angular Step (°) : "), wxDefaultPosition, wxDefaultSize, 0 );
	m_staticText189->Wrap( -1 );
	fgSizer1->Add( m_staticText189, 0, wxALIGN_CENTER_VERTICAL|wxALL, 5 );

	OutofPlaneStepNumericCtrl = new NumericTextCtrl( ExpertPanel, wxID_ANY, wxT("2.5"), wxDefaultPosition, wxDefaultSize, wxTE_PROCESS_ENTER );
	fgSizer1->Add( OutofPlaneStepNumericCtrl, 0, wxALL|wxEXPAND, 5 );

	m_staticText190 = new wxStaticText( ExpertPanel, wxID_ANY, wxT("In Plane Angular Step (°) :"), wxDefaultPosition, wxDefaultSize, 0 );
	m_staticText190->Wrap( -1 );
	fgSizer1->Add( m_staticText190, 0, wxALIGN_CENTER_VERTICAL|wxALL, 5 );

	InPlaneStepNumericCtrl = new NumericTextCtrl( ExpertPanel, wxID_ANY, wxT("1.5"), wxDefaultPosition, wxDefaultSize, wxTE_PROCESS_ENTER );
	fgSizer1->Add( InPlaneStepNumericCtrl, 0, wxALL|wxEXPAND, 5 );

	m_staticText190211 = new wxStaticText( ExpertPanel, wxID_ANY, wxT("High-Resolution Limit (Å) :"), wxDefaultPosition, wxDefaultSize, 0 );
	m_staticText190211->Wrap( -1 );
	fgSizer1->Add( m_staticText190211, 0, wxALIGN_CENTER_VERTICAL|wxALL, 5 );

	HighResolutionLimitNumericCtrl = new NumericTextCtrl( ExpertPanel, wxID_ANY, wxT("2.0"), wxDefaultPosition, wxDefaultSize, wxTE_PROCESS_ENTER );
	fgSizer1->Add( HighResolutionLimitNumericCtrl, 0, wxALL|wxEXPAND, 5 );

	m_staticText19021 = new wxStaticText( ExpertPanel, wxID_ANY, wxT("Pointgroup Symmetry :"), wxDefaultPosition, wxDefaultSize, 0 );
	m_staticText19021->Wrap( -1 );
	fgSizer1->Add( m_staticText19021, 0, wxALIGN_CENTER_VERTICAL|wxALL, 5 );

	SymmetryComboBox = new wxComboBox( ExpertPanel, wxID_ANY, wxT("C1"), wxDefaultPosition, wxDefaultSize, 0, NULL, 0 );
	fgSizer1->Add( SymmetryComboBox, 0, wxALL|wxEXPAND, 5 );

	m_staticText698 = new wxStaticText( ExpertPanel, wxID_ANY, wxT("Perform Defocus Search? "), wxDefaultPosition, wxDefaultSize, 0 );
	m_staticText698->Wrap( -1 );
	fgSizer1->Add( m_staticText698, 0, wxALIGN_CENTER_VERTICAL|wxALL, 5 );

	wxBoxSizer* bSizer265;
	bSizer265 = new wxBoxSizer( wxHORIZONTAL );

	DefocusSearchYesRadio = new wxRadioButton( ExpertPanel, wxID_ANY, wxT("Yes"), wxDefaultPosition, wxDefaultSize, wxRB_GROUP );
	bSizer265->Add( DefocusSearchYesRadio, 0, wxALL, 5 );

	DefocusSearchNoRadio = new wxRadioButton( ExpertPanel, wxID_ANY, wxT("No"), wxDefaultPosition, wxDefaultSize, 0 );
	bSizer265->Add( DefocusSearchNoRadio, 0, wxALL, 5 );


	fgSizer1->Add( bSizer265, 1, wxEXPAND, 5 );

	DefocusRangeStaticText = new wxStaticText( ExpertPanel, wxID_ANY, wxT("\tDefocus Range (Å) :"), wxDefaultPosition, wxDefaultSize, 0 );
	DefocusRangeStaticText->Wrap( -1 );
	DefocusRangeStaticText->Enable( false );

	fgSizer1->Add( DefocusRangeStaticText, 0, wxALIGN_CENTER_VERTICAL|wxALL, 5 );

	DefocusSearchRangeNumericCtrl = new NumericTextCtrl( ExpertPanel, wxID_ANY, wxT("1200"), wxDefaultPosition, wxDefaultSize, wxTE_PROCESS_ENTER );
	DefocusSearchRangeNumericCtrl->Enable( false );

	fgSizer1->Add( DefocusSearchRangeNumericCtrl, 0, wxALL|wxEXPAND, 5 );

	DefocusStepStaticText = new wxStaticText( ExpertPanel, wxID_ANY, wxT("\tDefocus Step (Å) :"), wxDefaultPosition, wxDefaultSize, 0 );
	DefocusStepStaticText->Wrap( -1 );
	DefocusStepStaticText->Enable( false );

	fgSizer1->Add( DefocusStepStaticText, 0, wxALIGN_CENTER_VERTICAL|wxALL, 5 );

	DefocusSearchStepNumericCtrl = new NumericTextCtrl( ExpertPanel, wxID_ANY, wxT("200"), wxDefaultPosition, wxDefaultSize, wxTE_PROCESS_ENTER );
	DefocusSearchStepNumericCtrl->Enable( false );

	fgSizer1->Add( DefocusSearchStepNumericCtrl, 0, wxALL|wxEXPAND, 5 );

	m_staticText699 = new wxStaticText( ExpertPanel, wxID_ANY, wxT("Perform Pixel Size Search? "), wxDefaultPosition, wxDefaultSize, 0 );
	m_staticText699->Wrap( -1 );
	fgSizer1->Add( m_staticText699, 0, wxALIGN_CENTER_VERTICAL|wxALL, 5 );

	wxBoxSizer* bSizer2651;
	bSizer2651 = new wxBoxSizer( wxHORIZONTAL );

	PixelSizeSearchYesRadio = new wxRadioButton( ExpertPanel, wxID_ANY, wxT("Yes"), wxDefaultPosition, wxDefaultSize, wxRB_GROUP );
	bSizer2651->Add( PixelSizeSearchYesRadio, 0, wxALL, 5 );

	PixelSizeSearchNoRadio = new wxRadioButton( ExpertPanel, wxID_ANY, wxT("No"), wxDefaultPosition, wxDefaultSize, 0 );
	bSizer2651->Add( PixelSizeSearchNoRadio, 0, wxALL, 5 );


	fgSizer1->Add( bSizer2651, 1, wxEXPAND, 5 );

	PixelSizeRangeStaticText = new wxStaticText( ExpertPanel, wxID_ANY, wxT("\tPixel Size Range (Å) :"), wxDefaultPosition, wxDefaultSize, 0 );
	PixelSizeRangeStaticText->Wrap( -1 );
	PixelSizeRangeStaticText->Enable( false );

	fgSizer1->Add( PixelSizeRangeStaticText, 0, wxALIGN_CENTER_VERTICAL|wxALL, 5 );

	PixelSizeSearchRangeNumericCtrl = new NumericTextCtrl( ExpertPanel, wxID_ANY, wxT("0.05"), wxDefaultPosition, wxDefaultSize, wxTE_PROCESS_ENTER );
	PixelSizeSearchRangeNumericCtrl->Enable( false );

	fgSizer1->Add( PixelSizeSearchRangeNumericCtrl, 0, wxALL|wxEXPAND, 5 );

	PixelSizeStepStaticText = new wxStaticText( ExpertPanel, wxID_ANY, wxT("\tPixel Size Step (Å) :"), wxDefaultPosition, wxDefaultSize, 0 );
	PixelSizeStepStaticText->Wrap( -1 );
	PixelSizeStepStaticText->Enable( false );

	fgSizer1->Add( PixelSizeStepStaticText, 0, wxALIGN_CENTER_VERTICAL|wxALL, 5 );

	PixelSizeSearchStepNumericCtrl = new NumericTextCtrl( ExpertPanel, wxID_ANY, wxT("0.01"), wxDefaultPosition, wxDefaultSize, wxTE_PROCESS_ENTER );
	PixelSizeSearchStepNumericCtrl->Enable( false );

	fgSizer1->Add( PixelSizeSearchStepNumericCtrl, 0, wxALL|wxEXPAND, 5 );

	m_staticText857 = new wxStaticText( ExpertPanel, wxID_ANY, wxT("Peak Selection"), wxDefaultPosition, wxDefaultSize, 0 );
	m_staticText857->Wrap( -1 );
	m_staticText857->SetFont( wxFont( 10, wxFONTFAMILY_SWISS, wxFONTSTYLE_NORMAL, wxFONTWEIGHT_BOLD, true, wxT("Sans") ) );

	fgSizer1->Add( m_staticText857, 0, wxALIGN_BOTTOM|wxALL, 5 );


	fgSizer1->Add( 0, 0, 1, wxEXPAND, 5 );

	m_staticText849 = new wxStaticText( ExpertPanel, wxID_ANY, wxT("Minimum peak radius (px.) :"), wxDefaultPosition, wxDefaultSize, 0 );
	m_staticText849->Wrap( -1 );
	fgSizer1->Add( m_staticText849, 0, wxALIGN_CENTER_VERTICAL|wxALL, 5 );

	MinPeakRadiusNumericCtrl = new NumericTextCtrl( ExpertPanel, wxID_ANY, wxT("2.5"), wxDefaultPosition, wxDefaultSize, wxTE_PROCESS_ENTER );
	fgSizer1->Add( MinPeakRadiusNumericCtrl, 0, wxALL|wxEXPAND, 5 );

m_staticText8571 = new wxStaticText( ExpertPanel, wxID_ANY, wxT("Gpu Configuration"), wxDefaultPosition, wxDefaultSize, 0 );
	m_staticText8571->Wrap( -1 );
	m_staticText8571->SetFont( wxFont( 10, wxFONTFAMILY_SWISS, wxFONTSTYLE_NORMAL, wxFONTWEIGHT_BOLD, true, wxT("Sans") ) );

	fgSizer1->Add( m_staticText8571, 0, wxALL, 5 );


	fgSizer1->Add( 0, 0, 1, wxEXPAND, 5 );

	m_staticText6991 = new wxStaticText( ExpertPanel, wxID_ANY, wxT("Use GPU?"), wxDefaultPosition, wxDefaultSize, 0 );
	m_staticText6991->Wrap( -1 );
	fgSizer1->Add( m_staticText6991, 0, wxALL, 5 );

	wxBoxSizer* bSizer26513;
	bSizer26513 = new wxBoxSizer( wxHORIZONTAL );

	UseGPURadioYes = new wxRadioButton( ExpertPanel, wxID_ANY, wxT("Yes"), wxDefaultPosition, wxDefaultSize, wxRB_GROUP );
	bSizer26513->Add( UseGPURadioYes, 0, wxALL, 5 );

	UseGPURadioNo = new wxRadioButton( ExpertPanel, wxID_ANY, wxT("No"), wxDefaultPosition, wxDefaultSize, 0 );
	bSizer26513->Add( UseGPURadioNo, 0, wxALL, 5 );


	fgSizer1->Add( bSizer26513, 1, wxEXPAND, 5 );

	m_staticText69911 = new wxStaticText( ExpertPanel, wxID_ANY, wxT("Use FastFFT library?"), wxDefaultPosition, wxDefaultSize, 0 );
	m_staticText69911->Wrap( -1 );
	fgSizer1->Add( m_staticText69911, 0, wxALL, 5 );

	wxBoxSizer* bSizer26512;
	bSizer26512 = new wxBoxSizer( wxHORIZONTAL );

	UseFastFFTRadioYes = new wxRadioButton( ExpertPanel, wxID_ANY, wxT("Yes"), wxDefaultPosition, wxDefaultSize, wxRB_GROUP );
	bSizer26512->Add( UseFastFFTRadioYes, 0, wxALL, 5 );

	UseFastFFTRadioNo = new wxRadioButton( ExpertPanel, wxID_ANY, wxT("No"), wxDefaultPosition, wxDefaultSize, 0 );
	bSizer26512->Add( UseFastFFTRadioNo, 0, wxALL, 5 );


	fgSizer1->Add( bSizer26512, 1, wxEXPAND, 5 );


	wxBoxSizer* bSizer26511;
	bSizer26511 = new wxBoxSizer( wxHORIZONTAL );




	fgSizer1->Add( bSizer26511, 1, wxEXPAND, 5 );


	InputSizer->Add( fgSizer1, 1, wxEXPAND, 5 );


	ExpertPanel->SetSizer( InputSizer );
	ExpertPanel->Layout();
	InputSizer->Fit( ExpertPanel );
	bSizer46->Add( ExpertPanel, 0, wxALL|wxEXPAND, 5 );

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

	InfoPanel = new wxPanel( this, wxID_ANY, wxDefaultPosition, wxDefaultSize, wxTAB_TRAVERSAL );
	wxBoxSizer* bSizer61;
	bSizer61 = new wxBoxSizer( wxVERTICAL );

	InfoText = new wxRichTextCtrl( InfoPanel, wxID_ANY, wxEmptyString, wxDefaultPosition, wxDefaultSize, wxTE_READONLY|wxHSCROLL|wxVSCROLL );
	bSizer61->Add( InfoText, 1, wxEXPAND | wxALL, 5 );


	InfoPanel->SetSizer( bSizer61 );
	InfoPanel->Layout();
	bSizer61->Fit( InfoPanel );
	bSizer46->Add( InfoPanel, 1, wxEXPAND | wxALL, 5 );

	ResultsPanel = new ShowTemplateMatchResultsPanel( this, wxID_ANY, wxDefaultPosition, wxDefaultSize, wxTAB_TRAVERSAL );
	ResultsPanel->Hide();

	bSizer46->Add( ResultsPanel, 80, wxEXPAND | wxALL, 5 );


	bSizer43->Add( bSizer46, 1, wxEXPAND, 5 );

	m_staticline11 = new wxStaticLine( this, wxID_ANY, wxDefaultPosition, wxDefaultSize, wxLI_HORIZONTAL );
	bSizer43->Add( m_staticline11, 0, wxEXPAND | wxALL, 5 );

	wxBoxSizer* bSizer48;
	bSizer48 = new wxBoxSizer( wxHORIZONTAL );

	wxBoxSizer* bSizer70;
	bSizer70 = new wxBoxSizer( wxHORIZONTAL );

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
	bSizer58->Add( RunProfileComboBox, 50, wxALIGN_CENTER_VERTICAL|wxALL, 5 );

	wxBoxSizer* bSizer60;
	bSizer60 = new wxBoxSizer( wxVERTICAL );

	StartEstimationButton = new wxButton( StartPanel, wxID_ANY, wxT("Start Search"), wxDefaultPosition, wxDefaultSize, 0 );
	bSizer60->Add( StartEstimationButton, 0, wxALL, 5 );


	bSizer58->Add( bSizer60, 50, wxEXPAND, 5 );

	ResumeRunCheckBox = new wxCheckBox( StartPanel, wxID_ANY, wxT("Resume Run"), wxDefaultPosition, wxDefaultSize, 0 );
	ResumeRunCheckBox->Enable( false );

	bSizer58->Add( ResumeRunCheckBox, 0, wxALL|wxEXPAND, 5 );


	StartPanel->SetSizer( bSizer58 );
	StartPanel->Layout();
	bSizer58->Fit( StartPanel );
	bSizer48->Add( StartPanel, 1, wxEXPAND | wxALL, 5 );


	bSizer43->Add( bSizer48, 0, wxEXPAND, 5 );


	this->SetSizer( bSizer43 );
	this->Layout();

	// Connect Events
	this->Connect( wxEVT_UPDATE_UI, wxUpdateUIEventHandler( MatchTemplatePanelParent::OnUpdateUI ) );
	ResetAllDefaultsButton->Connect( wxEVT_COMMAND_BUTTON_CLICKED, wxCommandEventHandler( MatchTemplatePanelParent::ResetAllDefaultsClick ), NULL, this );
	InfoText->Connect( wxEVT_COMMAND_TEXT_URL, wxTextUrlEventHandler( MatchTemplatePanelParent::OnInfoURL ), NULL, this );
	FinishButton->Connect( wxEVT_COMMAND_BUTTON_CLICKED, wxCommandEventHandler( MatchTemplatePanelParent::FinishButtonClick ), NULL, this );
	CancelAlignmentButton->Connect( wxEVT_COMMAND_BUTTON_CLICKED, wxCommandEventHandler( MatchTemplatePanelParent::TerminateButtonClick ), NULL, this );
	StartEstimationButton->Connect( wxEVT_COMMAND_BUTTON_CLICKED, wxCommandEventHandler( MatchTemplatePanelParent::StartEstimationClick ), NULL, this );
	ResumeRunCheckBox->Connect( wxEVT_COMMAND_CHECKBOX_CLICKED, wxCommandEventHandler( MatchTemplatePanelParent::ResumeRunCheckBoxOnCheckBox ), NULL, this );
}

MatchTemplatePanelParent::~MatchTemplatePanelParent()
{
	// Disconnect Events
	this->Disconnect( wxEVT_UPDATE_UI, wxUpdateUIEventHandler( MatchTemplatePanelParent::OnUpdateUI ) );
	ResetAllDefaultsButton->Disconnect( wxEVT_COMMAND_BUTTON_CLICKED, wxCommandEventHandler( MatchTemplatePanelParent::ResetAllDefaultsClick ), NULL, this );
	InfoText->Disconnect( wxEVT_COMMAND_TEXT_URL, wxTextUrlEventHandler( MatchTemplatePanelParent::OnInfoURL ), NULL, this );
	FinishButton->Disconnect( wxEVT_COMMAND_BUTTON_CLICKED, wxCommandEventHandler( MatchTemplatePanelParent::FinishButtonClick ), NULL, this );
	CancelAlignmentButton->Disconnect( wxEVT_COMMAND_BUTTON_CLICKED, wxCommandEventHandler( MatchTemplatePanelParent::TerminateButtonClick ), NULL, this );
	StartEstimationButton->Disconnect( wxEVT_COMMAND_BUTTON_CLICKED, wxCommandEventHandler( MatchTemplatePanelParent::StartEstimationClick ), NULL, this );
	ResumeRunCheckBox->Disconnect( wxEVT_COMMAND_CHECKBOX_CLICKED, wxCommandEventHandler( MatchTemplatePanelParent::ResumeRunCheckBoxOnCheckBox ), NULL, this );

}

RefineTemplatePanelParent::RefineTemplatePanelParent( wxWindow* parent, wxWindowID id, const wxPoint& pos, const wxSize& size, long style, const wxString& name ) : JobPanel( parent, id, pos, size, style, name )
{
	wxBoxSizer* bSizer43;
	bSizer43 = new wxBoxSizer( wxVERTICAL );

	m_staticline149 = new wxStaticLine( this, wxID_ANY, wxDefaultPosition, wxDefaultSize, wxLI_HORIZONTAL );
	bSizer43->Add( m_staticline149, 0, wxEXPAND | wxALL, 5 );

	InputPanel = new wxPanel( this, wxID_ANY, wxDefaultPosition, wxDefaultSize, wxTAB_TRAVERSAL );
	wxBoxSizer* bSizer534;
	bSizer534 = new wxBoxSizer( wxVERTICAL );

	wxBoxSizer* bSizer45;
	bSizer45 = new wxBoxSizer( wxHORIZONTAL );

	wxBoxSizer* bSizer557;
	bSizer557 = new wxBoxSizer( wxHORIZONTAL );

	wxFlexGridSizer* fgSizer15;
	fgSizer15 = new wxFlexGridSizer( 0, 2, 0, 0 );
	fgSizer15->SetFlexibleDirection( wxHORIZONTAL );
	fgSizer15->SetNonFlexibleGrowMode( wxFLEX_GROWMODE_SPECIFIED );

	m_staticText262 = new wxStaticText( InputPanel, wxID_ANY, wxT("Input Image Group :"), wxDefaultPosition, wxSize( -1,-1 ), 0 );
	m_staticText262->Wrap( -1 );
	fgSizer15->Add( m_staticText262, 0, wxALIGN_CENTER_VERTICAL|wxALIGN_RIGHT|wxALL, 5 );

	GroupComboBox = new ImageGroupPickerComboPanel( InputPanel, wxID_ANY, wxDefaultPosition, wxDefaultSize, wxTAB_TRAVERSAL );
	GroupComboBox->SetMinSize( wxSize( 350,-1 ) );
	GroupComboBox->SetMaxSize( wxSize( 350,-1 ) );

	fgSizer15->Add( GroupComboBox, 1, wxEXPAND | wxALL, 5 );

	m_staticText478 = new wxStaticText( InputPanel, wxID_ANY, wxT("Reference Volume :"), wxDefaultPosition, wxDefaultSize, 0 );
	m_staticText478->Wrap( -1 );
	fgSizer15->Add( m_staticText478, 0, wxALIGN_CENTER_VERTICAL|wxALIGN_RIGHT|wxALL, 5 );

	ReferenceSelectPanel = new VolumeAssetPickerComboPanel( InputPanel, wxID_ANY, wxDefaultPosition, wxDefaultSize, wxTAB_TRAVERSAL );
	ReferenceSelectPanel->SetMinSize( wxSize( 350,-1 ) );
	ReferenceSelectPanel->SetMaxSize( wxSize( 350,-1 ) );

	fgSizer15->Add( ReferenceSelectPanel, 100, wxALIGN_CENTER_VERTICAL|wxALL, 5 );


	bSizer557->Add( fgSizer15, 0, wxEXPAND, 5 );


	bSizer45->Add( bSizer557, 1, wxEXPAND, 5 );


	bSizer534->Add( bSizer45, 1, wxEXPAND, 5 );

	InputErrorText = new wxStaticText( InputPanel, wxID_ANY, wxT("Please run Match Template on all images in this group before running refine."), wxDefaultPosition, wxDefaultSize, 0 );
	InputErrorText->Wrap( -1 );
	InputErrorText->SetFont( wxFont( wxNORMAL_FONT->GetPointSize(), wxFONTFAMILY_DEFAULT, wxFONTSTYLE_NORMAL, wxFONTWEIGHT_BOLD, false, wxEmptyString ) );
	InputErrorText->SetForegroundColour( wxColour( 180, 0, 0 ) );

	bSizer534->Add( InputErrorText, 0, wxALL, 5 );

	m_staticline151 = new wxStaticLine( InputPanel, wxID_ANY, wxDefaultPosition, wxDefaultSize, wxLI_HORIZONTAL );
	bSizer534->Add( m_staticline151, 0, wxEXPAND | wxALL, 5 );


	InputPanel->SetSizer( bSizer534 );
	InputPanel->Layout();
	bSizer534->Fit( InputPanel );
	bSizer43->Add( InputPanel, 0, wxEXPAND | wxALL, 5 );

	wxBoxSizer* bSizer46;
	bSizer46 = new wxBoxSizer( wxHORIZONTAL );

	ExpertPanel = new wxScrolledWindow( this, wxID_ANY, wxDefaultPosition, wxSize( -1,-1 ), wxVSCROLL );
	ExpertPanel->SetScrollRate( 5, 5 );
	InputSizer = new wxBoxSizer( wxVERTICAL );

	wxFlexGridSizer* fgSizer1;
	fgSizer1 = new wxFlexGridSizer( 0, 2, 0, 0 );
	fgSizer1->SetFlexibleDirection( wxBOTH );
	fgSizer1->SetNonFlexibleGrowMode( wxFLEX_GROWMODE_SPECIFIED );

	m_staticText847 = new wxStaticText( ExpertPanel, wxID_ANY, wxT("Peak Selection"), wxDefaultPosition, wxDefaultSize, 0 );
	m_staticText847->Wrap( -1 );
	m_staticText847->SetFont( wxFont( 10, wxFONTFAMILY_SWISS, wxFONTSTYLE_NORMAL, wxFONTWEIGHT_BOLD, true, wxT("Sans") ) );

	fgSizer1->Add( m_staticText847, 0, wxALIGN_BOTTOM|wxALL, 5 );

	ResetAllDefaultsButton = new wxButton( ExpertPanel, wxID_ANY, wxT("Reset All Defaults"), wxDefaultPosition, wxDefaultSize, 0 );
	fgSizer1->Add( ResetAllDefaultsButton, 0, wxALIGN_RIGHT|wxALL, 5 );

	m_staticText849 = new wxStaticText( ExpertPanel, wxID_ANY, wxT("Minimum peak radius (px.) :"), wxDefaultPosition, wxDefaultSize, 0 );
	m_staticText849->Wrap( -1 );
	fgSizer1->Add( m_staticText849, 0, wxALIGN_CENTER_VERTICAL|wxALL, 5 );

	MinPeakRadiusNumericCtrl = new NumericTextCtrl( ExpertPanel, wxID_ANY, wxT("2.5"), wxDefaultPosition, wxDefaultSize, wxTE_PROCESS_ENTER );
	fgSizer1->Add( MinPeakRadiusNumericCtrl, 0, wxALL|wxEXPAND, 5 );

	m_staticText846 = new wxStaticText( ExpertPanel, wxID_ANY, wxT("Threshold for Peak Selection : "), wxDefaultPosition, wxDefaultSize, 0 );
	m_staticText846->Wrap( -1 );
	fgSizer1->Add( m_staticText846, 0, wxALIGN_CENTER_VERTICAL|wxALL, 5 );

	PeakSelectionThresholdNumericCtrl = new NumericTextCtrl( ExpertPanel, wxID_ANY, wxT("2.5"), wxDefaultPosition, wxDefaultSize, wxTE_PROCESS_ENTER );
	fgSizer1->Add( PeakSelectionThresholdNumericCtrl, 0, wxALL|wxEXPAND, 5 );

	m_staticText848 = new wxStaticText( ExpertPanel, wxID_ANY, wxT("Threshold for Results :"), wxDefaultPosition, wxDefaultSize, 0 );
	m_staticText848->Wrap( -1 );
	fgSizer1->Add( m_staticText848, 0, wxALIGN_CENTER_VERTICAL|wxALL, 5 );

	PeakPlottingThresholdNumericCtrl = new NumericTextCtrl( ExpertPanel, wxID_ANY, wxT("2.5"), wxDefaultPosition, wxDefaultSize, wxTE_PROCESS_ENTER );
	fgSizer1->Add( PeakPlottingThresholdNumericCtrl, 0, wxALL|wxEXPAND, 5 );

	mask_radius = new wxStaticText( ExpertPanel, wxID_ANY, wxT("Remove Highly Shifted Peaks?"), wxDefaultPosition, wxDefaultSize, 0 );
	mask_radius->Wrap( -1 );
	fgSizer1->Add( mask_radius, 0, wxALIGN_CENTER_VERTICAL|wxALL, 5 );

	wxBoxSizer* bSizer2652;
	bSizer2652 = new wxBoxSizer( wxHORIZONTAL );

	RemoveShiftedPeaksYesRadio = new wxRadioButton( ExpertPanel, wxID_ANY, wxT("Yes"), wxDefaultPosition, wxDefaultSize, wxRB_GROUP );
	bSizer2652->Add( RemoveShiftedPeaksYesRadio, 0, wxALL, 5 );

	RemoveShiftedPeaksNoRadio = new wxRadioButton( ExpertPanel, wxID_ANY, wxT("No"), wxDefaultPosition, wxDefaultSize, 0 );
	bSizer2652->Add( RemoveShiftedPeaksNoRadio, 0, wxALL, 5 );


	fgSizer1->Add( bSizer2652, 1, wxEXPAND, 5 );

	ShiftThresholdStaticText = new wxStaticText( ExpertPanel, wxID_ANY, wxT("\tShift Threshold (Å) :"), wxDefaultPosition, wxDefaultSize, 0 );
	ShiftThresholdStaticText->Wrap( -1 );
	ShiftThresholdStaticText->Enable( false );

	fgSizer1->Add( ShiftThresholdStaticText, 0, wxALIGN_CENTER_VERTICAL|wxALL, 5 );

	PeakChangeThresholdNumericTextCtrl = new NumericTextCtrl( ExpertPanel, wxID_ANY, wxT("2.5"), wxDefaultPosition, wxDefaultSize, wxTE_PROCESS_ENTER );
	PeakChangeThresholdNumericTextCtrl->Enable( false );

	fgSizer1->Add( PeakChangeThresholdNumericTextCtrl, 0, wxALL|wxEXPAND, 5 );

	m_staticText201 = new wxStaticText( ExpertPanel, wxID_ANY, wxT("Search Limits"), wxDefaultPosition, wxDefaultSize, 0 );
	m_staticText201->Wrap( -1 );
	m_staticText201->SetFont( wxFont( wxNORMAL_FONT->GetPointSize(), wxFONTFAMILY_DEFAULT, wxFONTSTYLE_NORMAL, wxFONTWEIGHT_BOLD, true, wxEmptyString ) );

	fgSizer1->Add( m_staticText201, 0, wxALIGN_BOTTOM|wxALL, 5 );


	fgSizer1->Add( 0, 0, 1, wxEXPAND, 5 );

	m_staticText852 = new wxStaticText( ExpertPanel, wxID_ANY, wxT("Mask Radius (Å) :"), wxDefaultPosition, wxDefaultSize, 0 );
	m_staticText852->Wrap( -1 );
	fgSizer1->Add( m_staticText852, 0, wxALIGN_CENTER_VERTICAL|wxALL, 5 );

	MaskRadiusNumericTextCtrl = new NumericTextCtrl( ExpertPanel, wxID_ANY, wxT("2.5"), wxDefaultPosition, wxDefaultSize, wxTE_PROCESS_ENTER );
	fgSizer1->Add( MaskRadiusNumericTextCtrl, 0, wxALL|wxEXPAND, 5 );

	m_staticText189 = new wxStaticText( ExpertPanel, wxID_ANY, wxT("Out of Plane Angular Step (°) : "), wxDefaultPosition, wxDefaultSize, 0 );
	m_staticText189->Wrap( -1 );
	fgSizer1->Add( m_staticText189, 0, wxALIGN_CENTER_VERTICAL|wxALL, 5 );

	OutofPlaneStepNumericCtrl = new NumericTextCtrl( ExpertPanel, wxID_ANY, wxT("2.5"), wxDefaultPosition, wxDefaultSize, wxTE_PROCESS_ENTER );
	fgSizer1->Add( OutofPlaneStepNumericCtrl, 0, wxALL|wxEXPAND, 5 );

	m_staticText190 = new wxStaticText( ExpertPanel, wxID_ANY, wxT("In Plane Angular Step (°) :"), wxDefaultPosition, wxDefaultSize, 0 );
	m_staticText190->Wrap( -1 );
	fgSizer1->Add( m_staticText190, 0, wxALIGN_CENTER_VERTICAL|wxALL, 5 );

	InPlaneStepNumericCtrl = new NumericTextCtrl( ExpertPanel, wxID_ANY, wxT("1.5"), wxDefaultPosition, wxDefaultSize, wxTE_PROCESS_ENTER );
	fgSizer1->Add( InPlaneStepNumericCtrl, 0, wxALL|wxEXPAND, 5 );

	m_staticText190211 = new wxStaticText( ExpertPanel, wxID_ANY, wxT("High-Resolution Limit (Å) :"), wxDefaultPosition, wxDefaultSize, 0 );
	m_staticText190211->Wrap( -1 );
	fgSizer1->Add( m_staticText190211, 0, wxALIGN_CENTER_VERTICAL|wxALL, 5 );

	HighResolutionLimitNumericCtrl = new NumericTextCtrl( ExpertPanel, wxID_ANY, wxT("2.0"), wxDefaultPosition, wxDefaultSize, wxTE_PROCESS_ENTER );
	fgSizer1->Add( HighResolutionLimitNumericCtrl, 0, wxALL|wxEXPAND, 5 );

	m_staticText19021 = new wxStaticText( ExpertPanel, wxID_ANY, wxT("Pointgroup Symmetry :"), wxDefaultPosition, wxDefaultSize, 0 );
	m_staticText19021->Wrap( -1 );
	fgSizer1->Add( m_staticText19021, 0, wxALIGN_CENTER_VERTICAL|wxALL, 5 );

	SymmetryComboBox = new wxComboBox( ExpertPanel, wxID_ANY, wxT("C1"), wxDefaultPosition, wxDefaultSize, 0, NULL, 0 );
	fgSizer1->Add( SymmetryComboBox, 0, wxALL|wxEXPAND, 5 );

	m_staticText698 = new wxStaticText( ExpertPanel, wxID_ANY, wxT("Perform Defocus Search? "), wxDefaultPosition, wxDefaultSize, 0 );
	m_staticText698->Wrap( -1 );
	fgSizer1->Add( m_staticText698, 0, wxALIGN_CENTER_VERTICAL|wxALL, 5 );

	wxBoxSizer* bSizer265;
	bSizer265 = new wxBoxSizer( wxHORIZONTAL );

	DefocusSearchYesRadio = new wxRadioButton( ExpertPanel, wxID_ANY, wxT("Yes"), wxDefaultPosition, wxDefaultSize, wxRB_GROUP );
	bSizer265->Add( DefocusSearchYesRadio, 0, wxALL, 5 );

	DefocusSearchNoRadio = new wxRadioButton( ExpertPanel, wxID_ANY, wxT("No"), wxDefaultPosition, wxDefaultSize, 0 );
	bSizer265->Add( DefocusSearchNoRadio, 0, wxALL, 5 );


	fgSizer1->Add( bSizer265, 1, wxEXPAND, 5 );

	DefocusRangeStaticText = new wxStaticText( ExpertPanel, wxID_ANY, wxT("\tDefocus Range (Å) :"), wxDefaultPosition, wxDefaultSize, 0 );
	DefocusRangeStaticText->Wrap( -1 );
	DefocusRangeStaticText->Enable( false );

	fgSizer1->Add( DefocusRangeStaticText, 0, wxALIGN_CENTER_VERTICAL|wxALL, 5 );

	DefocusSearchRangeNumericCtrl = new NumericTextCtrl( ExpertPanel, wxID_ANY, wxT("1200"), wxDefaultPosition, wxDefaultSize, wxTE_PROCESS_ENTER );
	DefocusSearchRangeNumericCtrl->Enable( false );

	fgSizer1->Add( DefocusSearchRangeNumericCtrl, 0, wxALL|wxEXPAND, 5 );

	DefocusStepStaticText = new wxStaticText( ExpertPanel, wxID_ANY, wxT("\tDefocus Step (Å) :"), wxDefaultPosition, wxDefaultSize, 0 );
	DefocusStepStaticText->Wrap( -1 );
	DefocusStepStaticText->Enable( false );

	fgSizer1->Add( DefocusStepStaticText, 0, wxALIGN_CENTER_VERTICAL|wxALL, 5 );

	DefocusSearchStepNumericCtrl = new NumericTextCtrl( ExpertPanel, wxID_ANY, wxT("200"), wxDefaultPosition, wxDefaultSize, wxTE_PROCESS_ENTER );
	DefocusSearchStepNumericCtrl->Enable( false );

	fgSizer1->Add( DefocusSearchStepNumericCtrl, 0, wxALL|wxEXPAND, 5 );

	m_staticText699 = new wxStaticText( ExpertPanel, wxID_ANY, wxT("Perform Pixel Size Search? "), wxDefaultPosition, wxDefaultSize, 0 );
	m_staticText699->Wrap( -1 );
	fgSizer1->Add( m_staticText699, 0, wxALIGN_CENTER_VERTICAL|wxALL, 5 );

	wxBoxSizer* bSizer2651;
	bSizer2651 = new wxBoxSizer( wxHORIZONTAL );

	PixelSizeSearchYesRadio = new wxRadioButton( ExpertPanel, wxID_ANY, wxT("Yes"), wxDefaultPosition, wxDefaultSize, wxRB_GROUP );
	bSizer2651->Add( PixelSizeSearchYesRadio, 0, wxALL, 5 );

	PixelSizeSearchNoRadio = new wxRadioButton( ExpertPanel, wxID_ANY, wxT("No"), wxDefaultPosition, wxDefaultSize, 0 );
	bSizer2651->Add( PixelSizeSearchNoRadio, 0, wxALL, 5 );


	fgSizer1->Add( bSizer2651, 1, wxEXPAND, 5 );

	PixelSizeRangeStaticText = new wxStaticText( ExpertPanel, wxID_ANY, wxT("\tPixel Size Range (Å) :"), wxDefaultPosition, wxDefaultSize, 0 );
	PixelSizeRangeStaticText->Wrap( -1 );
	PixelSizeRangeStaticText->Enable( false );

	fgSizer1->Add( PixelSizeRangeStaticText, 0, wxALIGN_CENTER_VERTICAL|wxALL, 5 );

	PixelSizeSearchRangeNumericCtrl = new NumericTextCtrl( ExpertPanel, wxID_ANY, wxT("0.05"), wxDefaultPosition, wxDefaultSize, wxTE_PROCESS_ENTER );
	PixelSizeSearchRangeNumericCtrl->Enable( false );

	fgSizer1->Add( PixelSizeSearchRangeNumericCtrl, 0, wxALL|wxEXPAND, 5 );

	PixelSizeStepStaticText = new wxStaticText( ExpertPanel, wxID_ANY, wxT("\tPixel Size Step (Å) :"), wxDefaultPosition, wxDefaultSize, 0 );
	PixelSizeStepStaticText->Wrap( -1 );
	PixelSizeStepStaticText->Enable( false );

	fgSizer1->Add( PixelSizeStepStaticText, 0, wxALIGN_CENTER_VERTICAL|wxALL, 5 );

	PixelSizeSearchStepNumericCtrl = new NumericTextCtrl( ExpertPanel, wxID_ANY, wxT("0.01"), wxDefaultPosition, wxDefaultSize, wxTE_PROCESS_ENTER );
	PixelSizeSearchStepNumericCtrl->Enable( false );

	fgSizer1->Add( PixelSizeSearchStepNumericCtrl, 0, wxALL|wxEXPAND, 5 );


	InputSizer->Add( fgSizer1, 1, wxEXPAND, 5 );


	ExpertPanel->SetSizer( InputSizer );
	ExpertPanel->Layout();
	InputSizer->Fit( ExpertPanel );
	bSizer46->Add( ExpertPanel, 0, wxALL|wxEXPAND, 5 );

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

	InfoPanel = new wxPanel( this, wxID_ANY, wxDefaultPosition, wxDefaultSize, wxTAB_TRAVERSAL );
	wxBoxSizer* bSizer61;
	bSizer61 = new wxBoxSizer( wxVERTICAL );

	InfoText = new wxRichTextCtrl( InfoPanel, wxID_ANY, wxEmptyString, wxDefaultPosition, wxDefaultSize, wxTE_READONLY|wxHSCROLL|wxVSCROLL );
	bSizer61->Add( InfoText, 1, wxEXPAND | wxALL, 5 );


	InfoPanel->SetSizer( bSizer61 );
	InfoPanel->Layout();
	bSizer61->Fit( InfoPanel );
	bSizer46->Add( InfoPanel, 1, wxEXPAND | wxALL, 5 );

	ResultsPanel = new ShowTemplateMatchResultsPanel( this, wxID_ANY, wxDefaultPosition, wxDefaultSize, wxTAB_TRAVERSAL );
	ResultsPanel->Hide();

	bSizer46->Add( ResultsPanel, 80, wxEXPAND | wxALL, 5 );


	bSizer43->Add( bSizer46, 1, wxEXPAND, 5 );

	m_staticline11 = new wxStaticLine( this, wxID_ANY, wxDefaultPosition, wxDefaultSize, wxLI_HORIZONTAL );
	bSizer43->Add( m_staticline11, 0, wxEXPAND | wxALL, 5 );

	wxBoxSizer* bSizer48;
	bSizer48 = new wxBoxSizer( wxHORIZONTAL );

	wxBoxSizer* bSizer70;
	bSizer70 = new wxBoxSizer( wxHORIZONTAL );

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
	bSizer58->Add( RunProfileComboBox, 50, wxALIGN_CENTER_VERTICAL|wxALL, 5 );

	wxBoxSizer* bSizer60;
	bSizer60 = new wxBoxSizer( wxVERTICAL );

	StartEstimationButton = new wxButton( StartPanel, wxID_ANY, wxT("Start Search"), wxDefaultPosition, wxDefaultSize, 0 );
	bSizer60->Add( StartEstimationButton, 0, wxALL, 5 );


	bSizer58->Add( bSizer60, 50, wxEXPAND, 5 );


	StartPanel->SetSizer( bSizer58 );
	StartPanel->Layout();
	bSizer58->Fit( StartPanel );
	bSizer48->Add( StartPanel, 1, wxEXPAND | wxALL, 5 );


	bSizer43->Add( bSizer48, 0, wxEXPAND, 5 );


	this->SetSizer( bSizer43 );
	this->Layout();

	// Connect Events
	this->Connect( wxEVT_UPDATE_UI, wxUpdateUIEventHandler( RefineTemplatePanelParent::OnUpdateUI ) );
	ResetAllDefaultsButton->Connect( wxEVT_COMMAND_BUTTON_CLICKED, wxCommandEventHandler( RefineTemplatePanelParent::ResetAllDefaultsClick ), NULL, this );
	InfoText->Connect( wxEVT_COMMAND_TEXT_URL, wxTextUrlEventHandler( RefineTemplatePanelParent::OnInfoURL ), NULL, this );
	FinishButton->Connect( wxEVT_COMMAND_BUTTON_CLICKED, wxCommandEventHandler( RefineTemplatePanelParent::FinishButtonClick ), NULL, this );
	CancelAlignmentButton->Connect( wxEVT_COMMAND_BUTTON_CLICKED, wxCommandEventHandler( RefineTemplatePanelParent::TerminateButtonClick ), NULL, this );
	StartEstimationButton->Connect( wxEVT_COMMAND_BUTTON_CLICKED, wxCommandEventHandler( RefineTemplatePanelParent::StartEstimationClick ), NULL, this );
}

RefineTemplatePanelParent::~RefineTemplatePanelParent()
{
	// Disconnect Events
	this->Disconnect( wxEVT_UPDATE_UI, wxUpdateUIEventHandler( RefineTemplatePanelParent::OnUpdateUI ) );
	ResetAllDefaultsButton->Disconnect( wxEVT_COMMAND_BUTTON_CLICKED, wxCommandEventHandler( RefineTemplatePanelParent::ResetAllDefaultsClick ), NULL, this );
	InfoText->Disconnect( wxEVT_COMMAND_TEXT_URL, wxTextUrlEventHandler( RefineTemplatePanelParent::OnInfoURL ), NULL, this );
	FinishButton->Disconnect( wxEVT_COMMAND_BUTTON_CLICKED, wxCommandEventHandler( RefineTemplatePanelParent::FinishButtonClick ), NULL, this );
	CancelAlignmentButton->Disconnect( wxEVT_COMMAND_BUTTON_CLICKED, wxCommandEventHandler( RefineTemplatePanelParent::TerminateButtonClick ), NULL, this );
	StartEstimationButton->Disconnect( wxEVT_COMMAND_BUTTON_CLICKED, wxCommandEventHandler( RefineTemplatePanelParent::StartEstimationClick ), NULL, this );

}

RefineTemplateDevPanelParent::RefineTemplateDevPanelParent( wxWindow* parent, wxWindowID id, const wxPoint& pos, const wxSize& size, long style, const wxString& name ) : JobPanel( parent, id, pos, size, style, name )
{
	wxBoxSizer* bSizer43;
	bSizer43 = new wxBoxSizer( wxVERTICAL );

	m_staticline149 = new wxStaticLine( this, wxID_ANY, wxDefaultPosition, wxDefaultSize, wxLI_HORIZONTAL );
	bSizer43->Add( m_staticline149, 0, wxEXPAND | wxALL, 5 );

	InputPanel = new wxPanel( this, wxID_ANY, wxDefaultPosition, wxDefaultSize, wxTAB_TRAVERSAL );
	wxBoxSizer* bSizer534;
	bSizer534 = new wxBoxSizer( wxVERTICAL );

	wxBoxSizer* bSizer45;
	bSizer45 = new wxBoxSizer( wxHORIZONTAL );

	wxBoxSizer* bSizer557;
	bSizer557 = new wxBoxSizer( wxHORIZONTAL );

	wxFlexGridSizer* fgSizer15;
	fgSizer15 = new wxFlexGridSizer( 0, 2, 0, 0 );
	fgSizer15->SetFlexibleDirection( wxHORIZONTAL );
	fgSizer15->SetNonFlexibleGrowMode( wxFLEX_GROWMODE_SPECIFIED );

	m_staticText262 = new wxStaticText( InputPanel, wxID_ANY, wxT("Input Image Group :"), wxDefaultPosition, wxSize( -1,-1 ), 0 );
	m_staticText262->Wrap( -1 );
	fgSizer15->Add( m_staticText262, 0, wxALIGN_CENTER_VERTICAL|wxALIGN_RIGHT|wxALL, 5 );

	GroupComboBox = new ImageGroupPickerComboPanel( InputPanel, wxID_ANY, wxDefaultPosition, wxDefaultSize, wxTAB_TRAVERSAL );
	GroupComboBox->SetMinSize( wxSize( 350,-1 ) );
	GroupComboBox->SetMaxSize( wxSize( 350,-1 ) );

	fgSizer15->Add( GroupComboBox, 1, wxEXPAND | wxALL, 5 );

	m_staticText478 = new wxStaticText( InputPanel, wxID_ANY, wxT("Reference Volume :"), wxDefaultPosition, wxDefaultSize, 0 );
	m_staticText478->Wrap( -1 );
	fgSizer15->Add( m_staticText478, 0, wxALIGN_CENTER_VERTICAL|wxALIGN_RIGHT|wxALL, 5 );

	ReferenceSelectPanel = new VolumeAssetPickerComboPanel( InputPanel, wxID_ANY, wxDefaultPosition, wxDefaultSize, wxTAB_TRAVERSAL );
	ReferenceSelectPanel->SetMinSize( wxSize( 350,-1 ) );
	ReferenceSelectPanel->SetMaxSize( wxSize( 350,-1 ) );

	fgSizer15->Add( ReferenceSelectPanel, 100, wxALIGN_CENTER_VERTICAL|wxALL, 5 );


	bSizer557->Add( fgSizer15, 0, wxEXPAND, 5 );


	bSizer45->Add( bSizer557, 1, wxEXPAND, 5 );


	bSizer534->Add( bSizer45, 1, wxEXPAND, 5 );

	InputErrorText = new wxStaticText( InputPanel, wxID_ANY, wxT("Please run Match Template on all images in this group before running refine."), wxDefaultPosition, wxDefaultSize, 0 );
	InputErrorText->Wrap( -1 );
	InputErrorText->SetFont( wxFont( wxNORMAL_FONT->GetPointSize(), wxFONTFAMILY_DEFAULT, wxFONTSTYLE_NORMAL, wxFONTWEIGHT_BOLD, false, wxEmptyString ) );
	InputErrorText->SetForegroundColour( wxColour( 180, 0, 0 ) );

	bSizer534->Add( InputErrorText, 0, wxALL, 5 );

	m_staticline151 = new wxStaticLine( InputPanel, wxID_ANY, wxDefaultPosition, wxDefaultSize, wxLI_HORIZONTAL );
	bSizer534->Add( m_staticline151, 0, wxEXPAND | wxALL, 5 );


	InputPanel->SetSizer( bSizer534 );
	InputPanel->Layout();
	bSizer534->Fit( InputPanel );
	bSizer43->Add( InputPanel, 0, wxEXPAND | wxALL, 5 );

	wxBoxSizer* bSizer46;
	bSizer46 = new wxBoxSizer( wxHORIZONTAL );

	ExpertPanel = new wxScrolledWindow( this, wxID_ANY, wxDefaultPosition, wxSize( -1,-1 ), wxVSCROLL );
	ExpertPanel->SetScrollRate( 5, 5 );
	InputSizer = new wxBoxSizer( wxVERTICAL );

	wxFlexGridSizer* fgSizer1;
	fgSizer1 = new wxFlexGridSizer( 0, 2, 0, 0 );
	fgSizer1->SetFlexibleDirection( wxBOTH );
	fgSizer1->SetNonFlexibleGrowMode( wxFLEX_GROWMODE_SPECIFIED );

	m_staticText847 = new wxStaticText( ExpertPanel, wxID_ANY, wxT("Peak Selection"), wxDefaultPosition, wxDefaultSize, 0 );
	m_staticText847->Wrap( -1 );
	m_staticText847->SetFont( wxFont( 10, wxFONTFAMILY_SWISS, wxFONTSTYLE_NORMAL, wxFONTWEIGHT_BOLD, true, wxT("Sans") ) );

	fgSizer1->Add( m_staticText847, 0, wxALIGN_BOTTOM|wxALL, 5 );

	ResetAllDefaultsButton = new wxButton( ExpertPanel, wxID_ANY, wxT("Reset All Defaults"), wxDefaultPosition, wxDefaultSize, 0 );
	fgSizer1->Add( ResetAllDefaultsButton, 0, wxALIGN_RIGHT|wxALL, 5 );

	m_staticText849 = new wxStaticText( ExpertPanel, wxID_ANY, wxT("Minimum peak radius (px.) :"), wxDefaultPosition, wxDefaultSize, 0 );
	m_staticText849->Wrap( -1 );
	fgSizer1->Add( m_staticText849, 0, wxALIGN_CENTER_VERTICAL|wxALL, 5 );

	MinPeakRadiusNumericCtrl = new NumericTextCtrl( ExpertPanel, wxID_ANY, wxT("10"), wxDefaultPosition, wxDefaultSize, wxTE_PROCESS_ENTER );
	fgSizer1->Add( MinPeakRadiusNumericCtrl, 0, wxALL|wxEXPAND, 5 );

	m_staticText846 = new wxStaticText( ExpertPanel, wxID_ANY, wxT("Threshold for Peak Selection : "), wxDefaultPosition, wxDefaultSize, 0 );
	m_staticText846->Wrap( -1 );
	fgSizer1->Add( m_staticText846, 0, wxALIGN_CENTER_VERTICAL|wxALL, 5 );

	PeakSelectionThresholdNumericCtrl = new NumericTextCtrl( ExpertPanel, wxID_ANY, wxT("2.5"), wxDefaultPosition, wxDefaultSize, wxTE_PROCESS_ENTER );
	fgSizer1->Add( PeakSelectionThresholdNumericCtrl, 0, wxALL|wxEXPAND, 5 );

	m_staticText848 = new wxStaticText( ExpertPanel, wxID_ANY, wxT("Threshold for Results :"), wxDefaultPosition, wxDefaultSize, 0 );
	m_staticText848->Wrap( -1 );
	fgSizer1->Add( m_staticText848, 0, wxALIGN_CENTER_VERTICAL|wxALL, 5 );

	PeakPlottingThresholdNumericCtrl = new NumericTextCtrl( ExpertPanel, wxID_ANY, wxT("2.5"), wxDefaultPosition, wxDefaultSize, wxTE_PROCESS_ENTER );
	fgSizer1->Add( PeakPlottingThresholdNumericCtrl, 0, wxALL|wxEXPAND, 5 );

	mask_radius = new wxStaticText( ExpertPanel, wxID_ANY, wxT("Remove Highly Shifted Peaks?"), wxDefaultPosition, wxDefaultSize, 0 );
	mask_radius->Wrap( -1 );
	fgSizer1->Add( mask_radius, 0, wxALIGN_CENTER_VERTICAL|wxALL, 5 );

	wxBoxSizer* bSizer2652;
	bSizer2652 = new wxBoxSizer( wxHORIZONTAL );

	RemoveShiftedPeaksYesRadio = new wxRadioButton( ExpertPanel, wxID_ANY, wxT("Yes"), wxDefaultPosition, wxDefaultSize, wxRB_GROUP );
	bSizer2652->Add( RemoveShiftedPeaksYesRadio, 0, wxALL, 5 );

	RemoveShiftedPeaksNoRadio = new wxRadioButton( ExpertPanel, wxID_ANY, wxT("No"), wxDefaultPosition, wxDefaultSize, 0 );
	bSizer2652->Add( RemoveShiftedPeaksNoRadio, 0, wxALL, 5 );


	fgSizer1->Add( bSizer2652, 1, wxEXPAND, 5 );

	ShiftThresholdStaticText = new wxStaticText( ExpertPanel, wxID_ANY, wxT("\tShift Threshold (Å) :"), wxDefaultPosition, wxDefaultSize, 0 );
	ShiftThresholdStaticText->Wrap( -1 );
	ShiftThresholdStaticText->Enable( false );

	fgSizer1->Add( ShiftThresholdStaticText, 0, wxALIGN_CENTER_VERTICAL|wxALL, 5 );

	PeakChangeThresholdNumericTextCtrl = new NumericTextCtrl( ExpertPanel, wxID_ANY, wxT("2.5"), wxDefaultPosition, wxDefaultSize, wxTE_PROCESS_ENTER );
	PeakChangeThresholdNumericTextCtrl->Enable( false );

	fgSizer1->Add( PeakChangeThresholdNumericTextCtrl, 0, wxALL|wxEXPAND, 5 );

	m_staticText201 = new wxStaticText( ExpertPanel, wxID_ANY, wxT("Search Limits"), wxDefaultPosition, wxDefaultSize, 0 );
	m_staticText201->Wrap( -1 );
	m_staticText201->SetFont( wxFont( wxNORMAL_FONT->GetPointSize(), wxFONTFAMILY_DEFAULT, wxFONTSTYLE_NORMAL, wxFONTWEIGHT_BOLD, true, wxEmptyString ) );

	fgSizer1->Add( m_staticText201, 0, wxALIGN_BOTTOM|wxALL, 5 );


	fgSizer1->Add( 0, 0, 1, wxEXPAND, 5 );

	m_staticText852 = new wxStaticText( ExpertPanel, wxID_ANY, wxT("Mask Radius (Å) :"), wxDefaultPosition, wxDefaultSize, 0 );
	m_staticText852->Wrap( -1 );
	fgSizer1->Add( m_staticText852, 0, wxALIGN_CENTER_VERTICAL|wxALL, 5 );

	MaskRadiusNumericTextCtrl = new NumericTextCtrl( ExpertPanel, wxID_ANY, wxT("2.5"), wxDefaultPosition, wxDefaultSize, wxTE_PROCESS_ENTER );
	fgSizer1->Add( MaskRadiusNumericTextCtrl, 0, wxALL|wxEXPAND, 5 );

	m_staticText189 = new wxStaticText( ExpertPanel, wxID_ANY, wxT("Out of Plane Angular Step (°) : "), wxDefaultPosition, wxDefaultSize, 0 );
	m_staticText189->Wrap( -1 );
	fgSizer1->Add( m_staticText189, 0, wxALIGN_CENTER_VERTICAL|wxALL, 5 );

	OutofPlaneStepNumericCtrl = new NumericTextCtrl( ExpertPanel, wxID_ANY, wxT("2.5"), wxDefaultPosition, wxDefaultSize, wxTE_PROCESS_ENTER );
	fgSizer1->Add( OutofPlaneStepNumericCtrl, 0, wxALL|wxEXPAND, 5 );

	m_staticText190 = new wxStaticText( ExpertPanel, wxID_ANY, wxT("In Plane Angular Step (°) :"), wxDefaultPosition, wxDefaultSize, 0 );
	m_staticText190->Wrap( -1 );
	fgSizer1->Add( m_staticText190, 0, wxALIGN_CENTER_VERTICAL|wxALL, 5 );

	InPlaneStepNumericCtrl = new NumericTextCtrl( ExpertPanel, wxID_ANY, wxT("1.5"), wxDefaultPosition, wxDefaultSize, wxTE_PROCESS_ENTER );
	fgSizer1->Add( InPlaneStepNumericCtrl, 0, wxALL|wxEXPAND, 5 );

	m_staticText190211 = new wxStaticText( ExpertPanel, wxID_ANY, wxT("High-Resolution Limit (Å) :"), wxDefaultPosition, wxDefaultSize, 0 );
	m_staticText190211->Wrap( -1 );
	fgSizer1->Add( m_staticText190211, 0, wxALIGN_CENTER_VERTICAL|wxALL, 5 );

	HighResolutionLimitNumericCtrl = new NumericTextCtrl( ExpertPanel, wxID_ANY, wxT("2.0"), wxDefaultPosition, wxDefaultSize, wxTE_PROCESS_ENTER );
	fgSizer1->Add( HighResolutionLimitNumericCtrl, 0, wxALL|wxEXPAND, 5 );

	m_staticText698 = new wxStaticText( ExpertPanel, wxID_ANY, wxT("Perform Defocus Search? "), wxDefaultPosition, wxDefaultSize, 0 );
	m_staticText698->Wrap( -1 );
	fgSizer1->Add( m_staticText698, 0, wxALIGN_CENTER_VERTICAL|wxALL, 5 );

	wxBoxSizer* bSizer265;
	bSizer265 = new wxBoxSizer( wxHORIZONTAL );

	DefocusSearchYesRadio = new wxRadioButton( ExpertPanel, wxID_ANY, wxT("Yes"), wxDefaultPosition, wxDefaultSize, wxRB_GROUP );
	bSizer265->Add( DefocusSearchYesRadio, 0, wxALL, 5 );

	DefocusSearchNoRadio = new wxRadioButton( ExpertPanel, wxID_ANY, wxT("No"), wxDefaultPosition, wxDefaultSize, 0 );
	bSizer265->Add( DefocusSearchNoRadio, 0, wxALL, 5 );


	fgSizer1->Add( bSizer265, 1, wxEXPAND, 5 );

	DefocusRangeStaticText = new wxStaticText( ExpertPanel, wxID_ANY, wxT("\tDefocus Range (Å) :"), wxDefaultPosition, wxDefaultSize, 0 );
	DefocusRangeStaticText->Wrap( -1 );
	DefocusRangeStaticText->Enable( false );

	fgSizer1->Add( DefocusRangeStaticText, 0, wxALIGN_CENTER_VERTICAL|wxALL, 5 );

	DefocusSearchRangeNumericCtrl = new NumericTextCtrl( ExpertPanel, wxID_ANY, wxT("200"), wxDefaultPosition, wxDefaultSize, wxTE_PROCESS_ENTER );
	DefocusSearchRangeNumericCtrl->Enable( false );

	fgSizer1->Add( DefocusSearchRangeNumericCtrl, 0, wxALL|wxEXPAND, 5 );

	DefocusStepStaticText = new wxStaticText( ExpertPanel, wxID_ANY, wxT("\tDefocus Step (Å) :"), wxDefaultPosition, wxDefaultSize, 0 );
	DefocusStepStaticText->Wrap( -1 );
	DefocusStepStaticText->Enable( false );

	fgSizer1->Add( DefocusStepStaticText, 0, wxALIGN_CENTER_VERTICAL|wxALL, 5 );

	DefocusSearchStepNumericCtrl = new NumericTextCtrl( ExpertPanel, wxID_ANY, wxT("10"), wxDefaultPosition, wxDefaultSize, wxTE_PROCESS_ENTER );
	DefocusSearchStepNumericCtrl->Enable( false );

	fgSizer1->Add( DefocusSearchStepNumericCtrl, 0, wxALL|wxEXPAND, 5 );

	m_staticText699 = new wxStaticText( ExpertPanel, wxID_ANY, wxT("Refine Astigmatism?"), wxDefaultPosition, wxDefaultSize, 0 );
	m_staticText699->Wrap( -1 );
	fgSizer1->Add( m_staticText699, 0, wxALIGN_CENTER_VERTICAL|wxALL, 5 );

	wxBoxSizer* bSizer2651;
	bSizer2651 = new wxBoxSizer( wxHORIZONTAL );

	AstigmatismSearchYesRadio = new wxRadioButton( ExpertPanel, wxID_ANY, wxT("Yes"), wxDefaultPosition, wxDefaultSize, wxRB_GROUP );
	bSizer2651->Add( AstigmatismSearchYesRadio, 0, wxALL, 5 );

	AstigmatismSearchNoRadio = new wxRadioButton( ExpertPanel, wxID_ANY, wxT("No"), wxDefaultPosition, wxDefaultSize, 0 );
	AstigmatismSearchNoRadio->SetValue( true );
	bSizer2651->Add( AstigmatismSearchNoRadio, 0, wxALL, 5 );


	fgSizer1->Add( bSizer2651, 1, wxEXPAND, 5 );

	AstigmatismConstraint = new wxStaticText( ExpertPanel, wxID_ANY, wxT("\tConstrain by N-neighbors :"), wxDefaultPosition, wxDefaultSize, 0 );
	AstigmatismConstraint->Wrap( -1 );
	AstigmatismConstraint->Enable( false );

	fgSizer1->Add( AstigmatismConstraint, 0, wxALIGN_CENTER_VERTICAL|wxALL, 5 );

	PixelSizeSearchRangeNumericCtrl = new NumericTextCtrl( ExpertPanel, wxID_ANY, wxT("0.05"), wxDefaultPosition, wxDefaultSize, wxTE_PROCESS_ENTER );
	PixelSizeSearchRangeNumericCtrl->Enable( false );

	fgSizer1->Add( PixelSizeSearchRangeNumericCtrl, 0, wxALL|wxEXPAND, 5 );

	m_staticText6992 = new wxStaticText( ExpertPanel, wxID_ANY, wxT("Refine Beam Tilt?"), wxDefaultPosition, wxDefaultSize, 0 );
	m_staticText6992->Wrap( -1 );
	fgSizer1->Add( m_staticText6992, 0, wxALL, 5 );

	wxBoxSizer* bSizer26511;
	bSizer26511 = new wxBoxSizer( wxHORIZONTAL );

	BeamTiltSearchYesRadio1 = new wxRadioButton( ExpertPanel, wxID_ANY, wxT("Yes"), wxDefaultPosition, wxDefaultSize, wxRB_GROUP );
	bSizer26511->Add( BeamTiltSearchYesRadio1, 0, wxALL, 5 );

	BeamTiltSearchNoRadio1 = new wxRadioButton( ExpertPanel, wxID_ANY, wxT("No"), wxDefaultPosition, wxDefaultSize, 0 );
	BeamTiltSearchNoRadio1->SetValue( true );
	bSizer26511->Add( BeamTiltSearchNoRadio1, 0, wxALL, 5 );


	fgSizer1->Add( bSizer26511, 1, wxEXPAND, 5 );


	InputSizer->Add( fgSizer1, 1, wxEXPAND, 5 );


	ExpertPanel->SetSizer( InputSizer );
	ExpertPanel->Layout();
	InputSizer->Fit( ExpertPanel );
	bSizer46->Add( ExpertPanel, 0, wxALL|wxEXPAND, 5 );

	m_staticText6991 = new wxStaticText( this, wxID_ANY, wxT("Perform Pixel Size Search? "), wxDefaultPosition, wxDefaultSize, 0 );
	m_staticText6991->Wrap( -1 );
	bSizer46->Add( m_staticText6991, 0, wxALL, 5 );

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

	InfoPanel = new wxPanel( this, wxID_ANY, wxDefaultPosition, wxDefaultSize, wxTAB_TRAVERSAL );
	wxBoxSizer* bSizer61;
	bSizer61 = new wxBoxSizer( wxVERTICAL );

	InfoText = new wxRichTextCtrl( InfoPanel, wxID_ANY, wxEmptyString, wxDefaultPosition, wxDefaultSize, wxTE_READONLY|wxHSCROLL|wxVSCROLL );
	bSizer61->Add( InfoText, 1, wxEXPAND | wxALL, 5 );


	InfoPanel->SetSizer( bSizer61 );
	InfoPanel->Layout();
	bSizer61->Fit( InfoPanel );
	bSizer46->Add( InfoPanel, 1, wxEXPAND | wxALL, 5 );

	ResultsPanel = new ShowTemplateMatchResultsPanel( this, wxID_ANY, wxDefaultPosition, wxDefaultSize, wxTAB_TRAVERSAL );
	ResultsPanel->Hide();

	bSizer46->Add( ResultsPanel, 80, wxEXPAND | wxALL, 5 );


	bSizer43->Add( bSizer46, 1, wxEXPAND, 5 );

	m_staticline11 = new wxStaticLine( this, wxID_ANY, wxDefaultPosition, wxDefaultSize, wxLI_HORIZONTAL );
	bSizer43->Add( m_staticline11, 0, wxEXPAND | wxALL, 5 );

	wxBoxSizer* bSizer48;
	bSizer48 = new wxBoxSizer( wxHORIZONTAL );

	wxBoxSizer* bSizer70;
	bSizer70 = new wxBoxSizer( wxHORIZONTAL );

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
	bSizer58->Add( RunProfileComboBox, 50, wxALIGN_CENTER_VERTICAL|wxALL, 5 );

	wxBoxSizer* bSizer60;
	bSizer60 = new wxBoxSizer( wxVERTICAL );

	StartEstimationButton = new wxButton( StartPanel, wxID_ANY, wxT("Start Search"), wxDefaultPosition, wxDefaultSize, 0 );
	bSizer60->Add( StartEstimationButton, 0, wxALL, 5 );


	bSizer58->Add( bSizer60, 50, wxEXPAND, 5 );


	StartPanel->SetSizer( bSizer58 );
	StartPanel->Layout();
	bSizer58->Fit( StartPanel );
	bSizer48->Add( StartPanel, 1, wxEXPAND | wxALL, 5 );


	bSizer43->Add( bSizer48, 0, wxEXPAND, 5 );


	this->SetSizer( bSizer43 );
	this->Layout();

	// Connect Events
	this->Connect( wxEVT_UPDATE_UI, wxUpdateUIEventHandler( RefineTemplateDevPanelParent::OnUpdateUI ) );
	ResetAllDefaultsButton->Connect( wxEVT_COMMAND_BUTTON_CLICKED, wxCommandEventHandler( RefineTemplateDevPanelParent::ResetAllDefaultsClick ), NULL, this );
	InfoText->Connect( wxEVT_COMMAND_TEXT_URL, wxTextUrlEventHandler( RefineTemplateDevPanelParent::OnInfoURL ), NULL, this );
	FinishButton->Connect( wxEVT_COMMAND_BUTTON_CLICKED, wxCommandEventHandler( RefineTemplateDevPanelParent::FinishButtonClick ), NULL, this );
	CancelAlignmentButton->Connect( wxEVT_COMMAND_BUTTON_CLICKED, wxCommandEventHandler( RefineTemplateDevPanelParent::TerminateButtonClick ), NULL, this );
	StartEstimationButton->Connect( wxEVT_COMMAND_BUTTON_CLICKED, wxCommandEventHandler( RefineTemplateDevPanelParent::StartEstimationClick ), NULL, this );
}

RefineTemplateDevPanelParent::~RefineTemplateDevPanelParent()
{
	// Disconnect Events
	this->Disconnect( wxEVT_UPDATE_UI, wxUpdateUIEventHandler( RefineTemplateDevPanelParent::OnUpdateUI ) );
	ResetAllDefaultsButton->Disconnect( wxEVT_COMMAND_BUTTON_CLICKED, wxCommandEventHandler( RefineTemplateDevPanelParent::ResetAllDefaultsClick ), NULL, this );
	InfoText->Disconnect( wxEVT_COMMAND_TEXT_URL, wxTextUrlEventHandler( RefineTemplateDevPanelParent::OnInfoURL ), NULL, this );
	FinishButton->Disconnect( wxEVT_COMMAND_BUTTON_CLICKED, wxCommandEventHandler( RefineTemplateDevPanelParent::FinishButtonClick ), NULL, this );
	CancelAlignmentButton->Disconnect( wxEVT_COMMAND_BUTTON_CLICKED, wxCommandEventHandler( RefineTemplateDevPanelParent::TerminateButtonClick ), NULL, this );
	StartEstimationButton->Disconnect( wxEVT_COMMAND_BUTTON_CLICKED, wxCommandEventHandler( RefineTemplateDevPanelParent::StartEstimationClick ), NULL, this );

}

// clang-format on