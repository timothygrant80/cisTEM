///////////////////////////////////////////////////////////////////////////
// C++ code generated with wxFormBuilder (version 3.10.0-4761b0c)
// http://www.wxformbuilder.org/
//
// PLEASE DO *NOT* EDIT THIS FILE!
///////////////////////////////////////////////////////////////////////////

#include "../core/gui_core_headers.h"

#include "AssetPickerComboPanel.h"
#include "BitmapPanel.h"
#include "CTF1DPanel.h"
#include "DisplayPanel.h"
#include "DisplayRefinementResultsPanel.h"
#include "MyFSCPanel.h"
#include "PlotCurvePanel.h"
#include "ResultsDataViewListCtrl.h"
#include "ShowCTFResultsPanel.h"
#include "my_controls.h"

#include "ProjectX_gui_ctffind.h"

///////////////////////////////////////////////////////////////////////////

ShowCTFResultsPanelParent::ShowCTFResultsPanelParent( wxWindow* parent, wxWindowID id, const wxPoint& pos, const wxSize& size, long style, const wxString& name ) : wxPanel( parent, id, pos, size, style, name )
{
	wxBoxSizer* bSizer92;
	bSizer92 = new wxBoxSizer( wxVERTICAL );

	m_splitter16 = new wxSplitterWindow( this, wxID_ANY, wxDefaultPosition, wxDefaultSize, wxSP_3D );
	m_splitter16->SetSashGravity( 0.5 );
	m_splitter16->Connect( wxEVT_IDLE, wxIdleEventHandler( ShowCTFResultsPanelParent::m_splitter16OnIdle ), NULL, this );

	m_panel87 = new wxPanel( m_splitter16, wxID_ANY, wxDefaultPosition, wxDefaultSize, wxTAB_TRAVERSAL );
	wxBoxSizer* bSizer301;
	bSizer301 = new wxBoxSizer( wxVERTICAL );

	m_splitter15 = new wxSplitterWindow( m_panel87, wxID_ANY, wxDefaultPosition, wxDefaultSize, wxSP_3D );
	m_splitter15->SetSashGravity( 0.5 );
	m_splitter15->Connect( wxEVT_IDLE, wxIdleEventHandler( ShowCTFResultsPanelParent::m_splitter15OnIdle ), NULL, this );

	m_panel88 = new wxPanel( m_splitter15, wxID_ANY, wxDefaultPosition, wxDefaultSize, wxTAB_TRAVERSAL );
	wxBoxSizer* bSizer302;
	bSizer302 = new wxBoxSizer( wxVERTICAL );

	m_staticText377 = new wxStaticText( m_panel88, wxID_ANY, wxT("2D CTF Fit Result"), wxDefaultPosition, wxDefaultSize, 0 );
	m_staticText377->Wrap( -1 );
	m_staticText377->SetFont( wxFont( 10, wxFONTFAMILY_SWISS, wxFONTSTYLE_NORMAL, wxFONTWEIGHT_BOLD, false, wxT("Sans") ) );

	bSizer302->Add( m_staticText377, 0, wxALL, 5 );

	m_staticline81 = new wxStaticLine( m_panel88, wxID_ANY, wxDefaultPosition, wxDefaultSize, wxLI_HORIZONTAL );
	bSizer302->Add( m_staticline81, 0, wxEXPAND | wxALL, 5 );

	CTF2DResultsPanel = new BitmapPanel( m_panel88, wxID_ANY, wxDefaultPosition, wxDefaultSize, wxTAB_TRAVERSAL );
	bSizer302->Add( CTF2DResultsPanel, 1, wxEXPAND | wxALL, 5 );


	m_panel88->SetSizer( bSizer302 );
	m_panel88->Layout();
	bSizer302->Fit( m_panel88 );
	m_panel89 = new wxPanel( m_splitter15, wxID_ANY, wxDefaultPosition, wxDefaultSize, wxTAB_TRAVERSAL );
	wxBoxSizer* bSizer303;
	bSizer303 = new wxBoxSizer( wxVERTICAL );

	m_staticText378 = new wxStaticText( m_panel89, wxID_ANY, wxT("1D CTF Fit Result"), wxDefaultPosition, wxDefaultSize, 0 );
	m_staticText378->Wrap( -1 );
	m_staticText378->SetFont( wxFont( 10, wxFONTFAMILY_SWISS, wxFONTSTYLE_NORMAL, wxFONTWEIGHT_BOLD, false, wxT("Sans") ) );

	bSizer303->Add( m_staticText378, 0, wxALL, 5 );

	m_staticline82 = new wxStaticLine( m_panel89, wxID_ANY, wxDefaultPosition, wxDefaultSize, wxLI_HORIZONTAL );
	bSizer303->Add( m_staticline82, 0, wxEXPAND | wxALL, 5 );

	CTFPlotPanel = new CTF1DPanel( m_panel89, wxID_ANY, wxDefaultPosition, wxDefaultSize, wxTAB_TRAVERSAL );
	bSizer303->Add( CTFPlotPanel, 1, wxEXPAND | wxALL, 5 );


	m_panel89->SetSizer( bSizer303 );
	m_panel89->Layout();
	bSizer303->Fit( m_panel89 );
	m_splitter15->SplitHorizontally( m_panel88, m_panel89, 0 );
	bSizer301->Add( m_splitter15, 1, wxEXPAND, 5 );


	m_panel87->SetSizer( bSizer301 );
	m_panel87->Layout();
	bSizer301->Fit( m_panel87 );
	m_panel86 = new wxPanel( m_splitter16, wxID_ANY, wxDefaultPosition, wxDefaultSize, wxTAB_TRAVERSAL );
	wxBoxSizer* bSizer304;
	bSizer304 = new wxBoxSizer( wxVERTICAL );

	m_staticText379 = new wxStaticText( m_panel86, wxID_ANY, wxT("Estimated CTF Parameters"), wxDefaultPosition, wxDefaultSize, 0 );
	m_staticText379->Wrap( -1 );
	m_staticText379->SetFont( wxFont( 10, wxFONTFAMILY_SWISS, wxFONTSTYLE_NORMAL, wxFONTWEIGHT_BOLD, false, wxT("Sans") ) );

	bSizer304->Add( m_staticText379, 0, wxALL, 5 );

	m_staticline78 = new wxStaticLine( m_panel86, wxID_ANY, wxDefaultPosition, wxDefaultSize, wxLI_HORIZONTAL );
	bSizer304->Add( m_staticline78, 0, wxEXPAND | wxALL, 5 );

	wxBoxSizer* bSizer305;
	bSizer305 = new wxBoxSizer( wxHORIZONTAL );

	wxFlexGridSizer* fgSizer43;
	fgSizer43 = new wxFlexGridSizer( 0, 4, 0, 0 );
	fgSizer43->SetFlexibleDirection( wxBOTH );
	fgSizer43->SetNonFlexibleGrowMode( wxFLEX_GROWMODE_SPECIFIED );

	m_staticText380 = new wxStaticText( m_panel86, wxID_ANY, wxT("\tDefocus 1 :"), wxDefaultPosition, wxDefaultSize, 0 );
	m_staticText380->Wrap( -1 );
	m_staticText380->SetFont( wxFont( 10, wxFONTFAMILY_SWISS, wxFONTSTYLE_NORMAL, wxFONTWEIGHT_BOLD, false, wxT("Sans") ) );

	fgSizer43->Add( m_staticText380, 0, wxALIGN_RIGHT|wxALL, 5 );

	Defocus1Text = new wxStaticText( m_panel86, wxID_ANY, wxEmptyString, wxDefaultPosition, wxDefaultSize, 0 );
	Defocus1Text->Wrap( -1 );
	Defocus1Text->SetMinSize( wxSize( 50,-1 ) );

	fgSizer43->Add( Defocus1Text, 0, wxALIGN_LEFT|wxALL, 5 );

	m_staticText389 = new wxStaticText( m_panel86, wxID_ANY, wxT("Score :"), wxDefaultPosition, wxDefaultSize, 0 );
	m_staticText389->Wrap( -1 );
	m_staticText389->SetFont( wxFont( 10, wxFONTFAMILY_SWISS, wxFONTSTYLE_NORMAL, wxFONTWEIGHT_BOLD, false, wxT("Sans") ) );

	fgSizer43->Add( m_staticText389, 0, wxALIGN_RIGHT|wxALL, 5 );

	ScoreText = new wxStaticText( m_panel86, wxID_ANY, wxEmptyString, wxDefaultPosition, wxDefaultSize, 0 );
	ScoreText->Wrap( -1 );
	ScoreText->SetMinSize( wxSize( 50,-1 ) );

	fgSizer43->Add( ScoreText, 0, wxALL, 5 );

	m_staticText382 = new wxStaticText( m_panel86, wxID_ANY, wxT("Defocus 2 :"), wxDefaultPosition, wxDefaultSize, 0 );
	m_staticText382->Wrap( -1 );
	m_staticText382->SetFont( wxFont( 10, wxFONTFAMILY_SWISS, wxFONTSTYLE_NORMAL, wxFONTWEIGHT_BOLD, false, wxT("Sans") ) );

	fgSizer43->Add( m_staticText382, 0, wxALIGN_RIGHT|wxALL, 5 );

	Defocus2Text = new wxStaticText( m_panel86, wxID_ANY, wxEmptyString, wxDefaultPosition, wxDefaultSize, 0 );
	Defocus2Text->Wrap( -1 );
	fgSizer43->Add( Defocus2Text, 0, wxALL, 5 );

	m_staticText391 = new wxStaticText( m_panel86, wxID_ANY, wxT("Fit Res. :"), wxDefaultPosition, wxDefaultSize, 0 );
	m_staticText391->Wrap( -1 );
	m_staticText391->SetFont( wxFont( 10, wxFONTFAMILY_SWISS, wxFONTSTYLE_NORMAL, wxFONTWEIGHT_BOLD, false, wxT("Sans") ) );

	fgSizer43->Add( m_staticText391, 0, wxALIGN_RIGHT|wxALL, 5 );

	FitResText = new wxStaticText( m_panel86, wxID_ANY, wxEmptyString, wxDefaultPosition, wxDefaultSize, 0 );
	FitResText->Wrap( -1 );
	fgSizer43->Add( FitResText, 0, wxALL, 5 );

	m_staticText384 = new wxStaticText( m_panel86, wxID_ANY, wxT("Angle :"), wxDefaultPosition, wxDefaultSize, 0 );
	m_staticText384->Wrap( -1 );
	m_staticText384->SetFont( wxFont( 10, wxFONTFAMILY_SWISS, wxFONTSTYLE_NORMAL, wxFONTWEIGHT_BOLD, false, wxT("Sans") ) );

	fgSizer43->Add( m_staticText384, 0, wxALIGN_RIGHT|wxALL, 5 );

	AngleText = new wxStaticText( m_panel86, wxID_ANY, wxEmptyString, wxDefaultPosition, wxDefaultSize, 0 );
	AngleText->Wrap( -1 );
	fgSizer43->Add( AngleText, 0, wxALL, 5 );

	m_staticText393 = new wxStaticText( m_panel86, wxID_ANY, wxT("Alias Res. :"), wxDefaultPosition, wxDefaultSize, 0 );
	m_staticText393->Wrap( -1 );
	m_staticText393->SetFont( wxFont( 10, wxFONTFAMILY_SWISS, wxFONTSTYLE_NORMAL, wxFONTWEIGHT_BOLD, false, wxT("Sans") ) );

	fgSizer43->Add( m_staticText393, 0, wxALIGN_RIGHT|wxALL, 5 );

	AliasResText = new wxStaticText( m_panel86, wxID_ANY, wxEmptyString, wxDefaultPosition, wxDefaultSize, 0 );
	AliasResText->Wrap( -1 );
	fgSizer43->Add( AliasResText, 0, wxALL, 5 );

	m_staticText386 = new wxStaticText( m_panel86, wxID_ANY, wxT("Phase Shift :"), wxDefaultPosition, wxDefaultSize, 0 );
	m_staticText386->Wrap( -1 );
	m_staticText386->SetFont( wxFont( 10, wxFONTFAMILY_SWISS, wxFONTSTYLE_NORMAL, wxFONTWEIGHT_BOLD, false, wxT("Sans") ) );

	fgSizer43->Add( m_staticText386, 0, wxALIGN_RIGHT|wxALL, 5 );

	PhaseShiftText = new wxStaticText( m_panel86, wxID_ANY, wxEmptyString, wxDefaultPosition, wxDefaultSize, 0 );
	PhaseShiftText->Wrap( -1 );
	fgSizer43->Add( PhaseShiftText, 0, wxALL, 5 );

	IcinessLabel = new wxStaticText( m_panel86, wxID_ANY, wxT("Iciness :"), wxDefaultPosition, wxDefaultSize, 0 );
	IcinessLabel->Wrap( -1 );
	IcinessLabel->SetFont( wxFont( 10, wxFONTFAMILY_SWISS, wxFONTSTYLE_NORMAL, wxFONTWEIGHT_BOLD, false, wxT("Sans") ) );

	fgSizer43->Add( IcinessLabel, 0, wxALIGN_RIGHT|wxALL, 5 );

	IcinessStaticText = new wxStaticText( m_panel86, wxID_ANY, wxEmptyString, wxDefaultPosition, wxDefaultSize, 0 );
	IcinessStaticText->Wrap( -1 );
	fgSizer43->Add( IcinessStaticText, 0, wxALL, 5 );

	m_staticText866 = new wxStaticText( m_panel86, wxID_ANY, wxT("Tilt Angle :"), wxDefaultPosition, wxDefaultSize, 0 );
	m_staticText866->Wrap( -1 );
	m_staticText866->SetFont( wxFont( 10, wxFONTFAMILY_SWISS, wxFONTSTYLE_NORMAL, wxFONTWEIGHT_BOLD, false, wxT("Sans") ) );

	fgSizer43->Add( m_staticText866, 0, wxALIGN_CENTER_VERTICAL|wxALIGN_RIGHT|wxALL, 5 );

	TiltAngleStaticText = new wxStaticText( m_panel86, wxID_ANY, wxEmptyString, wxDefaultPosition, wxDefaultSize, 0 );
	TiltAngleStaticText->Wrap( -1 );
	fgSizer43->Add( TiltAngleStaticText, 0, wxALL, 5 );

	m_staticText868 = new wxStaticText( m_panel86, wxID_ANY, wxT("Tilt Axis :"), wxDefaultPosition, wxDefaultSize, 0 );
	m_staticText868->Wrap( -1 );
	m_staticText868->SetFont( wxFont( 10, wxFONTFAMILY_SWISS, wxFONTSTYLE_NORMAL, wxFONTWEIGHT_BOLD, false, wxT("Sans") ) );

	fgSizer43->Add( m_staticText868, 0, wxALIGN_CENTER_VERTICAL|wxALIGN_RIGHT|wxALL, 5 );

	TiltAxisStaticText = new wxStaticText( m_panel86, wxID_ANY, wxEmptyString, wxDefaultPosition, wxDefaultSize, 0 );
	TiltAxisStaticText->Wrap( -1 );
	fgSizer43->Add( TiltAxisStaticText, 0, wxALL, 5 );


	fgSizer43->Add( 0, 0, 10, wxALL, 5 );


	fgSizer43->Add( 0, 0, 0, wxALL, 5 );

	m_staticText8681 = new wxStaticText( m_panel86, wxID_ANY, wxT("Est. Sample Thickness :"), wxDefaultPosition, wxDefaultSize, 0 );
	m_staticText8681->Wrap( -1 );
	m_staticText8681->SetFont( wxFont( 10, wxFONTFAMILY_SWISS, wxFONTSTYLE_NORMAL, wxFONTWEIGHT_BOLD, false, wxT("Sans") ) );

	fgSizer43->Add( m_staticText8681, 0, wxALL, 5 );

	ThicknessStaticText = new wxStaticText( m_panel86, wxID_ANY, wxEmptyString, wxDefaultPosition, wxDefaultSize, 0 );
	ThicknessStaticText->Wrap( -1 );
	fgSizer43->Add( ThicknessStaticText, 0, wxALL, 5 );


	bSizer305->Add( fgSizer43, 1, wxEXPAND, 5 );


	bSizer304->Add( bSizer305, 0, wxEXPAND, 5 );

	m_staticline83 = new wxStaticLine( m_panel86, wxID_ANY, wxDefaultPosition, wxDefaultSize, wxLI_HORIZONTAL );
	bSizer304->Add( m_staticline83, 0, wxEXPAND | wxALL, 5 );

	wxBoxSizer* bSizer306;
	bSizer306 = new wxBoxSizer( wxHORIZONTAL );

	m_staticText394 = new wxStaticText( m_panel86, wxID_ANY, wxT("Image / Aligned Movie Sum"), wxDefaultPosition, wxDefaultSize, 0 );
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
}

ShowCTFResultsPanelParent::~ShowCTFResultsPanelParent()
{
}

FindCTFResultsPanel::FindCTFResultsPanel( wxWindow* parent, wxWindowID id, const wxPoint& pos, const wxSize& size, long style, const wxString& name ) : wxPanel( parent, id, pos, size, style, name )
{
	wxBoxSizer* bSizer63;
	bSizer63 = new wxBoxSizer( wxVERTICAL );

	m_staticline25 = new wxStaticLine( this, wxID_ANY, wxDefaultPosition, wxDefaultSize, wxLI_HORIZONTAL );
	bSizer63->Add( m_staticline25, 0, wxEXPAND | wxALL, 5 );

	m_splitter4 = new wxSplitterWindow( this, wxID_ANY, wxDefaultPosition, wxDefaultSize, wxSP_3D );
	m_splitter4->SetSashGravity( 0.5 );
	m_splitter4->Connect( wxEVT_IDLE, wxIdleEventHandler( FindCTFResultsPanel::m_splitter4OnIdle ), NULL, this );

	m_panel13 = new wxPanel( m_splitter4, wxID_ANY, wxDefaultPosition, wxDefaultSize, wxTAB_TRAVERSAL );
	wxBoxSizer* bSizer66;
	bSizer66 = new wxBoxSizer( wxVERTICAL );

	wxBoxSizer* bSizer64;
	bSizer64 = new wxBoxSizer( wxHORIZONTAL );

	AllImagesButton = new wxRadioButton( m_panel13, wxID_ANY, wxT("All Images"), wxDefaultPosition, wxDefaultSize, 0 );
	bSizer64->Add( AllImagesButton, 0, wxALIGN_CENTER_VERTICAL|wxALL, 5 );

	ByFilterButton = new wxRadioButton( m_panel13, wxID_ANY, wxT("By Filter"), wxDefaultPosition, wxDefaultSize, 0 );
	bSizer64->Add( ByFilterButton, 0, wxALIGN_CENTER_VERTICAL|wxALL, 5 );

	FilterButton = new wxButton( m_panel13, wxID_ANY, wxT("Define Filter"), wxDefaultPosition, wxDefaultSize, 0 );
	FilterButton->Enable( false );

	bSizer64->Add( FilterButton, 0, wxALIGN_CENTER_VERTICAL|wxALL, 5 );


	bSizer64->Add( 0, 0, 1, wxEXPAND, 5 );

	m_staticline77 = new wxStaticLine( m_panel13, wxID_ANY, wxDefaultPosition, wxDefaultSize, wxLI_HORIZONTAL|wxLI_VERTICAL );
	bSizer64->Add( m_staticline77, 0, wxEXPAND | wxALL, 5 );

	PlotResultsButton = new NoFocusBitmapButton( m_panel13, wxID_ANY, wxNullBitmap, wxDefaultPosition, wxDefaultSize, wxBU_AUTODRAW|0 );

	PlotResultsButton->SetDefault();
	bSizer64->Add( PlotResultsButton, 0, wxALIGN_CENTER_VERTICAL|wxALL, 5 );

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

	InfoSizer = new wxFlexGridSizer( 0, 6, 0, 0 );
	InfoSizer->SetFlexibleDirection( wxBOTH );
	InfoSizer->SetNonFlexibleGrowMode( wxFLEX_GROWMODE_SPECIFIED );

	m_staticText72 = new wxStaticText( JobDetailsPanel, wxID_ANY, wxT("Estimation ID :"), wxDefaultPosition, wxDefaultSize, 0 );
	m_staticText72->Wrap( -1 );
	m_staticText72->SetFont( wxFont( 10, wxFONTFAMILY_SWISS, wxFONTSTYLE_NORMAL, wxFONTWEIGHT_BOLD, false, wxT("Sans") ) );

	InfoSizer->Add( m_staticText72, 0, wxALIGN_RIGHT|wxALL, 5 );

	EstimationIDStaticText = new wxStaticText( JobDetailsPanel, wxID_ANY, wxEmptyString, wxDefaultPosition, wxDefaultSize, 0 );
	EstimationIDStaticText->Wrap( -1 );
	InfoSizer->Add( EstimationIDStaticText, 0, wxALIGN_LEFT|wxALL, 5 );

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

	m_staticText82 = new wxStaticText( JobDetailsPanel, wxID_ANY, wxT("Cs :"), wxDefaultPosition, wxDefaultSize, 0 );
	m_staticText82->Wrap( -1 );
	m_staticText82->SetFont( wxFont( 10, wxFONTFAMILY_SWISS, wxFONTSTYLE_NORMAL, wxFONTWEIGHT_BOLD, false, wxT("Sans") ) );

	InfoSizer->Add( m_staticText82, 0, wxALIGN_RIGHT|wxALL, 5 );

	CsStaticText = new wxStaticText( JobDetailsPanel, wxID_ANY, wxEmptyString, wxDefaultPosition, wxDefaultSize, 0 );
	CsStaticText->Wrap( -1 );
	InfoSizer->Add( CsStaticText, 0, wxALIGN_LEFT|wxALL, 5 );

	m_staticText78 = new wxStaticText( JobDetailsPanel, wxID_ANY, wxT("Pixel Size :"), wxDefaultPosition, wxDefaultSize, 0 );
	m_staticText78->Wrap( -1 );
	m_staticText78->SetFont( wxFont( 10, wxFONTFAMILY_SWISS, wxFONTSTYLE_NORMAL, wxFONTWEIGHT_BOLD, false, wxT("Sans") ) );

	InfoSizer->Add( m_staticText78, 0, wxALIGN_RIGHT|wxALL, 5 );

	PixelSizeStaticText = new wxStaticText( JobDetailsPanel, wxID_ANY, wxEmptyString, wxDefaultPosition, wxDefaultSize, 0 );
	PixelSizeStaticText->Wrap( -1 );
	InfoSizer->Add( PixelSizeStaticText, 0, wxALIGN_LEFT|wxALL, 5 );

	m_staticText781 = new wxStaticText( JobDetailsPanel, wxID_ANY, wxT("Resample Pixel Size? :"), wxDefaultPosition, wxDefaultSize, 0 );
	m_staticText781->Wrap( -1 );
	m_staticText781->SetFont( wxFont( 10, wxFONTFAMILY_SWISS, wxFONTSTYLE_NORMAL, wxFONTWEIGHT_BOLD, false, wxT("Sans") ) );

	InfoSizer->Add( m_staticText781, 0, wxALIGN_RIGHT|wxALL, 5 );

	ResampleStaticText = new wxStaticText( JobDetailsPanel, wxID_ANY, wxEmptyString, wxDefaultPosition, wxDefaultSize, 0 );
	ResampleStaticText->Wrap( -1 );
	InfoSizer->Add( ResampleStaticText, 0, wxALL, 5 );

	m_staticText7811 = new wxStaticText( JobDetailsPanel, wxID_ANY, wxT("Pixel Size Target :"), wxDefaultPosition, wxDefaultSize, 0 );
	m_staticText7811->Wrap( -1 );
	m_staticText7811->SetFont( wxFont( 10, wxFONTFAMILY_SWISS, wxFONTSTYLE_NORMAL, wxFONTWEIGHT_BOLD, false, wxT("Sans") ) );

	InfoSizer->Add( m_staticText7811, 0, wxALIGN_RIGHT|wxALL, 5 );

	PixelSizeTargetStaticText = new wxStaticText( JobDetailsPanel, wxID_ANY, wxEmptyString, wxDefaultPosition, wxDefaultSize, 0 );
	PixelSizeTargetStaticText->Wrap( -1 );
	InfoSizer->Add( PixelSizeTargetStaticText, 0, wxALL, 5 );

	m_staticText78111 = new wxStaticText( JobDetailsPanel, wxID_ANY, wxT("Estimate Tilt? :"), wxDefaultPosition, wxDefaultSize, 0 );
	m_staticText78111->Wrap( -1 );
	m_staticText78111->SetFont( wxFont( 10, wxFONTFAMILY_SWISS, wxFONTSTYLE_NORMAL, wxFONTWEIGHT_BOLD, false, wxT("Sans") ) );

	InfoSizer->Add( m_staticText78111, 0, wxALIGN_RIGHT|wxALL, 5 );

	EstimateTiltStaticText = new wxStaticText( JobDetailsPanel, wxID_ANY, wxEmptyString, wxDefaultPosition, wxDefaultSize, 0 );
	EstimateTiltStaticText->Wrap( -1 );
	InfoSizer->Add( EstimateTiltStaticText, 0, wxALL, 5 );

	m_staticText96 = new wxStaticText( JobDetailsPanel, wxID_ANY, wxT("Amp. Contrast :"), wxDefaultPosition, wxDefaultSize, 0 );
	m_staticText96->Wrap( -1 );
	m_staticText96->SetFont( wxFont( 10, wxFONTFAMILY_SWISS, wxFONTSTYLE_NORMAL, wxFONTWEIGHT_BOLD, false, wxT("Sans") ) );

	InfoSizer->Add( m_staticText96, 0, wxALIGN_RIGHT|wxALL, 5 );

	AmplitudeContrastStaticText = new wxStaticText( JobDetailsPanel, wxID_ANY, wxEmptyString, wxDefaultPosition, wxDefaultSize, 0 );
	AmplitudeContrastStaticText->Wrap( -1 );
	InfoSizer->Add( AmplitudeContrastStaticText, 0, wxALIGN_LEFT|wxALL, 5 );

	m_staticText85 = new wxStaticText( JobDetailsPanel, wxID_ANY, wxT("Box Size :"), wxDefaultPosition, wxDefaultSize, 0 );
	m_staticText85->Wrap( -1 );
	m_staticText85->SetFont( wxFont( 10, wxFONTFAMILY_SWISS, wxFONTSTYLE_NORMAL, wxFONTWEIGHT_BOLD, false, wxT("Sans") ) );

	InfoSizer->Add( m_staticText85, 0, wxALIGN_RIGHT|wxALL, 5 );

	BoxSizeStaticText = new wxStaticText( JobDetailsPanel, wxID_ANY, wxEmptyString, wxDefaultPosition, wxDefaultSize, 0 );
	BoxSizeStaticText->Wrap( -1 );
	InfoSizer->Add( BoxSizeStaticText, 0, wxALIGN_LEFT|wxALL, 5 );

	m_staticText87 = new wxStaticText( JobDetailsPanel, wxID_ANY, wxT("Min. Res. :"), wxDefaultPosition, wxDefaultSize, 0 );
	m_staticText87->Wrap( -1 );
	m_staticText87->SetFont( wxFont( 10, wxFONTFAMILY_SWISS, wxFONTSTYLE_NORMAL, wxFONTWEIGHT_BOLD, false, wxT("Sans") ) );

	InfoSizer->Add( m_staticText87, 0, wxALIGN_RIGHT|wxALL, 5 );

	MinResStaticText = new wxStaticText( JobDetailsPanel, wxID_ANY, wxEmptyString, wxDefaultPosition, wxDefaultSize, 0 );
	MinResStaticText->Wrap( -1 );
	InfoSizer->Add( MinResStaticText, 0, wxALIGN_LEFT|wxALL, 5 );

	m_staticText89 = new wxStaticText( JobDetailsPanel, wxID_ANY, wxT("Max. Res. :"), wxDefaultPosition, wxDefaultSize, 0 );
	m_staticText89->Wrap( -1 );
	m_staticText89->SetFont( wxFont( 10, wxFONTFAMILY_SWISS, wxFONTSTYLE_NORMAL, wxFONTWEIGHT_BOLD, false, wxT("Sans") ) );

	InfoSizer->Add( m_staticText89, 0, wxALIGN_RIGHT|wxALL, 5 );

	MaxResStaticText = new wxStaticText( JobDetailsPanel, wxID_ANY, wxEmptyString, wxDefaultPosition, wxDefaultSize, 0 );
	MaxResStaticText->Wrap( -1 );
	InfoSizer->Add( MaxResStaticText, 0, wxALIGN_LEFT|wxALL, 5 );

	m_staticText91 = new wxStaticText( JobDetailsPanel, wxID_ANY, wxT("Min. Defocus :"), wxDefaultPosition, wxDefaultSize, 0 );
	m_staticText91->Wrap( -1 );
	m_staticText91->SetFont( wxFont( 10, wxFONTFAMILY_SWISS, wxFONTSTYLE_NORMAL, wxFONTWEIGHT_BOLD, false, wxT("Sans") ) );

	InfoSizer->Add( m_staticText91, 0, wxALIGN_RIGHT|wxALL, 5 );

	MinDefocusStaticText = new wxStaticText( JobDetailsPanel, wxID_ANY, wxEmptyString, wxDefaultPosition, wxDefaultSize, 0 );
	MinDefocusStaticText->Wrap( -1 );
	InfoSizer->Add( MinDefocusStaticText, 0, wxALIGN_LEFT|wxALL, 5 );

	m_staticText79 = new wxStaticText( JobDetailsPanel, wxID_ANY, wxT("Max. Defocus :"), wxDefaultPosition, wxDefaultSize, 0 );
	m_staticText79->Wrap( -1 );
	m_staticText79->SetFont( wxFont( 10, wxFONTFAMILY_SWISS, wxFONTSTYLE_NORMAL, wxFONTWEIGHT_BOLD, false, wxT("Sans") ) );

	InfoSizer->Add( m_staticText79, 0, wxALIGN_RIGHT|wxALL, 5 );

	MaxDefocusStaticText = new wxStaticText( JobDetailsPanel, wxID_ANY, wxEmptyString, wxDefaultPosition, wxDefaultSize, 0 );
	MaxDefocusStaticText->Wrap( -1 );
	InfoSizer->Add( MaxDefocusStaticText, 0, wxALIGN_LEFT|wxALL, 5 );

	m_staticText95 = new wxStaticText( JobDetailsPanel, wxID_ANY, wxT("Defocus Step :"), wxDefaultPosition, wxDefaultSize, 0 );
	m_staticText95->Wrap( -1 );
	m_staticText95->SetFont( wxFont( 10, wxFONTFAMILY_SWISS, wxFONTSTYLE_NORMAL, wxFONTWEIGHT_BOLD, false, wxT("Sans") ) );

	InfoSizer->Add( m_staticText95, 0, wxALIGN_RIGHT|wxALL, 5 );

	DefocusStepStaticText = new wxStaticText( JobDetailsPanel, wxID_ANY, wxEmptyString, wxDefaultPosition, wxDefaultSize, 0 );
	DefocusStepStaticText->Wrap( -1 );
	InfoSizer->Add( DefocusStepStaticText, 0, wxALIGN_LEFT|wxALL, 5 );

	LargeAstigExpectedLabel = new wxStaticText( JobDetailsPanel, wxID_ANY, wxT("Exhaustive Search? :"), wxDefaultPosition, wxDefaultSize, 0 );
	LargeAstigExpectedLabel->Wrap( -1 );
	LargeAstigExpectedLabel->SetFont( wxFont( 10, wxFONTFAMILY_SWISS, wxFONTSTYLE_NORMAL, wxFONTWEIGHT_BOLD, false, wxT("Sans") ) );

	InfoSizer->Add( LargeAstigExpectedLabel, 0, wxALIGN_RIGHT|wxALL, 5 );

	LargeAstigExpectedStaticText = new wxStaticText( JobDetailsPanel, wxID_ANY, wxEmptyString, wxDefaultPosition, wxDefaultSize, 0 );
	LargeAstigExpectedStaticText->Wrap( -1 );
	InfoSizer->Add( LargeAstigExpectedStaticText, 0, wxALL, 5 );

	m_staticText99 = new wxStaticText( JobDetailsPanel, wxID_ANY, wxT("Restrain Astig.?:"), wxDefaultPosition, wxDefaultSize, 0 );
	m_staticText99->Wrap( -1 );
	m_staticText99->SetFont( wxFont( 10, wxFONTFAMILY_SWISS, wxFONTSTYLE_NORMAL, wxFONTWEIGHT_BOLD, false, wxT("Sans") ) );

	InfoSizer->Add( m_staticText99, 0, wxALIGN_RIGHT|wxALL, 5 );

	RestrainAstigStaticText = new wxStaticText( JobDetailsPanel, wxID_ANY, wxEmptyString, wxDefaultPosition, wxDefaultSize, 0 );
	RestrainAstigStaticText->Wrap( -1 );
	InfoSizer->Add( RestrainAstigStaticText, 0, wxALIGN_LEFT|wxALL, 5 );

	ToleratedAstigLabel = new wxStaticText( JobDetailsPanel, wxID_ANY, wxT("Tolerated Astig. :"), wxDefaultPosition, wxDefaultSize, 0 );
	ToleratedAstigLabel->Wrap( -1 );
	ToleratedAstigLabel->SetFont( wxFont( 10, wxFONTFAMILY_SWISS, wxFONTSTYLE_NORMAL, wxFONTWEIGHT_BOLD, false, wxT("Sans") ) );

	InfoSizer->Add( ToleratedAstigLabel, 0, wxALIGN_RIGHT|wxALL, 5 );

	ToleratedAstigStaticText = new wxStaticText( JobDetailsPanel, wxID_ANY, wxEmptyString, wxDefaultPosition, wxDefaultSize, 0 );
	ToleratedAstigStaticText->Wrap( -1 );
	InfoSizer->Add( ToleratedAstigStaticText, 0, wxALIGN_LEFT|wxALL, 5 );

	NumberOfAveragedFramesLabel = new wxStaticText( JobDetailsPanel, wxID_ANY, wxT("Num. Averaged Frames :"), wxDefaultPosition, wxDefaultSize, 0 );
	NumberOfAveragedFramesLabel->Wrap( -1 );
	NumberOfAveragedFramesLabel->SetFont( wxFont( 10, wxFONTFAMILY_SWISS, wxFONTSTYLE_NORMAL, wxFONTWEIGHT_BOLD, false, wxT("Sans") ) );

	InfoSizer->Add( NumberOfAveragedFramesLabel, 0, wxALIGN_RIGHT|wxALL, 5 );

	NumberOfAveragedFramesStaticText = new wxStaticText( JobDetailsPanel, wxID_ANY, wxEmptyString, wxDefaultPosition, wxDefaultSize, 0 );
	NumberOfAveragedFramesStaticText->Wrap( -1 );
	InfoSizer->Add( NumberOfAveragedFramesStaticText, 0, wxALL, 5 );

	m_staticText103 = new wxStaticText( JobDetailsPanel, wxID_ANY, wxT("Add. Phase Shift :"), wxDefaultPosition, wxDefaultSize, 0 );
	m_staticText103->Wrap( -1 );
	m_staticText103->SetFont( wxFont( 10, wxFONTFAMILY_SWISS, wxFONTSTYLE_NORMAL, wxFONTWEIGHT_BOLD, false, wxT("Sans") ) );

	InfoSizer->Add( m_staticText103, 0, wxALIGN_RIGHT|wxALL, 5 );

	AddtionalPhaseShiftStaticText = new wxStaticText( JobDetailsPanel, wxID_ANY, wxEmptyString, wxDefaultPosition, wxDefaultSize, 0 );
	AddtionalPhaseShiftStaticText->Wrap( -1 );
	InfoSizer->Add( AddtionalPhaseShiftStaticText, 0, wxALIGN_LEFT|wxALL, 5 );

	MinPhaseShiftLabel = new wxStaticText( JobDetailsPanel, wxID_ANY, wxT("Min. Phase Shift :"), wxDefaultPosition, wxDefaultSize, 0 );
	MinPhaseShiftLabel->Wrap( -1 );
	MinPhaseShiftLabel->SetFont( wxFont( 10, wxFONTFAMILY_SWISS, wxFONTSTYLE_NORMAL, wxFONTWEIGHT_BOLD, false, wxT("Sans") ) );

	InfoSizer->Add( MinPhaseShiftLabel, 0, wxALIGN_RIGHT|wxALL, 5 );

	MinPhaseShiftStaticText = new wxStaticText( JobDetailsPanel, wxID_ANY, wxEmptyString, wxDefaultPosition, wxDefaultSize, 0 );
	MinPhaseShiftStaticText->Wrap( -1 );
	InfoSizer->Add( MinPhaseShiftStaticText, 0, wxALIGN_LEFT|wxALL, 5 );

	MaxPhaseShiftLabel = new wxStaticText( JobDetailsPanel, wxID_ANY, wxT("Max. Phase Shift :"), wxDefaultPosition, wxDefaultSize, 0 );
	MaxPhaseShiftLabel->Wrap( -1 );
	MaxPhaseShiftLabel->SetFont( wxFont( 10, wxFONTFAMILY_SWISS, wxFONTSTYLE_NORMAL, wxFONTWEIGHT_BOLD, false, wxT("Sans") ) );

	InfoSizer->Add( MaxPhaseShiftLabel, 0, wxALIGN_RIGHT|wxALL, 5 );

	MaxPhaseshiftStaticText = new wxStaticText( JobDetailsPanel, wxID_ANY, wxEmptyString, wxDefaultPosition, wxDefaultSize, 0 );
	MaxPhaseshiftStaticText->Wrap( -1 );
	InfoSizer->Add( MaxPhaseshiftStaticText, 0, wxALL, 5 );

	PhaseShiftStepLabel = new wxStaticText( JobDetailsPanel, wxID_ANY, wxT("Phase Shift Step :"), wxDefaultPosition, wxDefaultSize, 0 );
	PhaseShiftStepLabel->Wrap( -1 );
	PhaseShiftStepLabel->SetFont( wxFont( 10, wxFONTFAMILY_SWISS, wxFONTSTYLE_NORMAL, wxFONTWEIGHT_BOLD, false, wxT("Sans") ) );

	InfoSizer->Add( PhaseShiftStepLabel, 0, wxALIGN_RIGHT|wxALL, 5 );

	PhaseShiftStepStaticText = new wxStaticText( JobDetailsPanel, wxID_ANY, wxEmptyString, wxDefaultPosition, wxDefaultSize, 0 );
	PhaseShiftStepStaticText->Wrap( -1 );
	InfoSizer->Add( PhaseShiftStepStaticText, 0, wxALL, 5 );

	PhaseShiftStepLabel1 = new wxStaticText( JobDetailsPanel, wxID_ANY, wxT("Estimate Thickness? :"), wxDefaultPosition, wxDefaultSize, 0 );
	PhaseShiftStepLabel1->Wrap( -1 );
	PhaseShiftStepLabel1->SetFont( wxFont( 10, wxFONTFAMILY_SWISS, wxFONTSTYLE_NORMAL, wxFONTWEIGHT_BOLD, false, wxT("Sans") ) );

	InfoSizer->Add( PhaseShiftStepLabel1, 0, wxALIGN_RIGHT|wxALL, 5 );

	EstimateThicknessStaticText = new wxStaticText( JobDetailsPanel, wxID_ANY, wxEmptyString, wxDefaultPosition, wxDefaultSize, 0 );
	EstimateThicknessStaticText->Wrap( -1 );
	InfoSizer->Add( EstimateThicknessStaticText, 0, wxALL, 5 );

	ThicknessLabel1 = new wxStaticText( JobDetailsPanel, wxID_ANY, wxT("Thickness 1D search? :"), wxDefaultPosition, wxDefaultSize, 0 );
	ThicknessLabel1->Wrap( -1 );
	ThicknessLabel1->SetFont( wxFont( 10, wxFONTFAMILY_SWISS, wxFONTSTYLE_NORMAL, wxFONTWEIGHT_BOLD, false, wxT("Sans") ) );

	InfoSizer->Add( ThicknessLabel1, 0, wxALIGN_RIGHT|wxALL, 5 );

	Thickness1DStaticText = new wxStaticText( JobDetailsPanel, wxID_ANY, wxEmptyString, wxDefaultPosition, wxDefaultSize, 0 );
	Thickness1DStaticText->Wrap( -1 );
	InfoSizer->Add( Thickness1DStaticText, 0, wxALL, 5 );

	ThicknessLabel2 = new wxStaticText( JobDetailsPanel, wxID_ANY, wxT("Thickness 2D search? :"), wxDefaultPosition, wxDefaultSize, 0 );
	ThicknessLabel2->Wrap( -1 );
	ThicknessLabel2->SetFont( wxFont( 10, wxFONTFAMILY_SWISS, wxFONTSTYLE_NORMAL, wxFONTWEIGHT_BOLD, false, wxT("Sans") ) );

	InfoSizer->Add( ThicknessLabel2, 0, wxALIGN_RIGHT|wxALL, 5 );

	Thickness2DStaticText = new wxStaticText( JobDetailsPanel, wxID_ANY, wxEmptyString, wxDefaultPosition, wxDefaultSize, 0 );
	Thickness2DStaticText->Wrap( -1 );
	InfoSizer->Add( Thickness2DStaticText, 0, wxALL, 5 );

	ThicknessLabel3 = new wxStaticText( JobDetailsPanel, wxID_ANY, wxT("Thickness Min. Res. :"), wxDefaultPosition, wxDefaultSize, 0 );
	ThicknessLabel3->Wrap( -1 );
	ThicknessLabel3->SetFont( wxFont( 10, wxFONTFAMILY_SWISS, wxFONTSTYLE_NORMAL, wxFONTWEIGHT_BOLD, false, wxT("Sans") ) );

	InfoSizer->Add( ThicknessLabel3, 0, wxALIGN_RIGHT|wxALL, 5 );

	ThicknessMinResText = new wxStaticText( JobDetailsPanel, wxID_ANY, wxEmptyString, wxDefaultPosition, wxDefaultSize, 0 );
	ThicknessMinResText->Wrap( -1 );
	InfoSizer->Add( ThicknessMinResText, 0, wxALL, 5 );

	ThicknessLabel4 = new wxStaticText( JobDetailsPanel, wxID_ANY, wxT("Thickness Max. Res. :"), wxDefaultPosition, wxDefaultSize, 0 );
	ThicknessLabel4->Wrap( -1 );
	ThicknessLabel4->SetFont( wxFont( 10, wxFONTFAMILY_SWISS, wxFONTSTYLE_NORMAL, wxFONTWEIGHT_BOLD, false, wxT("Sans") ) );

	InfoSizer->Add( ThicknessLabel4, 0, wxALIGN_RIGHT|wxALL, 5 );

	ThicknessMaxResStaticText = new wxStaticText( JobDetailsPanel, wxID_ANY, wxEmptyString, wxDefaultPosition, wxDefaultSize, 0 );
	ThicknessMaxResStaticText->Wrap( -1 );
	InfoSizer->Add( ThicknessMaxResStaticText, 0, wxALL, 5 );

	ThicknessLabel5 = new wxStaticText( JobDetailsPanel, wxID_ANY, wxT("No-decay model? :"), wxDefaultPosition, wxDefaultSize, 0 );
	ThicknessLabel5->Wrap( -1 );
	ThicknessLabel5->SetFont( wxFont( 10, wxFONTFAMILY_SWISS, wxFONTSTYLE_NORMAL, wxFONTWEIGHT_BOLD, false, wxT("Sans") ) );

	InfoSizer->Add( ThicknessLabel5, 0, wxALIGN_RIGHT|wxALL, 5 );

	ThicknessNoDecayStaticText = new wxStaticText( JobDetailsPanel, wxID_ANY, wxEmptyString, wxDefaultPosition, wxDefaultSize, 0 );
	ThicknessNoDecayStaticText->Wrap( -1 );
	InfoSizer->Add( ThicknessNoDecayStaticText, 0, wxALL, 5 );

	ThicknessLabel6 = new wxStaticText( JobDetailsPanel, wxID_ANY, wxT("Downweight nodes? :"), wxDefaultPosition, wxDefaultSize, 0 );
	ThicknessLabel6->Wrap( -1 );
	ThicknessLabel6->SetFont( wxFont( 10, wxFONTFAMILY_SWISS, wxFONTSTYLE_NORMAL, wxFONTWEIGHT_BOLD, false, wxT("Sans") ) );

	InfoSizer->Add( ThicknessLabel6, 0, wxALIGN_RIGHT|wxALL, 5 );

	ThicknessDownweightNodesStaticText = new wxStaticText( JobDetailsPanel, wxID_ANY, wxEmptyString, wxDefaultPosition, wxDefaultSize, 0 );
	ThicknessDownweightNodesStaticText->Wrap( -1 );
	InfoSizer->Add( ThicknessDownweightNodesStaticText, 0, wxALIGN_RIGHT|wxALL, 5 );


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

	ResultPanel = new ShowCTFResultsPanel( RightPanel, wxID_ANY, wxDefaultPosition, wxDefaultSize, wxTAB_TRAVERSAL );
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
	this->Connect( wxEVT_UPDATE_UI, wxUpdateUIEventHandler( FindCTFResultsPanel::OnUpdateUI ) );
	AllImagesButton->Connect( wxEVT_COMMAND_RADIOBUTTON_SELECTED, wxCommandEventHandler( FindCTFResultsPanel::OnAllMoviesSelect ), NULL, this );
	ByFilterButton->Connect( wxEVT_COMMAND_RADIOBUTTON_SELECTED, wxCommandEventHandler( FindCTFResultsPanel::OnByFilterSelect ), NULL, this );
	FilterButton->Connect( wxEVT_COMMAND_BUTTON_CLICKED, wxCommandEventHandler( FindCTFResultsPanel::OnDefineFilterClick ), NULL, this );
	PlotResultsButton->Connect( wxEVT_COMMAND_BUTTON_CLICKED, wxCommandEventHandler( FindCTFResultsPanel::OnPlotResultsButtonClick ), NULL, this );
	JobDetailsToggleButton->Connect( wxEVT_COMMAND_TOGGLEBUTTON_CLICKED, wxCommandEventHandler( FindCTFResultsPanel::OnJobDetailsToggle ), NULL, this );
	PreviousButton->Connect( wxEVT_COMMAND_BUTTON_CLICKED, wxCommandEventHandler( FindCTFResultsPanel::OnPreviousButtonClick ), NULL, this );
	AddAllToGroupButton->Connect( wxEVT_COMMAND_BUTTON_CLICKED, wxCommandEventHandler( FindCTFResultsPanel::OnAddAllToGroupClick ), NULL, this );
	NextButton->Connect( wxEVT_COMMAND_BUTTON_CLICKED, wxCommandEventHandler( FindCTFResultsPanel::OnNextButtonClick ), NULL, this );
	DeleteFromGroupButton->Connect( wxEVT_COMMAND_BUTTON_CLICKED, wxCommandEventHandler( FindCTFResultsPanel::OnRemoveFromGroupClick ), NULL, this );
	AddToGroupButton->Connect( wxEVT_COMMAND_BUTTON_CLICKED, wxCommandEventHandler( FindCTFResultsPanel::OnAddToGroupClick ), NULL, this );
}

FindCTFResultsPanel::~FindCTFResultsPanel()
{
	// Disconnect Events
	this->Disconnect( wxEVT_UPDATE_UI, wxUpdateUIEventHandler( FindCTFResultsPanel::OnUpdateUI ) );
	AllImagesButton->Disconnect( wxEVT_COMMAND_RADIOBUTTON_SELECTED, wxCommandEventHandler( FindCTFResultsPanel::OnAllMoviesSelect ), NULL, this );
	ByFilterButton->Disconnect( wxEVT_COMMAND_RADIOBUTTON_SELECTED, wxCommandEventHandler( FindCTFResultsPanel::OnByFilterSelect ), NULL, this );
	FilterButton->Disconnect( wxEVT_COMMAND_BUTTON_CLICKED, wxCommandEventHandler( FindCTFResultsPanel::OnDefineFilterClick ), NULL, this );
	PlotResultsButton->Disconnect( wxEVT_COMMAND_BUTTON_CLICKED, wxCommandEventHandler( FindCTFResultsPanel::OnPlotResultsButtonClick ), NULL, this );
	JobDetailsToggleButton->Disconnect( wxEVT_COMMAND_TOGGLEBUTTON_CLICKED, wxCommandEventHandler( FindCTFResultsPanel::OnJobDetailsToggle ), NULL, this );
	PreviousButton->Disconnect( wxEVT_COMMAND_BUTTON_CLICKED, wxCommandEventHandler( FindCTFResultsPanel::OnPreviousButtonClick ), NULL, this );
	AddAllToGroupButton->Disconnect( wxEVT_COMMAND_BUTTON_CLICKED, wxCommandEventHandler( FindCTFResultsPanel::OnAddAllToGroupClick ), NULL, this );
	NextButton->Disconnect( wxEVT_COMMAND_BUTTON_CLICKED, wxCommandEventHandler( FindCTFResultsPanel::OnNextButtonClick ), NULL, this );
	DeleteFromGroupButton->Disconnect( wxEVT_COMMAND_BUTTON_CLICKED, wxCommandEventHandler( FindCTFResultsPanel::OnRemoveFromGroupClick ), NULL, this );
	AddToGroupButton->Disconnect( wxEVT_COMMAND_BUTTON_CLICKED, wxCommandEventHandler( FindCTFResultsPanel::OnAddToGroupClick ), NULL, this );

}

RefineCTFPanelParent::RefineCTFPanelParent( wxWindow* parent, wxWindowID id, const wxPoint& pos, const wxSize& size, long style, const wxString& name ) : JobPanel( parent, id, pos, size, style, name )
{
	wxBoxSizer* bSizer43;
	bSizer43 = new wxBoxSizer( wxVERTICAL );

	m_staticline12 = new wxStaticLine( this, wxID_ANY, wxDefaultPosition, wxDefaultSize, wxLI_HORIZONTAL );
	bSizer43->Add( m_staticline12, 0, wxEXPAND | wxALL, 5 );

	wxBoxSizer* bSizer451;
	bSizer451 = new wxBoxSizer( wxHORIZONTAL );

	InputParamsPanel = new wxPanel( this, wxID_ANY, wxDefaultPosition, wxDefaultSize, wxTAB_TRAVERSAL );
	wxBoxSizer* bSizer361;
	bSizer361 = new wxBoxSizer( wxVERTICAL );

	wxBoxSizer* bSizer441;
	bSizer441 = new wxBoxSizer( wxHORIZONTAL );

	wxBoxSizer* bSizer200;
	bSizer200 = new wxBoxSizer( wxHORIZONTAL );

	wxBoxSizer* bSizer359;
	bSizer359 = new wxBoxSizer( wxVERTICAL );


	bSizer200->Add( bSizer359, 1, wxEXPAND, 5 );

	wxFlexGridSizer* fgSizer15;
	fgSizer15 = new wxFlexGridSizer( 0, 2, 0, 0 );
	fgSizer15->SetFlexibleDirection( wxBOTH );
	fgSizer15->SetNonFlexibleGrowMode( wxFLEX_GROWMODE_SPECIFIED );

	m_staticText262 = new wxStaticText( InputParamsPanel, wxID_ANY, wxT("Input Refinement Package :"), wxDefaultPosition, wxSize( -1,-1 ), 0 );
	m_staticText262->Wrap( -1 );
	fgSizer15->Add( m_staticText262, 0, wxALIGN_CENTER_VERTICAL|wxALIGN_RIGHT|wxALL, 5 );

	RefinementPackageComboBox = new RefinementPackagePickerComboPanel( InputParamsPanel, wxID_ANY, wxDefaultPosition, wxDefaultSize, wxTAB_TRAVERSAL );
	RefinementPackageComboBox->SetMinSize( wxSize( 350,-1 ) );
	RefinementPackageComboBox->SetMaxSize( wxSize( 350,-1 ) );

	fgSizer15->Add( RefinementPackageComboBox, 1, wxEXPAND | wxALL, 5 );

	m_staticText263 = new wxStaticText( InputParamsPanel, wxID_ANY, wxT("Input Parameters :"), wxDefaultPosition, wxDefaultSize, 0 );
	m_staticText263->Wrap( -1 );
	fgSizer15->Add( m_staticText263, 0, wxALIGN_CENTER_VERTICAL|wxALIGN_RIGHT|wxALL, 5 );

	InputParametersComboBox = new RefinementPickerComboPanel( InputParamsPanel, wxID_ANY, wxDefaultPosition, wxDefaultSize, wxTAB_TRAVERSAL );
	InputParametersComboBox->SetMinSize( wxSize( 350,-1 ) );
	InputParametersComboBox->SetMaxSize( wxSize( 350,-1 ) );

	fgSizer15->Add( InputParametersComboBox, 1, wxEXPAND | wxALL, 5 );

	UseMaskCheckBox = new wxCheckBox( InputParamsPanel, wxID_ANY, wxT("Use a Mask? :"), wxDefaultPosition, wxDefaultSize, 0 );
	fgSizer15->Add( UseMaskCheckBox, 0, wxALIGN_CENTER_VERTICAL|wxALIGN_RIGHT|wxALL, 5 );

	MaskSelectPanel = new VolumeAssetPickerComboPanel( InputParamsPanel, wxID_ANY, wxDefaultPosition, wxDefaultSize, wxTAB_TRAVERSAL );
	MaskSelectPanel->SetMinSize( wxSize( 350,-1 ) );

	fgSizer15->Add( MaskSelectPanel, 1, wxALIGN_CENTER_VERTICAL|wxALL, 5 );


	bSizer200->Add( fgSizer15, 0, wxALIGN_CENTER, 5 );

	m_staticline52 = new wxStaticLine( InputParamsPanel, wxID_ANY, wxDefaultPosition, wxDefaultSize, wxLI_HORIZONTAL|wxLI_VERTICAL );
	bSizer200->Add( m_staticline52, 0, wxEXPAND | wxALL, 5 );

	wxBoxSizer* bSizer357;
	bSizer357 = new wxBoxSizer( wxVERTICAL );

	wxBoxSizer* bSizer215;
	bSizer215 = new wxBoxSizer( wxVERTICAL );

	RefineBeamTiltCheckBox = new wxCheckBox( InputParamsPanel, wxID_ANY, wxT("Refine Beam Tilt Parms."), wxDefaultPosition, wxDefaultSize, 0 );
	RefineBeamTiltCheckBox->SetValue(true);
	bSizer215->Add( RefineBeamTiltCheckBox, 0, wxALL, 5 );

	m_staticline142 = new wxStaticLine( InputParamsPanel, wxID_ANY, wxDefaultPosition, wxDefaultSize, wxLI_HORIZONTAL );
	bSizer215->Add( m_staticline142, 0, wxEXPAND | wxALL, 5 );

	RefineCTFCheckBox = new wxCheckBox( InputParamsPanel, wxID_ANY, wxT("Refine Defocus Params."), wxDefaultPosition, wxDefaultSize, 0 );
	RefineCTFCheckBox->SetValue(true);
	bSizer215->Add( RefineCTFCheckBox, 0, wxALL, 5 );

	wxBoxSizer* bSizer520;
	bSizer520 = new wxBoxSizer( wxHORIZONTAL );

	HiResLimitStaticText = new wxStaticText( InputParamsPanel, wxID_ANY, wxT("Hi-Res Limit (Å) :"), wxDefaultPosition, wxDefaultSize, 0 );
	HiResLimitStaticText->Wrap( -1 );
	bSizer520->Add( HiResLimitStaticText, 0, wxALIGN_CENTER_VERTICAL|wxALIGN_RIGHT|wxALL, 5 );

	HighResolutionLimitTextCtrl = new NumericTextCtrl( InputParamsPanel, wxID_ANY, wxT("3.5"), wxDefaultPosition, wxDefaultSize, 0 );
	bSizer520->Add( HighResolutionLimitTextCtrl, 0, wxALIGN_CENTER_VERTICAL|wxALL|wxEXPAND, 5 );


	bSizer215->Add( bSizer520, 1, wxEXPAND, 5 );

	m_staticline143 = new wxStaticLine( InputParamsPanel, wxID_ANY, wxDefaultPosition, wxDefaultSize, wxLI_HORIZONTAL );
	bSizer215->Add( m_staticline143, 0, wxEXPAND | wxALL, 5 );


	bSizer357->Add( bSizer215, 0, wxEXPAND, 5 );

	wxFlexGridSizer* fgSizer22;
	fgSizer22 = new wxFlexGridSizer( 0, 2, 0, 0 );
	fgSizer22->SetFlexibleDirection( wxBOTH );
	fgSizer22->SetNonFlexibleGrowMode( wxFLEX_GROWMODE_SPECIFIED );


	bSizer357->Add( fgSizer22, 1, wxEXPAND, 5 );

	wxBoxSizer* bSizer358;
	bSizer358 = new wxBoxSizer( wxHORIZONTAL );

	ExpertToggleButton = new wxToggleButton( InputParamsPanel, wxID_ANY, wxT("Show Expert Options"), wxDefaultPosition, wxDefaultSize, 0 );
	bSizer358->Add( ExpertToggleButton, 0, wxALIGN_BOTTOM|wxALIGN_CENTER|wxALL, 5 );


	bSizer357->Add( bSizer358, 0, wxALIGN_CENTER, 5 );


	bSizer200->Add( bSizer357, 1, wxEXPAND, 5 );

	m_staticline101 = new wxStaticLine( InputParamsPanel, wxID_ANY, wxDefaultPosition, wxDefaultSize, wxLI_HORIZONTAL|wxLI_VERTICAL );
	bSizer200->Add( m_staticline101, 0, wxEXPAND | wxALL, 5 );

	wxBoxSizer* bSizer364;
	bSizer364 = new wxBoxSizer( wxVERTICAL );

	Active3DReferencesListCtrl = new ReferenceVolumesListControlRefinement( InputParamsPanel, wxID_ANY, wxDefaultPosition, wxDefaultSize, wxLC_REPORT|wxLC_SINGLE_SEL|wxLC_VIRTUAL );
	Active3DReferencesListCtrl->SetMinSize( wxSize( -1,50 ) );

	bSizer364->Add( Active3DReferencesListCtrl, 30, wxALL|wxEXPAND, 5 );


	bSizer200->Add( bSizer364, 30, wxEXPAND, 5 );


	bSizer441->Add( bSizer200, 1, wxEXPAND, 5 );


	bSizer361->Add( bSizer441, 1, wxEXPAND, 5 );

	PleaseCreateRefinementPackageText = new wxStaticText( InputParamsPanel, wxID_ANY, wxT("Please create a refinement package (in the assets panel) in order to perform a 3D refinement."), wxDefaultPosition, wxDefaultSize, 0 );
	PleaseCreateRefinementPackageText->Wrap( -1 );
	PleaseCreateRefinementPackageText->SetFont( wxFont( wxNORMAL_FONT->GetPointSize(), wxFONTFAMILY_DEFAULT, wxFONTSTYLE_NORMAL, wxFONTWEIGHT_BOLD, false, wxEmptyString ) );
	PleaseCreateRefinementPackageText->SetForegroundColour( wxColour( 180, 0, 0 ) );
	PleaseCreateRefinementPackageText->Hide();

	bSizer361->Add( PleaseCreateRefinementPackageText, 0, wxALL, 5 );

	m_staticline10 = new wxStaticLine( InputParamsPanel, wxID_ANY, wxDefaultPosition, wxDefaultSize, wxLI_HORIZONTAL );
	bSizer361->Add( m_staticline10, 0, wxEXPAND | wxALL, 5 );


	InputParamsPanel->SetSizer( bSizer361 );
	InputParamsPanel->Layout();
	bSizer361->Fit( InputParamsPanel );
	bSizer451->Add( InputParamsPanel, 1, wxEXPAND | wxALL, 5 );


	bSizer43->Add( bSizer451, 0, wxEXPAND, 5 );

	wxBoxSizer* bSizer46;
	bSizer46 = new wxBoxSizer( wxHORIZONTAL );

	ExpertPanel = new wxScrolledWindow( this, wxID_ANY, wxDefaultPosition, wxSize( -1,-1 ), wxVSCROLL );
	ExpertPanel->SetScrollRate( 5, 5 );
	ExpertPanel->Hide();

	InputSizer = new wxBoxSizer( wxVERTICAL );

	wxBoxSizer* bSizer258;
	bSizer258 = new wxBoxSizer( wxHORIZONTAL );


	bSizer258->Add( 0, 0, 1, wxEXPAND, 5 );

	ResetAllDefaultsButton = new wxButton( ExpertPanel, wxID_ANY, wxT("Reset All Defaults"), wxDefaultPosition, wxDefaultSize, 0 );
	bSizer258->Add( ResetAllDefaultsButton, 0, wxALL, 5 );


	bSizer258->Add( 5, 0, 0, wxEXPAND, 5 );


	InputSizer->Add( bSizer258, 0, wxEXPAND, 5 );

	wxBoxSizer* bSizer257;
	bSizer257 = new wxBoxSizer( wxHORIZONTAL );

	wxGridSizer* gSizer14;
	gSizer14 = new wxGridSizer( 0, 3, 0, 0 );


	bSizer257->Add( gSizer14, 1, wxEXPAND, 5 );


	InputSizer->Add( bSizer257, 0, wxEXPAND, 5 );


	InputSizer->Add( 0, 5, 0, wxEXPAND, 5 );

	wxFlexGridSizer* fgSizer1;
	fgSizer1 = new wxFlexGridSizer( 0, 2, 0, 0 );
	fgSizer1->SetFlexibleDirection( wxBOTH );
	fgSizer1->SetNonFlexibleGrowMode( wxFLEX_GROWMODE_SPECIFIED );

	m_staticText202 = new wxStaticText( ExpertPanel, wxID_ANY, wxT("CTF Refinement"), wxDefaultPosition, wxDefaultSize, 0 );
	m_staticText202->Wrap( -1 );
	m_staticText202->SetFont( wxFont( wxNORMAL_FONT->GetPointSize(), wxFONTFAMILY_DEFAULT, wxFONTSTYLE_NORMAL, wxFONTWEIGHT_BOLD, true, wxEmptyString ) );

	fgSizer1->Add( m_staticText202, 0, wxALL, 5 );


	fgSizer1->Add( 0, 0, 1, wxEXPAND, 5 );

	NoMovieFramesStaticText = new wxStaticText( ExpertPanel, wxID_ANY, wxT("Low-Resolution Limit (Å) :"), wxDefaultPosition, wxDefaultSize, 0 );
	NoMovieFramesStaticText->Wrap( -1 );
	fgSizer1->Add( NoMovieFramesStaticText, 0, wxALIGN_CENTER_VERTICAL|wxALL, 5 );

	LowResolutionLimitTextCtrl = new NumericTextCtrl( ExpertPanel, wxID_ANY, wxT("300.00"), wxDefaultPosition, wxDefaultSize, 0 );
	fgSizer1->Add( LowResolutionLimitTextCtrl, 0, wxALL|wxEXPAND, 5 );

	m_staticText196 = new wxStaticText( ExpertPanel, wxID_ANY, wxT("Outer Mask Radius (Å) :"), wxDefaultPosition, wxDefaultSize, 0 );
	m_staticText196->Wrap( -1 );
	fgSizer1->Add( m_staticText196, 0, wxALIGN_CENTER_VERTICAL|wxALL, 5 );

	MaskRadiusTextCtrl = new NumericTextCtrl( ExpertPanel, wxID_ANY, wxT("100.00"), wxDefaultPosition, wxDefaultSize, wxTE_PROCESS_ENTER );
	fgSizer1->Add( MaskRadiusTextCtrl, 0, wxALL|wxEXPAND, 5 );

	m_staticText331 = new wxStaticText( ExpertPanel, wxID_ANY, wxT("Inner Mask Radius (Å) :"), wxDefaultPosition, wxDefaultSize, 0 );
	m_staticText331->Wrap( -1 );
	fgSizer1->Add( m_staticText331, 0, wxALIGN_CENTER_VERTICAL|wxALL, 5 );

	InnerMaskRadiusTextCtrl = new NumericTextCtrl( ExpertPanel, wxID_ANY, wxT("0.00"), wxDefaultPosition, wxDefaultSize, 0 );
	fgSizer1->Add( InnerMaskRadiusTextCtrl, 0, wxALL|wxEXPAND, 5 );

	m_staticText317 = new wxStaticText( ExpertPanel, wxID_ANY, wxT("Signed CC Resolution Limit (Å) :"), wxDefaultPosition, wxDefaultSize, 0 );
	m_staticText317->Wrap( -1 );
	fgSizer1->Add( m_staticText317, 0, wxALIGN_CENTER_VERTICAL|wxALL, 5 );

	SignedCCResolutionTextCtrl = new NumericTextCtrl( ExpertPanel, wxID_ANY, wxT("0.00"), wxDefaultPosition, wxDefaultSize, 0 );
	fgSizer1->Add( SignedCCResolutionTextCtrl, 0, wxALL|wxEXPAND, 5 );

	DefocusSearchRangeStaticText = new wxStaticText( ExpertPanel, wxID_ANY, wxT("Defocus Search Range (Å) :"), wxDefaultPosition, wxDefaultSize, 0 );
	DefocusSearchRangeStaticText->Wrap( -1 );
	fgSizer1->Add( DefocusSearchRangeStaticText, 0, wxALIGN_CENTER_VERTICAL|wxALL, 5 );

	DefocusSearchRangeTextCtrl = new NumericTextCtrl( ExpertPanel, wxID_ANY, wxT("500.00"), wxDefaultPosition, wxDefaultSize, 0 );
	fgSizer1->Add( DefocusSearchRangeTextCtrl, 0, wxALL|wxEXPAND, 5 );

	DefocusSearchStepStaticText = new wxStaticText( ExpertPanel, wxID_ANY, wxT("Defocus Search Step (Å) : "), wxDefaultPosition, wxDefaultSize, 0 );
	DefocusSearchStepStaticText->Wrap( -1 );
	fgSizer1->Add( DefocusSearchStepStaticText, 0, wxALIGN_CENTER_VERTICAL|wxALL, 5 );

	DefocusSearchStepTextCtrl = new NumericTextCtrl( ExpertPanel, wxID_ANY, wxT("100.00"), wxDefaultPosition, wxDefaultSize, 0 );
	fgSizer1->Add( DefocusSearchStepTextCtrl, 0, wxALL|wxEXPAND, 5 );

	m_staticText329 = new wxStaticText( ExpertPanel, wxID_ANY, wxT("Reconstruction"), wxDefaultPosition, wxDefaultSize, 0 );
	m_staticText329->Wrap( -1 );
	m_staticText329->SetFont( wxFont( 10, wxFONTFAMILY_SWISS, wxFONTSTYLE_NORMAL, wxFONTWEIGHT_BOLD, true, wxT("Sans") ) );

	fgSizer1->Add( m_staticText329, 0, wxALL, 5 );


	fgSizer1->Add( 0, 0, 1, wxEXPAND, 5 );

	m_staticText332 = new wxStaticText( ExpertPanel, wxID_ANY, wxT("Score to Weight Constant (Å²) :"), wxDefaultPosition, wxDefaultSize, 0 );
	m_staticText332->Wrap( -1 );
	fgSizer1->Add( m_staticText332, 0, wxALIGN_CENTER_VERTICAL|wxALL, 5 );

	ScoreToWeightConstantTextCtrl = new NumericTextCtrl( ExpertPanel, wxID_ANY, wxT("5.00"), wxDefaultPosition, wxDefaultSize, 0 );
	fgSizer1->Add( ScoreToWeightConstantTextCtrl, 0, wxALIGN_CENTER_VERTICAL|wxALL|wxEXPAND, 5 );

	m_staticText335 = new wxStaticText( ExpertPanel, wxID_ANY, wxT("Adjust Score for Defocus?"), wxDefaultPosition, wxDefaultSize, 0 );
	m_staticText335->Wrap( -1 );
	fgSizer1->Add( m_staticText335, 0, wxALIGN_CENTER_VERTICAL|wxALL, 5 );

	wxBoxSizer* bSizer265;
	bSizer265 = new wxBoxSizer( wxHORIZONTAL );

	AdjustScoreForDefocusYesRadio = new wxRadioButton( ExpertPanel, wxID_ANY, wxT("Yes"), wxDefaultPosition, wxDefaultSize, wxRB_GROUP );
	bSizer265->Add( AdjustScoreForDefocusYesRadio, 0, wxALL, 5 );

	AdjustScoreForDefocusNoRadio = new wxRadioButton( ExpertPanel, wxID_ANY, wxT("No"), wxDefaultPosition, wxDefaultSize, 0 );
	bSizer265->Add( AdjustScoreForDefocusNoRadio, 0, wxALL, 5 );


	fgSizer1->Add( bSizer265, 1, wxEXPAND, 5 );

	m_staticText333 = new wxStaticText( ExpertPanel, wxID_ANY, wxT("Score Threshold :"), wxDefaultPosition, wxDefaultSize, 0 );
	m_staticText333->Wrap( -1 );
	fgSizer1->Add( m_staticText333, 0, wxALIGN_CENTER_VERTICAL|wxALL, 5 );

	ReconstructionScoreThreshold = new NumericTextCtrl( ExpertPanel, wxID_ANY, wxT("-1.0"), wxDefaultPosition, wxDefaultSize, 0 );
	fgSizer1->Add( ReconstructionScoreThreshold, 0, wxALL|wxEXPAND, 5 );

	m_staticText334 = new wxStaticText( ExpertPanel, wxID_ANY, wxT("Resolution Limit (Å) :"), wxDefaultPosition, wxDefaultSize, 0 );
	m_staticText334->Wrap( -1 );
	fgSizer1->Add( m_staticText334, 0, wxALIGN_CENTER_VERTICAL|wxALL, 5 );

	ReconstructionResolutionLimitTextCtrl = new NumericTextCtrl( ExpertPanel, wxID_ANY, wxT("0.00"), wxDefaultPosition, wxDefaultSize, 0 );
	fgSizer1->Add( ReconstructionResolutionLimitTextCtrl, 0, wxALIGN_CENTER_VERTICAL|wxALL|wxEXPAND, 5 );

	m_staticText336 = new wxStaticText( ExpertPanel, wxID_ANY, wxT("Autocrop Images?"), wxDefaultPosition, wxDefaultSize, 0 );
	m_staticText336->Wrap( -1 );
	fgSizer1->Add( m_staticText336, 0, wxALIGN_CENTER_VERTICAL|wxALL, 5 );

	wxBoxSizer* bSizer266;
	bSizer266 = new wxBoxSizer( wxHORIZONTAL );

	AutoCropYesRadioButton = new wxRadioButton( ExpertPanel, wxID_ANY, wxT("Yes"), wxDefaultPosition, wxDefaultSize, wxRB_GROUP );
	bSizer266->Add( AutoCropYesRadioButton, 0, wxALL, 5 );

	AutoCropNoRadioButton = new wxRadioButton( ExpertPanel, wxID_ANY, wxT("No"), wxDefaultPosition, wxDefaultSize, 0 );
	bSizer266->Add( AutoCropNoRadioButton, 0, wxALL, 5 );


	fgSizer1->Add( bSizer266, 1, wxEXPAND, 5 );

	m_staticText363 = new wxStaticText( ExpertPanel, wxID_ANY, wxT("Apply Likelihood Blurring?"), wxDefaultPosition, wxDefaultSize, 0 );
	m_staticText363->Wrap( -1 );
	fgSizer1->Add( m_staticText363, 0, wxALIGN_CENTER_VERTICAL|wxALL, 5 );

	wxBoxSizer* bSizer2661;
	bSizer2661 = new wxBoxSizer( wxHORIZONTAL );

	ApplyBlurringYesRadioButton = new wxRadioButton( ExpertPanel, wxID_ANY, wxT("Yes"), wxDefaultPosition, wxDefaultSize, wxRB_GROUP );
	bSizer2661->Add( ApplyBlurringYesRadioButton, 0, wxALL, 5 );

	ApplyBlurringNoRadioButton = new wxRadioButton( ExpertPanel, wxID_ANY, wxT("No"), wxDefaultPosition, wxDefaultSize, 0 );
	bSizer2661->Add( ApplyBlurringNoRadioButton, 0, wxALL, 5 );


	fgSizer1->Add( bSizer2661, 1, wxEXPAND, 5 );

	SmoothingFactorStaticText = new wxStaticText( ExpertPanel, wxID_ANY, wxT("\tSmoothing Factor :"), wxDefaultPosition, wxDefaultSize, 0 );
	SmoothingFactorStaticText->Wrap( -1 );
	fgSizer1->Add( SmoothingFactorStaticText, 0, wxALIGN_CENTER_VERTICAL|wxALL, 5 );

	SmoothingFactorTextCtrl = new NumericTextCtrl( ExpertPanel, wxID_ANY, wxT("1.00"), wxDefaultPosition, wxDefaultSize, 0 );
	fgSizer1->Add( SmoothingFactorTextCtrl, 0, wxALL|wxEXPAND, 5 );

	m_staticText405 = new wxStaticText( ExpertPanel, wxID_ANY, wxT("Masking"), wxDefaultPosition, wxDefaultSize, 0 );
	m_staticText405->Wrap( -1 );
	m_staticText405->SetFont( wxFont( 10, wxFONTFAMILY_SWISS, wxFONTSTYLE_NORMAL, wxFONTWEIGHT_BOLD, true, wxT("Sans") ) );

	fgSizer1->Add( m_staticText405, 0, wxALL, 5 );


	fgSizer1->Add( 0, 0, 1, wxEXPAND, 5 );

	AutoMaskStaticText = new wxStaticText( ExpertPanel, wxID_ANY, wxT("Use AutoMasking?"), wxDefaultPosition, wxDefaultSize, 0 );
	AutoMaskStaticText->Wrap( -1 );
	fgSizer1->Add( AutoMaskStaticText, 0, wxALIGN_CENTER_VERTICAL|wxALL, 5 );

	wxBoxSizer* bSizer26612;
	bSizer26612 = new wxBoxSizer( wxHORIZONTAL );

	AutoMaskYesRadioButton = new wxRadioButton( ExpertPanel, wxID_ANY, wxT("Yes"), wxDefaultPosition, wxDefaultSize, wxRB_GROUP );
	bSizer26612->Add( AutoMaskYesRadioButton, 0, wxALL, 5 );

	AutoMaskNoRadioButton = new wxRadioButton( ExpertPanel, wxID_ANY, wxT("No"), wxDefaultPosition, wxDefaultSize, 0 );
	bSizer26612->Add( AutoMaskNoRadioButton, 0, wxALL, 5 );


	fgSizer1->Add( bSizer26612, 1, wxEXPAND, 5 );

	MaskEdgeStaticText = new wxStaticText( ExpertPanel, wxID_ANY, wxT("Mask Edge Width (Å) :"), wxDefaultPosition, wxDefaultSize, 0 );
	MaskEdgeStaticText->Wrap( -1 );
	fgSizer1->Add( MaskEdgeStaticText, 0, wxALIGN_CENTER_VERTICAL|wxALL, 5 );

	MaskEdgeTextCtrl = new NumericTextCtrl( ExpertPanel, wxID_ANY, wxT("10.00"), wxDefaultPosition, wxDefaultSize, 0 );
	fgSizer1->Add( MaskEdgeTextCtrl, 0, wxALL, 5 );

	MaskWeightStaticText = new wxStaticText( ExpertPanel, wxID_ANY, wxT("Outside Weight :"), wxDefaultPosition, wxDefaultSize, 0 );
	MaskWeightStaticText->Wrap( -1 );
	fgSizer1->Add( MaskWeightStaticText, 0, wxALIGN_CENTER_VERTICAL|wxALL, 5 );

	MaskWeightTextCtrl = new NumericTextCtrl( ExpertPanel, wxID_ANY, wxT("0.00"), wxDefaultPosition, wxDefaultSize, 0 );
	fgSizer1->Add( MaskWeightTextCtrl, 0, wxALL, 5 );

	LowPassYesNoStaticText = new wxStaticText( ExpertPanel, wxID_ANY, wxT("Low-Pass Filter Outside Mask?"), wxDefaultPosition, wxDefaultSize, 0 );
	LowPassYesNoStaticText->Wrap( -1 );
	fgSizer1->Add( LowPassYesNoStaticText, 0, wxALIGN_CENTER_VERTICAL|wxALL, 5 );

	wxBoxSizer* bSizer26611;
	bSizer26611 = new wxBoxSizer( wxHORIZONTAL );

	LowPassMaskYesRadio = new wxRadioButton( ExpertPanel, wxID_ANY, wxT("Yes"), wxDefaultPosition, wxDefaultSize, wxRB_GROUP );
	bSizer26611->Add( LowPassMaskYesRadio, 0, wxALL, 5 );

	LowPassMaskNoRadio = new wxRadioButton( ExpertPanel, wxID_ANY, wxT("No"), wxDefaultPosition, wxDefaultSize, 0 );
	bSizer26611->Add( LowPassMaskNoRadio, 0, wxALL, 5 );


	fgSizer1->Add( bSizer26611, 1, wxEXPAND, 5 );

	FilterResolutionStaticText = new wxStaticText( ExpertPanel, wxID_ANY, wxT("\tFilter Resolution (Å) :"), wxDefaultPosition, wxDefaultSize, 0 );
	FilterResolutionStaticText->Wrap( -1 );
	fgSizer1->Add( FilterResolutionStaticText, 0, wxALIGN_CENTER_VERTICAL|wxALL, 5 );

	MaskFilterResolutionText = new NumericTextCtrl( ExpertPanel, wxID_ANY, wxT("20.00"), wxDefaultPosition, wxDefaultSize, 0 );
	fgSizer1->Add( MaskFilterResolutionText, 0, wxALL, 5 );


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
	bSizer61 = new wxBoxSizer( wxHORIZONTAL );

	InfoText = new wxRichTextCtrl( InfoPanel, wxID_ANY, wxEmptyString, wxDefaultPosition, wxDefaultSize, wxTE_READONLY|wxHSCROLL|wxVSCROLL );
	bSizer61->Add( InfoText, 70, wxEXPAND | wxALL, 5 );


	InfoPanel->SetSizer( bSizer61 );
	InfoPanel->Layout();
	bSizer61->Fit( InfoPanel );
	bSizer46->Add( InfoPanel, 70, wxEXPAND | wxALL, 5 );

	ShowRefinementResultsPanel = new DisplayCTFRefinementResultsPanel( this, wxID_ANY, wxDefaultPosition, wxDefaultSize, wxTAB_TRAVERSAL );
	ShowRefinementResultsPanel->Hide();

	bSizer46->Add( ShowRefinementResultsPanel, 80, wxEXPAND | wxALL, 5 );


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

	NumberConnectedText = new wxStaticText( ProgressPanel, wxID_ANY, wxEmptyString, wxDefaultPosition, wxDefaultSize, 0 );
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

	wxBoxSizer* bSizer268;
	bSizer268 = new wxBoxSizer( wxVERTICAL );

	wxBoxSizer* bSizer267;
	bSizer267 = new wxBoxSizer( wxHORIZONTAL );

	RunProfileText = new wxStaticText( StartPanel, wxID_ANY, wxT("Refinement Run Profile :"), wxDefaultPosition, wxDefaultSize, 0 );
	RunProfileText->Wrap( -1 );
	bSizer267->Add( RunProfileText, 20, wxALIGN_CENTER_VERTICAL|wxALL, 5 );

	RefinementRunProfileComboBox = new MemoryComboBox( StartPanel, wxID_ANY, wxEmptyString, wxDefaultPosition, wxDefaultSize, 0, NULL, wxCB_READONLY );
	bSizer267->Add( RefinementRunProfileComboBox, 50, wxALIGN_CENTER_VERTICAL|wxALL|wxEXPAND, 5 );


	bSizer268->Add( bSizer267, 50, wxEXPAND, 5 );

	wxBoxSizer* bSizer2671;
	bSizer2671 = new wxBoxSizer( wxHORIZONTAL );

	RunProfileText1 = new wxStaticText( StartPanel, wxID_ANY, wxT("Reconstruction Run Profile :"), wxDefaultPosition, wxDefaultSize, 0 );
	RunProfileText1->Wrap( -1 );
	bSizer2671->Add( RunProfileText1, 20, wxALIGN_CENTER_VERTICAL|wxALL, 5 );

	ReconstructionRunProfileComboBox = new MemoryComboBox( StartPanel, wxID_ANY, wxEmptyString, wxDefaultPosition, wxDefaultSize, 0, NULL, wxCB_READONLY );
	bSizer2671->Add( ReconstructionRunProfileComboBox, 50, wxALIGN_CENTER_VERTICAL|wxALL|wxEXPAND, 5 );


	bSizer268->Add( bSizer2671, 0, wxEXPAND, 5 );


	bSizer58->Add( bSizer268, 50, wxEXPAND, 5 );

	wxBoxSizer* bSizer60;
	bSizer60 = new wxBoxSizer( wxVERTICAL );


	bSizer60->Add( 0, 0, 1, wxEXPAND, 5 );

	StartRefinementButton = new wxButton( StartPanel, wxID_ANY, wxT("Start Refinement"), wxDefaultPosition, wxDefaultSize, 0 );
	bSizer60->Add( StartRefinementButton, 0, wxALL, 5 );


	bSizer58->Add( bSizer60, 50, wxEXPAND, 5 );


	StartPanel->SetSizer( bSizer58 );
	StartPanel->Layout();
	bSizer58->Fit( StartPanel );
	bSizer48->Add( StartPanel, 1, wxEXPAND | wxALL, 5 );


	bSizer43->Add( bSizer48, 0, wxEXPAND, 5 );


	this->SetSizer( bSizer43 );
	this->Layout();

	// Connect Events
	this->Connect( wxEVT_UPDATE_UI, wxUpdateUIEventHandler( RefineCTFPanelParent::OnUpdateUI ) );
	UseMaskCheckBox->Connect( wxEVT_COMMAND_CHECKBOX_CLICKED, wxCommandEventHandler( RefineCTFPanelParent::OnUseMaskCheckBox ), NULL, this );
	HighResolutionLimitTextCtrl->Connect( wxEVT_COMMAND_TEXT_UPDATED, wxCommandEventHandler( RefineCTFPanelParent::OnHighResLimitChange ), NULL, this );
	ExpertToggleButton->Connect( wxEVT_COMMAND_TOGGLEBUTTON_CLICKED, wxCommandEventHandler( RefineCTFPanelParent::OnExpertOptionsToggle ), NULL, this );
	Active3DReferencesListCtrl->Connect( wxEVT_COMMAND_LIST_ITEM_ACTIVATED, wxListEventHandler( RefineCTFPanelParent::OnVolumeListItemActivated ), NULL, this );
	ResetAllDefaultsButton->Connect( wxEVT_COMMAND_BUTTON_CLICKED, wxCommandEventHandler( RefineCTFPanelParent::ResetAllDefaultsClick ), NULL, this );
	AutoMaskYesRadioButton->Connect( wxEVT_COMMAND_RADIOBUTTON_SELECTED, wxCommandEventHandler( RefineCTFPanelParent::OnAutoMaskButton ), NULL, this );
	AutoMaskNoRadioButton->Connect( wxEVT_COMMAND_RADIOBUTTON_SELECTED, wxCommandEventHandler( RefineCTFPanelParent::OnAutoMaskButton ), NULL, this );
	InfoText->Connect( wxEVT_COMMAND_TEXT_URL, wxTextUrlEventHandler( RefineCTFPanelParent::OnInfoURL ), NULL, this );
	FinishButton->Connect( wxEVT_COMMAND_BUTTON_CLICKED, wxCommandEventHandler( RefineCTFPanelParent::FinishButtonClick ), NULL, this );
	CancelAlignmentButton->Connect( wxEVT_COMMAND_BUTTON_CLICKED, wxCommandEventHandler( RefineCTFPanelParent::TerminateButtonClick ), NULL, this );
	StartRefinementButton->Connect( wxEVT_COMMAND_BUTTON_CLICKED, wxCommandEventHandler( RefineCTFPanelParent::StartRefinementClick ), NULL, this );
}

RefineCTFPanelParent::~RefineCTFPanelParent()
{
	// Disconnect Events
	this->Disconnect( wxEVT_UPDATE_UI, wxUpdateUIEventHandler( RefineCTFPanelParent::OnUpdateUI ) );
	UseMaskCheckBox->Disconnect( wxEVT_COMMAND_CHECKBOX_CLICKED, wxCommandEventHandler( RefineCTFPanelParent::OnUseMaskCheckBox ), NULL, this );
	HighResolutionLimitTextCtrl->Disconnect( wxEVT_COMMAND_TEXT_UPDATED, wxCommandEventHandler( RefineCTFPanelParent::OnHighResLimitChange ), NULL, this );
	ExpertToggleButton->Disconnect( wxEVT_COMMAND_TOGGLEBUTTON_CLICKED, wxCommandEventHandler( RefineCTFPanelParent::OnExpertOptionsToggle ), NULL, this );
	Active3DReferencesListCtrl->Disconnect( wxEVT_COMMAND_LIST_ITEM_ACTIVATED, wxListEventHandler( RefineCTFPanelParent::OnVolumeListItemActivated ), NULL, this );
	ResetAllDefaultsButton->Disconnect( wxEVT_COMMAND_BUTTON_CLICKED, wxCommandEventHandler( RefineCTFPanelParent::ResetAllDefaultsClick ), NULL, this );
	AutoMaskYesRadioButton->Disconnect( wxEVT_COMMAND_RADIOBUTTON_SELECTED, wxCommandEventHandler( RefineCTFPanelParent::OnAutoMaskButton ), NULL, this );
	AutoMaskNoRadioButton->Disconnect( wxEVT_COMMAND_RADIOBUTTON_SELECTED, wxCommandEventHandler( RefineCTFPanelParent::OnAutoMaskButton ), NULL, this );
	InfoText->Disconnect( wxEVT_COMMAND_TEXT_URL, wxTextUrlEventHandler( RefineCTFPanelParent::OnInfoURL ), NULL, this );
	FinishButton->Disconnect( wxEVT_COMMAND_BUTTON_CLICKED, wxCommandEventHandler( RefineCTFPanelParent::FinishButtonClick ), NULL, this );
	CancelAlignmentButton->Disconnect( wxEVT_COMMAND_BUTTON_CLICKED, wxCommandEventHandler( RefineCTFPanelParent::TerminateButtonClick ), NULL, this );
	StartRefinementButton->Disconnect( wxEVT_COMMAND_BUTTON_CLICKED, wxCommandEventHandler( RefineCTFPanelParent::StartRefinementClick ), NULL, this );

}

FindCTFPanel::FindCTFPanel( wxWindow* parent, wxWindowID id, const wxPoint& pos, const wxSize& size, long style, const wxString& name ) : JobPanel( parent, id, pos, size, style, name )
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

	GroupComboBox = new ImageGroupPickerComboPanel( this, wxID_ANY, wxDefaultPosition, wxDefaultSize, wxTAB_TRAVERSAL );
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
	InputSizer = new wxBoxSizer( wxVERTICAL );

	wxFlexGridSizer* fgSizer1;
	fgSizer1 = new wxFlexGridSizer( 0, 2, 0, 0 );
	fgSizer1->SetFlexibleDirection( wxBOTH );
	fgSizer1->SetNonFlexibleGrowMode( wxFLEX_GROWMODE_SPECIFIED );

	m_staticText202 = new wxStaticText( ExpertPanel, wxID_ANY, wxT("Search"), wxDefaultPosition, wxDefaultSize, 0 );
	m_staticText202->Wrap( -1 );
	m_staticText202->SetFont( wxFont( wxNORMAL_FONT->GetPointSize(), wxFONTFAMILY_DEFAULT, wxFONTSTYLE_NORMAL, wxFONTWEIGHT_BOLD, true, wxEmptyString ) );

	fgSizer1->Add( m_staticText202, 0, wxALL, 5 );


	fgSizer1->Add( 0, 0, 1, wxEXPAND, 5 );

	m_staticText186 = new wxStaticText( ExpertPanel, wxID_ANY, wxT("Estimate Using :"), wxDefaultPosition, wxDefaultSize, 0 );
	m_staticText186->Wrap( -1 );
	fgSizer1->Add( m_staticText186, 0, wxALIGN_CENTER_VERTICAL|wxALL, 5 );

	wxBoxSizer* bSizer123;
	bSizer123 = new wxBoxSizer( wxHORIZONTAL );

	MovieRadioButton = new wxRadioButton( ExpertPanel, wxID_ANY, wxT("Movies"), wxDefaultPosition, wxDefaultSize, wxRB_GROUP );
	bSizer123->Add( MovieRadioButton, 0, wxALL, 5 );

	ImageRadioButton = new wxRadioButton( ExpertPanel, wxID_ANY, wxT("Images"), wxDefaultPosition, wxDefaultSize, 0 );
	bSizer123->Add( ImageRadioButton, 0, wxALL, 5 );


	fgSizer1->Add( bSizer123, 1, wxEXPAND, 5 );

	NoMovieFramesStaticText = new wxStaticText( ExpertPanel, wxID_ANY, wxT("No. Movie Frames to Average :"), wxDefaultPosition, wxDefaultSize, 0 );
	NoMovieFramesStaticText->Wrap( -1 );
	fgSizer1->Add( NoMovieFramesStaticText, 0, wxALIGN_CENTER_VERTICAL|wxALL, 5 );

	NoFramesToAverageSpinCtrl = new wxSpinCtrl( ExpertPanel, wxID_ANY, wxEmptyString, wxDefaultPosition, wxDefaultSize, wxSP_ARROW_KEYS, 1, 999999999, 3 );
	fgSizer1->Add( NoFramesToAverageSpinCtrl, 0, wxALL|wxEXPAND, 5 );

	m_staticText188 = new wxStaticText( ExpertPanel, wxID_ANY, wxT("Box Size (px) :"), wxDefaultPosition, wxDefaultSize, 0 );
	m_staticText188->Wrap( -1 );
	fgSizer1->Add( m_staticText188, 0, wxALIGN_CENTER_VERTICAL|wxALL, 5 );

	BoxSizeSpinCtrl = new wxSpinCtrl( ExpertPanel, wxID_ANY, wxEmptyString, wxDefaultPosition, wxDefaultSize, wxSP_ARROW_KEYS, 128, 99999999, 512 );
	fgSizer1->Add( BoxSizeSpinCtrl, 0, wxALL|wxEXPAND, 5 );

	m_staticText196 = new wxStaticText( ExpertPanel, wxID_ANY, wxT("Amplitude Contrast :"), wxDefaultPosition, wxDefaultSize, 0 );
	m_staticText196->Wrap( -1 );
	fgSizer1->Add( m_staticText196, 0, wxALIGN_CENTER_VERTICAL|wxALL, 5 );

	AmplitudeContrastNumericCtrl = new NumericTextCtrl( ExpertPanel, wxID_ANY, wxT("0.07"), wxDefaultPosition, wxDefaultSize, wxTE_PROCESS_ENTER );
	fgSizer1->Add( AmplitudeContrastNumericCtrl, 0, wxALL|wxEXPAND, 5 );

	TiltStaticText = new wxStaticText( ExpertPanel, wxID_ANY, wxT("Also Search for Tilt?"), wxDefaultPosition, wxDefaultSize, 0 );
	TiltStaticText->Wrap( -1 );
	fgSizer1->Add( TiltStaticText, 0, wxALIGN_CENTER_VERTICAL|wxALL, 5 );

	wxBoxSizer* bSizer1231;
	bSizer1231 = new wxBoxSizer( wxHORIZONTAL );

	SearchTiltYesRadio = new wxRadioButton( ExpertPanel, wxID_ANY, wxT("Yes"), wxDefaultPosition, wxDefaultSize, wxRB_GROUP );
	bSizer1231->Add( SearchTiltYesRadio, 0, wxALL, 5 );

	SearchTiltNoRadio = new wxRadioButton( ExpertPanel, wxID_ANY, wxT("No"), wxDefaultPosition, wxDefaultSize, 0 );
	bSizer1231->Add( SearchTiltNoRadio, 0, wxALL, 5 );


	fgSizer1->Add( bSizer1231, 1, wxEXPAND, 5 );

	ResampleStaticText = new wxStaticText( ExpertPanel, wxID_ANY, wxT("Resample Pixel Size (Å)"), wxDefaultPosition, wxDefaultSize, 0 );
	ResampleStaticText->Wrap( -1 );
	fgSizer1->Add( ResampleStaticText, 0, wxALL, 5 );

	wxBoxSizer* bSizer12311;
	bSizer12311 = new wxBoxSizer( wxHORIZONTAL );

	ResamplePixelSizeCheckBox = new wxCheckBox( ExpertPanel, wxID_ANY, wxEmptyString, wxDefaultPosition, wxDefaultSize, 0 );
	ResamplePixelSizeCheckBox->SetValue(true);
	bSizer12311->Add( ResamplePixelSizeCheckBox, 0, wxALL, 5 );

	ResamplePixelSizeNumericCtrl = new NumericTextCtrl( ExpertPanel, wxID_ANY, wxT("1.4"), wxDefaultPosition, wxDefaultSize, wxTE_PROCESS_ENTER );
	bSizer12311->Add( ResamplePixelSizeNumericCtrl, 1, wxALL, 5 );


	fgSizer1->Add( bSizer12311, 1, wxEXPAND, 5 );

	m_staticText201 = new wxStaticText( ExpertPanel, wxID_ANY, wxT("Search Limits"), wxDefaultPosition, wxDefaultSize, 0 );
	m_staticText201->Wrap( -1 );
	m_staticText201->SetFont( wxFont( wxNORMAL_FONT->GetPointSize(), wxFONTFAMILY_DEFAULT, wxFONTSTYLE_NORMAL, wxFONTWEIGHT_BOLD, true, wxEmptyString ) );

	fgSizer1->Add( m_staticText201, 0, wxALL, 5 );


	fgSizer1->Add( 0, 0, 1, wxEXPAND, 5 );

	m_staticText189 = new wxStaticText( ExpertPanel, wxID_ANY, wxT("Min. Resolution of Fit (Å) :"), wxDefaultPosition, wxDefaultSize, 0 );
	m_staticText189->Wrap( -1 );
	fgSizer1->Add( m_staticText189, 0, wxALIGN_CENTER_VERTICAL|wxALL, 5 );

	MinResNumericCtrl = new NumericTextCtrl( ExpertPanel, wxID_ANY, wxT("30.00"), wxDefaultPosition, wxDefaultSize, wxTE_PROCESS_ENTER );
	fgSizer1->Add( MinResNumericCtrl, 0, wxALL|wxEXPAND, 5 );

	m_staticText190 = new wxStaticText( ExpertPanel, wxID_ANY, wxT("Max. Resolution of Fit (Å) :"), wxDefaultPosition, wxDefaultSize, 0 );
	m_staticText190->Wrap( -1 );
	fgSizer1->Add( m_staticText190, 0, wxALIGN_CENTER_VERTICAL|wxALL, 5 );

	MaxResNumericCtrl = new NumericTextCtrl( ExpertPanel, wxID_ANY, wxT("5.00"), wxDefaultPosition, wxDefaultSize, wxTE_PROCESS_ENTER );
	fgSizer1->Add( MaxResNumericCtrl, 0, wxALL|wxEXPAND, 5 );

	m_staticText191 = new wxStaticText( ExpertPanel, wxID_ANY, wxT("Low Defocus For Search (Å) :"), wxDefaultPosition, wxDefaultSize, 0 );
	m_staticText191->Wrap( -1 );
	fgSizer1->Add( m_staticText191, 0, wxALIGN_CENTER_VERTICAL|wxALL, 5 );

	LowDefocusNumericCtrl = new NumericTextCtrl( ExpertPanel, wxID_ANY, wxT("5000.00"), wxDefaultPosition, wxDefaultSize, wxTE_PROCESS_ENTER );
	fgSizer1->Add( LowDefocusNumericCtrl, 0, wxALL|wxEXPAND, 5 );

	m_staticText192 = new wxStaticText( ExpertPanel, wxID_ANY, wxT("High Defocus For Search (Å) :"), wxDefaultPosition, wxDefaultSize, 0 );
	m_staticText192->Wrap( -1 );
	fgSizer1->Add( m_staticText192, 0, wxALIGN_CENTER_VERTICAL|wxALL, 5 );

	HighDefocusNumericCtrl = new NumericTextCtrl( ExpertPanel, wxID_ANY, wxT("50000.00"), wxDefaultPosition, wxDefaultSize, wxTE_PROCESS_ENTER );
	fgSizer1->Add( HighDefocusNumericCtrl, 0, wxALL|wxEXPAND, 5 );

	m_staticText194 = new wxStaticText( ExpertPanel, wxID_ANY, wxT("Defocus Search Step (Å) :"), wxDefaultPosition, wxDefaultSize, 0 );
	m_staticText194->Wrap( -1 );
	fgSizer1->Add( m_staticText194, 0, wxALIGN_CENTER_VERTICAL|wxALL, 5 );

	DefocusStepNumericCtrl = new NumericTextCtrl( ExpertPanel, wxID_ANY, wxT("100.00"), wxDefaultPosition, wxDefaultSize, wxTE_PROCESS_ENTER );
	fgSizer1->Add( DefocusStepNumericCtrl, 0, wxALL|wxEXPAND, 5 );

	LargeAstigmatismExpectedCheckBox = new wxCheckBox( ExpertPanel, wxID_ANY, wxT("Use Slower, More Exhaustive Search?"), wxDefaultPosition, wxDefaultSize, 0 );
	fgSizer1->Add( LargeAstigmatismExpectedCheckBox, 0, wxALL, 5 );


	fgSizer1->Add( 0, 0, 1, wxEXPAND, 5 );

	RestrainAstigmatismCheckBox = new wxCheckBox( ExpertPanel, wxID_ANY, wxT("Restrain Astigmatism?"), wxDefaultPosition, wxDefaultSize, 0 );
	fgSizer1->Add( RestrainAstigmatismCheckBox, 0, wxALL, 5 );


	fgSizer1->Add( 0, 0, 1, wxEXPAND, 5 );

	ToleratedAstigmatismStaticText = new wxStaticText( ExpertPanel, wxID_ANY, wxT("Tolerated Astigmatism (Å) :"), wxDefaultPosition, wxDefaultSize, 0 );
	ToleratedAstigmatismStaticText->Wrap( -1 );
	fgSizer1->Add( ToleratedAstigmatismStaticText, 0, wxALIGN_CENTER_VERTICAL|wxALL, 5 );

	ToleratedAstigmatismNumericCtrl = new NumericTextCtrl( ExpertPanel, wxID_ANY, wxT("500.00"), wxDefaultPosition, wxDefaultSize, wxTE_PROCESS_ENTER );
	fgSizer1->Add( ToleratedAstigmatismNumericCtrl, 0, wxALL|wxEXPAND, 5 );

	m_staticText200 = new wxStaticText( ExpertPanel, wxID_ANY, wxT("Phase Plates"), wxDefaultPosition, wxDefaultSize, 0 );
	m_staticText200->Wrap( -1 );
	m_staticText200->SetFont( wxFont( 10, wxFONTFAMILY_SWISS, wxFONTSTYLE_NORMAL, wxFONTWEIGHT_BOLD, true, wxT("Sans") ) );

	fgSizer1->Add( m_staticText200, 0, wxALL, 5 );


	fgSizer1->Add( 0, 0, 1, wxEXPAND, 5 );

	AdditionalPhaseShiftCheckBox = new wxCheckBox( ExpertPanel, wxID_ANY, wxT("Find Additional Phase Shift?"), wxDefaultPosition, wxDefaultSize, 0 );
	fgSizer1->Add( AdditionalPhaseShiftCheckBox, 0, wxALL, 5 );


	fgSizer1->Add( 0, 0, 1, wxEXPAND, 5 );

	MinPhaseShiftStaticText = new wxStaticText( ExpertPanel, wxID_ANY, wxT("Min. Phase Shift (°) :"), wxDefaultPosition, wxDefaultSize, 0 );
	MinPhaseShiftStaticText->Wrap( -1 );
	MinPhaseShiftStaticText->Enable( false );

	fgSizer1->Add( MinPhaseShiftStaticText, 0, wxALIGN_CENTER_VERTICAL|wxALL, 5 );

	MinPhaseShiftNumericCtrl = new NumericTextCtrl( ExpertPanel, wxID_ANY, wxT("0.00"), wxDefaultPosition, wxDefaultSize, wxTE_PROCESS_ENTER );
	MinPhaseShiftNumericCtrl->Enable( false );

	fgSizer1->Add( MinPhaseShiftNumericCtrl, 0, wxALL|wxEXPAND, 5 );

	MaxPhaseShiftStaticText = new wxStaticText( ExpertPanel, wxID_ANY, wxT("Max. Phase Shift (°) :"), wxDefaultPosition, wxDefaultSize, 0 );
	MaxPhaseShiftStaticText->Wrap( -1 );
	MaxPhaseShiftStaticText->Enable( false );

	fgSizer1->Add( MaxPhaseShiftStaticText, 0, wxALIGN_CENTER_VERTICAL|wxALL, 5 );

	MaxPhaseShiftNumericCtrl = new NumericTextCtrl( ExpertPanel, wxID_ANY, wxT("180.00"), wxDefaultPosition, wxDefaultSize, wxTE_PROCESS_ENTER );
	MaxPhaseShiftNumericCtrl->Enable( false );

	fgSizer1->Add( MaxPhaseShiftNumericCtrl, 0, wxALL|wxEXPAND, 5 );

	PhaseShiftStepStaticText = new wxStaticText( ExpertPanel, wxID_ANY, wxT("Phase Shift Search Step (°) :"), wxDefaultPosition, wxDefaultSize, 0 );
	PhaseShiftStepStaticText->Wrap( -1 );
	PhaseShiftStepStaticText->Enable( false );

	fgSizer1->Add( PhaseShiftStepStaticText, 0, wxALIGN_CENTER_VERTICAL|wxALL, 5 );

	PhaseShiftStepNumericCtrl = new NumericTextCtrl( ExpertPanel, wxID_ANY, wxT("10.00"), wxDefaultPosition, wxDefaultSize, wxTE_PROCESS_ENTER );
	PhaseShiftStepNumericCtrl->Enable( false );

	fgSizer1->Add( PhaseShiftStepNumericCtrl, 0, wxALL|wxEXPAND, 5 );

	m_staticText2001 = new wxStaticText( ExpertPanel, wxID_ANY, wxT("Estimate Sample Thickness"), wxDefaultPosition, wxDefaultSize, 0 );
	m_staticText2001->Wrap( -1 );
	m_staticText2001->SetFont( wxFont( 10, wxFONTFAMILY_SWISS, wxFONTSTYLE_NORMAL, wxFONTWEIGHT_BOLD, true, wxT("Sans") ) );

	fgSizer1->Add( m_staticText2001, 0, wxALL, 5 );


	fgSizer1->Add( 0, 0, 1, wxEXPAND, 5 );

	FitNodesCheckBox = new wxCheckBox( ExpertPanel, wxID_ANY, wxT("Estimate Sample Thickness?"), wxDefaultPosition, wxDefaultSize, 0 );
	fgSizer1->Add( FitNodesCheckBox, 0, wxALL, 5 );


	fgSizer1->Add( 0, 0, 1, wxEXPAND, 5 );

	FitNodes1DCheckBox = new wxCheckBox( ExpertPanel, wxID_ANY, wxT("Do a brute-force search for sample thickness?"), wxDefaultPosition, wxDefaultSize, 0 );
	FitNodes1DCheckBox->SetValue(true);
	FitNodes1DCheckBox->Enable( false );

	fgSizer1->Add( FitNodes1DCheckBox, 0, wxALL, 5 );


	fgSizer1->Add( 0, 0, 1, wxEXPAND, 5 );

	FitNodes2DCheckBox = new wxCheckBox( ExpertPanel, wxID_ANY, wxT("Refine CTF and Astigmatism?"), wxDefaultPosition, wxDefaultSize, 0 );
	FitNodes2DCheckBox->SetValue(true);
	FitNodes2DCheckBox->Enable( false );

	fgSizer1->Add( FitNodes2DCheckBox, 0, wxALL, 5 );


	fgSizer1->Add( 0, 0, 1, wxEXPAND, 5 );

	FitNodesMinResStaticText = new wxStaticText( ExpertPanel, wxID_ANY, wxT("Min. Resolution of Fit (Å) :"), wxDefaultPosition, wxDefaultSize, 0 );
	FitNodesMinResStaticText->Wrap( -1 );
	FitNodesMinResStaticText->Enable( false );

	fgSizer1->Add( FitNodesMinResStaticText, 0, wxALL, 5 );

	FitNodesMinResNumericCtrl = new NumericTextCtrl( ExpertPanel, wxID_ANY, wxT("30.00"), wxDefaultPosition, wxDefaultSize, wxTE_PROCESS_ENTER );
	FitNodesMinResNumericCtrl->Enable( false );

	fgSizer1->Add( FitNodesMinResNumericCtrl, 0, wxALL, 5 );

	FitNodesMaxResStaticText = new wxStaticText( ExpertPanel, wxID_ANY, wxT("Max. Resolution of Fit (Å) :"), wxDefaultPosition, wxDefaultSize, 0 );
	FitNodesMaxResStaticText->Wrap( -1 );
	FitNodesMaxResStaticText->Enable( false );

	fgSizer1->Add( FitNodesMaxResStaticText, 0, wxALL, 5 );

	FitNodesMaxResNumericCtrl = new NumericTextCtrl( ExpertPanel, wxID_ANY, wxT("3.00"), wxDefaultPosition, wxDefaultSize, wxTE_PROCESS_ENTER );
	FitNodesMaxResNumericCtrl->Enable( false );

	fgSizer1->Add( FitNodesMaxResNumericCtrl, 0, wxALL, 5 );

	FitNodesRoundedSquareCheckBox = new wxCheckBox( ExpertPanel, wxID_ANY, wxT("Use no-decay model"), wxDefaultPosition, wxDefaultSize, 0 );
	FitNodesRoundedSquareCheckBox->Enable( false );

	fgSizer1->Add( FitNodesRoundedSquareCheckBox, 0, wxALL, 5 );


	fgSizer1->Add( 0, 0, 1, wxEXPAND, 5 );

	FitNodesWeightsCheckBox = new wxCheckBox( ExpertPanel, wxID_ANY, wxT("Ignore spectrum in nodes"), wxDefaultPosition, wxDefaultSize, 0 );
	FitNodesWeightsCheckBox->Enable( false );

	fgSizer1->Add( FitNodesWeightsCheckBox, 0, wxALL, 5 );


	fgSizer1->Add( 0, 0, 1, wxEXPAND, 5 );

	m_staticText20011 = new wxStaticText( ExpertPanel, wxID_ANY, wxT("Additional Options"), wxDefaultPosition, wxDefaultSize, 0 );
	m_staticText20011->Wrap( -1 );
	m_staticText20011->SetFont( wxFont( 10, wxFONTFAMILY_SWISS, wxFONTSTYLE_NORMAL, wxFONTWEIGHT_BOLD, true, wxT("Sans") ) );

	fgSizer1->Add( m_staticText20011, 0, wxALL, 5 );


	fgSizer1->Add( 0, 0, 1, wxEXPAND, 5 );

	FilterLowresSignalCheckBox = new wxCheckBox( ExpertPanel, wxID_ANY, wxT("Weight down low resolution signal?"), wxDefaultPosition, wxDefaultSize, 0 );
	FilterLowresSignalCheckBox->SetValue(true);
	fgSizer1->Add( FilterLowresSignalCheckBox, 0, wxALL, 5 );


	fgSizer1->Add( 0, 0, 1, wxEXPAND, 5 );


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

	CTFResultsPanel = new ShowCTFResultsPanel( this, wxID_ANY, wxDefaultPosition, wxDefaultSize, wxTAB_TRAVERSAL );
	CTFResultsPanel->Hide();

	bSizer46->Add( CTFResultsPanel, 80, wxEXPAND | wxALL, 5 );


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
	bSizer58->Add( RunProfileComboBox, 50, wxALIGN_CENTER_VERTICAL|wxALL, 5 );

	wxBoxSizer* bSizer60;
	bSizer60 = new wxBoxSizer( wxVERTICAL );

	StartEstimationButton = new wxButton( StartPanel, wxID_ANY, wxT("Start Estimation"), wxDefaultPosition, wxDefaultSize, 0 );
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
	this->Connect( wxEVT_UPDATE_UI, wxUpdateUIEventHandler( FindCTFPanel::OnUpdateUI ) );
	ExpertToggleButton->Connect( wxEVT_COMMAND_TOGGLEBUTTON_CLICKED, wxCommandEventHandler( FindCTFPanel::OnExpertOptionsToggle ), NULL, this );
	MovieRadioButton->Connect( wxEVT_COMMAND_RADIOBUTTON_SELECTED, wxCommandEventHandler( FindCTFPanel::OnMovieRadioButton ), NULL, this );
	ImageRadioButton->Connect( wxEVT_COMMAND_RADIOBUTTON_SELECTED, wxCommandEventHandler( FindCTFPanel::OnImageRadioButton ), NULL, this );
	SearchTiltYesRadio->Connect( wxEVT_COMMAND_RADIOBUTTON_SELECTED, wxCommandEventHandler( FindCTFPanel::OnMovieRadioButton ), NULL, this );
	SearchTiltNoRadio->Connect( wxEVT_COMMAND_RADIOBUTTON_SELECTED, wxCommandEventHandler( FindCTFPanel::OnImageRadioButton ), NULL, this );
	LargeAstigmatismExpectedCheckBox->Connect( wxEVT_COMMAND_CHECKBOX_CLICKED, wxCommandEventHandler( FindCTFPanel::OnLargeAstigmatismExpectedCheckBox ), NULL, this );
	RestrainAstigmatismCheckBox->Connect( wxEVT_COMMAND_CHECKBOX_CLICKED, wxCommandEventHandler( FindCTFPanel::OnRestrainAstigmatismCheckBox ), NULL, this );
	AdditionalPhaseShiftCheckBox->Connect( wxEVT_COMMAND_CHECKBOX_CLICKED, wxCommandEventHandler( FindCTFPanel::OnFindAdditionalPhaseCheckBox ), NULL, this );
	FitNodesCheckBox->Connect( wxEVT_COMMAND_CHECKBOX_CLICKED, wxCommandEventHandler( FindCTFPanel::OnFitNodesCheckBox ), NULL, this );
	FitNodes1DCheckBox->Connect( wxEVT_COMMAND_CHECKBOX_CLICKED, wxCommandEventHandler( FindCTFPanel::OnFindAdditionalPhaseCheckBox ), NULL, this );
	FitNodes2DCheckBox->Connect( wxEVT_COMMAND_CHECKBOX_CLICKED, wxCommandEventHandler( FindCTFPanel::OnFindAdditionalPhaseCheckBox ), NULL, this );
	FitNodesRoundedSquareCheckBox->Connect( wxEVT_COMMAND_CHECKBOX_CLICKED, wxCommandEventHandler( FindCTFPanel::OnFindAdditionalPhaseCheckBox ), NULL, this );
	FitNodesWeightsCheckBox->Connect( wxEVT_COMMAND_CHECKBOX_CLICKED, wxCommandEventHandler( FindCTFPanel::OnFindAdditionalPhaseCheckBox ), NULL, this );
	FilterLowresSignalCheckBox->Connect( wxEVT_COMMAND_CHECKBOX_CLICKED, wxCommandEventHandler( FindCTFPanel::OnFindAdditionalPhaseCheckBox ), NULL, this );
	InfoText->Connect( wxEVT_COMMAND_TEXT_URL, wxTextUrlEventHandler( FindCTFPanel::OnInfoURL ), NULL, this );
	FinishButton->Connect( wxEVT_COMMAND_BUTTON_CLICKED, wxCommandEventHandler( FindCTFPanel::FinishButtonClick ), NULL, this );
	CancelAlignmentButton->Connect( wxEVT_COMMAND_BUTTON_CLICKED, wxCommandEventHandler( FindCTFPanel::TerminateButtonClick ), NULL, this );
	StartEstimationButton->Connect( wxEVT_COMMAND_BUTTON_CLICKED, wxCommandEventHandler( FindCTFPanel::StartEstimationClick ), NULL, this );
}

FindCTFPanel::~FindCTFPanel()
{
	// Disconnect Events
	this->Disconnect( wxEVT_UPDATE_UI, wxUpdateUIEventHandler( FindCTFPanel::OnUpdateUI ) );
	ExpertToggleButton->Disconnect( wxEVT_COMMAND_TOGGLEBUTTON_CLICKED, wxCommandEventHandler( FindCTFPanel::OnExpertOptionsToggle ), NULL, this );
	MovieRadioButton->Disconnect( wxEVT_COMMAND_RADIOBUTTON_SELECTED, wxCommandEventHandler( FindCTFPanel::OnMovieRadioButton ), NULL, this );
	ImageRadioButton->Disconnect( wxEVT_COMMAND_RADIOBUTTON_SELECTED, wxCommandEventHandler( FindCTFPanel::OnImageRadioButton ), NULL, this );
	SearchTiltYesRadio->Disconnect( wxEVT_COMMAND_RADIOBUTTON_SELECTED, wxCommandEventHandler( FindCTFPanel::OnMovieRadioButton ), NULL, this );
	SearchTiltNoRadio->Disconnect( wxEVT_COMMAND_RADIOBUTTON_SELECTED, wxCommandEventHandler( FindCTFPanel::OnImageRadioButton ), NULL, this );
	LargeAstigmatismExpectedCheckBox->Disconnect( wxEVT_COMMAND_CHECKBOX_CLICKED, wxCommandEventHandler( FindCTFPanel::OnLargeAstigmatismExpectedCheckBox ), NULL, this );
	RestrainAstigmatismCheckBox->Disconnect( wxEVT_COMMAND_CHECKBOX_CLICKED, wxCommandEventHandler( FindCTFPanel::OnRestrainAstigmatismCheckBox ), NULL, this );
	AdditionalPhaseShiftCheckBox->Disconnect( wxEVT_COMMAND_CHECKBOX_CLICKED, wxCommandEventHandler( FindCTFPanel::OnFindAdditionalPhaseCheckBox ), NULL, this );
	FitNodesCheckBox->Disconnect( wxEVT_COMMAND_CHECKBOX_CLICKED, wxCommandEventHandler( FindCTFPanel::OnFitNodesCheckBox ), NULL, this );
	FitNodes1DCheckBox->Disconnect( wxEVT_COMMAND_CHECKBOX_CLICKED, wxCommandEventHandler( FindCTFPanel::OnFindAdditionalPhaseCheckBox ), NULL, this );
	FitNodes2DCheckBox->Disconnect( wxEVT_COMMAND_CHECKBOX_CLICKED, wxCommandEventHandler( FindCTFPanel::OnFindAdditionalPhaseCheckBox ), NULL, this );
	FitNodesRoundedSquareCheckBox->Disconnect( wxEVT_COMMAND_CHECKBOX_CLICKED, wxCommandEventHandler( FindCTFPanel::OnFindAdditionalPhaseCheckBox ), NULL, this );
	FitNodesWeightsCheckBox->Disconnect( wxEVT_COMMAND_CHECKBOX_CLICKED, wxCommandEventHandler( FindCTFPanel::OnFindAdditionalPhaseCheckBox ), NULL, this );
	FilterLowresSignalCheckBox->Disconnect( wxEVT_COMMAND_CHECKBOX_CLICKED, wxCommandEventHandler( FindCTFPanel::OnFindAdditionalPhaseCheckBox ), NULL, this );
	InfoText->Disconnect( wxEVT_COMMAND_TEXT_URL, wxTextUrlEventHandler( FindCTFPanel::OnInfoURL ), NULL, this );
	FinishButton->Disconnect( wxEVT_COMMAND_BUTTON_CLICKED, wxCommandEventHandler( FindCTFPanel::FinishButtonClick ), NULL, this );
	CancelAlignmentButton->Disconnect( wxEVT_COMMAND_BUTTON_CLICKED, wxCommandEventHandler( FindCTFPanel::TerminateButtonClick ), NULL, this );
	StartEstimationButton->Disconnect( wxEVT_COMMAND_BUTTON_CLICKED, wxCommandEventHandler( FindCTFPanel::StartEstimationClick ), NULL, this );

}

DisplayCTFRefinementResultsPanelParent::DisplayCTFRefinementResultsPanelParent( wxWindow* parent, wxWindowID id, const wxPoint& pos, const wxSize& size, long style, const wxString& name ) : wxPanel( parent, id, pos, size, style, name )
{
	wxBoxSizer* bSizer92;
	bSizer92 = new wxBoxSizer( wxVERTICAL );

	LeftRightSplitter = new wxSplitterWindow( this, wxID_ANY, wxDefaultPosition, wxDefaultSize, wxSP_3D|wxSP_PERMIT_UNSPLIT );
	LeftRightSplitter->SetSashGravity( 0.5 );
	LeftRightSplitter->Connect( wxEVT_IDLE, wxIdleEventHandler( DisplayCTFRefinementResultsPanelParent::LeftRightSplitterOnIdle ), NULL, this );

	LeftPanel = new wxPanel( LeftRightSplitter, wxID_ANY, wxDefaultPosition, wxDefaultSize, wxTAB_TRAVERSAL );
	wxBoxSizer* bSizer301;
	bSizer301 = new wxBoxSizer( wxVERTICAL );

	TopBottomSplitter = new wxSplitterWindow( LeftPanel, wxID_ANY, wxDefaultPosition, wxDefaultSize, wxSP_3D|wxSP_PERMIT_UNSPLIT );
	TopBottomSplitter->SetSashGravity( 0.5 );
	TopBottomSplitter->Connect( wxEVT_IDLE, wxIdleEventHandler( DisplayCTFRefinementResultsPanelParent::TopBottomSplitterOnIdle ), NULL, this );

	TopPanel = new wxPanel( TopBottomSplitter, wxID_ANY, wxDefaultPosition, wxDefaultSize, wxTAB_TRAVERSAL );
	wxBoxSizer* bSizer302;
	bSizer302 = new wxBoxSizer( wxVERTICAL );

	DefocusChangeText = new wxStaticText( TopPanel, wxID_ANY, wxT("Defocus Change Histogram"), wxDefaultPosition, wxDefaultSize, 0 );
	DefocusChangeText->Wrap( -1 );
	DefocusChangeText->SetFont( wxFont( 10, wxFONTFAMILY_SWISS, wxFONTSTYLE_NORMAL, wxFONTWEIGHT_BOLD, false, wxT("Sans") ) );

	bSizer302->Add( DefocusChangeText, 0, wxALL, 5 );

	DefocusPlotLine = new wxStaticLine( TopPanel, wxID_ANY, wxDefaultPosition, wxDefaultSize, wxLI_HORIZONTAL );
	bSizer302->Add( DefocusPlotLine, 0, wxEXPAND | wxALL, 5 );

	DefocusHistorgramPlotPanel = new PlotCurvePanel( TopPanel, wxID_ANY, wxDefaultPosition, wxDefaultSize, wxTAB_TRAVERSAL );
	bSizer302->Add( DefocusHistorgramPlotPanel, 1, wxEXPAND | wxALL, 5 );


	TopPanel->SetSizer( bSizer302 );
	TopPanel->Layout();
	bSizer302->Fit( TopPanel );
	BottomPanel = new wxPanel( TopBottomSplitter, wxID_ANY, wxDefaultPosition, wxDefaultSize, wxTAB_TRAVERSAL );
	wxBoxSizer* bSizer303;
	bSizer303 = new wxBoxSizer( wxVERTICAL );

	FSCResultsPanel = new MyFSCPanel( BottomPanel, wxID_ANY, wxDefaultPosition, wxDefaultSize, wxTAB_TRAVERSAL );
	bSizer303->Add( FSCResultsPanel, 1, wxEXPAND | wxALL, 5 );


	BottomPanel->SetSizer( bSizer303 );
	BottomPanel->Layout();
	bSizer303->Fit( BottomPanel );
	TopBottomSplitter->SplitHorizontally( TopPanel, BottomPanel, 0 );
	bSizer301->Add( TopBottomSplitter, 1, wxEXPAND, 5 );


	LeftPanel->SetSizer( bSizer301 );
	LeftPanel->Layout();
	bSizer301->Fit( LeftPanel );
	RightPanel = new wxPanel( LeftRightSplitter, wxID_ANY, wxDefaultPosition, wxDefaultSize, wxTAB_TRAVERSAL );
	wxBoxSizer* bSizer304;
	bSizer304 = new wxBoxSizer( wxVERTICAL );

	OrthText = new wxStaticText( RightPanel, wxID_ANY, wxT("Orthogonal Slices / Projections"), wxDefaultPosition, wxDefaultSize, 0 );
	OrthText->Wrap( -1 );
	OrthText->SetFont( wxFont( 10, wxFONTFAMILY_SWISS, wxFONTSTYLE_NORMAL, wxFONTWEIGHT_BOLD, false, wxT("Sans") ) );

	bSizer304->Add( OrthText, 0, wxALL, 5 );

	m_staticline109 = new wxStaticLine( RightPanel, wxID_ANY, wxDefaultPosition, wxDefaultSize, wxLI_HORIZONTAL );
	bSizer304->Add( m_staticline109, 0, wxEXPAND | wxALL, 5 );

	ShowOrthDisplayPanel = new DisplayPanel( RightPanel, wxID_ANY, wxDefaultPosition, wxDefaultSize, wxTAB_TRAVERSAL );
	bSizer304->Add( ShowOrthDisplayPanel, 66, wxEXPAND | wxALL, 5 );

	RoundPlotPanel = new wxPanel( RightPanel, wxID_ANY, wxDefaultPosition, wxDefaultSize, wxTAB_TRAVERSAL );
	RoundPlotPanel->Hide();

	bSizer304->Add( RoundPlotPanel, 33, wxEXPAND | wxALL, 5 );

	wxBoxSizer* bSizer306;
	bSizer306 = new wxBoxSizer( wxHORIZONTAL );


	bSizer304->Add( bSizer306, 0, wxEXPAND, 5 );


	RightPanel->SetSizer( bSizer304 );
	RightPanel->Layout();
	bSizer304->Fit( RightPanel );
	LeftRightSplitter->SplitVertically( LeftPanel, RightPanel, 600 );
	bSizer92->Add( LeftRightSplitter, 1, wxEXPAND, 5 );


	this->SetSizer( bSizer92 );
	this->Layout();
}

DisplayCTFRefinementResultsPanelParent::~DisplayCTFRefinementResultsPanelParent()
{
}
