///////////////////////////////////////////////////////////////////////////
// C++ code generated with wxFormBuilder (version 3.10.0-4761b0c)
// http://www.wxformbuilder.org/
//
// PLEASE DO *NOT* EDIT THIS FILE!
///////////////////////////////////////////////////////////////////////////

#include "AngularDistributionPlotPanel.h"
#include "AssetPickerComboPanel.h"
#include "DisplayPanel.h"
#include "DisplayRefinementResultsPanel.h"
#include "MyFSCPanel.h"
#include "PlotCurvePanel.h"
#include "my_controls.h"

#include "ProjectX_gui_refine3d.h"

///////////////////////////////////////////////////////////////////////////

RefinementResultsPanel::RefinementResultsPanel( wxWindow* parent, wxWindowID id, const wxPoint& pos, const wxSize& size, long style, const wxString& name ) : wxPanel( parent, id, pos, size, style, name )
{
	wxBoxSizer* bSizer202;
	bSizer202 = new wxBoxSizer( wxVERTICAL );

	m_splitter7 = new wxSplitterWindow( this, wxID_ANY, wxDefaultPosition, wxDefaultSize, wxSP_3D );
	m_splitter7->SetSashGravity( 0.5 );
	m_splitter7->SetSashSize( 10 );
	m_splitter7->Connect( wxEVT_IDLE, wxIdleEventHandler( RefinementResultsPanel::m_splitter7OnIdle ), NULL, this );

	m_panel49 = new wxPanel( m_splitter7, wxID_ANY, wxDefaultPosition, wxDefaultSize, wxTAB_TRAVERSAL );
	wxBoxSizer* bSizer206;
	bSizer206 = new wxBoxSizer( wxVERTICAL );

	wxFlexGridSizer* fgSizer11;
	fgSizer11 = new wxFlexGridSizer( 0, 2, 0, 0 );
	fgSizer11->AddGrowableCol( 1 );
	fgSizer11->SetFlexibleDirection( wxBOTH );
	fgSizer11->SetNonFlexibleGrowMode( wxFLEX_GROWMODE_SPECIFIED );

	m_staticText284 = new wxStaticText( m_panel49, wxID_ANY, wxT("Select Refinement Package :"), wxDefaultPosition, wxDefaultSize, 0 );
	m_staticText284->Wrap( -1 );
	fgSizer11->Add( m_staticText284, 0, wxALIGN_CENTER_VERTICAL|wxALIGN_RIGHT|wxALL, 5 );

	RefinementPackageComboBox = new RefinementPackagePickerComboPanel( m_panel49, wxID_ANY, wxDefaultPosition, wxDefaultSize, wxTAB_TRAVERSAL );
	RefinementPackageComboBox->SetMinSize( wxSize( 350,-1 ) );
	RefinementPackageComboBox->SetMaxSize( wxSize( 350,-1 ) );

	fgSizer11->Add( RefinementPackageComboBox, 0, wxALIGN_CENTER_VERTICAL|wxALL|wxEXPAND, 5 );

	m_staticText285 = new wxStaticText( m_panel49, wxID_ANY, wxT("Select Input Parameters :"), wxDefaultPosition, wxDefaultSize, 0 );
	m_staticText285->Wrap( -1 );
	fgSizer11->Add( m_staticText285, 0, wxALIGN_CENTER_VERTICAL|wxALIGN_RIGHT|wxALL, 5 );

	InputParametersComboBox = new RefinementPickerComboPanel( m_panel49, wxID_ANY, wxDefaultPosition, wxDefaultSize, wxTAB_TRAVERSAL );
	InputParametersComboBox->SetMinSize( wxSize( 350,-1 ) );
	InputParametersComboBox->SetMaxSize( wxSize( 350,-1 ) );

	fgSizer11->Add( InputParametersComboBox, 0, wxALIGN_CENTER_VERTICAL|wxALL|wxEXPAND, 5 );


	bSizer206->Add( fgSizer11, 0, wxEXPAND, 5 );

	m_splitter16 = new wxSplitterWindow( m_panel49, wxID_ANY, wxDefaultPosition, wxDefaultSize, wxSP_3D );
	m_splitter16->SetSashGravity( 0.5 );
	m_splitter16->Connect( wxEVT_IDLE, wxIdleEventHandler( RefinementResultsPanel::m_splitter16OnIdle ), NULL, this );

	m_panel124 = new wxPanel( m_splitter16, wxID_ANY, wxDefaultPosition, wxDefaultSize, wxTAB_TRAVERSAL );
	wxBoxSizer* bSizer383;
	bSizer383 = new wxBoxSizer( wxVERTICAL );

	FSCPlotPanel = new MyFSCPanel( m_panel124, wxID_ANY, wxDefaultPosition, wxDefaultSize, wxTAB_TRAVERSAL );
	bSizer383->Add( FSCPlotPanel, 60, wxEXPAND | wxALL, 5 );


	m_panel124->SetSizer( bSizer383 );
	m_panel124->Layout();
	bSizer383->Fit( m_panel124 );
	m_panel125 = new wxPanel( m_splitter16, wxID_ANY, wxDefaultPosition, wxDefaultSize, wxTAB_TRAVERSAL );
	wxBoxSizer* bSizer384;
	bSizer384 = new wxBoxSizer( wxVERTICAL );

	wxBoxSizer* bSizer495;
	bSizer495 = new wxBoxSizer( wxVERTICAL );

	wxBoxSizer* bSizer497;
	bSizer497 = new wxBoxSizer( wxHORIZONTAL );

	m_staticText708 = new wxStaticText( m_panel125, wxID_ANY, wxT("Angular Distribution"), wxDefaultPosition, wxDefaultSize, 0 );
	m_staticText708->Wrap( -1 );
	m_staticText708->SetFont( wxFont( 10, wxFONTFAMILY_SWISS, wxFONTSTYLE_NORMAL, wxFONTWEIGHT_BOLD, false, wxT("Sans") ) );

	bSizer497->Add( m_staticText708, 0, wxALIGN_BOTTOM|wxALL, 5 );


	bSizer497->Add( 0, 0, 1, wxEXPAND, 5 );

	m_staticline132 = new wxStaticLine( m_panel125, wxID_ANY, wxDefaultPosition, wxDefaultSize, wxLI_HORIZONTAL|wxLI_VERTICAL );
	bSizer497->Add( m_staticline132, 0, wxEXPAND | wxALL, 5 );

	ParametersDetailButton = new NoFocusBitmapButton( m_panel125, wxID_ANY, wxNullBitmap, wxDefaultPosition, wxDefaultSize, wxBU_AUTODRAW|0 );

	ParametersDetailButton->SetDefault();
	bSizer497->Add( ParametersDetailButton, 0, wxLEFT|wxTOP, 5 );

	AngularPlotDetailsButton = new NoFocusBitmapButton( m_panel125, wxID_ANY, wxNullBitmap, wxDefaultPosition, wxDefaultSize, wxBU_AUTODRAW|0 );

	AngularPlotDetailsButton->SetDefault();
	bSizer497->Add( AngularPlotDetailsButton, 0, wxRIGHT|wxTOP, 5 );


	bSizer495->Add( bSizer497, 0, wxEXPAND, 5 );

	m_staticline131 = new wxStaticLine( m_panel125, wxID_ANY, wxDefaultPosition, wxDefaultSize, wxLI_HORIZONTAL );
	bSizer495->Add( m_staticline131, 0, wxEXPAND | wxALL, 5 );

	AngularPlotPanel = new AngularDistributionPlotPanelHistogram( m_panel125, wxID_ANY, wxDefaultPosition, wxDefaultSize, wxTAB_TRAVERSAL );
	bSizer495->Add( AngularPlotPanel, 100, wxEXPAND | wxALL, 5 );


	bSizer384->Add( bSizer495, 1, wxEXPAND, 5 );


	m_panel125->SetSizer( bSizer384 );
	m_panel125->Layout();
	bSizer384->Fit( m_panel125 );
	m_splitter16->SplitHorizontally( m_panel124, m_panel125, 350 );
	bSizer206->Add( m_splitter16, 1, wxEXPAND, 5 );


	m_panel49->SetSizer( bSizer206 );
	m_panel49->Layout();
	bSizer206->Fit( m_panel49 );
	RightPanel = new wxPanel( m_splitter7, wxID_ANY, wxDefaultPosition, wxDefaultSize, wxTAB_TRAVERSAL );
	wxBoxSizer* bSizer203;
	bSizer203 = new wxBoxSizer( wxVERTICAL );

	wxBoxSizer* bSizer504;
	bSizer504 = new wxBoxSizer( wxVERTICAL );

	JobDetailsToggleButton = new wxToggleButton( RightPanel, wxID_ANY, wxT("Show Job Details"), wxDefaultPosition, wxDefaultSize, 0 );
	bSizer504->Add( JobDetailsToggleButton, 0, wxALL, 5 );


	bSizer203->Add( bSizer504, 0, wxEXPAND, 5 );

	m_staticline133 = new wxStaticLine( RightPanel, wxID_ANY, wxDefaultPosition, wxDefaultSize, wxLI_HORIZONTAL );
	bSizer203->Add( m_staticline133, 0, wxEXPAND | wxALL, 5 );

	JobDetailsPanel = new wxPanel( RightPanel, wxID_ANY, wxDefaultPosition, wxDefaultSize, wxTAB_TRAVERSAL );
	JobDetailsPanel->Hide();

	wxBoxSizer* bSizer270;
	bSizer270 = new wxBoxSizer( wxVERTICAL );

	wxBoxSizer* bSizer101;
	bSizer101 = new wxBoxSizer( wxVERTICAL );

	InfoSizer = new wxFlexGridSizer( 0, 6, 0, 0 );
	InfoSizer->SetFlexibleDirection( wxBOTH );
	InfoSizer->SetNonFlexibleGrowMode( wxFLEX_GROWMODE_SPECIFIED );

	m_staticText72 = new wxStaticText( JobDetailsPanel, wxID_ANY, wxT("Refinement ID :"), wxDefaultPosition, wxDefaultSize, 0 );
	m_staticText72->Wrap( -1 );
	m_staticText72->SetFont( wxFont( 10, wxFONTFAMILY_SWISS, wxFONTSTYLE_NORMAL, wxFONTWEIGHT_BOLD, false, wxT("Sans") ) );

	InfoSizer->Add( m_staticText72, 0, wxALIGN_RIGHT|wxALL, 5 );

	RefinementIDStaticText = new wxStaticText( JobDetailsPanel, wxID_ANY, wxEmptyString, wxDefaultPosition, wxDefaultSize, 0 );
	RefinementIDStaticText->Wrap( -1 );
	InfoSizer->Add( RefinementIDStaticText, 0, wxALIGN_LEFT|wxALL, 5 );

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

	m_staticText785 = new wxStaticText( JobDetailsPanel, wxID_ANY, wxT("Percent Used :"), wxDefaultPosition, wxDefaultSize, 0 );
	m_staticText785->Wrap( -1 );
	m_staticText785->SetFont( wxFont( 10, wxFONTFAMILY_SWISS, wxFONTSTYLE_NORMAL, wxFONTWEIGHT_BOLD, false, wxT("Sans") ) );

	InfoSizer->Add( m_staticText785, 0, wxALIGN_RIGHT|wxALL, 5 );

	PercentUsedStaticText = new wxStaticText( JobDetailsPanel, wxID_ANY, wxEmptyString, wxDefaultPosition, wxDefaultSize, 0 );
	PercentUsedStaticText->Wrap( -1 );
	InfoSizer->Add( PercentUsedStaticText, 0, wxALL, 5 );

	m_staticText83 = new wxStaticText( JobDetailsPanel, wxID_ANY, wxT("Ref. Volume ID :"), wxDefaultPosition, wxDefaultSize, 0 );
	m_staticText83->Wrap( -1 );
	m_staticText83->SetFont( wxFont( 10, wxFONTFAMILY_SWISS, wxFONTSTYLE_NORMAL, wxFONTWEIGHT_BOLD, false, wxT("Sans") ) );

	InfoSizer->Add( m_staticText83, 0, wxALIGN_RIGHT|wxALL, 5 );

	ReferenceVolumeIDStaticText = new wxStaticText( JobDetailsPanel, wxID_ANY, wxEmptyString, wxDefaultPosition, wxDefaultSize, 0 );
	ReferenceVolumeIDStaticText->Wrap( -1 );
	InfoSizer->Add( ReferenceVolumeIDStaticText, 0, wxALIGN_LEFT|wxALL, 5 );

	m_staticText82 = new wxStaticText( JobDetailsPanel, wxID_ANY, wxT("Ref. Refinement ID :"), wxDefaultPosition, wxDefaultSize, 0 );
	m_staticText82->Wrap( -1 );
	m_staticText82->SetFont( wxFont( 10, wxFONTFAMILY_SWISS, wxFONTSTYLE_NORMAL, wxFONTWEIGHT_BOLD, false, wxT("Sans") ) );

	InfoSizer->Add( m_staticText82, 0, wxALIGN_RIGHT|wxALL, 5 );

	ReferenceRefinementIDStaticText = new wxStaticText( JobDetailsPanel, wxID_ANY, wxEmptyString, wxDefaultPosition, wxDefaultSize, 0 );
	ReferenceRefinementIDStaticText->Wrap( -1 );
	InfoSizer->Add( ReferenceRefinementIDStaticText, 0, wxALIGN_LEFT|wxALL, 5 );

	m_staticText85 = new wxStaticText( JobDetailsPanel, wxID_ANY, wxT("Low Res. Limit :"), wxDefaultPosition, wxDefaultSize, 0 );
	m_staticText85->Wrap( -1 );
	m_staticText85->SetFont( wxFont( 10, wxFONTFAMILY_SWISS, wxFONTSTYLE_NORMAL, wxFONTWEIGHT_BOLD, false, wxT("Sans") ) );

	InfoSizer->Add( m_staticText85, 0, wxALIGN_RIGHT|wxALL, 5 );

	LowResLimitStaticText = new wxStaticText( JobDetailsPanel, wxID_ANY, wxEmptyString, wxDefaultPosition, wxDefaultSize, 0 );
	LowResLimitStaticText->Wrap( -1 );
	InfoSizer->Add( LowResLimitStaticText, 0, wxALIGN_LEFT|wxALL, 5 );

	m_staticText87 = new wxStaticText( JobDetailsPanel, wxID_ANY, wxT("High Res. Limit :"), wxDefaultPosition, wxDefaultSize, 0 );
	m_staticText87->Wrap( -1 );
	m_staticText87->SetFont( wxFont( 10, wxFONTFAMILY_SWISS, wxFONTSTYLE_NORMAL, wxFONTWEIGHT_BOLD, false, wxT("Sans") ) );

	InfoSizer->Add( m_staticText87, 0, wxALIGN_RIGHT|wxALL, 5 );

	HighResLimitStaticText = new wxStaticText( JobDetailsPanel, wxID_ANY, wxEmptyString, wxDefaultPosition, wxDefaultSize, 0 );
	HighResLimitStaticText->Wrap( -1 );
	InfoSizer->Add( HighResLimitStaticText, 0, wxALIGN_LEFT|wxALL, 5 );

	m_staticText89 = new wxStaticText( JobDetailsPanel, wxID_ANY, wxT("Mask Radius :"), wxDefaultPosition, wxDefaultSize, 0 );
	m_staticText89->Wrap( -1 );
	m_staticText89->SetFont( wxFont( 10, wxFONTFAMILY_SWISS, wxFONTSTYLE_NORMAL, wxFONTWEIGHT_BOLD, false, wxT("Sans") ) );

	InfoSizer->Add( m_staticText89, 0, wxALIGN_RIGHT|wxALL, 5 );

	MaskRadiusStaticText = new wxStaticText( JobDetailsPanel, wxID_ANY, wxEmptyString, wxDefaultPosition, wxDefaultSize, 0 );
	MaskRadiusStaticText->Wrap( -1 );
	InfoSizer->Add( MaskRadiusStaticText, 0, wxALIGN_LEFT|wxALL, 5 );

	m_staticText777 = new wxStaticText( JobDetailsPanel, wxID_ANY, wxT("Signed CC Res. Limit :"), wxDefaultPosition, wxDefaultSize, 0 );
	m_staticText777->Wrap( -1 );
	m_staticText777->SetFont( wxFont( 10, wxFONTFAMILY_SWISS, wxFONTSTYLE_NORMAL, wxFONTWEIGHT_BOLD, false, wxT("Sans") ) );

	InfoSizer->Add( m_staticText777, 0, wxALIGN_RIGHT|wxALL, 5 );

	SignedCCResLimitStaticText = new wxStaticText( JobDetailsPanel, wxID_ANY, wxEmptyString, wxDefaultPosition, wxDefaultSize, 0 );
	SignedCCResLimitStaticText->Wrap( -1 );
	InfoSizer->Add( SignedCCResLimitStaticText, 0, wxALL, 5 );

	m_staticText779 = new wxStaticText( JobDetailsPanel, wxID_ANY, wxT("Global Res. Limit :"), wxDefaultPosition, wxDefaultSize, 0 );
	m_staticText779->Wrap( -1 );
	m_staticText779->SetFont( wxFont( 10, wxFONTFAMILY_SWISS, wxFONTSTYLE_NORMAL, wxFONTWEIGHT_BOLD, false, wxT("Sans") ) );

	InfoSizer->Add( m_staticText779, 0, wxALIGN_RIGHT|wxALL, 5 );

	GlobalResLimitStaticText = new wxStaticText( JobDetailsPanel, wxID_ANY, wxEmptyString, wxDefaultPosition, wxDefaultSize, 0 );
	GlobalResLimitStaticText->Wrap( -1 );
	InfoSizer->Add( GlobalResLimitStaticText, 0, wxALL, 5 );

	m_staticText781 = new wxStaticText( JobDetailsPanel, wxID_ANY, wxT("Global Mask Radius :"), wxDefaultPosition, wxDefaultSize, 0 );
	m_staticText781->Wrap( -1 );
	m_staticText781->SetFont( wxFont( 10, wxFONTFAMILY_SWISS, wxFONTSTYLE_NORMAL, wxFONTWEIGHT_BOLD, false, wxT("Sans") ) );

	InfoSizer->Add( m_staticText781, 0, wxALIGN_RIGHT|wxALL, 5 );

	GlobalMaskRadiusStaticText = new wxStaticText( JobDetailsPanel, wxID_ANY, wxEmptyString, wxDefaultPosition, wxDefaultSize, 0 );
	GlobalMaskRadiusStaticText->Wrap( -1 );
	InfoSizer->Add( GlobalMaskRadiusStaticText, 0, wxALL, 5 );

	m_staticText783 = new wxStaticText( JobDetailsPanel, wxID_ANY, wxT("No. Results Refined :"), wxDefaultPosition, wxDefaultSize, 0 );
	m_staticText783->Wrap( -1 );
	m_staticText783->SetFont( wxFont( 10, wxFONTFAMILY_SWISS, wxFONTSTYLE_NORMAL, wxFONTWEIGHT_BOLD, false, wxT("Sans") ) );

	InfoSizer->Add( m_staticText783, 0, wxALIGN_RIGHT|wxALL, 5 );

	NumberResultsRefinedStaticText = new wxStaticText( JobDetailsPanel, wxID_ANY, wxEmptyString, wxDefaultPosition, wxDefaultSize, 0 );
	NumberResultsRefinedStaticText->Wrap( -1 );
	InfoSizer->Add( NumberResultsRefinedStaticText, 0, wxALL, 5 );

	m_staticText91 = new wxStaticText( JobDetailsPanel, wxID_ANY, wxT("Angular Search Step :"), wxDefaultPosition, wxDefaultSize, 0 );
	m_staticText91->Wrap( -1 );
	m_staticText91->SetFont( wxFont( 10, wxFONTFAMILY_SWISS, wxFONTSTYLE_NORMAL, wxFONTWEIGHT_BOLD, false, wxT("Sans") ) );

	InfoSizer->Add( m_staticText91, 0, wxALIGN_RIGHT|wxALL, 5 );

	AngularSearchStepStaticText = new wxStaticText( JobDetailsPanel, wxID_ANY, wxEmptyString, wxDefaultPosition, wxDefaultSize, 0 );
	AngularSearchStepStaticText->Wrap( -1 );
	InfoSizer->Add( AngularSearchStepStaticText, 0, wxALIGN_LEFT|wxALL, 5 );

	m_staticText79 = new wxStaticText( JobDetailsPanel, wxID_ANY, wxT("Search Range X :"), wxDefaultPosition, wxDefaultSize, 0 );
	m_staticText79->Wrap( -1 );
	m_staticText79->SetFont( wxFont( 10, wxFONTFAMILY_SWISS, wxFONTSTYLE_NORMAL, wxFONTWEIGHT_BOLD, false, wxT("Sans") ) );

	InfoSizer->Add( m_staticText79, 0, wxALIGN_RIGHT|wxALL, 5 );

	SearchRangeXStaticText = new wxStaticText( JobDetailsPanel, wxID_ANY, wxEmptyString, wxDefaultPosition, wxDefaultSize, 0 );
	SearchRangeXStaticText->Wrap( -1 );
	InfoSizer->Add( SearchRangeXStaticText, 0, wxALIGN_LEFT|wxALL, 5 );

	m_staticText99 = new wxStaticText( JobDetailsPanel, wxID_ANY, wxT("Search Range Y :"), wxDefaultPosition, wxDefaultSize, 0 );
	m_staticText99->Wrap( -1 );
	m_staticText99->SetFont( wxFont( 10, wxFONTFAMILY_SWISS, wxFONTSTYLE_NORMAL, wxFONTWEIGHT_BOLD, false, wxT("Sans") ) );

	InfoSizer->Add( m_staticText99, 0, wxALIGN_RIGHT|wxALL, 5 );

	SearchRangeYStaticText = new wxStaticText( JobDetailsPanel, wxID_ANY, wxEmptyString, wxDefaultPosition, wxDefaultSize, 0 );
	SearchRangeYStaticText->Wrap( -1 );
	InfoSizer->Add( SearchRangeYStaticText, 0, wxALIGN_LEFT|wxALL, 5 );

	m_staticText95 = new wxStaticText( JobDetailsPanel, wxID_ANY, wxT("Class. Res. Limit :"), wxDefaultPosition, wxDefaultSize, 0 );
	m_staticText95->Wrap( -1 );
	m_staticText95->SetFont( wxFont( 10, wxFONTFAMILY_SWISS, wxFONTSTYLE_NORMAL, wxFONTWEIGHT_BOLD, false, wxT("Sans") ) );

	InfoSizer->Add( m_staticText95, 0, wxALIGN_RIGHT|wxALL, 5 );

	ClassificationResLimitStaticText = new wxStaticText( JobDetailsPanel, wxID_ANY, wxEmptyString, wxDefaultPosition, wxDefaultSize, 0 );
	ClassificationResLimitStaticText->Wrap( -1 );
	InfoSizer->Add( ClassificationResLimitStaticText, 0, wxALIGN_LEFT|wxALL, 5 );

	LargeAstigExpectedLabel = new wxStaticText( JobDetailsPanel, wxID_ANY, wxT("Focus Classify? :"), wxDefaultPosition, wxDefaultSize, 0 );
	LargeAstigExpectedLabel->Wrap( -1 );
	LargeAstigExpectedLabel->SetFont( wxFont( 10, wxFONTFAMILY_SWISS, wxFONTSTYLE_NORMAL, wxFONTWEIGHT_BOLD, false, wxT("Sans") ) );

	InfoSizer->Add( LargeAstigExpectedLabel, 0, wxALIGN_RIGHT|wxALL, 5 );

	ShouldFocusClassifyStaticText = new wxStaticText( JobDetailsPanel, wxID_ANY, wxEmptyString, wxDefaultPosition, wxDefaultSize, 0 );
	ShouldFocusClassifyStaticText->Wrap( -1 );
	InfoSizer->Add( ShouldFocusClassifyStaticText, 0, wxALL, 5 );

	ToleratedAstigLabel = new wxStaticText( JobDetailsPanel, wxID_ANY, wxT("Sphere X Co-ord :"), wxDefaultPosition, wxDefaultSize, 0 );
	ToleratedAstigLabel->Wrap( -1 );
	ToleratedAstigLabel->SetFont( wxFont( 10, wxFONTFAMILY_SWISS, wxFONTSTYLE_NORMAL, wxFONTWEIGHT_BOLD, false, wxT("Sans") ) );

	InfoSizer->Add( ToleratedAstigLabel, 0, wxALIGN_RIGHT|wxALL, 5 );

	SphereXCoordStaticText = new wxStaticText( JobDetailsPanel, wxID_ANY, wxEmptyString, wxDefaultPosition, wxDefaultSize, 0 );
	SphereXCoordStaticText->Wrap( -1 );
	InfoSizer->Add( SphereXCoordStaticText, 0, wxALIGN_LEFT|wxALL, 5 );

	NumberOfAveragedFramesLabel = new wxStaticText( JobDetailsPanel, wxID_ANY, wxT("Sphere Y Co-ord :"), wxDefaultPosition, wxDefaultSize, 0 );
	NumberOfAveragedFramesLabel->Wrap( -1 );
	NumberOfAveragedFramesLabel->SetFont( wxFont( 10, wxFONTFAMILY_SWISS, wxFONTSTYLE_NORMAL, wxFONTWEIGHT_BOLD, false, wxT("Sans") ) );

	InfoSizer->Add( NumberOfAveragedFramesLabel, 0, wxALIGN_RIGHT|wxALL, 5 );

	SphereYCoordStaticText = new wxStaticText( JobDetailsPanel, wxID_ANY, wxEmptyString, wxDefaultPosition, wxDefaultSize, 0 );
	SphereYCoordStaticText->Wrap( -1 );
	InfoSizer->Add( SphereYCoordStaticText, 0, wxALL, 5 );

	m_staticText787 = new wxStaticText( JobDetailsPanel, wxID_ANY, wxT("Sphere Z Co-ord :"), wxDefaultPosition, wxDefaultSize, 0 );
	m_staticText787->Wrap( -1 );
	m_staticText787->SetFont( wxFont( 10, wxFONTFAMILY_SWISS, wxFONTSTYLE_NORMAL, wxFONTWEIGHT_BOLD, false, wxT("Sans") ) );

	InfoSizer->Add( m_staticText787, 0, wxALIGN_RIGHT|wxALL, 5 );

	SphereZCoordStaticText = new wxStaticText( JobDetailsPanel, wxID_ANY, wxEmptyString, wxDefaultPosition, wxDefaultSize, 0 );
	SphereZCoordStaticText->Wrap( -1 );
	InfoSizer->Add( SphereZCoordStaticText, 0, wxALL, 5 );

	m_staticText789 = new wxStaticText( JobDetailsPanel, wxID_ANY, wxT("Sphere Radius :"), wxDefaultPosition, wxDefaultSize, 0 );
	m_staticText789->Wrap( -1 );
	m_staticText789->SetFont( wxFont( 10, wxFONTFAMILY_SWISS, wxFONTSTYLE_NORMAL, wxFONTWEIGHT_BOLD, false, wxT("Sans") ) );

	InfoSizer->Add( m_staticText789, 0, wxALIGN_RIGHT|wxALL, 5 );

	SphereRadiusStaticText = new wxStaticText( JobDetailsPanel, wxID_ANY, wxEmptyString, wxDefaultPosition, wxDefaultSize, 0 );
	SphereRadiusStaticText->Wrap( -1 );
	InfoSizer->Add( SphereRadiusStaticText, 0, wxALL, 5 );

	m_staticText791 = new wxStaticText( JobDetailsPanel, wxID_ANY, wxT("Refine CTF? :"), wxDefaultPosition, wxDefaultSize, 0 );
	m_staticText791->Wrap( -1 );
	m_staticText791->SetFont( wxFont( 10, wxFONTFAMILY_SWISS, wxFONTSTYLE_NORMAL, wxFONTWEIGHT_BOLD, false, wxT("Sans") ) );

	InfoSizer->Add( m_staticText791, 0, wxALIGN_RIGHT|wxALL, 5 );

	ShouldRefineCTFStaticText = new wxStaticText( JobDetailsPanel, wxID_ANY, wxEmptyString, wxDefaultPosition, wxDefaultSize, 0 );
	ShouldRefineCTFStaticText->Wrap( -1 );
	InfoSizer->Add( ShouldRefineCTFStaticText, 0, wxALL, 5 );

	m_staticText793 = new wxStaticText( JobDetailsPanel, wxID_ANY, wxT("Defocus Search Range :"), wxDefaultPosition, wxDefaultSize, 0 );
	m_staticText793->Wrap( -1 );
	m_staticText793->SetFont( wxFont( 10, wxFONTFAMILY_SWISS, wxFONTSTYLE_NORMAL, wxFONTWEIGHT_BOLD, false, wxT("Sans") ) );

	InfoSizer->Add( m_staticText793, 0, wxALIGN_RIGHT|wxALL, 5 );

	DefocusSearchRangeStaticText = new wxStaticText( JobDetailsPanel, wxID_ANY, wxEmptyString, wxDefaultPosition, wxDefaultSize, 0 );
	DefocusSearchRangeStaticText->Wrap( -1 );
	InfoSizer->Add( DefocusSearchRangeStaticText, 0, wxALL, 5 );

	m_staticText795 = new wxStaticText( JobDetailsPanel, wxID_ANY, wxT("Defocus Search Step :"), wxDefaultPosition, wxDefaultSize, 0 );
	m_staticText795->Wrap( -1 );
	m_staticText795->SetFont( wxFont( 10, wxFONTFAMILY_SWISS, wxFONTSTYLE_NORMAL, wxFONTWEIGHT_BOLD, false, wxT("Sans") ) );

	InfoSizer->Add( m_staticText795, 0, wxALIGN_RIGHT|wxALL, 5 );

	DefocusSearchStepStaticText = new wxStaticText( JobDetailsPanel, wxID_ANY, wxEmptyString, wxDefaultPosition, wxDefaultSize, 0 );
	DefocusSearchStepStaticText->Wrap( -1 );
	InfoSizer->Add( DefocusSearchStepStaticText, 0, wxALL, 5 );

	m_staticText797 = new wxStaticText( JobDetailsPanel, wxID_ANY, wxT("AutoMask? :"), wxDefaultPosition, wxDefaultSize, 0 );
	m_staticText797->Wrap( -1 );
	m_staticText797->SetFont( wxFont( 10, wxFONTFAMILY_SWISS, wxFONTSTYLE_NORMAL, wxFONTWEIGHT_BOLD, false, wxT("Sans") ) );

	InfoSizer->Add( m_staticText797, 0, wxALIGN_RIGHT|wxALL, 5 );

	ShouldAutoMaskStaticText = new wxStaticText( JobDetailsPanel, wxID_ANY, wxEmptyString, wxDefaultPosition, wxDefaultSize, 0 );
	ShouldAutoMaskStaticText->Wrap( -1 );
	InfoSizer->Add( ShouldAutoMaskStaticText, 0, wxALL, 5 );

	m_staticText799 = new wxStaticText( JobDetailsPanel, wxID_ANY, wxT("Also Refine Input? :"), wxDefaultPosition, wxDefaultSize, 0 );
	m_staticText799->Wrap( -1 );
	m_staticText799->SetFont( wxFont( 10, wxFONTFAMILY_SWISS, wxFONTSTYLE_NORMAL, wxFONTWEIGHT_BOLD, false, wxT("Sans") ) );

	InfoSizer->Add( m_staticText799, 0, wxALIGN_RIGHT|wxALL, 5 );

	RefineInputParamsStaticText = new wxStaticText( JobDetailsPanel, wxID_ANY, wxEmptyString, wxDefaultPosition, wxDefaultSize, 0 );
	RefineInputParamsStaticText->Wrap( -1 );
	InfoSizer->Add( RefineInputParamsStaticText, 0, wxALL, 5 );

	m_staticText801 = new wxStaticText( JobDetailsPanel, wxID_ANY, wxT("Use Supplied Mask? :"), wxDefaultPosition, wxDefaultSize, 0 );
	m_staticText801->Wrap( -1 );
	m_staticText801->SetFont( wxFont( 10, wxFONTFAMILY_SWISS, wxFONTSTYLE_NORMAL, wxFONTWEIGHT_BOLD, false, wxT("Sans") ) );

	InfoSizer->Add( m_staticText801, 0, wxALIGN_RIGHT|wxALL, 5 );

	UseSuppliedMaskStaticText = new wxStaticText( JobDetailsPanel, wxID_ANY, wxEmptyString, wxDefaultPosition, wxDefaultSize, 0 );
	UseSuppliedMaskStaticText->Wrap( -1 );
	InfoSizer->Add( UseSuppliedMaskStaticText, 0, wxALL, 5 );

	m_staticText803 = new wxStaticText( JobDetailsPanel, wxID_ANY, wxT("Mask Asset ID :"), wxDefaultPosition, wxDefaultSize, 0 );
	m_staticText803->Wrap( -1 );
	m_staticText803->SetFont( wxFont( 10, wxFONTFAMILY_SWISS, wxFONTSTYLE_NORMAL, wxFONTWEIGHT_BOLD, false, wxT("Sans") ) );

	InfoSizer->Add( m_staticText803, 0, wxALIGN_RIGHT|wxALL, 5 );

	MaskAssetIDStaticText = new wxStaticText( JobDetailsPanel, wxID_ANY, wxEmptyString, wxDefaultPosition, wxDefaultSize, 0 );
	MaskAssetIDStaticText->Wrap( -1 );
	InfoSizer->Add( MaskAssetIDStaticText, 0, wxALL, 5 );

	m_staticText805 = new wxStaticText( JobDetailsPanel, wxID_ANY, wxT("Mask Edge Width :"), wxDefaultPosition, wxDefaultSize, 0 );
	m_staticText805->Wrap( -1 );
	m_staticText805->SetFont( wxFont( 10, wxFONTFAMILY_SWISS, wxFONTSTYLE_NORMAL, wxFONTWEIGHT_BOLD, false, wxT("Sans") ) );

	InfoSizer->Add( m_staticText805, 0, wxALIGN_RIGHT|wxALL, 5 );

	MaskEdgeWidthStaticText = new wxStaticText( JobDetailsPanel, wxID_ANY, wxEmptyString, wxDefaultPosition, wxDefaultSize, 0 );
	MaskEdgeWidthStaticText->Wrap( -1 );
	InfoSizer->Add( MaskEdgeWidthStaticText, 0, wxALL, 5 );

	m_staticText807 = new wxStaticText( JobDetailsPanel, wxID_ANY, wxT("Mask Out. Weight :"), wxDefaultPosition, wxDefaultSize, 0 );
	m_staticText807->Wrap( -1 );
	m_staticText807->SetFont( wxFont( 10, wxFONTFAMILY_SWISS, wxFONTSTYLE_NORMAL, wxFONTWEIGHT_BOLD, false, wxT("Sans") ) );

	InfoSizer->Add( m_staticText807, 0, wxALIGN_RIGHT|wxALL, 5 );

	MaskOutsideWeightStaticText = new wxStaticText( JobDetailsPanel, wxID_ANY, wxEmptyString, wxDefaultPosition, wxDefaultSize, 0 );
	MaskOutsideWeightStaticText->Wrap( -1 );
	InfoSizer->Add( MaskOutsideWeightStaticText, 0, wxALL, 5 );

	m_staticText809 = new wxStaticText( JobDetailsPanel, wxID_ANY, wxT("Filter Out. Mask? :"), wxDefaultPosition, wxDefaultSize, 0 );
	m_staticText809->Wrap( -1 );
	m_staticText809->SetFont( wxFont( 10, wxFONTFAMILY_SWISS, wxFONTSTYLE_NORMAL, wxFONTWEIGHT_BOLD, false, wxT("Sans") ) );

	InfoSizer->Add( m_staticText809, 0, wxALIGN_RIGHT|wxALL, 5 );

	ShouldFilterOutsideMaskStaticText = new wxStaticText( JobDetailsPanel, wxID_ANY, wxEmptyString, wxDefaultPosition, wxDefaultSize, 0 );
	ShouldFilterOutsideMaskStaticText->Wrap( -1 );
	InfoSizer->Add( ShouldFilterOutsideMaskStaticText, 0, wxALL, 5 );

	m_staticText811 = new wxStaticText( JobDetailsPanel, wxID_ANY, wxT("Mask Filter Res. :"), wxDefaultPosition, wxDefaultSize, 0 );
	m_staticText811->Wrap( -1 );
	m_staticText811->SetFont( wxFont( 10, wxFONTFAMILY_SWISS, wxFONTSTYLE_NORMAL, wxFONTWEIGHT_BOLD, false, wxT("Sans") ) );

	InfoSizer->Add( m_staticText811, 0, wxALIGN_RIGHT|wxALL, 5 );

	MaskFilterResolutionStaticText = new wxStaticText( JobDetailsPanel, wxID_ANY, wxEmptyString, wxDefaultPosition, wxDefaultSize, 0 );
	MaskFilterResolutionStaticText->Wrap( -1 );
	InfoSizer->Add( MaskFilterResolutionStaticText, 0, wxALL, 5 );

	m_staticText813 = new wxStaticText( JobDetailsPanel, wxID_ANY, wxT("Reconstruction ID :"), wxDefaultPosition, wxDefaultSize, 0 );
	m_staticText813->Wrap( -1 );
	m_staticText813->SetFont( wxFont( 10, wxFONTFAMILY_SWISS, wxFONTSTYLE_NORMAL, wxFONTWEIGHT_BOLD, false, wxT("Sans") ) );

	InfoSizer->Add( m_staticText813, 0, wxALIGN_RIGHT|wxALL, 5 );

	ReconstructionIDStaticText = new wxStaticText( JobDetailsPanel, wxID_ANY, wxEmptyString, wxDefaultPosition, wxDefaultSize, 0 );
	ReconstructionIDStaticText->Wrap( -1 );
	InfoSizer->Add( ReconstructionIDStaticText, 0, wxALL, 5 );

	m_staticText815 = new wxStaticText( JobDetailsPanel, wxID_ANY, wxT("Inner Mask Rad. :"), wxDefaultPosition, wxDefaultSize, 0 );
	m_staticText815->Wrap( -1 );
	m_staticText815->SetFont( wxFont( 10, wxFONTFAMILY_SWISS, wxFONTSTYLE_NORMAL, wxFONTWEIGHT_BOLD, false, wxT("Sans") ) );

	InfoSizer->Add( m_staticText815, 0, wxALIGN_RIGHT|wxALL, 5 );

	InnerMaskRadiusStaticText = new wxStaticText( JobDetailsPanel, wxID_ANY, wxEmptyString, wxDefaultPosition, wxDefaultSize, 0 );
	InnerMaskRadiusStaticText->Wrap( -1 );
	InfoSizer->Add( InnerMaskRadiusStaticText, 0, wxALL, 5 );

	m_staticText817 = new wxStaticText( JobDetailsPanel, wxID_ANY, wxT("Outer Mask Rad. :"), wxDefaultPosition, wxDefaultSize, 0 );
	m_staticText817->Wrap( -1 );
	m_staticText817->SetFont( wxFont( 10, wxFONTFAMILY_SWISS, wxFONTSTYLE_NORMAL, wxFONTWEIGHT_BOLD, false, wxT("Sans") ) );

	InfoSizer->Add( m_staticText817, 0, wxALIGN_RIGHT|wxALL, 5 );

	OuterMaskRadiusStaticText = new wxStaticText( JobDetailsPanel, wxID_ANY, wxEmptyString, wxDefaultPosition, wxDefaultSize, 0 );
	OuterMaskRadiusStaticText->Wrap( -1 );
	InfoSizer->Add( OuterMaskRadiusStaticText, 0, wxALL, 5 );

	m_staticText820 = new wxStaticText( JobDetailsPanel, wxID_ANY, wxT("Res. Cut-Off :"), wxDefaultPosition, wxDefaultSize, 0 );
	m_staticText820->Wrap( -1 );
	m_staticText820->SetFont( wxFont( 10, wxFONTFAMILY_SWISS, wxFONTSTYLE_NORMAL, wxFONTWEIGHT_BOLD, false, wxT("Sans") ) );

	InfoSizer->Add( m_staticText820, 0, wxALIGN_RIGHT|wxALL, 5 );

	ResolutionCutOffStaticText = new wxStaticText( JobDetailsPanel, wxID_ANY, wxEmptyString, wxDefaultPosition, wxDefaultSize, 0 );
	ResolutionCutOffStaticText->Wrap( -1 );
	InfoSizer->Add( ResolutionCutOffStaticText, 0, wxALL, 5 );

	Score = new wxStaticText( JobDetailsPanel, wxID_ANY, wxT("Weight Constant :"), wxDefaultPosition, wxDefaultSize, 0 );
	Score->Wrap( -1 );
	Score->SetFont( wxFont( 10, wxFONTFAMILY_SWISS, wxFONTSTYLE_NORMAL, wxFONTWEIGHT_BOLD, false, wxT("Sans") ) );

	InfoSizer->Add( Score, 0, wxALIGN_RIGHT|wxALL, 5 );

	ScoreWeightConstantStaticText = new wxStaticText( JobDetailsPanel, wxID_ANY, wxEmptyString, wxDefaultPosition, wxDefaultSize, 0 );
	ScoreWeightConstantStaticText->Wrap( -1 );
	InfoSizer->Add( ScoreWeightConstantStaticText, 0, wxALL, 5 );

	m_staticText823 = new wxStaticText( JobDetailsPanel, wxID_ANY, wxT("Adjust Scores? :"), wxDefaultPosition, wxDefaultSize, 0 );
	m_staticText823->Wrap( -1 );
	m_staticText823->SetFont( wxFont( 10, wxFONTFAMILY_SWISS, wxFONTSTYLE_NORMAL, wxFONTWEIGHT_BOLD, false, wxT("Sans") ) );

	InfoSizer->Add( m_staticText823, 0, wxALIGN_RIGHT|wxALL, 5 );

	AdjustScoresStaticText = new wxStaticText( JobDetailsPanel, wxID_ANY, wxEmptyString, wxDefaultPosition, wxDefaultSize, 0 );
	AdjustScoresStaticText->Wrap( -1 );
	InfoSizer->Add( AdjustScoresStaticText, 0, wxALL, 5 );

	m_staticText825 = new wxStaticText( JobDetailsPanel, wxID_ANY, wxT("Crop Images? :"), wxDefaultPosition, wxDefaultSize, 0 );
	m_staticText825->Wrap( -1 );
	m_staticText825->SetFont( wxFont( 10, wxFONTFAMILY_SWISS, wxFONTSTYLE_NORMAL, wxFONTWEIGHT_BOLD, false, wxT("Sans") ) );

	InfoSizer->Add( m_staticText825, 0, wxALIGN_RIGHT|wxALL, 5 );

	ShouldCropImagesStaticText = new wxStaticText( JobDetailsPanel, wxID_ANY, wxEmptyString, wxDefaultPosition, wxDefaultSize, 0 );
	ShouldCropImagesStaticText->Wrap( -1 );
	InfoSizer->Add( ShouldCropImagesStaticText, 0, wxALL, 5 );

	m_staticText827 = new wxStaticText( JobDetailsPanel, wxID_ANY, wxT("Likelihood Blur? :"), wxDefaultPosition, wxDefaultSize, 0 );
	m_staticText827->Wrap( -1 );
	m_staticText827->SetFont( wxFont( 10, wxFONTFAMILY_SWISS, wxFONTSTYLE_NORMAL, wxFONTWEIGHT_BOLD, false, wxT("Sans") ) );

	InfoSizer->Add( m_staticText827, 0, wxALIGN_RIGHT|wxALL, 5 );

	ShouldLikelihoodBlurStaticText = new wxStaticText( JobDetailsPanel, wxID_ANY, wxEmptyString, wxDefaultPosition, wxDefaultSize, 0 );
	ShouldLikelihoodBlurStaticText->Wrap( -1 );
	InfoSizer->Add( ShouldLikelihoodBlurStaticText, 0, wxALL, 5 );

	m_staticText829 = new wxStaticText( JobDetailsPanel, wxID_ANY, wxT("Smoothing Factor :"), wxDefaultPosition, wxDefaultSize, 0 );
	m_staticText829->Wrap( -1 );
	m_staticText829->SetFont( wxFont( 10, wxFONTFAMILY_SWISS, wxFONTSTYLE_NORMAL, wxFONTWEIGHT_BOLD, false, wxT("Sans") ) );

	InfoSizer->Add( m_staticText829, 0, wxALIGN_RIGHT|wxALL, 5 );

	SmoothingFactorStaticText = new wxStaticText( JobDetailsPanel, wxID_ANY, wxEmptyString, wxDefaultPosition, wxDefaultSize, 0 );
	SmoothingFactorStaticText->Wrap( -1 );
	InfoSizer->Add( SmoothingFactorStaticText, 0, wxALL, 5 );


	bSizer101->Add( InfoSizer, 1, wxEXPAND, 5 );

	m_staticline30 = new wxStaticLine( JobDetailsPanel, wxID_ANY, wxDefaultPosition, wxDefaultSize, wxLI_HORIZONTAL );
	bSizer101->Add( m_staticline30, 0, wxEXPAND | wxALL, 5 );


	bSizer270->Add( bSizer101, 1, wxEXPAND, 5 );


	JobDetailsPanel->SetSizer( bSizer270 );
	JobDetailsPanel->Layout();
	bSizer270->Fit( JobDetailsPanel );
	bSizer203->Add( JobDetailsPanel, 0, wxEXPAND | wxALL, 5 );

	OrthPanel = new DisplayPanel( RightPanel, wxID_ANY, wxDefaultPosition, wxDefaultSize, wxTAB_TRAVERSAL );
	bSizer203->Add( OrthPanel, 1, wxEXPAND | wxALL, 5 );


	RightPanel->SetSizer( bSizer203 );
	RightPanel->Layout();
	bSizer203->Fit( RightPanel );
	m_splitter7->SplitVertically( m_panel49, RightPanel, 900 );
	bSizer202->Add( m_splitter7, 1, wxEXPAND, 5 );


	this->SetSizer( bSizer202 );
	this->Layout();

	// Connect Events
	this->Connect( wxEVT_UPDATE_UI, wxUpdateUIEventHandler( RefinementResultsPanel::OnUpdateUI ) );
	ParametersDetailButton->Connect( wxEVT_COMMAND_BUTTON_CLICKED, wxCommandEventHandler( RefinementResultsPanel::PopupParametersClick ), NULL, this );
	AngularPlotDetailsButton->Connect( wxEVT_COMMAND_BUTTON_CLICKED, wxCommandEventHandler( RefinementResultsPanel::AngularPlotPopupClick ), NULL, this );
	JobDetailsToggleButton->Connect( wxEVT_COMMAND_TOGGLEBUTTON_CLICKED, wxCommandEventHandler( RefinementResultsPanel::OnJobDetailsToggle ), NULL, this );
}

RefinementResultsPanel::~RefinementResultsPanel()
{
	// Disconnect Events
	this->Disconnect( wxEVT_UPDATE_UI, wxUpdateUIEventHandler( RefinementResultsPanel::OnUpdateUI ) );
	ParametersDetailButton->Disconnect( wxEVT_COMMAND_BUTTON_CLICKED, wxCommandEventHandler( RefinementResultsPanel::PopupParametersClick ), NULL, this );
	AngularPlotDetailsButton->Disconnect( wxEVT_COMMAND_BUTTON_CLICKED, wxCommandEventHandler( RefinementResultsPanel::AngularPlotPopupClick ), NULL, this );
	JobDetailsToggleButton->Disconnect( wxEVT_COMMAND_TOGGLEBUTTON_CLICKED, wxCommandEventHandler( RefinementResultsPanel::OnJobDetailsToggle ), NULL, this );

}

Refine3DPanel::Refine3DPanel( wxWindow* parent, wxWindowID id, const wxPoint& pos, const wxSize& size, long style, const wxString& name ) : JobPanel( parent, id, pos, size, style, name )
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
	bSizer215 = new wxBoxSizer( wxHORIZONTAL );

	LocalRefinementRadio = new wxRadioButton( InputParamsPanel, wxID_ANY, wxT("Local Refinement"), wxDefaultPosition, wxDefaultSize, 0 );
	bSizer215->Add( LocalRefinementRadio, 0, wxALIGN_CENTER_VERTICAL|wxALIGN_RIGHT|wxALL, 5 );

	GlobalRefinementRadio = new wxRadioButton( InputParamsPanel, wxID_ANY, wxT("Global Search"), wxDefaultPosition, wxDefaultSize, 0 );
	bSizer215->Add( GlobalRefinementRadio, 0, wxALIGN_CENTER_VERTICAL|wxALIGN_RIGHT|wxALL, 5 );


	bSizer215->Add( 0, 0, 1, wxEXPAND, 5 );


	bSizer357->Add( bSizer215, 0, wxEXPAND, 5 );

	wxFlexGridSizer* fgSizer22;
	fgSizer22 = new wxFlexGridSizer( 0, 2, 0, 0 );
	fgSizer22->SetFlexibleDirection( wxBOTH );
	fgSizer22->SetNonFlexibleGrowMode( wxFLEX_GROWMODE_SPECIFIED );

	NoCycleStaticText = new wxStaticText( InputParamsPanel, wxID_ANY, wxT("No. of Cycles to Run :"), wxDefaultPosition, wxDefaultSize, 0 );
	NoCycleStaticText->Wrap( -1 );
	fgSizer22->Add( NoCycleStaticText, 0, wxALIGN_CENTER_VERTICAL|wxALIGN_RIGHT|wxALL, 5 );

	NumberRoundsSpinCtrl = new wxSpinCtrl( InputParamsPanel, wxID_ANY, wxEmptyString, wxDefaultPosition, wxDefaultSize, wxSP_ARROW_KEYS, 1, 100, 1 );
	NumberRoundsSpinCtrl->SetMinSize( wxSize( 100,-1 ) );

	fgSizer22->Add( NumberRoundsSpinCtrl, 1, wxALIGN_CENTER_VERTICAL|wxALL|wxEXPAND, 5 );

	HiResLimitStaticText = new wxStaticText( InputParamsPanel, wxID_ANY, wxT("Hi-Res Limit (Å) :"), wxDefaultPosition, wxDefaultSize, 0 );
	HiResLimitStaticText->Wrap( -1 );
	fgSizer22->Add( HiResLimitStaticText, 0, wxALIGN_CENTER_VERTICAL|wxALIGN_RIGHT|wxALL, 5 );

	HighResolutionLimitTextCtrl = new NumericTextCtrl( InputParamsPanel, wxID_ANY, wxT("30.00"), wxDefaultPosition, wxDefaultSize, 0 );
	fgSizer22->Add( HighResolutionLimitTextCtrl, 0, wxALIGN_CENTER_VERTICAL|wxALL|wxEXPAND, 5 );


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

	m_staticText318 = new wxStaticText( ExpertPanel, wxID_ANY, wxT("Parameters To Refine"), wxDefaultPosition, wxDefaultSize, 0 );
	m_staticText318->Wrap( -1 );
	m_staticText318->SetFont( wxFont( 10, wxFONTFAMILY_SWISS, wxFONTSTYLE_NORMAL, wxFONTWEIGHT_BOLD, true, wxT("Sans") ) );

	bSizer258->Add( m_staticText318, 0, wxALIGN_CENTER_VERTICAL|wxALL, 5 );


	bSizer258->Add( 0, 0, 1, wxEXPAND, 5 );

	ResetAllDefaultsButton = new wxButton( ExpertPanel, wxID_ANY, wxT("Reset All Defaults"), wxDefaultPosition, wxDefaultSize, 0 );
	bSizer258->Add( ResetAllDefaultsButton, 0, wxALL, 5 );


	bSizer258->Add( 5, 0, 0, wxEXPAND, 5 );


	InputSizer->Add( bSizer258, 0, wxEXPAND, 5 );

	wxBoxSizer* bSizer257;
	bSizer257 = new wxBoxSizer( wxHORIZONTAL );

	wxGridSizer* gSizer14;
	gSizer14 = new wxGridSizer( 0, 3, 0, 0 );

	RefinePsiCheckBox = new wxCheckBox( ExpertPanel, wxID_ANY, wxT("Psi"), wxDefaultPosition, wxDefaultSize, 0 );
	RefinePsiCheckBox->SetValue(true);
	gSizer14->Add( RefinePsiCheckBox, 0, wxALL, 5 );

	RefineThetaCheckBox = new wxCheckBox( ExpertPanel, wxID_ANY, wxT("Theta"), wxDefaultPosition, wxDefaultSize, 0 );
	RefineThetaCheckBox->SetValue(true);
	gSizer14->Add( RefineThetaCheckBox, 0, wxALL, 5 );

	RefinePhiCheckBox = new wxCheckBox( ExpertPanel, wxID_ANY, wxT("Phi"), wxDefaultPosition, wxDefaultSize, 0 );
	RefinePhiCheckBox->SetValue(true);
	gSizer14->Add( RefinePhiCheckBox, 0, wxALL, 5 );

	RefineXShiftCheckBox = new wxCheckBox( ExpertPanel, wxID_ANY, wxT("X Shift"), wxDefaultPosition, wxDefaultSize, 0 );
	RefineXShiftCheckBox->SetValue(true);
	gSizer14->Add( RefineXShiftCheckBox, 0, wxALL, 5 );

	RefineYShiftCheckBox = new wxCheckBox( ExpertPanel, wxID_ANY, wxT("Y Shift"), wxDefaultPosition, wxDefaultSize, 0 );
	RefineYShiftCheckBox->SetValue(true);
	gSizer14->Add( RefineYShiftCheckBox, 0, wxALL, 5 );

	RefineOccupanciesCheckBox = new wxCheckBox( ExpertPanel, wxID_ANY, wxT("Occupancy"), wxDefaultPosition, wxDefaultSize, 0 );
	RefineOccupanciesCheckBox->SetValue(true);
	gSizer14->Add( RefineOccupanciesCheckBox, 0, wxALL, 5 );


	bSizer257->Add( gSizer14, 1, wxEXPAND, 5 );


	InputSizer->Add( bSizer257, 0, wxEXPAND, 5 );


	InputSizer->Add( 0, 5, 0, wxEXPAND, 5 );

	wxFlexGridSizer* fgSizer1;
	fgSizer1 = new wxFlexGridSizer( 0, 2, 0, 0 );
	fgSizer1->SetFlexibleDirection( wxBOTH );
	fgSizer1->SetNonFlexibleGrowMode( wxFLEX_GROWMODE_SPECIFIED );

	m_staticText202 = new wxStaticText( ExpertPanel, wxID_ANY, wxT("General Refinement"), wxDefaultPosition, wxDefaultSize, 0 );
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

	m_staticText362 = new wxStaticText( ExpertPanel, wxID_ANY, wxT("Percent Used (%) :"), wxDefaultPosition, wxDefaultSize, 0 );
	m_staticText362->Wrap( -1 );
	fgSizer1->Add( m_staticText362, 0, wxALIGN_CENTER_VERTICAL|wxALL, 5 );

	PercentUsedTextCtrl = new NumericTextCtrl( ExpertPanel, wxID_ANY, wxT("100.00"), wxDefaultPosition, wxDefaultSize, 0 );
	fgSizer1->Add( PercentUsedTextCtrl, 0, wxALL|wxEXPAND, 5 );

	m_staticText201 = new wxStaticText( ExpertPanel, wxID_ANY, wxT("Global Search"), wxDefaultPosition, wxDefaultSize, 0 );
	m_staticText201->Wrap( -1 );
	m_staticText201->SetFont( wxFont( wxNORMAL_FONT->GetPointSize(), wxFONTFAMILY_DEFAULT, wxFONTSTYLE_NORMAL, wxFONTWEIGHT_BOLD, true, wxEmptyString ) );

	fgSizer1->Add( m_staticText201, 0, wxALL, 5 );


	fgSizer1->Add( 0, 0, 1, wxEXPAND, 5 );

	GlobalMaskRadiusStaticText = new wxStaticText( ExpertPanel, wxID_ANY, wxT("Global Mask Radius (Å) :"), wxDefaultPosition, wxDefaultSize, 0 );
	GlobalMaskRadiusStaticText->Wrap( -1 );
	fgSizer1->Add( GlobalMaskRadiusStaticText, 0, wxALIGN_CENTER_VERTICAL|wxALL, 5 );

	GlobalMaskRadiusTextCtrl = new NumericTextCtrl( ExpertPanel, wxID_ANY, wxT("30.00"), wxDefaultPosition, wxDefaultSize, wxTE_PROCESS_ENTER );
	fgSizer1->Add( GlobalMaskRadiusTextCtrl, 0, wxALL|wxEXPAND, 5 );

	NumberToRefineStaticText = new wxStaticText( ExpertPanel, wxID_ANY, wxT("Number of Results to Refine :"), wxDefaultPosition, wxDefaultSize, 0 );
	NumberToRefineStaticText->Wrap( -1 );
	fgSizer1->Add( NumberToRefineStaticText, 0, wxALIGN_CENTER_VERTICAL|wxALL, 5 );

	NumberToRefineSpinCtrl = new wxSpinCtrl( ExpertPanel, wxID_ANY, wxEmptyString, wxDefaultPosition, wxDefaultSize, wxSP_ARROW_KEYS, -100000, 100000, 20 );
	fgSizer1->Add( NumberToRefineSpinCtrl, 0, wxALL, 5 );

	AlsoRefineInputStaticText1 = new wxStaticText( ExpertPanel, wxID_ANY, wxT("\tAlso Refine Input Parameters?"), wxDefaultPosition, wxDefaultSize, 0 );
	AlsoRefineInputStaticText1->Wrap( -1 );
	AlsoRefineInputStaticText1->Enable( false );

	fgSizer1->Add( AlsoRefineInputStaticText1, 0, wxALL, 5 );

	wxBoxSizer* bSizer2631;
	bSizer2631 = new wxBoxSizer( wxHORIZONTAL );

	AlsoRefineInputYesRadio = new wxRadioButton( ExpertPanel, wxID_ANY, wxT("Yes"), wxDefaultPosition, wxDefaultSize, wxRB_GROUP );
	AlsoRefineInputYesRadio->SetValue( true );
	AlsoRefineInputYesRadio->Enable( false );

	bSizer2631->Add( AlsoRefineInputYesRadio, 0, wxALL, 5 );

	AlsoRefineInputNoRadio = new wxRadioButton( ExpertPanel, wxID_ANY, wxT("No"), wxDefaultPosition, wxDefaultSize, 0 );
	AlsoRefineInputNoRadio->Enable( false );

	bSizer2631->Add( AlsoRefineInputNoRadio, 0, wxALL, 5 );


	fgSizer1->Add( bSizer2631, 1, wxEXPAND, 5 );

	AngularStepStaticText = new wxStaticText( ExpertPanel, wxID_ANY, wxT("Angular Search Step (°) :"), wxDefaultPosition, wxDefaultSize, 0 );
	AngularStepStaticText->Wrap( -1 );
	fgSizer1->Add( AngularStepStaticText, 0, wxALIGN_CENTER_VERTICAL|wxALL, 5 );

	AngularStepTextCtrl = new NumericTextCtrl( ExpertPanel, wxID_ANY, wxT("20.00"), wxDefaultPosition, wxDefaultSize, wxTE_PROCESS_ENTER );
	fgSizer1->Add( AngularStepTextCtrl, 0, wxALL|wxEXPAND, 5 );

	SearchRangeXStaticText = new wxStaticText( ExpertPanel, wxID_ANY, wxT("Search Range in X (Å) :"), wxDefaultPosition, wxDefaultSize, 0 );
	SearchRangeXStaticText->Wrap( -1 );
	fgSizer1->Add( SearchRangeXStaticText, 0, wxALIGN_CENTER_VERTICAL|wxALL, 5 );

	SearchRangeXTextCtrl = new NumericTextCtrl( ExpertPanel, wxID_ANY, wxT("100.00"), wxDefaultPosition, wxDefaultSize, wxTE_PROCESS_ENTER );
	fgSizer1->Add( SearchRangeXTextCtrl, 0, wxALL|wxEXPAND, 5 );

	SearchRangeYStaticText = new wxStaticText( ExpertPanel, wxID_ANY, wxT("Search Range in Y (Å) :"), wxDefaultPosition, wxDefaultSize, 0 );
	SearchRangeYStaticText->Wrap( -1 );
	fgSizer1->Add( SearchRangeYStaticText, 0, wxALIGN_CENTER_VERTICAL|wxALL, 5 );

	SearchRangeYTextCtrl = new NumericTextCtrl( ExpertPanel, wxID_ANY, wxT("100.00"), wxDefaultPosition, wxDefaultSize, wxTE_PROCESS_ENTER );
	fgSizer1->Add( SearchRangeYTextCtrl, 0, wxALL|wxEXPAND, 5 );

	m_staticText200 = new wxStaticText( ExpertPanel, wxID_ANY, wxT("Classification"), wxDefaultPosition, wxDefaultSize, 0 );
	m_staticText200->Wrap( -1 );
	m_staticText200->SetFont( wxFont( 10, wxFONTFAMILY_SWISS, wxFONTSTYLE_NORMAL, wxFONTWEIGHT_BOLD, true, wxT("Sans") ) );

	fgSizer1->Add( m_staticText200, 0, wxALL, 5 );


	fgSizer1->Add( 0, 0, 1, wxEXPAND, 5 );

	MinPhaseShiftStaticText = new wxStaticText( ExpertPanel, wxID_ANY, wxT("High-Resolution Limit (Å) :"), wxDefaultPosition, wxDefaultSize, 0 );
	MinPhaseShiftStaticText->Wrap( -1 );
	fgSizer1->Add( MinPhaseShiftStaticText, 0, wxALIGN_CENTER_VERTICAL|wxALL, 5 );

	ClassificationHighResLimitTextCtrl = new NumericTextCtrl( ExpertPanel, wxID_ANY, wxT("10.00"), wxDefaultPosition, wxDefaultSize, wxTE_PROCESS_ENTER );
	fgSizer1->Add( ClassificationHighResLimitTextCtrl, 0, wxALL|wxEXPAND, 5 );

	PhaseShiftStepStaticText = new wxStaticText( ExpertPanel, wxID_ANY, wxT("Focused Classification?"), wxDefaultPosition, wxDefaultSize, 0 );
	PhaseShiftStepStaticText->Wrap( -1 );
	fgSizer1->Add( PhaseShiftStepStaticText, 0, wxALIGN_CENTER_VERTICAL|wxALL, 5 );

	wxBoxSizer* bSizer263;
	bSizer263 = new wxBoxSizer( wxHORIZONTAL );

	SphereClassificatonYesRadio = new wxRadioButton( ExpertPanel, wxID_ANY, wxT("Yes"), wxDefaultPosition, wxDefaultSize, wxRB_GROUP );
	bSizer263->Add( SphereClassificatonYesRadio, 0, wxALL, 5 );

	SphereClassificatonNoRadio = new wxRadioButton( ExpertPanel, wxID_ANY, wxT("No"), wxDefaultPosition, wxDefaultSize, 0 );
	bSizer263->Add( SphereClassificatonNoRadio, 0, wxALL, 5 );


	fgSizer1->Add( bSizer263, 1, wxEXPAND, 5 );

	SphereXStaticText = new wxStaticText( ExpertPanel, wxID_ANY, wxT("\tSphere X Co-ordinate (Å) :"), wxDefaultPosition, wxDefaultSize, 0 );
	SphereXStaticText->Wrap( -1 );
	SphereXStaticText->Enable( false );

	fgSizer1->Add( SphereXStaticText, 0, wxALIGN_CENTER_VERTICAL|wxALL, 5 );

	SphereXTextCtrl = new NumericTextCtrl( ExpertPanel, wxID_ANY, wxT("0.00"), wxDefaultPosition, wxDefaultSize, 0 );
	SphereXTextCtrl->Enable( false );

	fgSizer1->Add( SphereXTextCtrl, 0, wxALL|wxEXPAND, 5 );

	SphereYStaticText = new wxStaticText( ExpertPanel, wxID_ANY, wxT("\tSphere Y Co-ordinate (Å) :"), wxDefaultPosition, wxDefaultSize, 0 );
	SphereYStaticText->Wrap( -1 );
	SphereYStaticText->Enable( false );

	fgSizer1->Add( SphereYStaticText, 0, wxALIGN_CENTER_VERTICAL|wxALL, 5 );

	SphereYTextCtrl = new NumericTextCtrl( ExpertPanel, wxID_ANY, wxT("0.00"), wxDefaultPosition, wxDefaultSize, 0 );
	SphereYTextCtrl->Enable( false );

	fgSizer1->Add( SphereYTextCtrl, 0, wxALL|wxEXPAND, 5 );

	SphereZStaticText = new wxStaticText( ExpertPanel, wxID_ANY, wxT("\tSphere Z Co-ordinate (Å) :"), wxDefaultPosition, wxDefaultSize, 0 );
	SphereZStaticText->Wrap( -1 );
	SphereZStaticText->Enable( false );

	fgSizer1->Add( SphereZStaticText, 0, wxALIGN_CENTER_VERTICAL|wxALIGN_LEFT|wxALL, 5 );

	SphereZTextCtrl = new NumericTextCtrl( ExpertPanel, wxID_ANY, wxT("0.00"), wxDefaultPosition, wxDefaultSize, 0 );
	SphereZTextCtrl->Enable( false );

	fgSizer1->Add( SphereZTextCtrl, 0, wxALL|wxEXPAND, 5 );

	SphereRadiusStaticText = new wxStaticText( ExpertPanel, wxID_ANY, wxT("\tSphere Radius (Å) :"), wxDefaultPosition, wxDefaultSize, 0 );
	SphereRadiusStaticText->Wrap( -1 );
	SphereRadiusStaticText->Enable( false );

	fgSizer1->Add( SphereRadiusStaticText, 0, wxALIGN_LEFT|wxALL, 5 );

	SphereRadiusTextCtrl = new NumericTextCtrl( ExpertPanel, wxID_ANY, wxT("0.00"), wxDefaultPosition, wxDefaultSize, 0 );
	SphereRadiusTextCtrl->Enable( false );

	fgSizer1->Add( SphereRadiusTextCtrl, 0, wxALL|wxEXPAND, 5 );

	m_staticText323 = new wxStaticText( ExpertPanel, wxID_ANY, wxT("CTF Refinement"), wxDefaultPosition, wxDefaultSize, 0 );
	m_staticText323->Wrap( -1 );
	m_staticText323->SetFont( wxFont( 10, wxFONTFAMILY_SWISS, wxFONTSTYLE_NORMAL, wxFONTWEIGHT_BOLD, true, wxT("Sans") ) );

	fgSizer1->Add( m_staticText323, 0, wxALL, 5 );


	fgSizer1->Add( 0, 0, 1, wxEXPAND, 5 );

	m_staticText324 = new wxStaticText( ExpertPanel, wxID_ANY, wxT("Refine Defocus?"), wxDefaultPosition, wxDefaultSize, 0 );
	m_staticText324->Wrap( -1 );
	fgSizer1->Add( m_staticText324, 0, wxALIGN_CENTER_VERTICAL|wxALL, 5 );

	wxBoxSizer* bSizer264;
	bSizer264 = new wxBoxSizer( wxHORIZONTAL );

	RefineCTFYesRadio = new wxRadioButton( ExpertPanel, wxID_ANY, wxT("Yes"), wxDefaultPosition, wxDefaultSize, wxRB_GROUP );
	bSizer264->Add( RefineCTFYesRadio, 0, wxALL, 5 );

	RefineCTFNoRadio = new wxRadioButton( ExpertPanel, wxID_ANY, wxT("No"), wxDefaultPosition, wxDefaultSize, 0 );
	bSizer264->Add( RefineCTFNoRadio, 0, wxALL, 5 );


	fgSizer1->Add( bSizer264, 1, wxEXPAND, 5 );

	DefocusSearchRangeStaticText = new wxStaticText( ExpertPanel, wxID_ANY, wxT("\tDefocus Search Range (Å) :"), wxDefaultPosition, wxDefaultSize, 0 );
	DefocusSearchRangeStaticText->Wrap( -1 );
	DefocusSearchRangeStaticText->Enable( false );

	fgSizer1->Add( DefocusSearchRangeStaticText, 0, wxALIGN_CENTER_VERTICAL|wxALL, 5 );

	DefocusSearchRangeTextCtrl = new NumericTextCtrl( ExpertPanel, wxID_ANY, wxT("500.00"), wxDefaultPosition, wxDefaultSize, 0 );
	DefocusSearchRangeTextCtrl->Enable( false );

	fgSizer1->Add( DefocusSearchRangeTextCtrl, 0, wxALL|wxEXPAND, 5 );

	DefocusSearchStepStaticText = new wxStaticText( ExpertPanel, wxID_ANY, wxT("\tDefocus Search Step (Å) : "), wxDefaultPosition, wxDefaultSize, 0 );
	DefocusSearchStepStaticText->Wrap( -1 );
	DefocusSearchStepStaticText->Enable( false );

	fgSizer1->Add( DefocusSearchStepStaticText, 0, wxALIGN_CENTER_VERTICAL|wxALL, 5 );

	DefocusSearchStepTextCtrl = new NumericTextCtrl( ExpertPanel, wxID_ANY, wxT("100.00"), wxDefaultPosition, wxDefaultSize, 0 );
	DefocusSearchStepTextCtrl->Enable( false );

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

	AutoCenterStaticText = new wxStaticText( ExpertPanel, wxID_ANY, wxT("Autocenter reconstruction?"), wxDefaultPosition, wxDefaultSize, 0 );
	AutoCenterStaticText->Wrap( -1 );
	fgSizer1->Add( AutoCenterStaticText, 0, wxALL, 5 );

	wxBoxSizer* bSizer266121;
	bSizer266121 = new wxBoxSizer( wxHORIZONTAL );

	AutoCenterYesRadioButton = new wxRadioButton( ExpertPanel, wxID_ANY, wxT("Yes"), wxDefaultPosition, wxDefaultSize, wxRB_GROUP );
	bSizer266121->Add( AutoCenterYesRadioButton, 0, wxALL, 5 );

	AutoCenterNoRadioButton = new wxRadioButton( ExpertPanel, wxID_ANY, wxT("No"), wxDefaultPosition, wxDefaultSize, 0 );
	bSizer266121->Add( AutoCenterNoRadioButton, 0, wxALL, 5 );


	fgSizer1->Add( bSizer266121, 1, wxEXPAND, 5 );

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

	ShowRefinementResultsPanel = new DisplayRefinementResultsPanel( this, wxID_ANY, wxDefaultPosition, wxDefaultSize, wxTAB_TRAVERSAL );
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
	this->Connect( wxEVT_UPDATE_UI, wxUpdateUIEventHandler( Refine3DPanel::OnUpdateUI ) );
	UseMaskCheckBox->Connect( wxEVT_COMMAND_CHECKBOX_CLICKED, wxCommandEventHandler( Refine3DPanel::OnUseMaskCheckBox ), NULL, this );
	HighResolutionLimitTextCtrl->Connect( wxEVT_COMMAND_TEXT_UPDATED, wxCommandEventHandler( Refine3DPanel::OnHighResLimitChange ), NULL, this );
	ExpertToggleButton->Connect( wxEVT_COMMAND_TOGGLEBUTTON_CLICKED, wxCommandEventHandler( Refine3DPanel::OnExpertOptionsToggle ), NULL, this );
	Active3DReferencesListCtrl->Connect( wxEVT_COMMAND_LIST_ITEM_ACTIVATED, wxListEventHandler( Refine3DPanel::OnVolumeListItemActivated ), NULL, this );
	ResetAllDefaultsButton->Connect( wxEVT_COMMAND_BUTTON_CLICKED, wxCommandEventHandler( Refine3DPanel::ResetAllDefaultsClick ), NULL, this );
	AutoMaskYesRadioButton->Connect( wxEVT_COMMAND_RADIOBUTTON_SELECTED, wxCommandEventHandler( Refine3DPanel::OnAutoMaskButton ), NULL, this );
	AutoMaskNoRadioButton->Connect( wxEVT_COMMAND_RADIOBUTTON_SELECTED, wxCommandEventHandler( Refine3DPanel::OnAutoMaskButton ), NULL, this );
	InfoText->Connect( wxEVT_COMMAND_TEXT_URL, wxTextUrlEventHandler( Refine3DPanel::OnInfoURL ), NULL, this );
	FinishButton->Connect( wxEVT_COMMAND_BUTTON_CLICKED, wxCommandEventHandler( Refine3DPanel::FinishButtonClick ), NULL, this );
	CancelAlignmentButton->Connect( wxEVT_COMMAND_BUTTON_CLICKED, wxCommandEventHandler( Refine3DPanel::TerminateButtonClick ), NULL, this );
	StartRefinementButton->Connect( wxEVT_COMMAND_BUTTON_CLICKED, wxCommandEventHandler( Refine3DPanel::StartRefinementClick ), NULL, this );
}

Refine3DPanel::~Refine3DPanel()
{
	// Disconnect Events
	this->Disconnect( wxEVT_UPDATE_UI, wxUpdateUIEventHandler( Refine3DPanel::OnUpdateUI ) );
	UseMaskCheckBox->Disconnect( wxEVT_COMMAND_CHECKBOX_CLICKED, wxCommandEventHandler( Refine3DPanel::OnUseMaskCheckBox ), NULL, this );
	HighResolutionLimitTextCtrl->Disconnect( wxEVT_COMMAND_TEXT_UPDATED, wxCommandEventHandler( Refine3DPanel::OnHighResLimitChange ), NULL, this );
	ExpertToggleButton->Disconnect( wxEVT_COMMAND_TOGGLEBUTTON_CLICKED, wxCommandEventHandler( Refine3DPanel::OnExpertOptionsToggle ), NULL, this );
	Active3DReferencesListCtrl->Disconnect( wxEVT_COMMAND_LIST_ITEM_ACTIVATED, wxListEventHandler( Refine3DPanel::OnVolumeListItemActivated ), NULL, this );
	ResetAllDefaultsButton->Disconnect( wxEVT_COMMAND_BUTTON_CLICKED, wxCommandEventHandler( Refine3DPanel::ResetAllDefaultsClick ), NULL, this );
	AutoMaskYesRadioButton->Disconnect( wxEVT_COMMAND_RADIOBUTTON_SELECTED, wxCommandEventHandler( Refine3DPanel::OnAutoMaskButton ), NULL, this );
	AutoMaskNoRadioButton->Disconnect( wxEVT_COMMAND_RADIOBUTTON_SELECTED, wxCommandEventHandler( Refine3DPanel::OnAutoMaskButton ), NULL, this );
	InfoText->Disconnect( wxEVT_COMMAND_TEXT_URL, wxTextUrlEventHandler( Refine3DPanel::OnInfoURL ), NULL, this );
	FinishButton->Disconnect( wxEVT_COMMAND_BUTTON_CLICKED, wxCommandEventHandler( Refine3DPanel::FinishButtonClick ), NULL, this );
	CancelAlignmentButton->Disconnect( wxEVT_COMMAND_BUTTON_CLICKED, wxCommandEventHandler( Refine3DPanel::TerminateButtonClick ), NULL, this );
	StartRefinementButton->Disconnect( wxEVT_COMMAND_BUTTON_CLICKED, wxCommandEventHandler( Refine3DPanel::StartRefinementClick ), NULL, this );

}

Sharpen3DPanelParent::Sharpen3DPanelParent( wxWindow* parent, wxWindowID id, const wxPoint& pos, const wxSize& size, long style, const wxString& name ) : wxPanel( parent, id, pos, size, style, name )
{
	wxBoxSizer* bSizer589;
	bSizer589 = new wxBoxSizer( wxVERTICAL );

	wxFlexGridSizer* fgSizer15;
	fgSizer15 = new wxFlexGridSizer( 0, 2, 0, 0 );
	fgSizer15->SetFlexibleDirection( wxBOTH );
	fgSizer15->SetNonFlexibleGrowMode( wxFLEX_GROWMODE_SPECIFIED );

	m_staticText262 = new wxStaticText( this, wxID_ANY, wxT("Input Volume :"), wxDefaultPosition, wxSize( -1,-1 ), 0 );
	m_staticText262->Wrap( -1 );
	fgSizer15->Add( m_staticText262, 0, wxALIGN_CENTER_VERTICAL|wxALIGN_RIGHT|wxALL, 5 );

	VolumeComboBox = new VolumeAssetPickerComboPanel( this, wxID_ANY, wxDefaultPosition, wxDefaultSize, wxTAB_TRAVERSAL );
	VolumeComboBox->SetMinSize( wxSize( 350,-1 ) );

	fgSizer15->Add( VolumeComboBox, 1, wxEXPAND | wxALL, 5 );

	UseMaskCheckBox = new wxCheckBox( this, wxID_ANY, wxT("Supply a Mask? :"), wxDefaultPosition, wxDefaultSize, 0 );
	fgSizer15->Add( UseMaskCheckBox, 0, wxALIGN_CENTER_VERTICAL|wxALIGN_RIGHT|wxALL, 5 );

	MaskSelectPanel = new VolumeAssetPickerComboPanel( this, wxID_ANY, wxDefaultPosition, wxDefaultSize, wxTAB_TRAVERSAL );
	MaskSelectPanel->SetMinSize( wxSize( 350,-1 ) );

	fgSizer15->Add( MaskSelectPanel, 1, wxALIGN_CENTER_VERTICAL|wxALL|wxEXPAND, 5 );


	bSizer589->Add( fgSizer15, 0, wxEXPAND, 5 );

	m_staticline129 = new wxStaticLine( this, wxID_ANY, wxDefaultPosition, wxDefaultSize, wxLI_HORIZONTAL );
	bSizer589->Add( m_staticline129, 0, wxEXPAND | wxALL, 5 );

	wxBoxSizer* bSizer479;
	bSizer479 = new wxBoxSizer( wxHORIZONTAL );

	wxBoxSizer* bSizer488;
	bSizer488 = new wxBoxSizer( wxHORIZONTAL );

	InputSizer = new wxBoxSizer( wxVERTICAL );

	wxBoxSizer* bSizer491;
	bSizer491 = new wxBoxSizer( wxHORIZONTAL );

	wxFlexGridSizer* fgSizer1;
	fgSizer1 = new wxFlexGridSizer( 0, 2, 0, 0 );
	fgSizer1->SetFlexibleDirection( wxBOTH );
	fgSizer1->SetNonFlexibleGrowMode( wxFLEX_GROWMODE_SPECIFIED );

	m_staticText1006 = new wxStaticText( this, wxID_ANY, wxT("Filtering"), wxDefaultPosition, wxDefaultSize, 0 );
	m_staticText1006->Wrap( -1 );
	m_staticText1006->SetFont( wxFont( 10, wxFONTFAMILY_SWISS, wxFONTSTYLE_NORMAL, wxFONTWEIGHT_BOLD, true, wxT("Sans") ) );

	fgSizer1->Add( m_staticText1006, 0, wxALL, 5 );


	fgSizer1->Add( 0, 0, 1, wxEXPAND, 5 );


	fgSizer1->Add( 0, 0, 1, wxEXPAND, 5 );


	fgSizer1->Add( 0, 0, 1, wxEXPAND, 5 );

	m_staticText1007 = new wxStaticText( this, wxID_ANY, wxT("Flatten From Res. (Å) :"), wxDefaultPosition, wxDefaultSize, 0 );
	m_staticText1007->Wrap( -1 );
	fgSizer1->Add( m_staticText1007, 0, wxALIGN_CENTER_VERTICAL|wxALIGN_RIGHT|wxALL, 5 );

	FlattenFromTextCtrl = new NumericTextCtrl( this, wxID_ANY, wxT("8.00"), wxDefaultPosition, wxDefaultSize, 0 );
	fgSizer1->Add( FlattenFromTextCtrl, 0, wxALIGN_CENTER_VERTICAL|wxALL, 5 );

	m_staticText10081 = new wxStaticText( this, wxID_ANY, wxT("Resolution Cut-Off (Å) :"), wxDefaultPosition, wxDefaultSize, 0 );
	m_staticText10081->Wrap( -1 );
	fgSizer1->Add( m_staticText10081, 0, wxALIGN_CENTER_VERTICAL|wxALIGN_RIGHT|wxALL, 5 );

	CutOffResTextCtrl = new NumericTextCtrl( this, wxID_ANY, wxT("0.00"), wxDefaultPosition, wxDefaultSize, 0 );
	fgSizer1->Add( CutOffResTextCtrl, 0, wxALL, 5 );

	m_staticText638 = new wxStaticText( this, wxID_ANY, wxT("Pre-Cut-Off B-Factor (Å²) :"), wxDefaultPosition, wxDefaultSize, 0 );
	m_staticText638->Wrap( -1 );
	fgSizer1->Add( m_staticText638, 0, wxALIGN_CENTER_VERTICAL|wxALIGN_RIGHT|wxALL, 5 );

	AdditionalLowBFactorTextCtrl = new NumericTextCtrl( this, wxID_ANY, wxT("-90.0"), wxDefaultPosition, wxDefaultSize, 0 );
	fgSizer1->Add( AdditionalLowBFactorTextCtrl, 0, wxALL, 5 );

	m_staticText600 = new wxStaticText( this, wxID_ANY, wxT("Post-Cut-Off B-Factor (Å²) :"), wxDefaultPosition, wxDefaultSize, 0 );
	m_staticText600->Wrap( -1 );
	fgSizer1->Add( m_staticText600, 0, wxALIGN_CENTER_VERTICAL|wxALIGN_RIGHT|wxALL, 5 );

	AdditionalHighBFactorTextCtrl = new NumericTextCtrl( this, wxID_ANY, wxT("0.00"), wxDefaultPosition, wxDefaultSize, 0 );
	fgSizer1->Add( AdditionalHighBFactorTextCtrl, 0, wxALL, 5 );

	m_staticText10111 = new wxStaticText( this, wxID_ANY, wxT("Filter Edge-Width (Å) :"), wxDefaultPosition, wxDefaultSize, 0 );
	m_staticText10111->Wrap( -1 );
	fgSizer1->Add( m_staticText10111, 0, wxALIGN_CENTER_VERTICAL|wxALIGN_RIGHT|wxALL, 5 );

	FilterEdgeWidthTextCtrl = new NumericTextCtrl( this, wxID_ANY, wxT("20.00"), wxDefaultPosition, wxDefaultSize, 0 );
	fgSizer1->Add( FilterEdgeWidthTextCtrl, 0, wxALL, 5 );

	UseFSCWeightingStaticText = new wxStaticText( this, wxID_ANY, wxT("Use FOM Weighting? :"), wxDefaultPosition, wxDefaultSize, 0 );
	UseFSCWeightingStaticText->Wrap( -1 );
	fgSizer1->Add( UseFSCWeightingStaticText, 0, wxALIGN_CENTER_VERTICAL|wxALIGN_RIGHT|wxALL, 5 );

	wxBoxSizer* bSizer266121;
	bSizer266121 = new wxBoxSizer( wxHORIZONTAL );

	UseFSCWeightingYesButton = new wxRadioButton( this, wxID_ANY, wxT("Yes"), wxDefaultPosition, wxDefaultSize, wxRB_GROUP );
	bSizer266121->Add( UseFSCWeightingYesButton, 0, wxALL, 5 );

	UseFSCWeightingNoButton = new wxRadioButton( this, wxID_ANY, wxT("No"), wxDefaultPosition, wxDefaultSize, 0 );
	bSizer266121->Add( UseFSCWeightingNoButton, 0, wxALL, 5 );


	fgSizer1->Add( bSizer266121, 1, wxEXPAND, 5 );

	SSNRScaleFactorText = new wxStaticText( this, wxID_ANY, wxT("\tSSNR Scale Factor :"), wxDefaultPosition, wxDefaultSize, 0 );
	SSNRScaleFactorText->Wrap( -1 );
	fgSizer1->Add( SSNRScaleFactorText, 0, wxALIGN_CENTER_VERTICAL|wxALL, 5 );

	SSNRScaleFactorTextCtrl = new NumericTextCtrl( this, wxID_ANY, wxT("1.00"), wxDefaultPosition, wxDefaultSize, 0 );
	fgSizer1->Add( SSNRScaleFactorTextCtrl, 0, wxALL, 5 );


	bSizer491->Add( fgSizer1, 0, wxEXPAND, 5 );

	m_staticline136 = new wxStaticLine( this, wxID_ANY, wxDefaultPosition, wxDefaultSize, wxLI_HORIZONTAL|wxLI_VERTICAL );
	bSizer491->Add( m_staticline136, 0, wxEXPAND | wxALL, 5 );

	wxFlexGridSizer* fgSizer11;
	fgSizer11 = new wxFlexGridSizer( 0, 2, 0, 0 );
	fgSizer11->SetFlexibleDirection( wxBOTH );
	fgSizer11->SetNonFlexibleGrowMode( wxFLEX_GROWMODE_SPECIFIED );

	m_staticText202 = new wxStaticText( this, wxID_ANY, wxT("Masking"), wxDefaultPosition, wxDefaultSize, 0 );
	m_staticText202->Wrap( -1 );
	m_staticText202->SetFont( wxFont( wxNORMAL_FONT->GetPointSize(), wxFONTFAMILY_DEFAULT, wxFONTSTYLE_NORMAL, wxFONTWEIGHT_BOLD, true, wxEmptyString ) );

	fgSizer11->Add( m_staticText202, 0, wxALL, 5 );


	fgSizer11->Add( 0, 0, 1, wxEXPAND, 5 );

	InnerMaskRadiusStaticText = new wxStaticText( this, wxID_ANY, wxT("Inner Mask Radius (Å) :"), wxDefaultPosition, wxDefaultSize, 0 );
	InnerMaskRadiusStaticText->Wrap( -1 );
	fgSizer11->Add( InnerMaskRadiusStaticText, 0, wxALIGN_CENTER_VERTICAL|wxALIGN_RIGHT|wxALL, 5 );

	InnerMaskRadiusTextCtrl = new NumericTextCtrl( this, wxID_ANY, wxT("0.00"), wxDefaultPosition, wxDefaultSize, 0 );
	fgSizer11->Add( InnerMaskRadiusTextCtrl, 0, wxALL|wxEXPAND, 5 );

	OuterMaskRadiusStaticText = new wxStaticText( this, wxID_ANY, wxT("Outer Mask Radius (Å) :"), wxDefaultPosition, wxDefaultSize, 0 );
	OuterMaskRadiusStaticText->Wrap( -1 );
	fgSizer11->Add( OuterMaskRadiusStaticText, 0, wxALIGN_CENTER_VERTICAL|wxALIGN_RIGHT|wxALL, 5 );

	OuterMaskRadiusTextCtrl = new NumericTextCtrl( this, wxID_ANY, wxT("100.00"), wxDefaultPosition, wxDefaultSize, wxTE_PROCESS_ENTER );
	fgSizer11->Add( OuterMaskRadiusTextCtrl, 0, wxALL|wxEXPAND, 5 );

	UseAutoMaskingStaticText = new wxStaticText( this, wxID_ANY, wxT("Use AutoMasking? :"), wxDefaultPosition, wxDefaultSize, 0 );
	UseAutoMaskingStaticText->Wrap( -1 );
	fgSizer11->Add( UseAutoMaskingStaticText, 0, wxALIGN_CENTER_VERTICAL|wxALIGN_RIGHT|wxALL, 5 );

	wxBoxSizer* bSizer26612111;
	bSizer26612111 = new wxBoxSizer( wxHORIZONTAL );

	UseAutoMaskingYesButton = new wxRadioButton( this, wxID_ANY, wxT("Yes"), wxDefaultPosition, wxDefaultSize, wxRB_GROUP );
	bSizer26612111->Add( UseAutoMaskingYesButton, 0, wxALL, 5 );

	UseAutoMaskingNoButton = new wxRadioButton( this, wxID_ANY, wxT("No"), wxDefaultPosition, wxDefaultSize, 0 );
	bSizer26612111->Add( UseAutoMaskingNoButton, 0, wxALL, 5 );


	fgSizer11->Add( bSizer26612111, 1, wxEXPAND, 5 );

	m_staticText671 = new wxStaticText( this, wxID_ANY, wxT("Additional"), wxDefaultPosition, wxDefaultSize, 0 );
	m_staticText671->Wrap( -1 );
	m_staticText671->SetFont( wxFont( 10, wxFONTFAMILY_SWISS, wxFONTSTYLE_NORMAL, wxFONTWEIGHT_BOLD, true, wxT("Sans") ) );

	fgSizer11->Add( m_staticText671, 0, wxALL, 5 );


	fgSizer11->Add( 0, 0, 1, wxEXPAND, 5 );

	m_staticText642 = new wxStaticText( this, wxID_ANY, wxT("Invert Handedness? :"), wxDefaultPosition, wxDefaultSize, 0 );
	m_staticText642->Wrap( -1 );
	fgSizer11->Add( m_staticText642, 0, wxALIGN_CENTER_VERTICAL|wxALIGN_RIGHT|wxALL, 5 );

	wxBoxSizer* bSizer266122;
	bSizer266122 = new wxBoxSizer( wxHORIZONTAL );

	InvertHandednessYesButton = new wxRadioButton( this, wxID_ANY, wxT("Yes"), wxDefaultPosition, wxDefaultSize, wxRB_GROUP );
	bSizer266122->Add( InvertHandednessYesButton, 0, wxALL, 5 );

	InvertHandednessNoButton = new wxRadioButton( this, wxID_ANY, wxT("No"), wxDefaultPosition, wxDefaultSize, 0 );
	bSizer266122->Add( InvertHandednessNoButton, 0, wxALL, 5 );


	fgSizer11->Add( bSizer266122, 1, wxEXPAND, 5 );

	m_staticText6721 = new wxStaticText( this, wxID_ANY, wxT("Correct Gridding Error? :"), wxDefaultPosition, wxDefaultSize, 0 );
	m_staticText6721->Wrap( -1 );
	fgSizer11->Add( m_staticText6721, 0, wxALIGN_CENTER_VERTICAL|wxALIGN_RIGHT|wxALL, 5 );

	wxBoxSizer* bSizer2661211;
	bSizer2661211 = new wxBoxSizer( wxHORIZONTAL );

	CorrectGriddingYesButton = new wxRadioButton( this, wxID_ANY, wxT("Yes"), wxDefaultPosition, wxDefaultSize, wxRB_GROUP );
	bSizer2661211->Add( CorrectGriddingYesButton, 0, wxALL, 5 );

	CorrectGriddingNoButton = new wxRadioButton( this, wxID_ANY, wxT("No"), wxDefaultPosition, wxDefaultSize, 0 );
	bSizer2661211->Add( CorrectGriddingNoButton, 0, wxALL, 5 );


	fgSizer11->Add( bSizer2661211, 1, wxEXPAND, 5 );


	bSizer491->Add( fgSizer11, 0, wxEXPAND, 5 );


	InputSizer->Add( bSizer491, 0, wxEXPAND, 5 );

	m_staticline138 = new wxStaticLine( this, wxID_ANY, wxDefaultPosition, wxDefaultSize, wxLI_HORIZONTAL );
	InputSizer->Add( m_staticline138, 0, wxEXPAND | wxALL, 5 );


	InputSizer->Add( 0, 0, 0, wxEXPAND, 5 );


	InputSizer->Add( 0, 0, 1, wxEXPAND, 5 );

	m_staticText699 = new wxStaticText( this, wxID_ANY, wxT("Plot of Relative Log Amplitudes"), wxDefaultPosition, wxDefaultSize, 0 );
	m_staticText699->Wrap( -1 );
	m_staticText699->SetFont( wxFont( 10, wxFONTFAMILY_SWISS, wxFONTSTYLE_NORMAL, wxFONTWEIGHT_BOLD, true, wxT("Sans") ) );

	InputSizer->Add( m_staticText699, 0, wxALL, 5 );

	m_staticline137 = new wxStaticLine( this, wxID_ANY, wxDefaultPosition, wxDefaultSize, wxLI_HORIZONTAL );
	InputSizer->Add( m_staticline137, 0, wxEXPAND | wxALL, 5 );

	GuinierPlot = new PlotCurvePanel( this, wxID_ANY, wxDefaultPosition, wxDefaultSize, wxTAB_TRAVERSAL );
	GuinierPlot->SetMinSize( wxSize( -1,300 ) );
	GuinierPlot->SetMaxSize( wxSize( -1,400 ) );

	InputSizer->Add( GuinierPlot, 100, wxALIGN_BOTTOM|wxALL|wxEXPAND, 5 );


	bSizer488->Add( InputSizer, 0, wxEXPAND, 5 );

	wxBoxSizer* bSizer487;
	bSizer487 = new wxBoxSizer( wxHORIZONTAL );

	m_staticline135 = new wxStaticLine( this, wxID_ANY, wxDefaultPosition, wxDefaultSize, wxLI_HORIZONTAL|wxLI_VERTICAL );
	bSizer487->Add( m_staticline135, 0, wxEXPAND | wxALL, 5 );

	ResultDisplayPanel = new DisplayPanel( this, wxID_ANY, wxDefaultPosition, wxDefaultSize, wxTAB_TRAVERSAL );
	ResultDisplayPanel->Hide();

	bSizer487->Add( ResultDisplayPanel, 1, wxEXPAND | wxALL, 5 );

	InfoPanel = new wxPanel( this, wxID_ANY, wxDefaultPosition, wxDefaultSize, wxTAB_TRAVERSAL );
	wxBoxSizer* bSizer61;
	bSizer61 = new wxBoxSizer( wxVERTICAL );

	InfoText = new wxRichTextCtrl( InfoPanel, wxID_ANY, wxEmptyString, wxDefaultPosition, wxDefaultSize, wxTE_READONLY|wxHSCROLL|wxVSCROLL );
	bSizer61->Add( InfoText, 1, wxEXPAND | wxALL, 5 );


	InfoPanel->SetSizer( bSizer61 );
	InfoPanel->Layout();
	bSizer61->Fit( InfoPanel );
	bSizer487->Add( InfoPanel, 1, wxEXPAND | wxALL, 5 );


	bSizer488->Add( bSizer487, 1, wxEXPAND, 5 );


	bSizer479->Add( bSizer488, 1, wxEXPAND, 5 );


	bSizer589->Add( bSizer479, 1, wxEXPAND, 5 );

	m_staticline130 = new wxStaticLine( this, wxID_ANY, wxDefaultPosition, wxDefaultSize, wxLI_HORIZONTAL );
	bSizer589->Add( m_staticline130, 0, wxEXPAND | wxALL, 5 );

	wxBoxSizer* bSizer478;
	bSizer478 = new wxBoxSizer( wxVERTICAL );

	wxBoxSizer* bSizer477;
	bSizer477 = new wxBoxSizer( wxHORIZONTAL );

	RunJobButton = new wxButton( this, wxID_ANY, wxT("Run"), wxDefaultPosition, wxDefaultSize, 0 );
	bSizer477->Add( RunJobButton, 0, wxALL, 5 );

	ProgressGuage = new wxGauge( this, wxID_ANY, 100, wxDefaultPosition, wxDefaultSize, wxGA_HORIZONTAL );
	ProgressGuage->SetValue( 0 );
	bSizer477->Add( ProgressGuage, 1, wxALIGN_CENTER_VERTICAL|wxALL, 5 );

	ImportResultButton = new wxButton( this, wxID_ANY, wxT("Import Current Result"), wxDefaultPosition, wxDefaultSize, 0 );
	ImportResultButton->Hide();

	bSizer477->Add( ImportResultButton, 0, wxALL, 5 );

	SaveResultButton = new wxButton( this, wxID_ANY, wxT("Save Current Result"), wxDefaultPosition, wxDefaultSize, 0 );
	bSizer477->Add( SaveResultButton, 0, wxALL, 5 );


	bSizer478->Add( bSizer477, 0, wxEXPAND, 5 );


	bSizer589->Add( bSizer478, 0, wxEXPAND, 5 );


	this->SetSizer( bSizer589 );
	this->Layout();

	// Connect Events
	this->Connect( wxEVT_UPDATE_UI, wxUpdateUIEventHandler( Sharpen3DPanelParent::OnUpdateUI ) );
	UseMaskCheckBox->Connect( wxEVT_COMMAND_CHECKBOX_CLICKED, wxCommandEventHandler( Sharpen3DPanelParent::OnUseMaskCheckBox ), NULL, this );
	UseFSCWeightingYesButton->Connect( wxEVT_COMMAND_RADIOBUTTON_SELECTED, wxCommandEventHandler( Sharpen3DPanelParent::OnAutoMaskButton ), NULL, this );
	UseFSCWeightingNoButton->Connect( wxEVT_COMMAND_RADIOBUTTON_SELECTED, wxCommandEventHandler( Sharpen3DPanelParent::OnAutoMaskButton ), NULL, this );
	UseAutoMaskingYesButton->Connect( wxEVT_COMMAND_RADIOBUTTON_SELECTED, wxCommandEventHandler( Sharpen3DPanelParent::OnAutoMaskButton ), NULL, this );
	UseAutoMaskingNoButton->Connect( wxEVT_COMMAND_RADIOBUTTON_SELECTED, wxCommandEventHandler( Sharpen3DPanelParent::OnAutoMaskButton ), NULL, this );
	InvertHandednessYesButton->Connect( wxEVT_COMMAND_RADIOBUTTON_SELECTED, wxCommandEventHandler( Sharpen3DPanelParent::OnAutoMaskButton ), NULL, this );
	InvertHandednessNoButton->Connect( wxEVT_COMMAND_RADIOBUTTON_SELECTED, wxCommandEventHandler( Sharpen3DPanelParent::OnAutoMaskButton ), NULL, this );
	CorrectGriddingYesButton->Connect( wxEVT_COMMAND_RADIOBUTTON_SELECTED, wxCommandEventHandler( Sharpen3DPanelParent::OnAutoMaskButton ), NULL, this );
	CorrectGriddingNoButton->Connect( wxEVT_COMMAND_RADIOBUTTON_SELECTED, wxCommandEventHandler( Sharpen3DPanelParent::OnAutoMaskButton ), NULL, this );
	InfoText->Connect( wxEVT_COMMAND_TEXT_URL, wxTextUrlEventHandler( Sharpen3DPanelParent::OnInfoURL ), NULL, this );
	RunJobButton->Connect( wxEVT_COMMAND_BUTTON_CLICKED, wxCommandEventHandler( Sharpen3DPanelParent::OnRunButtonClick ), NULL, this );
	ImportResultButton->Connect( wxEVT_COMMAND_BUTTON_CLICKED, wxCommandEventHandler( Sharpen3DPanelParent::OnImportResultClick ), NULL, this );
	SaveResultButton->Connect( wxEVT_COMMAND_BUTTON_CLICKED, wxCommandEventHandler( Sharpen3DPanelParent::OnSaveResultClick ), NULL, this );
}

Sharpen3DPanelParent::~Sharpen3DPanelParent()
{
	// Disconnect Events
	this->Disconnect( wxEVT_UPDATE_UI, wxUpdateUIEventHandler( Sharpen3DPanelParent::OnUpdateUI ) );
	UseMaskCheckBox->Disconnect( wxEVT_COMMAND_CHECKBOX_CLICKED, wxCommandEventHandler( Sharpen3DPanelParent::OnUseMaskCheckBox ), NULL, this );
	UseFSCWeightingYesButton->Disconnect( wxEVT_COMMAND_RADIOBUTTON_SELECTED, wxCommandEventHandler( Sharpen3DPanelParent::OnAutoMaskButton ), NULL, this );
	UseFSCWeightingNoButton->Disconnect( wxEVT_COMMAND_RADIOBUTTON_SELECTED, wxCommandEventHandler( Sharpen3DPanelParent::OnAutoMaskButton ), NULL, this );
	UseAutoMaskingYesButton->Disconnect( wxEVT_COMMAND_RADIOBUTTON_SELECTED, wxCommandEventHandler( Sharpen3DPanelParent::OnAutoMaskButton ), NULL, this );
	UseAutoMaskingNoButton->Disconnect( wxEVT_COMMAND_RADIOBUTTON_SELECTED, wxCommandEventHandler( Sharpen3DPanelParent::OnAutoMaskButton ), NULL, this );
	InvertHandednessYesButton->Disconnect( wxEVT_COMMAND_RADIOBUTTON_SELECTED, wxCommandEventHandler( Sharpen3DPanelParent::OnAutoMaskButton ), NULL, this );
	InvertHandednessNoButton->Disconnect( wxEVT_COMMAND_RADIOBUTTON_SELECTED, wxCommandEventHandler( Sharpen3DPanelParent::OnAutoMaskButton ), NULL, this );
	CorrectGriddingYesButton->Disconnect( wxEVT_COMMAND_RADIOBUTTON_SELECTED, wxCommandEventHandler( Sharpen3DPanelParent::OnAutoMaskButton ), NULL, this );
	CorrectGriddingNoButton->Disconnect( wxEVT_COMMAND_RADIOBUTTON_SELECTED, wxCommandEventHandler( Sharpen3DPanelParent::OnAutoMaskButton ), NULL, this );
	InfoText->Disconnect( wxEVT_COMMAND_TEXT_URL, wxTextUrlEventHandler( Sharpen3DPanelParent::OnInfoURL ), NULL, this );
	RunJobButton->Disconnect( wxEVT_COMMAND_BUTTON_CLICKED, wxCommandEventHandler( Sharpen3DPanelParent::OnRunButtonClick ), NULL, this );
	ImportResultButton->Disconnect( wxEVT_COMMAND_BUTTON_CLICKED, wxCommandEventHandler( Sharpen3DPanelParent::OnImportResultClick ), NULL, this );
	SaveResultButton->Disconnect( wxEVT_COMMAND_BUTTON_CLICKED, wxCommandEventHandler( Sharpen3DPanelParent::OnSaveResultClick ), NULL, this );

}

Generate3DPanelParent::Generate3DPanelParent( wxWindow* parent, wxWindowID id, const wxPoint& pos, const wxSize& size, long style, const wxString& name ) : JobPanel( parent, id, pos, size, style, name )
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


	bSizer200->Add( fgSizer15, 0, wxALIGN_CENTER, 5 );


	bSizer200->Add( 0, 0, 1, wxEXPAND, 5 );

	wxBoxSizer* bSizer357;
	bSizer357 = new wxBoxSizer( wxVERTICAL );

	wxBoxSizer* bSizer358;
	bSizer358 = new wxBoxSizer( wxHORIZONTAL );

	ExpertToggleButton = new wxToggleButton( InputParamsPanel, wxID_ANY, wxT("Show Expert Options"), wxDefaultPosition, wxDefaultSize, 0 );
	bSizer358->Add( ExpertToggleButton, 0, wxALIGN_BOTTOM|wxALIGN_CENTER|wxALL, 5 );


	bSizer357->Add( bSizer358, 0, wxALIGN_CENTER, 5 );


	bSizer200->Add( bSizer357, 0, wxALIGN_BOTTOM, 5 );


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

	m_staticText329 = new wxStaticText( ExpertPanel, wxID_ANY, wxT("Reconstruction"), wxDefaultPosition, wxDefaultSize, 0 );
	m_staticText329->Wrap( -1 );
	m_staticText329->SetFont( wxFont( 10, wxFONTFAMILY_SWISS, wxFONTSTYLE_NORMAL, wxFONTWEIGHT_BOLD, true, wxT("Sans") ) );

	bSizer258->Add( m_staticText329, 0, wxALIGN_BOTTOM|wxALL, 5 );


	bSizer258->Add( 0, 0, 1, wxEXPAND, 5 );

	ResetAllDefaultsButton = new wxButton( ExpertPanel, wxID_ANY, wxT("Reset All Defaults"), wxDefaultPosition, wxDefaultSize, 0 );
	bSizer258->Add( ResetAllDefaultsButton, 0, wxALL, 5 );


	bSizer258->Add( 5, 0, 0, wxEXPAND, 5 );


	InputSizer->Add( bSizer258, 0, wxEXPAND, 5 );


	InputSizer->Add( 0, 5, 0, wxEXPAND, 5 );

	wxFlexGridSizer* fgSizer1;
	fgSizer1 = new wxFlexGridSizer( 0, 2, 0, 0 );
	fgSizer1->SetFlexibleDirection( wxBOTH );
	fgSizer1->SetNonFlexibleGrowMode( wxFLEX_GROWMODE_SPECIFIED );

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

	m_staticText628 = new wxStaticText( ExpertPanel, wxID_ANY, wxT("Also Save Half-Maps?"), wxDefaultPosition, wxDefaultSize, 0 );
	m_staticText628->Wrap( -1 );
	fgSizer1->Add( m_staticText628, 0, wxALIGN_CENTER_VERTICAL|wxALL, 5 );

	wxBoxSizer* bSizer26611;
	bSizer26611 = new wxBoxSizer( wxHORIZONTAL );

	SaveHalfMapsYesButton = new wxRadioButton( ExpertPanel, wxID_ANY, wxT("Yes"), wxDefaultPosition, wxDefaultSize, wxRB_GROUP );
	bSizer26611->Add( SaveHalfMapsYesButton, 0, wxALL, 5 );

	SaveHalfMapsNoButton = new wxRadioButton( ExpertPanel, wxID_ANY, wxT("No"), wxDefaultPosition, wxDefaultSize, 0 );
	bSizer26611->Add( SaveHalfMapsNoButton, 0, wxALL, 5 );


	fgSizer1->Add( bSizer26611, 0, 0, 5 );

	m_staticText631 = new wxStaticText( ExpertPanel, wxID_ANY, wxT("Overwrite Statistics?"), wxDefaultPosition, wxDefaultSize, 0 );
	m_staticText631->Wrap( -1 );
	fgSizer1->Add( m_staticText631, 0, wxALIGN_CENTER_VERTICAL|wxALL, 5 );

	wxBoxSizer* bSizer26612;
	bSizer26612 = new wxBoxSizer( wxHORIZONTAL );

	OverwriteStatisticsYesButton = new wxRadioButton( ExpertPanel, wxID_ANY, wxT("Yes"), wxDefaultPosition, wxDefaultSize, wxRB_GROUP );
	bSizer26612->Add( OverwriteStatisticsYesButton, 0, wxALL, 5 );

	OverwriteStatisticsNoButton = new wxRadioButton( ExpertPanel, wxID_ANY, wxT("No"), wxDefaultPosition, wxDefaultSize, 0 );
	bSizer26612->Add( OverwriteStatisticsNoButton, 0, wxALL, 5 );


	fgSizer1->Add( bSizer26612, 1, wxEXPAND, 5 );

	m_staticText879 = new wxStaticText( ExpertPanel, wxID_ANY, wxT("Apply Ewald Sphere correction?"), wxDefaultPosition, wxDefaultSize, 0 );
	m_staticText879->Wrap( -1 );
	fgSizer1->Add( m_staticText879, 0, wxALIGN_CENTER_VERTICAL|wxALL, 5 );

	wxBoxSizer* bSizer266121;
	bSizer266121 = new wxBoxSizer( wxHORIZONTAL );

	ApplyEwaldSphereCorrectionYesButton = new wxRadioButton( ExpertPanel, wxID_ANY, wxT("Yes"), wxDefaultPosition, wxDefaultSize, wxRB_GROUP );
	bSizer266121->Add( ApplyEwaldSphereCorrectionYesButton, 0, wxALL, 5 );

	ApplyEwaldSphereCorrectionNoButton = new wxRadioButton( ExpertPanel, wxID_ANY, wxT("No"), wxDefaultPosition, wxDefaultSize, 0 );
	bSizer266121->Add( ApplyEwaldSphereCorrectionNoButton, 0, wxALL, 5 );


	fgSizer1->Add( bSizer266121, 1, wxEXPAND, 5 );

	ApplyInverseHandLabelText = new wxStaticText( ExpertPanel, wxID_ANY, wxT("\tApply For Inverse Hand?"), wxDefaultPosition, wxDefaultSize, 0 );
	ApplyInverseHandLabelText->Wrap( -1 );
	fgSizer1->Add( ApplyInverseHandLabelText, 0, wxALIGN_CENTER_VERTICAL|wxALL, 5 );

	wxBoxSizer* bSizer2661211;
	bSizer2661211 = new wxBoxSizer( wxHORIZONTAL );

	ApplyEwaldInverseHandYesButton = new wxRadioButton( ExpertPanel, wxID_ANY, wxT("Yes"), wxDefaultPosition, wxDefaultSize, wxRB_GROUP );
	bSizer2661211->Add( ApplyEwaldInverseHandYesButton, 0, wxALL, 5 );

	ApplyEwaldInverseHandNoButton = new wxRadioButton( ExpertPanel, wxID_ANY, wxT("No"), wxDefaultPosition, wxDefaultSize, 0 );
	bSizer2661211->Add( ApplyEwaldInverseHandNoButton, 0, wxALL, 5 );


	fgSizer1->Add( bSizer2661211, 1, wxEXPAND, 5 );


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

	ShowRefinementResultsPanel = new DisplayRefinementResultsPanel( this, wxID_ANY, wxDefaultPosition, wxDefaultSize, wxTAB_TRAVERSAL );
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

	wxBoxSizer* bSizer268;
	bSizer268 = new wxBoxSizer( wxHORIZONTAL );

	RunProfileText1 = new wxStaticText( StartPanel, wxID_ANY, wxT("Reconstruction Run Profile :"), wxDefaultPosition, wxDefaultSize, 0 );
	RunProfileText1->Wrap( -1 );
	bSizer268->Add( RunProfileText1, 20, wxALIGN_CENTER_VERTICAL|wxALL, 5 );

	ReconstructionRunProfileComboBox = new MemoryComboBox( StartPanel, wxID_ANY, wxEmptyString, wxDefaultPosition, wxDefaultSize, 0, NULL, wxCB_READONLY );
	bSizer268->Add( ReconstructionRunProfileComboBox, 50, wxALIGN_CENTER_VERTICAL|wxALL|wxEXPAND, 5 );


	bSizer58->Add( bSizer268, 50, wxALIGN_CENTER_VERTICAL, 5 );

	wxBoxSizer* bSizer60;
	bSizer60 = new wxBoxSizer( wxVERTICAL );


	bSizer60->Add( 0, 0, 1, wxEXPAND, 5 );

	StartReconstructionButton = new wxButton( StartPanel, wxID_ANY, wxT("Start Reconstruction"), wxDefaultPosition, wxDefaultSize, 0 );
	bSizer60->Add( StartReconstructionButton, 0, wxALL, 5 );


	bSizer58->Add( bSizer60, 50, wxEXPAND, 5 );


	StartPanel->SetSizer( bSizer58 );
	StartPanel->Layout();
	bSizer58->Fit( StartPanel );
	bSizer48->Add( StartPanel, 1, wxEXPAND | wxALL, 5 );


	bSizer43->Add( bSizer48, 0, wxEXPAND, 5 );


	this->SetSizer( bSizer43 );
	this->Layout();

	// Connect Events
	this->Connect( wxEVT_UPDATE_UI, wxUpdateUIEventHandler( Generate3DPanelParent::OnUpdateUI ) );
	ExpertToggleButton->Connect( wxEVT_COMMAND_TOGGLEBUTTON_CLICKED, wxCommandEventHandler( Generate3DPanelParent::OnExpertOptionsToggle ), NULL, this );
	ResetAllDefaultsButton->Connect( wxEVT_COMMAND_BUTTON_CLICKED, wxCommandEventHandler( Generate3DPanelParent::ResetAllDefaultsClick ), NULL, this );
	InfoText->Connect( wxEVT_COMMAND_TEXT_URL, wxTextUrlEventHandler( Generate3DPanelParent::OnInfoURL ), NULL, this );
	FinishButton->Connect( wxEVT_COMMAND_BUTTON_CLICKED, wxCommandEventHandler( Generate3DPanelParent::FinishButtonClick ), NULL, this );
	CancelAlignmentButton->Connect( wxEVT_COMMAND_BUTTON_CLICKED, wxCommandEventHandler( Generate3DPanelParent::TerminateButtonClick ), NULL, this );
	StartReconstructionButton->Connect( wxEVT_COMMAND_BUTTON_CLICKED, wxCommandEventHandler( Generate3DPanelParent::StartReconstructionClick ), NULL, this );
}

Generate3DPanelParent::~Generate3DPanelParent()
{
	// Disconnect Events
	this->Disconnect( wxEVT_UPDATE_UI, wxUpdateUIEventHandler( Generate3DPanelParent::OnUpdateUI ) );
	ExpertToggleButton->Disconnect( wxEVT_COMMAND_TOGGLEBUTTON_CLICKED, wxCommandEventHandler( Generate3DPanelParent::OnExpertOptionsToggle ), NULL, this );
	ResetAllDefaultsButton->Disconnect( wxEVT_COMMAND_BUTTON_CLICKED, wxCommandEventHandler( Generate3DPanelParent::ResetAllDefaultsClick ), NULL, this );
	InfoText->Disconnect( wxEVT_COMMAND_TEXT_URL, wxTextUrlEventHandler( Generate3DPanelParent::OnInfoURL ), NULL, this );
	FinishButton->Disconnect( wxEVT_COMMAND_BUTTON_CLICKED, wxCommandEventHandler( Generate3DPanelParent::FinishButtonClick ), NULL, this );
	CancelAlignmentButton->Disconnect( wxEVT_COMMAND_BUTTON_CLICKED, wxCommandEventHandler( Generate3DPanelParent::TerminateButtonClick ), NULL, this );
	StartReconstructionButton->Disconnect( wxEVT_COMMAND_BUTTON_CLICKED, wxCommandEventHandler( Generate3DPanelParent::StartReconstructionClick ), NULL, this );

}
