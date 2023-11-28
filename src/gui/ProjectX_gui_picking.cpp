///////////////////////////////////////////////////////////////////////////
// C++ code generated with wxFormBuilder (version 3.10.0-4761b0c)
// http://www.wxformbuilder.org/
//
// PLEASE DO *NOT* EDIT THIS FILE!
///////////////////////////////////////////////////////////////////////////

#include "../core/gui_core_headers.h"
#include "AssetPickerComboPanel.h"
#include "PickingResultsDisplayPanel.h"
#include "ResultsDataViewListCtrl.h"
#include "my_controls.h"

#include "ProjectX_gui_picking.h"

///////////////////////////////////////////////////////////////////////////

PickingResultsDisplayPanelParent::PickingResultsDisplayPanelParent( wxWindow* parent, wxWindowID id, const wxPoint& pos, const wxSize& size, long style, const wxString& name ) : wxPanel( parent, id, pos, size, style, name )
{
	wxBoxSizer* bSizer92;
	bSizer92 = new wxBoxSizer( wxHORIZONTAL );

	PickingResultsImagePanel = new PickingBitmapPanel( this, wxID_ANY, wxDefaultPosition, wxDefaultSize, wxTAB_TRAVERSAL );
	bSizer92->Add( PickingResultsImagePanel, 5, wxEXPAND | wxALL, 5 );

	wxBoxSizer* bSizer93;
	bSizer93 = new wxBoxSizer( wxHORIZONTAL );

	wxBoxSizer* bSizer94;
	bSizer94 = new wxBoxSizer( wxVERTICAL );

	CirclesAroundParticlesCheckBox = new wxCheckBox( this, wxID_ANY, wxT("Circles"), wxDefaultPosition, wxDefaultSize, 0 );
	CirclesAroundParticlesCheckBox->SetValue(true);
	CirclesAroundParticlesCheckBox->SetToolTip( wxT("Draw circles around picked particles") );

	bSizer94->Add( CirclesAroundParticlesCheckBox, 0, wxALIGN_CENTER_VERTICAL|wxALL, 5 );

	ScaleBarCheckBox = new wxCheckBox( this, wxID_ANY, wxT("Scale"), wxDefaultPosition, wxDefaultSize, 0 );
	ScaleBarCheckBox->SetValue(true);
	ScaleBarCheckBox->SetToolTip( wxT("Display a scale bar") );

	bSizer94->Add( ScaleBarCheckBox, 0, wxALIGN_CENTER_VERTICAL|wxALL, 5 );

	HighPassFilterCheckBox = new wxCheckBox( this, wxID_ANY, wxT("High-pass"), wxDefaultPosition, wxDefaultSize, 0 );
	HighPassFilterCheckBox->SetValue(true);
	HighPassFilterCheckBox->SetToolTip( wxT("Filter the image to remove density ramps") );

	bSizer94->Add( HighPassFilterCheckBox, 0, wxALIGN_CENTER_VERTICAL|wxALL, 5 );

	wxBoxSizer* bSizer487;
	bSizer487 = new wxBoxSizer( wxHORIZONTAL );

	LowPassFilterCheckBox = new wxCheckBox( this, wxID_ANY, wxT("Low-pass"), wxDefaultPosition, wxDefaultSize, 0 );
	LowPassFilterCheckBox->SetToolTip( wxT("Filter the image to remove density ramps") );

	bSizer487->Add( LowPassFilterCheckBox, 0, wxALIGN_CENTER_VERTICAL|wxALL, 5 );

	LowResFilterTextCtrl = new NumericTextCtrl( this, wxID_ANY, wxT("40"), wxDefaultPosition, wxSize( 40,-1 ), wxTE_PROCESS_ENTER );
	LowResFilterTextCtrl->Enable( false );

	bSizer487->Add( LowResFilterTextCtrl, 0, wxALIGN_CENTER_VERTICAL|wxALL, 5 );

	LowAngstromStatic = new wxStaticText( this, wxID_ANY, wxT("Å"), wxDefaultPosition, wxDefaultSize, 0 );
	LowAngstromStatic->Wrap( -1 );
	LowAngstromStatic->Enable( false );

	bSizer487->Add( LowAngstromStatic, 0, wxALIGN_CENTER_VERTICAL|wxALL, 5 );


	bSizer94->Add( bSizer487, 0, wxEXPAND, 5 );

	WienerFilterCheckBox = new wxCheckBox( this, wxID_ANY, wxT("Wiener filter"), wxDefaultPosition, wxDefaultSize, 0 );
	WienerFilterCheckBox->SetValue(true);
	WienerFilterCheckBox->SetToolTip( wxT("Filter the image to remove density ramps") );

	bSizer94->Add( WienerFilterCheckBox, 0, wxALIGN_CENTER_VERTICAL|wxALL, 5 );

	m_staticline831 = new wxStaticLine( this, wxID_ANY, wxDefaultPosition, wxDefaultSize, wxLI_HORIZONTAL );
	bSizer94->Add( m_staticline831, 0, wxEXPAND | wxALL, 5 );

	ImageIDStaticText = new wxStaticText( this, wxID_ANY, wxT("Image ID: -1"), wxDefaultPosition, wxDefaultSize, 0 );
	ImageIDStaticText->Wrap( -1 );
	bSizer94->Add( ImageIDStaticText, 0, wxALL, 5 );

	DefocusStaticText = new wxStaticText( this, wxID_ANY, wxT("Defocus: 0.0 μm"), wxDefaultPosition, wxDefaultSize, 0 );
	DefocusStaticText->Wrap( -1 );
	bSizer94->Add( DefocusStaticText, 0, wxALL, 5 );

	IcinessStaticText = new wxStaticText( this, wxID_ANY, wxT("Iciness: 0.0"), wxDefaultPosition, wxDefaultSize, 0 );
	IcinessStaticText->Wrap( -1 );
	bSizer94->Add( IcinessStaticText, 0, wxALL, 5 );

	NumberOfPicksStaticText = new wxStaticText( this, wxID_ANY, wxT("0 picked coordinates"), wxDefaultPosition, wxDefaultSize, 0 );
	NumberOfPicksStaticText->Wrap( -1 );
	bSizer94->Add( NumberOfPicksStaticText, 0, wxALL, 5 );

	m_staticline8311 = new wxStaticLine( this, wxID_ANY, wxDefaultPosition, wxDefaultSize, wxLI_HORIZONTAL );
	bSizer94->Add( m_staticline8311, 0, wxEXPAND | wxALL, 5 );

	wxBoxSizer* bSizer488;
	bSizer488 = new wxBoxSizer( wxHORIZONTAL );

	UndoButton = new wxButton( this, wxID_ANY, wxT("Undo"), wxDefaultPosition, wxDefaultSize, 0 );
	UndoButton->Enable( false );
	UndoButton->SetToolTip( wxT("Undo manual change of particle coordinates") );

	bSizer488->Add( UndoButton, 0, wxALIGN_CENTER_VERTICAL|wxALL, 5 );

	RedoButton = new wxButton( this, wxID_ANY, wxT("Redo"), wxDefaultPosition, wxDefaultSize, 0 );
	RedoButton->Enable( false );
	RedoButton->SetToolTip( wxT("Redo manual change of particle coordinates") );

	bSizer488->Add( RedoButton, 0, wxALIGN_CENTER_VERTICAL|wxALL, 5 );


	bSizer94->Add( bSizer488, 0, wxEXPAND, 5 );


	bSizer93->Add( bSizer94, 1, wxALIGN_CENTER|wxEXPAND, 5 );

	m_staticline26 = new wxStaticLine( this, wxID_ANY, wxDefaultPosition, wxDefaultSize, wxLI_HORIZONTAL );
	bSizer93->Add( m_staticline26, 0, wxEXPAND | wxALL, 5 );


	bSizer92->Add( bSizer93, 1, wxEXPAND, 5 );


	this->SetSizer( bSizer92 );
	this->Layout();

	// Connect Events
	CirclesAroundParticlesCheckBox->Connect( wxEVT_COMMAND_CHECKBOX_CLICKED, wxCommandEventHandler( PickingResultsDisplayPanelParent::OnCirclesAroundParticlesCheckBox ), NULL, this );
	ScaleBarCheckBox->Connect( wxEVT_COMMAND_CHECKBOX_CLICKED, wxCommandEventHandler( PickingResultsDisplayPanelParent::OnScaleBarCheckBox ), NULL, this );
	HighPassFilterCheckBox->Connect( wxEVT_COMMAND_CHECKBOX_CLICKED, wxCommandEventHandler( PickingResultsDisplayPanelParent::OnHighPassFilterCheckBox ), NULL, this );
	LowPassFilterCheckBox->Connect( wxEVT_COMMAND_CHECKBOX_CLICKED, wxCommandEventHandler( PickingResultsDisplayPanelParent::OnLowPassFilterCheckBox ), NULL, this );
	LowResFilterTextCtrl->Connect( wxEVT_KILL_FOCUS, wxFocusEventHandler( PickingResultsDisplayPanelParent::OnLowPassKillFocus ), NULL, this );
	LowResFilterTextCtrl->Connect( wxEVT_COMMAND_TEXT_ENTER, wxCommandEventHandler( PickingResultsDisplayPanelParent::OnLowPassEnter ), NULL, this );
	WienerFilterCheckBox->Connect( wxEVT_COMMAND_CHECKBOX_CLICKED, wxCommandEventHandler( PickingResultsDisplayPanelParent::OnWienerFilterCheckBox ), NULL, this );
	UndoButton->Connect( wxEVT_COMMAND_BUTTON_CLICKED, wxCommandEventHandler( PickingResultsDisplayPanelParent::OnUndoButtonClick ), NULL, this );
	RedoButton->Connect( wxEVT_COMMAND_BUTTON_CLICKED, wxCommandEventHandler( PickingResultsDisplayPanelParent::OnRedoButtonClick ), NULL, this );
}

PickingResultsDisplayPanelParent::~PickingResultsDisplayPanelParent()
{
	// Disconnect Events
	CirclesAroundParticlesCheckBox->Disconnect( wxEVT_COMMAND_CHECKBOX_CLICKED, wxCommandEventHandler( PickingResultsDisplayPanelParent::OnCirclesAroundParticlesCheckBox ), NULL, this );
	ScaleBarCheckBox->Disconnect( wxEVT_COMMAND_CHECKBOX_CLICKED, wxCommandEventHandler( PickingResultsDisplayPanelParent::OnScaleBarCheckBox ), NULL, this );
	HighPassFilterCheckBox->Disconnect( wxEVT_COMMAND_CHECKBOX_CLICKED, wxCommandEventHandler( PickingResultsDisplayPanelParent::OnHighPassFilterCheckBox ), NULL, this );
	LowPassFilterCheckBox->Disconnect( wxEVT_COMMAND_CHECKBOX_CLICKED, wxCommandEventHandler( PickingResultsDisplayPanelParent::OnLowPassFilterCheckBox ), NULL, this );
	LowResFilterTextCtrl->Disconnect( wxEVT_KILL_FOCUS, wxFocusEventHandler( PickingResultsDisplayPanelParent::OnLowPassKillFocus ), NULL, this );
	LowResFilterTextCtrl->Disconnect( wxEVT_COMMAND_TEXT_ENTER, wxCommandEventHandler( PickingResultsDisplayPanelParent::OnLowPassEnter ), NULL, this );
	WienerFilterCheckBox->Disconnect( wxEVT_COMMAND_CHECKBOX_CLICKED, wxCommandEventHandler( PickingResultsDisplayPanelParent::OnWienerFilterCheckBox ), NULL, this );
	UndoButton->Disconnect( wxEVT_COMMAND_BUTTON_CLICKED, wxCommandEventHandler( PickingResultsDisplayPanelParent::OnUndoButtonClick ), NULL, this );
	RedoButton->Disconnect( wxEVT_COMMAND_BUTTON_CLICKED, wxCommandEventHandler( PickingResultsDisplayPanelParent::OnRedoButtonClick ), NULL, this );

}

PickingResultsPanel::PickingResultsPanel( wxWindow* parent, wxWindowID id, const wxPoint& pos, const wxSize& size, long style, const wxString& name ) : wxPanel( parent, id, pos, size, style, name )
{
	wxBoxSizer* bSizer63;
	bSizer63 = new wxBoxSizer( wxVERTICAL );

	m_staticline25 = new wxStaticLine( this, wxID_ANY, wxDefaultPosition, wxDefaultSize, wxLI_HORIZONTAL );
	bSizer63->Add( m_staticline25, 0, wxEXPAND | wxALL, 5 );

	m_splitter4 = new wxSplitterWindow( this, wxID_ANY, wxDefaultPosition, wxDefaultSize, wxSP_3D );
	m_splitter4->SetSashGravity( 0.5 );
	m_splitter4->Connect( wxEVT_IDLE, wxIdleEventHandler( PickingResultsPanel::m_splitter4OnIdle ), NULL, this );

	m_panel13 = new wxPanel( m_splitter4, wxID_ANY, wxDefaultPosition, wxDefaultSize, wxTAB_TRAVERSAL );
	wxBoxSizer* bSizer66;
	bSizer66 = new wxBoxSizer( wxVERTICAL );

	wxBoxSizer* bSizer64;
	bSizer64 = new wxBoxSizer( wxHORIZONTAL );

	AllImagesButton = new wxRadioButton( m_panel13, wxID_ANY, wxT("All Images"), wxDefaultPosition, wxDefaultSize, 0 );
	bSizer64->Add( AllImagesButton, 0, wxALIGN_CENTER_VERTICAL|wxALL, 5 );

	ByFilterButton = new wxRadioButton( m_panel13, wxID_ANY, wxT("By Filter"), wxDefaultPosition, wxDefaultSize, 0 );
	ByFilterButton->Enable( false );

	bSizer64->Add( ByFilterButton, 0, wxALIGN_CENTER_VERTICAL|wxALL, 5 );

	FilterButton = new wxButton( m_panel13, wxID_ANY, wxT("Define Filter"), wxDefaultPosition, wxDefaultSize, 0 );
	FilterButton->Enable( false );

	bSizer64->Add( FilterButton, 0, wxALIGN_CENTER_VERTICAL|wxALL, 5 );

	m_staticline85 = new wxStaticLine( m_panel13, wxID_ANY, wxDefaultPosition, wxDefaultSize, wxLI_HORIZONTAL|wxLI_VERTICAL );
	bSizer64->Add( m_staticline85, 0, wxEXPAND | wxALL, 5 );


	bSizer64->Add( 0, 0, 1, wxEXPAND, 5 );

	JobDetailsToggleButton = new wxToggleButton( m_panel13, wxID_ANY, wxT("Show Job Details"), wxDefaultPosition, wxDefaultSize, 0 );
	bSizer64->Add( JobDetailsToggleButton, 0, wxALIGN_CENTER_VERTICAL|wxALIGN_RIGHT|wxALL, 5 );


	bSizer66->Add( bSizer64, 0, wxEXPAND, 5 );

	ResultDataView = new ResultsDataViewListCtrl( m_panel13, wxID_ANY, wxDefaultPosition, wxDefaultSize, wxDV_VERT_RULES );
	bSizer66->Add( ResultDataView, 1, wxALL|wxEXPAND, 5 );

	wxBoxSizer* bSizer68;
	bSizer68 = new wxBoxSizer( wxHORIZONTAL );

	PreviousButton = new wxButton( m_panel13, wxID_ANY, wxT("&Previous"), wxDefaultPosition, wxDefaultSize, 0 );
	bSizer68->Add( PreviousButton, 0, wxALL, 5 );


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

	wxBoxSizer* bSizer101;
	bSizer101 = new wxBoxSizer( wxVERTICAL );

	InfoSizer = new wxFlexGridSizer( 0, 6, 0, 0 );
	InfoSizer->SetFlexibleDirection( wxBOTH );
	InfoSizer->SetNonFlexibleGrowMode( wxFLEX_GROWMODE_SPECIFIED );

	m_staticText72 = new wxStaticText( JobDetailsPanel, wxID_ANY, wxT("Pick ID :"), wxDefaultPosition, wxDefaultSize, 0 );
	m_staticText72->Wrap( -1 );
	m_staticText72->SetFont( wxFont( 10, wxFONTFAMILY_SWISS, wxFONTSTYLE_NORMAL, wxFONTWEIGHT_BOLD, false, wxT("Sans") ) );

	InfoSizer->Add( m_staticText72, 0, wxALIGN_RIGHT|wxALL, 5 );

	PickIDStaticText = new wxStaticText( JobDetailsPanel, wxID_ANY, wxEmptyString, wxDefaultPosition, wxDefaultSize, 0 );
	PickIDStaticText->Wrap( -1 );
	InfoSizer->Add( PickIDStaticText, 0, wxALIGN_LEFT|wxALL, 5 );

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

	m_staticText83 = new wxStaticText( JobDetailsPanel, wxID_ANY, wxT("Algorithm :"), wxDefaultPosition, wxDefaultSize, 0 );
	m_staticText83->Wrap( -1 );
	m_staticText83->SetFont( wxFont( 10, wxFONTFAMILY_SWISS, wxFONTSTYLE_NORMAL, wxFONTWEIGHT_BOLD, false, wxT("Sans") ) );

	InfoSizer->Add( m_staticText83, 0, wxALIGN_RIGHT|wxALL, 5 );

	AlgorithmStaticText = new wxStaticText( JobDetailsPanel, wxID_ANY, wxEmptyString, wxDefaultPosition, wxDefaultSize, 0 );
	AlgorithmStaticText->Wrap( -1 );
	InfoSizer->Add( AlgorithmStaticText, 0, wxALIGN_LEFT|wxALL, 5 );

	m_staticText82 = new wxStaticText( JobDetailsPanel, wxID_ANY, wxT("Manual edit :"), wxDefaultPosition, wxDefaultSize, 0 );
	m_staticText82->Wrap( -1 );
	m_staticText82->SetFont( wxFont( 10, wxFONTFAMILY_SWISS, wxFONTSTYLE_NORMAL, wxFONTWEIGHT_BOLD, false, wxT("Sans") ) );

	InfoSizer->Add( m_staticText82, 0, wxALIGN_RIGHT|wxALL, 5 );

	ManualEditStaticText = new wxStaticText( JobDetailsPanel, wxID_ANY, wxEmptyString, wxDefaultPosition, wxDefaultSize, 0 );
	ManualEditStaticText->Wrap( -1 );
	InfoSizer->Add( ManualEditStaticText, 0, wxALIGN_LEFT|wxALL, 5 );

	m_staticText78 = new wxStaticText( JobDetailsPanel, wxID_ANY, wxT("Threshold :"), wxDefaultPosition, wxDefaultSize, 0 );
	m_staticText78->Wrap( -1 );
	m_staticText78->SetFont( wxFont( 10, wxFONTFAMILY_SWISS, wxFONTSTYLE_NORMAL, wxFONTWEIGHT_BOLD, false, wxT("Sans") ) );

	InfoSizer->Add( m_staticText78, 0, wxALIGN_RIGHT|wxALL, 5 );

	ThresholdStaticText = new wxStaticText( JobDetailsPanel, wxID_ANY, wxEmptyString, wxDefaultPosition, wxDefaultSize, 0 );
	ThresholdStaticText->Wrap( -1 );
	InfoSizer->Add( ThresholdStaticText, 0, wxALIGN_LEFT|wxALL, 5 );

	m_staticText96 = new wxStaticText( JobDetailsPanel, wxID_ANY, wxT("Max. Radius :"), wxDefaultPosition, wxDefaultSize, 0 );
	m_staticText96->Wrap( -1 );
	m_staticText96->SetFont( wxFont( 10, wxFONTFAMILY_SWISS, wxFONTSTYLE_NORMAL, wxFONTWEIGHT_BOLD, false, wxT("Sans") ) );

	InfoSizer->Add( m_staticText96, 0, wxALIGN_RIGHT|wxALL, 5 );

	MaximumRadiusStaticText = new wxStaticText( JobDetailsPanel, wxID_ANY, wxEmptyString, wxDefaultPosition, wxDefaultSize, 0 );
	MaximumRadiusStaticText->Wrap( -1 );
	InfoSizer->Add( MaximumRadiusStaticText, 0, wxALIGN_LEFT|wxALL, 5 );

	m_staticText85 = new wxStaticText( JobDetailsPanel, wxID_ANY, wxT("Charact. Radius :"), wxDefaultPosition, wxDefaultSize, 0 );
	m_staticText85->Wrap( -1 );
	m_staticText85->SetFont( wxFont( 10, wxFONTFAMILY_SWISS, wxFONTSTYLE_NORMAL, wxFONTWEIGHT_BOLD, false, wxT("Sans") ) );

	InfoSizer->Add( m_staticText85, 0, wxALIGN_RIGHT|wxALL, 5 );

	CharacteristicRadiusStaticText = new wxStaticText( JobDetailsPanel, wxID_ANY, wxEmptyString, wxDefaultPosition, wxDefaultSize, 0 );
	CharacteristicRadiusStaticText->Wrap( -1 );
	InfoSizer->Add( CharacteristicRadiusStaticText, 0, wxALIGN_LEFT|wxALL, 5 );

	m_staticText87 = new wxStaticText( JobDetailsPanel, wxID_ANY, wxT("Highest Res. :"), wxDefaultPosition, wxDefaultSize, 0 );
	m_staticText87->Wrap( -1 );
	m_staticText87->SetFont( wxFont( 10, wxFONTFAMILY_SWISS, wxFONTSTYLE_NORMAL, wxFONTWEIGHT_BOLD, false, wxT("Sans") ) );

	InfoSizer->Add( m_staticText87, 0, wxALIGN_RIGHT|wxALL, 5 );

	HighestResStaticText = new wxStaticText( JobDetailsPanel, wxID_ANY, wxEmptyString, wxDefaultPosition, wxDefaultSize, 0 );
	HighestResStaticText->Wrap( -1 );
	InfoSizer->Add( HighestResStaticText, 0, wxALIGN_LEFT|wxALL, 5 );

	m_staticText89 = new wxStaticText( JobDetailsPanel, wxID_ANY, wxT("Min. Edge Dist. :"), wxDefaultPosition, wxDefaultSize, 0 );
	m_staticText89->Wrap( -1 );
	m_staticText89->SetFont( wxFont( 10, wxFONTFAMILY_SWISS, wxFONTSTYLE_NORMAL, wxFONTWEIGHT_BOLD, false, wxT("Sans") ) );

	InfoSizer->Add( m_staticText89, 0, wxALIGN_RIGHT|wxALL, 5 );

	MinEdgeDistStaticText = new wxStaticText( JobDetailsPanel, wxID_ANY, wxEmptyString, wxDefaultPosition, wxDefaultSize, 0 );
	MinEdgeDistStaticText->Wrap( -1 );
	InfoSizer->Add( MinEdgeDistStaticText, 0, wxALIGN_LEFT|wxALL, 5 );

	m_staticText91 = new wxStaticText( JobDetailsPanel, wxID_ANY, wxT("Avoid High Var. :"), wxDefaultPosition, wxDefaultSize, 0 );
	m_staticText91->Wrap( -1 );
	m_staticText91->SetFont( wxFont( 10, wxFONTFAMILY_SWISS, wxFONTSTYLE_NORMAL, wxFONTWEIGHT_BOLD, false, wxT("Sans") ) );

	InfoSizer->Add( m_staticText91, 0, wxALIGN_RIGHT|wxALL, 5 );

	AvoidHighVarStaticText = new wxStaticText( JobDetailsPanel, wxID_ANY, wxEmptyString, wxDefaultPosition, wxDefaultSize, 0 );
	AvoidHighVarStaticText->Wrap( -1 );
	InfoSizer->Add( AvoidHighVarStaticText, 0, wxALIGN_LEFT|wxALL, 5 );

	m_staticText79 = new wxStaticText( JobDetailsPanel, wxID_ANY, wxT("Avoid Hi/Lo Mean :"), wxDefaultPosition, wxDefaultSize, 0 );
	m_staticText79->Wrap( -1 );
	m_staticText79->SetFont( wxFont( 10, wxFONTFAMILY_SWISS, wxFONTSTYLE_NORMAL, wxFONTWEIGHT_BOLD, false, wxT("Sans") ) );

	InfoSizer->Add( m_staticText79, 0, wxALIGN_RIGHT|wxALL, 5 );

	AvoidHighLowMeanStaticText = new wxStaticText( JobDetailsPanel, wxID_ANY, wxEmptyString, wxDefaultPosition, wxDefaultSize, 0 );
	AvoidHighLowMeanStaticText->Wrap( -1 );
	InfoSizer->Add( AvoidHighLowMeanStaticText, 0, wxALIGN_LEFT|wxALL, 5 );

	m_staticText95 = new wxStaticText( JobDetailsPanel, wxID_ANY, wxT("Num. Bckgd. Boxes :"), wxDefaultPosition, wxDefaultSize, 0 );
	m_staticText95->Wrap( -1 );
	m_staticText95->SetFont( wxFont( 10, wxFONTFAMILY_SWISS, wxFONTSTYLE_NORMAL, wxFONTWEIGHT_BOLD, false, wxT("Sans") ) );

	InfoSizer->Add( m_staticText95, 0, wxALIGN_RIGHT|wxALL, 5 );

	NumBackgroundBoxesStaticText = new wxStaticText( JobDetailsPanel, wxID_ANY, wxEmptyString, wxDefaultPosition, wxDefaultSize, 0 );
	NumBackgroundBoxesStaticText->Wrap( -1 );
	InfoSizer->Add( NumBackgroundBoxesStaticText, 0, wxALIGN_LEFT|wxALL, 5 );


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

	ResultDisplayPanel = new PickingResultsDisplayPanel( RightPanel, wxID_ANY, wxDefaultPosition, wxDefaultSize, wxTAB_TRAVERSAL );
	bSizer681->Add( ResultDisplayPanel, 1, wxEXPAND | wxALL, 5 );

	wxBoxSizer* bSizer69;
	bSizer69 = new wxBoxSizer( wxHORIZONTAL );

	m_staticText469 = new wxStaticText( RightPanel, wxID_ANY, wxT("Left-Click on particles to select/deselect.\nCtrl-Left-Click to rubberband area and deselect.\nShift-Left-Click to paint area and deselect."), wxDefaultPosition, wxDefaultSize, 0 );
	m_staticText469->Wrap( -1 );
	bSizer69->Add( m_staticText469, 0, wxALIGN_CENTER_VERTICAL|wxALL, 5 );


	bSizer69->Add( 0, 0, 1, wxEXPAND, 5 );

	DeleteFromGroupButton = new wxButton( RightPanel, wxID_ANY, wxT("Delete Image From Group"), wxDefaultPosition, wxDefaultSize, 0 );
	bSizer69->Add( DeleteFromGroupButton, 0, wxALIGN_CENTER_VERTICAL|wxALL, 5 );

	AddToGroupButton = new wxButton( RightPanel, wxID_ANY, wxT("Add Image To Group"), wxDefaultPosition, wxDefaultSize, 0 );
	bSizer69->Add( AddToGroupButton, 0, wxALIGN_CENTER_VERTICAL|wxALL, 5 );

	GroupComboBox = new MemoryComboBox( RightPanel, wxID_ANY, wxT("Combo!"), wxDefaultPosition, wxDefaultSize, 0, NULL, wxCB_READONLY );
	GroupComboBox->SetMinSize( wxSize( 200,-1 ) );

	bSizer69->Add( GroupComboBox, 0, wxALIGN_CENTER_VERTICAL|wxALL, 5 );


	bSizer681->Add( bSizer69, 0, wxALIGN_RIGHT|wxEXPAND, 5 );


	RightPanel->SetSizer( bSizer681 );
	RightPanel->Layout();
	bSizer681->Fit( RightPanel );
	m_splitter4->SplitVertically( m_panel13, RightPanel, 450 );
	bSizer63->Add( m_splitter4, 1, wxEXPAND, 5 );


	this->SetSizer( bSizer63 );
	this->Layout();

	// Connect Events
	this->Connect( wxEVT_UPDATE_UI, wxUpdateUIEventHandler( PickingResultsPanel::OnUpdateUI ) );
	AllImagesButton->Connect( wxEVT_COMMAND_RADIOBUTTON_SELECTED, wxCommandEventHandler( PickingResultsPanel::OnAllMoviesSelect ), NULL, this );
	ByFilterButton->Connect( wxEVT_COMMAND_RADIOBUTTON_SELECTED, wxCommandEventHandler( PickingResultsPanel::OnByFilterSelect ), NULL, this );
	FilterButton->Connect( wxEVT_COMMAND_BUTTON_CLICKED, wxCommandEventHandler( PickingResultsPanel::OnDefineFilterClick ), NULL, this );
	JobDetailsToggleButton->Connect( wxEVT_COMMAND_TOGGLEBUTTON_CLICKED, wxCommandEventHandler( PickingResultsPanel::OnJobDetailsToggle ), NULL, this );
	PreviousButton->Connect( wxEVT_COMMAND_BUTTON_CLICKED, wxCommandEventHandler( PickingResultsPanel::OnPreviousButtonClick ), NULL, this );
	NextButton->Connect( wxEVT_COMMAND_BUTTON_CLICKED, wxCommandEventHandler( PickingResultsPanel::OnNextButtonClick ), NULL, this );
	DeleteFromGroupButton->Connect( wxEVT_COMMAND_BUTTON_CLICKED, wxCommandEventHandler( PickingResultsPanel::OnRemoveFromGroupClick ), NULL, this );
	AddToGroupButton->Connect( wxEVT_COMMAND_BUTTON_CLICKED, wxCommandEventHandler( PickingResultsPanel::OnAddToGroupClick ), NULL, this );
}

PickingResultsPanel::~PickingResultsPanel()
{
	// Disconnect Events
	this->Disconnect( wxEVT_UPDATE_UI, wxUpdateUIEventHandler( PickingResultsPanel::OnUpdateUI ) );
	AllImagesButton->Disconnect( wxEVT_COMMAND_RADIOBUTTON_SELECTED, wxCommandEventHandler( PickingResultsPanel::OnAllMoviesSelect ), NULL, this );
	ByFilterButton->Disconnect( wxEVT_COMMAND_RADIOBUTTON_SELECTED, wxCommandEventHandler( PickingResultsPanel::OnByFilterSelect ), NULL, this );
	FilterButton->Disconnect( wxEVT_COMMAND_BUTTON_CLICKED, wxCommandEventHandler( PickingResultsPanel::OnDefineFilterClick ), NULL, this );
	JobDetailsToggleButton->Disconnect( wxEVT_COMMAND_TOGGLEBUTTON_CLICKED, wxCommandEventHandler( PickingResultsPanel::OnJobDetailsToggle ), NULL, this );
	PreviousButton->Disconnect( wxEVT_COMMAND_BUTTON_CLICKED, wxCommandEventHandler( PickingResultsPanel::OnPreviousButtonClick ), NULL, this );
	NextButton->Disconnect( wxEVT_COMMAND_BUTTON_CLICKED, wxCommandEventHandler( PickingResultsPanel::OnNextButtonClick ), NULL, this );
	DeleteFromGroupButton->Disconnect( wxEVT_COMMAND_BUTTON_CLICKED, wxCommandEventHandler( PickingResultsPanel::OnRemoveFromGroupClick ), NULL, this );
	AddToGroupButton->Disconnect( wxEVT_COMMAND_BUTTON_CLICKED, wxCommandEventHandler( PickingResultsPanel::OnAddToGroupClick ), NULL, this );

}

FindParticlesPanel::FindParticlesPanel( wxWindow* parent, wxWindowID id, const wxPoint& pos, const wxSize& size, long style, const wxString& name ) : JobPanel( parent, id, pos, size, style, name )
{
	wxBoxSizer* bSizer43;
	bSizer43 = new wxBoxSizer( wxVERTICAL );

	m_staticline12 = new wxStaticLine( this, wxID_ANY, wxDefaultPosition, wxDefaultSize, wxLI_HORIZONTAL );
	bSizer43->Add( m_staticline12, 0, wxEXPAND | wxALL, 5 );

	wxBoxSizer* bSizer1211;
	bSizer1211 = new wxBoxSizer( wxVERTICAL );

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

	bSizer44->Add( GroupComboBox, 1, wxEXPAND | wxALL, 5 );

	PickingAlgorithStaticText = new wxStaticText( this, wxID_ANY, wxT("Picking algorithm :"), wxDefaultPosition, wxDefaultSize, 0 );
	PickingAlgorithStaticText->Wrap( -1 );
	PickingAlgorithStaticText->Enable( false );

	bSizer44->Add( PickingAlgorithStaticText, 0, wxALIGN_CENTER_VERTICAL|wxALL, 5 );

	PickingAlgorithmComboBox = new wxComboBox( this, wxID_ANY, wxEmptyString, wxDefaultPosition, wxDefaultSize, 0, NULL, wxCB_READONLY );
	PickingAlgorithmComboBox->Append( wxT("default") );
	PickingAlgorithmComboBox->SetSelection( 0 );
	PickingAlgorithmComboBox->Enable( false );
	PickingAlgorithmComboBox->SetMinSize( wxSize( 150,-1 ) );

	bSizer44->Add( PickingAlgorithmComboBox, 0, wxALIGN_CENTER_VERTICAL|wxALL, 5 );


	bSizer44->Add( 0, 0, 60, wxEXPAND, 5 );


	bSizer45->Add( bSizer44, 1, wxEXPAND, 5 );

	ExpertToggleButton = new wxToggleButton( this, wxID_ANY, wxT("Show Expert Options"), wxDefaultPosition, wxDefaultSize, 0 );
	ExpertToggleButton->Enable( false );

	bSizer45->Add( ExpertToggleButton, 0, wxALL, 5 );


	bSizer1211->Add( bSizer45, 1, wxEXPAND, 5 );

	PleaseEstimateCTFStaticText = new wxStaticText( this, wxID_ANY, wxT("Please run CTF estimation on this group before picking particles"), wxDefaultPosition, wxDefaultSize, 0 );
	PleaseEstimateCTFStaticText->Wrap( -1 );
	PleaseEstimateCTFStaticText->SetFont( wxFont( wxNORMAL_FONT->GetPointSize(), wxFONTFAMILY_DEFAULT, wxFONTSTYLE_NORMAL, wxFONTWEIGHT_BOLD, false, wxEmptyString ) );
	PleaseEstimateCTFStaticText->SetForegroundColour( wxColour( 180, 0, 0 ) );

	bSizer1211->Add( PleaseEstimateCTFStaticText, 0, wxALL, 5 );


	bSizer43->Add( bSizer1211, 0, wxEXPAND, 5 );

	m_staticline10 = new wxStaticLine( this, wxID_ANY, wxDefaultPosition, wxDefaultSize, wxLI_HORIZONTAL );
	bSizer43->Add( m_staticline10, 0, wxEXPAND | wxALL, 5 );

	FindParticlesSplitterWindow = new wxSplitterWindow( this, wxID_ANY, wxDefaultPosition, wxDefaultSize, wxSP_3D|wxSP_PERMIT_UNSPLIT );
	FindParticlesSplitterWindow->Connect( wxEVT_IDLE, wxIdleEventHandler( FindParticlesPanel::FindParticlesSplitterWindowOnIdle ), NULL, this );
	FindParticlesSplitterWindow->SetMinimumPaneSize( 10 );

	LeftPanel = new wxPanel( FindParticlesSplitterWindow, wxID_ANY, wxDefaultPosition, wxDefaultSize, wxTAB_TRAVERSAL );
	wxBoxSizer* bSizer214;
	bSizer214 = new wxBoxSizer( wxVERTICAL );

	PickingParametersPanel = new wxScrolledWindow( LeftPanel, wxID_ANY, wxDefaultPosition, wxSize( -1,-1 ), wxVSCROLL );
	PickingParametersPanel->SetScrollRate( 5, 5 );
	InputSizer = new wxBoxSizer( wxVERTICAL );

	wxFlexGridSizer* fgSizer1;
	fgSizer1 = new wxFlexGridSizer( 0, 2, 0, 0 );
	fgSizer1->AddGrowableCol( 1 );
	fgSizer1->SetFlexibleDirection( wxBOTH );
	fgSizer1->SetNonFlexibleGrowMode( wxFLEX_GROWMODE_SPECIFIED );

	m_staticText196 = new wxStaticText( PickingParametersPanel, wxID_ANY, wxT("Exclusion radius (Å) :"), wxDefaultPosition, wxDefaultSize, 0 );
	m_staticText196->Wrap( -1 );
	fgSizer1->Add( m_staticText196, 0, wxALIGN_CENTER_VERTICAL|wxALL, 5 );

	ExclusionRadiusNumericCtrl = new NumericTextCtrl( PickingParametersPanel, wxID_ANY, wxT("120"), wxDefaultPosition, wxDefaultSize, wxTE_PROCESS_ENTER );
	fgSizer1->Add( ExclusionRadiusNumericCtrl, 0, wxALL|wxEXPAND, 5 );

	CharacteristicParticleRadiusStaticText = new wxStaticText( PickingParametersPanel, wxID_ANY, wxT("Template radius (Å) :"), wxDefaultPosition, wxDefaultSize, 0 );
	CharacteristicParticleRadiusStaticText->Wrap( -1 );
	fgSizer1->Add( CharacteristicParticleRadiusStaticText, 0, wxALIGN_CENTER_VERTICAL|wxALL, 5 );

	TemplateRadiusNumericCtrl = new NumericTextCtrl( PickingParametersPanel, wxID_ANY, wxT("80"), wxDefaultPosition, wxDefaultSize, wxTE_PROCESS_ENTER );
	fgSizer1->Add( TemplateRadiusNumericCtrl, 0, wxALL|wxEXPAND, 5 );

	ThresholdPeakHeightStaticText1 = new wxStaticText( PickingParametersPanel, wxID_ANY, wxT("Threshold peak height :"), wxDefaultPosition, wxDefaultSize, 0 );
	ThresholdPeakHeightStaticText1->Wrap( -1 );
	fgSizer1->Add( ThresholdPeakHeightStaticText1, 0, wxALL, 5 );

	ThresholdPeakHeightNumericCtrl = new NumericTextCtrl( PickingParametersPanel, wxID_ANY, wxT("6.0"), wxDefaultPosition, wxDefaultSize, wxTE_PROCESS_ENTER );
	fgSizer1->Add( ThresholdPeakHeightNumericCtrl, 0, wxALL|wxEXPAND, 5 );

	AvoidLowVarianceAreasCheckBox = new wxCheckBox( PickingParametersPanel, wxID_ANY, wxT("Avoid low variance areas"), wxDefaultPosition, wxDefaultSize, 0 );
	AvoidLowVarianceAreasCheckBox->SetValue(true);
	fgSizer1->Add( AvoidLowVarianceAreasCheckBox, 0, wxALL, 5 );

	LowVarianceThresholdNumericCtrl = new NumericTextCtrl( PickingParametersPanel, wxID_ANY, wxT("-0.5"), wxDefaultPosition, wxDefaultSize, wxTE_PROCESS_ENTER );
	fgSizer1->Add( LowVarianceThresholdNumericCtrl, 0, wxALL|wxEXPAND, 5 );

	AvoidHighVarianceAreasCheckBox = new wxCheckBox( PickingParametersPanel, wxID_ANY, wxT("Avoid high variance areas"), wxDefaultPosition, wxDefaultSize, 0 );
	fgSizer1->Add( AvoidHighVarianceAreasCheckBox, 0, wxALL, 5 );

	HighVarianceThresholdNumericCtrl = new NumericTextCtrl( PickingParametersPanel, wxID_ANY, wxT("2.0"), wxDefaultPosition, wxDefaultSize, wxTE_PROCESS_ENTER );
	HighVarianceThresholdNumericCtrl->Enable( false );

	fgSizer1->Add( HighVarianceThresholdNumericCtrl, 0, wxALL|wxEXPAND, 5 );


	InputSizer->Add( fgSizer1, 0, wxEXPAND, 5 );

	m_staticline106 = new wxStaticLine( PickingParametersPanel, wxID_ANY, wxDefaultPosition, wxDefaultSize, wxLI_HORIZONTAL );
	InputSizer->Add( m_staticline106, 0, wxEXPAND | wxALL, 5 );

	m_staticText440 = new wxStaticText( PickingParametersPanel, wxID_ANY, wxT("Select preview image :"), wxDefaultPosition, wxDefaultSize, 0 );
	m_staticText440->Wrap( -1 );
	InputSizer->Add( m_staticText440, 0, wxALL, 5 );

	ImageComboBox = new ImagesPickerComboPanel( PickingParametersPanel, wxID_ANY, wxDefaultPosition, wxDefaultSize, wxTAB_TRAVERSAL );
	ImageComboBox->SetMinSize( wxSize( 350,-1 ) );
	ImageComboBox->SetMaxSize( wxSize( 350,-1 ) );

	InputSizer->Add( ImageComboBox, 0, wxALIGN_CENTER_VERTICAL|wxALL, 5 );

	wxBoxSizer* bSizer212;
	bSizer212 = new wxBoxSizer( wxHORIZONTAL );

	TestOnCurrentMicrographButton = new wxButton( PickingParametersPanel, wxID_ANY, wxT("Preview"), wxDefaultPosition, wxDefaultSize, 0 );
	bSizer212->Add( TestOnCurrentMicrographButton, 0, wxALL, 5 );

	AutoPickRefreshCheckBox = new wxCheckBox( PickingParametersPanel, wxID_ANY, wxT("Auto preview"), wxDefaultPosition, wxDefaultSize, 0 );
	bSizer212->Add( AutoPickRefreshCheckBox, 0, wxALIGN_CENTER_VERTICAL|wxALL, 5 );


	InputSizer->Add( bSizer212, 0, 0, 5 );

	ExpertOptionsPanel = new wxPanel( PickingParametersPanel, wxID_ANY, wxDefaultPosition, wxDefaultSize, wxTAB_TRAVERSAL );
	ExpertOptionsPanel->Hide();

	ExpertInputSizer = new wxBoxSizer( wxVERTICAL );

	m_staticline35 = new wxStaticLine( ExpertOptionsPanel, wxID_ANY, wxDefaultPosition, wxDefaultSize, wxLI_HORIZONTAL );
	ExpertInputSizer->Add( m_staticline35, 0, wxEXPAND | wxALL, 5 );

	wxFlexGridSizer* ExpertOptionsSizer;
	ExpertOptionsSizer = new wxFlexGridSizer( 0, 2, 0, 0 );
	ExpertOptionsSizer->SetFlexibleDirection( wxBOTH );
	ExpertOptionsSizer->SetNonFlexibleGrowMode( wxFLEX_GROWMODE_SPECIFIED );

	ExpertOptionsStaticText = new wxStaticText( ExpertOptionsPanel, wxID_ANY, wxT("Expert Options"), wxDefaultPosition, wxDefaultSize, 0 );
	ExpertOptionsStaticText->Wrap( -1 );
	ExpertOptionsStaticText->SetFont( wxFont( wxNORMAL_FONT->GetPointSize(), wxFONTFAMILY_DEFAULT, wxFONTSTYLE_NORMAL, wxFONTWEIGHT_BOLD, true, wxEmptyString ) );

	ExpertOptionsSizer->Add( ExpertOptionsStaticText, 0, wxALL|wxEXPAND, 5 );


	ExpertOptionsSizer->Add( 0, 0, 1, wxEXPAND, 5 );

	HighestResolutionStaticText = new wxStaticText( ExpertOptionsPanel, wxID_ANY, wxT("Highest resolution used (Å) :"), wxDefaultPosition, wxDefaultSize, 0 );
	HighestResolutionStaticText->Wrap( -1 );
	ExpertOptionsSizer->Add( HighestResolutionStaticText, 0, wxALIGN_CENTER_VERTICAL|wxALL|wxEXPAND, 5 );

	HighestResolutionNumericCtrl = new NumericTextCtrl( ExpertOptionsPanel, wxID_ANY, wxT("30.00"), wxDefaultPosition, wxDefaultSize, wxTE_PROCESS_ENTER );
	ExpertOptionsSizer->Add( HighestResolutionNumericCtrl, 0, wxALL|wxEXPAND, 5 );

	SetMinimumDistanceFromEdgesCheckBox = new wxCheckBox( ExpertOptionsPanel, wxID_ANY, wxT("Min. edge distance (pix.) : "), wxDefaultPosition, wxDefaultSize, 0 );
	ExpertOptionsSizer->Add( SetMinimumDistanceFromEdgesCheckBox, 0, wxALL|wxEXPAND, 5 );

	MinimumDistanceFromEdgesSpinCtrl = new wxSpinCtrl( ExpertOptionsPanel, wxID_ANY, wxEmptyString, wxDefaultPosition, wxDefaultSize, wxSP_ARROW_KEYS, 1, 999999999, 128 );
	MinimumDistanceFromEdgesSpinCtrl->Enable( false );

	ExpertOptionsSizer->Add( MinimumDistanceFromEdgesSpinCtrl, 0, wxALL|wxEXPAND, 5 );

	m_checkBox9 = new wxCheckBox( ExpertOptionsPanel, wxID_ANY, wxT("Number of template rotations : "), wxDefaultPosition, wxDefaultSize, 0 );
	m_checkBox9->Enable( false );

	ExpertOptionsSizer->Add( m_checkBox9, 0, wxALL|wxEXPAND, 5 );

	NumberOfTemplateRotationsSpinCtrl = new wxSpinCtrl( ExpertOptionsPanel, wxID_ANY, wxEmptyString, wxDefaultPosition, wxDefaultSize, wxSP_ARROW_KEYS, 1, 360, 72 );
	NumberOfTemplateRotationsSpinCtrl->Enable( false );

	ExpertOptionsSizer->Add( NumberOfTemplateRotationsSpinCtrl, 0, wxALL|wxEXPAND, 5 );

	AvoidAbnormalLocalMeanAreasCheckBox = new wxCheckBox( ExpertOptionsPanel, wxID_ANY, wxT("Avoid abnormal local mean"), wxDefaultPosition, wxDefaultSize, 0 );
	AvoidAbnormalLocalMeanAreasCheckBox->SetValue(true);
	ExpertOptionsSizer->Add( AvoidAbnormalLocalMeanAreasCheckBox, 0, wxALL|wxEXPAND, 5 );


	ExpertOptionsSizer->Add( 0, 0, 1, wxEXPAND, 5 );

	m_staticText170 = new wxStaticText( ExpertOptionsPanel, wxID_ANY, wxT("Number of background boxes : "), wxDefaultPosition, wxDefaultSize, 0 );
	m_staticText170->Wrap( -1 );
	ExpertOptionsSizer->Add( m_staticText170, 0, wxALIGN_CENTER_VERTICAL|wxALL|wxEXPAND, 5 );

	NumberOfBackgroundBoxesSpinCtrl = new wxSpinCtrl( ExpertOptionsPanel, wxID_ANY, wxEmptyString, wxDefaultPosition, wxDefaultSize, wxSP_ARROW_KEYS, 1, 999, 50 );
	ExpertOptionsSizer->Add( NumberOfBackgroundBoxesSpinCtrl, 0, wxALL|wxEXPAND, 5 );

	m_staticText169 = new wxStaticText( ExpertOptionsPanel, wxID_ANY, wxT("Find background areas by : "), wxDefaultPosition, wxDefaultSize, 0 );
	m_staticText169->Wrap( -1 );
	ExpertOptionsSizer->Add( m_staticText169, 0, wxALIGN_CENTER_VERTICAL|wxALL|wxEXPAND, 5 );

	wxString AlgorithmToFindBackgroundChoiceChoices[] = { wxT("Lowest variance"), wxT("Variance near mode") };
	int AlgorithmToFindBackgroundChoiceNChoices = sizeof( AlgorithmToFindBackgroundChoiceChoices ) / sizeof( wxString );
	AlgorithmToFindBackgroundChoice = new wxChoice( ExpertOptionsPanel, wxID_ANY, wxDefaultPosition, wxDefaultSize, AlgorithmToFindBackgroundChoiceNChoices, AlgorithmToFindBackgroundChoiceChoices, 0 );
	AlgorithmToFindBackgroundChoice->SetSelection( 0 );
	ExpertOptionsSizer->Add( AlgorithmToFindBackgroundChoice, 0, wxALL|wxEXPAND, 5 );


	ExpertInputSizer->Add( ExpertOptionsSizer, 1, wxEXPAND, 5 );


	ExpertOptionsPanel->SetSizer( ExpertInputSizer );
	ExpertOptionsPanel->Layout();
	ExpertInputSizer->Fit( ExpertOptionsPanel );
	InputSizer->Add( ExpertOptionsPanel, 0, wxALL|wxEXPAND, 0 );


	PickingParametersPanel->SetSizer( InputSizer );
	PickingParametersPanel->Layout();
	InputSizer->Fit( PickingParametersPanel );
	bSizer214->Add( PickingParametersPanel, 1, wxALL|wxEXPAND, 5 );

	OutputTextPanel = new wxPanel( LeftPanel, wxID_ANY, wxDefaultPosition, wxDefaultSize, wxTAB_TRAVERSAL );
	OutputTextPanel->Hide();

	wxBoxSizer* bSizer56;
	bSizer56 = new wxBoxSizer( wxVERTICAL );

	output_textctrl = new wxTextCtrl( OutputTextPanel, wxID_ANY, wxEmptyString, wxDefaultPosition, wxDefaultSize, wxTE_MULTILINE|wxTE_READONLY );
	bSizer56->Add( output_textctrl, 1, wxALL|wxEXPAND, 5 );


	OutputTextPanel->SetSizer( bSizer56 );
	OutputTextPanel->Layout();
	bSizer56->Fit( OutputTextPanel );
	bSizer214->Add( OutputTextPanel, 30, wxEXPAND | wxALL, 5 );


	LeftPanel->SetSizer( bSizer214 );
	LeftPanel->Layout();
	bSizer214->Fit( LeftPanel );
	RightPanel = new wxPanel( FindParticlesSplitterWindow, wxID_ANY, wxDefaultPosition, wxDefaultSize, wxTAB_TRAVERSAL );
	wxBoxSizer* bSizer215;
	bSizer215 = new wxBoxSizer( wxVERTICAL );

	PickingResultsPanel = new PickingResultsDisplayPanel( RightPanel, wxID_ANY, wxDefaultPosition, wxDefaultSize, wxTAB_TRAVERSAL );
	PickingResultsPanel->Hide();

	bSizer215->Add( PickingResultsPanel, 1, wxEXPAND | wxALL, 5 );

	InfoPanel = new wxPanel( RightPanel, wxID_ANY, wxDefaultPosition, wxDefaultSize, wxTAB_TRAVERSAL );
	wxBoxSizer* bSizer61;
	bSizer61 = new wxBoxSizer( wxVERTICAL );

	InfoText = new wxRichTextCtrl( InfoPanel, wxID_ANY, wxEmptyString, wxDefaultPosition, wxDefaultSize, wxTE_READONLY|wxHSCROLL|wxVSCROLL );
	bSizer61->Add( InfoText, 1, wxEXPAND | wxALL, 5 );


	InfoPanel->SetSizer( bSizer61 );
	InfoPanel->Layout();
	bSizer61->Fit( InfoPanel );
	bSizer215->Add( InfoPanel, 1, wxEXPAND | wxALL, 5 );


	RightPanel->SetSizer( bSizer215 );
	RightPanel->Layout();
	bSizer215->Fit( RightPanel );
	FindParticlesSplitterWindow->SplitVertically( LeftPanel, RightPanel, 350 );
	bSizer43->Add( FindParticlesSplitterWindow, 1, wxEXPAND, 5 );

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

	StartPickingButton = new wxButton( StartPanel, wxID_ANY, wxT("Start Picking"), wxDefaultPosition, wxDefaultSize, 0 );
	bSizer60->Add( StartPickingButton, 0, wxALL, 5 );


	bSizer58->Add( bSizer60, 50, wxEXPAND, 5 );


	StartPanel->SetSizer( bSizer58 );
	StartPanel->Layout();
	bSizer58->Fit( StartPanel );
	bSizer48->Add( StartPanel, 1, wxEXPAND | wxALL, 5 );


	bSizer43->Add( bSizer48, 0, wxEXPAND, 5 );


	this->SetSizer( bSizer43 );
	this->Layout();

	// Connect Events
	this->Connect( wxEVT_UPDATE_UI, wxUpdateUIEventHandler( FindParticlesPanel::OnUpdateUI ) );
	PickingAlgorithmComboBox->Connect( wxEVT_COMMAND_COMBOBOX_SELECTED, wxCommandEventHandler( FindParticlesPanel::OnPickingAlgorithmComboBox ), NULL, this );
	ExpertToggleButton->Connect( wxEVT_COMMAND_TOGGLEBUTTON_CLICKED, wxCommandEventHandler( FindParticlesPanel::OnExpertOptionsToggle ), NULL, this );
	ExclusionRadiusNumericCtrl->Connect( wxEVT_KILL_FOCUS, wxFocusEventHandler( FindParticlesPanel::OnExclusionRadiusNumericTextKillFocus ), NULL, this );
	ExclusionRadiusNumericCtrl->Connect( wxEVT_SET_FOCUS, wxFocusEventHandler( FindParticlesPanel::OnExclusionRadiusNumericTextSetFocus ), NULL, this );
	ExclusionRadiusNumericCtrl->Connect( wxEVT_COMMAND_TEXT_ENTER, wxCommandEventHandler( FindParticlesPanel::OnExclusionRadiusNumericTextEnter ), NULL, this );
	TemplateRadiusNumericCtrl->Connect( wxEVT_KILL_FOCUS, wxFocusEventHandler( FindParticlesPanel::OnTemplateRadiusNumericTextKillFocus ), NULL, this );
	TemplateRadiusNumericCtrl->Connect( wxEVT_SET_FOCUS, wxFocusEventHandler( FindParticlesPanel::OnTemplateRadiusNumericTextSetFocus ), NULL, this );
	TemplateRadiusNumericCtrl->Connect( wxEVT_COMMAND_TEXT_ENTER, wxCommandEventHandler( FindParticlesPanel::OnTemplateRadiusNumericTextEnter ), NULL, this );
	ThresholdPeakHeightNumericCtrl->Connect( wxEVT_KILL_FOCUS, wxFocusEventHandler( FindParticlesPanel::OnThresholdPeakHeightNumericTextKillFocus ), NULL, this );
	ThresholdPeakHeightNumericCtrl->Connect( wxEVT_SET_FOCUS, wxFocusEventHandler( FindParticlesPanel::OnThresholdPeakHeightNumericTextSetFocus ), NULL, this );
	ThresholdPeakHeightNumericCtrl->Connect( wxEVT_COMMAND_TEXT_ENTER, wxCommandEventHandler( FindParticlesPanel::OnThresholdPeakHeightNumericTextEnter ), NULL, this );
	AvoidLowVarianceAreasCheckBox->Connect( wxEVT_COMMAND_CHECKBOX_CLICKED, wxCommandEventHandler( FindParticlesPanel::OnAvoidLowVarianceAreasCheckBox ), NULL, this );
	LowVarianceThresholdNumericCtrl->Connect( wxEVT_KILL_FOCUS, wxFocusEventHandler( FindParticlesPanel::OnThresholdPeakHeightNumericTextKillFocus ), NULL, this );
	LowVarianceThresholdNumericCtrl->Connect( wxEVT_SET_FOCUS, wxFocusEventHandler( FindParticlesPanel::OnThresholdPeakHeightNumericTextSetFocus ), NULL, this );
	LowVarianceThresholdNumericCtrl->Connect( wxEVT_COMMAND_TEXT_ENTER, wxCommandEventHandler( FindParticlesPanel::OnLowVarianceThresholdNumericTextEnter ), NULL, this );
	AvoidHighVarianceAreasCheckBox->Connect( wxEVT_COMMAND_CHECKBOX_CLICKED, wxCommandEventHandler( FindParticlesPanel::OnAvoidHighVarianceAreasCheckBox ), NULL, this );
	HighVarianceThresholdNumericCtrl->Connect( wxEVT_KILL_FOCUS, wxFocusEventHandler( FindParticlesPanel::OnThresholdPeakHeightNumericTextKillFocus ), NULL, this );
	HighVarianceThresholdNumericCtrl->Connect( wxEVT_SET_FOCUS, wxFocusEventHandler( FindParticlesPanel::OnThresholdPeakHeightNumericTextSetFocus ), NULL, this );
	HighVarianceThresholdNumericCtrl->Connect( wxEVT_COMMAND_TEXT_ENTER, wxCommandEventHandler( FindParticlesPanel::OnHighVarianceThresholdNumericTextEnter ), NULL, this );
	TestOnCurrentMicrographButton->Connect( wxEVT_COMMAND_BUTTON_CLICKED, wxCommandEventHandler( FindParticlesPanel::OnTestOnCurrentMicrographButtonClick ), NULL, this );
	AutoPickRefreshCheckBox->Connect( wxEVT_COMMAND_CHECKBOX_CLICKED, wxCommandEventHandler( FindParticlesPanel::OnAutoPickRefreshCheckBox ), NULL, this );
	HighestResolutionNumericCtrl->Connect( wxEVT_KILL_FOCUS, wxFocusEventHandler( FindParticlesPanel::OnHighestResolutionNumericKillFocus ), NULL, this );
	HighestResolutionNumericCtrl->Connect( wxEVT_SET_FOCUS, wxFocusEventHandler( FindParticlesPanel::OnHighestResolutionNumericSetFocus ), NULL, this );
	HighestResolutionNumericCtrl->Connect( wxEVT_COMMAND_TEXT_ENTER, wxCommandEventHandler( FindParticlesPanel::OnHighestResolutionNumericTextEnter ), NULL, this );
	SetMinimumDistanceFromEdgesCheckBox->Connect( wxEVT_COMMAND_CHECKBOX_CLICKED, wxCommandEventHandler( FindParticlesPanel::OnSetMinimumDistanceFromEdgesCheckBox ), NULL, this );
	MinimumDistanceFromEdgesSpinCtrl->Connect( wxEVT_COMMAND_SPINCTRL_UPDATED, wxSpinEventHandler( FindParticlesPanel::OnMinimumDistanceFromEdgesSpinCtrl ), NULL, this );
	AvoidAbnormalLocalMeanAreasCheckBox->Connect( wxEVT_COMMAND_CHECKBOX_CLICKED, wxCommandEventHandler( FindParticlesPanel::OnAvoidAbnormalLocalMeanAreasCheckBox ), NULL, this );
	NumberOfBackgroundBoxesSpinCtrl->Connect( wxEVT_COMMAND_SPINCTRL_UPDATED, wxSpinEventHandler( FindParticlesPanel::OnNumberOfBackgroundBoxesSpinCtrl ), NULL, this );
	AlgorithmToFindBackgroundChoice->Connect( wxEVT_COMMAND_CHOICE_SELECTED, wxCommandEventHandler( FindParticlesPanel::OnAlgorithmToFindBackgroundChoice ), NULL, this );
	InfoText->Connect( wxEVT_COMMAND_TEXT_URL, wxTextUrlEventHandler( FindParticlesPanel::OnInfoURL ), NULL, this );
	FinishButton->Connect( wxEVT_COMMAND_BUTTON_CLICKED, wxCommandEventHandler( FindParticlesPanel::FinishButtonClick ), NULL, this );
	CancelAlignmentButton->Connect( wxEVT_COMMAND_BUTTON_CLICKED, wxCommandEventHandler( FindParticlesPanel::TerminateButtonClick ), NULL, this );
	StartPickingButton->Connect( wxEVT_COMMAND_BUTTON_CLICKED, wxCommandEventHandler( FindParticlesPanel::StartPickingClick ), NULL, this );
}

FindParticlesPanel::~FindParticlesPanel()
{
	// Disconnect Events
	this->Disconnect( wxEVT_UPDATE_UI, wxUpdateUIEventHandler( FindParticlesPanel::OnUpdateUI ) );
	PickingAlgorithmComboBox->Disconnect( wxEVT_COMMAND_COMBOBOX_SELECTED, wxCommandEventHandler( FindParticlesPanel::OnPickingAlgorithmComboBox ), NULL, this );
	ExpertToggleButton->Disconnect( wxEVT_COMMAND_TOGGLEBUTTON_CLICKED, wxCommandEventHandler( FindParticlesPanel::OnExpertOptionsToggle ), NULL, this );
	ExclusionRadiusNumericCtrl->Disconnect( wxEVT_KILL_FOCUS, wxFocusEventHandler( FindParticlesPanel::OnExclusionRadiusNumericTextKillFocus ), NULL, this );
	ExclusionRadiusNumericCtrl->Disconnect( wxEVT_SET_FOCUS, wxFocusEventHandler( FindParticlesPanel::OnExclusionRadiusNumericTextSetFocus ), NULL, this );
	ExclusionRadiusNumericCtrl->Disconnect( wxEVT_COMMAND_TEXT_ENTER, wxCommandEventHandler( FindParticlesPanel::OnExclusionRadiusNumericTextEnter ), NULL, this );
	TemplateRadiusNumericCtrl->Disconnect( wxEVT_KILL_FOCUS, wxFocusEventHandler( FindParticlesPanel::OnTemplateRadiusNumericTextKillFocus ), NULL, this );
	TemplateRadiusNumericCtrl->Disconnect( wxEVT_SET_FOCUS, wxFocusEventHandler( FindParticlesPanel::OnTemplateRadiusNumericTextSetFocus ), NULL, this );
	TemplateRadiusNumericCtrl->Disconnect( wxEVT_COMMAND_TEXT_ENTER, wxCommandEventHandler( FindParticlesPanel::OnTemplateRadiusNumericTextEnter ), NULL, this );
	ThresholdPeakHeightNumericCtrl->Disconnect( wxEVT_KILL_FOCUS, wxFocusEventHandler( FindParticlesPanel::OnThresholdPeakHeightNumericTextKillFocus ), NULL, this );
	ThresholdPeakHeightNumericCtrl->Disconnect( wxEVT_SET_FOCUS, wxFocusEventHandler( FindParticlesPanel::OnThresholdPeakHeightNumericTextSetFocus ), NULL, this );
	ThresholdPeakHeightNumericCtrl->Disconnect( wxEVT_COMMAND_TEXT_ENTER, wxCommandEventHandler( FindParticlesPanel::OnThresholdPeakHeightNumericTextEnter ), NULL, this );
	AvoidLowVarianceAreasCheckBox->Disconnect( wxEVT_COMMAND_CHECKBOX_CLICKED, wxCommandEventHandler( FindParticlesPanel::OnAvoidLowVarianceAreasCheckBox ), NULL, this );
	LowVarianceThresholdNumericCtrl->Disconnect( wxEVT_KILL_FOCUS, wxFocusEventHandler( FindParticlesPanel::OnThresholdPeakHeightNumericTextKillFocus ), NULL, this );
	LowVarianceThresholdNumericCtrl->Disconnect( wxEVT_SET_FOCUS, wxFocusEventHandler( FindParticlesPanel::OnThresholdPeakHeightNumericTextSetFocus ), NULL, this );
	LowVarianceThresholdNumericCtrl->Disconnect( wxEVT_COMMAND_TEXT_ENTER, wxCommandEventHandler( FindParticlesPanel::OnLowVarianceThresholdNumericTextEnter ), NULL, this );
	AvoidHighVarianceAreasCheckBox->Disconnect( wxEVT_COMMAND_CHECKBOX_CLICKED, wxCommandEventHandler( FindParticlesPanel::OnAvoidHighVarianceAreasCheckBox ), NULL, this );
	HighVarianceThresholdNumericCtrl->Disconnect( wxEVT_KILL_FOCUS, wxFocusEventHandler( FindParticlesPanel::OnThresholdPeakHeightNumericTextKillFocus ), NULL, this );
	HighVarianceThresholdNumericCtrl->Disconnect( wxEVT_SET_FOCUS, wxFocusEventHandler( FindParticlesPanel::OnThresholdPeakHeightNumericTextSetFocus ), NULL, this );
	HighVarianceThresholdNumericCtrl->Disconnect( wxEVT_COMMAND_TEXT_ENTER, wxCommandEventHandler( FindParticlesPanel::OnHighVarianceThresholdNumericTextEnter ), NULL, this );
	TestOnCurrentMicrographButton->Disconnect( wxEVT_COMMAND_BUTTON_CLICKED, wxCommandEventHandler( FindParticlesPanel::OnTestOnCurrentMicrographButtonClick ), NULL, this );
	AutoPickRefreshCheckBox->Disconnect( wxEVT_COMMAND_CHECKBOX_CLICKED, wxCommandEventHandler( FindParticlesPanel::OnAutoPickRefreshCheckBox ), NULL, this );
	HighestResolutionNumericCtrl->Disconnect( wxEVT_KILL_FOCUS, wxFocusEventHandler( FindParticlesPanel::OnHighestResolutionNumericKillFocus ), NULL, this );
	HighestResolutionNumericCtrl->Disconnect( wxEVT_SET_FOCUS, wxFocusEventHandler( FindParticlesPanel::OnHighestResolutionNumericSetFocus ), NULL, this );
	HighestResolutionNumericCtrl->Disconnect( wxEVT_COMMAND_TEXT_ENTER, wxCommandEventHandler( FindParticlesPanel::OnHighestResolutionNumericTextEnter ), NULL, this );
	SetMinimumDistanceFromEdgesCheckBox->Disconnect( wxEVT_COMMAND_CHECKBOX_CLICKED, wxCommandEventHandler( FindParticlesPanel::OnSetMinimumDistanceFromEdgesCheckBox ), NULL, this );
	MinimumDistanceFromEdgesSpinCtrl->Disconnect( wxEVT_COMMAND_SPINCTRL_UPDATED, wxSpinEventHandler( FindParticlesPanel::OnMinimumDistanceFromEdgesSpinCtrl ), NULL, this );
	AvoidAbnormalLocalMeanAreasCheckBox->Disconnect( wxEVT_COMMAND_CHECKBOX_CLICKED, wxCommandEventHandler( FindParticlesPanel::OnAvoidAbnormalLocalMeanAreasCheckBox ), NULL, this );
	NumberOfBackgroundBoxesSpinCtrl->Disconnect( wxEVT_COMMAND_SPINCTRL_UPDATED, wxSpinEventHandler( FindParticlesPanel::OnNumberOfBackgroundBoxesSpinCtrl ), NULL, this );
	AlgorithmToFindBackgroundChoice->Disconnect( wxEVT_COMMAND_CHOICE_SELECTED, wxCommandEventHandler( FindParticlesPanel::OnAlgorithmToFindBackgroundChoice ), NULL, this );
	InfoText->Disconnect( wxEVT_COMMAND_TEXT_URL, wxTextUrlEventHandler( FindParticlesPanel::OnInfoURL ), NULL, this );
	FinishButton->Disconnect( wxEVT_COMMAND_BUTTON_CLICKED, wxCommandEventHandler( FindParticlesPanel::FinishButtonClick ), NULL, this );
	CancelAlignmentButton->Disconnect( wxEVT_COMMAND_BUTTON_CLICKED, wxCommandEventHandler( FindParticlesPanel::TerminateButtonClick ), NULL, this );
	StartPickingButton->Disconnect( wxEVT_COMMAND_BUTTON_CLICKED, wxCommandEventHandler( FindParticlesPanel::StartPickingClick ), NULL, this );

}

ParticlePositionExportDialog::ParticlePositionExportDialog( wxWindow* parent, wxWindowID id, const wxString& title, const wxPoint& pos, const wxSize& size, long style ) : wxDialog( parent, id, title, pos, size, style )
{
	this->SetSizeHints( wxDefaultSize, wxDefaultSize );

	wxBoxSizer* bSizer133;
	bSizer133 = new wxBoxSizer( wxVERTICAL );

	m_panel38 = new wxPanel( this, wxID_ANY, wxDefaultPosition, wxDefaultSize, wxTAB_TRAVERSAL );
	wxBoxSizer* bSizer135;
	bSizer135 = new wxBoxSizer( wxVERTICAL );

	wxStaticBoxSizer* sbSizer3;
	sbSizer3 = new wxStaticBoxSizer( new wxStaticBox( m_panel38, wxID_ANY, wxT("Export from these images") ), wxVERTICAL );

	GroupComboBox = new wxComboBox( sbSizer3->GetStaticBox(), wxID_ANY, wxEmptyString, wxDefaultPosition, wxDefaultSize, 0, NULL, wxCB_READONLY );
	sbSizer3->Add( GroupComboBox, 0, wxALL|wxEXPAND, 5 );


	bSizer135->Add( sbSizer3, 0, wxEXPAND, 25 );

	wxStaticBoxSizer* sbSizer4;
	sbSizer4 = new wxStaticBoxSizer( new wxStaticBox( m_panel38, wxID_ANY, wxT("Destination directory") ), wxVERTICAL );

	DestinationDirectoryPickerCtrl = new wxDirPickerCtrl( sbSizer4->GetStaticBox(), wxID_ANY, wxEmptyString, wxT("Select a folder"), wxDefaultPosition, wxDefaultSize, wxDIRP_DEFAULT_STYLE );
	sbSizer4->Add( DestinationDirectoryPickerCtrl, 0, wxALL|wxEXPAND, 5 );


	bSizer135->Add( sbSizer4, 0, wxEXPAND, 5 );


	bSizer135->Add( 0, 0, 0, wxEXPAND, 5 );

	WarningText = new wxStaticText( m_panel38, wxID_ANY, wxT("Warning: running jobs \nmay affect exported coordinates"), wxDefaultPosition, wxDefaultSize, wxALIGN_CENTER_HORIZONTAL );
	WarningText->Wrap( -1 );
	WarningText->SetForegroundColour( wxColour( 180, 0, 0 ) );
	WarningText->Hide();

	bSizer135->Add( WarningText, 0, wxALL, 5 );

	wxBoxSizer* bSizer137;
	bSizer137 = new wxBoxSizer( wxHORIZONTAL );


	bSizer137->Add( 0, 0, 1, wxEXPAND, 5 );

	CancelButton = new wxButton( m_panel38, wxID_ANY, wxT("Cancel"), wxDefaultPosition, wxDefaultSize, 0 );
	bSizer137->Add( CancelButton, 0, wxALL, 5 );

	ExportButton = new wxButton( m_panel38, wxID_ANY, wxT("Export"), wxDefaultPosition, wxDefaultSize, 0 );

	ExportButton->SetDefault();
	ExportButton->Enable( false );

	bSizer137->Add( ExportButton, 0, wxALL, 5 );


	bSizer135->Add( bSizer137, 0, wxEXPAND, 5 );


	m_panel38->SetSizer( bSizer135 );
	m_panel38->Layout();
	bSizer135->Fit( m_panel38 );
	bSizer133->Add( m_panel38, 0, wxEXPAND | wxALL, 5 );


	this->SetSizer( bSizer133 );
	this->Layout();
	bSizer133->Fit( this );

	this->Centre( wxBOTH );

	// Connect Events
	DestinationDirectoryPickerCtrl->Connect( wxEVT_COMMAND_DIRPICKER_CHANGED, wxFileDirPickerEventHandler( ParticlePositionExportDialog::OnDirChanged ), NULL, this );
	CancelButton->Connect( wxEVT_COMMAND_BUTTON_CLICKED, wxCommandEventHandler( ParticlePositionExportDialog::OnCancelButtonClick ), NULL, this );
	ExportButton->Connect( wxEVT_COMMAND_BUTTON_CLICKED, wxCommandEventHandler( ParticlePositionExportDialog::OnExportButtonClick ), NULL, this );
}

ParticlePositionExportDialog::~ParticlePositionExportDialog()
{
	// Disconnect Events
	DestinationDirectoryPickerCtrl->Disconnect( wxEVT_COMMAND_DIRPICKER_CHANGED, wxFileDirPickerEventHandler( ParticlePositionExportDialog::OnDirChanged ), NULL, this );
	CancelButton->Disconnect( wxEVT_COMMAND_BUTTON_CLICKED, wxCommandEventHandler( ParticlePositionExportDialog::OnCancelButtonClick ), NULL, this );
	ExportButton->Disconnect( wxEVT_COMMAND_BUTTON_CLICKED, wxCommandEventHandler( ParticlePositionExportDialog::OnExportButtonClick ), NULL, this );

}
