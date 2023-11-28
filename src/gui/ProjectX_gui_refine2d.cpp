///////////////////////////////////////////////////////////////////////////
// C++ code generated with wxFormBuilder (version 3.10.0-4761b0c)
// http://www.wxformbuilder.org/
//
// PLEASE DO *NOT* EDIT THIS FILE!
///////////////////////////////////////////////////////////////////////////

#include "AssetPickerComboPanel.h"
#include "ClassificationPlotPanel.h"
#include "DisplayPanel.h"
#include "my_controls.h"

#include "ProjectX_gui_refine2d.h"

///////////////////////////////////////////////////////////////////////////

Refine2DPanel::Refine2DPanel( wxWindow* parent, wxWindowID id, const wxPoint& pos, const wxSize& size, long style, const wxString& name ) : JobPanel( parent, id, pos, size, style, name )
{
	wxBoxSizer* bSizer43;
	bSizer43 = new wxBoxSizer( wxVERTICAL );

	m_staticline12 = new wxStaticLine( this, wxID_ANY, wxDefaultPosition, wxDefaultSize, wxLI_HORIZONTAL );
	bSizer43->Add( m_staticline12, 0, wxEXPAND | wxALL, 5 );

	InputParamsPanel = new wxPanel( this, wxID_ANY, wxDefaultPosition, wxDefaultSize, wxTAB_TRAVERSAL );
	wxBoxSizer* bSizer451;
	bSizer451 = new wxBoxSizer( wxHORIZONTAL );

	wxBoxSizer* bSizer441;
	bSizer441 = new wxBoxSizer( wxVERTICAL );

	wxBoxSizer* bSizer200;
	bSizer200 = new wxBoxSizer( wxHORIZONTAL );

	wxFlexGridSizer* fgSizer14;
	fgSizer14 = new wxFlexGridSizer( 0, 2, 0, 0 );
	fgSizer14->SetFlexibleDirection( wxBOTH );
	fgSizer14->SetNonFlexibleGrowMode( wxFLEX_GROWMODE_SPECIFIED );

	m_staticText262 = new wxStaticText( InputParamsPanel, wxID_ANY, wxT("Input Refinement Package :"), wxDefaultPosition, wxSize( -1,-1 ), 0 );
	m_staticText262->Wrap( -1 );
	fgSizer14->Add( m_staticText262, 0, wxALIGN_CENTER_VERTICAL|wxALIGN_RIGHT|wxALL, 5 );

	RefinementPackageComboBox = new RefinementPackagePickerComboPanel( InputParamsPanel, wxID_ANY, wxDefaultPosition, wxDefaultSize, wxTAB_TRAVERSAL );
	RefinementPackageComboBox->SetMinSize( wxSize( 350,-1 ) );
	RefinementPackageComboBox->SetMaxSize( wxSize( 350,-1 ) );

	fgSizer14->Add( RefinementPackageComboBox, 0, wxALIGN_CENTER_VERTICAL|wxALL, 5 );

	m_staticText263 = new wxStaticText( InputParamsPanel, wxID_ANY, wxT("Input Starting References :"), wxDefaultPosition, wxDefaultSize, 0 );
	m_staticText263->Wrap( -1 );
	fgSizer14->Add( m_staticText263, 0, wxALIGN_CENTER_VERTICAL|wxALIGN_RIGHT|wxALL, 5 );

	InputParametersComboBox = new ClassificationPickerComboPanel( InputParamsPanel, wxID_ANY, wxDefaultPosition, wxDefaultSize, wxTAB_TRAVERSAL );
	InputParametersComboBox->SetMinSize( wxSize( 350,-1 ) );
	InputParametersComboBox->SetMaxSize( wxSize( 350,-1 ) );

	fgSizer14->Add( InputParametersComboBox, 0, wxALIGN_CENTER_VERTICAL|wxALL, 5 );


	bSizer200->Add( fgSizer14, 0, wxEXPAND, 5 );

	m_staticline57 = new wxStaticLine( InputParamsPanel, wxID_ANY, wxDefaultPosition, wxDefaultSize, wxLI_HORIZONTAL|wxLI_VERTICAL );
	bSizer200->Add( m_staticline57, 0, wxEXPAND | wxALL, 5 );

	wxGridSizer* gSizer10;
	gSizer10 = new wxGridSizer( 0, 1, 0, 0 );

	wxBoxSizer* bSizer215;
	bSizer215 = new wxBoxSizer( wxHORIZONTAL );


	bSizer215->Add( 0, 0, 1, wxEXPAND, 5 );

	m_staticText3321 = new wxStaticText( InputParamsPanel, wxID_ANY, wxT("No. of Classes :"), wxDefaultPosition, wxDefaultSize, 0 );
	m_staticText3321->Wrap( -1 );
	bSizer215->Add( m_staticText3321, 0, wxALIGN_CENTER_VERTICAL|wxALL, 5 );

	NumberClassesSpinCtrl = new wxSpinCtrl( InputParamsPanel, wxID_ANY, wxEmptyString, wxDefaultPosition, wxDefaultSize, wxSP_ARROW_KEYS, 1, 1000, 50 );
	NumberClassesSpinCtrl->SetMinSize( wxSize( 100,-1 ) );

	bSizer215->Add( NumberClassesSpinCtrl, 0, wxALL, 5 );


	gSizer10->Add( bSizer215, 1, wxEXPAND, 5 );

	wxBoxSizer* bSizer216;
	bSizer216 = new wxBoxSizer( wxHORIZONTAL );

	m_staticText264 = new wxStaticText( InputParamsPanel, wxID_ANY, wxT("No. of Cycles to Run :"), wxDefaultPosition, wxDefaultSize, 0 );
	m_staticText264->Wrap( -1 );
	bSizer216->Add( m_staticText264, 0, wxALIGN_CENTER_VERTICAL|wxALIGN_RIGHT|wxALL, 5 );

	NumberRoundsSpinCtrl = new wxSpinCtrl( InputParamsPanel, wxID_ANY, wxEmptyString, wxDefaultPosition, wxDefaultSize, wxSP_ARROW_KEYS, 1, 100, 20 );
	NumberRoundsSpinCtrl->SetMinSize( wxSize( 100,-1 ) );

	bSizer216->Add( NumberRoundsSpinCtrl, 0, wxALL, 5 );


	gSizer10->Add( bSizer216, 1, wxEXPAND, 5 );


	bSizer200->Add( gSizer10, 0, wxEXPAND, 5 );

	wxGridSizer* gSizer11;
	gSizer11 = new wxGridSizer( 2, 1, 0, 0 );

	wxBoxSizer* bSizer214;
	bSizer214 = new wxBoxSizer( wxHORIZONTAL );


	gSizer11->Add( bSizer214, 1, wxEXPAND, 5 );


	bSizer200->Add( gSizer11, 0, wxEXPAND, 5 );


	bSizer200->Add( 0, 0, 1, wxEXPAND, 5 );

	ExpertToggleButton = new wxToggleButton( InputParamsPanel, wxID_ANY, wxT("Show Expert Options"), wxDefaultPosition, wxDefaultSize, 0 );
	bSizer200->Add( ExpertToggleButton, 0, wxALIGN_BOTTOM|wxALL, 5 );


	bSizer441->Add( bSizer200, 1, wxEXPAND, 5 );

	PleaseCreateRefinementPackageText = new wxStaticText( InputParamsPanel, wxID_ANY, wxT("Please create a refinement package (in the assets panel) in order to perform a 2D classification."), wxDefaultPosition, wxDefaultSize, 0 );
	PleaseCreateRefinementPackageText->Wrap( -1 );
	PleaseCreateRefinementPackageText->SetFont( wxFont( wxNORMAL_FONT->GetPointSize(), wxFONTFAMILY_DEFAULT, wxFONTSTYLE_NORMAL, wxFONTWEIGHT_BOLD, false, wxEmptyString ) );
	PleaseCreateRefinementPackageText->SetForegroundColour( wxColour( 180, 0, 0 ) );
	PleaseCreateRefinementPackageText->Hide();

	bSizer441->Add( PleaseCreateRefinementPackageText, 0, wxALL, 5 );

	m_staticline10 = new wxStaticLine( InputParamsPanel, wxID_ANY, wxDefaultPosition, wxDefaultSize, wxLI_HORIZONTAL );
	bSizer441->Add( m_staticline10, 0, wxEXPAND | wxALL, 5 );


	bSizer451->Add( bSizer441, 1, wxEXPAND, 5 );


	InputParamsPanel->SetSizer( bSizer451 );
	InputParamsPanel->Layout();
	bSizer451->Fit( InputParamsPanel );
	bSizer43->Add( InputParamsPanel, 0, wxEXPAND | wxALL, 5 );

	wxBoxSizer* bSizer46;
	bSizer46 = new wxBoxSizer( wxHORIZONTAL );

	wxBoxSizer* bSizer279;
	bSizer279 = new wxBoxSizer( wxVERTICAL );

	OutputTextPanel = new wxPanel( this, wxID_ANY, wxDefaultPosition, wxDefaultSize, wxTAB_TRAVERSAL );
	OutputTextPanel->Hide();

	wxBoxSizer* bSizer56;
	bSizer56 = new wxBoxSizer( wxVERTICAL );

	output_textctrl = new wxTextCtrl( OutputTextPanel, wxID_ANY, wxEmptyString, wxDefaultPosition, wxDefaultSize, wxTE_MULTILINE|wxTE_READONLY );
	bSizer56->Add( output_textctrl, 1, wxALL|wxEXPAND, 5 );


	OutputTextPanel->SetSizer( bSizer56 );
	OutputTextPanel->Layout();
	bSizer56->Fit( OutputTextPanel );
	bSizer279->Add( OutputTextPanel, 40, wxEXPAND | wxALL, 5 );

	PlotPanel = new ClassificationPlotPanel( this, wxID_ANY, wxDefaultPosition, wxDefaultSize, wxTAB_TRAVERSAL );
	PlotPanel->Hide();

	bSizer279->Add( PlotPanel, 60, wxEXPAND | wxALL, 5 );


	bSizer46->Add( bSizer279, 40, wxEXPAND, 5 );

	ResultDisplayPanel = new DisplayPanel( this, wxID_ANY, wxDefaultPosition, wxDefaultSize, wxTAB_TRAVERSAL );
	ResultDisplayPanel->Hide();

	bSizer46->Add( ResultDisplayPanel, 80, wxEXPAND | wxALL, 5 );

	ExpertPanel = new wxScrolledWindow( this, wxID_ANY, wxDefaultPosition, wxSize( -1,-1 ), wxVSCROLL );
	ExpertPanel->SetScrollRate( 5, 5 );
	ExpertPanel->Hide();

	InputSizer = new wxBoxSizer( wxVERTICAL );

	wxBoxSizer* bSizer258;
	bSizer258 = new wxBoxSizer( wxHORIZONTAL );

	m_staticText318 = new wxStaticText( ExpertPanel, wxID_ANY, wxT("Expert Options"), wxDefaultPosition, wxDefaultSize, 0 );
	m_staticText318->Wrap( -1 );
	m_staticText318->SetFont( wxFont( 10, wxFONTFAMILY_SWISS, wxFONTSTYLE_NORMAL, wxFONTWEIGHT_BOLD, true, wxT("Sans") ) );

	bSizer258->Add( m_staticText318, 0, wxALIGN_CENTER_VERTICAL|wxALL, 5 );


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

	NoMovieFramesStaticText = new wxStaticText( ExpertPanel, wxID_ANY, wxT("Low-Resolution Limit (Å) :"), wxDefaultPosition, wxDefaultSize, 0 );
	NoMovieFramesStaticText->Wrap( -1 );
	fgSizer1->Add( NoMovieFramesStaticText, 0, wxALIGN_CENTER_VERTICAL|wxALL, 5 );

	LowResolutionLimitTextCtrl = new NumericTextCtrl( ExpertPanel, wxID_ANY, wxT("300.00"), wxDefaultPosition, wxDefaultSize, 0 );
	fgSizer1->Add( LowResolutionLimitTextCtrl, 0, wxALL|wxEXPAND, 5 );

	m_staticText188 = new wxStaticText( ExpertPanel, wxID_ANY, wxT("High-Res. Limit (start)  (Å) : "), wxDefaultPosition, wxDefaultSize, 0 );
	m_staticText188->Wrap( -1 );
	fgSizer1->Add( m_staticText188, 0, wxALIGN_CENTER_VERTICAL|wxALL, 5 );

	HighResolutionLimitStartTextCtrl = new NumericTextCtrl( ExpertPanel, wxID_ANY, wxT("40.00"), wxDefaultPosition, wxDefaultSize, 0 );
	fgSizer1->Add( HighResolutionLimitStartTextCtrl, 1, wxALL|wxEXPAND, 5 );

	m_staticText1881 = new wxStaticText( ExpertPanel, wxID_ANY, wxT("High-Res. Limit (finish) (Å) : "), wxDefaultPosition, wxDefaultSize, 0 );
	m_staticText1881->Wrap( -1 );
	fgSizer1->Add( m_staticText1881, 0, wxALL, 5 );

	HighResolutionLimitFinishTextCtrl = new NumericTextCtrl( ExpertPanel, wxID_ANY, wxT("8.00"), wxDefaultPosition, wxDefaultSize, 0 );
	fgSizer1->Add( HighResolutionLimitFinishTextCtrl, 0, wxALL|wxEXPAND, 5 );

	m_staticText196 = new wxStaticText( ExpertPanel, wxID_ANY, wxT("Mask Radius (Å) :"), wxDefaultPosition, wxDefaultSize, 0 );
	m_staticText196->Wrap( -1 );
	fgSizer1->Add( m_staticText196, 0, wxALIGN_CENTER_VERTICAL|wxALL, 5 );

	MaskRadiusTextCtrl = new NumericTextCtrl( ExpertPanel, wxID_ANY, wxT("100.00"), wxDefaultPosition, wxDefaultSize, wxTE_PROCESS_ENTER );
	fgSizer1->Add( MaskRadiusTextCtrl, 0, wxALL|wxEXPAND, 5 );

	AngularStepStaticText = new wxStaticText( ExpertPanel, wxID_ANY, wxT("Angular Search Step (°) :"), wxDefaultPosition, wxDefaultSize, 0 );
	AngularStepStaticText->Wrap( -1 );
	fgSizer1->Add( AngularStepStaticText, 0, wxALIGN_CENTER_VERTICAL|wxALL, 5 );

	AngularStepTextCtrl = new NumericTextCtrl( ExpertPanel, wxID_ANY, wxT("15.00"), wxDefaultPosition, wxDefaultSize, wxTE_PROCESS_ENTER );
	fgSizer1->Add( AngularStepTextCtrl, 0, wxALL|wxEXPAND, 5 );

	SearchRangeXStaticText = new wxStaticText( ExpertPanel, wxID_ANY, wxT("Max Search Range (Å) :"), wxDefaultPosition, wxDefaultSize, 0 );
	SearchRangeXStaticText->Wrap( -1 );
	fgSizer1->Add( SearchRangeXStaticText, 0, wxALIGN_CENTER_VERTICAL|wxALL, 5 );

	MaxSearchRangeTextCtrl = new NumericTextCtrl( ExpertPanel, wxID_ANY, wxT("100.00"), wxDefaultPosition, wxDefaultSize, wxTE_PROCESS_ENTER );
	fgSizer1->Add( MaxSearchRangeTextCtrl, 0, wxALL|wxEXPAND, 5 );

	m_staticText330 = new wxStaticText( ExpertPanel, wxID_ANY, wxT("Smoothing Factor :"), wxDefaultPosition, wxDefaultSize, 0 );
	m_staticText330->Wrap( -1 );
	fgSizer1->Add( m_staticText330, 0, wxALIGN_CENTER_VERTICAL|wxALL, 5 );

	SmoothingFactorTextCtrl = new NumericTextCtrl( ExpertPanel, wxID_ANY, wxT("1"), wxDefaultPosition, wxDefaultSize, 0 );
	fgSizer1->Add( SmoothingFactorTextCtrl, 0, wxALL|wxEXPAND, 5 );

	PhaseShiftStepStaticText = new wxStaticText( ExpertPanel, wxID_ANY, wxT("Exclude Blank Edges?"), wxDefaultPosition, wxDefaultSize, 0 );
	PhaseShiftStepStaticText->Wrap( -1 );
	fgSizer1->Add( PhaseShiftStepStaticText, 0, wxALIGN_CENTER_VERTICAL|wxALL, 5 );

	wxBoxSizer* bSizer263;
	bSizer263 = new wxBoxSizer( wxHORIZONTAL );

	ExcludeBlankEdgesYesRadio = new wxRadioButton( ExpertPanel, wxID_ANY, wxT("Yes"), wxDefaultPosition, wxDefaultSize, wxRB_GROUP );
	bSizer263->Add( ExcludeBlankEdgesYesRadio, 0, wxALL, 5 );

	ExcludeBlankEdgesNoRadio = new wxRadioButton( ExpertPanel, wxID_ANY, wxT("No"), wxDefaultPosition, wxDefaultSize, 0 );
	bSizer263->Add( ExcludeBlankEdgesNoRadio, 0, wxALL, 5 );


	fgSizer1->Add( bSizer263, 1, wxEXPAND, 5 );

	m_staticText334 = new wxStaticText( ExpertPanel, wxID_ANY, wxT("Auto Percent Used?"), wxDefaultPosition, wxDefaultSize, 0 );
	m_staticText334->Wrap( -1 );
	fgSizer1->Add( m_staticText334, 0, wxALIGN_CENTER_VERTICAL|wxALL, 5 );

	wxBoxSizer* bSizer2631;
	bSizer2631 = new wxBoxSizer( wxHORIZONTAL );

	AutoPercentUsedRadioYes = new wxRadioButton( ExpertPanel, wxID_ANY, wxT("Yes"), wxDefaultPosition, wxDefaultSize, wxRB_GROUP );
	bSizer2631->Add( AutoPercentUsedRadioYes, 0, wxALL, 5 );

	SpherAutoPercentUsedRadioNo = new wxRadioButton( ExpertPanel, wxID_ANY, wxT("No"), wxDefaultPosition, wxDefaultSize, 0 );
	bSizer2631->Add( SpherAutoPercentUsedRadioNo, 0, wxALL, 5 );


	fgSizer1->Add( bSizer2631, 1, wxEXPAND, 5 );

	PercentUsedStaticText = new wxStaticText( ExpertPanel, wxID_ANY, wxT("\tPercent Used :"), wxDefaultPosition, wxDefaultSize, 0 );
	PercentUsedStaticText->Wrap( -1 );
	PercentUsedStaticText->Enable( false );

	fgSizer1->Add( PercentUsedStaticText, 0, wxALIGN_CENTER_VERTICAL|wxALL, 5 );

	PercentUsedTextCtrl = new NumericTextCtrl( ExpertPanel, wxID_ANY, wxT("100.00"), wxDefaultPosition, wxDefaultSize, 0 );
	PercentUsedTextCtrl->Enable( false );

	fgSizer1->Add( PercentUsedTextCtrl, 0, wxALL, 5 );

	m_staticText650 = new wxStaticText( ExpertPanel, wxID_ANY, wxT("AutoMask Averages?"), wxDefaultPosition, wxDefaultSize, 0 );
	m_staticText650->Wrap( -1 );
	fgSizer1->Add( m_staticText650, 0, wxALIGN_CENTER_VERTICAL|wxALL, 5 );

	wxBoxSizer* bSizer26311;
	bSizer26311 = new wxBoxSizer( wxHORIZONTAL );

	AutoMaskRadioYes = new wxRadioButton( ExpertPanel, wxID_ANY, wxT("Yes"), wxDefaultPosition, wxDefaultSize, wxRB_GROUP );
	bSizer26311->Add( AutoMaskRadioYes, 0, wxALL, 5 );

	AutoMaskRadioNo = new wxRadioButton( ExpertPanel, wxID_ANY, wxT("No"), wxDefaultPosition, wxDefaultSize, 0 );
	bSizer26311->Add( AutoMaskRadioNo, 0, wxALL, 5 );


	fgSizer1->Add( bSizer26311, 1, wxEXPAND, 5 );

	m_staticText651 = new wxStaticText( ExpertPanel, wxID_ANY, wxT("Auto Centre Averages?"), wxDefaultPosition, wxDefaultSize, 0 );
	m_staticText651->Wrap( -1 );
	fgSizer1->Add( m_staticText651, 0, wxALIGN_CENTER_VERTICAL|wxALL, 5 );

	wxBoxSizer* bSizer26312;
	bSizer26312 = new wxBoxSizer( wxHORIZONTAL );

	AutoCentreRadioYes = new wxRadioButton( ExpertPanel, wxID_ANY, wxT("Yes"), wxDefaultPosition, wxDefaultSize, wxRB_GROUP );
	bSizer26312->Add( AutoCentreRadioYes, 0, wxALL, 5 );

	AutoCentreRadioNo = new wxRadioButton( ExpertPanel, wxID_ANY, wxT("No"), wxDefaultPosition, wxDefaultSize, 0 );
	bSizer26312->Add( AutoCentreRadioNo, 0, wxALL, 5 );


	fgSizer1->Add( bSizer26312, 1, wxEXPAND, 5 );


	InputSizer->Add( fgSizer1, 1, wxEXPAND, 5 );


	ExpertPanel->SetSizer( InputSizer );
	ExpertPanel->Layout();
	InputSizer->Fit( ExpertPanel );
	bSizer46->Add( ExpertPanel, 0, wxALL|wxEXPAND, 5 );

	InfoPanel = new wxPanel( this, wxID_ANY, wxDefaultPosition, wxDefaultSize, wxTAB_TRAVERSAL );
	wxBoxSizer* bSizer61;
	bSizer61 = new wxBoxSizer( wxVERTICAL );

	InfoText = new wxRichTextCtrl( InfoPanel, wxID_ANY, wxEmptyString, wxDefaultPosition, wxDefaultSize, wxTE_READONLY|wxHSCROLL|wxVSCROLL );
	bSizer61->Add( InfoText, 1, wxEXPAND | wxALL, 5 );


	InfoPanel->SetSizer( bSizer61 );
	InfoPanel->Layout();
	bSizer61->Fit( InfoPanel );
	bSizer46->Add( InfoPanel, 20, wxEXPAND | wxALL, 5 );


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
	bSizer48->Add( ProgressPanel, 1, wxEXPAND | wxALL, 5 );

	StartPanel = new wxPanel( this, wxID_ANY, wxDefaultPosition, wxDefaultSize, wxTAB_TRAVERSAL );
	wxBoxSizer* bSizer58;
	bSizer58 = new wxBoxSizer( wxHORIZONTAL );

	wxBoxSizer* bSizer268;
	bSizer268 = new wxBoxSizer( wxVERTICAL );

	wxBoxSizer* bSizer267;
	bSizer267 = new wxBoxSizer( wxHORIZONTAL );

	RunProfileText = new wxStaticText( StartPanel, wxID_ANY, wxT("Classification Run Profile :"), wxDefaultPosition, wxDefaultSize, 0 );
	RunProfileText->Wrap( -1 );
	bSizer267->Add( RunProfileText, 20, wxALIGN_CENTER_VERTICAL|wxALL, 5 );

	RefinementRunProfileComboBox = new MemoryComboBox( StartPanel, wxID_ANY, wxEmptyString, wxDefaultPosition, wxDefaultSize, 0, NULL, wxCB_READONLY );
	bSizer267->Add( RefinementRunProfileComboBox, 50, wxALIGN_CENTER_VERTICAL|wxALL|wxEXPAND, 5 );


	bSizer268->Add( bSizer267, 50, wxEXPAND, 5 );

	wxBoxSizer* bSizer2671;
	bSizer2671 = new wxBoxSizer( wxHORIZONTAL );


	bSizer268->Add( bSizer2671, 0, wxEXPAND, 5 );


	bSizer58->Add( bSizer268, 50, wxEXPAND, 5 );

	wxBoxSizer* bSizer60;
	bSizer60 = new wxBoxSizer( wxVERTICAL );


	bSizer60->Add( 0, 0, 1, wxEXPAND, 5 );

	StartRefinementButton = new wxButton( StartPanel, wxID_ANY, wxT("Start Classification"), wxDefaultPosition, wxDefaultSize, 0 );
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
	this->Connect( wxEVT_UPDATE_UI, wxUpdateUIEventHandler( Refine2DPanel::OnUpdateUI ) );
	ExpertToggleButton->Connect( wxEVT_COMMAND_TOGGLEBUTTON_CLICKED, wxCommandEventHandler( Refine2DPanel::OnExpertOptionsToggle ), NULL, this );
	ResetAllDefaultsButton->Connect( wxEVT_COMMAND_BUTTON_CLICKED, wxCommandEventHandler( Refine2DPanel::ResetAllDefaultsClick ), NULL, this );
	HighResolutionLimitStartTextCtrl->Connect( wxEVT_COMMAND_TEXT_UPDATED, wxCommandEventHandler( Refine2DPanel::OnHighResLimitChange ), NULL, this );
	HighResolutionLimitFinishTextCtrl->Connect( wxEVT_COMMAND_TEXT_UPDATED, wxCommandEventHandler( Refine2DPanel::OnHighResLimitChange ), NULL, this );
	InfoText->Connect( wxEVT_COMMAND_TEXT_URL, wxTextUrlEventHandler( Refine2DPanel::OnInfoURL ), NULL, this );
	FinishButton->Connect( wxEVT_COMMAND_BUTTON_CLICKED, wxCommandEventHandler( Refine2DPanel::FinishButtonClick ), NULL, this );
	CancelAlignmentButton->Connect( wxEVT_COMMAND_BUTTON_CLICKED, wxCommandEventHandler( Refine2DPanel::TerminateButtonClick ), NULL, this );
	StartRefinementButton->Connect( wxEVT_COMMAND_BUTTON_CLICKED, wxCommandEventHandler( Refine2DPanel::StartClassificationClick ), NULL, this );
}

Refine2DPanel::~Refine2DPanel()
{
	// Disconnect Events
	this->Disconnect( wxEVT_UPDATE_UI, wxUpdateUIEventHandler( Refine2DPanel::OnUpdateUI ) );
	ExpertToggleButton->Disconnect( wxEVT_COMMAND_TOGGLEBUTTON_CLICKED, wxCommandEventHandler( Refine2DPanel::OnExpertOptionsToggle ), NULL, this );
	ResetAllDefaultsButton->Disconnect( wxEVT_COMMAND_BUTTON_CLICKED, wxCommandEventHandler( Refine2DPanel::ResetAllDefaultsClick ), NULL, this );
	HighResolutionLimitStartTextCtrl->Disconnect( wxEVT_COMMAND_TEXT_UPDATED, wxCommandEventHandler( Refine2DPanel::OnHighResLimitChange ), NULL, this );
	HighResolutionLimitFinishTextCtrl->Disconnect( wxEVT_COMMAND_TEXT_UPDATED, wxCommandEventHandler( Refine2DPanel::OnHighResLimitChange ), NULL, this );
	InfoText->Disconnect( wxEVT_COMMAND_TEXT_URL, wxTextUrlEventHandler( Refine2DPanel::OnInfoURL ), NULL, this );
	FinishButton->Disconnect( wxEVT_COMMAND_BUTTON_CLICKED, wxCommandEventHandler( Refine2DPanel::FinishButtonClick ), NULL, this );
	CancelAlignmentButton->Disconnect( wxEVT_COMMAND_BUTTON_CLICKED, wxCommandEventHandler( Refine2DPanel::TerminateButtonClick ), NULL, this );
	StartRefinementButton->Disconnect( wxEVT_COMMAND_BUTTON_CLICKED, wxCommandEventHandler( Refine2DPanel::StartClassificationClick ), NULL, this );

}

Refine2DResultsPanelParent::Refine2DResultsPanelParent( wxWindow* parent, wxWindowID id, const wxPoint& pos, const wxSize& size, long style, const wxString& name ) : wxPanel( parent, id, pos, size, style, name )
{
	wxBoxSizer* bSizer202;
	bSizer202 = new wxBoxSizer( wxVERTICAL );

	wxBoxSizer* bSizer265;
	bSizer265 = new wxBoxSizer( wxVERTICAL );

	m_staticline67 = new wxStaticLine( this, wxID_ANY, wxDefaultPosition, wxDefaultSize, wxLI_HORIZONTAL );
	bSizer265->Add( m_staticline67, 0, wxEXPAND | wxALL, 5 );


	bSizer202->Add( bSizer265, 0, wxEXPAND, 5 );

	m_splitter7 = new wxSplitterWindow( this, wxID_ANY, wxDefaultPosition, wxDefaultSize, wxSP_3D|wxSP_3DBORDER|wxSP_3DSASH|wxSP_BORDER );
	m_splitter7->SetSashGravity( 0.5 );
	m_splitter7->Connect( wxEVT_IDLE, wxIdleEventHandler( Refine2DResultsPanelParent::m_splitter7OnIdle ), NULL, this );

	LeftPanel = new wxPanel( m_splitter7, wxID_ANY, wxDefaultPosition, wxDefaultSize, wxTAB_TRAVERSAL );
	wxBoxSizer* bSizer203;
	bSizer203 = new wxBoxSizer( wxVERTICAL );

	wxFlexGridSizer* fgSizer11;
	fgSizer11 = new wxFlexGridSizer( 0, 2, 0, 0 );
	fgSizer11->AddGrowableCol( 0 );
	fgSizer11->SetFlexibleDirection( wxBOTH );
	fgSizer11->SetNonFlexibleGrowMode( wxFLEX_GROWMODE_SPECIFIED );

	wxFlexGridSizer* fgSizer16;
	fgSizer16 = new wxFlexGridSizer( 0, 2, 0, 0 );
	fgSizer16->AddGrowableCol( 1 );
	fgSizer16->SetFlexibleDirection( wxBOTH );
	fgSizer16->SetNonFlexibleGrowMode( wxFLEX_GROWMODE_SPECIFIED );

	m_staticText284 = new wxStaticText( LeftPanel, wxID_ANY, wxT("Select Refinement Package :"), wxDefaultPosition, wxDefaultSize, 0 );
	m_staticText284->Wrap( -1 );
	fgSizer16->Add( m_staticText284, 0, wxALIGN_CENTER_VERTICAL|wxALIGN_RIGHT|wxALL, 5 );

	RefinementPackageComboBox = new RefinementPackagePickerComboPanel( LeftPanel, wxID_ANY, wxDefaultPosition, wxDefaultSize, wxTAB_TRAVERSAL );
	RefinementPackageComboBox->SetMinSize( wxSize( 350,-1 ) );
	RefinementPackageComboBox->SetMaxSize( wxSize( 350,-1 ) );

	fgSizer16->Add( RefinementPackageComboBox, 0, wxALIGN_CENTER_VERTICAL|wxALL, 5 );

	m_staticText285 = new wxStaticText( LeftPanel, wxID_ANY, wxT("Select Classification :"), wxDefaultPosition, wxDefaultSize, 0 );
	m_staticText285->Wrap( -1 );
	fgSizer16->Add( m_staticText285, 0, wxALIGN_CENTER_VERTICAL|wxALIGN_RIGHT|wxALL, 5 );

	InputParametersComboBox = new ClassificationPickerComboPanel( LeftPanel, wxID_ANY, wxDefaultPosition, wxDefaultSize, wxTAB_TRAVERSAL );
	InputParametersComboBox->SetMinSize( wxSize( 350,-1 ) );
	InputParametersComboBox->SetMaxSize( wxSize( 350,-1 ) );

	fgSizer16->Add( InputParametersComboBox, 0, wxALIGN_CENTER_VERTICAL|wxALL, 5 );


	fgSizer11->Add( fgSizer16, 0, wxEXPAND, 5 );


	bSizer203->Add( fgSizer11, 0, wxALIGN_CENTER_VERTICAL|wxEXPAND, 5 );

	m_staticline66 = new wxStaticLine( LeftPanel, wxID_ANY, wxDefaultPosition, wxDefaultSize, wxLI_HORIZONTAL );
	bSizer203->Add( m_staticline66, 0, wxEXPAND | wxALL, 5 );

	JobDetailsPanel = new wxPanel( LeftPanel, wxID_ANY, wxDefaultPosition, wxDefaultSize, wxTAB_TRAVERSAL );
	JobDetailsPanel->Hide();

	wxBoxSizer* bSizer270;
	bSizer270 = new wxBoxSizer( wxVERTICAL );

	wxBoxSizer* bSizer101;
	bSizer101 = new wxBoxSizer( wxVERTICAL );

	InfoSizer = new wxFlexGridSizer( 0, 6, 0, 0 );
	InfoSizer->SetFlexibleDirection( wxBOTH );
	InfoSizer->SetNonFlexibleGrowMode( wxFLEX_GROWMODE_SPECIFIED );

	m_staticText72 = new wxStaticText( JobDetailsPanel, wxID_ANY, wxT("Classification ID :"), wxDefaultPosition, wxDefaultSize, 0 );
	m_staticText72->Wrap( -1 );
	m_staticText72->SetFont( wxFont( 10, wxFONTFAMILY_SWISS, wxFONTSTYLE_NORMAL, wxFONTWEIGHT_BOLD, false, wxT("Sans") ) );

	InfoSizer->Add( m_staticText72, 0, wxALIGN_RIGHT|wxALL, 5 );

	ClassificationIDStaticText = new wxStaticText( JobDetailsPanel, wxID_ANY, wxEmptyString, wxDefaultPosition, wxDefaultSize, 0 );
	ClassificationIDStaticText->Wrap( -1 );
	InfoSizer->Add( ClassificationIDStaticText, 0, wxALIGN_LEFT|wxALL, 5 );

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

	m_staticText83 = new wxStaticText( JobDetailsPanel, wxID_ANY, wxT("Refinement Package ID :"), wxDefaultPosition, wxDefaultSize, 0 );
	m_staticText83->Wrap( -1 );
	m_staticText83->SetFont( wxFont( 10, wxFONTFAMILY_SWISS, wxFONTSTYLE_NORMAL, wxFONTWEIGHT_BOLD, false, wxT("Sans") ) );

	InfoSizer->Add( m_staticText83, 0, wxALIGN_RIGHT|wxALL, 5 );

	RefinementPackageIDStaticText = new wxStaticText( JobDetailsPanel, wxID_ANY, wxEmptyString, wxDefaultPosition, wxDefaultSize, 0 );
	RefinementPackageIDStaticText->Wrap( -1 );
	InfoSizer->Add( RefinementPackageIDStaticText, 0, wxALIGN_LEFT|wxALL, 5 );

	m_staticText82 = new wxStaticText( JobDetailsPanel, wxID_ANY, wxT("Ref. Classification ID :"), wxDefaultPosition, wxDefaultSize, 0 );
	m_staticText82->Wrap( -1 );
	m_staticText82->SetFont( wxFont( 10, wxFONTFAMILY_SWISS, wxFONTSTYLE_NORMAL, wxFONTWEIGHT_BOLD, false, wxT("Sans") ) );

	InfoSizer->Add( m_staticText82, 0, wxALIGN_RIGHT|wxALL, 5 );

	StartClassificationIDStaticText = new wxStaticText( JobDetailsPanel, wxID_ANY, wxEmptyString, wxDefaultPosition, wxDefaultSize, 0 );
	StartClassificationIDStaticText->Wrap( -1 );
	InfoSizer->Add( StartClassificationIDStaticText, 0, wxALIGN_LEFT|wxALL, 5 );

	m_staticText78 = new wxStaticText( JobDetailsPanel, wxID_ANY, wxT("No. Classes :"), wxDefaultPosition, wxDefaultSize, 0 );
	m_staticText78->Wrap( -1 );
	m_staticText78->SetFont( wxFont( 10, wxFONTFAMILY_SWISS, wxFONTSTYLE_NORMAL, wxFONTWEIGHT_BOLD, false, wxT("Sans") ) );

	InfoSizer->Add( m_staticText78, 0, wxALIGN_RIGHT|wxALL, 5 );

	NumberClassesStaticText = new wxStaticText( JobDetailsPanel, wxID_ANY, wxEmptyString, wxDefaultPosition, wxDefaultSize, 0 );
	NumberClassesStaticText->Wrap( -1 );
	InfoSizer->Add( NumberClassesStaticText, 0, wxALIGN_LEFT|wxALL, 5 );

	m_staticText96 = new wxStaticText( JobDetailsPanel, wxID_ANY, wxT("No. Input Particles :"), wxDefaultPosition, wxDefaultSize, 0 );
	m_staticText96->Wrap( -1 );
	m_staticText96->SetFont( wxFont( 10, wxFONTFAMILY_SWISS, wxFONTSTYLE_NORMAL, wxFONTWEIGHT_BOLD, false, wxT("Sans") ) );

	InfoSizer->Add( m_staticText96, 0, wxALIGN_RIGHT|wxALL, 5 );

	NumberParticlesStaticText = new wxStaticText( JobDetailsPanel, wxID_ANY, wxEmptyString, wxDefaultPosition, wxDefaultSize, 0 );
	NumberParticlesStaticText->Wrap( -1 );
	InfoSizer->Add( NumberParticlesStaticText, 0, wxALIGN_LEFT|wxALL, 5 );

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

	m_staticText95 = new wxStaticText( JobDetailsPanel, wxID_ANY, wxT("Smoothing Factor :"), wxDefaultPosition, wxDefaultSize, 0 );
	m_staticText95->Wrap( -1 );
	m_staticText95->SetFont( wxFont( 10, wxFONTFAMILY_SWISS, wxFONTSTYLE_NORMAL, wxFONTWEIGHT_BOLD, false, wxT("Sans") ) );

	InfoSizer->Add( m_staticText95, 0, wxALIGN_RIGHT|wxALL, 5 );

	SmoothingFactorStaticText = new wxStaticText( JobDetailsPanel, wxID_ANY, wxEmptyString, wxDefaultPosition, wxDefaultSize, 0 );
	SmoothingFactorStaticText->Wrap( -1 );
	InfoSizer->Add( SmoothingFactorStaticText, 0, wxALIGN_LEFT|wxALL, 5 );

	LargeAstigExpectedLabel = new wxStaticText( JobDetailsPanel, wxID_ANY, wxT("Exclude Blank Edges? :"), wxDefaultPosition, wxDefaultSize, 0 );
	LargeAstigExpectedLabel->Wrap( -1 );
	LargeAstigExpectedLabel->SetFont( wxFont( 10, wxFONTFAMILY_SWISS, wxFONTSTYLE_NORMAL, wxFONTWEIGHT_BOLD, false, wxT("Sans") ) );

	InfoSizer->Add( LargeAstigExpectedLabel, 0, wxALIGN_RIGHT|wxALL, 5 );

	ExcludeBlankEdgesStaticText = new wxStaticText( JobDetailsPanel, wxID_ANY, wxEmptyString, wxDefaultPosition, wxDefaultSize, 0 );
	ExcludeBlankEdgesStaticText->Wrap( -1 );
	InfoSizer->Add( ExcludeBlankEdgesStaticText, 0, wxALL, 5 );

	m_staticText99 = new wxStaticText( JobDetailsPanel, wxID_ANY, wxT("Search Range Y :"), wxDefaultPosition, wxDefaultSize, 0 );
	m_staticText99->Wrap( -1 );
	m_staticText99->SetFont( wxFont( 10, wxFONTFAMILY_SWISS, wxFONTSTYLE_NORMAL, wxFONTWEIGHT_BOLD, false, wxT("Sans") ) );

	InfoSizer->Add( m_staticText99, 0, wxALIGN_RIGHT|wxALL, 5 );

	SearchRangeYStaticText = new wxStaticText( JobDetailsPanel, wxID_ANY, wxEmptyString, wxDefaultPosition, wxDefaultSize, 0 );
	SearchRangeYStaticText->Wrap( -1 );
	InfoSizer->Add( SearchRangeYStaticText, 0, wxALIGN_LEFT|wxALL, 5 );

	ToleratedAstigLabel = new wxStaticText( JobDetailsPanel, wxID_ANY, wxT("Auto Percent Used? :"), wxDefaultPosition, wxDefaultSize, 0 );
	ToleratedAstigLabel->Wrap( -1 );
	ToleratedAstigLabel->SetFont( wxFont( 10, wxFONTFAMILY_SWISS, wxFONTSTYLE_NORMAL, wxFONTWEIGHT_BOLD, false, wxT("Sans") ) );

	InfoSizer->Add( ToleratedAstigLabel, 0, wxALIGN_RIGHT|wxALL, 5 );

	AutoPercentUsedStaticText = new wxStaticText( JobDetailsPanel, wxID_ANY, wxEmptyString, wxDefaultPosition, wxDefaultSize, 0 );
	AutoPercentUsedStaticText->Wrap( -1 );
	InfoSizer->Add( AutoPercentUsedStaticText, 0, wxALIGN_LEFT|wxALL, 5 );

	NumberOfAveragedFramesLabel = new wxStaticText( JobDetailsPanel, wxID_ANY, wxT("Percent Used :"), wxDefaultPosition, wxDefaultSize, 0 );
	NumberOfAveragedFramesLabel->Wrap( -1 );
	NumberOfAveragedFramesLabel->SetFont( wxFont( 10, wxFONTFAMILY_SWISS, wxFONTSTYLE_NORMAL, wxFONTWEIGHT_BOLD, false, wxT("Sans") ) );

	InfoSizer->Add( NumberOfAveragedFramesLabel, 0, wxALIGN_RIGHT|wxALL, 5 );

	PercentUsedStaticText = new wxStaticText( JobDetailsPanel, wxID_ANY, wxEmptyString, wxDefaultPosition, wxDefaultSize, 0 );
	PercentUsedStaticText->Wrap( -1 );
	InfoSizer->Add( PercentUsedStaticText, 0, wxALL, 5 );


	bSizer101->Add( InfoSizer, 1, wxEXPAND, 5 );

	m_staticline30 = new wxStaticLine( JobDetailsPanel, wxID_ANY, wxDefaultPosition, wxDefaultSize, wxLI_HORIZONTAL );
	bSizer101->Add( m_staticline30, 0, wxEXPAND | wxALL, 5 );


	bSizer270->Add( bSizer101, 1, wxEXPAND, 5 );


	JobDetailsPanel->SetSizer( bSizer270 );
	JobDetailsPanel->Layout();
	bSizer270->Fit( JobDetailsPanel );
	bSizer203->Add( JobDetailsPanel, 0, wxEXPAND | wxALL, 5 );

	ClassumDisplayPanel = new DisplayPanel( LeftPanel, wxID_ANY, wxDefaultPosition, wxDefaultSize, wxTAB_TRAVERSAL );
	bSizer203->Add( ClassumDisplayPanel, 1, wxEXPAND | wxALL, 5 );


	LeftPanel->SetSizer( bSizer203 );
	LeftPanel->Layout();
	bSizer203->Fit( LeftPanel );
	m_panel49 = new wxPanel( m_splitter7, wxID_ANY, wxDefaultPosition, wxDefaultSize, wxTAB_TRAVERSAL );
	wxBoxSizer* bSizer206;
	bSizer206 = new wxBoxSizer( wxVERTICAL );

	wxBoxSizer* bSizer269;
	bSizer269 = new wxBoxSizer( wxHORIZONTAL );

	m_staticText321 = new wxStaticText( m_panel49, wxID_ANY, wxT("Manage class average selections"), wxDefaultPosition, wxDefaultSize, 0 );
	m_staticText321->Wrap( -1 );
	m_staticText321->SetFont( wxFont( 10, wxFONTFAMILY_SWISS, wxFONTSTYLE_NORMAL, wxFONTWEIGHT_BOLD, false, wxT("Sans") ) );

	bSizer269->Add( m_staticText321, 0, wxALIGN_BOTTOM|wxALL, 5 );


	bSizer269->Add( 0, 0, 1, wxEXPAND, 5 );

	JobDetailsToggleButton = new wxToggleButton( m_panel49, wxID_ANY, wxT("Show Job Details"), wxDefaultPosition, wxDefaultSize, 0 );
	bSizer269->Add( JobDetailsToggleButton, 0, wxALL, 5 );


	bSizer206->Add( bSizer269, 1, wxEXPAND, 5 );

	m_staticline671 = new wxStaticLine( m_panel49, wxID_ANY, wxDefaultPosition, wxDefaultSize, wxLI_HORIZONTAL );
	bSizer206->Add( m_staticline671, 0, wxEXPAND | wxALL, 5 );

	SelectionPanel = new wxPanel( m_panel49, wxID_ANY, wxDefaultPosition, wxDefaultSize, wxTAB_TRAVERSAL );
	wxBoxSizer* bSizer271;
	bSizer271 = new wxBoxSizer( wxVERTICAL );

	wxBoxSizer* bSizer267;
	bSizer267 = new wxBoxSizer( wxVERTICAL );

	SelectionManagerListCtrl = new ClassificationSelectionListCtrl( SelectionPanel, wxID_ANY, wxDefaultPosition, wxDefaultSize, wxLC_REPORT|wxLC_SINGLE_SEL|wxLC_VIRTUAL );
	bSizer267->Add( SelectionManagerListCtrl, 0, wxALL|wxEXPAND, 5 );

	wxBoxSizer* bSizer268;
	bSizer268 = new wxBoxSizer( wxHORIZONTAL );

	AddButton = new wxButton( SelectionPanel, wxID_ANY, wxT("Add"), wxDefaultPosition, wxDefaultSize, 0 );
	bSizer268->Add( AddButton, 0, wxALL, 5 );

	DeleteButton = new wxButton( SelectionPanel, wxID_ANY, wxT("Delete"), wxDefaultPosition, wxDefaultSize, 0 );
	bSizer268->Add( DeleteButton, 0, wxALL, 5 );

	RenameButton = new wxButton( SelectionPanel, wxID_ANY, wxT("Rename"), wxDefaultPosition, wxDefaultSize, 0 );
	bSizer268->Add( RenameButton, 0, wxALL, 5 );

	CopyOtherButton = new wxButton( SelectionPanel, wxID_ANY, wxT("Copy Other"), wxDefaultPosition, wxDefaultSize, 0 );
	bSizer268->Add( CopyOtherButton, 0, wxALL, 5 );

	m_staticline64 = new wxStaticLine( SelectionPanel, wxID_ANY, wxDefaultPosition, wxDefaultSize, wxLI_VERTICAL );
	bSizer268->Add( m_staticline64, 0, wxEXPAND | wxALL, 5 );

	ClearButton = new wxButton( SelectionPanel, wxID_ANY, wxT("Clear"), wxDefaultPosition, wxDefaultSize, 0 );
	bSizer268->Add( ClearButton, 0, wxALL, 5 );

	InvertButton = new wxButton( SelectionPanel, wxID_ANY, wxT("Invert"), wxDefaultPosition, wxDefaultSize, 0 );
	bSizer268->Add( InvertButton, 0, wxALL, 5 );


	bSizer267->Add( bSizer268, 0, wxEXPAND, 5 );


	bSizer271->Add( bSizer267, 1, wxEXPAND, 5 );


	SelectionPanel->SetSizer( bSizer271 );
	SelectionPanel->Layout();
	bSizer271->Fit( SelectionPanel );
	bSizer206->Add( SelectionPanel, 0, wxEXPAND | wxALL, 5 );

	ClassNumberStaticText = new wxStaticText( m_panel49, wxID_ANY, wxT("Images for class #1"), wxDefaultPosition, wxDefaultSize, 0 );
	ClassNumberStaticText->Wrap( -1 );
	ClassNumberStaticText->SetFont( wxFont( 10, wxFONTFAMILY_SWISS, wxFONTSTYLE_NORMAL, wxFONTWEIGHT_BOLD, false, wxT("Sans") ) );

	bSizer206->Add( ClassNumberStaticText, 0, wxALIGN_LEFT|wxALL|wxBOTTOM, 5 );

	m_staticline65 = new wxStaticLine( m_panel49, wxID_ANY, wxDefaultPosition, wxDefaultSize, wxLI_HORIZONTAL );
	bSizer206->Add( m_staticline65, 0, wxEXPAND | wxALL, 5 );

	ParticleDisplayPanel = new DisplayPanel( m_panel49, wxID_ANY, wxDefaultPosition, wxDefaultSize, wxTAB_TRAVERSAL );
	bSizer206->Add( ParticleDisplayPanel, 90, wxEXPAND | wxALL, 5 );


	m_panel49->SetSizer( bSizer206 );
	m_panel49->Layout();
	bSizer206->Fit( m_panel49 );
	m_splitter7->SplitVertically( LeftPanel, m_panel49, -800 );
	bSizer202->Add( m_splitter7, 1, wxEXPAND, 5 );


	this->SetSizer( bSizer202 );
	this->Layout();

	// Connect Events
	this->Connect( wxEVT_UPDATE_UI, wxUpdateUIEventHandler( Refine2DResultsPanelParent::OnUpdateUI ) );
	JobDetailsToggleButton->Connect( wxEVT_COMMAND_TOGGLEBUTTON_CLICKED, wxCommandEventHandler( Refine2DResultsPanelParent::OnJobDetailsToggle ), NULL, this );
	SelectionManagerListCtrl->Connect( wxEVT_COMMAND_LIST_BEGIN_LABEL_EDIT, wxListEventHandler( Refine2DResultsPanelParent::OnBeginLabelEdit ), NULL, this );
	SelectionManagerListCtrl->Connect( wxEVT_COMMAND_LIST_END_LABEL_EDIT, wxListEventHandler( Refine2DResultsPanelParent::OnEndLabelEdit ), NULL, this );
	SelectionManagerListCtrl->Connect( wxEVT_COMMAND_LIST_ITEM_ACTIVATED, wxListEventHandler( Refine2DResultsPanelParent::OnActivated ), NULL, this );
	SelectionManagerListCtrl->Connect( wxEVT_COMMAND_LIST_ITEM_DESELECTED, wxListEventHandler( Refine2DResultsPanelParent::OnDeselected ), NULL, this );
	SelectionManagerListCtrl->Connect( wxEVT_COMMAND_LIST_ITEM_SELECTED, wxListEventHandler( Refine2DResultsPanelParent::OnSelected ), NULL, this );
	AddButton->Connect( wxEVT_COMMAND_BUTTON_CLICKED, wxCommandEventHandler( Refine2DResultsPanelParent::OnAddButtonClick ), NULL, this );
	DeleteButton->Connect( wxEVT_COMMAND_BUTTON_CLICKED, wxCommandEventHandler( Refine2DResultsPanelParent::OnDeleteButtonClick ), NULL, this );
	RenameButton->Connect( wxEVT_COMMAND_BUTTON_CLICKED, wxCommandEventHandler( Refine2DResultsPanelParent::OnRenameButtonClick ), NULL, this );
	CopyOtherButton->Connect( wxEVT_COMMAND_BUTTON_CLICKED, wxCommandEventHandler( Refine2DResultsPanelParent::OnCopyOtherButtonClick ), NULL, this );
	ClearButton->Connect( wxEVT_COMMAND_BUTTON_CLICKED, wxCommandEventHandler( Refine2DResultsPanelParent::OnClearButtonClick ), NULL, this );
	InvertButton->Connect( wxEVT_COMMAND_BUTTON_CLICKED, wxCommandEventHandler( Refine2DResultsPanelParent::OnInvertButtonClick ), NULL, this );
}

Refine2DResultsPanelParent::~Refine2DResultsPanelParent()
{
	// Disconnect Events
	this->Disconnect( wxEVT_UPDATE_UI, wxUpdateUIEventHandler( Refine2DResultsPanelParent::OnUpdateUI ) );
	JobDetailsToggleButton->Disconnect( wxEVT_COMMAND_TOGGLEBUTTON_CLICKED, wxCommandEventHandler( Refine2DResultsPanelParent::OnJobDetailsToggle ), NULL, this );
	SelectionManagerListCtrl->Disconnect( wxEVT_COMMAND_LIST_BEGIN_LABEL_EDIT, wxListEventHandler( Refine2DResultsPanelParent::OnBeginLabelEdit ), NULL, this );
	SelectionManagerListCtrl->Disconnect( wxEVT_COMMAND_LIST_END_LABEL_EDIT, wxListEventHandler( Refine2DResultsPanelParent::OnEndLabelEdit ), NULL, this );
	SelectionManagerListCtrl->Disconnect( wxEVT_COMMAND_LIST_ITEM_ACTIVATED, wxListEventHandler( Refine2DResultsPanelParent::OnActivated ), NULL, this );
	SelectionManagerListCtrl->Disconnect( wxEVT_COMMAND_LIST_ITEM_DESELECTED, wxListEventHandler( Refine2DResultsPanelParent::OnDeselected ), NULL, this );
	SelectionManagerListCtrl->Disconnect( wxEVT_COMMAND_LIST_ITEM_SELECTED, wxListEventHandler( Refine2DResultsPanelParent::OnSelected ), NULL, this );
	AddButton->Disconnect( wxEVT_COMMAND_BUTTON_CLICKED, wxCommandEventHandler( Refine2DResultsPanelParent::OnAddButtonClick ), NULL, this );
	DeleteButton->Disconnect( wxEVT_COMMAND_BUTTON_CLICKED, wxCommandEventHandler( Refine2DResultsPanelParent::OnDeleteButtonClick ), NULL, this );
	RenameButton->Disconnect( wxEVT_COMMAND_BUTTON_CLICKED, wxCommandEventHandler( Refine2DResultsPanelParent::OnRenameButtonClick ), NULL, this );
	CopyOtherButton->Disconnect( wxEVT_COMMAND_BUTTON_CLICKED, wxCommandEventHandler( Refine2DResultsPanelParent::OnCopyOtherButtonClick ), NULL, this );
	ClearButton->Disconnect( wxEVT_COMMAND_BUTTON_CLICKED, wxCommandEventHandler( Refine2DResultsPanelParent::OnClearButtonClick ), NULL, this );
	InvertButton->Disconnect( wxEVT_COMMAND_BUTTON_CLICKED, wxCommandEventHandler( Refine2DResultsPanelParent::OnInvertButtonClick ), NULL, this );

}
