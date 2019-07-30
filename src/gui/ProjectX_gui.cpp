///////////////////////////////////////////////////////////////////////////
// C++ code generated with wxFormBuilder (version Jun 20 2017)
// http://www.wxformbuilder.org/
//
// PLEASE DO "NOT" EDIT THIS FILE!
///////////////////////////////////////////////////////////////////////////

#include "../core/gui_core_headers.h"
#include "AbInitioPlotPanel.h"
#include "AngularDistributionPlotPanel.h"
#include "AssetPickerComboPanel.h"
#include "BitmapPanel.h"
#include "CTF1DPanel.h"
#include "ClassificationPlotPanel.h"
#include "DisplayPanel.h"
#include "DisplayRefinementResultsPanel.h"
#include "MyFSCPanel.h"
#include "PickingResultsDisplayPanel.h"
#include "PlotCurvePanel.h"
#include "PlotFSCPanel.h"
#include "ResultsDataViewListCtrl.h"
#include "ShowCTFResultsPanel.h"
#include "UnblurResultsPanel.h"
#include "my_controls.h"

#include "ProjectX_gui.h"

///////////////////////////////////////////////////////////////////////////

MainFrame::MainFrame( wxWindow* parent, wxWindowID id, const wxString& title, const wxPoint& pos, const wxSize& size, long style ) : wxFrame( parent, id, title, pos, size, style )
{
	this->SetSizeHints( wxSize( 1366,768 ), wxDefaultSize );
	
	wxBoxSizer* bSizer2;
	bSizer2 = new wxBoxSizer( wxVERTICAL );
	
	LeftPanel = new wxPanel( this, wxID_ANY, wxDefaultPosition, wxDefaultSize, wxRAISED_BORDER|wxTAB_TRAVERSAL );
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
	this->Connect( FileNewProject->GetId(), wxEVT_COMMAND_MENU_SELECTED, wxCommandEventHandler( MainFrame::OnFileNewProject ) );
	this->Connect( FileOpenProject->GetId(), wxEVT_COMMAND_MENU_SELECTED, wxCommandEventHandler( MainFrame::OnFileOpenProject ) );
	this->Connect( FileCloseProject->GetId(), wxEVT_COMMAND_MENU_SELECTED, wxCommandEventHandler( MainFrame::OnFileCloseProject ) );
	this->Connect( FileExit->GetId(), wxEVT_COMMAND_MENU_SELECTED, wxCommandEventHandler( MainFrame::OnFileExit ) );
	this->Connect( OnlineHelpLaunch->GetId(), wxEVT_COMMAND_MENU_SELECTED, wxCommandEventHandler( MainFrame::OnHelpLaunch ) );
	this->Connect( AboutLaunch->GetId(), wxEVT_COMMAND_MENU_SELECTED, wxCommandEventHandler( MainFrame::OnAboutLaunch ) );
}

MainFrame::~MainFrame()
{
	// Disconnect Events
	MenuBook->Disconnect( wxEVT_COMMAND_LISTBOOK_PAGE_CHANGED, wxListbookEventHandler( MainFrame::OnMenuBookChange ), NULL, this );
	m_menubar1->Disconnect( wxEVT_UPDATE_UI, wxUpdateUIEventHandler( MainFrame::OnFileMenuUpdate ), NULL, this );
	this->Disconnect( wxID_ANY, wxEVT_COMMAND_MENU_SELECTED, wxCommandEventHandler( MainFrame::OnFileNewProject ) );
	this->Disconnect( wxID_ANY, wxEVT_COMMAND_MENU_SELECTED, wxCommandEventHandler( MainFrame::OnFileOpenProject ) );
	this->Disconnect( wxID_ANY, wxEVT_COMMAND_MENU_SELECTED, wxCommandEventHandler( MainFrame::OnFileCloseProject ) );
	this->Disconnect( wxID_ANY, wxEVT_COMMAND_MENU_SELECTED, wxCommandEventHandler( MainFrame::OnFileExit ) );
	this->Disconnect( wxID_ANY, wxEVT_COMMAND_MENU_SELECTED, wxCommandEventHandler( MainFrame::OnHelpLaunch ) );
	this->Disconnect( wxID_ANY, wxEVT_COMMAND_MENU_SELECTED, wxCommandEventHandler( MainFrame::OnAboutLaunch ) );
	
}

AssetPickerComboPanelParent::AssetPickerComboPanelParent( wxWindow* parent, wxWindowID id, const wxPoint& pos, const wxSize& size, long style ) : wxPanel( parent, id, pos, size, style )
{
	wxBoxSizer* bSizer436;
	bSizer436 = new wxBoxSizer( wxHORIZONTAL );
	
	AssetComboBox = new MemoryComboBox( this, wxID_ANY, wxEmptyString, wxDefaultPosition, wxDefaultSize, 0, NULL, wxCB_READONLY ); 
	bSizer436->Add( AssetComboBox, 100, wxALIGN_CENTER_VERTICAL|wxEXPAND, 0 );
	
	wxBoxSizer* bSizer494;
	bSizer494 = new wxBoxSizer( wxVERTICAL );
	
	PreviousButton = new NoFocusBitmapButton( this, wxID_ANY, wxNullBitmap, wxDefaultPosition, wxDefaultSize, wxBU_AUTODRAW );
	PreviousButton->SetDefault(); 
	bSizer494->Add( PreviousButton, 0, wxALIGN_CENTER_VERTICAL|wxALL, 0 );
	
	NextButton = new NoFocusBitmapButton( this, wxID_ADD, wxNullBitmap, wxDefaultPosition, wxDefaultSize, wxBU_AUTODRAW );
	NextButton->SetDefault(); 
	bSizer494->Add( NextButton, 0, wxALIGN_CENTER_VERTICAL|wxALL, 0 );
	
	
	bSizer436->Add( bSizer494, 0, wxALIGN_CENTER_VERTICAL, 5 );
	
	WindowSelectButton = new NoFocusBitmapButton( this, wxID_ANY, wxNullBitmap, wxDefaultPosition, wxDefaultSize, wxBU_AUTODRAW );
	WindowSelectButton->SetDefault(); 
	bSizer436->Add( WindowSelectButton, 0, wxALIGN_CENTER_VERTICAL|wxLEFT, 5 );
	
	
	this->SetSizer( bSizer436 );
	this->Layout();
	bSizer436->Fit( this );
	
	// Connect Events
	this->Connect( wxEVT_SIZE, wxSizeEventHandler( AssetPickerComboPanelParent::OnSize ) );
	this->Connect( wxEVT_UPDATE_UI, wxUpdateUIEventHandler( AssetPickerComboPanelParent::OnUpdateUI ) );
	PreviousButton->Connect( wxEVT_COMMAND_BUTTON_CLICKED, wxCommandEventHandler( AssetPickerComboPanelParent::OnPreviousButtonClick ), NULL, this );
	NextButton->Connect( wxEVT_COMMAND_BUTTON_CLICKED, wxCommandEventHandler( AssetPickerComboPanelParent::OnNextButtonClick ), NULL, this );
}

AssetPickerComboPanelParent::~AssetPickerComboPanelParent()
{
	// Disconnect Events
	this->Disconnect( wxEVT_SIZE, wxSizeEventHandler( AssetPickerComboPanelParent::OnSize ) );
	this->Disconnect( wxEVT_UPDATE_UI, wxUpdateUIEventHandler( AssetPickerComboPanelParent::OnUpdateUI ) );
	PreviousButton->Disconnect( wxEVT_COMMAND_BUTTON_CLICKED, wxCommandEventHandler( AssetPickerComboPanelParent::OnPreviousButtonClick ), NULL, this );
	NextButton->Disconnect( wxEVT_COMMAND_BUTTON_CLICKED, wxCommandEventHandler( AssetPickerComboPanelParent::OnNextButtonClick ), NULL, this );
	
}

AbInitio3DPanelParent::AbInitio3DPanelParent( wxWindow* parent, wxWindowID id, const wxPoint& pos, const wxSize& size, long style ) : JobPanel( parent, id, pos, size, style )
{
	wxBoxSizer* bSizer43;
	bSizer43 = new wxBoxSizer( wxVERTICAL );
	
	TopStaticLine = new wxStaticLine( this, wxID_ANY, wxDefaultPosition, wxDefaultSize, wxLI_HORIZONTAL );
	bSizer43->Add( TopStaticLine, 0, wxEXPAND | wxALL, 5 );
	
	wxBoxSizer* bSizer504;
	bSizer504 = new wxBoxSizer( wxHORIZONTAL );
	
	ImageOrClassAverageStaticText = new wxStaticText( this, wxID_ANY, wxT("Use Images or Class Averages? :"), wxDefaultPosition, wxDefaultSize, 0 );
	ImageOrClassAverageStaticText->Wrap( -1 );
	bSizer504->Add( ImageOrClassAverageStaticText, 0, wxALIGN_CENTER_VERTICAL|wxALIGN_RIGHT|wxALL, 5 );
	
	wxBoxSizer* bSizer502;
	bSizer502 = new wxBoxSizer( wxHORIZONTAL );
	
	ImageInputRadioButton = new wxRadioButton( this, wxID_ANY, wxT("Images"), wxDefaultPosition, wxDefaultSize, 0 );
	ImageInputRadioButton->SetValue( true ); 
	bSizer502->Add( ImageInputRadioButton, 0, wxALIGN_CENTER_VERTICAL|wxALL, 5 );
	
	ClassAverageInputRadioButton = new wxRadioButton( this, wxID_ANY, wxT("Class Averages"), wxDefaultPosition, wxDefaultSize, 0 );
	bSizer502->Add( ClassAverageInputRadioButton, 0, wxALIGN_CENTER_VERTICAL|wxALL, 5 );
	
	
	bSizer504->Add( bSizer502, 1, wxEXPAND, 5 );
	
	
	bSizer43->Add( bSizer504, 0, wxEXPAND, 5 );
	
	BottomStaticLine = new wxStaticLine( this, wxID_ANY, wxDefaultPosition, wxDefaultSize, wxLI_HORIZONTAL );
	bSizer43->Add( BottomStaticLine, 0, wxEXPAND | wxALL, 5 );
	
	wxBoxSizer* bSizer451;
	bSizer451 = new wxBoxSizer( wxHORIZONTAL );
	
	InputParamsPanel = new wxPanel( this, wxID_ANY, wxDefaultPosition, wxDefaultSize, wxTAB_TRAVERSAL );
	wxBoxSizer* bSizer441;
	bSizer441 = new wxBoxSizer( wxVERTICAL );
	
	wxBoxSizer* bSizer200;
	bSizer200 = new wxBoxSizer( wxHORIZONTAL );
	
	wxFlexGridSizer* fgSizer15;
	fgSizer15 = new wxFlexGridSizer( 0, 2, 0, 0 );
	fgSizer15->SetFlexibleDirection( wxBOTH );
	fgSizer15->SetNonFlexibleGrowMode( wxFLEX_GROWMODE_SPECIFIED );
	
	InputRefinementPackageStaticText = new wxStaticText( InputParamsPanel, wxID_ANY, wxT("Input Refinement Package :"), wxDefaultPosition, wxSize( -1,-1 ), 0 );
	InputRefinementPackageStaticText->Wrap( -1 );
	fgSizer15->Add( InputRefinementPackageStaticText, 0, wxALIGN_CENTER_VERTICAL|wxALIGN_RIGHT|wxALL, 5 );
	
	RefinementPackageComboBox = new RefinementPackagePickerComboPanel( InputParamsPanel, wxID_ANY, wxDefaultPosition, wxDefaultSize, wxTAB_TRAVERSAL );
	RefinementPackageComboBox->Enable( false );
	RefinementPackageComboBox->SetMinSize( wxSize( 350,-1 ) );
	RefinementPackageComboBox->SetMaxSize( wxSize( 350,-1 ) );
	
	fgSizer15->Add( RefinementPackageComboBox, 1, wxEXPAND | wxALL, 5 );
	
	InputClassificationSelectionStaticText = new wxStaticText( InputParamsPanel, wxID_ANY, wxT("Input Classification Selection :"), wxDefaultPosition, wxDefaultSize, 0 );
	InputClassificationSelectionStaticText->Wrap( -1 );
	InputClassificationSelectionStaticText->Enable( false );
	
	fgSizer15->Add( InputClassificationSelectionStaticText, 0, wxALIGN_CENTER_VERTICAL|wxALIGN_RIGHT|wxALL, 5 );
	
	ClassSelectionComboBox = new ClassSelectionPickerComboPanel( InputParamsPanel, wxID_ANY, wxDefaultPosition, wxDefaultSize, wxTAB_TRAVERSAL );
	ClassSelectionComboBox->Enable( false );
	ClassSelectionComboBox->SetMinSize( wxSize( 350,-1 ) );
	ClassSelectionComboBox->SetMaxSize( wxSize( 350,-1 ) );
	
	fgSizer15->Add( ClassSelectionComboBox, 1, wxEXPAND | wxALL, 5 );
	
	
	bSizer200->Add( fgSizer15, 0, wxALIGN_CENTER_VERTICAL, 5 );
	
	m_staticline52 = new wxStaticLine( InputParamsPanel, wxID_ANY, wxDefaultPosition, wxDefaultSize, wxLI_HORIZONTAL|wxLI_VERTICAL );
	bSizer200->Add( m_staticline52, 0, wxEXPAND | wxALL, 5 );
	
	wxFlexGridSizer* fgSizer35;
	fgSizer35 = new wxFlexGridSizer( 0, 2, 0, 0 );
	fgSizer35->SetFlexibleDirection( wxBOTH );
	fgSizer35->SetNonFlexibleGrowMode( wxFLEX_GROWMODE_SPECIFIED );
	
	NoClassesStaticText = new wxStaticText( InputParamsPanel, wxID_ANY, wxT("No. Classes :"), wxDefaultPosition, wxDefaultSize, 0 );
	NoClassesStaticText->Wrap( -1 );
	NoClassesStaticText->Enable( false );
	
	fgSizer35->Add( NoClassesStaticText, 0, wxALIGN_CENTER_VERTICAL|wxALIGN_RIGHT|wxALL, 5 );
	
	NumberClassesSpinCtrl = new wxSpinCtrl( InputParamsPanel, wxID_ANY, wxEmptyString, wxDefaultPosition, wxDefaultSize, wxSP_ARROW_KEYS, 1, 50, 1 );
	NumberClassesSpinCtrl->Enable( false );
	NumberClassesSpinCtrl->SetMinSize( wxSize( 100,-1 ) );
	
	fgSizer35->Add( NumberClassesSpinCtrl, 0, wxALL, 5 );
	
	m_staticText415 = new wxStaticText( InputParamsPanel, wxID_ANY, wxT("No. of Starts :"), wxDefaultPosition, wxDefaultSize, 0 );
	m_staticText415->Wrap( -1 );
	fgSizer35->Add( m_staticText415, 0, wxALIGN_CENTER_VERTICAL|wxALL, 5 );
	
	NumberStartsSpinCtrl = new wxSpinCtrl( InputParamsPanel, wxID_ANY, wxT("2"), wxDefaultPosition, wxDefaultSize, wxSP_ARROW_KEYS, 1, 1000, 0 );
	NumberStartsSpinCtrl->SetMinSize( wxSize( 100,-1 ) );
	
	fgSizer35->Add( NumberStartsSpinCtrl, 0, wxALL, 5 );
	
	
	bSizer200->Add( fgSizer35, 0, wxALIGN_CENTER_VERTICAL, 5 );
	
	m_staticline144 = new wxStaticLine( InputParamsPanel, wxID_ANY, wxDefaultPosition, wxDefaultSize, wxLI_HORIZONTAL|wxLI_VERTICAL );
	bSizer200->Add( m_staticline144, 0, wxEXPAND | wxALL, 5 );
	
	wxFlexGridSizer* fgSizer36;
	fgSizer36 = new wxFlexGridSizer( 0, 2, 0, 0 );
	fgSizer36->SetFlexibleDirection( wxBOTH );
	fgSizer36->SetNonFlexibleGrowMode( wxFLEX_GROWMODE_SPECIFIED );
	
	SymmetryStaticText = new wxStaticText( InputParamsPanel, wxID_ANY, wxT("Symmetry : "), wxDefaultPosition, wxDefaultSize, 0 );
	SymmetryStaticText->Wrap( -1 );
	SymmetryStaticText->Enable( false );
	
	fgSizer36->Add( SymmetryStaticText, 0, wxALIGN_CENTER_VERTICAL|wxALIGN_RIGHT|wxALL, 5 );
	
	SymmetryComboBox = new wxComboBox( InputParamsPanel, wxID_ANY, wxT("C1"), wxDefaultPosition, wxDefaultSize, 0, NULL, 0 ); 
	SymmetryComboBox->Enable( false );
	SymmetryComboBox->SetMinSize( wxSize( 100,-1 ) );
	
	fgSizer36->Add( SymmetryComboBox, 0, wxALL, 5 );
	
	m_staticText264 = new wxStaticText( InputParamsPanel, wxID_ANY, wxT("No. of Cycles :"), wxDefaultPosition, wxDefaultSize, 0 );
	m_staticText264->Wrap( -1 );
	fgSizer36->Add( m_staticText264, 0, wxALIGN_CENTER_VERTICAL|wxALIGN_RIGHT|wxALL, 5 );
	
	NumberRoundsSpinCtrl = new wxSpinCtrl( InputParamsPanel, wxID_ANY, wxEmptyString, wxDefaultPosition, wxDefaultSize, wxSP_ARROW_KEYS, 5, 100, 40 );
	NumberRoundsSpinCtrl->SetMinSize( wxSize( 100,-1 ) );
	
	fgSizer36->Add( NumberRoundsSpinCtrl, 1, wxALL|wxEXPAND, 5 );
	
	
	bSizer200->Add( fgSizer36, 0, wxALIGN_CENTER_VERTICAL, 5 );
	
	
	bSizer200->Add( 0, 0, 1, wxEXPAND, 5 );
	
	ExpertToggleButton = new wxToggleButton( InputParamsPanel, wxID_ANY, wxT("Show Expert Options"), wxDefaultPosition, wxDefaultSize, 0 );
	bSizer200->Add( ExpertToggleButton, 0, wxALIGN_BOTTOM|wxALL, 5 );
	
	
	bSizer441->Add( bSizer200, 1, wxEXPAND, 5 );
	
	PleaseCreateRefinementPackageText = new wxStaticText( InputParamsPanel, wxID_ANY, wxT("Please create a refinement package (in the assets panel) in order to perform a 3D refinement."), wxDefaultPosition, wxDefaultSize, 0 );
	PleaseCreateRefinementPackageText->Wrap( -1 );
	PleaseCreateRefinementPackageText->SetFont( wxFont( wxNORMAL_FONT->GetPointSize(), 70, 90, 92, false, wxEmptyString ) );
	PleaseCreateRefinementPackageText->SetForegroundColour( wxColour( 180, 0, 0 ) );
	PleaseCreateRefinementPackageText->Hide();
	
	bSizer441->Add( PleaseCreateRefinementPackageText, 0, wxALL, 5 );
	
	
	InputParamsPanel->SetSizer( bSizer441 );
	InputParamsPanel->Layout();
	bSizer441->Fit( InputParamsPanel );
	bSizer451->Add( InputParamsPanel, 1, wxEXPAND | wxALL, 5 );
	
	
	bSizer43->Add( bSizer451, 0, wxEXPAND, 5 );
	
	m_staticline141 = new wxStaticLine( this, wxID_ANY, wxDefaultPosition, wxDefaultSize, wxLI_HORIZONTAL );
	bSizer43->Add( m_staticline141, 0, wxEXPAND | wxALL, 5 );
	
	wxBoxSizer* bSizer46;
	bSizer46 = new wxBoxSizer( wxHORIZONTAL );
	
	ExpertPanel = new wxScrolledWindow( this, wxID_ANY, wxDefaultPosition, wxSize( -1,-1 ), wxVSCROLL );
	ExpertPanel->SetScrollRate( 5, 5 );
	InputSizer = new wxBoxSizer( wxVERTICAL );
	
	wxBoxSizer* bSizer258;
	bSizer258 = new wxBoxSizer( wxHORIZONTAL );
	
	m_staticText531 = new wxStaticText( ExpertPanel, wxID_ANY, wxT("Refinement"), wxDefaultPosition, wxDefaultSize, 0 );
	m_staticText531->Wrap( -1 );
	m_staticText531->SetFont( wxFont( 10, 74, 90, 92, true, wxT("Sans") ) );
	
	bSizer258->Add( m_staticText531, 0, wxALIGN_BOTTOM|wxALL, 5 );
	
	
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
	
	NoMovieFramesStaticText = new wxStaticText( ExpertPanel, wxID_ANY, wxT("Initial Resolution Limit (Å) :"), wxDefaultPosition, wxDefaultSize, 0 );
	NoMovieFramesStaticText->Wrap( -1 );
	fgSizer1->Add( NoMovieFramesStaticText, 0, wxALIGN_CENTER_VERTICAL|wxALL, 5 );
	
	InitialResolutionLimitTextCtrl = new NumericTextCtrl( ExpertPanel, wxID_ANY, wxT("40.00"), wxDefaultPosition, wxDefaultSize, 0 );
	fgSizer1->Add( InitialResolutionLimitTextCtrl, 0, wxALL|wxEXPAND, 5 );
	
	m_staticText196 = new wxStaticText( ExpertPanel, wxID_ANY, wxT("Final Resolution Limit (Å) :"), wxDefaultPosition, wxDefaultSize, 0 );
	m_staticText196->Wrap( -1 );
	fgSizer1->Add( m_staticText196, 0, wxALIGN_CENTER_VERTICAL|wxALL, 5 );
	
	FinalResolutionLimitTextCtrl = new NumericTextCtrl( ExpertPanel, wxID_ANY, wxT("9.00"), wxDefaultPosition, wxDefaultSize, wxTE_PROCESS_ENTER );
	fgSizer1->Add( FinalResolutionLimitTextCtrl, 0, wxALL|wxEXPAND, 5 );
	
	GlobalMaskRadiusStaticText = new wxStaticText( ExpertPanel, wxID_ANY, wxT("Mask Radius (Å) :"), wxDefaultPosition, wxDefaultSize, 0 );
	GlobalMaskRadiusStaticText->Wrap( -1 );
	fgSizer1->Add( GlobalMaskRadiusStaticText, 0, wxALIGN_CENTER_VERTICAL|wxALL, 5 );
	
	GlobalMaskRadiusTextCtrl = new NumericTextCtrl( ExpertPanel, wxID_ANY, wxT("30.00"), wxDefaultPosition, wxDefaultSize, wxTE_PROCESS_ENTER );
	fgSizer1->Add( GlobalMaskRadiusTextCtrl, 0, wxALL, 5 );
	
	InnerMaskRadiusStaticText1 = new wxStaticText( ExpertPanel, wxID_ANY, wxT("Inner Mask Radius (Å) :"), wxDefaultPosition, wxDefaultSize, 0 );
	InnerMaskRadiusStaticText1->Wrap( -1 );
	fgSizer1->Add( InnerMaskRadiusStaticText1, 0, wxALIGN_CENTER_VERTICAL|wxALL, 5 );
	
	InnerMaskRadiusTextCtrl = new NumericTextCtrl( ExpertPanel, wxID_ANY, wxT("30.00"), wxDefaultPosition, wxDefaultSize, wxTE_PROCESS_ENTER );
	fgSizer1->Add( InnerMaskRadiusTextCtrl, 0, wxALL, 5 );
	
	SearchRangeXStaticText = new wxStaticText( ExpertPanel, wxID_ANY, wxT("Search Range in X (Å) :"), wxDefaultPosition, wxDefaultSize, 0 );
	SearchRangeXStaticText->Wrap( -1 );
	fgSizer1->Add( SearchRangeXStaticText, 0, wxALIGN_CENTER_VERTICAL|wxALL, 5 );
	
	SearchRangeXTextCtrl = new NumericTextCtrl( ExpertPanel, wxID_ANY, wxT("100.00"), wxDefaultPosition, wxDefaultSize, wxTE_PROCESS_ENTER );
	fgSizer1->Add( SearchRangeXTextCtrl, 0, wxALL, 5 );
	
	SearchRangeYStaticText = new wxStaticText( ExpertPanel, wxID_ANY, wxT("Search Range in Y (Å) :"), wxDefaultPosition, wxDefaultSize, 0 );
	SearchRangeYStaticText->Wrap( -1 );
	fgSizer1->Add( SearchRangeYStaticText, 0, wxALIGN_CENTER_VERTICAL|wxALL, 5 );
	
	SearchRangeYTextCtrl = new NumericTextCtrl( ExpertPanel, wxID_ANY, wxT("100.00"), wxDefaultPosition, wxDefaultSize, wxTE_PROCESS_ENTER );
	fgSizer1->Add( SearchRangeYTextCtrl, 0, wxALL, 5 );
	
	UseAutoMaskingStaticText = new wxStaticText( ExpertPanel, wxID_ANY, wxT("Use Auto-Masking?"), wxDefaultPosition, wxDefaultSize, 0 );
	UseAutoMaskingStaticText->Wrap( -1 );
	fgSizer1->Add( UseAutoMaskingStaticText, 0, wxALIGN_CENTER_VERTICAL|wxALL, 5 );
	
	wxBoxSizer* bSizer264;
	bSizer264 = new wxBoxSizer( wxHORIZONTAL );
	
	AutoMaskYesRadio = new wxRadioButton( ExpertPanel, wxID_ANY, wxT("Yes"), wxDefaultPosition, wxDefaultSize, wxRB_GROUP );
	bSizer264->Add( AutoMaskYesRadio, 0, wxALL, 5 );
	
	AutoMaskNoRadio = new wxRadioButton( ExpertPanel, wxID_ANY, wxT("No"), wxDefaultPosition, wxDefaultSize, 0 );
	bSizer264->Add( AutoMaskNoRadio, 0, wxALL, 5 );
	
	
	fgSizer1->Add( bSizer264, 1, wxEXPAND, 5 );
	
	m_staticText4151 = new wxStaticText( ExpertPanel, wxID_ANY, wxT("Auto Percent Used?"), wxDefaultPosition, wxDefaultSize, 0 );
	m_staticText4151->Wrap( -1 );
	fgSizer1->Add( m_staticText4151, 0, wxALL, 5 );
	
	wxBoxSizer* bSizer2641;
	bSizer2641 = new wxBoxSizer( wxHORIZONTAL );
	
	AutoPercentUsedYesRadio = new wxRadioButton( ExpertPanel, wxID_ANY, wxT("Yes"), wxDefaultPosition, wxDefaultSize, wxRB_GROUP );
	bSizer2641->Add( AutoPercentUsedYesRadio, 0, wxALL, 5 );
	
	AutoPercentUsedNoRadio = new wxRadioButton( ExpertPanel, wxID_ANY, wxT("No"), wxDefaultPosition, wxDefaultSize, 0 );
	bSizer2641->Add( AutoPercentUsedNoRadio, 0, wxALL, 5 );
	
	
	fgSizer1->Add( bSizer2641, 1, wxEXPAND, 5 );
	
	InitialPercentUsedStaticText = new wxStaticText( ExpertPanel, wxID_ANY, wxT("\tInitial % Used :"), wxDefaultPosition, wxDefaultSize, 0 );
	InitialPercentUsedStaticText->Wrap( -1 );
	fgSizer1->Add( InitialPercentUsedStaticText, 0, wxALIGN_CENTER_VERTICAL|wxALL, 5 );
	
	StartPercentUsedTextCtrl = new NumericTextCtrl( ExpertPanel, wxID_ANY, wxT("9.00"), wxDefaultPosition, wxDefaultSize, wxTE_PROCESS_ENTER );
	fgSizer1->Add( StartPercentUsedTextCtrl, 0, wxALL, 5 );
	
	FinalPercentUsedStaticText = new wxStaticText( ExpertPanel, wxID_ANY, wxT("\tFinal % Used :"), wxDefaultPosition, wxDefaultSize, 0 );
	FinalPercentUsedStaticText->Wrap( -1 );
	fgSizer1->Add( FinalPercentUsedStaticText, 0, wxALIGN_CENTER_VERTICAL|wxALL, 5 );
	
	EndPercentUsedTextCtrl = new NumericTextCtrl( ExpertPanel, wxID_ANY, wxT("9.00"), wxDefaultPosition, wxDefaultSize, wxTE_PROCESS_ENTER );
	fgSizer1->Add( EndPercentUsedTextCtrl, 0, wxALL, 5 );
	
	AlwaysApplySymmetryStaticText = new wxStaticText( ExpertPanel, wxID_ANY, wxT("Always Apply Symmetry?"), wxDefaultPosition, wxDefaultSize, 0 );
	AlwaysApplySymmetryStaticText->Wrap( -1 );
	fgSizer1->Add( AlwaysApplySymmetryStaticText, 0, wxALIGN_CENTER_VERTICAL|wxALL, 5 );
	
	wxBoxSizer* bSizer26611;
	bSizer26611 = new wxBoxSizer( wxHORIZONTAL );
	
	AlwaysApplySymmetryYesButton = new wxRadioButton( ExpertPanel, wxID_ANY, wxT("Yes"), wxDefaultPosition, wxDefaultSize, wxRB_GROUP );
	bSizer26611->Add( AlwaysApplySymmetryYesButton, 0, wxALL, 5 );
	
	AlwaysApplySymmetryNoButton = new wxRadioButton( ExpertPanel, wxID_ANY, wxT("No"), wxDefaultPosition, wxDefaultSize, 0 );
	bSizer26611->Add( AlwaysApplySymmetryNoButton, 0, wxALL, 5 );
	
	
	fgSizer1->Add( bSizer26611, 1, wxALIGN_CENTER_VERTICAL|wxEXPAND, 5 );
	
	m_staticText532 = new wxStaticText( ExpertPanel, wxID_ANY, wxT("3D Reconstruction"), wxDefaultPosition, wxDefaultSize, 0 );
	m_staticText532->Wrap( -1 );
	m_staticText532->SetFont( wxFont( 10, 74, 90, 92, true, wxT("Sans") ) );
	
	fgSizer1->Add( m_staticText532, 0, wxALL, 5 );
	
	
	fgSizer1->Add( 0, 0, 1, wxEXPAND, 5 );
	
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
	
	m_staticText662 = new wxStaticText( ExpertPanel, wxID_ANY, wxT("Class Averages"), wxDefaultPosition, wxDefaultSize, 0 );
	m_staticText662->Wrap( -1 );
	m_staticText662->SetFont( wxFont( 10, 74, 90, 92, true, wxT("Sans") ) );
	
	fgSizer1->Add( m_staticText662, 0, wxALL, 5 );
	
	
	fgSizer1->Add( 0, 0, 1, wxEXPAND, 5 );
	
	m_staticText663 = new wxStaticText( ExpertPanel, wxID_ANY, wxT("Images per Class :"), wxDefaultPosition, wxDefaultSize, 0 );
	m_staticText663->Wrap( -1 );
	fgSizer1->Add( m_staticText663, 0, wxALIGN_CENTER_VERTICAL|wxALL, 5 );
	
	ImagesPerClassSpinCtrl = new wxSpinCtrl( ExpertPanel, wxID_ANY, wxEmptyString, wxDefaultPosition, wxDefaultSize, wxSP_ARROW_KEYS, 1, 500, 5 );
	ImagesPerClassSpinCtrl->SetMinSize( wxSize( 100,-1 ) );
	
	fgSizer1->Add( ImagesPerClassSpinCtrl, 0, wxALIGN_CENTER_VERTICAL|wxALL, 5 );
	
	
	InputSizer->Add( fgSizer1, 1, wxEXPAND, 5 );
	
	
	ExpertPanel->SetSizer( InputSizer );
	ExpertPanel->Layout();
	InputSizer->Fit( ExpertPanel );
	bSizer46->Add( ExpertPanel, 0, wxALL|wxEXPAND, 5 );
	
	wxBoxSizer* bSizer325;
	bSizer325 = new wxBoxSizer( wxVERTICAL );
	
	OutputTextPanel = new wxPanel( this, wxID_ANY, wxDefaultPosition, wxDefaultSize, wxTAB_TRAVERSAL );
	OutputTextPanel->Hide();
	
	wxBoxSizer* bSizer56;
	bSizer56 = new wxBoxSizer( wxVERTICAL );
	
	output_textctrl = new wxTextCtrl( OutputTextPanel, wxID_ANY, wxEmptyString, wxDefaultPosition, wxDefaultSize, wxTE_MULTILINE|wxTE_READONLY );
	bSizer56->Add( output_textctrl, 1, wxALL|wxEXPAND, 5 );
	
	
	OutputTextPanel->SetSizer( bSizer56 );
	OutputTextPanel->Layout();
	bSizer56->Fit( OutputTextPanel );
	bSizer325->Add( OutputTextPanel, 40, wxEXPAND | wxALL, 5 );
	
	PlotPanel = new AbInitioPlotPanel( this, wxID_ANY, wxDefaultPosition, wxDefaultSize, wxTAB_TRAVERSAL );
	PlotPanel->Hide();
	
	bSizer325->Add( PlotPanel, 60, wxEXPAND | wxALL, 5 );
	
	
	bSizer46->Add( bSizer325, 40, wxEXPAND, 5 );
	
	InfoPanel = new wxPanel( this, wxID_ANY, wxDefaultPosition, wxDefaultSize, wxTAB_TRAVERSAL );
	wxBoxSizer* bSizer61;
	bSizer61 = new wxBoxSizer( wxVERTICAL );
	
	InfoText = new wxRichTextCtrl( InfoPanel, wxID_ANY, wxEmptyString, wxDefaultPosition, wxDefaultSize, wxTE_READONLY|wxHSCROLL|wxVSCROLL );
	bSizer61->Add( InfoText, 1, wxEXPAND | wxALL, 5 );
	
	
	InfoPanel->SetSizer( bSizer61 );
	InfoPanel->Layout();
	bSizer61->Fit( InfoPanel );
	bSizer46->Add( InfoPanel, 1, wxEXPAND | wxALL, 5 );
	
	OrthResultsPanel = new wxPanel( this, wxID_ANY, wxDefaultPosition, wxDefaultSize, wxTAB_TRAVERSAL );
	OrthResultsPanel->Hide();
	
	wxBoxSizer* bSizer326;
	bSizer326 = new wxBoxSizer( wxVERTICAL );
	
	m_staticText377 = new wxStaticText( OrthResultsPanel, wxID_ANY, wxT("Orthogonal Slices / Projections"), wxDefaultPosition, wxDefaultSize, 0 );
	m_staticText377->Wrap( -1 );
	m_staticText377->SetFont( wxFont( 10, 74, 90, 92, false, wxT("Sans") ) );
	
	bSizer326->Add( m_staticText377, 0, wxALL, 5 );
	
	m_staticline91 = new wxStaticLine( OrthResultsPanel, wxID_ANY, wxDefaultPosition, wxDefaultSize, wxLI_HORIZONTAL );
	bSizer326->Add( m_staticline91, 0, wxEXPAND | wxALL, 5 );
	
	ShowOrthDisplayPanel = new DisplayPanel( OrthResultsPanel, wxID_ANY, wxDefaultPosition, wxDefaultSize, wxTAB_TRAVERSAL );
	bSizer326->Add( ShowOrthDisplayPanel, 1, wxEXPAND | wxALL, 5 );
	
	
	OrthResultsPanel->SetSizer( bSizer326 );
	OrthResultsPanel->Layout();
	bSizer326->Fit( OrthResultsPanel );
	bSizer46->Add( OrthResultsPanel, 80, wxEXPAND | wxALL, 5 );
	
	
	bSizer43->Add( bSizer46, 1, wxEXPAND, 5 );
	
	m_staticline11 = new wxStaticLine( this, wxID_ANY, wxDefaultPosition, wxDefaultSize, wxLI_HORIZONTAL );
	bSizer43->Add( m_staticline11, 0, wxEXPAND | wxALL, 5 );
	
	wxBoxSizer* bSizer48;
	bSizer48 = new wxBoxSizer( wxHORIZONTAL );
	
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
	
	TimeRemainingText = new wxStaticText( ProgressPanel, wxID_ANY, wxT("Time Remaining : ???h:??m:??s   "), wxDefaultPosition, wxDefaultSize, wxALIGN_CENTRE );
	TimeRemainingText->Wrap( -1 );
	bSizer59->Add( TimeRemainingText, 0, wxALIGN_CENTER_VERTICAL|wxALIGN_RIGHT|wxALL, 5 );
	
	m_staticline60 = new wxStaticLine( ProgressPanel, wxID_ANY, wxDefaultPosition, wxDefaultSize, wxLI_HORIZONTAL|wxLI_VERTICAL );
	bSizer59->Add( m_staticline60, 0, wxEXPAND | wxALL, 5 );
	
	FinishButton = new wxButton( ProgressPanel, wxID_ANY, wxT("Finish"), wxDefaultPosition, wxDefaultSize, 0 );
	FinishButton->Hide();
	
	bSizer59->Add( FinishButton, 0, wxALIGN_CENTER_HORIZONTAL|wxALIGN_CENTER_VERTICAL|wxALL, 5 );
	
	CancelAlignmentButton = new wxButton( ProgressPanel, wxID_ANY, wxT("Terminate Job"), wxDefaultPosition, wxDefaultSize, 0 );
	bSizer59->Add( CancelAlignmentButton, 0, wxALIGN_CENTER_HORIZONTAL|wxALIGN_CENTER_VERTICAL|wxALL, 5 );
	
	CurrentLineOne = new wxStaticLine( ProgressPanel, wxID_ANY, wxDefaultPosition, wxDefaultSize, wxLI_HORIZONTAL|wxLI_VERTICAL );
	bSizer59->Add( CurrentLineOne, 0, wxEXPAND | wxALL, 5 );
	
	TakeCurrentResultButton = new wxButton( ProgressPanel, wxID_ANY, wxT("Take Current"), wxDefaultPosition, wxDefaultSize, 0 );
	TakeCurrentResultButton->SetToolTip( wxT("This will terminate the job") );
	
	bSizer59->Add( TakeCurrentResultButton, 0, wxALIGN_CENTER_VERTICAL|wxALL, 5 );
	
	CurrentLineTwo = new wxStaticLine( ProgressPanel, wxID_ANY, wxDefaultPosition, wxDefaultSize, wxLI_HORIZONTAL|wxLI_VERTICAL );
	bSizer59->Add( CurrentLineTwo, 0, wxEXPAND | wxALL, 5 );
	
	TakeLastStartResultButton = new wxButton( ProgressPanel, wxID_ANY, wxT("Take Last Start"), wxDefaultPosition, wxDefaultSize, 0 );
	TakeLastStartResultButton->SetToolTip( wxT("This will terminate the job") );
	
	bSizer59->Add( TakeLastStartResultButton, 0, wxALIGN_CENTER_VERTICAL|wxALL, 5 );
	
	
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
	this->Connect( wxEVT_UPDATE_UI, wxUpdateUIEventHandler( AbInitio3DPanelParent::OnUpdateUI ) );
	ImageInputRadioButton->Connect( wxEVT_COMMAND_RADIOBUTTON_SELECTED, wxCommandEventHandler( AbInitio3DPanelParent::OnMethodChange ), NULL, this );
	ClassAverageInputRadioButton->Connect( wxEVT_COMMAND_RADIOBUTTON_SELECTED, wxCommandEventHandler( AbInitio3DPanelParent::OnMethodChange ), NULL, this );
	ExpertToggleButton->Connect( wxEVT_COMMAND_TOGGLEBUTTON_CLICKED, wxCommandEventHandler( AbInitio3DPanelParent::OnExpertOptionsToggle ), NULL, this );
	ResetAllDefaultsButton->Connect( wxEVT_COMMAND_BUTTON_CLICKED, wxCommandEventHandler( AbInitio3DPanelParent::ResetAllDefaultsClick ), NULL, this );
	InfoText->Connect( wxEVT_COMMAND_TEXT_URL, wxTextUrlEventHandler( AbInitio3DPanelParent::OnInfoURL ), NULL, this );
	FinishButton->Connect( wxEVT_COMMAND_BUTTON_CLICKED, wxCommandEventHandler( AbInitio3DPanelParent::FinishButtonClick ), NULL, this );
	CancelAlignmentButton->Connect( wxEVT_COMMAND_BUTTON_CLICKED, wxCommandEventHandler( AbInitio3DPanelParent::TerminateButtonClick ), NULL, this );
	TakeCurrentResultButton->Connect( wxEVT_COMMAND_BUTTON_CLICKED, wxCommandEventHandler( AbInitio3DPanelParent::TakeCurrentClicked ), NULL, this );
	TakeLastStartResultButton->Connect( wxEVT_COMMAND_BUTTON_CLICKED, wxCommandEventHandler( AbInitio3DPanelParent::TakeLastStartClicked ), NULL, this );
	StartRefinementButton->Connect( wxEVT_COMMAND_BUTTON_CLICKED, wxCommandEventHandler( AbInitio3DPanelParent::StartRefinementClick ), NULL, this );
}

AbInitio3DPanelParent::~AbInitio3DPanelParent()
{
	// Disconnect Events
	this->Disconnect( wxEVT_UPDATE_UI, wxUpdateUIEventHandler( AbInitio3DPanelParent::OnUpdateUI ) );
	ImageInputRadioButton->Disconnect( wxEVT_COMMAND_RADIOBUTTON_SELECTED, wxCommandEventHandler( AbInitio3DPanelParent::OnMethodChange ), NULL, this );
	ClassAverageInputRadioButton->Disconnect( wxEVT_COMMAND_RADIOBUTTON_SELECTED, wxCommandEventHandler( AbInitio3DPanelParent::OnMethodChange ), NULL, this );
	ExpertToggleButton->Disconnect( wxEVT_COMMAND_TOGGLEBUTTON_CLICKED, wxCommandEventHandler( AbInitio3DPanelParent::OnExpertOptionsToggle ), NULL, this );
	ResetAllDefaultsButton->Disconnect( wxEVT_COMMAND_BUTTON_CLICKED, wxCommandEventHandler( AbInitio3DPanelParent::ResetAllDefaultsClick ), NULL, this );
	InfoText->Disconnect( wxEVT_COMMAND_TEXT_URL, wxTextUrlEventHandler( AbInitio3DPanelParent::OnInfoURL ), NULL, this );
	FinishButton->Disconnect( wxEVT_COMMAND_BUTTON_CLICKED, wxCommandEventHandler( AbInitio3DPanelParent::FinishButtonClick ), NULL, this );
	CancelAlignmentButton->Disconnect( wxEVT_COMMAND_BUTTON_CLICKED, wxCommandEventHandler( AbInitio3DPanelParent::TerminateButtonClick ), NULL, this );
	TakeCurrentResultButton->Disconnect( wxEVT_COMMAND_BUTTON_CLICKED, wxCommandEventHandler( AbInitio3DPanelParent::TakeCurrentClicked ), NULL, this );
	TakeLastStartResultButton->Disconnect( wxEVT_COMMAND_BUTTON_CLICKED, wxCommandEventHandler( AbInitio3DPanelParent::TakeLastStartClicked ), NULL, this );
	StartRefinementButton->Disconnect( wxEVT_COMMAND_BUTTON_CLICKED, wxCommandEventHandler( AbInitio3DPanelParent::StartRefinementClick ), NULL, this );
	
}

Refine2DPanel::Refine2DPanel( wxWindow* parent, wxWindowID id, const wxPoint& pos, const wxSize& size, long style ) : JobPanel( parent, id, pos, size, style )
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
	PleaseCreateRefinementPackageText->SetFont( wxFont( wxNORMAL_FONT->GetPointSize(), 70, 90, 92, false, wxEmptyString ) );
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
	m_staticText318->SetFont( wxFont( 10, 74, 90, 92, true, wxT("Sans") ) );
	
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
	
	TimeRemainingText = new wxStaticText( ProgressPanel, wxID_ANY, wxT("Time Remaining : ???h:??m:??s   "), wxDefaultPosition, wxDefaultSize, wxALIGN_CENTRE );
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

RefinementResultsPanel::RefinementResultsPanel( wxWindow* parent, wxWindowID id, const wxPoint& pos, const wxSize& size, long style ) : wxPanel( parent, id, pos, size, style )
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
	m_staticText708->SetFont( wxFont( 10, 74, 90, 92, false, wxT("Sans") ) );
	
	bSizer497->Add( m_staticText708, 0, wxALIGN_BOTTOM|wxALL, 5 );
	
	
	bSizer497->Add( 0, 0, 1, wxEXPAND, 5 );
	
	m_staticline132 = new wxStaticLine( m_panel125, wxID_ANY, wxDefaultPosition, wxDefaultSize, wxLI_HORIZONTAL|wxLI_VERTICAL );
	bSizer497->Add( m_staticline132, 0, wxEXPAND | wxALL, 5 );
	
	ParametersDetailButton = new NoFocusBitmapButton( m_panel125, wxID_ANY, wxNullBitmap, wxDefaultPosition, wxDefaultSize, wxBU_AUTODRAW );
	ParametersDetailButton->SetDefault(); 
	bSizer497->Add( ParametersDetailButton, 0, wxLEFT|wxTOP, 5 );
	
	AngularPlotDetailsButton = new NoFocusBitmapButton( m_panel125, wxID_ANY, wxNullBitmap, wxDefaultPosition, wxDefaultSize, wxBU_AUTODRAW );
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
	m_staticText72->SetFont( wxFont( 10, 74, 90, 92, false, wxT("Sans") ) );
	
	InfoSizer->Add( m_staticText72, 0, wxALIGN_RIGHT|wxALL, 5 );
	
	RefinementIDStaticText = new wxStaticText( JobDetailsPanel, wxID_ANY, wxEmptyString, wxDefaultPosition, wxDefaultSize, 0 );
	RefinementIDStaticText->Wrap( -1 );
	InfoSizer->Add( RefinementIDStaticText, 0, wxALIGN_LEFT|wxALL, 5 );
	
	m_staticText74 = new wxStaticText( JobDetailsPanel, wxID_ANY, wxT("Date of Run :"), wxDefaultPosition, wxDefaultSize, 0 );
	m_staticText74->Wrap( -1 );
	m_staticText74->SetFont( wxFont( 10, 74, 90, 92, false, wxT("Sans") ) );
	
	InfoSizer->Add( m_staticText74, 0, wxALIGN_RIGHT|wxALL, 5 );
	
	DateOfRunStaticText = new wxStaticText( JobDetailsPanel, wxID_ANY, wxEmptyString, wxDefaultPosition, wxDefaultSize, 0 );
	DateOfRunStaticText->Wrap( -1 );
	InfoSizer->Add( DateOfRunStaticText, 0, wxALL, 5 );
	
	m_staticText93 = new wxStaticText( JobDetailsPanel, wxID_ANY, wxT("Time Of Run :"), wxDefaultPosition, wxDefaultSize, 0 );
	m_staticText93->Wrap( -1 );
	m_staticText93->SetFont( wxFont( 10, 74, 90, 92, false, wxT("Sans") ) );
	
	InfoSizer->Add( m_staticText93, 0, wxALIGN_RIGHT|wxALL, 5 );
	
	TimeOfRunStaticText = new wxStaticText( JobDetailsPanel, wxID_ANY, wxEmptyString, wxDefaultPosition, wxDefaultSize, 0 );
	TimeOfRunStaticText->Wrap( -1 );
	InfoSizer->Add( TimeOfRunStaticText, 0, wxALIGN_LEFT|wxALL, 5 );
	
	m_staticText785 = new wxStaticText( JobDetailsPanel, wxID_ANY, wxT("Percent Used :"), wxDefaultPosition, wxDefaultSize, 0 );
	m_staticText785->Wrap( -1 );
	m_staticText785->SetFont( wxFont( 10, 74, 90, 92, false, wxT("Sans") ) );
	
	InfoSizer->Add( m_staticText785, 0, wxALIGN_RIGHT|wxALL, 5 );
	
	PercentUsedStaticText = new wxStaticText( JobDetailsPanel, wxID_ANY, wxEmptyString, wxDefaultPosition, wxDefaultSize, 0 );
	PercentUsedStaticText->Wrap( -1 );
	InfoSizer->Add( PercentUsedStaticText, 0, wxALL, 5 );
	
	m_staticText83 = new wxStaticText( JobDetailsPanel, wxID_ANY, wxT("Ref. Volume ID :"), wxDefaultPosition, wxDefaultSize, 0 );
	m_staticText83->Wrap( -1 );
	m_staticText83->SetFont( wxFont( 10, 74, 90, 92, false, wxT("Sans") ) );
	
	InfoSizer->Add( m_staticText83, 0, wxALIGN_RIGHT|wxALL, 5 );
	
	ReferenceVolumeIDStaticText = new wxStaticText( JobDetailsPanel, wxID_ANY, wxEmptyString, wxDefaultPosition, wxDefaultSize, 0 );
	ReferenceVolumeIDStaticText->Wrap( -1 );
	InfoSizer->Add( ReferenceVolumeIDStaticText, 0, wxALIGN_LEFT|wxALL, 5 );
	
	m_staticText82 = new wxStaticText( JobDetailsPanel, wxID_ANY, wxT("Ref. Refinement ID :"), wxDefaultPosition, wxDefaultSize, 0 );
	m_staticText82->Wrap( -1 );
	m_staticText82->SetFont( wxFont( 10, 74, 90, 92, false, wxT("Sans") ) );
	
	InfoSizer->Add( m_staticText82, 0, wxALIGN_RIGHT|wxALL, 5 );
	
	ReferenceRefinementIDStaticText = new wxStaticText( JobDetailsPanel, wxID_ANY, wxEmptyString, wxDefaultPosition, wxDefaultSize, 0 );
	ReferenceRefinementIDStaticText->Wrap( -1 );
	InfoSizer->Add( ReferenceRefinementIDStaticText, 0, wxALIGN_LEFT|wxALL, 5 );
	
	m_staticText85 = new wxStaticText( JobDetailsPanel, wxID_ANY, wxT("Low Res. Limit :"), wxDefaultPosition, wxDefaultSize, 0 );
	m_staticText85->Wrap( -1 );
	m_staticText85->SetFont( wxFont( 10, 74, 90, 92, false, wxT("Sans") ) );
	
	InfoSizer->Add( m_staticText85, 0, wxALIGN_RIGHT|wxALL, 5 );
	
	LowResLimitStaticText = new wxStaticText( JobDetailsPanel, wxID_ANY, wxEmptyString, wxDefaultPosition, wxDefaultSize, 0 );
	LowResLimitStaticText->Wrap( -1 );
	InfoSizer->Add( LowResLimitStaticText, 0, wxALIGN_LEFT|wxALL, 5 );
	
	m_staticText87 = new wxStaticText( JobDetailsPanel, wxID_ANY, wxT("High Res. Limit :"), wxDefaultPosition, wxDefaultSize, 0 );
	m_staticText87->Wrap( -1 );
	m_staticText87->SetFont( wxFont( 10, 74, 90, 92, false, wxT("Sans") ) );
	
	InfoSizer->Add( m_staticText87, 0, wxALIGN_RIGHT|wxALL, 5 );
	
	HighResLimitStaticText = new wxStaticText( JobDetailsPanel, wxID_ANY, wxEmptyString, wxDefaultPosition, wxDefaultSize, 0 );
	HighResLimitStaticText->Wrap( -1 );
	InfoSizer->Add( HighResLimitStaticText, 0, wxALIGN_LEFT|wxALL, 5 );
	
	m_staticText89 = new wxStaticText( JobDetailsPanel, wxID_ANY, wxT("Mask Radius :"), wxDefaultPosition, wxDefaultSize, 0 );
	m_staticText89->Wrap( -1 );
	m_staticText89->SetFont( wxFont( 10, 74, 90, 92, false, wxT("Sans") ) );
	
	InfoSizer->Add( m_staticText89, 0, wxALIGN_RIGHT|wxALL, 5 );
	
	MaskRadiusStaticText = new wxStaticText( JobDetailsPanel, wxID_ANY, wxEmptyString, wxDefaultPosition, wxDefaultSize, 0 );
	MaskRadiusStaticText->Wrap( -1 );
	InfoSizer->Add( MaskRadiusStaticText, 0, wxALIGN_LEFT|wxALL, 5 );
	
	m_staticText777 = new wxStaticText( JobDetailsPanel, wxID_ANY, wxT("Signed CC Res. Limit :"), wxDefaultPosition, wxDefaultSize, 0 );
	m_staticText777->Wrap( -1 );
	m_staticText777->SetFont( wxFont( 10, 74, 90, 92, false, wxT("Sans") ) );
	
	InfoSizer->Add( m_staticText777, 0, wxALIGN_RIGHT|wxALL, 5 );
	
	SignedCCResLimitStaticText = new wxStaticText( JobDetailsPanel, wxID_ANY, wxEmptyString, wxDefaultPosition, wxDefaultSize, 0 );
	SignedCCResLimitStaticText->Wrap( -1 );
	InfoSizer->Add( SignedCCResLimitStaticText, 0, wxALL, 5 );
	
	m_staticText779 = new wxStaticText( JobDetailsPanel, wxID_ANY, wxT("Global Res. Limit :"), wxDefaultPosition, wxDefaultSize, 0 );
	m_staticText779->Wrap( -1 );
	m_staticText779->SetFont( wxFont( 10, 74, 90, 92, false, wxT("Sans") ) );
	
	InfoSizer->Add( m_staticText779, 0, wxALIGN_RIGHT|wxALL, 5 );
	
	GlobalResLimitStaticText = new wxStaticText( JobDetailsPanel, wxID_ANY, wxEmptyString, wxDefaultPosition, wxDefaultSize, 0 );
	GlobalResLimitStaticText->Wrap( -1 );
	InfoSizer->Add( GlobalResLimitStaticText, 0, wxALL, 5 );
	
	m_staticText781 = new wxStaticText( JobDetailsPanel, wxID_ANY, wxT("Global Mask Radius :"), wxDefaultPosition, wxDefaultSize, 0 );
	m_staticText781->Wrap( -1 );
	m_staticText781->SetFont( wxFont( 10, 74, 90, 92, false, wxT("Sans") ) );
	
	InfoSizer->Add( m_staticText781, 0, wxALIGN_RIGHT|wxALL, 5 );
	
	GlobalMaskRadiusStaticText = new wxStaticText( JobDetailsPanel, wxID_ANY, wxEmptyString, wxDefaultPosition, wxDefaultSize, 0 );
	GlobalMaskRadiusStaticText->Wrap( -1 );
	InfoSizer->Add( GlobalMaskRadiusStaticText, 0, wxALL, 5 );
	
	m_staticText783 = new wxStaticText( JobDetailsPanel, wxID_ANY, wxT("No. Results Refined :"), wxDefaultPosition, wxDefaultSize, 0 );
	m_staticText783->Wrap( -1 );
	m_staticText783->SetFont( wxFont( 10, 74, 90, 92, false, wxT("Sans") ) );
	
	InfoSizer->Add( m_staticText783, 0, wxALIGN_RIGHT|wxALL, 5 );
	
	NumberResultsRefinedStaticText = new wxStaticText( JobDetailsPanel, wxID_ANY, wxEmptyString, wxDefaultPosition, wxDefaultSize, 0 );
	NumberResultsRefinedStaticText->Wrap( -1 );
	InfoSizer->Add( NumberResultsRefinedStaticText, 0, wxALL, 5 );
	
	m_staticText91 = new wxStaticText( JobDetailsPanel, wxID_ANY, wxT("Angular Search Step :"), wxDefaultPosition, wxDefaultSize, 0 );
	m_staticText91->Wrap( -1 );
	m_staticText91->SetFont( wxFont( 10, 74, 90, 92, false, wxT("Sans") ) );
	
	InfoSizer->Add( m_staticText91, 0, wxALIGN_RIGHT|wxALL, 5 );
	
	AngularSearchStepStaticText = new wxStaticText( JobDetailsPanel, wxID_ANY, wxEmptyString, wxDefaultPosition, wxDefaultSize, 0 );
	AngularSearchStepStaticText->Wrap( -1 );
	InfoSizer->Add( AngularSearchStepStaticText, 0, wxALIGN_LEFT|wxALL, 5 );
	
	m_staticText79 = new wxStaticText( JobDetailsPanel, wxID_ANY, wxT("Search Range X :"), wxDefaultPosition, wxDefaultSize, 0 );
	m_staticText79->Wrap( -1 );
	m_staticText79->SetFont( wxFont( 10, 74, 90, 92, false, wxT("Sans") ) );
	
	InfoSizer->Add( m_staticText79, 0, wxALIGN_RIGHT|wxALL, 5 );
	
	SearchRangeXStaticText = new wxStaticText( JobDetailsPanel, wxID_ANY, wxEmptyString, wxDefaultPosition, wxDefaultSize, 0 );
	SearchRangeXStaticText->Wrap( -1 );
	InfoSizer->Add( SearchRangeXStaticText, 0, wxALIGN_LEFT|wxALL, 5 );
	
	m_staticText99 = new wxStaticText( JobDetailsPanel, wxID_ANY, wxT("Search Range Y :"), wxDefaultPosition, wxDefaultSize, 0 );
	m_staticText99->Wrap( -1 );
	m_staticText99->SetFont( wxFont( 10, 74, 90, 92, false, wxT("Sans") ) );
	
	InfoSizer->Add( m_staticText99, 0, wxALIGN_RIGHT|wxALL, 5 );
	
	SearchRangeYStaticText = new wxStaticText( JobDetailsPanel, wxID_ANY, wxEmptyString, wxDefaultPosition, wxDefaultSize, 0 );
	SearchRangeYStaticText->Wrap( -1 );
	InfoSizer->Add( SearchRangeYStaticText, 0, wxALIGN_LEFT|wxALL, 5 );
	
	m_staticText95 = new wxStaticText( JobDetailsPanel, wxID_ANY, wxT("Class. Res. Limit :"), wxDefaultPosition, wxDefaultSize, 0 );
	m_staticText95->Wrap( -1 );
	m_staticText95->SetFont( wxFont( 10, 74, 90, 92, false, wxT("Sans") ) );
	
	InfoSizer->Add( m_staticText95, 0, wxALIGN_RIGHT|wxALL, 5 );
	
	ClassificationResLimitStaticText = new wxStaticText( JobDetailsPanel, wxID_ANY, wxEmptyString, wxDefaultPosition, wxDefaultSize, 0 );
	ClassificationResLimitStaticText->Wrap( -1 );
	InfoSizer->Add( ClassificationResLimitStaticText, 0, wxALIGN_LEFT|wxALL, 5 );
	
	LargeAstigExpectedLabel = new wxStaticText( JobDetailsPanel, wxID_ANY, wxT("Focus Classify? :"), wxDefaultPosition, wxDefaultSize, 0 );
	LargeAstigExpectedLabel->Wrap( -1 );
	LargeAstigExpectedLabel->SetFont( wxFont( 10, 74, 90, 92, false, wxT("Sans") ) );
	
	InfoSizer->Add( LargeAstigExpectedLabel, 0, wxALIGN_RIGHT|wxALL, 5 );
	
	ShouldFocusClassifyStaticText = new wxStaticText( JobDetailsPanel, wxID_ANY, wxEmptyString, wxDefaultPosition, wxDefaultSize, 0 );
	ShouldFocusClassifyStaticText->Wrap( -1 );
	InfoSizer->Add( ShouldFocusClassifyStaticText, 0, wxALL, 5 );
	
	ToleratedAstigLabel = new wxStaticText( JobDetailsPanel, wxID_ANY, wxT("Sphere X Co-ord :"), wxDefaultPosition, wxDefaultSize, 0 );
	ToleratedAstigLabel->Wrap( -1 );
	ToleratedAstigLabel->SetFont( wxFont( 10, 74, 90, 92, false, wxT("Sans") ) );
	
	InfoSizer->Add( ToleratedAstigLabel, 0, wxALIGN_RIGHT|wxALL, 5 );
	
	SphereXCoordStaticText = new wxStaticText( JobDetailsPanel, wxID_ANY, wxEmptyString, wxDefaultPosition, wxDefaultSize, 0 );
	SphereXCoordStaticText->Wrap( -1 );
	InfoSizer->Add( SphereXCoordStaticText, 0, wxALIGN_LEFT|wxALL, 5 );
	
	NumberOfAveragedFramesLabel = new wxStaticText( JobDetailsPanel, wxID_ANY, wxT("Sphere Y Co-ord :"), wxDefaultPosition, wxDefaultSize, 0 );
	NumberOfAveragedFramesLabel->Wrap( -1 );
	NumberOfAveragedFramesLabel->SetFont( wxFont( 10, 74, 90, 92, false, wxT("Sans") ) );
	
	InfoSizer->Add( NumberOfAveragedFramesLabel, 0, wxALIGN_RIGHT|wxALL, 5 );
	
	SphereYCoordStaticText = new wxStaticText( JobDetailsPanel, wxID_ANY, wxEmptyString, wxDefaultPosition, wxDefaultSize, 0 );
	SphereYCoordStaticText->Wrap( -1 );
	InfoSizer->Add( SphereYCoordStaticText, 0, wxALL, 5 );
	
	m_staticText787 = new wxStaticText( JobDetailsPanel, wxID_ANY, wxT("Sphere Z Co-ord :"), wxDefaultPosition, wxDefaultSize, 0 );
	m_staticText787->Wrap( -1 );
	m_staticText787->SetFont( wxFont( 10, 74, 90, 92, false, wxT("Sans") ) );
	
	InfoSizer->Add( m_staticText787, 0, wxALIGN_RIGHT|wxALL, 5 );
	
	SphereZCoordStaticText = new wxStaticText( JobDetailsPanel, wxID_ANY, wxEmptyString, wxDefaultPosition, wxDefaultSize, 0 );
	SphereZCoordStaticText->Wrap( -1 );
	InfoSizer->Add( SphereZCoordStaticText, 0, wxALL, 5 );
	
	m_staticText789 = new wxStaticText( JobDetailsPanel, wxID_ANY, wxT("Sphere Radius :"), wxDefaultPosition, wxDefaultSize, 0 );
	m_staticText789->Wrap( -1 );
	m_staticText789->SetFont( wxFont( 10, 74, 90, 92, false, wxT("Sans") ) );
	
	InfoSizer->Add( m_staticText789, 0, wxALIGN_RIGHT|wxALL, 5 );
	
	SphereRadiusStaticText = new wxStaticText( JobDetailsPanel, wxID_ANY, wxEmptyString, wxDefaultPosition, wxDefaultSize, 0 );
	SphereRadiusStaticText->Wrap( -1 );
	InfoSizer->Add( SphereRadiusStaticText, 0, wxALL, 5 );
	
	m_staticText791 = new wxStaticText( JobDetailsPanel, wxID_ANY, wxT("Refine CTF? :"), wxDefaultPosition, wxDefaultSize, 0 );
	m_staticText791->Wrap( -1 );
	m_staticText791->SetFont( wxFont( 10, 74, 90, 92, false, wxT("Sans") ) );
	
	InfoSizer->Add( m_staticText791, 0, wxALIGN_RIGHT|wxALL, 5 );
	
	ShouldRefineCTFStaticText = new wxStaticText( JobDetailsPanel, wxID_ANY, wxEmptyString, wxDefaultPosition, wxDefaultSize, 0 );
	ShouldRefineCTFStaticText->Wrap( -1 );
	InfoSizer->Add( ShouldRefineCTFStaticText, 0, wxALL, 5 );
	
	m_staticText793 = new wxStaticText( JobDetailsPanel, wxID_ANY, wxT("Defocus Search Range :"), wxDefaultPosition, wxDefaultSize, 0 );
	m_staticText793->Wrap( -1 );
	m_staticText793->SetFont( wxFont( 10, 74, 90, 92, false, wxT("Sans") ) );
	
	InfoSizer->Add( m_staticText793, 0, wxALIGN_RIGHT|wxALL, 5 );
	
	DefocusSearchRangeStaticText = new wxStaticText( JobDetailsPanel, wxID_ANY, wxEmptyString, wxDefaultPosition, wxDefaultSize, 0 );
	DefocusSearchRangeStaticText->Wrap( -1 );
	InfoSizer->Add( DefocusSearchRangeStaticText, 0, wxALL, 5 );
	
	m_staticText795 = new wxStaticText( JobDetailsPanel, wxID_ANY, wxT("Defocus Search Step :"), wxDefaultPosition, wxDefaultSize, 0 );
	m_staticText795->Wrap( -1 );
	m_staticText795->SetFont( wxFont( 10, 74, 90, 92, false, wxT("Sans") ) );
	
	InfoSizer->Add( m_staticText795, 0, wxALIGN_RIGHT|wxALL, 5 );
	
	DefocusSearchStepStaticText = new wxStaticText( JobDetailsPanel, wxID_ANY, wxEmptyString, wxDefaultPosition, wxDefaultSize, 0 );
	DefocusSearchStepStaticText->Wrap( -1 );
	InfoSizer->Add( DefocusSearchStepStaticText, 0, wxALL, 5 );
	
	m_staticText797 = new wxStaticText( JobDetailsPanel, wxID_ANY, wxT("AutoMask? :"), wxDefaultPosition, wxDefaultSize, 0 );
	m_staticText797->Wrap( -1 );
	m_staticText797->SetFont( wxFont( 10, 74, 90, 92, false, wxT("Sans") ) );
	
	InfoSizer->Add( m_staticText797, 0, wxALIGN_RIGHT|wxALL, 5 );
	
	ShouldAutoMaskStaticText = new wxStaticText( JobDetailsPanel, wxID_ANY, wxEmptyString, wxDefaultPosition, wxDefaultSize, 0 );
	ShouldAutoMaskStaticText->Wrap( -1 );
	InfoSizer->Add( ShouldAutoMaskStaticText, 0, wxALL, 5 );
	
	m_staticText799 = new wxStaticText( JobDetailsPanel, wxID_ANY, wxT("Also Refine Input? :"), wxDefaultPosition, wxDefaultSize, 0 );
	m_staticText799->Wrap( -1 );
	m_staticText799->SetFont( wxFont( 10, 74, 90, 92, false, wxT("Sans") ) );
	
	InfoSizer->Add( m_staticText799, 0, wxALIGN_RIGHT|wxALL, 5 );
	
	RefineInputParamsStaticText = new wxStaticText( JobDetailsPanel, wxID_ANY, wxEmptyString, wxDefaultPosition, wxDefaultSize, 0 );
	RefineInputParamsStaticText->Wrap( -1 );
	InfoSizer->Add( RefineInputParamsStaticText, 0, wxALL, 5 );
	
	m_staticText801 = new wxStaticText( JobDetailsPanel, wxID_ANY, wxT("Use Supplied Mask? :"), wxDefaultPosition, wxDefaultSize, 0 );
	m_staticText801->Wrap( -1 );
	m_staticText801->SetFont( wxFont( 10, 74, 90, 92, false, wxT("Sans") ) );
	
	InfoSizer->Add( m_staticText801, 0, wxALIGN_RIGHT|wxALL, 5 );
	
	UseSuppliedMaskStaticText = new wxStaticText( JobDetailsPanel, wxID_ANY, wxEmptyString, wxDefaultPosition, wxDefaultSize, 0 );
	UseSuppliedMaskStaticText->Wrap( -1 );
	InfoSizer->Add( UseSuppliedMaskStaticText, 0, wxALL, 5 );
	
	m_staticText803 = new wxStaticText( JobDetailsPanel, wxID_ANY, wxT("Mask Asset ID :"), wxDefaultPosition, wxDefaultSize, 0 );
	m_staticText803->Wrap( -1 );
	m_staticText803->SetFont( wxFont( 10, 74, 90, 92, false, wxT("Sans") ) );
	
	InfoSizer->Add( m_staticText803, 0, wxALIGN_RIGHT|wxALL, 5 );
	
	MaskAssetIDStaticText = new wxStaticText( JobDetailsPanel, wxID_ANY, wxEmptyString, wxDefaultPosition, wxDefaultSize, 0 );
	MaskAssetIDStaticText->Wrap( -1 );
	InfoSizer->Add( MaskAssetIDStaticText, 0, wxALL, 5 );
	
	m_staticText805 = new wxStaticText( JobDetailsPanel, wxID_ANY, wxT("Mask Edge Width :"), wxDefaultPosition, wxDefaultSize, 0 );
	m_staticText805->Wrap( -1 );
	m_staticText805->SetFont( wxFont( 10, 74, 90, 92, false, wxT("Sans") ) );
	
	InfoSizer->Add( m_staticText805, 0, wxALIGN_RIGHT|wxALL, 5 );
	
	MaskEdgeWidthStaticText = new wxStaticText( JobDetailsPanel, wxID_ANY, wxEmptyString, wxDefaultPosition, wxDefaultSize, 0 );
	MaskEdgeWidthStaticText->Wrap( -1 );
	InfoSizer->Add( MaskEdgeWidthStaticText, 0, wxALL, 5 );
	
	m_staticText807 = new wxStaticText( JobDetailsPanel, wxID_ANY, wxT("Mask Out. Weight :"), wxDefaultPosition, wxDefaultSize, 0 );
	m_staticText807->Wrap( -1 );
	m_staticText807->SetFont( wxFont( 10, 74, 90, 92, false, wxT("Sans") ) );
	
	InfoSizer->Add( m_staticText807, 0, wxALIGN_RIGHT|wxALL, 5 );
	
	MaskOutsideWeightStaticText = new wxStaticText( JobDetailsPanel, wxID_ANY, wxEmptyString, wxDefaultPosition, wxDefaultSize, 0 );
	MaskOutsideWeightStaticText->Wrap( -1 );
	InfoSizer->Add( MaskOutsideWeightStaticText, 0, wxALL, 5 );
	
	m_staticText809 = new wxStaticText( JobDetailsPanel, wxID_ANY, wxT("Filter Out. Mask? :"), wxDefaultPosition, wxDefaultSize, 0 );
	m_staticText809->Wrap( -1 );
	m_staticText809->SetFont( wxFont( 10, 74, 90, 92, false, wxT("Sans") ) );
	
	InfoSizer->Add( m_staticText809, 0, wxALIGN_RIGHT|wxALL, 5 );
	
	ShouldFilterOutsideMaskStaticText = new wxStaticText( JobDetailsPanel, wxID_ANY, wxEmptyString, wxDefaultPosition, wxDefaultSize, 0 );
	ShouldFilterOutsideMaskStaticText->Wrap( -1 );
	InfoSizer->Add( ShouldFilterOutsideMaskStaticText, 0, wxALL, 5 );
	
	m_staticText811 = new wxStaticText( JobDetailsPanel, wxID_ANY, wxT("Mask Filter Res. :"), wxDefaultPosition, wxDefaultSize, 0 );
	m_staticText811->Wrap( -1 );
	m_staticText811->SetFont( wxFont( 10, 74, 90, 92, false, wxT("Sans") ) );
	
	InfoSizer->Add( m_staticText811, 0, wxALIGN_RIGHT|wxALL, 5 );
	
	MaskFilterResolutionStaticText = new wxStaticText( JobDetailsPanel, wxID_ANY, wxEmptyString, wxDefaultPosition, wxDefaultSize, 0 );
	MaskFilterResolutionStaticText->Wrap( -1 );
	InfoSizer->Add( MaskFilterResolutionStaticText, 0, wxALL, 5 );
	
	m_staticText813 = new wxStaticText( JobDetailsPanel, wxID_ANY, wxT("Reconstruction ID :"), wxDefaultPosition, wxDefaultSize, 0 );
	m_staticText813->Wrap( -1 );
	m_staticText813->SetFont( wxFont( 10, 74, 90, 92, false, wxT("Sans") ) );
	
	InfoSizer->Add( m_staticText813, 0, wxALIGN_RIGHT|wxALL, 5 );
	
	ReconstructionIDStaticText = new wxStaticText( JobDetailsPanel, wxID_ANY, wxEmptyString, wxDefaultPosition, wxDefaultSize, 0 );
	ReconstructionIDStaticText->Wrap( -1 );
	InfoSizer->Add( ReconstructionIDStaticText, 0, wxALL, 5 );
	
	m_staticText815 = new wxStaticText( JobDetailsPanel, wxID_ANY, wxT("Inner Mask Rad. :"), wxDefaultPosition, wxDefaultSize, 0 );
	m_staticText815->Wrap( -1 );
	m_staticText815->SetFont( wxFont( 10, 74, 90, 92, false, wxT("Sans") ) );
	
	InfoSizer->Add( m_staticText815, 0, wxALIGN_RIGHT|wxALL, 5 );
	
	InnerMaskRadiusStaticText = new wxStaticText( JobDetailsPanel, wxID_ANY, wxEmptyString, wxDefaultPosition, wxDefaultSize, 0 );
	InnerMaskRadiusStaticText->Wrap( -1 );
	InfoSizer->Add( InnerMaskRadiusStaticText, 0, wxALL, 5 );
	
	m_staticText817 = new wxStaticText( JobDetailsPanel, wxID_ANY, wxT("Outer Mask Rad. :"), wxDefaultPosition, wxDefaultSize, 0 );
	m_staticText817->Wrap( -1 );
	m_staticText817->SetFont( wxFont( 10, 74, 90, 92, false, wxT("Sans") ) );
	
	InfoSizer->Add( m_staticText817, 0, wxALIGN_RIGHT|wxALL, 5 );
	
	OuterMaskRadiusStaticText = new wxStaticText( JobDetailsPanel, wxID_ANY, wxEmptyString, wxDefaultPosition, wxDefaultSize, 0 );
	OuterMaskRadiusStaticText->Wrap( -1 );
	InfoSizer->Add( OuterMaskRadiusStaticText, 0, wxALL, 5 );
	
	m_staticText820 = new wxStaticText( JobDetailsPanel, wxID_ANY, wxT("Res. Cut-Off :"), wxDefaultPosition, wxDefaultSize, 0 );
	m_staticText820->Wrap( -1 );
	m_staticText820->SetFont( wxFont( 10, 74, 90, 92, false, wxT("Sans") ) );
	
	InfoSizer->Add( m_staticText820, 0, wxALIGN_RIGHT|wxALL, 5 );
	
	ResolutionCutOffStaticText = new wxStaticText( JobDetailsPanel, wxID_ANY, wxEmptyString, wxDefaultPosition, wxDefaultSize, 0 );
	ResolutionCutOffStaticText->Wrap( -1 );
	InfoSizer->Add( ResolutionCutOffStaticText, 0, wxALL, 5 );
	
	Score = new wxStaticText( JobDetailsPanel, wxID_ANY, wxT("Weight Constant :"), wxDefaultPosition, wxDefaultSize, 0 );
	Score->Wrap( -1 );
	Score->SetFont( wxFont( 10, 74, 90, 92, false, wxT("Sans") ) );
	
	InfoSizer->Add( Score, 0, wxALIGN_RIGHT|wxALL, 5 );
	
	ScoreWeightConstantStaticText = new wxStaticText( JobDetailsPanel, wxID_ANY, wxEmptyString, wxDefaultPosition, wxDefaultSize, 0 );
	ScoreWeightConstantStaticText->Wrap( -1 );
	InfoSizer->Add( ScoreWeightConstantStaticText, 0, wxALL, 5 );
	
	m_staticText823 = new wxStaticText( JobDetailsPanel, wxID_ANY, wxT("Adjust Scores? :"), wxDefaultPosition, wxDefaultSize, 0 );
	m_staticText823->Wrap( -1 );
	m_staticText823->SetFont( wxFont( 10, 74, 90, 92, false, wxT("Sans") ) );
	
	InfoSizer->Add( m_staticText823, 0, wxALIGN_RIGHT|wxALL, 5 );
	
	AdjustScoresStaticText = new wxStaticText( JobDetailsPanel, wxID_ANY, wxEmptyString, wxDefaultPosition, wxDefaultSize, 0 );
	AdjustScoresStaticText->Wrap( -1 );
	InfoSizer->Add( AdjustScoresStaticText, 0, wxALL, 5 );
	
	m_staticText825 = new wxStaticText( JobDetailsPanel, wxID_ANY, wxT("Crop Images? :"), wxDefaultPosition, wxDefaultSize, 0 );
	m_staticText825->Wrap( -1 );
	m_staticText825->SetFont( wxFont( 10, 74, 90, 92, false, wxT("Sans") ) );
	
	InfoSizer->Add( m_staticText825, 0, wxALIGN_RIGHT|wxALL, 5 );
	
	ShouldCropImagesStaticText = new wxStaticText( JobDetailsPanel, wxID_ANY, wxEmptyString, wxDefaultPosition, wxDefaultSize, 0 );
	ShouldCropImagesStaticText->Wrap( -1 );
	InfoSizer->Add( ShouldCropImagesStaticText, 0, wxALL, 5 );
	
	m_staticText827 = new wxStaticText( JobDetailsPanel, wxID_ANY, wxT("Likelihood Blur? :"), wxDefaultPosition, wxDefaultSize, 0 );
	m_staticText827->Wrap( -1 );
	m_staticText827->SetFont( wxFont( 10, 74, 90, 92, false, wxT("Sans") ) );
	
	InfoSizer->Add( m_staticText827, 0, wxALIGN_RIGHT|wxALL, 5 );
	
	ShouldLikelihoodBlurStaticText = new wxStaticText( JobDetailsPanel, wxID_ANY, wxEmptyString, wxDefaultPosition, wxDefaultSize, 0 );
	ShouldLikelihoodBlurStaticText->Wrap( -1 );
	InfoSizer->Add( ShouldLikelihoodBlurStaticText, 0, wxALL, 5 );
	
	m_staticText829 = new wxStaticText( JobDetailsPanel, wxID_ANY, wxT("Smoothing Factor :"), wxDefaultPosition, wxDefaultSize, 0 );
	m_staticText829->Wrap( -1 );
	m_staticText829->SetFont( wxFont( 10, 74, 90, 92, false, wxT("Sans") ) );
	
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

ShowCTFResultsParentPanel::ShowCTFResultsParentPanel( wxWindow* parent, wxWindowID id, const wxPoint& pos, const wxSize& size, long style ) : wxPanel( parent, id, pos, size, style )
{
	wxBoxSizer* bSizer92;
	bSizer92 = new wxBoxSizer( wxVERTICAL );
	
	m_splitter16 = new wxSplitterWindow( this, wxID_ANY, wxDefaultPosition, wxDefaultSize, wxSP_3D );
	m_splitter16->SetSashGravity( 0.5 );
	m_splitter16->Connect( wxEVT_IDLE, wxIdleEventHandler( ShowCTFResultsParentPanel::m_splitter16OnIdle ), NULL, this );
	
	m_panel87 = new wxPanel( m_splitter16, wxID_ANY, wxDefaultPosition, wxDefaultSize, wxTAB_TRAVERSAL );
	wxBoxSizer* bSizer301;
	bSizer301 = new wxBoxSizer( wxVERTICAL );
	
	m_splitter15 = new wxSplitterWindow( m_panel87, wxID_ANY, wxDefaultPosition, wxDefaultSize, wxSP_3D );
	m_splitter15->SetSashGravity( 0.5 );
	m_splitter15->Connect( wxEVT_IDLE, wxIdleEventHandler( ShowCTFResultsParentPanel::m_splitter15OnIdle ), NULL, this );
	
	m_panel88 = new wxPanel( m_splitter15, wxID_ANY, wxDefaultPosition, wxDefaultSize, wxTAB_TRAVERSAL );
	wxBoxSizer* bSizer302;
	bSizer302 = new wxBoxSizer( wxVERTICAL );
	
	m_staticText377 = new wxStaticText( m_panel88, wxID_ANY, wxT("2D CTF Fit Result"), wxDefaultPosition, wxDefaultSize, 0 );
	m_staticText377->Wrap( -1 );
	m_staticText377->SetFont( wxFont( 10, 74, 90, 92, false, wxT("Sans") ) );
	
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
	m_staticText378->SetFont( wxFont( 10, 74, 90, 92, false, wxT("Sans") ) );
	
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
	m_staticText379->SetFont( wxFont( 10, 74, 90, 92, false, wxT("Sans") ) );
	
	bSizer304->Add( m_staticText379, 0, wxALL, 5 );
	
	m_staticline78 = new wxStaticLine( m_panel86, wxID_ANY, wxDefaultPosition, wxDefaultSize, wxLI_HORIZONTAL );
	bSizer304->Add( m_staticline78, 0, wxEXPAND | wxALL, 5 );
	
	wxBoxSizer* bSizer305;
	bSizer305 = new wxBoxSizer( wxHORIZONTAL );
	
	wxGridSizer* gSizer14;
	gSizer14 = new wxGridSizer( 0, 4, 0, 0 );
	
	m_staticText380 = new wxStaticText( m_panel86, wxID_ANY, wxT("\tDefocus 1 :"), wxDefaultPosition, wxDefaultSize, 0 );
	m_staticText380->Wrap( -1 );
	m_staticText380->SetFont( wxFont( 10, 74, 90, 92, false, wxT("Sans") ) );
	
	gSizer14->Add( m_staticText380, 0, wxALIGN_RIGHT|wxALL, 5 );
	
	Defocus1Text = new wxStaticText( m_panel86, wxID_ANY, wxEmptyString, wxDefaultPosition, wxDefaultSize, 0 );
	Defocus1Text->Wrap( -1 );
	gSizer14->Add( Defocus1Text, 0, wxALIGN_LEFT|wxALL, 5 );
	
	m_staticText389 = new wxStaticText( m_panel86, wxID_ANY, wxT("Score :"), wxDefaultPosition, wxDefaultSize, 0 );
	m_staticText389->Wrap( -1 );
	m_staticText389->SetFont( wxFont( 10, 74, 90, 92, false, wxT("Sans") ) );
	
	gSizer14->Add( m_staticText389, 0, wxALIGN_RIGHT|wxALL, 5 );
	
	ScoreText = new wxStaticText( m_panel86, wxID_ANY, wxEmptyString, wxDefaultPosition, wxDefaultSize, 0 );
	ScoreText->Wrap( -1 );
	gSizer14->Add( ScoreText, 0, wxALL, 5 );
	
	m_staticText382 = new wxStaticText( m_panel86, wxID_ANY, wxT("Defocus 2 :"), wxDefaultPosition, wxDefaultSize, 0 );
	m_staticText382->Wrap( -1 );
	m_staticText382->SetFont( wxFont( 10, 74, 90, 92, false, wxT("Sans") ) );
	
	gSizer14->Add( m_staticText382, 0, wxALIGN_RIGHT|wxALL, 5 );
	
	Defocus2Text = new wxStaticText( m_panel86, wxID_ANY, wxEmptyString, wxDefaultPosition, wxDefaultSize, 0 );
	Defocus2Text->Wrap( -1 );
	gSizer14->Add( Defocus2Text, 0, wxALL, 5 );
	
	m_staticText391 = new wxStaticText( m_panel86, wxID_ANY, wxT("Fit Res. :"), wxDefaultPosition, wxDefaultSize, 0 );
	m_staticText391->Wrap( -1 );
	m_staticText391->SetFont( wxFont( 10, 74, 90, 92, false, wxT("Sans") ) );
	
	gSizer14->Add( m_staticText391, 0, wxALIGN_RIGHT|wxALL, 5 );
	
	FitResText = new wxStaticText( m_panel86, wxID_ANY, wxEmptyString, wxDefaultPosition, wxDefaultSize, 0 );
	FitResText->Wrap( -1 );
	gSizer14->Add( FitResText, 0, wxALL, 5 );
	
	m_staticText384 = new wxStaticText( m_panel86, wxID_ANY, wxT("Angle :"), wxDefaultPosition, wxDefaultSize, 0 );
	m_staticText384->Wrap( -1 );
	m_staticText384->SetFont( wxFont( 10, 74, 90, 92, false, wxT("Sans") ) );
	
	gSizer14->Add( m_staticText384, 0, wxALIGN_RIGHT|wxALL, 5 );
	
	AngleText = new wxStaticText( m_panel86, wxID_ANY, wxEmptyString, wxDefaultPosition, wxDefaultSize, 0 );
	AngleText->Wrap( -1 );
	gSizer14->Add( AngleText, 0, wxALL, 5 );
	
	m_staticText393 = new wxStaticText( m_panel86, wxID_ANY, wxT("Alias Res. : "), wxDefaultPosition, wxDefaultSize, 0 );
	m_staticText393->Wrap( -1 );
	m_staticText393->SetFont( wxFont( 10, 74, 90, 92, false, wxT("Sans") ) );
	
	gSizer14->Add( m_staticText393, 0, wxALIGN_RIGHT|wxALL, 5 );
	
	AliasResText = new wxStaticText( m_panel86, wxID_ANY, wxEmptyString, wxDefaultPosition, wxDefaultSize, 0 );
	AliasResText->Wrap( -1 );
	gSizer14->Add( AliasResText, 0, wxALL, 5 );
	
	m_staticText386 = new wxStaticText( m_panel86, wxID_ANY, wxT("Phase Shift :"), wxDefaultPosition, wxDefaultSize, 0 );
	m_staticText386->Wrap( -1 );
	m_staticText386->SetFont( wxFont( 10, 74, 90, 92, false, wxT("Sans") ) );
	
	gSizer14->Add( m_staticText386, 0, wxALIGN_RIGHT|wxALL, 5 );
	
	PhaseShiftText = new wxStaticText( m_panel86, wxID_ANY, wxEmptyString, wxDefaultPosition, wxDefaultSize, 0 );
	PhaseShiftText->Wrap( -1 );
	gSizer14->Add( PhaseShiftText, 0, wxALL, 5 );
	
	IcinessLabel = new wxStaticText( m_panel86, wxID_ANY, wxT("Iciness :"), wxDefaultPosition, wxDefaultSize, 0 );
	IcinessLabel->Wrap( -1 );
	IcinessLabel->SetFont( wxFont( 10, 74, 90, 92, false, wxT("Sans") ) );
	
	gSizer14->Add( IcinessLabel, 0, wxALIGN_RIGHT|wxALL, 5 );
	
	IcinessStaticText = new wxStaticText( m_panel86, wxID_ANY, wxEmptyString, wxDefaultPosition, wxDefaultSize, 0 );
	IcinessStaticText->Wrap( -1 );
	gSizer14->Add( IcinessStaticText, 0, wxALL, 5 );
	
	
	bSizer305->Add( gSizer14, 0, 0, 5 );
	
	
	bSizer305->Add( 0, 0, 1, wxEXPAND, 5 );
	
	
	bSizer304->Add( bSizer305, 0, wxEXPAND, 5 );
	
	m_staticline83 = new wxStaticLine( m_panel86, wxID_ANY, wxDefaultPosition, wxDefaultSize, wxLI_HORIZONTAL );
	bSizer304->Add( m_staticline83, 0, wxEXPAND | wxALL, 5 );
	
	wxBoxSizer* bSizer306;
	bSizer306 = new wxBoxSizer( wxHORIZONTAL );
	
	m_staticText394 = new wxStaticText( m_panel86, wxID_ANY, wxT("Image / Aligned Movie Sum"), wxDefaultPosition, wxDefaultSize, 0 );
	m_staticText394->Wrap( -1 );
	m_staticText394->SetFont( wxFont( 10, 74, 90, 92, false, wxT("Sans") ) );
	
	bSizer306->Add( m_staticText394, 0, wxALL, 5 );
	
	ImageFileText = new wxStaticText( m_panel86, wxID_ANY, wxEmptyString, wxDefaultPosition, wxDefaultSize, 0 );
	ImageFileText->Wrap( -1 );
	ImageFileText->SetFont( wxFont( 10, 74, 90, 92, false, wxT("Sans") ) );
	
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

ShowCTFResultsParentPanel::~ShowCTFResultsParentPanel()
{
}

Refine2DResultsPanelParent::Refine2DResultsPanelParent( wxWindow* parent, wxWindowID id, const wxPoint& pos, const wxSize& size, long style ) : wxPanel( parent, id, pos, size, style )
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
	m_staticText72->SetFont( wxFont( 10, 74, 90, 92, false, wxT("Sans") ) );
	
	InfoSizer->Add( m_staticText72, 0, wxALIGN_RIGHT|wxALL, 5 );
	
	ClassificationIDStaticText = new wxStaticText( JobDetailsPanel, wxID_ANY, wxEmptyString, wxDefaultPosition, wxDefaultSize, 0 );
	ClassificationIDStaticText->Wrap( -1 );
	InfoSizer->Add( ClassificationIDStaticText, 0, wxALIGN_LEFT|wxALL, 5 );
	
	m_staticText74 = new wxStaticText( JobDetailsPanel, wxID_ANY, wxT("Date of Run :"), wxDefaultPosition, wxDefaultSize, 0 );
	m_staticText74->Wrap( -1 );
	m_staticText74->SetFont( wxFont( 10, 74, 90, 92, false, wxT("Sans") ) );
	
	InfoSizer->Add( m_staticText74, 0, wxALIGN_RIGHT|wxALL, 5 );
	
	DateOfRunStaticText = new wxStaticText( JobDetailsPanel, wxID_ANY, wxEmptyString, wxDefaultPosition, wxDefaultSize, 0 );
	DateOfRunStaticText->Wrap( -1 );
	InfoSizer->Add( DateOfRunStaticText, 0, wxALL, 5 );
	
	m_staticText93 = new wxStaticText( JobDetailsPanel, wxID_ANY, wxT("Time Of Run :"), wxDefaultPosition, wxDefaultSize, 0 );
	m_staticText93->Wrap( -1 );
	m_staticText93->SetFont( wxFont( 10, 74, 90, 92, false, wxT("Sans") ) );
	
	InfoSizer->Add( m_staticText93, 0, wxALIGN_RIGHT|wxALL, 5 );
	
	TimeOfRunStaticText = new wxStaticText( JobDetailsPanel, wxID_ANY, wxEmptyString, wxDefaultPosition, wxDefaultSize, 0 );
	TimeOfRunStaticText->Wrap( -1 );
	InfoSizer->Add( TimeOfRunStaticText, 0, wxALIGN_LEFT|wxALL, 5 );
	
	m_staticText83 = new wxStaticText( JobDetailsPanel, wxID_ANY, wxT("Refinement Package ID :"), wxDefaultPosition, wxDefaultSize, 0 );
	m_staticText83->Wrap( -1 );
	m_staticText83->SetFont( wxFont( 10, 74, 90, 92, false, wxT("Sans") ) );
	
	InfoSizer->Add( m_staticText83, 0, wxALIGN_RIGHT|wxALL, 5 );
	
	RefinementPackageIDStaticText = new wxStaticText( JobDetailsPanel, wxID_ANY, wxEmptyString, wxDefaultPosition, wxDefaultSize, 0 );
	RefinementPackageIDStaticText->Wrap( -1 );
	InfoSizer->Add( RefinementPackageIDStaticText, 0, wxALIGN_LEFT|wxALL, 5 );
	
	m_staticText82 = new wxStaticText( JobDetailsPanel, wxID_ANY, wxT("Ref. Classification ID :"), wxDefaultPosition, wxDefaultSize, 0 );
	m_staticText82->Wrap( -1 );
	m_staticText82->SetFont( wxFont( 10, 74, 90, 92, false, wxT("Sans") ) );
	
	InfoSizer->Add( m_staticText82, 0, wxALIGN_RIGHT|wxALL, 5 );
	
	StartClassificationIDStaticText = new wxStaticText( JobDetailsPanel, wxID_ANY, wxEmptyString, wxDefaultPosition, wxDefaultSize, 0 );
	StartClassificationIDStaticText->Wrap( -1 );
	InfoSizer->Add( StartClassificationIDStaticText, 0, wxALIGN_LEFT|wxALL, 5 );
	
	m_staticText78 = new wxStaticText( JobDetailsPanel, wxID_ANY, wxT("No. Classes :"), wxDefaultPosition, wxDefaultSize, 0 );
	m_staticText78->Wrap( -1 );
	m_staticText78->SetFont( wxFont( 10, 74, 90, 92, false, wxT("Sans") ) );
	
	InfoSizer->Add( m_staticText78, 0, wxALIGN_RIGHT|wxALL, 5 );
	
	NumberClassesStaticText = new wxStaticText( JobDetailsPanel, wxID_ANY, wxEmptyString, wxDefaultPosition, wxDefaultSize, 0 );
	NumberClassesStaticText->Wrap( -1 );
	InfoSizer->Add( NumberClassesStaticText, 0, wxALIGN_LEFT|wxALL, 5 );
	
	m_staticText96 = new wxStaticText( JobDetailsPanel, wxID_ANY, wxT("No. Input Particles :"), wxDefaultPosition, wxDefaultSize, 0 );
	m_staticText96->Wrap( -1 );
	m_staticText96->SetFont( wxFont( 10, 74, 90, 92, false, wxT("Sans") ) );
	
	InfoSizer->Add( m_staticText96, 0, wxALIGN_RIGHT|wxALL, 5 );
	
	NumberParticlesStaticText = new wxStaticText( JobDetailsPanel, wxID_ANY, wxEmptyString, wxDefaultPosition, wxDefaultSize, 0 );
	NumberParticlesStaticText->Wrap( -1 );
	InfoSizer->Add( NumberParticlesStaticText, 0, wxALIGN_LEFT|wxALL, 5 );
	
	m_staticText85 = new wxStaticText( JobDetailsPanel, wxID_ANY, wxT("Low Res. Limit :"), wxDefaultPosition, wxDefaultSize, 0 );
	m_staticText85->Wrap( -1 );
	m_staticText85->SetFont( wxFont( 10, 74, 90, 92, false, wxT("Sans") ) );
	
	InfoSizer->Add( m_staticText85, 0, wxALIGN_RIGHT|wxALL, 5 );
	
	LowResLimitStaticText = new wxStaticText( JobDetailsPanel, wxID_ANY, wxEmptyString, wxDefaultPosition, wxDefaultSize, 0 );
	LowResLimitStaticText->Wrap( -1 );
	InfoSizer->Add( LowResLimitStaticText, 0, wxALIGN_LEFT|wxALL, 5 );
	
	m_staticText87 = new wxStaticText( JobDetailsPanel, wxID_ANY, wxT("High Res. Limit :"), wxDefaultPosition, wxDefaultSize, 0 );
	m_staticText87->Wrap( -1 );
	m_staticText87->SetFont( wxFont( 10, 74, 90, 92, false, wxT("Sans") ) );
	
	InfoSizer->Add( m_staticText87, 0, wxALIGN_RIGHT|wxALL, 5 );
	
	HighResLimitStaticText = new wxStaticText( JobDetailsPanel, wxID_ANY, wxEmptyString, wxDefaultPosition, wxDefaultSize, 0 );
	HighResLimitStaticText->Wrap( -1 );
	InfoSizer->Add( HighResLimitStaticText, 0, wxALIGN_LEFT|wxALL, 5 );
	
	m_staticText89 = new wxStaticText( JobDetailsPanel, wxID_ANY, wxT("Mask Radius :"), wxDefaultPosition, wxDefaultSize, 0 );
	m_staticText89->Wrap( -1 );
	m_staticText89->SetFont( wxFont( 10, 74, 90, 92, false, wxT("Sans") ) );
	
	InfoSizer->Add( m_staticText89, 0, wxALIGN_RIGHT|wxALL, 5 );
	
	MaskRadiusStaticText = new wxStaticText( JobDetailsPanel, wxID_ANY, wxEmptyString, wxDefaultPosition, wxDefaultSize, 0 );
	MaskRadiusStaticText->Wrap( -1 );
	InfoSizer->Add( MaskRadiusStaticText, 0, wxALIGN_LEFT|wxALL, 5 );
	
	m_staticText91 = new wxStaticText( JobDetailsPanel, wxID_ANY, wxT("Angular Search Step :"), wxDefaultPosition, wxDefaultSize, 0 );
	m_staticText91->Wrap( -1 );
	m_staticText91->SetFont( wxFont( 10, 74, 90, 92, false, wxT("Sans") ) );
	
	InfoSizer->Add( m_staticText91, 0, wxALIGN_RIGHT|wxALL, 5 );
	
	AngularSearchStepStaticText = new wxStaticText( JobDetailsPanel, wxID_ANY, wxEmptyString, wxDefaultPosition, wxDefaultSize, 0 );
	AngularSearchStepStaticText->Wrap( -1 );
	InfoSizer->Add( AngularSearchStepStaticText, 0, wxALIGN_LEFT|wxALL, 5 );
	
	m_staticText79 = new wxStaticText( JobDetailsPanel, wxID_ANY, wxT("Search Range X :"), wxDefaultPosition, wxDefaultSize, 0 );
	m_staticText79->Wrap( -1 );
	m_staticText79->SetFont( wxFont( 10, 74, 90, 92, false, wxT("Sans") ) );
	
	InfoSizer->Add( m_staticText79, 0, wxALIGN_RIGHT|wxALL, 5 );
	
	SearchRangeXStaticText = new wxStaticText( JobDetailsPanel, wxID_ANY, wxEmptyString, wxDefaultPosition, wxDefaultSize, 0 );
	SearchRangeXStaticText->Wrap( -1 );
	InfoSizer->Add( SearchRangeXStaticText, 0, wxALIGN_LEFT|wxALL, 5 );
	
	m_staticText95 = new wxStaticText( JobDetailsPanel, wxID_ANY, wxT("Smoothing Factor :"), wxDefaultPosition, wxDefaultSize, 0 );
	m_staticText95->Wrap( -1 );
	m_staticText95->SetFont( wxFont( 10, 74, 90, 92, false, wxT("Sans") ) );
	
	InfoSizer->Add( m_staticText95, 0, wxALIGN_RIGHT|wxALL, 5 );
	
	SmoothingFactorStaticText = new wxStaticText( JobDetailsPanel, wxID_ANY, wxEmptyString, wxDefaultPosition, wxDefaultSize, 0 );
	SmoothingFactorStaticText->Wrap( -1 );
	InfoSizer->Add( SmoothingFactorStaticText, 0, wxALIGN_LEFT|wxALL, 5 );
	
	LargeAstigExpectedLabel = new wxStaticText( JobDetailsPanel, wxID_ANY, wxT("Exclude Blank Edges? :"), wxDefaultPosition, wxDefaultSize, 0 );
	LargeAstigExpectedLabel->Wrap( -1 );
	LargeAstigExpectedLabel->SetFont( wxFont( 10, 74, 90, 92, false, wxT("Sans") ) );
	
	InfoSizer->Add( LargeAstigExpectedLabel, 0, wxALIGN_RIGHT|wxALL, 5 );
	
	ExcludeBlankEdgesStaticText = new wxStaticText( JobDetailsPanel, wxID_ANY, wxEmptyString, wxDefaultPosition, wxDefaultSize, 0 );
	ExcludeBlankEdgesStaticText->Wrap( -1 );
	InfoSizer->Add( ExcludeBlankEdgesStaticText, 0, wxALL, 5 );
	
	m_staticText99 = new wxStaticText( JobDetailsPanel, wxID_ANY, wxT("Search Range Y :"), wxDefaultPosition, wxDefaultSize, 0 );
	m_staticText99->Wrap( -1 );
	m_staticText99->SetFont( wxFont( 10, 74, 90, 92, false, wxT("Sans") ) );
	
	InfoSizer->Add( m_staticText99, 0, wxALIGN_RIGHT|wxALL, 5 );
	
	SearchRangeYStaticText = new wxStaticText( JobDetailsPanel, wxID_ANY, wxEmptyString, wxDefaultPosition, wxDefaultSize, 0 );
	SearchRangeYStaticText->Wrap( -1 );
	InfoSizer->Add( SearchRangeYStaticText, 0, wxALIGN_LEFT|wxALL, 5 );
	
	ToleratedAstigLabel = new wxStaticText( JobDetailsPanel, wxID_ANY, wxT("Auto Percent Used? :"), wxDefaultPosition, wxDefaultSize, 0 );
	ToleratedAstigLabel->Wrap( -1 );
	ToleratedAstigLabel->SetFont( wxFont( 10, 74, 90, 92, false, wxT("Sans") ) );
	
	InfoSizer->Add( ToleratedAstigLabel, 0, wxALIGN_RIGHT|wxALL, 5 );
	
	AutoPercentUsedStaticText = new wxStaticText( JobDetailsPanel, wxID_ANY, wxEmptyString, wxDefaultPosition, wxDefaultSize, 0 );
	AutoPercentUsedStaticText->Wrap( -1 );
	InfoSizer->Add( AutoPercentUsedStaticText, 0, wxALIGN_LEFT|wxALL, 5 );
	
	NumberOfAveragedFramesLabel = new wxStaticText( JobDetailsPanel, wxID_ANY, wxT("Percent Used :"), wxDefaultPosition, wxDefaultSize, 0 );
	NumberOfAveragedFramesLabel->Wrap( -1 );
	NumberOfAveragedFramesLabel->SetFont( wxFont( 10, 74, 90, 92, false, wxT("Sans") ) );
	
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
	m_staticText321->SetFont( wxFont( 10, 74, 90, 92, false, wxT("Sans") ) );
	
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
	ClassNumberStaticText->SetFont( wxFont( 10, 74, 90, 92, false, wxT("Sans") ) );
	
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

PickingResultsDisplayParentPanel::PickingResultsDisplayParentPanel( wxWindow* parent, wxWindowID id, const wxPoint& pos, const wxSize& size, long style ) : wxPanel( parent, id, pos, size, style )
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
	CirclesAroundParticlesCheckBox->Connect( wxEVT_COMMAND_CHECKBOX_CLICKED, wxCommandEventHandler( PickingResultsDisplayParentPanel::OnCirclesAroundParticlesCheckBox ), NULL, this );
	ScaleBarCheckBox->Connect( wxEVT_COMMAND_CHECKBOX_CLICKED, wxCommandEventHandler( PickingResultsDisplayParentPanel::OnScaleBarCheckBox ), NULL, this );
	HighPassFilterCheckBox->Connect( wxEVT_COMMAND_CHECKBOX_CLICKED, wxCommandEventHandler( PickingResultsDisplayParentPanel::OnHighPassFilterCheckBox ), NULL, this );
	LowPassFilterCheckBox->Connect( wxEVT_COMMAND_CHECKBOX_CLICKED, wxCommandEventHandler( PickingResultsDisplayParentPanel::OnLowPassFilterCheckBox ), NULL, this );
	LowResFilterTextCtrl->Connect( wxEVT_KILL_FOCUS, wxFocusEventHandler( PickingResultsDisplayParentPanel::OnLowPassKillFocus ), NULL, this );
	LowResFilterTextCtrl->Connect( wxEVT_COMMAND_TEXT_ENTER, wxCommandEventHandler( PickingResultsDisplayParentPanel::OnLowPassEnter ), NULL, this );
	WienerFilterCheckBox->Connect( wxEVT_COMMAND_CHECKBOX_CLICKED, wxCommandEventHandler( PickingResultsDisplayParentPanel::OnWienerFilterCheckBox ), NULL, this );
	UndoButton->Connect( wxEVT_COMMAND_BUTTON_CLICKED, wxCommandEventHandler( PickingResultsDisplayParentPanel::OnUndoButtonClick ), NULL, this );
	RedoButton->Connect( wxEVT_COMMAND_BUTTON_CLICKED, wxCommandEventHandler( PickingResultsDisplayParentPanel::OnRedoButtonClick ), NULL, this );
}

PickingResultsDisplayParentPanel::~PickingResultsDisplayParentPanel()
{
	// Disconnect Events
	CirclesAroundParticlesCheckBox->Disconnect( wxEVT_COMMAND_CHECKBOX_CLICKED, wxCommandEventHandler( PickingResultsDisplayParentPanel::OnCirclesAroundParticlesCheckBox ), NULL, this );
	ScaleBarCheckBox->Disconnect( wxEVT_COMMAND_CHECKBOX_CLICKED, wxCommandEventHandler( PickingResultsDisplayParentPanel::OnScaleBarCheckBox ), NULL, this );
	HighPassFilterCheckBox->Disconnect( wxEVT_COMMAND_CHECKBOX_CLICKED, wxCommandEventHandler( PickingResultsDisplayParentPanel::OnHighPassFilterCheckBox ), NULL, this );
	LowPassFilterCheckBox->Disconnect( wxEVT_COMMAND_CHECKBOX_CLICKED, wxCommandEventHandler( PickingResultsDisplayParentPanel::OnLowPassFilterCheckBox ), NULL, this );
	LowResFilterTextCtrl->Disconnect( wxEVT_KILL_FOCUS, wxFocusEventHandler( PickingResultsDisplayParentPanel::OnLowPassKillFocus ), NULL, this );
	LowResFilterTextCtrl->Disconnect( wxEVT_COMMAND_TEXT_ENTER, wxCommandEventHandler( PickingResultsDisplayParentPanel::OnLowPassEnter ), NULL, this );
	WienerFilterCheckBox->Disconnect( wxEVT_COMMAND_CHECKBOX_CLICKED, wxCommandEventHandler( PickingResultsDisplayParentPanel::OnWienerFilterCheckBox ), NULL, this );
	UndoButton->Disconnect( wxEVT_COMMAND_BUTTON_CLICKED, wxCommandEventHandler( PickingResultsDisplayParentPanel::OnUndoButtonClick ), NULL, this );
	RedoButton->Disconnect( wxEVT_COMMAND_BUTTON_CLICKED, wxCommandEventHandler( PickingResultsDisplayParentPanel::OnRedoButtonClick ), NULL, this );
	
}

FindCTFResultsPanel::FindCTFResultsPanel( wxWindow* parent, wxWindowID id, const wxPoint& pos, const wxSize& size, long style ) : wxPanel( parent, id, pos, size, style )
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
	
	PlotResultsButton = new NoFocusBitmapButton( m_panel13, wxID_ANY, wxNullBitmap, wxDefaultPosition, wxDefaultSize, wxBU_AUTODRAW );
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
	JobDetailsPanel->Hide();
	
	wxBoxSizer* bSizer101;
	bSizer101 = new wxBoxSizer( wxVERTICAL );
	
	InfoSizer = new wxFlexGridSizer( 0, 6, 0, 0 );
	InfoSizer->SetFlexibleDirection( wxBOTH );
	InfoSizer->SetNonFlexibleGrowMode( wxFLEX_GROWMODE_SPECIFIED );
	
	m_staticText72 = new wxStaticText( JobDetailsPanel, wxID_ANY, wxT("Estimation ID :"), wxDefaultPosition, wxDefaultSize, 0 );
	m_staticText72->Wrap( -1 );
	m_staticText72->SetFont( wxFont( 10, 74, 90, 92, false, wxT("Sans") ) );
	
	InfoSizer->Add( m_staticText72, 0, wxALIGN_RIGHT|wxALL, 5 );
	
	EstimationIDStaticText = new wxStaticText( JobDetailsPanel, wxID_ANY, wxEmptyString, wxDefaultPosition, wxDefaultSize, 0 );
	EstimationIDStaticText->Wrap( -1 );
	InfoSizer->Add( EstimationIDStaticText, 0, wxALIGN_LEFT|wxALL, 5 );
	
	m_staticText74 = new wxStaticText( JobDetailsPanel, wxID_ANY, wxT("Date of Run :"), wxDefaultPosition, wxDefaultSize, 0 );
	m_staticText74->Wrap( -1 );
	m_staticText74->SetFont( wxFont( 10, 74, 90, 92, false, wxT("Sans") ) );
	
	InfoSizer->Add( m_staticText74, 0, wxALIGN_RIGHT|wxALL, 5 );
	
	DateOfRunStaticText = new wxStaticText( JobDetailsPanel, wxID_ANY, wxEmptyString, wxDefaultPosition, wxDefaultSize, 0 );
	DateOfRunStaticText->Wrap( -1 );
	InfoSizer->Add( DateOfRunStaticText, 0, wxALL, 5 );
	
	m_staticText93 = new wxStaticText( JobDetailsPanel, wxID_ANY, wxT("Time Of Run :"), wxDefaultPosition, wxDefaultSize, 0 );
	m_staticText93->Wrap( -1 );
	m_staticText93->SetFont( wxFont( 10, 74, 90, 92, false, wxT("Sans") ) );
	
	InfoSizer->Add( m_staticText93, 0, wxALIGN_RIGHT|wxALL, 5 );
	
	TimeOfRunStaticText = new wxStaticText( JobDetailsPanel, wxID_ANY, wxEmptyString, wxDefaultPosition, wxDefaultSize, 0 );
	TimeOfRunStaticText->Wrap( -1 );
	InfoSizer->Add( TimeOfRunStaticText, 0, wxALIGN_LEFT|wxALL, 5 );
	
	m_staticText83 = new wxStaticText( JobDetailsPanel, wxID_ANY, wxT("Voltage :"), wxDefaultPosition, wxDefaultSize, 0 );
	m_staticText83->Wrap( -1 );
	m_staticText83->SetFont( wxFont( 10, 74, 90, 92, false, wxT("Sans") ) );
	
	InfoSizer->Add( m_staticText83, 0, wxALIGN_RIGHT|wxALL, 5 );
	
	VoltageStaticText = new wxStaticText( JobDetailsPanel, wxID_ANY, wxEmptyString, wxDefaultPosition, wxDefaultSize, 0 );
	VoltageStaticText->Wrap( -1 );
	InfoSizer->Add( VoltageStaticText, 0, wxALIGN_LEFT|wxALL, 5 );
	
	m_staticText82 = new wxStaticText( JobDetailsPanel, wxID_ANY, wxT("Cs :"), wxDefaultPosition, wxDefaultSize, 0 );
	m_staticText82->Wrap( -1 );
	m_staticText82->SetFont( wxFont( 10, 74, 90, 92, false, wxT("Sans") ) );
	
	InfoSizer->Add( m_staticText82, 0, wxALIGN_RIGHT|wxALL, 5 );
	
	CsStaticText = new wxStaticText( JobDetailsPanel, wxID_ANY, wxEmptyString, wxDefaultPosition, wxDefaultSize, 0 );
	CsStaticText->Wrap( -1 );
	InfoSizer->Add( CsStaticText, 0, wxALIGN_LEFT|wxALL, 5 );
	
	m_staticText78 = new wxStaticText( JobDetailsPanel, wxID_ANY, wxT("Pixel Size :"), wxDefaultPosition, wxDefaultSize, 0 );
	m_staticText78->Wrap( -1 );
	m_staticText78->SetFont( wxFont( 10, 74, 90, 92, false, wxT("Sans") ) );
	
	InfoSizer->Add( m_staticText78, 0, wxALIGN_RIGHT|wxALL, 5 );
	
	PixelSizeStaticText = new wxStaticText( JobDetailsPanel, wxID_ANY, wxEmptyString, wxDefaultPosition, wxDefaultSize, 0 );
	PixelSizeStaticText->Wrap( -1 );
	InfoSizer->Add( PixelSizeStaticText, 0, wxALIGN_LEFT|wxALL, 5 );
	
	m_staticText96 = new wxStaticText( JobDetailsPanel, wxID_ANY, wxT("Amp. Contrast :"), wxDefaultPosition, wxDefaultSize, 0 );
	m_staticText96->Wrap( -1 );
	m_staticText96->SetFont( wxFont( 10, 74, 90, 92, false, wxT("Sans") ) );
	
	InfoSizer->Add( m_staticText96, 0, wxALIGN_RIGHT|wxALL, 5 );
	
	AmplitudeContrastStaticText = new wxStaticText( JobDetailsPanel, wxID_ANY, wxEmptyString, wxDefaultPosition, wxDefaultSize, 0 );
	AmplitudeContrastStaticText->Wrap( -1 );
	InfoSizer->Add( AmplitudeContrastStaticText, 0, wxALIGN_LEFT|wxALL, 5 );
	
	m_staticText85 = new wxStaticText( JobDetailsPanel, wxID_ANY, wxT("Box Size :"), wxDefaultPosition, wxDefaultSize, 0 );
	m_staticText85->Wrap( -1 );
	m_staticText85->SetFont( wxFont( 10, 74, 90, 92, false, wxT("Sans") ) );
	
	InfoSizer->Add( m_staticText85, 0, wxALIGN_RIGHT|wxALL, 5 );
	
	BoxSizeStaticText = new wxStaticText( JobDetailsPanel, wxID_ANY, wxEmptyString, wxDefaultPosition, wxDefaultSize, 0 );
	BoxSizeStaticText->Wrap( -1 );
	InfoSizer->Add( BoxSizeStaticText, 0, wxALIGN_LEFT|wxALL, 5 );
	
	m_staticText87 = new wxStaticText( JobDetailsPanel, wxID_ANY, wxT("Min. Res. :"), wxDefaultPosition, wxDefaultSize, 0 );
	m_staticText87->Wrap( -1 );
	m_staticText87->SetFont( wxFont( 10, 74, 90, 92, false, wxT("Sans") ) );
	
	InfoSizer->Add( m_staticText87, 0, wxALIGN_RIGHT|wxALL, 5 );
	
	MinResStaticText = new wxStaticText( JobDetailsPanel, wxID_ANY, wxEmptyString, wxDefaultPosition, wxDefaultSize, 0 );
	MinResStaticText->Wrap( -1 );
	InfoSizer->Add( MinResStaticText, 0, wxALIGN_LEFT|wxALL, 5 );
	
	m_staticText89 = new wxStaticText( JobDetailsPanel, wxID_ANY, wxT("Max. Res. :"), wxDefaultPosition, wxDefaultSize, 0 );
	m_staticText89->Wrap( -1 );
	m_staticText89->SetFont( wxFont( 10, 74, 90, 92, false, wxT("Sans") ) );
	
	InfoSizer->Add( m_staticText89, 0, wxALIGN_RIGHT|wxALL, 5 );
	
	MaxResStaticText = new wxStaticText( JobDetailsPanel, wxID_ANY, wxEmptyString, wxDefaultPosition, wxDefaultSize, 0 );
	MaxResStaticText->Wrap( -1 );
	InfoSizer->Add( MaxResStaticText, 0, wxALIGN_LEFT|wxALL, 5 );
	
	m_staticText91 = new wxStaticText( JobDetailsPanel, wxID_ANY, wxT("Min. Defocus :"), wxDefaultPosition, wxDefaultSize, 0 );
	m_staticText91->Wrap( -1 );
	m_staticText91->SetFont( wxFont( 10, 74, 90, 92, false, wxT("Sans") ) );
	
	InfoSizer->Add( m_staticText91, 0, wxALIGN_RIGHT|wxALL, 5 );
	
	MinDefocusStaticText = new wxStaticText( JobDetailsPanel, wxID_ANY, wxEmptyString, wxDefaultPosition, wxDefaultSize, 0 );
	MinDefocusStaticText->Wrap( -1 );
	InfoSizer->Add( MinDefocusStaticText, 0, wxALIGN_LEFT|wxALL, 5 );
	
	m_staticText79 = new wxStaticText( JobDetailsPanel, wxID_ANY, wxT("Max. Defocus :"), wxDefaultPosition, wxDefaultSize, 0 );
	m_staticText79->Wrap( -1 );
	m_staticText79->SetFont( wxFont( 10, 74, 90, 92, false, wxT("Sans") ) );
	
	InfoSizer->Add( m_staticText79, 0, wxALIGN_RIGHT|wxALL, 5 );
	
	MaxDefocusStaticText = new wxStaticText( JobDetailsPanel, wxID_ANY, wxEmptyString, wxDefaultPosition, wxDefaultSize, 0 );
	MaxDefocusStaticText->Wrap( -1 );
	InfoSizer->Add( MaxDefocusStaticText, 0, wxALIGN_LEFT|wxALL, 5 );
	
	m_staticText95 = new wxStaticText( JobDetailsPanel, wxID_ANY, wxT("Defocus Step :"), wxDefaultPosition, wxDefaultSize, 0 );
	m_staticText95->Wrap( -1 );
	m_staticText95->SetFont( wxFont( 10, 74, 90, 92, false, wxT("Sans") ) );
	
	InfoSizer->Add( m_staticText95, 0, wxALIGN_RIGHT|wxALL, 5 );
	
	DefocusStepStaticText = new wxStaticText( JobDetailsPanel, wxID_ANY, wxEmptyString, wxDefaultPosition, wxDefaultSize, 0 );
	DefocusStepStaticText->Wrap( -1 );
	InfoSizer->Add( DefocusStepStaticText, 0, wxALIGN_LEFT|wxALL, 5 );
	
	LargeAstigExpectedLabel = new wxStaticText( JobDetailsPanel, wxID_ANY, wxT("Exhaustive Search? :"), wxDefaultPosition, wxDefaultSize, 0 );
	LargeAstigExpectedLabel->Wrap( -1 );
	LargeAstigExpectedLabel->SetFont( wxFont( 10, 74, 90, 92, false, wxT("Sans") ) );
	
	InfoSizer->Add( LargeAstigExpectedLabel, 0, wxALIGN_RIGHT|wxALL, 5 );
	
	LargeAstigExpectedStaticText = new wxStaticText( JobDetailsPanel, wxID_ANY, wxEmptyString, wxDefaultPosition, wxDefaultSize, 0 );
	LargeAstigExpectedStaticText->Wrap( -1 );
	InfoSizer->Add( LargeAstigExpectedStaticText, 0, wxALL, 5 );
	
	m_staticText99 = new wxStaticText( JobDetailsPanel, wxID_ANY, wxT("Restrain Astig.?:"), wxDefaultPosition, wxDefaultSize, 0 );
	m_staticText99->Wrap( -1 );
	m_staticText99->SetFont( wxFont( 10, 74, 90, 92, false, wxT("Sans") ) );
	
	InfoSizer->Add( m_staticText99, 0, wxALIGN_RIGHT|wxALL, 5 );
	
	RestrainAstigStaticText = new wxStaticText( JobDetailsPanel, wxID_ANY, wxEmptyString, wxDefaultPosition, wxDefaultSize, 0 );
	RestrainAstigStaticText->Wrap( -1 );
	InfoSizer->Add( RestrainAstigStaticText, 0, wxALIGN_LEFT|wxALL, 5 );
	
	ToleratedAstigLabel = new wxStaticText( JobDetailsPanel, wxID_ANY, wxT("Tolerated Astig. :"), wxDefaultPosition, wxDefaultSize, 0 );
	ToleratedAstigLabel->Wrap( -1 );
	ToleratedAstigLabel->SetFont( wxFont( 10, 74, 90, 92, false, wxT("Sans") ) );
	
	InfoSizer->Add( ToleratedAstigLabel, 0, wxALIGN_RIGHT|wxALL, 5 );
	
	ToleratedAstigStaticText = new wxStaticText( JobDetailsPanel, wxID_ANY, wxEmptyString, wxDefaultPosition, wxDefaultSize, 0 );
	ToleratedAstigStaticText->Wrap( -1 );
	InfoSizer->Add( ToleratedAstigStaticText, 0, wxALIGN_LEFT|wxALL, 5 );
	
	NumberOfAveragedFramesLabel = new wxStaticText( JobDetailsPanel, wxID_ANY, wxT("Num. Averaged Frames :"), wxDefaultPosition, wxDefaultSize, 0 );
	NumberOfAveragedFramesLabel->Wrap( -1 );
	NumberOfAveragedFramesLabel->SetFont( wxFont( 10, 74, 90, 92, false, wxT("Sans") ) );
	
	InfoSizer->Add( NumberOfAveragedFramesLabel, 0, wxALL, 5 );
	
	NumberOfAveragedFramesStaticText = new wxStaticText( JobDetailsPanel, wxID_ANY, wxEmptyString, wxDefaultPosition, wxDefaultSize, 0 );
	NumberOfAveragedFramesStaticText->Wrap( -1 );
	InfoSizer->Add( NumberOfAveragedFramesStaticText, 0, wxALL, 5 );
	
	m_staticText103 = new wxStaticText( JobDetailsPanel, wxID_ANY, wxT("Add. Phase Shift :"), wxDefaultPosition, wxDefaultSize, 0 );
	m_staticText103->Wrap( -1 );
	m_staticText103->SetFont( wxFont( 10, 74, 90, 92, false, wxT("Sans") ) );
	
	InfoSizer->Add( m_staticText103, 0, wxALIGN_RIGHT|wxALL, 5 );
	
	AddtionalPhaseShiftStaticText = new wxStaticText( JobDetailsPanel, wxID_ANY, wxEmptyString, wxDefaultPosition, wxDefaultSize, 0 );
	AddtionalPhaseShiftStaticText->Wrap( -1 );
	InfoSizer->Add( AddtionalPhaseShiftStaticText, 0, wxALIGN_LEFT|wxALL, 5 );
	
	MinPhaseShiftLabel = new wxStaticText( JobDetailsPanel, wxID_ANY, wxT("Min. Phase Shift :"), wxDefaultPosition, wxDefaultSize, 0 );
	MinPhaseShiftLabel->Wrap( -1 );
	MinPhaseShiftLabel->SetFont( wxFont( 10, 74, 90, 92, false, wxT("Sans") ) );
	
	InfoSizer->Add( MinPhaseShiftLabel, 0, wxALIGN_RIGHT|wxALL, 5 );
	
	MinPhaseShiftStaticText = new wxStaticText( JobDetailsPanel, wxID_ANY, wxEmptyString, wxDefaultPosition, wxDefaultSize, 0 );
	MinPhaseShiftStaticText->Wrap( -1 );
	InfoSizer->Add( MinPhaseShiftStaticText, 0, wxALIGN_LEFT|wxALL, 5 );
	
	MaxPhaseShiftLabel = new wxStaticText( JobDetailsPanel, wxID_ANY, wxT("Max. Phase Shift :"), wxDefaultPosition, wxDefaultSize, 0 );
	MaxPhaseShiftLabel->Wrap( -1 );
	MaxPhaseShiftLabel->SetFont( wxFont( 10, 74, 90, 92, false, wxT("Sans") ) );
	
	InfoSizer->Add( MaxPhaseShiftLabel, 0, wxALIGN_RIGHT|wxALL, 5 );
	
	MaxPhaseshiftStaticText = new wxStaticText( JobDetailsPanel, wxID_ANY, wxEmptyString, wxDefaultPosition, wxDefaultSize, 0 );
	MaxPhaseshiftStaticText->Wrap( -1 );
	InfoSizer->Add( MaxPhaseshiftStaticText, 0, wxALL, 5 );
	
	PhaseShiftStepLabel = new wxStaticText( JobDetailsPanel, wxID_ANY, wxT("Phase Shift Step :"), wxDefaultPosition, wxDefaultSize, 0 );
	PhaseShiftStepLabel->Wrap( -1 );
	PhaseShiftStepLabel->SetFont( wxFont( 10, 74, 90, 92, false, wxT("Sans") ) );
	
	InfoSizer->Add( PhaseShiftStepLabel, 0, wxALIGN_RIGHT|wxALL, 5 );
	
	PhaseShiftStepStaticText = new wxStaticText( JobDetailsPanel, wxID_ANY, wxEmptyString, wxDefaultPosition, wxDefaultSize, 0 );
	PhaseShiftStepStaticText->Wrap( -1 );
	InfoSizer->Add( PhaseShiftStepStaticText, 0, wxALL, 5 );
	
	IcinessStaticText = new wxStaticText( JobDetailsPanel, wxID_ANY, wxEmptyString, wxDefaultPosition, wxDefaultSize, 0 );
	IcinessStaticText->Wrap( -1 );
	InfoSizer->Add( IcinessStaticText, 0, wxALL, 5 );
	
	
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

PickingResultsPanel::PickingResultsPanel( wxWindow* parent, wxWindowID id, const wxPoint& pos, const wxSize& size, long style ) : wxPanel( parent, id, pos, size, style )
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
	m_staticText72->SetFont( wxFont( 10, 74, 90, 92, false, wxT("Sans") ) );
	
	InfoSizer->Add( m_staticText72, 0, wxALIGN_RIGHT|wxALL, 5 );
	
	PickIDStaticText = new wxStaticText( JobDetailsPanel, wxID_ANY, wxEmptyString, wxDefaultPosition, wxDefaultSize, 0 );
	PickIDStaticText->Wrap( -1 );
	InfoSizer->Add( PickIDStaticText, 0, wxALIGN_LEFT|wxALL, 5 );
	
	m_staticText74 = new wxStaticText( JobDetailsPanel, wxID_ANY, wxT("Date of Run :"), wxDefaultPosition, wxDefaultSize, 0 );
	m_staticText74->Wrap( -1 );
	m_staticText74->SetFont( wxFont( 10, 74, 90, 92, false, wxT("Sans") ) );
	
	InfoSizer->Add( m_staticText74, 0, wxALIGN_RIGHT|wxALL, 5 );
	
	DateOfRunStaticText = new wxStaticText( JobDetailsPanel, wxID_ANY, wxEmptyString, wxDefaultPosition, wxDefaultSize, 0 );
	DateOfRunStaticText->Wrap( -1 );
	InfoSizer->Add( DateOfRunStaticText, 0, wxALL, 5 );
	
	m_staticText93 = new wxStaticText( JobDetailsPanel, wxID_ANY, wxT("Time Of Run :"), wxDefaultPosition, wxDefaultSize, 0 );
	m_staticText93->Wrap( -1 );
	m_staticText93->SetFont( wxFont( 10, 74, 90, 92, false, wxT("Sans") ) );
	
	InfoSizer->Add( m_staticText93, 0, wxALIGN_RIGHT|wxALL, 5 );
	
	TimeOfRunStaticText = new wxStaticText( JobDetailsPanel, wxID_ANY, wxEmptyString, wxDefaultPosition, wxDefaultSize, 0 );
	TimeOfRunStaticText->Wrap( -1 );
	InfoSizer->Add( TimeOfRunStaticText, 0, wxALIGN_LEFT|wxALL, 5 );
	
	m_staticText83 = new wxStaticText( JobDetailsPanel, wxID_ANY, wxT("Algorithm :"), wxDefaultPosition, wxDefaultSize, 0 );
	m_staticText83->Wrap( -1 );
	m_staticText83->SetFont( wxFont( 10, 74, 90, 92, false, wxT("Sans") ) );
	
	InfoSizer->Add( m_staticText83, 0, wxALIGN_RIGHT|wxALL, 5 );
	
	AlgorithmStaticText = new wxStaticText( JobDetailsPanel, wxID_ANY, wxEmptyString, wxDefaultPosition, wxDefaultSize, 0 );
	AlgorithmStaticText->Wrap( -1 );
	InfoSizer->Add( AlgorithmStaticText, 0, wxALIGN_LEFT|wxALL, 5 );
	
	m_staticText82 = new wxStaticText( JobDetailsPanel, wxID_ANY, wxT("Manual edit :"), wxDefaultPosition, wxDefaultSize, 0 );
	m_staticText82->Wrap( -1 );
	m_staticText82->SetFont( wxFont( 10, 74, 90, 92, false, wxT("Sans") ) );
	
	InfoSizer->Add( m_staticText82, 0, wxALIGN_RIGHT|wxALL, 5 );
	
	ManualEditStaticText = new wxStaticText( JobDetailsPanel, wxID_ANY, wxEmptyString, wxDefaultPosition, wxDefaultSize, 0 );
	ManualEditStaticText->Wrap( -1 );
	InfoSizer->Add( ManualEditStaticText, 0, wxALIGN_LEFT|wxALL, 5 );
	
	m_staticText78 = new wxStaticText( JobDetailsPanel, wxID_ANY, wxT("Threshold :"), wxDefaultPosition, wxDefaultSize, 0 );
	m_staticText78->Wrap( -1 );
	m_staticText78->SetFont( wxFont( 10, 74, 90, 92, false, wxT("Sans") ) );
	
	InfoSizer->Add( m_staticText78, 0, wxALIGN_RIGHT|wxALL, 5 );
	
	ThresholdStaticText = new wxStaticText( JobDetailsPanel, wxID_ANY, wxEmptyString, wxDefaultPosition, wxDefaultSize, 0 );
	ThresholdStaticText->Wrap( -1 );
	InfoSizer->Add( ThresholdStaticText, 0, wxALIGN_LEFT|wxALL, 5 );
	
	m_staticText96 = new wxStaticText( JobDetailsPanel, wxID_ANY, wxT("Max. Radius :"), wxDefaultPosition, wxDefaultSize, 0 );
	m_staticText96->Wrap( -1 );
	m_staticText96->SetFont( wxFont( 10, 74, 90, 92, false, wxT("Sans") ) );
	
	InfoSizer->Add( m_staticText96, 0, wxALIGN_RIGHT|wxALL, 5 );
	
	MaximumRadiusStaticText = new wxStaticText( JobDetailsPanel, wxID_ANY, wxEmptyString, wxDefaultPosition, wxDefaultSize, 0 );
	MaximumRadiusStaticText->Wrap( -1 );
	InfoSizer->Add( MaximumRadiusStaticText, 0, wxALIGN_LEFT|wxALL, 5 );
	
	m_staticText85 = new wxStaticText( JobDetailsPanel, wxID_ANY, wxT("Charact. Radius :"), wxDefaultPosition, wxDefaultSize, 0 );
	m_staticText85->Wrap( -1 );
	m_staticText85->SetFont( wxFont( 10, 74, 90, 92, false, wxT("Sans") ) );
	
	InfoSizer->Add( m_staticText85, 0, wxALIGN_RIGHT|wxALL, 5 );
	
	CharacteristicRadiusStaticText = new wxStaticText( JobDetailsPanel, wxID_ANY, wxEmptyString, wxDefaultPosition, wxDefaultSize, 0 );
	CharacteristicRadiusStaticText->Wrap( -1 );
	InfoSizer->Add( CharacteristicRadiusStaticText, 0, wxALIGN_LEFT|wxALL, 5 );
	
	m_staticText87 = new wxStaticText( JobDetailsPanel, wxID_ANY, wxT("Highest Res. :"), wxDefaultPosition, wxDefaultSize, 0 );
	m_staticText87->Wrap( -1 );
	m_staticText87->SetFont( wxFont( 10, 74, 90, 92, false, wxT("Sans") ) );
	
	InfoSizer->Add( m_staticText87, 0, wxALIGN_RIGHT|wxALL, 5 );
	
	HighestResStaticText = new wxStaticText( JobDetailsPanel, wxID_ANY, wxEmptyString, wxDefaultPosition, wxDefaultSize, 0 );
	HighestResStaticText->Wrap( -1 );
	InfoSizer->Add( HighestResStaticText, 0, wxALIGN_LEFT|wxALL, 5 );
	
	m_staticText89 = new wxStaticText( JobDetailsPanel, wxID_ANY, wxT("Min. Edge Dist. :"), wxDefaultPosition, wxDefaultSize, 0 );
	m_staticText89->Wrap( -1 );
	m_staticText89->SetFont( wxFont( 10, 74, 90, 92, false, wxT("Sans") ) );
	
	InfoSizer->Add( m_staticText89, 0, wxALIGN_RIGHT|wxALL, 5 );
	
	MinEdgeDistStaticText = new wxStaticText( JobDetailsPanel, wxID_ANY, wxEmptyString, wxDefaultPosition, wxDefaultSize, 0 );
	MinEdgeDistStaticText->Wrap( -1 );
	InfoSizer->Add( MinEdgeDistStaticText, 0, wxALIGN_LEFT|wxALL, 5 );
	
	m_staticText91 = new wxStaticText( JobDetailsPanel, wxID_ANY, wxT("Avoid High Var. :"), wxDefaultPosition, wxDefaultSize, 0 );
	m_staticText91->Wrap( -1 );
	m_staticText91->SetFont( wxFont( 10, 74, 90, 92, false, wxT("Sans") ) );
	
	InfoSizer->Add( m_staticText91, 0, wxALIGN_RIGHT|wxALL, 5 );
	
	AvoidHighVarStaticText = new wxStaticText( JobDetailsPanel, wxID_ANY, wxEmptyString, wxDefaultPosition, wxDefaultSize, 0 );
	AvoidHighVarStaticText->Wrap( -1 );
	InfoSizer->Add( AvoidHighVarStaticText, 0, wxALIGN_LEFT|wxALL, 5 );
	
	m_staticText79 = new wxStaticText( JobDetailsPanel, wxID_ANY, wxT("Avoid Hi/Lo Mean :"), wxDefaultPosition, wxDefaultSize, 0 );
	m_staticText79->Wrap( -1 );
	m_staticText79->SetFont( wxFont( 10, 74, 90, 92, false, wxT("Sans") ) );
	
	InfoSizer->Add( m_staticText79, 0, wxALIGN_RIGHT|wxALL, 5 );
	
	AvoidHighLowMeanStaticText = new wxStaticText( JobDetailsPanel, wxID_ANY, wxEmptyString, wxDefaultPosition, wxDefaultSize, 0 );
	AvoidHighLowMeanStaticText->Wrap( -1 );
	InfoSizer->Add( AvoidHighLowMeanStaticText, 0, wxALIGN_LEFT|wxALL, 5 );
	
	m_staticText95 = new wxStaticText( JobDetailsPanel, wxID_ANY, wxT("Num. Bckgd. Boxes :"), wxDefaultPosition, wxDefaultSize, 0 );
	m_staticText95->Wrap( -1 );
	m_staticText95->SetFont( wxFont( 10, 74, 90, 92, false, wxT("Sans") ) );
	
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

MovieAlignResultsPanel::MovieAlignResultsPanel( wxWindow* parent, wxWindowID id, const wxPoint& pos, const wxSize& size, long style ) : wxPanel( parent, id, pos, size, style )
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
	m_staticText72->SetFont( wxFont( 10, 74, 90, 92, false, wxT("Sans") ) );
	
	InfoSizer->Add( m_staticText72, 0, wxALIGN_RIGHT|wxALL, 5 );
	
	AlignmentIDStaticText = new wxStaticText( JobDetailsPanel, wxID_ANY, wxEmptyString, wxDefaultPosition, wxDefaultSize, 0 );
	AlignmentIDStaticText->Wrap( -1 );
	InfoSizer->Add( AlignmentIDStaticText, 0, wxALIGN_LEFT|wxALL, 5 );
	
	m_staticText74 = new wxStaticText( JobDetailsPanel, wxID_ANY, wxT("Date of Run :"), wxDefaultPosition, wxDefaultSize, 0 );
	m_staticText74->Wrap( -1 );
	m_staticText74->SetFont( wxFont( 10, 74, 90, 92, false, wxT("Sans") ) );
	
	InfoSizer->Add( m_staticText74, 0, wxALIGN_RIGHT|wxALL, 5 );
	
	DateOfRunStaticText = new wxStaticText( JobDetailsPanel, wxID_ANY, wxEmptyString, wxDefaultPosition, wxDefaultSize, 0 );
	DateOfRunStaticText->Wrap( -1 );
	InfoSizer->Add( DateOfRunStaticText, 0, wxALL, 5 );
	
	m_staticText93 = new wxStaticText( JobDetailsPanel, wxID_ANY, wxT("Time Of Run :"), wxDefaultPosition, wxDefaultSize, 0 );
	m_staticText93->Wrap( -1 );
	m_staticText93->SetFont( wxFont( 10, 74, 90, 92, false, wxT("Sans") ) );
	
	InfoSizer->Add( m_staticText93, 0, wxALIGN_RIGHT|wxALL, 5 );
	
	TimeOfRunStaticText = new wxStaticText( JobDetailsPanel, wxID_ANY, wxEmptyString, wxDefaultPosition, wxDefaultSize, 0 );
	TimeOfRunStaticText->Wrap( -1 );
	InfoSizer->Add( TimeOfRunStaticText, 0, wxALIGN_LEFT|wxALL, 5 );
	
	m_staticText83 = new wxStaticText( JobDetailsPanel, wxID_ANY, wxT("Voltage :"), wxDefaultPosition, wxDefaultSize, 0 );
	m_staticText83->Wrap( -1 );
	m_staticText83->SetFont( wxFont( 10, 74, 90, 92, false, wxT("Sans") ) );
	
	InfoSizer->Add( m_staticText83, 0, wxALIGN_RIGHT|wxALL, 5 );
	
	VoltageStaticText = new wxStaticText( JobDetailsPanel, wxID_ANY, wxEmptyString, wxDefaultPosition, wxDefaultSize, 0 );
	VoltageStaticText->Wrap( -1 );
	InfoSizer->Add( VoltageStaticText, 0, wxALIGN_LEFT|wxALL, 5 );
	
	m_staticText78 = new wxStaticText( JobDetailsPanel, wxID_ANY, wxT("Pixel Size :"), wxDefaultPosition, wxDefaultSize, 0 );
	m_staticText78->Wrap( -1 );
	m_staticText78->SetFont( wxFont( 10, 74, 90, 92, false, wxT("Sans") ) );
	
	InfoSizer->Add( m_staticText78, 0, wxALIGN_RIGHT|wxALL, 5 );
	
	PixelSizeStaticText = new wxStaticText( JobDetailsPanel, wxID_ANY, wxEmptyString, wxDefaultPosition, wxDefaultSize, 0 );
	PixelSizeStaticText->Wrap( -1 );
	InfoSizer->Add( PixelSizeStaticText, 0, wxALIGN_LEFT|wxALL, 5 );
	
	m_staticText82 = new wxStaticText( JobDetailsPanel, wxID_ANY, wxT("Exp. per Frame :"), wxDefaultPosition, wxDefaultSize, 0 );
	m_staticText82->Wrap( -1 );
	m_staticText82->SetFont( wxFont( 10, 74, 90, 92, false, wxT("Sans") ) );
	
	InfoSizer->Add( m_staticText82, 0, wxALIGN_RIGHT|wxALL, 5 );
	
	ExposureStaticText = new wxStaticText( JobDetailsPanel, wxID_ANY, wxEmptyString, wxDefaultPosition, wxDefaultSize, 0 );
	ExposureStaticText->Wrap( -1 );
	InfoSizer->Add( ExposureStaticText, 0, wxALIGN_LEFT|wxALL, 5 );
	
	m_staticText96 = new wxStaticText( JobDetailsPanel, wxID_ANY, wxT("Pre Exp. :"), wxDefaultPosition, wxDefaultSize, 0 );
	m_staticText96->Wrap( -1 );
	m_staticText96->SetFont( wxFont( 10, 74, 90, 92, false, wxT("Sans") ) );
	
	InfoSizer->Add( m_staticText96, 0, wxALIGN_RIGHT|wxALL, 5 );
	
	PreExposureStaticText = new wxStaticText( JobDetailsPanel, wxID_ANY, wxEmptyString, wxDefaultPosition, wxDefaultSize, 0 );
	PreExposureStaticText->Wrap( -1 );
	InfoSizer->Add( PreExposureStaticText, 0, wxALIGN_LEFT|wxALL, 5 );
	
	m_staticText85 = new wxStaticText( JobDetailsPanel, wxID_ANY, wxT("Min. Shift :"), wxDefaultPosition, wxDefaultSize, 0 );
	m_staticText85->Wrap( -1 );
	m_staticText85->SetFont( wxFont( 10, 74, 90, 92, false, wxT("Sans") ) );
	
	InfoSizer->Add( m_staticText85, 0, wxALIGN_RIGHT|wxALL, 5 );
	
	MinShiftStaticText = new wxStaticText( JobDetailsPanel, wxID_ANY, wxEmptyString, wxDefaultPosition, wxDefaultSize, 0 );
	MinShiftStaticText->Wrap( -1 );
	InfoSizer->Add( MinShiftStaticText, 0, wxALIGN_LEFT|wxALL, 5 );
	
	m_staticText87 = new wxStaticText( JobDetailsPanel, wxID_ANY, wxT("Max. Shift :"), wxDefaultPosition, wxDefaultSize, 0 );
	m_staticText87->Wrap( -1 );
	m_staticText87->SetFont( wxFont( 10, 74, 90, 92, false, wxT("Sans") ) );
	
	InfoSizer->Add( m_staticText87, 0, wxALIGN_RIGHT|wxALL, 5 );
	
	MaxShiftStaticText = new wxStaticText( JobDetailsPanel, wxID_ANY, wxEmptyString, wxDefaultPosition, wxDefaultSize, 0 );
	MaxShiftStaticText->Wrap( -1 );
	InfoSizer->Add( MaxShiftStaticText, 0, wxALIGN_LEFT|wxALL, 5 );
	
	m_staticText89 = new wxStaticText( JobDetailsPanel, wxID_ANY, wxT("Term. Threshold :"), wxDefaultPosition, wxDefaultSize, 0 );
	m_staticText89->Wrap( -1 );
	m_staticText89->SetFont( wxFont( 10, 74, 90, 92, false, wxT("Sans") ) );
	
	InfoSizer->Add( m_staticText89, 0, wxALIGN_RIGHT|wxALL, 5 );
	
	TerminationThresholdStaticText = new wxStaticText( JobDetailsPanel, wxID_ANY, wxEmptyString, wxDefaultPosition, wxDefaultSize, 0 );
	TerminationThresholdStaticText->Wrap( -1 );
	InfoSizer->Add( TerminationThresholdStaticText, 0, wxALIGN_LEFT|wxALL, 5 );
	
	m_staticText91 = new wxStaticText( JobDetailsPanel, wxID_ANY, wxT("Max. Iterations :"), wxDefaultPosition, wxDefaultSize, 0 );
	m_staticText91->Wrap( -1 );
	m_staticText91->SetFont( wxFont( 10, 74, 90, 92, false, wxT("Sans") ) );
	
	InfoSizer->Add( m_staticText91, 0, wxALIGN_RIGHT|wxALL, 5 );
	
	MaxIterationsStaticText = new wxStaticText( JobDetailsPanel, wxID_ANY, wxEmptyString, wxDefaultPosition, wxDefaultSize, 0 );
	MaxIterationsStaticText->Wrap( -1 );
	InfoSizer->Add( MaxIterationsStaticText, 0, wxALIGN_LEFT|wxALL, 5 );
	
	m_staticText79 = new wxStaticText( JobDetailsPanel, wxID_ANY, wxT("b-factor :"), wxDefaultPosition, wxDefaultSize, 0 );
	m_staticText79->Wrap( -1 );
	m_staticText79->SetFont( wxFont( 10, 74, 90, 92, false, wxT("Sans") ) );
	
	InfoSizer->Add( m_staticText79, 0, wxALIGN_RIGHT|wxALL, 5 );
	
	BfactorStaticText = new wxStaticText( JobDetailsPanel, wxID_ANY, wxEmptyString, wxDefaultPosition, wxDefaultSize, 0 );
	BfactorStaticText->Wrap( -1 );
	InfoSizer->Add( BfactorStaticText, 0, wxALIGN_LEFT|wxALL, 5 );
	
	m_staticText95 = new wxStaticText( JobDetailsPanel, wxID_ANY, wxT("Exp. Filter :"), wxDefaultPosition, wxDefaultSize, 0 );
	m_staticText95->Wrap( -1 );
	m_staticText95->SetFont( wxFont( 10, 74, 90, 92, false, wxT("Sans") ) );
	
	InfoSizer->Add( m_staticText95, 0, wxALIGN_RIGHT|wxALL, 5 );
	
	ExposureFilterStaticText = new wxStaticText( JobDetailsPanel, wxID_ANY, wxEmptyString, wxDefaultPosition, wxDefaultSize, 0 );
	ExposureFilterStaticText->Wrap( -1 );
	InfoSizer->Add( ExposureFilterStaticText, 0, wxALIGN_LEFT|wxALL, 5 );
	
	m_staticText99 = new wxStaticText( JobDetailsPanel, wxID_ANY, wxT("Restore Power :"), wxDefaultPosition, wxDefaultSize, 0 );
	m_staticText99->Wrap( -1 );
	m_staticText99->SetFont( wxFont( 10, 74, 90, 92, false, wxT("Sans") ) );
	
	InfoSizer->Add( m_staticText99, 0, wxALIGN_RIGHT|wxALL, 5 );
	
	RestorePowerStaticText = new wxStaticText( JobDetailsPanel, wxID_ANY, wxEmptyString, wxDefaultPosition, wxDefaultSize, 0 );
	RestorePowerStaticText->Wrap( -1 );
	InfoSizer->Add( RestorePowerStaticText, 0, wxALIGN_LEFT|wxALL, 5 );
	
	m_staticText101 = new wxStaticText( JobDetailsPanel, wxID_ANY, wxT("Mask Cross :"), wxDefaultPosition, wxDefaultSize, 0 );
	m_staticText101->Wrap( -1 );
	m_staticText101->SetFont( wxFont( 10, 74, 90, 92, false, wxT("Sans") ) );
	
	InfoSizer->Add( m_staticText101, 0, wxALIGN_RIGHT|wxALL, 5 );
	
	MaskCrossStaticText = new wxStaticText( JobDetailsPanel, wxID_ANY, wxEmptyString, wxDefaultPosition, wxDefaultSize, 0 );
	MaskCrossStaticText->Wrap( -1 );
	InfoSizer->Add( MaskCrossStaticText, 0, wxALIGN_LEFT|wxALL, 5 );
	
	m_staticText103 = new wxStaticText( JobDetailsPanel, wxID_ANY, wxT("Horiz. Mask :"), wxDefaultPosition, wxDefaultSize, 0 );
	m_staticText103->Wrap( -1 );
	m_staticText103->SetFont( wxFont( 10, 74, 90, 92, false, wxT("Sans") ) );
	
	InfoSizer->Add( m_staticText103, 0, wxALIGN_RIGHT|wxALL, 5 );
	
	HorizontalMaskStaticText = new wxStaticText( JobDetailsPanel, wxID_ANY, wxEmptyString, wxDefaultPosition, wxDefaultSize, 0 );
	HorizontalMaskStaticText->Wrap( -1 );
	InfoSizer->Add( HorizontalMaskStaticText, 0, wxALIGN_LEFT|wxALL, 5 );
	
	m_staticText105 = new wxStaticText( JobDetailsPanel, wxID_ANY, wxT("Vert. Mask :"), wxDefaultPosition, wxDefaultSize, 0 );
	m_staticText105->Wrap( -1 );
	m_staticText105->SetFont( wxFont( 10, 74, 90, 92, false, wxT("Sans") ) );
	
	InfoSizer->Add( m_staticText105, 0, wxALIGN_RIGHT|wxALL, 5 );
	
	VerticalMaskStaticText = new wxStaticText( JobDetailsPanel, wxID_ANY, wxEmptyString, wxDefaultPosition, wxDefaultSize, 0 );
	VerticalMaskStaticText->Wrap( -1 );
	InfoSizer->Add( VerticalMaskStaticText, 0, wxALIGN_LEFT|wxALL, 5 );
	
	m_staticText1051 = new wxStaticText( JobDetailsPanel, wxID_ANY, wxT("Sum all Frames :"), wxDefaultPosition, wxDefaultSize, 0 );
	m_staticText1051->Wrap( -1 );
	m_staticText1051->SetFont( wxFont( 10, 74, 90, 92, false, wxT("Sans") ) );
	
	InfoSizer->Add( m_staticText1051, 0, wxALIGN_RIGHT|wxALL, 5 );
	
	IncludeAllFramesStaticText = new wxStaticText( JobDetailsPanel, wxID_ANY, wxEmptyString, wxDefaultPosition, wxDefaultSize, 0 );
	IncludeAllFramesStaticText->Wrap( -1 );
	InfoSizer->Add( IncludeAllFramesStaticText, 0, wxALIGN_LEFT|wxALL, 5 );
	
	m_staticText1052 = new wxStaticText( JobDetailsPanel, wxID_ANY, wxT("First Frame :"), wxDefaultPosition, wxDefaultSize, 0 );
	m_staticText1052->Wrap( -1 );
	m_staticText1052->SetFont( wxFont( 10, 74, 90, 92, false, wxT("Sans") ) );
	
	InfoSizer->Add( m_staticText1052, 0, wxALIGN_RIGHT|wxALL, 5 );
	
	FirstFrameStaticText = new wxStaticText( JobDetailsPanel, wxID_ANY, wxEmptyString, wxDefaultPosition, wxDefaultSize, 0 );
	FirstFrameStaticText->Wrap( -1 );
	InfoSizer->Add( FirstFrameStaticText, 0, wxALIGN_LEFT|wxALL, 5 );
	
	m_staticText1053 = new wxStaticText( JobDetailsPanel, wxID_ANY, wxT("Last Frame :"), wxDefaultPosition, wxDefaultSize, 0 );
	m_staticText1053->Wrap( -1 );
	m_staticText1053->SetFont( wxFont( 10, 74, 90, 92, false, wxT("Sans") ) );
	
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

ResultsPanel::ResultsPanel( wxWindow* parent, wxWindowID id, const wxPoint& pos, const wxSize& size, long style ) : wxPanel( parent, id, pos, size, style )
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

AssetsPanel::AssetsPanel( wxWindow* parent, wxWindowID id, const wxPoint& pos, const wxSize& size, long style ) : wxPanel( parent, id, pos, size, style )
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
}

AssetsPanel::~AssetsPanel()
{
}

ExperimentalPanel::ExperimentalPanel( wxWindow* parent, wxWindowID id, const wxPoint& pos, const wxSize& size, long style ) : wxPanel( parent, id, pos, size, style )
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
}

ExperimentalPanel::~ExperimentalPanel()
{
}

OverviewPanel::OverviewPanel( wxWindow* parent, wxWindowID id, const wxPoint& pos, const wxSize& size, long style ) : wxPanel( parent, id, pos, size, style )
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

FindParticlesPanel::FindParticlesPanel( wxWindow* parent, wxWindowID id, const wxPoint& pos, const wxSize& size, long style ) : JobPanel( parent, id, pos, size, style )
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
	PleaseEstimateCTFStaticText->SetFont( wxFont( wxNORMAL_FONT->GetPointSize(), 70, 90, 92, false, wxEmptyString ) );
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
	ExpertOptionsStaticText->SetFont( wxFont( wxNORMAL_FONT->GetPointSize(), 70, 90, 92, true, wxEmptyString ) );
	
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
	
	TimeRemainingText = new wxStaticText( ProgressPanel, wxID_ANY, wxT("Time Remaining : ???h:??m:??s   "), wxDefaultPosition, wxDefaultSize, wxALIGN_CENTRE );
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

VolumeImportDialog::VolumeImportDialog( wxWindow* parent, wxWindowID id, const wxString& title, const wxPoint& pos, const wxSize& size, long style ) : wxDialog( parent, id, title, pos, size, style )
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
	
	wxBoxSizer* bSizer29;
	bSizer29 = new wxBoxSizer( wxHORIZONTAL );
	
	m_staticText20 = new wxStaticText( this, wxID_ANY, wxT("Pixel Size (Å) :"), wxDefaultPosition, wxDefaultSize, 0 );
	m_staticText20->Wrap( -1 );
	bSizer29->Add( m_staticText20, 50, wxALIGN_CENTER_VERTICAL|wxALL, 5 );
	
	PixelSizeText = new wxTextCtrl( this, wxID_ANY, wxEmptyString, wxDefaultPosition, wxDefaultSize, 0 );
	bSizer29->Add( PixelSizeText, 50, wxALIGN_CENTER_VERTICAL|wxALL, 5 );
	
	
	bSizer26->Add( bSizer29, 0, wxEXPAND, 5 );
	
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
	m_button10->Connect( wxEVT_COMMAND_BUTTON_CLICKED, wxCommandEventHandler( VolumeImportDialog::AddFilesClick ), NULL, this );
	m_button11->Connect( wxEVT_COMMAND_BUTTON_CLICKED, wxCommandEventHandler( VolumeImportDialog::AddDirectoryClick ), NULL, this );
	ClearButton->Connect( wxEVT_COMMAND_BUTTON_CLICKED, wxCommandEventHandler( VolumeImportDialog::ClearClick ), NULL, this );
	PixelSizeText->Connect( wxEVT_CHAR, wxKeyEventHandler( VolumeImportDialog::OnTextKeyPress ), NULL, this );
	PixelSizeText->Connect( wxEVT_COMMAND_TEXT_UPDATED, wxCommandEventHandler( VolumeImportDialog::TextChanged ), NULL, this );
	m_button13->Connect( wxEVT_COMMAND_BUTTON_CLICKED, wxCommandEventHandler( VolumeImportDialog::CancelClick ), NULL, this );
	ImportButton->Connect( wxEVT_COMMAND_BUTTON_CLICKED, wxCommandEventHandler( VolumeImportDialog::ImportClick ), NULL, this );
}

VolumeImportDialog::~VolumeImportDialog()
{
	// Disconnect Events
	m_button10->Disconnect( wxEVT_COMMAND_BUTTON_CLICKED, wxCommandEventHandler( VolumeImportDialog::AddFilesClick ), NULL, this );
	m_button11->Disconnect( wxEVT_COMMAND_BUTTON_CLICKED, wxCommandEventHandler( VolumeImportDialog::AddDirectoryClick ), NULL, this );
	ClearButton->Disconnect( wxEVT_COMMAND_BUTTON_CLICKED, wxCommandEventHandler( VolumeImportDialog::ClearClick ), NULL, this );
	PixelSizeText->Disconnect( wxEVT_CHAR, wxKeyEventHandler( VolumeImportDialog::OnTextKeyPress ), NULL, this );
	PixelSizeText->Disconnect( wxEVT_COMMAND_TEXT_UPDATED, wxCommandEventHandler( VolumeImportDialog::TextChanged ), NULL, this );
	m_button13->Disconnect( wxEVT_COMMAND_BUTTON_CLICKED, wxCommandEventHandler( VolumeImportDialog::CancelClick ), NULL, this );
	ImportButton->Disconnect( wxEVT_COMMAND_BUTTON_CLICKED, wxCommandEventHandler( VolumeImportDialog::ImportClick ), NULL, this );
	
}

MovieImportDialog::MovieImportDialog( wxWindow* parent, wxWindowID id, const wxString& title, const wxPoint& pos, const wxSize& size, long style ) : wxDialog( parent, id, title, pos, size, style )
{
	this->SetSizeHints( wxDefaultSize, wxDefaultSize );
	
	wxBoxSizer* bSizer26;
	bSizer26 = new wxBoxSizer( wxVERTICAL );
	
	PathListCtrl = new wxListCtrl( this, wxID_ANY, wxDefaultPosition, wxDefaultSize, wxLC_LIST|wxLC_NO_HEADER|wxLC_REPORT|wxVSCROLL );
	bSizer26->Add( PathListCtrl, 0, wxALL|wxEXPAND, 5 );
	
	wxBoxSizer* bSizer27;
	bSizer27 = new wxBoxSizer( wxHORIZONTAL );
	
	m_button10 = new wxButton( this, wxID_ANY, wxT("Add Files"), wxDefaultPosition, wxDefaultSize, 0 );
	bSizer27->Add( m_button10, 33, wxALL, 5 );
	
	m_button11 = new wxButton( this, wxID_ANY, wxT("Add Directory"), wxDefaultPosition, wxDefaultSize, 0 );
	bSizer27->Add( m_button11, 33, wxALL, 5 );
	
	s = new wxButton( this, wxID_ANY, wxT("Clear"), wxDefaultPosition, wxDefaultSize, 0 );
	bSizer27->Add( s, 33, wxALL, 5 );
	
	
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
	
	wxBoxSizer* bSizer30;
	bSizer30 = new wxBoxSizer( wxHORIZONTAL );
	
	m_staticText21 = new wxStaticText( this, wxID_ANY, wxT("Spherical Aberration (mm) :"), wxDefaultPosition, wxDefaultSize, 0 );
	m_staticText21->Wrap( -1 );
	bSizer30->Add( m_staticText21, 50, wxALIGN_CENTER_VERTICAL|wxALL, 5 );
	
	CsText = new wxTextCtrl( this, wxID_ANY, wxEmptyString, wxDefaultPosition, wxDefaultSize, 0 );
	bSizer30->Add( CsText, 50, wxALIGN_CENTER_VERTICAL|wxALL, 5 );
	
	
	bSizer26->Add( bSizer30, 0, wxEXPAND, 5 );
	
	wxBoxSizer* bSizer29;
	bSizer29 = new wxBoxSizer( wxHORIZONTAL );
	
	m_staticText20 = new wxStaticText( this, wxID_ANY, wxT("Pixel Size (Å) :"), wxDefaultPosition, wxDefaultSize, 0 );
	m_staticText20->Wrap( -1 );
	bSizer29->Add( m_staticText20, 50, wxALIGN_CENTER_VERTICAL|wxALL, 5 );
	
	PixelSizeText = new wxTextCtrl( this, wxID_ANY, wxEmptyString, wxDefaultPosition, wxDefaultSize, 0 );
	bSizer29->Add( PixelSizeText, 50, wxALIGN_CENTER_VERTICAL|wxALL, 5 );
	
	
	bSizer26->Add( bSizer29, 0, wxEXPAND, 5 );
	
	wxBoxSizer* bSizer32;
	bSizer32 = new wxBoxSizer( wxHORIZONTAL );
	
	m_staticText22 = new wxStaticText( this, wxID_ANY, wxT("Exposure per frame (e¯/Å²) :"), wxDefaultPosition, wxDefaultSize, 0 );
	m_staticText22->Wrap( -1 );
	bSizer32->Add( m_staticText22, 50, wxALIGN_CENTER_VERTICAL|wxALL, 5 );
	
	DoseText = new wxTextCtrl( this, wxID_ANY, wxEmptyString, wxDefaultPosition, wxDefaultSize, 0 );
	bSizer32->Add( DoseText, 50, wxALIGN_CENTER_VERTICAL|wxALL, 5 );
	
	
	bSizer26->Add( bSizer32, 0, wxEXPAND, 5 );
	
	wxBoxSizer* bSizer2292;
	bSizer2292 = new wxBoxSizer( wxHORIZONTAL );
	
	ApplyDarkImageCheckbox = new wxCheckBox( this, wxID_ANY, wxT("Apply dark image :"), wxDefaultPosition, wxDefaultSize, 0 );
	bSizer2292->Add( ApplyDarkImageCheckbox, 50, wxALIGN_CENTER_VERTICAL|wxALL, 5 );
	
	DarkFilePicker = new wxFilePickerCtrl( this, wxID_ANY, wxEmptyString, wxT("Select a file"), wxT("Any supported type (*.mrc;*.tif;*.dm;*.dm3;*.dm4)|*.mrc;*.tif;*.dm;*.dm3;*.dm4|MRC files (*.mrc) | *.mrc |TIFF files (*.tif) | *.tif | DM files (*.dm;*.dm3;*.dm4)|*.dm;*.dm3;*.dm4"), wxDefaultPosition, wxDefaultSize, wxFLP_DEFAULT_STYLE|wxFLP_FILE_MUST_EXIST );
	DarkFilePicker->Enable( false );
	
	bSizer2292->Add( DarkFilePicker, 50, wxALIGN_CENTER_VERTICAL|wxALL, 5 );
	
	
	bSizer26->Add( bSizer2292, 0, wxEXPAND, 5 );
	
	wxBoxSizer* bSizer229;
	bSizer229 = new wxBoxSizer( wxHORIZONTAL );
	
	ApplyGainImageCheckbox = new wxCheckBox( this, wxID_ANY, wxT("Apply gain image :"), wxDefaultPosition, wxDefaultSize, 0 );
	bSizer229->Add( ApplyGainImageCheckbox, 50, wxALL, 5 );
	
	GainFilePicker = new wxFilePickerCtrl( this, wxID_ANY, wxEmptyString, wxT("Select a file"), wxT("Any supported type (*.mrc;*.tif;*.dm;*.dm3;*.dm4)|*.mrc;*.tif;*.dm;*.dm3;*.dm4|MRC files (*.mrc) | *.mrc |TIFF files (*.tif) | *.tif | DM files (*.dm;*.dm3;*.dm4)|*.dm;*.dm3;*.dm4"), wxDefaultPosition, wxDefaultSize, wxFLP_DEFAULT_STYLE|wxFLP_FILE_MUST_EXIST );
	GainFilePicker->Enable( false );
	
	bSizer229->Add( GainFilePicker, 50, wxALIGN_CENTER_VERTICAL|wxALL, 5 );
	
	
	bSizer26->Add( bSizer229, 0, wxEXPAND, 5 );
	
	wxBoxSizer* bSizer3211;
	bSizer3211 = new wxBoxSizer( wxHORIZONTAL );
	
	ResampleMoviesCheckBox = new wxCheckBox( this, wxID_ANY, wxT("Resample movies during processing"), wxDefaultPosition, wxDefaultSize, 0 );
	bSizer3211->Add( ResampleMoviesCheckBox, 50, wxALIGN_CENTER_VERTICAL|wxALL, 5 );
	
	
	bSizer3211->Add( 0, 0, 50, wxEXPAND, 5 );
	
	
	bSizer26->Add( bSizer3211, 0, wxEXPAND, 5 );
	
	wxBoxSizer* bSizer2291;
	bSizer2291 = new wxBoxSizer( wxHORIZONTAL );
	
	DesiredPixelSizeStaticText = new wxStaticText( this, wxID_ANY, wxT("        Desired pixel size (Å) :"), wxDefaultPosition, wxDefaultSize, 0 );
	DesiredPixelSizeStaticText->Wrap( -1 );
	DesiredPixelSizeStaticText->Enable( false );
	
	bSizer2291->Add( DesiredPixelSizeStaticText, 50, wxALIGN_CENTER_VERTICAL|wxALL, 5 );
	
	DesiredPixelSizeTextCtrl = new NumericTextCtrl( this, wxID_ANY, wxT("1.00"), wxDefaultPosition, wxDefaultSize, 0 );
	DesiredPixelSizeTextCtrl->Enable( false );
	
	bSizer2291->Add( DesiredPixelSizeTextCtrl, 50, wxALL, 5 );
	
	
	bSizer26->Add( bSizer2291, 0, wxEXPAND, 5 );
	
	wxBoxSizer* bSizer282;
	bSizer282 = new wxBoxSizer( wxVERTICAL );
	
	CorrectMagDistortionCheckBox = new wxCheckBox( this, wxID_ANY, wxT("Correct magnification distortion"), wxDefaultPosition, wxDefaultSize, 0 );
	bSizer282->Add( CorrectMagDistortionCheckBox, 0, wxALL, 5 );
	
	wxBoxSizer* bSizer283;
	bSizer283 = new wxBoxSizer( wxHORIZONTAL );
	
	DistortionAngleStaticText = new wxStaticText( this, wxID_ANY, wxT("        Distortion Angle (°) :"), wxDefaultPosition, wxDefaultSize, 0 );
	DistortionAngleStaticText->Wrap( -1 );
	DistortionAngleStaticText->Enable( false );
	
	bSizer283->Add( DistortionAngleStaticText, 50, wxALIGN_CENTER_VERTICAL|wxALL, 5 );
	
	DistortionAngleTextCtrl = new NumericTextCtrl( this, wxID_ANY, wxT("0.00"), wxDefaultPosition, wxDefaultSize, 0 );
	DistortionAngleTextCtrl->Enable( false );
	
	bSizer283->Add( DistortionAngleTextCtrl, 50, wxALL, 5 );
	
	
	bSizer282->Add( bSizer283, 1, wxEXPAND, 5 );
	
	wxBoxSizer* bSizer2831;
	bSizer2831 = new wxBoxSizer( wxHORIZONTAL );
	
	MajorScaleStaticText = new wxStaticText( this, wxID_ANY, wxT("        Major Scale  :"), wxDefaultPosition, wxDefaultSize, 0 );
	MajorScaleStaticText->Wrap( -1 );
	MajorScaleStaticText->Enable( false );
	
	bSizer2831->Add( MajorScaleStaticText, 50, wxALIGN_CENTER_VERTICAL|wxALL, 5 );
	
	MajorScaleTextCtrl = new NumericTextCtrl( this, wxID_ANY, wxT("1.000"), wxDefaultPosition, wxDefaultSize, 0 );
	MajorScaleTextCtrl->Enable( false );
	
	bSizer2831->Add( MajorScaleTextCtrl, 50, wxALL, 5 );
	
	
	bSizer282->Add( bSizer2831, 1, wxEXPAND, 5 );
	
	wxBoxSizer* bSizer28311;
	bSizer28311 = new wxBoxSizer( wxHORIZONTAL );
	
	MinorScaleStaticText = new wxStaticText( this, wxID_ANY, wxT("        Minor Scale  :"), wxDefaultPosition, wxDefaultSize, 0 );
	MinorScaleStaticText->Wrap( -1 );
	MinorScaleStaticText->Enable( false );
	
	bSizer28311->Add( MinorScaleStaticText, 50, wxALIGN_CENTER_VERTICAL|wxALL, 5 );
	
	MinorScaleTextCtrl = new NumericTextCtrl( this, wxID_ANY, wxT("1.000"), wxDefaultPosition, wxDefaultSize, 0 );
	MinorScaleTextCtrl->Enable( false );
	
	bSizer28311->Add( MinorScaleTextCtrl, 50, wxALL, 5 );
	
	
	bSizer282->Add( bSizer28311, 1, wxEXPAND, 5 );
	
	
	bSizer26->Add( bSizer282, 0, wxEXPAND, 5 );
	
	wxBoxSizer* bSizer3212;
	bSizer3212 = new wxBoxSizer( wxHORIZONTAL );
	
	MoviesHaveInvertedContrast = new wxCheckBox( this, wxID_ANY, wxT("Particles are white"), wxDefaultPosition, wxDefaultSize, 0 );
	bSizer3212->Add( MoviesHaveInvertedContrast, 50, wxALIGN_CENTER_VERTICAL|wxALL, 5 );
	
	
	bSizer3212->Add( 0, 0, 50, wxEXPAND, 5 );
	
	
	bSizer26->Add( bSizer3212, 1, wxEXPAND, 5 );
	
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
	
	
	bSizer26->Add( bSizer33, 0, wxEXPAND, 5 );
	
	
	this->SetSizer( bSizer26 );
	this->Layout();
	bSizer26->Fit( this );
	
	this->Centre( wxBOTH );
	
	// Connect Events
	m_button10->Connect( wxEVT_COMMAND_BUTTON_CLICKED, wxCommandEventHandler( MovieImportDialog::AddFilesClick ), NULL, this );
	m_button11->Connect( wxEVT_COMMAND_BUTTON_CLICKED, wxCommandEventHandler( MovieImportDialog::AddDirectoryClick ), NULL, this );
	s->Connect( wxEVT_COMMAND_BUTTON_CLICKED, wxCommandEventHandler( MovieImportDialog::ClearClick ), NULL, this );
	VoltageCombo->Connect( wxEVT_CHAR, wxKeyEventHandler( MovieImportDialog::OnTextKeyPress ), NULL, this );
	VoltageCombo->Connect( wxEVT_COMMAND_TEXT_UPDATED, wxCommandEventHandler( MovieImportDialog::TextChanged ), NULL, this );
	CsText->Connect( wxEVT_CHAR, wxKeyEventHandler( MovieImportDialog::OnTextKeyPress ), NULL, this );
	CsText->Connect( wxEVT_COMMAND_TEXT_UPDATED, wxCommandEventHandler( MovieImportDialog::TextChanged ), NULL, this );
	PixelSizeText->Connect( wxEVT_CHAR, wxKeyEventHandler( MovieImportDialog::OnTextKeyPress ), NULL, this );
	PixelSizeText->Connect( wxEVT_COMMAND_TEXT_UPDATED, wxCommandEventHandler( MovieImportDialog::TextChanged ), NULL, this );
	DoseText->Connect( wxEVT_CHAR, wxKeyEventHandler( MovieImportDialog::OnTextKeyPress ), NULL, this );
	DoseText->Connect( wxEVT_COMMAND_TEXT_UPDATED, wxCommandEventHandler( MovieImportDialog::TextChanged ), NULL, this );
	ApplyDarkImageCheckbox->Connect( wxEVT_COMMAND_CHECKBOX_CLICKED, wxCommandEventHandler( MovieImportDialog::OnMoviesAreGainCorrectedCheckBox ), NULL, this );
	DarkFilePicker->Connect( wxEVT_COMMAND_FILEPICKER_CHANGED, wxFileDirPickerEventHandler( MovieImportDialog::OnGainFilePickerChanged ), NULL, this );
	ApplyGainImageCheckbox->Connect( wxEVT_COMMAND_CHECKBOX_CLICKED, wxCommandEventHandler( MovieImportDialog::OnMoviesAreGainCorrectedCheckBox ), NULL, this );
	GainFilePicker->Connect( wxEVT_COMMAND_FILEPICKER_CHANGED, wxFileDirPickerEventHandler( MovieImportDialog::OnGainFilePickerChanged ), NULL, this );
	ResampleMoviesCheckBox->Connect( wxEVT_COMMAND_CHECKBOX_CLICKED, wxCommandEventHandler( MovieImportDialog::OnResampleMoviesCheckBox ), NULL, this );
	DesiredPixelSizeTextCtrl->Connect( wxEVT_COMMAND_TEXT_UPDATED, wxCommandEventHandler( MovieImportDialog::TextChanged ), NULL, this );
	CorrectMagDistortionCheckBox->Connect( wxEVT_COMMAND_CHECKBOX_CLICKED, wxCommandEventHandler( MovieImportDialog::OnCorrectMagDistortionCheckBox ), NULL, this );
	DistortionAngleTextCtrl->Connect( wxEVT_COMMAND_TEXT_UPDATED, wxCommandEventHandler( MovieImportDialog::TextChanged ), NULL, this );
	MajorScaleTextCtrl->Connect( wxEVT_COMMAND_TEXT_UPDATED, wxCommandEventHandler( MovieImportDialog::TextChanged ), NULL, this );
	MinorScaleTextCtrl->Connect( wxEVT_COMMAND_TEXT_UPDATED, wxCommandEventHandler( MovieImportDialog::TextChanged ), NULL, this );
	MoviesHaveInvertedContrast->Connect( wxEVT_COMMAND_CHECKBOX_CLICKED, wxCommandEventHandler( MovieImportDialog::OnMoviesHaveInvertedContrastCheckBox ), NULL, this );
	m_button13->Connect( wxEVT_COMMAND_BUTTON_CLICKED, wxCommandEventHandler( MovieImportDialog::CancelClick ), NULL, this );
	ImportButton->Connect( wxEVT_COMMAND_BUTTON_CLICKED, wxCommandEventHandler( MovieImportDialog::ImportClick ), NULL, this );
}

MovieImportDialog::~MovieImportDialog()
{
	// Disconnect Events
	m_button10->Disconnect( wxEVT_COMMAND_BUTTON_CLICKED, wxCommandEventHandler( MovieImportDialog::AddFilesClick ), NULL, this );
	m_button11->Disconnect( wxEVT_COMMAND_BUTTON_CLICKED, wxCommandEventHandler( MovieImportDialog::AddDirectoryClick ), NULL, this );
	s->Disconnect( wxEVT_COMMAND_BUTTON_CLICKED, wxCommandEventHandler( MovieImportDialog::ClearClick ), NULL, this );
	VoltageCombo->Disconnect( wxEVT_CHAR, wxKeyEventHandler( MovieImportDialog::OnTextKeyPress ), NULL, this );
	VoltageCombo->Disconnect( wxEVT_COMMAND_TEXT_UPDATED, wxCommandEventHandler( MovieImportDialog::TextChanged ), NULL, this );
	CsText->Disconnect( wxEVT_CHAR, wxKeyEventHandler( MovieImportDialog::OnTextKeyPress ), NULL, this );
	CsText->Disconnect( wxEVT_COMMAND_TEXT_UPDATED, wxCommandEventHandler( MovieImportDialog::TextChanged ), NULL, this );
	PixelSizeText->Disconnect( wxEVT_CHAR, wxKeyEventHandler( MovieImportDialog::OnTextKeyPress ), NULL, this );
	PixelSizeText->Disconnect( wxEVT_COMMAND_TEXT_UPDATED, wxCommandEventHandler( MovieImportDialog::TextChanged ), NULL, this );
	DoseText->Disconnect( wxEVT_CHAR, wxKeyEventHandler( MovieImportDialog::OnTextKeyPress ), NULL, this );
	DoseText->Disconnect( wxEVT_COMMAND_TEXT_UPDATED, wxCommandEventHandler( MovieImportDialog::TextChanged ), NULL, this );
	ApplyDarkImageCheckbox->Disconnect( wxEVT_COMMAND_CHECKBOX_CLICKED, wxCommandEventHandler( MovieImportDialog::OnMoviesAreGainCorrectedCheckBox ), NULL, this );
	DarkFilePicker->Disconnect( wxEVT_COMMAND_FILEPICKER_CHANGED, wxFileDirPickerEventHandler( MovieImportDialog::OnGainFilePickerChanged ), NULL, this );
	ApplyGainImageCheckbox->Disconnect( wxEVT_COMMAND_CHECKBOX_CLICKED, wxCommandEventHandler( MovieImportDialog::OnMoviesAreGainCorrectedCheckBox ), NULL, this );
	GainFilePicker->Disconnect( wxEVT_COMMAND_FILEPICKER_CHANGED, wxFileDirPickerEventHandler( MovieImportDialog::OnGainFilePickerChanged ), NULL, this );
	ResampleMoviesCheckBox->Disconnect( wxEVT_COMMAND_CHECKBOX_CLICKED, wxCommandEventHandler( MovieImportDialog::OnResampleMoviesCheckBox ), NULL, this );
	DesiredPixelSizeTextCtrl->Disconnect( wxEVT_COMMAND_TEXT_UPDATED, wxCommandEventHandler( MovieImportDialog::TextChanged ), NULL, this );
	CorrectMagDistortionCheckBox->Disconnect( wxEVT_COMMAND_CHECKBOX_CLICKED, wxCommandEventHandler( MovieImportDialog::OnCorrectMagDistortionCheckBox ), NULL, this );
	DistortionAngleTextCtrl->Disconnect( wxEVT_COMMAND_TEXT_UPDATED, wxCommandEventHandler( MovieImportDialog::TextChanged ), NULL, this );
	MajorScaleTextCtrl->Disconnect( wxEVT_COMMAND_TEXT_UPDATED, wxCommandEventHandler( MovieImportDialog::TextChanged ), NULL, this );
	MinorScaleTextCtrl->Disconnect( wxEVT_COMMAND_TEXT_UPDATED, wxCommandEventHandler( MovieImportDialog::TextChanged ), NULL, this );
	MoviesHaveInvertedContrast->Disconnect( wxEVT_COMMAND_CHECKBOX_CLICKED, wxCommandEventHandler( MovieImportDialog::OnMoviesHaveInvertedContrastCheckBox ), NULL, this );
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
	
	wxBoxSizer* bSizer30;
	bSizer30 = new wxBoxSizer( wxHORIZONTAL );
	
	m_staticText21 = new wxStaticText( this, wxID_ANY, wxT("Spherical Aberration (mm) :"), wxDefaultPosition, wxDefaultSize, 0 );
	m_staticText21->Wrap( -1 );
	bSizer30->Add( m_staticText21, 50, wxALIGN_CENTER_VERTICAL|wxALL, 5 );
	
	CsText = new wxTextCtrl( this, wxID_ANY, wxEmptyString, wxDefaultPosition, wxDefaultSize, 0 );
	bSizer30->Add( CsText, 50, wxALIGN_CENTER_VERTICAL|wxALL, 5 );
	
	
	bSizer26->Add( bSizer30, 0, wxEXPAND, 5 );
	
	wxBoxSizer* bSizer29;
	bSizer29 = new wxBoxSizer( wxHORIZONTAL );
	
	m_staticText20 = new wxStaticText( this, wxID_ANY, wxT("Pixel Size (Å) :"), wxDefaultPosition, wxDefaultSize, 0 );
	m_staticText20->Wrap( -1 );
	bSizer29->Add( m_staticText20, 50, wxALIGN_CENTER_VERTICAL|wxALL, 5 );
	
	PixelSizeText = new wxTextCtrl( this, wxID_ANY, wxEmptyString, wxDefaultPosition, wxDefaultSize, 0 );
	bSizer29->Add( PixelSizeText, 50, wxALIGN_CENTER_VERTICAL|wxALL, 5 );
	
	
	bSizer26->Add( bSizer29, 0, wxEXPAND, 5 );
	
	wxBoxSizer* bSizer3212;
	bSizer3212 = new wxBoxSizer( wxVERTICAL );
	
	SaveScaledSumCheckbox = new wxCheckBox( this, wxID_ANY, wxT("Create Scaled Version to Speed up Visualisation?"), wxDefaultPosition, wxDefaultSize, 0 );
	SaveScaledSumCheckbox->SetValue(true); 
	bSizer3212->Add( SaveScaledSumCheckbox, 0, wxALL, 5 );
	
	ImagesHaveInvertedContrast = new wxCheckBox( this, wxID_ANY, wxT("Particles are white"), wxDefaultPosition, wxDefaultSize, 0 );
	bSizer3212->Add( ImagesHaveInvertedContrast, 50, wxALIGN_CENTER_VERTICAL|wxALL, 5 );
	
	
	bSizer26->Add( bSizer3212, 1, wxEXPAND, 5 );
	
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
	CsText->Connect( wxEVT_CHAR, wxKeyEventHandler( ImageImportDialog::OnTextKeyPress ), NULL, this );
	CsText->Connect( wxEVT_COMMAND_TEXT_UPDATED, wxCommandEventHandler( ImageImportDialog::TextChanged ), NULL, this );
	PixelSizeText->Connect( wxEVT_CHAR, wxKeyEventHandler( ImageImportDialog::OnTextKeyPress ), NULL, this );
	PixelSizeText->Connect( wxEVT_COMMAND_TEXT_UPDATED, wxCommandEventHandler( ImageImportDialog::TextChanged ), NULL, this );
	ImagesHaveInvertedContrast->Connect( wxEVT_COMMAND_CHECKBOX_CLICKED, wxCommandEventHandler( ImageImportDialog::OnImagesHaveInvertedContrastCheckBox ), NULL, this );
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
	CsText->Disconnect( wxEVT_CHAR, wxKeyEventHandler( ImageImportDialog::OnTextKeyPress ), NULL, this );
	CsText->Disconnect( wxEVT_COMMAND_TEXT_UPDATED, wxCommandEventHandler( ImageImportDialog::TextChanged ), NULL, this );
	PixelSizeText->Disconnect( wxEVT_CHAR, wxKeyEventHandler( ImageImportDialog::OnTextKeyPress ), NULL, this );
	PixelSizeText->Disconnect( wxEVT_COMMAND_TEXT_UPDATED, wxCommandEventHandler( ImageImportDialog::TextChanged ), NULL, this );
	ImagesHaveInvertedContrast->Disconnect( wxEVT_COMMAND_CHECKBOX_CLICKED, wxCommandEventHandler( ImageImportDialog::OnImagesHaveInvertedContrastCheckBox ), NULL, this );
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
	
	SplitterWindow = new wxSplitterWindow( this, wxID_ANY, wxDefaultPosition, wxDefaultSize, wxSP_3D );
	SplitterWindow->SetSashGravity( 0.2 );
	SplitterWindow->Connect( wxEVT_IDLE, wxIdleEventHandler( AssetParentPanel::SplitterWindowOnIdle ), NULL, this );
	
	LeftPanel = new wxPanel( SplitterWindow, wxID_ANY, wxDefaultPosition, wxDefaultSize, wxTAB_TRAVERSAL );
	wxBoxSizer* bSizer27;
	bSizer27 = new wxBoxSizer( wxVERTICAL );
	
	m_staticText18 = new wxStaticText( LeftPanel, wxID_ANY, wxT("Groups:"), wxDefaultPosition, wxDefaultSize, 0 );
	m_staticText18->Wrap( -1 );
	bSizer27->Add( m_staticText18, 0, wxALL, 5 );
	
	wxBoxSizer* bSizer221;
	bSizer221 = new wxBoxSizer( wxHORIZONTAL );
	
	GroupListBox = new wxListCtrl( LeftPanel, wxID_ANY, wxDefaultPosition, wxDefaultSize, wxLC_EDIT_LABELS|wxLC_NO_HEADER|wxLC_REPORT|wxLC_SINGLE_SEL );
	bSizer221->Add( GroupListBox, 1, wxALL|wxEXPAND, 5 );
	
	
	bSizer27->Add( bSizer221, 1, wxEXPAND, 5 );
	
	wxBoxSizer* bSizer49;
	bSizer49 = new wxBoxSizer( wxHORIZONTAL );
	
	AddGroupButton = new wxButton( LeftPanel, wxID_ANY, wxT("Add"), wxDefaultPosition, wxDefaultSize, 0 );
	bSizer49->Add( AddGroupButton, 0, wxALL, 5 );
	
	RenameGroupButton = new wxButton( LeftPanel, wxID_ANY, wxT("Rename"), wxDefaultPosition, wxDefaultSize, 0 );
	RenameGroupButton->Hide();
	
	bSizer49->Add( RenameGroupButton, 0, wxALL, 5 );
	
	RemoveGroupButton = new wxButton( LeftPanel, wxID_ANY, wxT("Remove"), wxDefaultPosition, wxDefaultSize, 0 );
	bSizer49->Add( RemoveGroupButton, 0, wxALL, 5 );
	
	InvertGroupButton = new wxButton( LeftPanel, wxID_ANY, wxT("Invert"), wxDefaultPosition, wxDefaultSize, 0 );
	bSizer49->Add( InvertGroupButton, 0, wxALL, 5 );
	
	NewFromParentButton = new wxButton( LeftPanel, wxID_ANY, wxT("New From Parent"), wxDefaultPosition, wxDefaultSize, 0 );
	NewFromParentButton->Enable( false );
	
	bSizer49->Add( NewFromParentButton, 0, wxALL, 5 );
	
	
	bSizer27->Add( bSizer49, 0, wxEXPAND, 5 );
	
	
	LeftPanel->SetSizer( bSizer27 );
	LeftPanel->Layout();
	bSizer27->Fit( LeftPanel );
	m_panel3 = new wxPanel( SplitterWindow, wxID_ANY, wxDefaultPosition, wxDefaultSize, wxTAB_TRAVERSAL );
	wxBoxSizer* bSizer30;
	bSizer30 = new wxBoxSizer( wxVERTICAL );
	
	AssetTypeText = new wxStaticText( m_panel3, wxID_ANY, wxT("Asset Type :"), wxDefaultPosition, wxDefaultSize, 0 );
	AssetTypeText->Wrap( -1 );
	bSizer30->Add( AssetTypeText, 0, wxALL, 5 );
	
	wxBoxSizer* bSizer25;
	bSizer25 = new wxBoxSizer( wxVERTICAL );
	
	ContentsListBox = new ContentsList( m_panel3, wxID_ANY, wxDefaultPosition, wxDefaultSize, wxLC_ALIGN_LEFT|wxLC_NO_SORT_HEADER|wxLC_REPORT|wxLC_VIRTUAL|wxLC_VRULES );
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
	
	RenameAssetButton = new wxButton( m_panel3, wxID_ANY, wxT("Rename"), wxDefaultPosition, wxDefaultSize, 0 );
	RenameAssetButton->Enable( false );
	
	bSizer28->Add( RenameAssetButton, 0, wxALL, 5 );
	
	AddSelectedAssetButton = new wxButton( m_panel3, wxID_ANY, wxT("Add To Group"), wxDefaultPosition, wxDefaultSize, 0 );
	AddSelectedAssetButton->Enable( false );
	
	bSizer28->Add( AddSelectedAssetButton, 0, wxALL|wxEXPAND, 5 );
	
	DisplayButton = new wxButton( m_panel3, wxID_ANY, wxT("Display"), wxDefaultPosition, wxDefaultSize, 0 );
	bSizer28->Add( DisplayButton, 0, wxALL, 5 );
	
	
	bSizer25->Add( bSizer28, 0, wxEXPAND, 5 );
	
	
	bSizer30->Add( bSizer25, 100, wxEXPAND, 5 );
	
	
	m_panel3->SetSizer( bSizer30 );
	m_panel3->Layout();
	bSizer30->Fit( m_panel3 );
	SplitterWindow->SplitVertically( LeftPanel, m_panel3, 405 );
	bSizer20->Add( SplitterWindow, 100, wxEXPAND, 5 );
	
	
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
	InvertGroupButton->Connect( wxEVT_COMMAND_BUTTON_CLICKED, wxCommandEventHandler( AssetParentPanel::InvertGroupClick ), NULL, this );
	NewFromParentButton->Connect( wxEVT_COMMAND_BUTTON_CLICKED, wxCommandEventHandler( AssetParentPanel::NewFromParentClick ), NULL, this );
	ContentsListBox->Connect( wxEVT_LEFT_DCLICK, wxMouseEventHandler( AssetParentPanel::MouseCheckContentsVeto ), NULL, this );
	ContentsListBox->Connect( wxEVT_LEFT_DOWN, wxMouseEventHandler( AssetParentPanel::MouseCheckContentsVeto ), NULL, this );
	ContentsListBox->Connect( wxEVT_LEFT_UP, wxMouseEventHandler( AssetParentPanel::MouseVeto ), NULL, this );
	ContentsListBox->Connect( wxEVT_COMMAND_LIST_BEGIN_DRAG, wxListEventHandler( AssetParentPanel::OnBeginContentsDrag ), NULL, this );
	ContentsListBox->Connect( wxEVT_COMMAND_LIST_ITEM_ACTIVATED, wxListEventHandler( AssetParentPanel::OnAssetActivated ), NULL, this );
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
	RenameAssetButton->Connect( wxEVT_COMMAND_BUTTON_CLICKED, wxCommandEventHandler( AssetParentPanel::RenameAssetClick ), NULL, this );
	AddSelectedAssetButton->Connect( wxEVT_COMMAND_BUTTON_CLICKED, wxCommandEventHandler( AssetParentPanel::AddSelectedAssetClick ), NULL, this );
	DisplayButton->Connect( wxEVT_COMMAND_BUTTON_CLICKED, wxCommandEventHandler( AssetParentPanel::OnDisplayButtonClick ), NULL, this );
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
	InvertGroupButton->Disconnect( wxEVT_COMMAND_BUTTON_CLICKED, wxCommandEventHandler( AssetParentPanel::InvertGroupClick ), NULL, this );
	NewFromParentButton->Disconnect( wxEVT_COMMAND_BUTTON_CLICKED, wxCommandEventHandler( AssetParentPanel::NewFromParentClick ), NULL, this );
	ContentsListBox->Disconnect( wxEVT_LEFT_DCLICK, wxMouseEventHandler( AssetParentPanel::MouseCheckContentsVeto ), NULL, this );
	ContentsListBox->Disconnect( wxEVT_LEFT_DOWN, wxMouseEventHandler( AssetParentPanel::MouseCheckContentsVeto ), NULL, this );
	ContentsListBox->Disconnect( wxEVT_LEFT_UP, wxMouseEventHandler( AssetParentPanel::MouseVeto ), NULL, this );
	ContentsListBox->Disconnect( wxEVT_COMMAND_LIST_BEGIN_DRAG, wxListEventHandler( AssetParentPanel::OnBeginContentsDrag ), NULL, this );
	ContentsListBox->Disconnect( wxEVT_COMMAND_LIST_ITEM_ACTIVATED, wxListEventHandler( AssetParentPanel::OnAssetActivated ), NULL, this );
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
	RenameAssetButton->Disconnect( wxEVT_COMMAND_BUTTON_CLICKED, wxCommandEventHandler( AssetParentPanel::RenameAssetClick ), NULL, this );
	AddSelectedAssetButton->Disconnect( wxEVT_COMMAND_BUTTON_CLICKED, wxCommandEventHandler( AssetParentPanel::AddSelectedAssetClick ), NULL, this );
	DisplayButton->Disconnect( wxEVT_COMMAND_BUTTON_CLICKED, wxCommandEventHandler( AssetParentPanel::OnDisplayButtonClick ), NULL, this );
	
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
	
	CommandErrorStaticText = new wxStaticText( CommandsPanel, wxID_ANY, wxEmptyString, wxDefaultPosition, wxDefaultSize, wxALIGN_CENTRE );
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
	InputSizer = new wxBoxSizer( wxVERTICAL );
	
	wxFlexGridSizer* fgSizer1;
	fgSizer1 = new wxFlexGridSizer( 0, 2, 0, 0 );
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
	
	maximum_shift_text = new wxTextCtrl( ExpertPanel, wxID_ANY, wxT("40"), wxDefaultPosition, wxDefaultSize, 0 );
	maximum_shift_text->SetToolTip( wxT("Maximum shift for each alignment round") );
	
	fgSizer1->Add( maximum_shift_text, 0, wxALL|wxEXPAND, 5 );
	
	m_staticText44 = new wxStaticText( ExpertPanel, wxID_ANY, wxT("Exposure Filter"), wxDefaultPosition, wxDefaultSize, 0 );
	m_staticText44->Wrap( -1 );
	m_staticText44->SetFont( wxFont( 10, 74, 90, 92, true, wxT("Sans") ) );
	
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
	m_staticText481->SetFont( wxFont( wxNORMAL_FONT->GetPointSize(), 70, 90, 92, true, wxEmptyString ) );
	
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
	
	TimeRemainingText = new wxStaticText( ProgressPanel, wxID_ANY, wxT("Time Remaining : ???h:??m:??s   "), wxDefaultPosition, wxDefaultSize, wxALIGN_CENTRE );
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

Refine3DPanel::Refine3DPanel( wxWindow* parent, wxWindowID id, const wxPoint& pos, const wxSize& size, long style ) : JobPanel( parent, id, pos, size, style )
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
	PleaseCreateRefinementPackageText->SetFont( wxFont( wxNORMAL_FONT->GetPointSize(), 70, 90, 92, false, wxEmptyString ) );
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
	m_staticText318->SetFont( wxFont( 10, 74, 90, 92, true, wxT("Sans") ) );
	
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
	m_staticText202->SetFont( wxFont( wxNORMAL_FONT->GetPointSize(), 70, 90, 92, true, wxEmptyString ) );
	
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
	m_staticText201->SetFont( wxFont( wxNORMAL_FONT->GetPointSize(), 70, 90, 92, true, wxEmptyString ) );
	
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
	
	NumberToRefineSpinCtrl = new wxSpinCtrl( ExpertPanel, wxID_ANY, wxEmptyString, wxDefaultPosition, wxDefaultSize, wxSP_ARROW_KEYS, -100000, 100000, -21 );
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
	m_staticText200->SetFont( wxFont( 10, 74, 90, 92, true, wxT("Sans") ) );
	
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
	m_staticText323->SetFont( wxFont( 10, 74, 90, 92, true, wxT("Sans") ) );
	
	fgSizer1->Add( m_staticText323, 0, wxALL, 5 );
	
	
	fgSizer1->Add( 0, 0, 1, wxEXPAND, 5 );
	
	m_staticText324 = new wxStaticText( ExpertPanel, wxID_ANY, wxT("Refine CTF?"), wxDefaultPosition, wxDefaultSize, 0 );
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
	m_staticText329->SetFont( wxFont( 10, 74, 90, 92, true, wxT("Sans") ) );
	
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
	m_staticText405->SetFont( wxFont( 10, 74, 90, 92, true, wxT("Sans") ) );
	
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
	
	TimeRemainingText = new wxStaticText( ProgressPanel, wxID_ANY, wxT("Time Remaining : ???h:??m:??s   "), wxDefaultPosition, wxDefaultSize, wxALIGN_CENTRE );
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

RefineCTFParentPanel::RefineCTFParentPanel( wxWindow* parent, wxWindowID id, const wxPoint& pos, const wxSize& size, long style ) : JobPanel( parent, id, pos, size, style )
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
	
	m_checkBox56 = new wxCheckBox( InputParamsPanel, wxID_ANY, wxT("Refine CTF Parameters"), wxDefaultPosition, wxDefaultSize, 0 );
	bSizer215->Add( m_checkBox56, 0, wxALL, 5 );
	
	m_checkBox57 = new wxCheckBox( InputParamsPanel, wxID_ANY, wxT("Refine Beam Tilt"), wxDefaultPosition, wxDefaultSize, 0 );
	bSizer215->Add( m_checkBox57, 0, wxALL, 5 );
	
	
	bSizer215->Add( 0, 0, 1, wxEXPAND, 5 );
	
	
	bSizer357->Add( bSizer215, 0, wxEXPAND, 5 );
	
	wxFlexGridSizer* fgSizer22;
	fgSizer22 = new wxFlexGridSizer( 0, 2, 0, 0 );
	fgSizer22->SetFlexibleDirection( wxBOTH );
	fgSizer22->SetNonFlexibleGrowMode( wxFLEX_GROWMODE_SPECIFIED );
	
	HiResLimitStaticText = new wxStaticText( InputParamsPanel, wxID_ANY, wxT("Hi-Res Limit (Å) :"), wxDefaultPosition, wxDefaultSize, 0 );
	HiResLimitStaticText->Wrap( -1 );
	fgSizer22->Add( HiResLimitStaticText, 0, wxALIGN_CENTER_VERTICAL|wxALIGN_RIGHT|wxALL, 5 );
	
	HighResolutionLimitTextCtrl = new NumericTextCtrl( InputParamsPanel, wxID_ANY, wxT("3.5"), wxDefaultPosition, wxDefaultSize, 0 );
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
	PleaseCreateRefinementPackageText->SetFont( wxFont( wxNORMAL_FONT->GetPointSize(), 70, 90, 92, false, wxEmptyString ) );
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
	
	m_staticText202 = new wxStaticText( ExpertPanel, wxID_ANY, wxT("General Refinement"), wxDefaultPosition, wxDefaultSize, 0 );
	m_staticText202->Wrap( -1 );
	m_staticText202->SetFont( wxFont( wxNORMAL_FONT->GetPointSize(), 70, 90, 92, true, wxEmptyString ) );
	
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
	
	m_staticText323 = new wxStaticText( ExpertPanel, wxID_ANY, wxT("CTF Refinement"), wxDefaultPosition, wxDefaultSize, 0 );
	m_staticText323->Wrap( -1 );
	m_staticText323->SetFont( wxFont( 10, 74, 90, 92, true, wxT("Sans") ) );
	
	fgSizer1->Add( m_staticText323, 0, wxALL, 5 );
	
	
	fgSizer1->Add( 0, 0, 1, wxEXPAND, 5 );
	
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
	m_staticText329->SetFont( wxFont( 10, 74, 90, 92, true, wxT("Sans") ) );
	
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
	m_staticText405->SetFont( wxFont( 10, 74, 90, 92, true, wxT("Sans") ) );
	
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
	
	TimeRemainingText = new wxStaticText( ProgressPanel, wxID_ANY, wxT("Time Remaining : ???h:??m:??s   "), wxDefaultPosition, wxDefaultSize, wxALIGN_CENTRE );
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
	this->Connect( wxEVT_UPDATE_UI, wxUpdateUIEventHandler( RefineCTFParentPanel::OnUpdateUI ) );
	UseMaskCheckBox->Connect( wxEVT_COMMAND_CHECKBOX_CLICKED, wxCommandEventHandler( RefineCTFParentPanel::OnUseMaskCheckBox ), NULL, this );
	HighResolutionLimitTextCtrl->Connect( wxEVT_COMMAND_TEXT_UPDATED, wxCommandEventHandler( RefineCTFParentPanel::OnHighResLimitChange ), NULL, this );
	ExpertToggleButton->Connect( wxEVT_COMMAND_TOGGLEBUTTON_CLICKED, wxCommandEventHandler( RefineCTFParentPanel::OnExpertOptionsToggle ), NULL, this );
	Active3DReferencesListCtrl->Connect( wxEVT_COMMAND_LIST_ITEM_ACTIVATED, wxListEventHandler( RefineCTFParentPanel::OnVolumeListItemActivated ), NULL, this );
	ResetAllDefaultsButton->Connect( wxEVT_COMMAND_BUTTON_CLICKED, wxCommandEventHandler( RefineCTFParentPanel::ResetAllDefaultsClick ), NULL, this );
	AutoMaskYesRadioButton->Connect( wxEVT_COMMAND_RADIOBUTTON_SELECTED, wxCommandEventHandler( RefineCTFParentPanel::OnAutoMaskButton ), NULL, this );
	AutoMaskNoRadioButton->Connect( wxEVT_COMMAND_RADIOBUTTON_SELECTED, wxCommandEventHandler( RefineCTFParentPanel::OnAutoMaskButton ), NULL, this );
	InfoText->Connect( wxEVT_COMMAND_TEXT_URL, wxTextUrlEventHandler( RefineCTFParentPanel::OnInfoURL ), NULL, this );
	FinishButton->Connect( wxEVT_COMMAND_BUTTON_CLICKED, wxCommandEventHandler( RefineCTFParentPanel::FinishButtonClick ), NULL, this );
	CancelAlignmentButton->Connect( wxEVT_COMMAND_BUTTON_CLICKED, wxCommandEventHandler( RefineCTFParentPanel::TerminateButtonClick ), NULL, this );
	StartRefinementButton->Connect( wxEVT_COMMAND_BUTTON_CLICKED, wxCommandEventHandler( RefineCTFParentPanel::StartRefinementClick ), NULL, this );
}

RefineCTFParentPanel::~RefineCTFParentPanel()
{
	// Disconnect Events
	this->Disconnect( wxEVT_UPDATE_UI, wxUpdateUIEventHandler( RefineCTFParentPanel::OnUpdateUI ) );
	UseMaskCheckBox->Disconnect( wxEVT_COMMAND_CHECKBOX_CLICKED, wxCommandEventHandler( RefineCTFParentPanel::OnUseMaskCheckBox ), NULL, this );
	HighResolutionLimitTextCtrl->Disconnect( wxEVT_COMMAND_TEXT_UPDATED, wxCommandEventHandler( RefineCTFParentPanel::OnHighResLimitChange ), NULL, this );
	ExpertToggleButton->Disconnect( wxEVT_COMMAND_TOGGLEBUTTON_CLICKED, wxCommandEventHandler( RefineCTFParentPanel::OnExpertOptionsToggle ), NULL, this );
	Active3DReferencesListCtrl->Disconnect( wxEVT_COMMAND_LIST_ITEM_ACTIVATED, wxListEventHandler( RefineCTFParentPanel::OnVolumeListItemActivated ), NULL, this );
	ResetAllDefaultsButton->Disconnect( wxEVT_COMMAND_BUTTON_CLICKED, wxCommandEventHandler( RefineCTFParentPanel::ResetAllDefaultsClick ), NULL, this );
	AutoMaskYesRadioButton->Disconnect( wxEVT_COMMAND_RADIOBUTTON_SELECTED, wxCommandEventHandler( RefineCTFParentPanel::OnAutoMaskButton ), NULL, this );
	AutoMaskNoRadioButton->Disconnect( wxEVT_COMMAND_RADIOBUTTON_SELECTED, wxCommandEventHandler( RefineCTFParentPanel::OnAutoMaskButton ), NULL, this );
	InfoText->Disconnect( wxEVT_COMMAND_TEXT_URL, wxTextUrlEventHandler( RefineCTFParentPanel::OnInfoURL ), NULL, this );
	FinishButton->Disconnect( wxEVT_COMMAND_BUTTON_CLICKED, wxCommandEventHandler( RefineCTFParentPanel::FinishButtonClick ), NULL, this );
	CancelAlignmentButton->Disconnect( wxEVT_COMMAND_BUTTON_CLICKED, wxCommandEventHandler( RefineCTFParentPanel::TerminateButtonClick ), NULL, this );
	StartRefinementButton->Disconnect( wxEVT_COMMAND_BUTTON_CLICKED, wxCommandEventHandler( RefineCTFParentPanel::StartRefinementClick ), NULL, this );
	
}

Sharpen3DPanelParent::Sharpen3DPanelParent( wxWindow* parent, wxWindowID id, const wxPoint& pos, const wxSize& size, long style ) : wxPanel( parent, id, pos, size, style )
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
	m_staticText1006->SetFont( wxFont( 10, 74, 90, 92, true, wxT("Sans") ) );
	
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
	m_staticText202->SetFont( wxFont( wxNORMAL_FONT->GetPointSize(), 70, 90, 92, true, wxEmptyString ) );
	
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
	
	m_staticText671 = new wxStaticText( this, wxID_ANY, wxT("Additional"), wxDefaultPosition, wxDefaultSize, 0 );
	m_staticText671->Wrap( -1 );
	m_staticText671->SetFont( wxFont( 10, 74, 90, 92, true, wxT("Sans") ) );
	
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
	m_staticText699->SetFont( wxFont( 10, 74, 90, 92, true, wxT("Sans") ) );
	
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
	InvertHandednessYesButton->Disconnect( wxEVT_COMMAND_RADIOBUTTON_SELECTED, wxCommandEventHandler( Sharpen3DPanelParent::OnAutoMaskButton ), NULL, this );
	InvertHandednessNoButton->Disconnect( wxEVT_COMMAND_RADIOBUTTON_SELECTED, wxCommandEventHandler( Sharpen3DPanelParent::OnAutoMaskButton ), NULL, this );
	CorrectGriddingYesButton->Disconnect( wxEVT_COMMAND_RADIOBUTTON_SELECTED, wxCommandEventHandler( Sharpen3DPanelParent::OnAutoMaskButton ), NULL, this );
	CorrectGriddingNoButton->Disconnect( wxEVT_COMMAND_RADIOBUTTON_SELECTED, wxCommandEventHandler( Sharpen3DPanelParent::OnAutoMaskButton ), NULL, this );
	InfoText->Disconnect( wxEVT_COMMAND_TEXT_URL, wxTextUrlEventHandler( Sharpen3DPanelParent::OnInfoURL ), NULL, this );
	RunJobButton->Disconnect( wxEVT_COMMAND_BUTTON_CLICKED, wxCommandEventHandler( Sharpen3DPanelParent::OnRunButtonClick ), NULL, this );
	ImportResultButton->Disconnect( wxEVT_COMMAND_BUTTON_CLICKED, wxCommandEventHandler( Sharpen3DPanelParent::OnImportResultClick ), NULL, this );
	SaveResultButton->Disconnect( wxEVT_COMMAND_BUTTON_CLICKED, wxCommandEventHandler( Sharpen3DPanelParent::OnSaveResultClick ), NULL, this );
	
}

Generate3DPanelParent::Generate3DPanelParent( wxWindow* parent, wxWindowID id, const wxPoint& pos, const wxSize& size, long style ) : JobPanel( parent, id, pos, size, style )
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
	PleaseCreateRefinementPackageText->SetFont( wxFont( wxNORMAL_FONT->GetPointSize(), 70, 90, 92, false, wxEmptyString ) );
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
	m_staticText329->SetFont( wxFont( 10, 74, 90, 92, true, wxT("Sans") ) );
	
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
	
	TimeRemainingText = new wxStaticText( ProgressPanel, wxID_ANY, wxT("Time Remaining : ???h:??m:??s   "), wxDefaultPosition, wxDefaultSize, wxALIGN_CENTRE );
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

AutoRefine3DPanelParent::AutoRefine3DPanelParent( wxWindow* parent, wxWindowID id, const wxPoint& pos, const wxSize& size, long style ) : JobPanel( parent, id, pos, size, style )
{
	wxBoxSizer* bSizer43;
	bSizer43 = new wxBoxSizer( wxVERTICAL );
	
	m_staticline12 = new wxStaticLine( this, wxID_ANY, wxDefaultPosition, wxDefaultSize, wxLI_HORIZONTAL );
	bSizer43->Add( m_staticline12, 0, wxEXPAND | wxALL, 5 );
	
	wxBoxSizer* bSizer451;
	bSizer451 = new wxBoxSizer( wxHORIZONTAL );
	
	wxBoxSizer* bSizer364;
	bSizer364 = new wxBoxSizer( wxVERTICAL );
	
	InputParamsPanel = new wxPanel( this, wxID_ANY, wxDefaultPosition, wxDefaultSize, wxTAB_TRAVERSAL );
	wxBoxSizer* bSizer441;
	bSizer441 = new wxBoxSizer( wxVERTICAL );
	
	wxBoxSizer* bSizer200;
	bSizer200 = new wxBoxSizer( wxHORIZONTAL );
	
	wxFlexGridSizer* fgSizer15;
	fgSizer15 = new wxFlexGridSizer( 0, 2, 0, 0 );
	fgSizer15->SetFlexibleDirection( wxHORIZONTAL );
	fgSizer15->SetNonFlexibleGrowMode( wxFLEX_GROWMODE_SPECIFIED );
	
	m_staticText262 = new wxStaticText( InputParamsPanel, wxID_ANY, wxT("Input Refinement Package :"), wxDefaultPosition, wxSize( -1,-1 ), 0 );
	m_staticText262->Wrap( -1 );
	fgSizer15->Add( m_staticText262, 0, wxALIGN_CENTER_VERTICAL|wxALIGN_RIGHT|wxALL, 5 );
	
	RefinementPackageSelectPanel = new RefinementPackagePickerComboPanel( InputParamsPanel, wxID_ANY, wxDefaultPosition, wxDefaultSize, wxTAB_TRAVERSAL );
	RefinementPackageSelectPanel->SetMinSize( wxSize( 350,-1 ) );
	RefinementPackageSelectPanel->SetMaxSize( wxSize( 350,-1 ) );
	
	fgSizer15->Add( RefinementPackageSelectPanel, 100, wxALIGN_CENTER_VERTICAL|wxALL, 5 );
	
	m_staticText478 = new wxStaticText( InputParamsPanel, wxID_ANY, wxT("Starting Reference :"), wxDefaultPosition, wxDefaultSize, 0 );
	m_staticText478->Wrap( -1 );
	fgSizer15->Add( m_staticText478, 0, wxALIGN_CENTER_VERTICAL|wxALIGN_RIGHT|wxALL, 5 );
	
	ReferenceSelectPanel = new VolumeAssetPickerComboPanel( InputParamsPanel, wxID_ANY, wxDefaultPosition, wxDefaultSize, wxTAB_TRAVERSAL );
	ReferenceSelectPanel->SetMinSize( wxSize( 350,-1 ) );
	ReferenceSelectPanel->SetMaxSize( wxSize( 350,-1 ) );
	
	fgSizer15->Add( ReferenceSelectPanel, 100, wxALIGN_CENTER_VERTICAL|wxALL, 5 );
	
	UseMaskCheckBox = new wxCheckBox( InputParamsPanel, wxID_ANY, wxT("Use a Mask? :"), wxDefaultPosition, wxDefaultSize, 0 );
	fgSizer15->Add( UseMaskCheckBox, 0, wxALIGN_CENTER_VERTICAL|wxALIGN_RIGHT|wxALL, 5 );
	
	MaskSelectPanel = new VolumeAssetPickerComboPanel( InputParamsPanel, wxID_ANY, wxDefaultPosition, wxDefaultSize, wxTAB_TRAVERSAL );
	MaskSelectPanel->SetMinSize( wxSize( 350,-1 ) );
	
	fgSizer15->Add( MaskSelectPanel, 1, wxEXPAND | wxALL, 5 );
	
	
	bSizer200->Add( fgSizer15, 0, wxEXPAND, 5 );
	
	m_staticline54 = new wxStaticLine( InputParamsPanel, wxID_ANY, wxDefaultPosition, wxDefaultSize, wxLI_HORIZONTAL|wxLI_VERTICAL );
	bSizer200->Add( m_staticline54, 0, wxEXPAND | wxALL, 5 );
	
	wxGridSizer* gSizer11;
	gSizer11 = new wxGridSizer( 1, 1, 0, 0 );
	
	wxBoxSizer* bSizer214;
	bSizer214 = new wxBoxSizer( wxHORIZONTAL );
	
	InitialResLimitStaticText = new wxStaticText( InputParamsPanel, wxID_ANY, wxT("Initial Res. Limit (Å) : "), wxDefaultPosition, wxDefaultSize, 0 );
	InitialResLimitStaticText->Wrap( -1 );
	bSizer214->Add( InitialResLimitStaticText, 0, wxALIGN_CENTER_VERTICAL|wxALIGN_RIGHT|wxALL, 5 );
	
	HighResolutionLimitTextCtrl = new NumericTextCtrl( InputParamsPanel, wxID_ANY, wxT("30.00"), wxDefaultPosition, wxDefaultSize, 0 );
	bSizer214->Add( HighResolutionLimitTextCtrl, 0, wxALIGN_CENTER_VERTICAL|wxALL, 5 );
	
	
	gSizer11->Add( bSizer214, 0, 0, 5 );
	
	
	bSizer200->Add( gSizer11, 0, wxEXPAND, 5 );
	
	
	bSizer200->Add( 0, 0, 1, wxEXPAND, 5 );
	
	ExpertToggleButton = new wxToggleButton( InputParamsPanel, wxID_ANY, wxT("Show Expert Options"), wxDefaultPosition, wxDefaultSize, 0 );
	bSizer200->Add( ExpertToggleButton, 0, wxALIGN_BOTTOM|wxALL, 5 );
	
	
	bSizer441->Add( bSizer200, 1, wxEXPAND, 5 );
	
	PleaseCreateRefinementPackageText = new wxStaticText( InputParamsPanel, wxID_ANY, wxT("Please create a refinement package (in the assets panel) in order to perform a 3D refinement."), wxDefaultPosition, wxDefaultSize, 0 );
	PleaseCreateRefinementPackageText->Wrap( -1 );
	PleaseCreateRefinementPackageText->SetFont( wxFont( wxNORMAL_FONT->GetPointSize(), 70, 90, 92, false, wxEmptyString ) );
	PleaseCreateRefinementPackageText->SetForegroundColour( wxColour( 180, 0, 0 ) );
	PleaseCreateRefinementPackageText->Hide();
	
	bSizer441->Add( PleaseCreateRefinementPackageText, 0, wxALL, 5 );
	
	m_staticline105 = new wxStaticLine( InputParamsPanel, wxID_ANY, wxDefaultPosition, wxDefaultSize, wxLI_HORIZONTAL );
	bSizer441->Add( m_staticline105, 0, wxEXPAND | wxALL, 5 );
	
	
	InputParamsPanel->SetSizer( bSizer441 );
	InputParamsPanel->Layout();
	bSizer441->Fit( InputParamsPanel );
	bSizer364->Add( InputParamsPanel, 1, wxEXPAND | wxALL, 5 );
	
	
	bSizer451->Add( bSizer364, 1, wxEXPAND, 5 );
	
	
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
	
	
	InputSizer->Add( 0, 5, 0, wxEXPAND, 5 );
	
	wxFlexGridSizer* fgSizer1;
	fgSizer1 = new wxFlexGridSizer( 0, 2, 0, 0 );
	fgSizer1->SetFlexibleDirection( wxBOTH );
	fgSizer1->SetNonFlexibleGrowMode( wxFLEX_GROWMODE_SPECIFIED );
	
	m_staticText202 = new wxStaticText( ExpertPanel, wxID_ANY, wxT("General Refinement"), wxDefaultPosition, wxDefaultSize, 0 );
	m_staticText202->Wrap( -1 );
	m_staticText202->SetFont( wxFont( wxNORMAL_FONT->GetPointSize(), 70, 90, 92, true, wxEmptyString ) );
	
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
	
	m_staticText330 = new wxStaticText( ExpertPanel, wxID_ANY, wxT("Inner Mask Radius (Å) :"), wxDefaultPosition, wxDefaultSize, 0 );
	m_staticText330->Wrap( -1 );
	fgSizer1->Add( m_staticText330, 0, wxALIGN_CENTER_VERTICAL|wxALL, 5 );
	
	InnerMaskRadiusTextCtrl = new NumericTextCtrl( ExpertPanel, wxID_ANY, wxT("0.00"), wxDefaultPosition, wxDefaultSize, 0 );
	fgSizer1->Add( InnerMaskRadiusTextCtrl, 0, wxALL|wxEXPAND, 5 );
	
	m_staticText201 = new wxStaticText( ExpertPanel, wxID_ANY, wxT("Global Search"), wxDefaultPosition, wxDefaultSize, 0 );
	m_staticText201->Wrap( -1 );
	m_staticText201->SetFont( wxFont( wxNORMAL_FONT->GetPointSize(), 70, 90, 92, true, wxEmptyString ) );
	
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
	
	NumberToRefineSpinCtrl = new wxSpinCtrl( ExpertPanel, wxID_ANY, wxEmptyString, wxDefaultPosition, wxDefaultSize, wxSP_ARROW_KEYS, -100000, 100000, 16 );
	fgSizer1->Add( NumberToRefineSpinCtrl, 0, wxALL, 5 );
	
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
	
	m_staticText329 = new wxStaticText( ExpertPanel, wxID_ANY, wxT("Reconstruction"), wxDefaultPosition, wxDefaultSize, 0 );
	m_staticText329->Wrap( -1 );
	m_staticText329->SetFont( wxFont( wxNORMAL_FONT->GetPointSize(), 70, 90, 92, true, wxT("Sans") ) );
	
	fgSizer1->Add( m_staticText329, 0, wxALL, 5 );
	
	
	fgSizer1->Add( 0, 0, 1, wxEXPAND, 5 );
	
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
	
	wxBoxSizer* bSizer26612;
	bSizer26612 = new wxBoxSizer( wxHORIZONTAL );
	
	AutoCenterYesRadioButton = new wxRadioButton( ExpertPanel, wxID_ANY, wxT("Yes"), wxDefaultPosition, wxDefaultSize, wxRB_GROUP );
	bSizer26612->Add( AutoCenterYesRadioButton, 0, wxALL, 5 );
	
	AutoCenterNoRadioButton = new wxRadioButton( ExpertPanel, wxID_ANY, wxT("No"), wxDefaultPosition, wxDefaultSize, 0 );
	bSizer26612->Add( AutoCenterNoRadioButton, 0, wxALL, 5 );
	
	
	fgSizer1->Add( bSizer26612, 1, wxEXPAND, 5 );
	
	m_staticText405 = new wxStaticText( ExpertPanel, wxID_ANY, wxT("Masking"), wxDefaultPosition, wxDefaultSize, 0 );
	m_staticText405->Wrap( -1 );
	m_staticText405->SetFont( wxFont( wxNORMAL_FONT->GetPointSize(), 70, 90, 92, true, wxT("Sans") ) );
	
	fgSizer1->Add( m_staticText405, 0, wxALL, 5 );
	
	
	fgSizer1->Add( 0, 0, 1, wxEXPAND, 5 );
	
	AutoMaskStaticText = new wxStaticText( ExpertPanel, wxID_ANY, wxT("Use Auto-Masking?"), wxDefaultPosition, wxDefaultSize, 0 );
	AutoMaskStaticText->Wrap( -1 );
	fgSizer1->Add( AutoMaskStaticText, 0, wxALL, 5 );
	
	wxBoxSizer* bSizer2641;
	bSizer2641 = new wxBoxSizer( wxHORIZONTAL );
	
	AutoMaskYesRadioButton = new wxRadioButton( ExpertPanel, wxID_ANY, wxT("Yes"), wxDefaultPosition, wxDefaultSize, wxRB_GROUP );
	bSizer2641->Add( AutoMaskYesRadioButton, 0, wxALL, 5 );
	
	AutoMaskNoRadioButton = new wxRadioButton( ExpertPanel, wxID_ANY, wxT("No"), wxDefaultPosition, wxDefaultSize, 0 );
	bSizer2641->Add( AutoMaskNoRadioButton, 0, wxALL, 5 );
	
	
	fgSizer1->Add( bSizer2641, 1, wxEXPAND, 5 );
	
	MaskEdgeStaticText = new wxStaticText( ExpertPanel, wxID_ANY, wxT("Mask Edge Width (Å) :"), wxDefaultPosition, wxDefaultSize, 0 );
	MaskEdgeStaticText->Wrap( -1 );
	fgSizer1->Add( MaskEdgeStaticText, 0, wxALL, 5 );
	
	MaskEdgeTextCtrl = new NumericTextCtrl( ExpertPanel, wxID_ANY, wxT("10.00"), wxDefaultPosition, wxDefaultSize, 0 );
	fgSizer1->Add( MaskEdgeTextCtrl, 0, wxALL, 5 );
	
	MaskWeightStaticText = new wxStaticText( ExpertPanel, wxID_ANY, wxT("Outside Weight :"), wxDefaultPosition, wxDefaultSize, 0 );
	MaskWeightStaticText->Wrap( -1 );
	fgSizer1->Add( MaskWeightStaticText, 0, wxALL, 5 );
	
	MaskWeightTextCtrl = new NumericTextCtrl( ExpertPanel, wxID_ANY, wxT("0.00"), wxDefaultPosition, wxDefaultSize, 0 );
	fgSizer1->Add( MaskWeightTextCtrl, 0, wxALL, 5 );
	
	LowPassYesNoStaticText = new wxStaticText( ExpertPanel, wxID_ANY, wxT("Low-Pass Filter Outside Mask?"), wxDefaultPosition, wxDefaultSize, 0 );
	LowPassYesNoStaticText->Wrap( -1 );
	fgSizer1->Add( LowPassYesNoStaticText, 0, wxALL, 5 );
	
	wxBoxSizer* bSizer26611;
	bSizer26611 = new wxBoxSizer( wxHORIZONTAL );
	
	LowPassMaskYesRadio = new wxRadioButton( ExpertPanel, wxID_ANY, wxT("Yes"), wxDefaultPosition, wxDefaultSize, wxRB_GROUP );
	bSizer26611->Add( LowPassMaskYesRadio, 0, wxALL, 5 );
	
	LowPassMaskNoRadio = new wxRadioButton( ExpertPanel, wxID_ANY, wxT("No"), wxDefaultPosition, wxDefaultSize, 0 );
	bSizer26611->Add( LowPassMaskNoRadio, 0, wxALL, 5 );
	
	
	fgSizer1->Add( bSizer26611, 1, wxEXPAND, 5 );
	
	FilterResolutionStaticText = new wxStaticText( ExpertPanel, wxID_ANY, wxT("\tFilter Resolution (Å) :"), wxDefaultPosition, wxDefaultSize, 0 );
	FilterResolutionStaticText->Wrap( -1 );
	fgSizer1->Add( FilterResolutionStaticText, 0, wxALL, 5 );
	
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
	bSizer61 = new wxBoxSizer( wxVERTICAL );
	
	InfoText = new wxRichTextCtrl( InfoPanel, wxID_ANY, wxEmptyString, wxDefaultPosition, wxDefaultSize, wxTE_READONLY|wxHSCROLL|wxVSCROLL );
	bSizer61->Add( InfoText, 1, wxEXPAND | wxALL, 5 );
	
	
	InfoPanel->SetSizer( bSizer61 );
	InfoPanel->Layout();
	bSizer61->Fit( InfoPanel );
	bSizer46->Add( InfoPanel, 1, wxEXPAND | wxALL, 5 );
	
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
	
	TimeRemainingText = new wxStaticText( ProgressPanel, wxID_ANY, wxT("Time Remaining : ???h:??m:??s   "), wxDefaultPosition, wxDefaultSize, wxALIGN_CENTRE );
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
	this->Connect( wxEVT_UPDATE_UI, wxUpdateUIEventHandler( AutoRefine3DPanelParent::OnUpdateUI ) );
	UseMaskCheckBox->Connect( wxEVT_COMMAND_CHECKBOX_CLICKED, wxCommandEventHandler( AutoRefine3DPanelParent::OnUseMaskCheckBox ), NULL, this );
	ExpertToggleButton->Connect( wxEVT_COMMAND_TOGGLEBUTTON_CLICKED, wxCommandEventHandler( AutoRefine3DPanelParent::OnExpertOptionsToggle ), NULL, this );
	ResetAllDefaultsButton->Connect( wxEVT_COMMAND_BUTTON_CLICKED, wxCommandEventHandler( AutoRefine3DPanelParent::ResetAllDefaultsClick ), NULL, this );
	AutoMaskYesRadioButton->Connect( wxEVT_COMMAND_RADIOBUTTON_SELECTED, wxCommandEventHandler( AutoRefine3DPanelParent::OnAutoMaskButton ), NULL, this );
	AutoMaskNoRadioButton->Connect( wxEVT_COMMAND_RADIOBUTTON_SELECTED, wxCommandEventHandler( AutoRefine3DPanelParent::OnAutoMaskButton ), NULL, this );
	InfoText->Connect( wxEVT_COMMAND_TEXT_URL, wxTextUrlEventHandler( AutoRefine3DPanelParent::OnInfoURL ), NULL, this );
	FinishButton->Connect( wxEVT_COMMAND_BUTTON_CLICKED, wxCommandEventHandler( AutoRefine3DPanelParent::FinishButtonClick ), NULL, this );
	CancelAlignmentButton->Connect( wxEVT_COMMAND_BUTTON_CLICKED, wxCommandEventHandler( AutoRefine3DPanelParent::TerminateButtonClick ), NULL, this );
	StartRefinementButton->Connect( wxEVT_COMMAND_BUTTON_CLICKED, wxCommandEventHandler( AutoRefine3DPanelParent::StartRefinementClick ), NULL, this );
}

AutoRefine3DPanelParent::~AutoRefine3DPanelParent()
{
	// Disconnect Events
	this->Disconnect( wxEVT_UPDATE_UI, wxUpdateUIEventHandler( AutoRefine3DPanelParent::OnUpdateUI ) );
	UseMaskCheckBox->Disconnect( wxEVT_COMMAND_CHECKBOX_CLICKED, wxCommandEventHandler( AutoRefine3DPanelParent::OnUseMaskCheckBox ), NULL, this );
	ExpertToggleButton->Disconnect( wxEVT_COMMAND_TOGGLEBUTTON_CLICKED, wxCommandEventHandler( AutoRefine3DPanelParent::OnExpertOptionsToggle ), NULL, this );
	ResetAllDefaultsButton->Disconnect( wxEVT_COMMAND_BUTTON_CLICKED, wxCommandEventHandler( AutoRefine3DPanelParent::ResetAllDefaultsClick ), NULL, this );
	AutoMaskYesRadioButton->Disconnect( wxEVT_COMMAND_RADIOBUTTON_SELECTED, wxCommandEventHandler( AutoRefine3DPanelParent::OnAutoMaskButton ), NULL, this );
	AutoMaskNoRadioButton->Disconnect( wxEVT_COMMAND_RADIOBUTTON_SELECTED, wxCommandEventHandler( AutoRefine3DPanelParent::OnAutoMaskButton ), NULL, this );
	InfoText->Disconnect( wxEVT_COMMAND_TEXT_URL, wxTextUrlEventHandler( AutoRefine3DPanelParent::OnInfoURL ), NULL, this );
	FinishButton->Disconnect( wxEVT_COMMAND_BUTTON_CLICKED, wxCommandEventHandler( AutoRefine3DPanelParent::FinishButtonClick ), NULL, this );
	CancelAlignmentButton->Disconnect( wxEVT_COMMAND_BUTTON_CLICKED, wxCommandEventHandler( AutoRefine3DPanelParent::TerminateButtonClick ), NULL, this );
	StartRefinementButton->Disconnect( wxEVT_COMMAND_BUTTON_CLICKED, wxCommandEventHandler( AutoRefine3DPanelParent::StartRefinementClick ), NULL, this );
	
}

FindCTFPanel::FindCTFPanel( wxWindow* parent, wxWindowID id, const wxPoint& pos, const wxSize& size, long style ) : JobPanel( parent, id, pos, size, style )
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
	ExpertPanel->Hide();
	
	InputSizer = new wxBoxSizer( wxVERTICAL );
	
	wxFlexGridSizer* fgSizer1;
	fgSizer1 = new wxFlexGridSizer( 0, 2, 0, 0 );
	fgSizer1->SetFlexibleDirection( wxBOTH );
	fgSizer1->SetNonFlexibleGrowMode( wxFLEX_GROWMODE_SPECIFIED );
	
	m_staticText202 = new wxStaticText( ExpertPanel, wxID_ANY, wxT("Search"), wxDefaultPosition, wxDefaultSize, 0 );
	m_staticText202->Wrap( -1 );
	m_staticText202->SetFont( wxFont( wxNORMAL_FONT->GetPointSize(), 70, 90, 92, true, wxEmptyString ) );
	
	fgSizer1->Add( m_staticText202, 0, wxALL, 5 );
	
	
	fgSizer1->Add( 0, 0, 1, wxEXPAND, 5 );
	
	m_staticText186 = new wxStaticText( ExpertPanel, wxID_ANY, wxT("Estimate Using :"), wxDefaultPosition, wxDefaultSize, 0 );
	m_staticText186->Wrap( -1 );
	fgSizer1->Add( m_staticText186, 0, wxALIGN_CENTER_VERTICAL|wxALL, 5 );
	
	wxBoxSizer* bSizer123;
	bSizer123 = new wxBoxSizer( wxHORIZONTAL );
	
	MovieRadioButton = new wxRadioButton( ExpertPanel, wxID_ANY, wxT("Movies"), wxDefaultPosition, wxDefaultSize, 0 );
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
	
	m_staticText201 = new wxStaticText( ExpertPanel, wxID_ANY, wxT("Search Limits"), wxDefaultPosition, wxDefaultSize, 0 );
	m_staticText201->Wrap( -1 );
	m_staticText201->SetFont( wxFont( wxNORMAL_FONT->GetPointSize(), 70, 90, 92, true, wxEmptyString ) );
	
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
	m_staticText200->SetFont( wxFont( 10, 74, 90, 92, true, wxT("Sans") ) );
	
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
	
	TimeRemainingText = new wxStaticText( ProgressPanel, wxID_ANY, wxT("Time Remaining : ???h:??m:??s   "), wxDefaultPosition, wxDefaultSize, wxALIGN_CENTRE );
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
	LargeAstigmatismExpectedCheckBox->Connect( wxEVT_COMMAND_CHECKBOX_CLICKED, wxCommandEventHandler( FindCTFPanel::OnLargeAstigmatismExpectedCheckBox ), NULL, this );
	RestrainAstigmatismCheckBox->Connect( wxEVT_COMMAND_CHECKBOX_CLICKED, wxCommandEventHandler( FindCTFPanel::OnRestrainAstigmatismCheckBox ), NULL, this );
	AdditionalPhaseShiftCheckBox->Connect( wxEVT_COMMAND_CHECKBOX_CLICKED, wxCommandEventHandler( FindCTFPanel::OnFindAdditionalPhaseCheckBox ), NULL, this );
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
	LargeAstigmatismExpectedCheckBox->Disconnect( wxEVT_COMMAND_CHECKBOX_CLICKED, wxCommandEventHandler( FindCTFPanel::OnLargeAstigmatismExpectedCheckBox ), NULL, this );
	RestrainAstigmatismCheckBox->Disconnect( wxEVT_COMMAND_CHECKBOX_CLICKED, wxCommandEventHandler( FindCTFPanel::OnRestrainAstigmatismCheckBox ), NULL, this );
	AdditionalPhaseShiftCheckBox->Disconnect( wxEVT_COMMAND_CHECKBOX_CLICKED, wxCommandEventHandler( FindCTFPanel::OnFindAdditionalPhaseCheckBox ), NULL, this );
	InfoText->Disconnect( wxEVT_COMMAND_TEXT_URL, wxTextUrlEventHandler( FindCTFPanel::OnInfoURL ), NULL, this );
	FinishButton->Disconnect( wxEVT_COMMAND_BUTTON_CLICKED, wxCommandEventHandler( FindCTFPanel::FinishButtonClick ), NULL, this );
	CancelAlignmentButton->Disconnect( wxEVT_COMMAND_BUTTON_CLICKED, wxCommandEventHandler( FindCTFPanel::TerminateButtonClick ), NULL, this );
	StartEstimationButton->Disconnect( wxEVT_COMMAND_BUTTON_CLICKED, wxCommandEventHandler( FindCTFPanel::StartEstimationClick ), NULL, this );
	
}

MatchTemplateParentPanel::MatchTemplateParentPanel( wxWindow* parent, wxWindowID id, const wxPoint& pos, const wxSize& size, long style ) : JobPanel( parent, id, pos, size, style )
{
	wxBoxSizer* bSizer43;
	bSizer43 = new wxBoxSizer( wxVERTICAL );
	
	m_staticline12 = new wxStaticLine( this, wxID_ANY, wxDefaultPosition, wxDefaultSize, wxLI_HORIZONTAL );
	bSizer43->Add( m_staticline12, 0, wxEXPAND | wxALL, 5 );
	
	wxBoxSizer* bSizer45;
	bSizer45 = new wxBoxSizer( wxHORIZONTAL );
	
	wxBoxSizer* bSizer557;
	bSizer557 = new wxBoxSizer( wxHORIZONTAL );
	
	wxFlexGridSizer* fgSizer15;
	fgSizer15 = new wxFlexGridSizer( 0, 2, 0, 0 );
	fgSizer15->SetFlexibleDirection( wxHORIZONTAL );
	fgSizer15->SetNonFlexibleGrowMode( wxFLEX_GROWMODE_SPECIFIED );
	
	m_staticText262 = new wxStaticText( this, wxID_ANY, wxT("Input Image Group :"), wxDefaultPosition, wxSize( -1,-1 ), 0 );
	m_staticText262->Wrap( -1 );
	fgSizer15->Add( m_staticText262, 0, wxALIGN_CENTER_VERTICAL|wxALIGN_RIGHT|wxALL, 5 );
	
	GroupComboBox = new ImageGroupPickerComboPanel( this, wxID_ANY, wxDefaultPosition, wxDefaultSize, wxTAB_TRAVERSAL );
	GroupComboBox->SetMinSize( wxSize( 350,-1 ) );
	GroupComboBox->SetMaxSize( wxSize( 350,-1 ) );
	
	fgSizer15->Add( GroupComboBox, 1, wxEXPAND | wxALL, 5 );
	
	m_staticText478 = new wxStaticText( this, wxID_ANY, wxT("Reference Volume :"), wxDefaultPosition, wxDefaultSize, 0 );
	m_staticText478->Wrap( -1 );
	fgSizer15->Add( m_staticText478, 0, wxALIGN_CENTER_VERTICAL|wxALIGN_RIGHT|wxALL, 5 );
	
	ReferenceSelectPanel = new VolumeAssetPickerComboPanel( this, wxID_ANY, wxDefaultPosition, wxDefaultSize, wxTAB_TRAVERSAL );
	ReferenceSelectPanel->SetMinSize( wxSize( 350,-1 ) );
	ReferenceSelectPanel->SetMaxSize( wxSize( 350,-1 ) );
	
	fgSizer15->Add( ReferenceSelectPanel, 100, wxALIGN_CENTER_VERTICAL|wxALL, 5 );
	
	
	bSizer557->Add( fgSizer15, 0, wxEXPAND, 5 );
	
	m_staticline155 = new wxStaticLine( this, wxID_ANY, wxDefaultPosition, wxDefaultSize, wxLI_HORIZONTAL|wxLI_VERTICAL );
	bSizer557->Add( m_staticline155, 0, wxEXPAND | wxALL, 5 );
	
	wxBoxSizer* bSizer559;
	bSizer559 = new wxBoxSizer( wxVERTICAL );
	
	wxBoxSizer* bSizer560;
	bSizer560 = new wxBoxSizer( wxHORIZONTAL );
	
	
	bSizer559->Add( bSizer560, 0, wxEXPAND, 5 );
	
	
	bSizer557->Add( bSizer559, 0, wxEXPAND, 5 );
	
	
	bSizer45->Add( bSizer557, 1, wxEXPAND, 5 );
	
	ExpertToggleButton = new wxToggleButton( this, wxID_ANY, wxT("Show Expert Options"), wxDefaultPosition, wxDefaultSize, 0 );
	bSizer45->Add( ExpertToggleButton, 0, wxALIGN_BOTTOM|wxALIGN_RIGHT|wxALL, 5 );
	
	
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
	
	m_staticText201 = new wxStaticText( ExpertPanel, wxID_ANY, wxT("Search Limits"), wxDefaultPosition, wxDefaultSize, 0 );
	m_staticText201->Wrap( -1 );
	m_staticText201->SetFont( wxFont( wxNORMAL_FONT->GetPointSize(), 70, 90, 92, true, wxEmptyString ) );
	
	fgSizer1->Add( m_staticText201, 0, wxALL, 5 );
	
	
	fgSizer1->Add( 0, 0, 1, wxEXPAND, 5 );
	
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
	
	m_staticText764 = new wxStaticText( ExpertPanel, wxID_ANY, wxT("Est. Particle Size (Å) : "), wxDefaultPosition, wxDefaultSize, 0 );
	m_staticText764->Wrap( -1 );
	fgSizer1->Add( m_staticText764, 0, wxALIGN_CENTER_VERTICAL|wxALL, 5 );
	
	EstimatedParticleSizeTextCtrl = new NumericTextCtrl( ExpertPanel, wxID_ANY, wxT("200"), wxDefaultPosition, wxDefaultSize, 0 );
	fgSizer1->Add( EstimatedParticleSizeTextCtrl, 0, wxALL|wxEXPAND, 5 );
	
	m_staticText1901 = new wxStaticText( ExpertPanel, wxID_ANY, wxT("Defocus Search Range (Å) :"), wxDefaultPosition, wxDefaultSize, 0 );
	m_staticText1901->Wrap( -1 );
	fgSizer1->Add( m_staticText1901, 0, wxALIGN_CENTER_VERTICAL|wxALL, 5 );
	
	DefocusSearchRangeNumericCtrl = new NumericTextCtrl( ExpertPanel, wxID_ANY, wxT("0.0"), wxDefaultPosition, wxDefaultSize, wxTE_PROCESS_ENTER );
	fgSizer1->Add( DefocusSearchRangeNumericCtrl, 0, wxALL|wxEXPAND, 5 );
	
	m_staticText19011 = new wxStaticText( ExpertPanel, wxID_ANY, wxT("Defocus Search Step (Å) :"), wxDefaultPosition, wxDefaultSize, 0 );
	m_staticText19011->Wrap( -1 );
	fgSizer1->Add( m_staticText19011, 0, wxALIGN_CENTER_VERTICAL|wxALL, 5 );
	
	DefocusSearchStepNumericCtrl = new NumericTextCtrl( ExpertPanel, wxID_ANY, wxT("200"), wxDefaultPosition, wxDefaultSize, wxTE_PROCESS_ENTER );
	fgSizer1->Add( DefocusSearchStepNumericCtrl, 0, wxALL|wxEXPAND, 5 );
	
	m_staticText1902 = new wxStaticText( ExpertPanel, wxID_ANY, wxT("Pixel Size Search Range (Å) :"), wxDefaultPosition, wxDefaultSize, 0 );
	m_staticText1902->Wrap( -1 );
	fgSizer1->Add( m_staticText1902, 0, wxALIGN_CENTER_VERTICAL|wxALL, 5 );
	
	PixelSizeSearchRangeNumericCtrl = new NumericTextCtrl( ExpertPanel, wxID_ANY, wxT("0.0"), wxDefaultPosition, wxDefaultSize, wxTE_PROCESS_ENTER );
	fgSizer1->Add( PixelSizeSearchRangeNumericCtrl, 0, wxALL|wxEXPAND, 5 );
	
	m_staticText19022 = new wxStaticText( ExpertPanel, wxID_ANY, wxT("Pixel Size Search Step (Å) :"), wxDefaultPosition, wxDefaultSize, 0 );
	m_staticText19022->Wrap( -1 );
	fgSizer1->Add( m_staticText19022, 0, wxALIGN_CENTER_VERTICAL|wxALL, 5 );
	
	PixelSizeSearchStepNumericCtrl = new NumericTextCtrl( ExpertPanel, wxID_ANY, wxT("0.01"), wxDefaultPosition, wxDefaultSize, wxTE_PROCESS_ENTER );
	fgSizer1->Add( PixelSizeSearchStepNumericCtrl, 0, wxALL|wxEXPAND, 5 );
	
	m_staticText19021 = new wxStaticText( ExpertPanel, wxID_ANY, wxT("Pointgroup Symmetry :"), wxDefaultPosition, wxDefaultSize, 0 );
	m_staticText19021->Wrap( -1 );
	fgSizer1->Add( m_staticText19021, 0, wxALIGN_CENTER_VERTICAL|wxALL, 5 );
	
	SymmetryComboBox = new wxComboBox( ExpertPanel, wxID_ANY, wxT("C1"), wxDefaultPosition, wxDefaultSize, 0, NULL, 0 ); 
	fgSizer1->Add( SymmetryComboBox, 0, wxALL|wxEXPAND, 5 );
	
	
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
	
	TimeRemainingText = new wxStaticText( ProgressPanel, wxID_ANY, wxT("Time Remaining : ???h:??m:??s   "), wxDefaultPosition, wxDefaultSize, wxALIGN_CENTRE );
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
	this->Connect( wxEVT_UPDATE_UI, wxUpdateUIEventHandler( MatchTemplateParentPanel::OnUpdateUI ) );
	ExpertToggleButton->Connect( wxEVT_COMMAND_TOGGLEBUTTON_CLICKED, wxCommandEventHandler( MatchTemplateParentPanel::OnExpertOptionsToggle ), NULL, this );
	InfoText->Connect( wxEVT_COMMAND_TEXT_URL, wxTextUrlEventHandler( MatchTemplateParentPanel::OnInfoURL ), NULL, this );
	FinishButton->Connect( wxEVT_COMMAND_BUTTON_CLICKED, wxCommandEventHandler( MatchTemplateParentPanel::FinishButtonClick ), NULL, this );
	CancelAlignmentButton->Connect( wxEVT_COMMAND_BUTTON_CLICKED, wxCommandEventHandler( MatchTemplateParentPanel::TerminateButtonClick ), NULL, this );
	StartEstimationButton->Connect( wxEVT_COMMAND_BUTTON_CLICKED, wxCommandEventHandler( MatchTemplateParentPanel::StartEstimationClick ), NULL, this );
}

MatchTemplateParentPanel::~MatchTemplateParentPanel()
{
	// Disconnect Events
	this->Disconnect( wxEVT_UPDATE_UI, wxUpdateUIEventHandler( MatchTemplateParentPanel::OnUpdateUI ) );
	ExpertToggleButton->Disconnect( wxEVT_COMMAND_TOGGLEBUTTON_CLICKED, wxCommandEventHandler( MatchTemplateParentPanel::OnExpertOptionsToggle ), NULL, this );
	InfoText->Disconnect( wxEVT_COMMAND_TEXT_URL, wxTextUrlEventHandler( MatchTemplateParentPanel::OnInfoURL ), NULL, this );
	FinishButton->Disconnect( wxEVT_COMMAND_BUTTON_CLICKED, wxCommandEventHandler( MatchTemplateParentPanel::FinishButtonClick ), NULL, this );
	CancelAlignmentButton->Disconnect( wxEVT_COMMAND_BUTTON_CLICKED, wxCommandEventHandler( MatchTemplateParentPanel::TerminateButtonClick ), NULL, this );
	StartEstimationButton->Disconnect( wxEVT_COMMAND_BUTTON_CLICKED, wxCommandEventHandler( MatchTemplateParentPanel::StartEstimationClick ), NULL, this );
	
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

ExportRefinementPackageWizardParent::ExportRefinementPackageWizardParent( wxWindow* parent, wxWindowID id, const wxString& title, const wxBitmap& bitmap, const wxPoint& pos, long style ) 
{
	this->Create( parent, id, title, bitmap, pos, style );
	this->SetSizeHints( wxSize( 700,400 ), wxDefaultSize );
	
	wxWizardPageSimple* ChooseParamsPage = new wxWizardPageSimple( this );
	m_pages.Add( ChooseParamsPage );
	
	wxBoxSizer* bSizer39311;
	bSizer39311 = new wxBoxSizer( wxVERTICAL );
	
	m_staticText46511 = new wxStaticText( ChooseParamsPage, wxID_ANY, wxT("Export which parameters? :-"), wxDefaultPosition, wxDefaultSize, 0 );
	m_staticText46511->Wrap( -1 );
	bSizer39311->Add( m_staticText46511, 0, wxALL, 5 );
	
	m_staticline10511 = new wxStaticLine( ChooseParamsPage, wxID_ANY, wxDefaultPosition, wxDefaultSize, wxLI_HORIZONTAL );
	bSizer39311->Add( m_staticline10511, 0, wxEXPAND | wxALL, 5 );
	
	wxBoxSizer* bSizer419;
	bSizer419 = new wxBoxSizer( wxHORIZONTAL );
	
	ParameterSelectPanel = new RefinementPickerComboPanel( ChooseParamsPage, wxID_ANY, wxDefaultPosition, wxDefaultSize, wxTAB_TRAVERSAL );
	bSizer419->Add( ParameterSelectPanel, 1, wxEXPAND | wxALL, 5 );
	
	m_staticline123 = new wxStaticLine( ChooseParamsPage, wxID_ANY, wxDefaultPosition, wxDefaultSize, wxLI_VERTICAL );
	bSizer419->Add( m_staticline123, 0, wxEXPAND | wxALL, 5 );
	
	ClassComboBox = new wxComboBox( ChooseParamsPage, wxID_ANY, wxEmptyString, wxDefaultPosition, wxDefaultSize, 0, NULL, wxCB_READONLY ); 
	ClassComboBox->SetMinSize( wxSize( 150,-1 ) );
	
	bSizer419->Add( ClassComboBox, 0, wxALL, 5 );
	
	
	bSizer39311->Add( bSizer419, 0, wxEXPAND, 5 );
	
	
	ChooseParamsPage->SetSizer( bSizer39311 );
	ChooseParamsPage->Layout();
	bSizer39311->Fit( ChooseParamsPage );
	wxWizardPageSimple* ExportTypePage = new wxWizardPageSimple( this );
	m_pages.Add( ExportTypePage );
	
	wxBoxSizer* bSizer3931;
	bSizer3931 = new wxBoxSizer( wxVERTICAL );
	
	m_staticText4651 = new wxStaticText( ExportTypePage, wxID_ANY, wxT("Export to what format? :-"), wxDefaultPosition, wxDefaultSize, 0 );
	m_staticText4651->Wrap( -1 );
	bSizer3931->Add( m_staticText4651, 0, wxALL, 5 );
	
	m_staticline1051 = new wxStaticLine( ExportTypePage, wxID_ANY, wxDefaultPosition, wxDefaultSize, wxLI_HORIZONTAL );
	bSizer3931->Add( m_staticline1051, 0, wxEXPAND | wxALL, 5 );
	
	wxBoxSizer* bSizer3941;
	bSizer3941 = new wxBoxSizer( wxVERTICAL );
	
	FrealignRadioButton = new wxRadioButton( ExportTypePage, wxID_ANY, wxT("Frealign"), wxDefaultPosition, wxDefaultSize, 0 );
	bSizer3941->Add( FrealignRadioButton, 0, wxALL, 5 );
	
	RelionRadioButton = new wxRadioButton( ExportTypePage, wxID_ANY, wxT("Relion "), wxDefaultPosition, wxDefaultSize, 0 );
	bSizer3941->Add( RelionRadioButton, 0, wxALL, 5 );
	
	
	bSizer3931->Add( bSizer3941, 1, wxEXPAND, 5 );
	
	
	ExportTypePage->SetSizer( bSizer3931 );
	ExportTypePage->Layout();
	bSizer3931->Fit( ExportTypePage );
	wxWizardPageSimple* GetPathPage = new wxWizardPageSimple( this );
	m_pages.Add( GetPathPage );
	
	GetPathPage->SetMinSize( wxSize( 600,300 ) );
	
	wxBoxSizer* bSizer471;
	bSizer471 = new wxBoxSizer( wxVERTICAL );
	
	m_staticText4741 = new wxStaticText( GetPathPage, wxID_ANY, wxT("Please provide the stack and metadata files"), wxDefaultPosition, wxDefaultSize, 0 );
	m_staticText4741->Wrap( -1 );
	bSizer471->Add( m_staticText4741, 0, wxALL, 5 );
	
	m_staticline1061 = new wxStaticLine( GetPathPage, wxID_ANY, wxDefaultPosition, wxDefaultSize, wxLI_HORIZONTAL );
	bSizer471->Add( m_staticline1061, 0, wxEXPAND | wxALL, 5 );
	
	m_staticText411 = new wxStaticText( GetPathPage, wxID_ANY, wxT("Output Particle Stack Filename :-"), wxDefaultPosition, wxDefaultSize, 0 );
	m_staticText411->Wrap( -1 );
	bSizer471->Add( m_staticText411, 0, wxALIGN_CENTER_VERTICAL|wxALL, 5 );
	
	wxBoxSizer* bSizer5011;
	bSizer5011 = new wxBoxSizer( wxHORIZONTAL );
	
	ParticleStackFileTextCtrl = new wxTextCtrl( GetPathPage, wxID_ANY, wxEmptyString, wxDefaultPosition, wxDefaultSize, 0 );
	bSizer5011->Add( ParticleStackFileTextCtrl, 1, wxALIGN_CENTER_VERTICAL|wxALL, 5 );
	
	m_button2411 = new wxButton( GetPathPage, wxID_ANY, wxT("Browse..."), wxDefaultPosition, wxDefaultSize, 0 );
	bSizer5011->Add( m_button2411, 0, wxALL, 5 );
	
	
	bSizer471->Add( bSizer5011, 0, wxEXPAND, 5 );
	
	MetaFilenameStaticText = new wxStaticText( GetPathPage, wxID_ANY, wxT("Output PAR / STAR Filename :-"), wxDefaultPosition, wxDefaultSize, 0 );
	MetaFilenameStaticText->Wrap( -1 );
	bSizer471->Add( MetaFilenameStaticText, 0, wxALIGN_CENTER_VERTICAL|wxALL, 5 );
	
	wxBoxSizer* bSizer502;
	bSizer502 = new wxBoxSizer( wxHORIZONTAL );
	
	MetaDataFileTextCtrl = new wxTextCtrl( GetPathPage, wxID_ANY, wxEmptyString, wxDefaultPosition, wxDefaultSize, 0 );
	bSizer502->Add( MetaDataFileTextCtrl, 1, wxALIGN_CENTER_VERTICAL|wxALL, 5 );
	
	m_button242 = new wxButton( GetPathPage, wxID_ANY, wxT("Browse..."), wxDefaultPosition, wxDefaultSize, 0 );
	bSizer502->Add( m_button242, 0, wxALL, 5 );
	
	
	bSizer471->Add( bSizer502, 0, wxEXPAND, 5 );
	
	
	GetPathPage->SetSizer( bSizer471 );
	GetPathPage->Layout();
	bSizer471->Fit( GetPathPage );
	
	this->Centre( wxBOTH );
	
	for ( unsigned int i = 1; i < m_pages.GetCount(); i++ )
	{
		m_pages.Item( i )->SetPrev( m_pages.Item( i - 1 ) );
		m_pages.Item( i - 1 )->SetNext( m_pages.Item( i ) );
	}
	
	// Connect Events
	this->Connect( wxEVT_UPDATE_UI, wxUpdateUIEventHandler( ExportRefinementPackageWizardParent::OnUpdateUI ) );
	this->Connect( wxID_ANY, wxEVT_WIZARD_FINISHED, wxWizardEventHandler( ExportRefinementPackageWizardParent::OnFinished ) );
	this->Connect( wxID_ANY, wxEVT_WIZARD_PAGE_CHANGED, wxWizardEventHandler( ExportRefinementPackageWizardParent::OnPageChanged ) );
	this->Connect( wxID_ANY, wxEVT_WIZARD_PAGE_CHANGING, wxWizardEventHandler( ExportRefinementPackageWizardParent::OnPageChanging ) );
	ParticleStackFileTextCtrl->Connect( wxEVT_COMMAND_TEXT_UPDATED, wxCommandEventHandler( ExportRefinementPackageWizardParent::OnPathChange ), NULL, this );
	m_button2411->Connect( wxEVT_COMMAND_BUTTON_CLICKED, wxCommandEventHandler( ExportRefinementPackageWizardParent::OnStackBrowseButtonClick ), NULL, this );
	MetaDataFileTextCtrl->Connect( wxEVT_COMMAND_TEXT_UPDATED, wxCommandEventHandler( ExportRefinementPackageWizardParent::OnPathChange ), NULL, this );
	m_button242->Connect( wxEVT_COMMAND_BUTTON_CLICKED, wxCommandEventHandler( ExportRefinementPackageWizardParent::OnMetaBrowseButtonClick ), NULL, this );
}

ExportRefinementPackageWizardParent::~ExportRefinementPackageWizardParent()
{
	// Disconnect Events
	this->Disconnect( wxEVT_UPDATE_UI, wxUpdateUIEventHandler( ExportRefinementPackageWizardParent::OnUpdateUI ) );
	this->Disconnect( wxID_ANY, wxEVT_WIZARD_FINISHED, wxWizardEventHandler( ExportRefinementPackageWizardParent::OnFinished ) );
	this->Disconnect( wxID_ANY, wxEVT_WIZARD_PAGE_CHANGED, wxWizardEventHandler( ExportRefinementPackageWizardParent::OnPageChanged ) );
	this->Disconnect( wxID_ANY, wxEVT_WIZARD_PAGE_CHANGING, wxWizardEventHandler( ExportRefinementPackageWizardParent::OnPageChanging ) );
	ParticleStackFileTextCtrl->Disconnect( wxEVT_COMMAND_TEXT_UPDATED, wxCommandEventHandler( ExportRefinementPackageWizardParent::OnPathChange ), NULL, this );
	m_button2411->Disconnect( wxEVT_COMMAND_BUTTON_CLICKED, wxCommandEventHandler( ExportRefinementPackageWizardParent::OnStackBrowseButtonClick ), NULL, this );
	MetaDataFileTextCtrl->Disconnect( wxEVT_COMMAND_TEXT_UPDATED, wxCommandEventHandler( ExportRefinementPackageWizardParent::OnPathChange ), NULL, this );
	m_button242->Disconnect( wxEVT_COMMAND_BUTTON_CLICKED, wxCommandEventHandler( ExportRefinementPackageWizardParent::OnMetaBrowseButtonClick ), NULL, this );
	
	m_pages.Clear();
}

ImportRefinementPackageWizardParent::ImportRefinementPackageWizardParent( wxWindow* parent, wxWindowID id, const wxString& title, const wxBitmap& bitmap, const wxPoint& pos, long style ) 
{
	this->Create( parent, id, title, bitmap, pos, style );
	this->SetSizeHints( wxSize( 700,400 ), wxDefaultSize );
	
	wxWizardPageSimple* ImportTypePage = new wxWizardPageSimple( this );
	m_pages.Add( ImportTypePage );
	
	wxBoxSizer* bSizer3931;
	bSizer3931 = new wxBoxSizer( wxVERTICAL );
	
	m_staticText4651 = new wxStaticText( ImportTypePage, wxID_ANY, wxT("Import from what source? :-"), wxDefaultPosition, wxDefaultSize, 0 );
	m_staticText4651->Wrap( -1 );
	bSizer3931->Add( m_staticText4651, 0, wxALL, 5 );
	
	m_staticline1051 = new wxStaticLine( ImportTypePage, wxID_ANY, wxDefaultPosition, wxDefaultSize, wxLI_HORIZONTAL );
	bSizer3931->Add( m_staticline1051, 0, wxEXPAND | wxALL, 5 );
	
	wxBoxSizer* bSizer3941;
	bSizer3941 = new wxBoxSizer( wxVERTICAL );
	
	FrealignRadioButton = new wxRadioButton( ImportTypePage, wxID_ANY, wxT("Frealign (Requires particle stack and PAR file)"), wxDefaultPosition, wxDefaultSize, 0 );
	bSizer3941->Add( FrealignRadioButton, 0, wxALL, 5 );
	
	RelionRadioButton = new wxRadioButton( ImportTypePage, wxID_ANY, wxT("Relion (Requires particle stack and STAR file)"), wxDefaultPosition, wxDefaultSize, 0 );
	bSizer3941->Add( RelionRadioButton, 0, wxALL, 5 );
	
	
	bSizer3931->Add( bSizer3941, 1, wxEXPAND, 5 );
	
	
	ImportTypePage->SetSizer( bSizer3931 );
	ImportTypePage->Layout();
	bSizer3931->Fit( ImportTypePage );
	wxWizardPageSimple* GetPathPage = new wxWizardPageSimple( this );
	m_pages.Add( GetPathPage );
	
	GetPathPage->SetMinSize( wxSize( 600,300 ) );
	
	wxBoxSizer* bSizer47;
	bSizer47 = new wxBoxSizer( wxVERTICAL );
	
	m_staticText474 = new wxStaticText( GetPathPage, wxID_ANY, wxT("Please provide the stack and metadata files"), wxDefaultPosition, wxDefaultSize, 0 );
	m_staticText474->Wrap( -1 );
	bSizer47->Add( m_staticText474, 0, wxALL, 5 );
	
	m_staticline106 = new wxStaticLine( GetPathPage, wxID_ANY, wxDefaultPosition, wxDefaultSize, wxLI_HORIZONTAL );
	bSizer47->Add( m_staticline106, 0, wxEXPAND | wxALL, 5 );
	
	m_staticText41 = new wxStaticText( GetPathPage, wxID_ANY, wxT("Particle Stack Filename :-"), wxDefaultPosition, wxDefaultSize, 0 );
	m_staticText41->Wrap( -1 );
	bSizer47->Add( m_staticText41, 0, wxALIGN_CENTER_VERTICAL|wxALL, 5 );
	
	wxBoxSizer* bSizer501;
	bSizer501 = new wxBoxSizer( wxHORIZONTAL );
	
	ParticleStackFileTextCtrl = new wxTextCtrl( GetPathPage, wxID_ANY, wxEmptyString, wxDefaultPosition, wxDefaultSize, 0 );
	bSizer501->Add( ParticleStackFileTextCtrl, 1, wxALIGN_CENTER_VERTICAL|wxALL, 5 );
	
	m_button241 = new wxButton( GetPathPage, wxID_ANY, wxT("Browse..."), wxDefaultPosition, wxDefaultSize, 0 );
	bSizer501->Add( m_button241, 0, wxALL, 5 );
	
	
	bSizer47->Add( bSizer501, 0, wxEXPAND, 5 );
	
	MetaFilenameStaticText = new wxStaticText( GetPathPage, wxID_ANY, wxT("PAR / STAR Filename :-"), wxDefaultPosition, wxDefaultSize, 0 );
	MetaFilenameStaticText->Wrap( -1 );
	bSizer47->Add( MetaFilenameStaticText, 0, wxALIGN_CENTER_VERTICAL|wxALL, 5 );
	
	wxBoxSizer* bSizer50;
	bSizer50 = new wxBoxSizer( wxHORIZONTAL );
	
	MetaDataFileTextCtrl = new wxTextCtrl( GetPathPage, wxID_ANY, wxEmptyString, wxDefaultPosition, wxDefaultSize, 0 );
	bSizer50->Add( MetaDataFileTextCtrl, 1, wxALIGN_CENTER_VERTICAL|wxALL, 5 );
	
	m_button24 = new wxButton( GetPathPage, wxID_ANY, wxT("Browse..."), wxDefaultPosition, wxDefaultSize, 0 );
	bSizer50->Add( m_button24, 0, wxALL, 5 );
	
	
	bSizer47->Add( bSizer50, 0, wxEXPAND, 5 );
	
	
	GetPathPage->SetSizer( bSizer47 );
	GetPathPage->Layout();
	bSizer47->Fit( GetPathPage );
	wxWizardPageSimple* GetParametersPage = new wxWizardPageSimple( this );
	m_pages.Add( GetParametersPage );
	
	wxBoxSizer* bSizer402;
	bSizer402 = new wxBoxSizer( wxVERTICAL );
	
	m_staticText476 = new wxStaticText( GetParametersPage, wxID_ANY, wxT("Please provide the following information :-"), wxDefaultPosition, wxDefaultSize, 0 );
	m_staticText476->Wrap( -1 );
	bSizer402->Add( m_staticText476, 0, wxALL, 5 );
	
	m_staticline107 = new wxStaticLine( GetParametersPage, wxID_ANY, wxDefaultPosition, wxDefaultSize, wxLI_HORIZONTAL );
	bSizer402->Add( m_staticline107, 0, wxEXPAND | wxALL, 5 );
	
	wxFlexGridSizer* fgSizer23;
	fgSizer23 = new wxFlexGridSizer( 0, 2, 0, 0 );
	fgSizer23->SetFlexibleDirection( wxBOTH );
	fgSizer23->SetNonFlexibleGrowMode( wxFLEX_GROWMODE_SPECIFIED );
	
	m_staticText477 = new wxStaticText( GetParametersPage, wxID_ANY, wxT("Pixel Size (Å) : "), wxDefaultPosition, wxDefaultSize, 0 );
	m_staticText477->Wrap( -1 );
	fgSizer23->Add( m_staticText477, 0, wxALIGN_CENTER_VERTICAL|wxALIGN_RIGHT|wxALL, 5 );
	
	PixelSizeTextCtrl = new NumericTextCtrl( GetParametersPage, wxID_ANY, wxT("1.00"), wxDefaultPosition, wxDefaultSize, 0 );
	PixelSizeTextCtrl->SetMinSize( wxSize( 100,-1 ) );
	
	fgSizer23->Add( PixelSizeTextCtrl, 0, wxALL, 5 );
	
	m_staticText478 = new wxStaticText( GetParametersPage, wxID_ANY, wxT("Microscope Voltage (kV) : "), wxDefaultPosition, wxDefaultSize, 0 );
	m_staticText478->Wrap( -1 );
	fgSizer23->Add( m_staticText478, 0, wxALIGN_CENTER_VERTICAL|wxALIGN_RIGHT|wxALL, 5 );
	
	MicroscopeVoltageTextCtrl = new NumericTextCtrl( GetParametersPage, wxID_ANY, wxT("300.00"), wxDefaultPosition, wxDefaultSize, 0 );
	MicroscopeVoltageTextCtrl->SetMinSize( wxSize( 100,-1 ) );
	
	fgSizer23->Add( MicroscopeVoltageTextCtrl, 0, wxALL, 5 );
	
	m_staticText479 = new wxStaticText( GetParametersPage, wxID_ANY, wxT("Microscope Cs (mm) : "), wxDefaultPosition, wxDefaultSize, 0 );
	m_staticText479->Wrap( -1 );
	fgSizer23->Add( m_staticText479, 0, wxALIGN_CENTER_VERTICAL|wxALIGN_RIGHT|wxALL, 5 );
	
	SphericalAberrationTextCtrl = new NumericTextCtrl( GetParametersPage, wxID_ANY, wxT("2.70"), wxDefaultPosition, wxDefaultSize, 0 );
	SphericalAberrationTextCtrl->SetMinSize( wxSize( 100,-1 ) );
	
	fgSizer23->Add( SphericalAberrationTextCtrl, 0, wxALL, 5 );
	
	m_staticText480 = new wxStaticText( GetParametersPage, wxID_ANY, wxT("Amplitude Contrast : "), wxDefaultPosition, wxDefaultSize, 0 );
	m_staticText480->Wrap( -1 );
	fgSizer23->Add( m_staticText480, 0, wxALIGN_CENTER_VERTICAL|wxALIGN_RIGHT|wxALL, 5 );
	
	AmplitudeContrastTextCtrl = new NumericTextCtrl( GetParametersPage, wxID_ANY, wxT("0.07"), wxDefaultPosition, wxDefaultSize, 0 );
	AmplitudeContrastTextCtrl->SetMinSize( wxSize( 100,-1 ) );
	
	fgSizer23->Add( AmplitudeContrastTextCtrl, 0, wxALL, 5 );
	
	m_staticText459 = new wxStaticText( GetParametersPage, wxID_ANY, wxT("Pointgroup Symmetry : "), wxDefaultPosition, wxDefaultSize, 0 );
	m_staticText459->Wrap( -1 );
	fgSizer23->Add( m_staticText459, 0, wxALIGN_CENTER_VERTICAL|wxALIGN_RIGHT|wxALL, 5 );
	
	SymmetryComboBox = new wxComboBox( GetParametersPage, wxID_ANY, wxT("C1"), wxDefaultPosition, wxDefaultSize, 0, NULL, 0 ); 
	SymmetryComboBox->SetMinSize( wxSize( 100,-1 ) );
	
	fgSizer23->Add( SymmetryComboBox, 0, wxALL, 5 );
	
	m_staticText460 = new wxStaticText( GetParametersPage, wxID_ANY, wxT("Estimated Molecular Weight (kDa) : "), wxDefaultPosition, wxDefaultSize, 0 );
	m_staticText460->Wrap( -1 );
	fgSizer23->Add( m_staticText460, 0, wxALIGN_CENTER_VERTICAL|wxALIGN_RIGHT|wxALL, 5 );
	
	MolecularWeightTextCtrl = new NumericTextCtrl( GetParametersPage, wxID_ANY, wxT("400.00"), wxDefaultPosition, wxDefaultSize, 0 );
	MolecularWeightTextCtrl->SetMinSize( wxSize( 100,-1 ) );
	
	fgSizer23->Add( MolecularWeightTextCtrl, 0, wxALL, 5 );
	
	m_staticText214 = new wxStaticText( GetParametersPage, wxID_ANY, wxT("Estimated Largest Dimension (Å) : "), wxDefaultPosition, wxDefaultSize, 0 );
	m_staticText214->Wrap( -1 );
	fgSizer23->Add( m_staticText214, 0, wxALIGN_CENTER_VERTICAL|wxALIGN_RIGHT|wxALL, 5 );
	
	LargestDimensionTextCtrl = new NumericTextCtrl( GetParametersPage, wxID_ANY, wxT("150.00"), wxDefaultPosition, wxDefaultSize, 0 );
	LargestDimensionTextCtrl->SetMinSize( wxSize( 100,-1 ) );
	
	fgSizer23->Add( LargestDimensionTextCtrl, 0, wxALL, 5 );
	
	m_staticText462 = new wxStaticText( GetParametersPage, wxID_ANY, wxT("Protein Density in Stack is : "), wxDefaultPosition, wxDefaultSize, 0 );
	m_staticText462->Wrap( -1 );
	fgSizer23->Add( m_staticText462, 0, wxALIGN_CENTER_VERTICAL|wxALIGN_RIGHT|wxALL, 5 );
	
	wxBoxSizer* bSizer379;
	bSizer379 = new wxBoxSizer( wxHORIZONTAL );
	
	BlackProteinRadioButton = new wxRadioButton( GetParametersPage, wxID_ANY, wxT("Black (Frealign Default)"), wxDefaultPosition, wxDefaultSize, 0 );
	bSizer379->Add( BlackProteinRadioButton, 0, wxALL, 5 );
	
	WhiteProteinRadioButton = new wxRadioButton( GetParametersPage, wxID_ANY, wxT("White (Relion Default)"), wxDefaultPosition, wxDefaultSize, 0 );
	bSizer379->Add( WhiteProteinRadioButton, 0, wxALL, 5 );
	
	
	fgSizer23->Add( bSizer379, 1, wxEXPAND, 5 );
	
	
	bSizer402->Add( fgSizer23, 1, wxEXPAND, 5 );
	
	
	GetParametersPage->SetSizer( bSizer402 );
	GetParametersPage->Layout();
	bSizer402->Fit( GetParametersPage );
	
	this->Centre( wxBOTH );
	
	for ( unsigned int i = 1; i < m_pages.GetCount(); i++ )
	{
		m_pages.Item( i )->SetPrev( m_pages.Item( i - 1 ) );
		m_pages.Item( i - 1 )->SetNext( m_pages.Item( i ) );
	}
	
	// Connect Events
	this->Connect( wxEVT_UPDATE_UI, wxUpdateUIEventHandler( ImportRefinementPackageWizardParent::OnUpdateUI ) );
	this->Connect( wxID_ANY, wxEVT_WIZARD_FINISHED, wxWizardEventHandler( ImportRefinementPackageWizardParent::OnFinished ) );
	this->Connect( wxID_ANY, wxEVT_WIZARD_PAGE_CHANGED, wxWizardEventHandler( ImportRefinementPackageWizardParent::OnPageChanged ) );
	this->Connect( wxID_ANY, wxEVT_WIZARD_PAGE_CHANGING, wxWizardEventHandler( ImportRefinementPackageWizardParent::OnPageChanging ) );
	ParticleStackFileTextCtrl->Connect( wxEVT_COMMAND_TEXT_UPDATED, wxCommandEventHandler( ImportRefinementPackageWizardParent::OnPathChange ), NULL, this );
	m_button241->Connect( wxEVT_COMMAND_BUTTON_CLICKED, wxCommandEventHandler( ImportRefinementPackageWizardParent::OnStackBrowseButtonClick ), NULL, this );
	MetaDataFileTextCtrl->Connect( wxEVT_COMMAND_TEXT_UPDATED, wxCommandEventHandler( ImportRefinementPackageWizardParent::OnPathChange ), NULL, this );
	m_button24->Connect( wxEVT_COMMAND_BUTTON_CLICKED, wxCommandEventHandler( ImportRefinementPackageWizardParent::OnMetaBrowseButtonClick ), NULL, this );
}

ImportRefinementPackageWizardParent::~ImportRefinementPackageWizardParent()
{
	// Disconnect Events
	this->Disconnect( wxEVT_UPDATE_UI, wxUpdateUIEventHandler( ImportRefinementPackageWizardParent::OnUpdateUI ) );
	this->Disconnect( wxID_ANY, wxEVT_WIZARD_FINISHED, wxWizardEventHandler( ImportRefinementPackageWizardParent::OnFinished ) );
	this->Disconnect( wxID_ANY, wxEVT_WIZARD_PAGE_CHANGED, wxWizardEventHandler( ImportRefinementPackageWizardParent::OnPageChanged ) );
	this->Disconnect( wxID_ANY, wxEVT_WIZARD_PAGE_CHANGING, wxWizardEventHandler( ImportRefinementPackageWizardParent::OnPageChanging ) );
	ParticleStackFileTextCtrl->Disconnect( wxEVT_COMMAND_TEXT_UPDATED, wxCommandEventHandler( ImportRefinementPackageWizardParent::OnPathChange ), NULL, this );
	m_button241->Disconnect( wxEVT_COMMAND_BUTTON_CLICKED, wxCommandEventHandler( ImportRefinementPackageWizardParent::OnStackBrowseButtonClick ), NULL, this );
	MetaDataFileTextCtrl->Disconnect( wxEVT_COMMAND_TEXT_UPDATED, wxCommandEventHandler( ImportRefinementPackageWizardParent::OnPathChange ), NULL, this );
	m_button24->Disconnect( wxEVT_COMMAND_BUTTON_CLICKED, wxCommandEventHandler( ImportRefinementPackageWizardParent::OnMetaBrowseButtonClick ), NULL, this );
	
	m_pages.Clear();
}

AddRunCommandDialog::AddRunCommandDialog( wxWindow* parent, wxWindowID id, const wxString& title, const wxPoint& pos, const wxSize& size, long style ) : wxDialog( parent, id, title, pos, size, style )
{
	this->SetSizeHints( wxDefaultSize, wxDefaultSize );
	
	wxBoxSizer* bSizer45;
	bSizer45 = new wxBoxSizer( wxVERTICAL );
	
	wxFlexGridSizer* fgSizer2;
	fgSizer2 = new wxFlexGridSizer( 3, 2, 0, 0 );
	fgSizer2->AddGrowableCol( 1 );
	fgSizer2->SetFlexibleDirection( wxBOTH );
	fgSizer2->SetNonFlexibleGrowMode( wxFLEX_GROWMODE_SPECIFIED );
	
	m_staticText45 = new wxStaticText( this, wxID_ANY, wxT("Command :"), wxDefaultPosition, wxDefaultSize, 0 );
	m_staticText45->Wrap( -1 );
	fgSizer2->Add( m_staticText45, 0, wxALIGN_CENTER_VERTICAL|wxALIGN_RIGHT|wxALL, 5 );
	
	CommandTextCtrl = new wxTextCtrl( this, wxID_ANY, wxT("$command"), wxDefaultPosition, wxDefaultSize, wxTE_PROCESS_ENTER );
	CommandTextCtrl->SetMinSize( wxSize( 300,-1 ) );
	
	fgSizer2->Add( CommandTextCtrl, 1, wxALL|wxEXPAND, 5 );
	
	m_staticText46 = new wxStaticText( this, wxID_ANY, wxT("No. Copies :"), wxDefaultPosition, wxDefaultSize, 0 );
	m_staticText46->Wrap( -1 );
	fgSizer2->Add( m_staticText46, 0, wxALIGN_CENTER_VERTICAL|wxALIGN_RIGHT|wxALL, 5 );
	
	NumberCopiesSpinCtrl = new wxSpinCtrl( this, wxID_ANY, wxEmptyString, wxDefaultPosition, wxDefaultSize, wxSP_ARROW_KEYS, 1, 999999, 1 );
	fgSizer2->Add( NumberCopiesSpinCtrl, 1, wxALL|wxEXPAND, 5 );
	
	m_staticText58 = new wxStaticText( this, wxID_ANY, wxT("Delay (ms) :"), wxDefaultPosition, wxDefaultSize, 0 );
	m_staticText58->Wrap( -1 );
	fgSizer2->Add( m_staticText58, 0, wxALIGN_CENTER_VERTICAL|wxALL, 5 );
	
	DelayTimeSpinCtrl = new wxSpinCtrl( this, wxID_ANY, wxEmptyString, wxDefaultPosition, wxDefaultSize, wxSP_ARROW_KEYS, 0, 10000, 100 );
	fgSizer2->Add( DelayTimeSpinCtrl, 0, wxALL|wxEXPAND, 5 );
	
	
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

RenameDialog::RenameDialog( wxWindow* parent, wxWindowID id, const wxString& title, const wxPoint& pos, const wxSize& size, long style ) : wxDialog( parent, id, title, pos, size, style )
{
	this->SetSizeHints( wxSize( -1,-1 ), wxDefaultSize );
	
	MainBoxSizer = new wxBoxSizer( wxVERTICAL );
	
	m_staticText246 = new wxStaticText( this, wxID_ANY, wxT("Set Asset Names :-"), wxDefaultPosition, wxDefaultSize, 0 );
	m_staticText246->Wrap( -1 );
	MainBoxSizer->Add( m_staticText246, 0, wxALL, 5 );
	
	m_staticline18 = new wxStaticLine( this, wxID_ANY, wxDefaultPosition, wxDefaultSize, wxLI_HORIZONTAL );
	MainBoxSizer->Add( m_staticline18, 0, wxEXPAND | wxALL, 5 );
	
	RenameScrollPanel = new wxScrolledWindow( this, wxID_ANY, wxDefaultPosition, wxDefaultSize, wxVSCROLL );
	RenameScrollPanel->SetScrollRate( 5, 5 );
	RenameBoxSizer = new wxBoxSizer( wxVERTICAL );
	
	
	RenameScrollPanel->SetSizer( RenameBoxSizer );
	RenameScrollPanel->Layout();
	RenameBoxSizer->Fit( RenameScrollPanel );
	MainBoxSizer->Add( RenameScrollPanel, 1, wxEXPAND, 5 );
	
	m_staticline19 = new wxStaticLine( this, wxID_ANY, wxDefaultPosition, wxDefaultSize, wxLI_HORIZONTAL );
	MainBoxSizer->Add( m_staticline19, 0, wxEXPAND | wxALL, 5 );
	
	wxBoxSizer* bSizer90;
	bSizer90 = new wxBoxSizer( wxHORIZONTAL );
	
	
	bSizer90->Add( 0, 0, 1, wxEXPAND, 5 );
	
	CancelButton = new wxButton( this, wxID_ANY, wxT("Cancel"), wxDefaultPosition, wxDefaultSize, 0 );
	bSizer90->Add( CancelButton, 0, wxALL, 5 );
	
	RenameButton = new wxButton( this, wxID_ANY, wxT("Set Name"), wxDefaultPosition, wxDefaultSize, 0 );
	bSizer90->Add( RenameButton, 0, wxALL, 5 );
	
	
	bSizer90->Add( 0, 0, 1, wxEXPAND, 5 );
	
	
	MainBoxSizer->Add( bSizer90, 0, wxEXPAND, 5 );
	
	
	this->SetSizer( MainBoxSizer );
	this->Layout();
	MainBoxSizer->Fit( this );
	
	this->Centre( wxBOTH );
	
	// Connect Events
	CancelButton->Connect( wxEVT_COMMAND_BUTTON_CLICKED, wxCommandEventHandler( RenameDialog::OnCancelClick ), NULL, this );
	RenameButton->Connect( wxEVT_COMMAND_BUTTON_CLICKED, wxCommandEventHandler( RenameDialog::OnRenameClick ), NULL, this );
}

RenameDialog::~RenameDialog()
{
	// Disconnect Events
	CancelButton->Disconnect( wxEVT_COMMAND_BUTTON_CLICKED, wxCommandEventHandler( RenameDialog::OnCancelClick ), NULL, this );
	RenameButton->Disconnect( wxEVT_COMMAND_BUTTON_CLICKED, wxCommandEventHandler( RenameDialog::OnRenameClick ), NULL, this );
	
}

NewRefinementPackageWizard::NewRefinementPackageWizard( wxWindow* parent, wxWindowID id, const wxString& title, const wxBitmap& bitmap, const wxPoint& pos, long style ) 
{
	this->Create( parent, id, title, bitmap, pos, style );
	this->SetSizeHints( wxSize( -1,-1 ), wxDefaultSize );
	
	
	this->Centre( wxBOTH );
	
	
	// Connect Events
	this->Connect( wxID_ANY, wxEVT_WIZARD_FINISHED, wxWizardEventHandler( NewRefinementPackageWizard::OnFinished ) );
	this->Connect( wxID_ANY, wxEVT_WIZARD_PAGE_CHANGED, wxWizardEventHandler( NewRefinementPackageWizard::PageChanged ) );
	this->Connect( wxID_ANY, wxEVT_WIZARD_PAGE_CHANGING, wxWizardEventHandler( NewRefinementPackageWizard::PageChanging ) );
}

NewRefinementPackageWizard::~NewRefinementPackageWizard()
{
	// Disconnect Events
	this->Disconnect( wxID_ANY, wxEVT_WIZARD_FINISHED, wxWizardEventHandler( NewRefinementPackageWizard::OnFinished ) );
	this->Disconnect( wxID_ANY, wxEVT_WIZARD_PAGE_CHANGED, wxWizardEventHandler( NewRefinementPackageWizard::PageChanged ) );
	this->Disconnect( wxID_ANY, wxEVT_WIZARD_PAGE_CHANGING, wxWizardEventHandler( NewRefinementPackageWizard::PageChanging ) );
	
	m_pages.Clear();
}

VolumeChooserDialog::VolumeChooserDialog( wxWindow* parent, wxWindowID id, const wxString& title, const wxPoint& pos, const wxSize& size, long style ) : wxDialog( parent, id, title, pos, size, style )
{
	this->SetSizeHints( wxSize( -1,-1 ), wxDefaultSize );
	
	MainBoxSizer = new wxBoxSizer( wxVERTICAL );
	
	m_staticText246 = new wxStaticText( this, wxID_ANY, wxT("Select New Volume :-"), wxDefaultPosition, wxDefaultSize, 0 );
	m_staticText246->Wrap( -1 );
	MainBoxSizer->Add( m_staticText246, 0, wxALL, 5 );
	
	m_staticline18 = new wxStaticLine( this, wxID_ANY, wxDefaultPosition, wxDefaultSize, wxLI_HORIZONTAL );
	MainBoxSizer->Add( m_staticline18, 0, wxEXPAND | wxALL, 5 );
	
	ComboBox = new VolumeAssetPickerComboPanel( this, wxID_ANY, wxDefaultPosition, wxDefaultSize, wxTAB_TRAVERSAL );
	ComboBox->SetMinSize( wxSize( 350,-1 ) );
	
	MainBoxSizer->Add( ComboBox, 1, wxEXPAND | wxALL, 5 );
	
	m_staticline19 = new wxStaticLine( this, wxID_ANY, wxDefaultPosition, wxDefaultSize, wxLI_HORIZONTAL );
	MainBoxSizer->Add( m_staticline19, 0, wxEXPAND | wxALL, 5 );
	
	wxBoxSizer* bSizer90;
	bSizer90 = new wxBoxSizer( wxHORIZONTAL );
	
	
	bSizer90->Add( 0, 0, 1, wxEXPAND, 5 );
	
	CancelButton = new wxButton( this, wxID_ANY, wxT("Cancel"), wxDefaultPosition, wxDefaultSize, 0 );
	bSizer90->Add( CancelButton, 0, wxALL, 5 );
	
	SetButton = new wxButton( this, wxID_ANY, wxT("Set Reference"), wxDefaultPosition, wxDefaultSize, 0 );
	bSizer90->Add( SetButton, 0, wxALL, 5 );
	
	
	bSizer90->Add( 0, 0, 1, wxEXPAND, 5 );
	
	
	MainBoxSizer->Add( bSizer90, 0, wxEXPAND, 5 );
	
	
	this->SetSizer( MainBoxSizer );
	this->Layout();
	MainBoxSizer->Fit( this );
	
	this->Centre( wxBOTH );
	
	// Connect Events
	CancelButton->Connect( wxEVT_COMMAND_BUTTON_CLICKED, wxCommandEventHandler( VolumeChooserDialog::OnCancelClick ), NULL, this );
	SetButton->Connect( wxEVT_COMMAND_BUTTON_CLICKED, wxCommandEventHandler( VolumeChooserDialog::OnRenameClick ), NULL, this );
}

VolumeChooserDialog::~VolumeChooserDialog()
{
	// Disconnect Events
	CancelButton->Disconnect( wxEVT_COMMAND_BUTTON_CLICKED, wxCommandEventHandler( VolumeChooserDialog::OnCancelClick ), NULL, this );
	SetButton->Disconnect( wxEVT_COMMAND_BUTTON_CLICKED, wxCommandEventHandler( VolumeChooserDialog::OnRenameClick ), NULL, this );
	
}

FilterDialog::FilterDialog( wxWindow* parent, wxWindowID id, const wxString& title, const wxPoint& pos, const wxSize& size, long style ) : wxDialog( parent, id, title, pos, size, style )
{
	this->SetSizeHints( wxSize( -1,-1 ), wxDefaultSize );
	
	MainBoxSizer = new wxBoxSizer( wxVERTICAL );
	
	m_staticText64 = new wxStaticText( this, wxID_ANY, wxT("Filter By :-"), wxDefaultPosition, wxDefaultSize, 0 );
	m_staticText64->Wrap( -1 );
	m_staticText64->SetFont( wxFont( wxNORMAL_FONT->GetPointSize(), 70, 90, 92, false, wxEmptyString ) );
	
	MainBoxSizer->Add( m_staticText64, 0, wxALL, 5 );
	
	m_staticline18 = new wxStaticLine( this, wxID_ANY, wxDefaultPosition, wxDefaultSize, wxLI_HORIZONTAL );
	MainBoxSizer->Add( m_staticline18, 0, wxEXPAND | wxALL, 5 );
	
	FilterScrollPanel = new wxScrolledWindow( this, wxID_ANY, wxDefaultPosition, wxDefaultSize, wxSUNKEN_BORDER|wxVSCROLL );
	FilterScrollPanel->SetScrollRate( 5, 5 );
	FilterBoxSizer = new wxBoxSizer( wxVERTICAL );
	
	
	FilterScrollPanel->SetSizer( FilterBoxSizer );
	FilterScrollPanel->Layout();
	FilterBoxSizer->Fit( FilterScrollPanel );
	MainBoxSizer->Add( FilterScrollPanel, 1, wxALL|wxEXPAND, 5 );
	
	m_staticText81 = new wxStaticText( this, wxID_ANY, wxT("\nSort By :-"), wxDefaultPosition, wxDefaultSize, 0 );
	m_staticText81->Wrap( -1 );
	m_staticText81->SetFont( wxFont( wxNORMAL_FONT->GetPointSize(), 70, 90, 92, false, wxEmptyString ) );
	
	MainBoxSizer->Add( m_staticText81, 0, wxALL, 5 );
	
	m_staticline19 = new wxStaticLine( this, wxID_ANY, wxDefaultPosition, wxDefaultSize, wxLI_HORIZONTAL );
	MainBoxSizer->Add( m_staticline19, 0, wxEXPAND | wxALL, 5 );
	
	SortScrollPanel = new wxScrolledWindow( this, wxID_ANY, wxDefaultPosition, wxDefaultSize, wxVSCROLL );
	SortScrollPanel->SetScrollRate( 5, 5 );
	SortSizer = new wxGridSizer( 0, 3, 0, 0 );
	
	
	SortScrollPanel->SetSizer( SortSizer );
	SortScrollPanel->Layout();
	SortSizer->Fit( SortScrollPanel );
	MainBoxSizer->Add( SortScrollPanel, 0, wxEXPAND | wxALL, 5 );
	
	m_staticline21 = new wxStaticLine( this, wxID_ANY, wxDefaultPosition, wxDefaultSize, wxLI_HORIZONTAL );
	MainBoxSizer->Add( m_staticline21, 0, wxEXPAND | wxALL, 5 );
	
	wxBoxSizer* bSizer90;
	bSizer90 = new wxBoxSizer( wxHORIZONTAL );
	
	
	bSizer90->Add( 0, 0, 1, wxEXPAND, 5 );
	
	CancelButton = new wxButton( this, wxID_ANY, wxT("Cancel"), wxDefaultPosition, wxDefaultSize, 0 );
	bSizer90->Add( CancelButton, 0, wxALL, 5 );
	
	FilterButton = new wxButton( this, wxID_ANY, wxT("Filter/Sort"), wxDefaultPosition, wxDefaultSize, 0 );
	bSizer90->Add( FilterButton, 0, wxALL, 5 );
	
	
	bSizer90->Add( 0, 0, 1, wxEXPAND, 5 );
	
	
	MainBoxSizer->Add( bSizer90, 0, wxEXPAND, 5 );
	
	
	this->SetSizer( MainBoxSizer );
	this->Layout();
	MainBoxSizer->Fit( this );
	
	this->Centre( wxBOTH );
	
	// Connect Events
	CancelButton->Connect( wxEVT_COMMAND_BUTTON_CLICKED, wxCommandEventHandler( FilterDialog::OnCancelClick ), NULL, this );
	FilterButton->Connect( wxEVT_COMMAND_BUTTON_CLICKED, wxCommandEventHandler( FilterDialog::OnFilterClick ), NULL, this );
}

FilterDialog::~FilterDialog()
{
	// Disconnect Events
	CancelButton->Disconnect( wxEVT_COMMAND_BUTTON_CLICKED, wxCommandEventHandler( FilterDialog::OnCancelClick ), NULL, this );
	FilterButton->Disconnect( wxEVT_COMMAND_BUTTON_CLICKED, wxCommandEventHandler( FilterDialog::OnFilterClick ), NULL, this );
	
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
	
	GroupComboBox = new wxComboBox( m_panel38, wxID_ANY, wxEmptyString, wxDefaultPosition, wxDefaultSize, 0, NULL, wxCB_READONLY ); 
	sbSizer3->Add( GroupComboBox, 0, wxALL|wxEXPAND, 5 );
	
	
	bSizer135->Add( sbSizer3, 0, wxEXPAND, 25 );
	
	wxStaticBoxSizer* sbSizer4;
	sbSizer4 = new wxStaticBoxSizer( new wxStaticBox( m_panel38, wxID_ANY, wxT("Destination directory") ), wxVERTICAL );
	
	DestinationDirectoryPickerCtrl = new wxDirPickerCtrl( m_panel38, wxID_ANY, wxEmptyString, wxT("Select a folder"), wxDefaultPosition, wxDefaultSize, wxDIRP_DEFAULT_STYLE );
	sbSizer4->Add( DestinationDirectoryPickerCtrl, 0, wxALL|wxEXPAND, 5 );
	
	
	bSizer135->Add( sbSizer4, 0, wxEXPAND, 5 );
	
	
	bSizer135->Add( 0, 0, 0, wxEXPAND, 5 );
	
	WarningText = new wxStaticText( m_panel38, wxID_ANY, wxT("Warning: running jobs \nmay affect exported coordinates"), wxDefaultPosition, wxDefaultSize, wxALIGN_CENTRE );
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

FrealignExportDialog::FrealignExportDialog( wxWindow* parent, wxWindowID id, const wxString& title, const wxPoint& pos, const wxSize& size, long style ) : wxDialog( parent, id, title, pos, size, style )
{
	this->SetSizeHints( wxDefaultSize, wxDefaultSize );
	
	wxBoxSizer* bSizer133;
	bSizer133 = new wxBoxSizer( wxVERTICAL );
	
	m_panel38 = new wxPanel( this, wxID_ANY, wxDefaultPosition, wxDefaultSize, wxTAB_TRAVERSAL );
	wxBoxSizer* bSizer135;
	bSizer135 = new wxBoxSizer( wxVERTICAL );
	
	wxStaticBoxSizer* sbSizer3;
	sbSizer3 = new wxStaticBoxSizer( new wxStaticBox( m_panel38, wxID_ANY, wxT("Export from these images") ), wxVERTICAL );
	
	GroupComboBox = new wxComboBox( m_panel38, wxID_ANY, wxEmptyString, wxDefaultPosition, wxDefaultSize, 0, NULL, wxCB_READONLY ); 
	sbSizer3->Add( GroupComboBox, 0, wxALL|wxEXPAND, 5 );
	
	
	bSizer135->Add( sbSizer3, 0, wxEXPAND, 25 );
	
	wxBoxSizer* bSizer141;
	bSizer141 = new wxBoxSizer( wxVERTICAL );
	
	wxBoxSizer* bSizer142;
	bSizer142 = new wxBoxSizer( wxHORIZONTAL );
	
	m_staticText202 = new wxStaticText( m_panel38, wxID_ANY, wxT("Downsampling factor : "), wxDefaultPosition, wxDefaultSize, 0 );
	m_staticText202->Wrap( -1 );
	bSizer142->Add( m_staticText202, 0, wxALIGN_CENTER_VERTICAL|wxALL, 5 );
	
	DownsamplingFactorSpinCtrl = new wxSpinCtrl( m_panel38, wxID_ANY, wxEmptyString, wxDefaultPosition, wxDefaultSize, wxSP_ARROW_KEYS, 1, 10, 1 );
	bSizer142->Add( DownsamplingFactorSpinCtrl, 0, wxALL, 5 );
	
	
	bSizer141->Add( bSizer142, 0, wxEXPAND, 5 );
	
	
	bSizer135->Add( bSizer141, 1, wxEXPAND, 5 );
	
	wxBoxSizer* bSizer145;
	bSizer145 = new wxBoxSizer( wxHORIZONTAL );
	
	wxBoxSizer* bSizer143;
	bSizer143 = new wxBoxSizer( wxHORIZONTAL );
	
	m_staticText203 = new wxStaticText( m_panel38, wxID_ANY, wxT("Box size after downsampling : "), wxDefaultPosition, wxDefaultSize, 0 );
	m_staticText203->Wrap( -1 );
	bSizer143->Add( m_staticText203, 0, wxALIGN_CENTER_VERTICAL|wxALL, 5 );
	
	BoxSizeSpinCtrl = new wxSpinCtrl( m_panel38, wxID_ANY, wxEmptyString, wxDefaultPosition, wxDefaultSize, wxSP_ARROW_KEYS, 1, 999999999, 512 );
	bSizer143->Add( BoxSizeSpinCtrl, 0, wxALL, 5 );
	
	
	bSizer145->Add( bSizer143, 1, wxEXPAND, 5 );
	
	
	bSizer135->Add( bSizer145, 1, wxEXPAND, 5 );
	
	wxBoxSizer* bSizer140;
	bSizer140 = new wxBoxSizer( wxVERTICAL );
	
	NormalizeCheckBox = new wxCheckBox( m_panel38, wxID_ANY, wxT("Normalize images"), wxDefaultPosition, wxDefaultSize, 0 );
	bSizer140->Add( NormalizeCheckBox, 0, wxALL, 5 );
	
	
	bSizer135->Add( bSizer140, 1, wxEXPAND, 5 );
	
	wxBoxSizer* bSizer213;
	bSizer213 = new wxBoxSizer( wxVERTICAL );
	
	FlipCTFCheckBox = new wxCheckBox( m_panel38, wxID_ANY, wxT("Flip CTF"), wxDefaultPosition, wxDefaultSize, 0 );
	bSizer213->Add( FlipCTFCheckBox, 0, wxALL, 5 );
	
	
	bSizer135->Add( bSizer213, 1, wxEXPAND, 5 );
	
	wxStaticBoxSizer* sbSizer4;
	sbSizer4 = new wxStaticBoxSizer( new wxStaticBox( m_panel38, wxID_ANY, wxT("Output image stack") ), wxVERTICAL );
	
	OutputImageStackPicker = new wxFilePickerCtrl( m_panel38, wxID_ANY, wxEmptyString, wxT("Select a file"), wxT("MRC files (*.mrc, *.mrcs)|*.mrc;*.mrcs"), wxDefaultPosition, wxDefaultSize, wxFLP_OVERWRITE_PROMPT|wxFLP_SAVE|wxFLP_USE_TEXTCTRL );
	sbSizer4->Add( OutputImageStackPicker, 0, wxALL|wxEXPAND, 5 );
	
	
	bSizer135->Add( sbSizer4, 0, wxEXPAND, 5 );
	
	
	bSizer135->Add( 0, 0, 0, wxEXPAND, 5 );
	
	WarningText = new wxStaticText( m_panel38, wxID_ANY, wxT("Warning: running jobs \nmay affect exported coordinates"), wxDefaultPosition, wxDefaultSize, wxALIGN_CENTRE );
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
	FlipCTFCheckBox->Connect( wxEVT_COMMAND_CHECKBOX_CLICKED, wxCommandEventHandler( FrealignExportDialog::OnFlipCTFCheckBox ), NULL, this );
	OutputImageStackPicker->Connect( wxEVT_COMMAND_FILEPICKER_CHANGED, wxFileDirPickerEventHandler( FrealignExportDialog::OnOutputImageStackFileChanged ), NULL, this );
	CancelButton->Connect( wxEVT_COMMAND_BUTTON_CLICKED, wxCommandEventHandler( FrealignExportDialog::OnCancelButtonClick ), NULL, this );
	ExportButton->Connect( wxEVT_COMMAND_BUTTON_CLICKED, wxCommandEventHandler( FrealignExportDialog::OnExportButtonClick ), NULL, this );
}

FrealignExportDialog::~FrealignExportDialog()
{
	// Disconnect Events
	FlipCTFCheckBox->Disconnect( wxEVT_COMMAND_CHECKBOX_CLICKED, wxCommandEventHandler( FrealignExportDialog::OnFlipCTFCheckBox ), NULL, this );
	OutputImageStackPicker->Disconnect( wxEVT_COMMAND_FILEPICKER_CHANGED, wxFileDirPickerEventHandler( FrealignExportDialog::OnOutputImageStackFileChanged ), NULL, this );
	CancelButton->Disconnect( wxEVT_COMMAND_BUTTON_CLICKED, wxCommandEventHandler( FrealignExportDialog::OnCancelButtonClick ), NULL, this );
	ExportButton->Disconnect( wxEVT_COMMAND_BUTTON_CLICKED, wxCommandEventHandler( FrealignExportDialog::OnExportButtonClick ), NULL, this );
	
}

RelionExportDialog::RelionExportDialog( wxWindow* parent, wxWindowID id, const wxString& title, const wxPoint& pos, const wxSize& size, long style ) : wxDialog( parent, id, title, pos, size, style )
{
	this->SetSizeHints( wxDefaultSize, wxDefaultSize );
	
	wxBoxSizer* bSizer133;
	bSizer133 = new wxBoxSizer( wxVERTICAL );
	
	m_panel38 = new wxPanel( this, wxID_ANY, wxDefaultPosition, wxDefaultSize, wxTAB_TRAVERSAL );
	wxBoxSizer* bSizer135;
	bSizer135 = new wxBoxSizer( wxVERTICAL );
	
	wxStaticBoxSizer* sbSizer3;
	sbSizer3 = new wxStaticBoxSizer( new wxStaticBox( m_panel38, wxID_ANY, wxT("Export from these images") ), wxVERTICAL );
	
	GroupComboBox = new wxComboBox( m_panel38, wxID_ANY, wxEmptyString, wxDefaultPosition, wxDefaultSize, 0, NULL, wxCB_READONLY ); 
	sbSizer3->Add( GroupComboBox, 0, wxALL|wxEXPAND, 5 );
	
	
	bSizer135->Add( sbSizer3, 0, wxEXPAND, 25 );
	
	wxBoxSizer* bSizer141;
	bSizer141 = new wxBoxSizer( wxVERTICAL );
	
	wxBoxSizer* bSizer142;
	bSizer142 = new wxBoxSizer( wxHORIZONTAL );
	
	m_staticText202 = new wxStaticText( m_panel38, wxID_ANY, wxT("Downsampling factor : "), wxDefaultPosition, wxDefaultSize, 0 );
	m_staticText202->Wrap( -1 );
	bSizer142->Add( m_staticText202, 0, wxALIGN_CENTER_VERTICAL|wxALL, 5 );
	
	DownsamplingFactorSpinCtrl = new wxSpinCtrl( m_panel38, wxID_ANY, wxEmptyString, wxDefaultPosition, wxDefaultSize, wxSP_ARROW_KEYS, 1, 10, 1 );
	bSizer142->Add( DownsamplingFactorSpinCtrl, 0, wxALL|wxEXPAND, 5 );
	
	
	bSizer141->Add( bSizer142, 0, wxEXPAND, 5 );
	
	
	bSizer135->Add( bSizer141, 1, wxEXPAND, 5 );
	
	wxBoxSizer* bSizer145;
	bSizer145 = new wxBoxSizer( wxHORIZONTAL );
	
	wxBoxSizer* bSizer143;
	bSizer143 = new wxBoxSizer( wxHORIZONTAL );
	
	m_staticText203 = new wxStaticText( m_panel38, wxID_ANY, wxT("Box size after downsampling : "), wxDefaultPosition, wxDefaultSize, 0 );
	m_staticText203->Wrap( -1 );
	bSizer143->Add( m_staticText203, 0, wxALIGN_CENTER_VERTICAL|wxALL, 5 );
	
	BoxSizeSpinCtrl = new wxSpinCtrl( m_panel38, wxID_ANY, wxEmptyString, wxDefaultPosition, wxDefaultSize, wxSP_ARROW_KEYS, 1, 999999999, 512 );
	bSizer143->Add( BoxSizeSpinCtrl, 0, wxALL|wxEXPAND, 5 );
	
	
	bSizer145->Add( bSizer143, 1, wxEXPAND, 5 );
	
	
	bSizer135->Add( bSizer145, 1, wxEXPAND, 5 );
	
	wxBoxSizer* bSizer140;
	bSizer140 = new wxBoxSizer( wxVERTICAL );
	
	NormalizeCheckBox = new wxCheckBox( m_panel38, wxID_ANY, wxT("Normalize images"), wxDefaultPosition, wxDefaultSize, 0 );
	bSizer140->Add( NormalizeCheckBox, 0, wxALL, 5 );
	
	
	bSizer135->Add( bSizer140, 1, wxEXPAND, 5 );
	
	wxBoxSizer* bSizer1451;
	bSizer1451 = new wxBoxSizer( wxHORIZONTAL );
	
	wxBoxSizer* bSizer1431;
	bSizer1431 = new wxBoxSizer( wxHORIZONTAL );
	
	particleRadiusStaticText = new wxStaticText( m_panel38, wxID_ANY, wxT("Particle radius (A) : "), wxDefaultPosition, wxDefaultSize, 0 );
	particleRadiusStaticText->Wrap( -1 );
	particleRadiusStaticText->Enable( false );
	
	bSizer1431->Add( particleRadiusStaticText, 0, wxALIGN_CENTER_VERTICAL|wxALL, 5 );
	
	particleRadiusTextCtrl = new wxTextCtrl( m_panel38, wxID_ANY, wxEmptyString, wxDefaultPosition, wxDefaultSize, 0 );
	particleRadiusTextCtrl->Enable( false );
	
	bSizer1431->Add( particleRadiusTextCtrl, 0, wxALL|wxEXPAND, 5 );
	
	
	bSizer1451->Add( bSizer1431, 1, wxEXPAND, 5 );
	
	
	bSizer135->Add( bSizer1451, 1, wxEXPAND, 5 );
	
	wxBoxSizer* bSizer213;
	bSizer213 = new wxBoxSizer( wxVERTICAL );
	
	FlipCTFCheckBox = new wxCheckBox( m_panel38, wxID_ANY, wxT("Flip CTF"), wxDefaultPosition, wxDefaultSize, 0 );
	bSizer213->Add( FlipCTFCheckBox, 0, wxALL, 5 );
	
	
	bSizer135->Add( bSizer213, 1, wxEXPAND, 5 );
	
	wxStaticBoxSizer* sbSizer4;
	sbSizer4 = new wxStaticBoxSizer( new wxStaticBox( m_panel38, wxID_ANY, wxT("Output image stack") ), wxVERTICAL );
	
	wxBoxSizer* bSizer229;
	bSizer229 = new wxBoxSizer( wxHORIZONTAL );
	
	OutputImageStackPicker = new wxFilePickerCtrl( m_panel38, wxID_ANY, wxEmptyString, wxT("Select a file"), wxT("MRC files (*.mrcs)|*.mrcs"), wxDefaultPosition, wxDefaultSize, wxFLP_OVERWRITE_PROMPT|wxFLP_SAVE );
	bSizer229->Add( OutputImageStackPicker, 0, wxALIGN_CENTER_VERTICAL|wxALL, 5 );
	
	FileNameStaticText = new wxStaticText( m_panel38, wxID_ANY, wxEmptyString, wxDefaultPosition, wxDefaultSize, 0 );
	FileNameStaticText->Wrap( -1 );
	bSizer229->Add( FileNameStaticText, 0, wxALIGN_CENTER_VERTICAL|wxALL|wxEXPAND, 5 );
	
	
	sbSizer4->Add( bSizer229, 1, wxEXPAND, 5 );
	
	
	bSizer135->Add( sbSizer4, 0, wxEXPAND, 5 );
	
	
	bSizer135->Add( 0, 0, 0, wxEXPAND, 5 );
	
	WarningText = new wxStaticText( m_panel38, wxID_ANY, wxT("Warning: running jobs \nmay affect exported coordinates"), wxDefaultPosition, wxDefaultSize, wxALIGN_CENTRE );
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
	NormalizeCheckBox->Connect( wxEVT_COMMAND_CHECKBOX_CLICKED, wxCommandEventHandler( RelionExportDialog::OnNormalizeCheckBox ), NULL, this );
	FlipCTFCheckBox->Connect( wxEVT_COMMAND_CHECKBOX_CLICKED, wxCommandEventHandler( RelionExportDialog::OnFlipCTFCheckBox ), NULL, this );
	OutputImageStackPicker->Connect( wxEVT_COMMAND_FILEPICKER_CHANGED, wxFileDirPickerEventHandler( RelionExportDialog::OnOutputImageStackFileChanged ), NULL, this );
	CancelButton->Connect( wxEVT_COMMAND_BUTTON_CLICKED, wxCommandEventHandler( RelionExportDialog::OnCancelButtonClick ), NULL, this );
	ExportButton->Connect( wxEVT_COMMAND_BUTTON_CLICKED, wxCommandEventHandler( RelionExportDialog::OnExportButtonClick ), NULL, this );
}

RelionExportDialog::~RelionExportDialog()
{
	// Disconnect Events
	NormalizeCheckBox->Disconnect( wxEVT_COMMAND_CHECKBOX_CLICKED, wxCommandEventHandler( RelionExportDialog::OnNormalizeCheckBox ), NULL, this );
	FlipCTFCheckBox->Disconnect( wxEVT_COMMAND_CHECKBOX_CLICKED, wxCommandEventHandler( RelionExportDialog::OnFlipCTFCheckBox ), NULL, this );
	OutputImageStackPicker->Disconnect( wxEVT_COMMAND_FILEPICKER_CHANGED, wxFileDirPickerEventHandler( RelionExportDialog::OnOutputImageStackFileChanged ), NULL, this );
	CancelButton->Disconnect( wxEVT_COMMAND_BUTTON_CLICKED, wxCommandEventHandler( RelionExportDialog::OnCancelButtonClick ), NULL, this );
	ExportButton->Disconnect( wxEVT_COMMAND_BUTTON_CLICKED, wxCommandEventHandler( RelionExportDialog::OnExportButtonClick ), NULL, this );
	
}

RefinementPackageAssetPanel::RefinementPackageAssetPanel( wxWindow* parent, wxWindowID id, const wxPoint& pos, const wxSize& size, long style ) : wxPanel( parent, id, pos, size, style )
{
	wxBoxSizer* bSizer187;
	bSizer187 = new wxBoxSizer( wxVERTICAL );
	
	m_staticline52 = new wxStaticLine( this, wxID_ANY, wxDefaultPosition, wxDefaultSize, wxLI_HORIZONTAL );
	bSizer187->Add( m_staticline52, 0, wxEXPAND | wxALL, 5 );
	
	m_splitter11 = new wxSplitterWindow( this, wxID_ANY, wxDefaultPosition, wxDefaultSize, wxSP_3D );
	m_splitter11->Connect( wxEVT_IDLE, wxIdleEventHandler( RefinementPackageAssetPanel::m_splitter11OnIdle ), NULL, this );
	
	m_panel50 = new wxPanel( m_splitter11, wxID_ANY, wxDefaultPosition, wxDefaultSize, wxTAB_TRAVERSAL );
	wxBoxSizer* bSizer188;
	bSizer188 = new wxBoxSizer( wxVERTICAL );
	
	m_staticText313 = new wxStaticText( m_panel50, wxID_ANY, wxT("Refinement Packages :"), wxDefaultPosition, wxDefaultSize, 0 );
	m_staticText313->Wrap( -1 );
	bSizer188->Add( m_staticText313, 0, wxALL, 5 );
	
	wxBoxSizer* bSizer145;
	bSizer145 = new wxBoxSizer( wxHORIZONTAL );
	
	wxBoxSizer* bSizer193;
	bSizer193 = new wxBoxSizer( wxVERTICAL );
	
	CreateButton = new wxButton( m_panel50, wxID_ANY, wxT("Create"), wxDefaultPosition, wxDefaultSize, 0 );
	bSizer193->Add( CreateButton, 0, wxALL, 5 );
	
	RenameButton = new wxButton( m_panel50, wxID_ANY, wxT("Rename"), wxDefaultPosition, wxDefaultSize, 0 );
	bSizer193->Add( RenameButton, 0, wxALL, 5 );
	
	DeleteButton = new wxButton( m_panel50, wxID_ANY, wxT("Delete"), wxDefaultPosition, wxDefaultSize, 0 );
	bSizer193->Add( DeleteButton, 0, wxALL, 5 );
	
	m_staticline122 = new wxStaticLine( m_panel50, wxID_ANY, wxDefaultPosition, wxDefaultSize, wxLI_HORIZONTAL );
	bSizer193->Add( m_staticline122, 0, wxEXPAND | wxALL, 5 );
	
	ImportButton = new wxButton( m_panel50, wxID_ANY, wxT("Import"), wxDefaultPosition, wxDefaultSize, 0 );
	bSizer193->Add( ImportButton, 0, wxALL, 5 );
	
	ExportButton = new wxButton( m_panel50, wxID_ANY, wxT("Export"), wxDefaultPosition, wxDefaultSize, 0 );
	bSizer193->Add( ExportButton, 0, wxALL, 5 );
	
	
	bSizer145->Add( bSizer193, 0, wxEXPAND, 5 );
	
	RefinementPackageListCtrl = new RefinementPackageListControl( m_panel50, wxID_ANY, wxDefaultPosition, wxDefaultSize, wxLC_EDIT_LABELS|wxLC_NO_HEADER|wxLC_REPORT|wxLC_SINGLE_SEL|wxLC_VIRTUAL );
	bSizer145->Add( RefinementPackageListCtrl, 1, wxALL|wxEXPAND, 5 );
	
	
	bSizer188->Add( bSizer145, 1, wxEXPAND, 5 );
	
	
	m_panel50->SetSizer( bSizer188 );
	m_panel50->Layout();
	bSizer188->Fit( m_panel50 );
	m_panel51 = new wxPanel( m_splitter11, wxID_ANY, wxDefaultPosition, wxDefaultSize, wxTAB_TRAVERSAL );
	wxBoxSizer* bSizer191;
	bSizer191 = new wxBoxSizer( wxVERTICAL );
	
	ContainedParticlesStaticText = new wxStaticText( m_panel51, wxID_ANY, wxT("Contained Particles :"), wxDefaultPosition, wxDefaultSize, 0 );
	ContainedParticlesStaticText->Wrap( -1 );
	bSizer191->Add( ContainedParticlesStaticText, 0, wxALL, 5 );
	
	ContainedParticlesListCtrl = new ContainedParticleListControl( m_panel51, wxID_ANY, wxDefaultPosition, wxDefaultSize, wxLC_REPORT|wxLC_SINGLE_SEL|wxLC_VIRTUAL );
	bSizer191->Add( ContainedParticlesListCtrl, 1, wxALL|wxEXPAND, 5 );
	
	wxBoxSizer* bSizer199;
	bSizer199 = new wxBoxSizer( wxHORIZONTAL );
	
	m_staticText230 = new wxStaticText( m_panel51, wxID_ANY, wxT("Active 3D References :"), wxDefaultPosition, wxDefaultSize, 0 );
	m_staticText230->Wrap( -1 );
	bSizer199->Add( m_staticText230, 0, wxALIGN_BOTTOM|wxALL, 5 );
	
	
	bSizer199->Add( 0, 0, 1, wxEXPAND, 5 );
	
	DisplayStackButton = new wxButton( m_panel51, wxID_ANY, wxT("Display Stack"), wxDefaultPosition, wxDefaultSize, 0 );
	bSizer199->Add( DisplayStackButton, 0, wxALL, 5 );
	
	
	bSizer191->Add( bSizer199, 0, wxEXPAND, 5 );
	
	Active3DReferencesListCtrl = new ReferenceVolumesListControl( m_panel51, wxID_ANY, wxDefaultPosition, wxDefaultSize, wxLC_REPORT|wxLC_SINGLE_SEL|wxLC_VIRTUAL );
	bSizer191->Add( Active3DReferencesListCtrl, 0, wxALL|wxEXPAND, 5 );
	
	
	m_panel51->SetSizer( bSizer191 );
	m_panel51->Layout();
	bSizer191->Fit( m_panel51 );
	m_splitter11->SplitVertically( m_panel50, m_panel51, 600 );
	bSizer187->Add( m_splitter11, 1, wxEXPAND, 5 );
	
	m_staticline53 = new wxStaticLine( this, wxID_ANY, wxDefaultPosition, wxDefaultSize, wxLI_HORIZONTAL );
	bSizer187->Add( m_staticline53, 0, wxEXPAND | wxALL, 5 );
	
	wxBoxSizer* bSizer192;
	bSizer192 = new wxBoxSizer( wxHORIZONTAL );
	
	wxGridSizer* gSizer12;
	gSizer12 = new wxGridSizer( 0, 6, 0, 0 );
	
	m_staticText319 = new wxStaticText( this, wxID_ANY, wxT("Stack Filename :"), wxDefaultPosition, wxDefaultSize, 0 );
	m_staticText319->Wrap( -1 );
	gSizer12->Add( m_staticText319, 0, wxALIGN_RIGHT|wxALL, 5 );
	
	StackFileNameText = new wxStaticText( this, wxID_ANY, wxEmptyString, wxDefaultPosition, wxDefaultSize, 0 );
	StackFileNameText->Wrap( -1 );
	gSizer12->Add( StackFileNameText, 0, wxALIGN_CENTER_VERTICAL|wxALL, 5 );
	
	
	gSizer12->Add( 0, 0, 1, wxEXPAND, 5 );
	
	
	gSizer12->Add( 0, 0, 1, wxEXPAND, 5 );
	
	
	gSizer12->Add( 0, 0, 1, wxEXPAND, 5 );
	
	
	gSizer12->Add( 0, 0, 1, wxEXPAND, 5 );
	
	m_staticText210 = new wxStaticText( this, wxID_ANY, wxT("Stack Box Size :"), wxDefaultPosition, wxDefaultSize, 0 );
	m_staticText210->Wrap( -1 );
	gSizer12->Add( m_staticText210, 0, wxALIGN_RIGHT|wxALL, 5 );
	
	StackBoxSizeText = new wxStaticText( this, wxID_ANY, wxEmptyString, wxDefaultPosition, wxDefaultSize, 0 );
	StackBoxSizeText->Wrap( -1 );
	gSizer12->Add( StackBoxSizeText, 0, wxALL, 5 );
	
	m_staticText315 = new wxStaticText( this, wxID_ANY, wxT("No. Classes :"), wxDefaultPosition, wxDefaultSize, 0 );
	m_staticText315->Wrap( -1 );
	gSizer12->Add( m_staticText315, 0, wxALIGN_CENTER_VERTICAL|wxALIGN_RIGHT|wxALL, 5 );
	
	NumberofClassesText = new wxStaticText( this, wxID_ANY, wxEmptyString, wxDefaultPosition, wxDefaultSize, 0 );
	NumberofClassesText->Wrap( -1 );
	gSizer12->Add( NumberofClassesText, 0, wxALIGN_CENTER_VERTICAL|wxALL, 5 );
	
	m_staticText279 = new wxStaticText( this, wxID_ANY, wxT("Symmetry :"), wxDefaultPosition, wxDefaultSize, 0 );
	m_staticText279->Wrap( -1 );
	gSizer12->Add( m_staticText279, 0, wxALIGN_RIGHT|wxALL, 5 );
	
	SymmetryText = new wxStaticText( this, wxID_ANY, wxEmptyString, wxDefaultPosition, wxDefaultSize, 0 );
	SymmetryText->Wrap( -1 );
	gSizer12->Add( SymmetryText, 0, wxALL, 5 );
	
	m_staticText281 = new wxStaticText( this, wxID_ANY, wxT("M.W. :"), wxDefaultPosition, wxDefaultSize, 0 );
	m_staticText281->Wrap( -1 );
	gSizer12->Add( m_staticText281, 0, wxALIGN_RIGHT|wxALL, 5 );
	
	MolecularWeightText = new wxStaticText( this, wxID_ANY, wxEmptyString, wxDefaultPosition, wxDefaultSize, 0 );
	MolecularWeightText->Wrap( -1 );
	gSizer12->Add( MolecularWeightText, 0, wxALL, 5 );
	
	m_staticText283 = new wxStaticText( this, wxID_ANY, wxT("Largest Dimension :"), wxDefaultPosition, wxDefaultSize, 0 );
	m_staticText283->Wrap( -1 );
	gSizer12->Add( m_staticText283, 0, wxALIGN_RIGHT|wxALL, 5 );
	
	LargestDimensionText = new wxStaticText( this, wxID_ANY, wxEmptyString, wxDefaultPosition, wxDefaultSize, 0 );
	LargestDimensionText->Wrap( -1 );
	gSizer12->Add( LargestDimensionText, 0, wxALL, 5 );
	
	m_staticText317 = new wxStaticText( this, wxID_ANY, wxT("No. Refinements :"), wxDefaultPosition, wxDefaultSize, 0 );
	m_staticText317->Wrap( -1 );
	gSizer12->Add( m_staticText317, 0, wxALIGN_CENTER_VERTICAL|wxALIGN_RIGHT|wxALL, 5 );
	
	NumberofRefinementsText = new wxStaticText( this, wxID_ANY, wxEmptyString, wxDefaultPosition, wxDefaultSize, 0 );
	NumberofRefinementsText->Wrap( -1 );
	gSizer12->Add( NumberofRefinementsText, 0, wxALIGN_CENTER_VERTICAL|wxALL, 5 );
	
	m_staticText212 = new wxStaticText( this, wxID_ANY, wxT("Last Refinement ID :"), wxDefaultPosition, wxDefaultSize, 0 );
	m_staticText212->Wrap( -1 );
	gSizer12->Add( m_staticText212, 0, wxALIGN_RIGHT|wxALL, 5 );
	
	LastRefinementIDText = new wxStaticText( this, wxID_ANY, wxEmptyString, wxDefaultPosition, wxDefaultSize, 0 );
	LastRefinementIDText->Wrap( -1 );
	gSizer12->Add( LastRefinementIDText, 0, wxALL, 5 );
	
	
	bSizer192->Add( gSizer12, 1, wxEXPAND, 5 );
	
	
	bSizer187->Add( bSizer192, 0, wxEXPAND, 5 );
	
	
	this->SetSizer( bSizer187 );
	this->Layout();
	
	// Connect Events
	this->Connect( wxEVT_UPDATE_UI, wxUpdateUIEventHandler( RefinementPackageAssetPanel::OnUpdateUI ) );
	CreateButton->Connect( wxEVT_COMMAND_BUTTON_CLICKED, wxCommandEventHandler( RefinementPackageAssetPanel::OnCreateClick ), NULL, this );
	RenameButton->Connect( wxEVT_COMMAND_BUTTON_CLICKED, wxCommandEventHandler( RefinementPackageAssetPanel::OnRenameClick ), NULL, this );
	DeleteButton->Connect( wxEVT_COMMAND_BUTTON_CLICKED, wxCommandEventHandler( RefinementPackageAssetPanel::OnDeleteClick ), NULL, this );
	ImportButton->Connect( wxEVT_COMMAND_BUTTON_CLICKED, wxCommandEventHandler( RefinementPackageAssetPanel::OnImportClick ), NULL, this );
	ExportButton->Connect( wxEVT_COMMAND_BUTTON_CLICKED, wxCommandEventHandler( RefinementPackageAssetPanel::OnExportClick ), NULL, this );
	RefinementPackageListCtrl->Connect( wxEVT_LEFT_DCLICK, wxMouseEventHandler( RefinementPackageAssetPanel::MouseCheckPackagesVeto ), NULL, this );
	RefinementPackageListCtrl->Connect( wxEVT_LEFT_DOWN, wxMouseEventHandler( RefinementPackageAssetPanel::MouseCheckPackagesVeto ), NULL, this );
	RefinementPackageListCtrl->Connect( wxEVT_LEFT_UP, wxMouseEventHandler( RefinementPackageAssetPanel::MouseVeto ), NULL, this );
	RefinementPackageListCtrl->Connect( wxEVT_COMMAND_LIST_BEGIN_LABEL_EDIT, wxListEventHandler( RefinementPackageAssetPanel::OnBeginEdit ), NULL, this );
	RefinementPackageListCtrl->Connect( wxEVT_COMMAND_LIST_END_LABEL_EDIT, wxListEventHandler( RefinementPackageAssetPanel::OnEndEdit ), NULL, this );
	RefinementPackageListCtrl->Connect( wxEVT_COMMAND_LIST_ITEM_ACTIVATED, wxListEventHandler( RefinementPackageAssetPanel::OnPackageActivated ), NULL, this );
	RefinementPackageListCtrl->Connect( wxEVT_COMMAND_LIST_ITEM_FOCUSED, wxListEventHandler( RefinementPackageAssetPanel::OnPackageFocusChange ), NULL, this );
	RefinementPackageListCtrl->Connect( wxEVT_MIDDLE_DCLICK, wxMouseEventHandler( RefinementPackageAssetPanel::MouseVeto ), NULL, this );
	RefinementPackageListCtrl->Connect( wxEVT_MIDDLE_DOWN, wxMouseEventHandler( RefinementPackageAssetPanel::MouseVeto ), NULL, this );
	RefinementPackageListCtrl->Connect( wxEVT_MIDDLE_UP, wxMouseEventHandler( RefinementPackageAssetPanel::MouseVeto ), NULL, this );
	RefinementPackageListCtrl->Connect( wxEVT_MOTION, wxMouseEventHandler( RefinementPackageAssetPanel::MouseVeto ), NULL, this );
	RefinementPackageListCtrl->Connect( wxEVT_RIGHT_DCLICK, wxMouseEventHandler( RefinementPackageAssetPanel::MouseVeto ), NULL, this );
	RefinementPackageListCtrl->Connect( wxEVT_RIGHT_DOWN, wxMouseEventHandler( RefinementPackageAssetPanel::MouseVeto ), NULL, this );
	RefinementPackageListCtrl->Connect( wxEVT_RIGHT_UP, wxMouseEventHandler( RefinementPackageAssetPanel::MouseVeto ), NULL, this );
	ContainedParticlesListCtrl->Connect( wxEVT_LEFT_DCLICK, wxMouseEventHandler( RefinementPackageAssetPanel::MouseCheckParticlesVeto ), NULL, this );
	ContainedParticlesListCtrl->Connect( wxEVT_LEFT_DOWN, wxMouseEventHandler( RefinementPackageAssetPanel::MouseCheckParticlesVeto ), NULL, this );
	ContainedParticlesListCtrl->Connect( wxEVT_LEFT_UP, wxMouseEventHandler( RefinementPackageAssetPanel::MouseVeto ), NULL, this );
	ContainedParticlesListCtrl->Connect( wxEVT_MIDDLE_DCLICK, wxMouseEventHandler( RefinementPackageAssetPanel::MouseVeto ), NULL, this );
	ContainedParticlesListCtrl->Connect( wxEVT_MIDDLE_DOWN, wxMouseEventHandler( RefinementPackageAssetPanel::MouseVeto ), NULL, this );
	ContainedParticlesListCtrl->Connect( wxEVT_MIDDLE_UP, wxMouseEventHandler( RefinementPackageAssetPanel::MouseVeto ), NULL, this );
	ContainedParticlesListCtrl->Connect( wxEVT_MOTION, wxMouseEventHandler( RefinementPackageAssetPanel::MouseVeto ), NULL, this );
	ContainedParticlesListCtrl->Connect( wxEVT_RIGHT_DCLICK, wxMouseEventHandler( RefinementPackageAssetPanel::MouseVeto ), NULL, this );
	ContainedParticlesListCtrl->Connect( wxEVT_RIGHT_DOWN, wxMouseEventHandler( RefinementPackageAssetPanel::MouseVeto ), NULL, this );
	ContainedParticlesListCtrl->Connect( wxEVT_RIGHT_UP, wxMouseEventHandler( RefinementPackageAssetPanel::MouseVeto ), NULL, this );
	DisplayStackButton->Connect( wxEVT_COMMAND_BUTTON_CLICKED, wxCommandEventHandler( RefinementPackageAssetPanel::OnDisplayStackButton ), NULL, this );
	Active3DReferencesListCtrl->Connect( wxEVT_COMMAND_LIST_ITEM_ACTIVATED, wxListEventHandler( RefinementPackageAssetPanel::OnVolumeListItemActivated ), NULL, this );
}

RefinementPackageAssetPanel::~RefinementPackageAssetPanel()
{
	// Disconnect Events
	this->Disconnect( wxEVT_UPDATE_UI, wxUpdateUIEventHandler( RefinementPackageAssetPanel::OnUpdateUI ) );
	CreateButton->Disconnect( wxEVT_COMMAND_BUTTON_CLICKED, wxCommandEventHandler( RefinementPackageAssetPanel::OnCreateClick ), NULL, this );
	RenameButton->Disconnect( wxEVT_COMMAND_BUTTON_CLICKED, wxCommandEventHandler( RefinementPackageAssetPanel::OnRenameClick ), NULL, this );
	DeleteButton->Disconnect( wxEVT_COMMAND_BUTTON_CLICKED, wxCommandEventHandler( RefinementPackageAssetPanel::OnDeleteClick ), NULL, this );
	ImportButton->Disconnect( wxEVT_COMMAND_BUTTON_CLICKED, wxCommandEventHandler( RefinementPackageAssetPanel::OnImportClick ), NULL, this );
	ExportButton->Disconnect( wxEVT_COMMAND_BUTTON_CLICKED, wxCommandEventHandler( RefinementPackageAssetPanel::OnExportClick ), NULL, this );
	RefinementPackageListCtrl->Disconnect( wxEVT_LEFT_DCLICK, wxMouseEventHandler( RefinementPackageAssetPanel::MouseCheckPackagesVeto ), NULL, this );
	RefinementPackageListCtrl->Disconnect( wxEVT_LEFT_DOWN, wxMouseEventHandler( RefinementPackageAssetPanel::MouseCheckPackagesVeto ), NULL, this );
	RefinementPackageListCtrl->Disconnect( wxEVT_LEFT_UP, wxMouseEventHandler( RefinementPackageAssetPanel::MouseVeto ), NULL, this );
	RefinementPackageListCtrl->Disconnect( wxEVT_COMMAND_LIST_BEGIN_LABEL_EDIT, wxListEventHandler( RefinementPackageAssetPanel::OnBeginEdit ), NULL, this );
	RefinementPackageListCtrl->Disconnect( wxEVT_COMMAND_LIST_END_LABEL_EDIT, wxListEventHandler( RefinementPackageAssetPanel::OnEndEdit ), NULL, this );
	RefinementPackageListCtrl->Disconnect( wxEVT_COMMAND_LIST_ITEM_ACTIVATED, wxListEventHandler( RefinementPackageAssetPanel::OnPackageActivated ), NULL, this );
	RefinementPackageListCtrl->Disconnect( wxEVT_COMMAND_LIST_ITEM_FOCUSED, wxListEventHandler( RefinementPackageAssetPanel::OnPackageFocusChange ), NULL, this );
	RefinementPackageListCtrl->Disconnect( wxEVT_MIDDLE_DCLICK, wxMouseEventHandler( RefinementPackageAssetPanel::MouseVeto ), NULL, this );
	RefinementPackageListCtrl->Disconnect( wxEVT_MIDDLE_DOWN, wxMouseEventHandler( RefinementPackageAssetPanel::MouseVeto ), NULL, this );
	RefinementPackageListCtrl->Disconnect( wxEVT_MIDDLE_UP, wxMouseEventHandler( RefinementPackageAssetPanel::MouseVeto ), NULL, this );
	RefinementPackageListCtrl->Disconnect( wxEVT_MOTION, wxMouseEventHandler( RefinementPackageAssetPanel::MouseVeto ), NULL, this );
	RefinementPackageListCtrl->Disconnect( wxEVT_RIGHT_DCLICK, wxMouseEventHandler( RefinementPackageAssetPanel::MouseVeto ), NULL, this );
	RefinementPackageListCtrl->Disconnect( wxEVT_RIGHT_DOWN, wxMouseEventHandler( RefinementPackageAssetPanel::MouseVeto ), NULL, this );
	RefinementPackageListCtrl->Disconnect( wxEVT_RIGHT_UP, wxMouseEventHandler( RefinementPackageAssetPanel::MouseVeto ), NULL, this );
	ContainedParticlesListCtrl->Disconnect( wxEVT_LEFT_DCLICK, wxMouseEventHandler( RefinementPackageAssetPanel::MouseCheckParticlesVeto ), NULL, this );
	ContainedParticlesListCtrl->Disconnect( wxEVT_LEFT_DOWN, wxMouseEventHandler( RefinementPackageAssetPanel::MouseCheckParticlesVeto ), NULL, this );
	ContainedParticlesListCtrl->Disconnect( wxEVT_LEFT_UP, wxMouseEventHandler( RefinementPackageAssetPanel::MouseVeto ), NULL, this );
	ContainedParticlesListCtrl->Disconnect( wxEVT_MIDDLE_DCLICK, wxMouseEventHandler( RefinementPackageAssetPanel::MouseVeto ), NULL, this );
	ContainedParticlesListCtrl->Disconnect( wxEVT_MIDDLE_DOWN, wxMouseEventHandler( RefinementPackageAssetPanel::MouseVeto ), NULL, this );
	ContainedParticlesListCtrl->Disconnect( wxEVT_MIDDLE_UP, wxMouseEventHandler( RefinementPackageAssetPanel::MouseVeto ), NULL, this );
	ContainedParticlesListCtrl->Disconnect( wxEVT_MOTION, wxMouseEventHandler( RefinementPackageAssetPanel::MouseVeto ), NULL, this );
	ContainedParticlesListCtrl->Disconnect( wxEVT_RIGHT_DCLICK, wxMouseEventHandler( RefinementPackageAssetPanel::MouseVeto ), NULL, this );
	ContainedParticlesListCtrl->Disconnect( wxEVT_RIGHT_DOWN, wxMouseEventHandler( RefinementPackageAssetPanel::MouseVeto ), NULL, this );
	ContainedParticlesListCtrl->Disconnect( wxEVT_RIGHT_UP, wxMouseEventHandler( RefinementPackageAssetPanel::MouseVeto ), NULL, this );
	DisplayStackButton->Disconnect( wxEVT_COMMAND_BUTTON_CLICKED, wxCommandEventHandler( RefinementPackageAssetPanel::OnDisplayStackButton ), NULL, this );
	Active3DReferencesListCtrl->Disconnect( wxEVT_COMMAND_LIST_ITEM_ACTIVATED, wxListEventHandler( RefinementPackageAssetPanel::OnVolumeListItemActivated ), NULL, this );
	
}

TemplateWizardPanel::TemplateWizardPanel( wxWindow* parent, wxWindowID id, const wxPoint& pos, const wxSize& size, long style ) : wxPanel( parent, id, pos, size, style )
{
	wxBoxSizer* bSizer153;
	bSizer153 = new wxBoxSizer( wxVERTICAL );
	
	wxBoxSizer* bSizer147;
	bSizer147 = new wxBoxSizer( wxHORIZONTAL );
	
	m_staticText214 = new wxStaticText( this, wxID_ANY, wxT("Template Refinement Package :"), wxDefaultPosition, wxDefaultSize, 0 );
	m_staticText214->Wrap( -1 );
	bSizer147->Add( m_staticText214, 0, wxALIGN_CENTER_VERTICAL|wxALL, 5 );
	
	GroupComboBox = new wxComboBox( this, wxID_ANY, wxT("Combo!"), wxDefaultPosition, wxDefaultSize, 0, NULL, wxCB_READONLY ); 
	bSizer147->Add( GroupComboBox, 1, wxALL, 5 );
	
	
	bSizer153->Add( bSizer147, 0, wxEXPAND, 5 );
	
	
	bSizer153->Add( 0, 0, 1, wxEXPAND, 5 );
	
	InfoText = new AutoWrapStaticText( this, wxID_ANY, wxT("Select \"New Refinement Package\" to create a package from scratch.  Select \"Create From 2D Class Average Selection\" to create a new package based on one or more selections of 2D class averages. Alternatively, an existing refinement package can be used as a template.  Template based refinement packages will have the same particle stack as their template, but you will be able to change the other parameters, you may wish to do this to change the number of classes or symmetry for example."), wxDefaultPosition, wxDefaultSize, 0 );
	InfoText->Wrap( -1 );
	bSizer153->Add( InfoText, 0, wxALL|wxEXPAND, 5 );
	
	
	this->SetSizer( bSizer153 );
	this->Layout();
}

TemplateWizardPanel::~TemplateWizardPanel()
{
}

InputParameterWizardPanel::InputParameterWizardPanel( wxWindow* parent, wxWindowID id, const wxPoint& pos, const wxSize& size, long style ) : wxPanel( parent, id, pos, size, style )
{
	wxBoxSizer* bSizer153;
	bSizer153 = new wxBoxSizer( wxVERTICAL );
	
	wxBoxSizer* bSizer147;
	bSizer147 = new wxBoxSizer( wxHORIZONTAL );
	
	m_staticText214 = new wxStaticText( this, wxID_ANY, wxT("Input Parameters :"), wxDefaultPosition, wxDefaultSize, 0 );
	m_staticText214->Wrap( -1 );
	bSizer147->Add( m_staticText214, 0, wxALIGN_CENTER_VERTICAL|wxALL, 5 );
	
	GroupComboBox = new wxComboBox( this, wxID_ANY, wxT("Combo!"), wxDefaultPosition, wxDefaultSize, 0, NULL, wxCB_READONLY ); 
	bSizer147->Add( GroupComboBox, 1, wxALL, 5 );
	
	
	bSizer153->Add( bSizer147, 0, wxEXPAND, 5 );
	
	
	bSizer153->Add( 0, 0, 1, wxEXPAND, 5 );
	
	InfoText = new AutoWrapStaticText( this, wxID_ANY, wxT("Please select the parameters to use to create the new refinement package."), wxDefaultPosition, wxDefaultSize, 0 );
	InfoText->Wrap( -1 );
	bSizer153->Add( InfoText, 0, wxALL|wxEXPAND, 5 );
	
	
	this->SetSizer( bSizer153 );
	this->Layout();
}

InputParameterWizardPanel::~InputParameterWizardPanel()
{
}

ClassSelectionWizardPanel::ClassSelectionWizardPanel( wxWindow* parent, wxWindowID id, const wxPoint& pos, const wxSize& size, long style ) : wxPanel( parent, id, pos, size, style )
{
	wxBoxSizer* bSizer153;
	bSizer153 = new wxBoxSizer( wxVERTICAL );
	
	m_staticText416 = new wxStaticText( this, wxID_ANY, wxT("Select class average selections to use :"), wxDefaultPosition, wxDefaultSize, 0 );
	m_staticText416->Wrap( -1 );
	bSizer153->Add( m_staticText416, 0, wxALL, 5 );
	
	wxBoxSizer* bSizer278;
	bSizer278 = new wxBoxSizer( wxHORIZONTAL );
	
	SelectionListCtrl = new wxListCtrl( this, wxID_ANY, wxDefaultPosition, wxDefaultSize, wxLC_EDIT_LABELS|wxLC_REPORT );
	bSizer278->Add( SelectionListCtrl, 1, wxALL|wxEXPAND, 5 );
	
	
	bSizer153->Add( bSizer278, 1, wxEXPAND, 5 );
	
	InfoText = new AutoWrapStaticText( this, wxID_ANY, wxT("If you have performed any 2D classifications, then you can use one or more selections of class averages (selected from the classification results panel) to copy over ONLY the particles in the selected class averages to the new refinement package.  This may be done to clean the dataset for example."), wxDefaultPosition, wxDefaultSize, 0 );
	InfoText->Wrap( -1 );
	bSizer153->Add( InfoText, 0, wxALL|wxEXPAND, 5 );
	
	
	this->SetSizer( bSizer153 );
	this->Layout();
}

ClassSelectionWizardPanel::~ClassSelectionWizardPanel()
{
}

SymmetryWizardPanel::SymmetryWizardPanel( wxWindow* parent, wxWindowID id, const wxPoint& pos, const wxSize& size, long style ) : wxPanel( parent, id, pos, size, style )
{
	wxBoxSizer* bSizer153;
	bSizer153 = new wxBoxSizer( wxVERTICAL );
	
	wxBoxSizer* bSizer147;
	bSizer147 = new wxBoxSizer( wxHORIZONTAL );
	
	m_staticText214 = new wxStaticText( this, wxID_ANY, wxT("Wanted Pointgroup Symmetry :"), wxDefaultPosition, wxDefaultSize, 0 );
	m_staticText214->Wrap( -1 );
	bSizer147->Add( m_staticText214, 0, wxALIGN_CENTER_VERTICAL|wxALL, 5 );
	
	SymmetryComboBox = new wxComboBox( this, wxID_ANY, wxT("C1"), wxDefaultPosition, wxDefaultSize, 0, NULL, 0 ); 
	bSizer147->Add( SymmetryComboBox, 1, wxALL, 5 );
	
	
	bSizer153->Add( bSizer147, 0, wxEXPAND, 5 );
	
	
	bSizer153->Add( 0, 0, 1, wxEXPAND, 5 );
	
	InfoText = new AutoWrapStaticText( this, wxID_ANY, wxT("Please select the pointgroup symmetry to use for the refinement.  Whilst a number of symmetries can be selected from the menu for convenience, you may enter any valid pointgroup symmetry (in Schoenflies notation)."), wxDefaultPosition, wxDefaultSize, 0 );
	InfoText->Wrap( -1 );
	bSizer153->Add( InfoText, 0, wxALL|wxEXPAND, 5 );
	
	
	this->SetSizer( bSizer153 );
	this->Layout();
}

SymmetryWizardPanel::~SymmetryWizardPanel()
{
}

MolecularWeightWizardPanel::MolecularWeightWizardPanel( wxWindow* parent, wxWindowID id, const wxPoint& pos, const wxSize& size, long style ) : wxPanel( parent, id, pos, size, style )
{
	wxBoxSizer* bSizer153;
	bSizer153 = new wxBoxSizer( wxVERTICAL );
	
	wxBoxSizer* bSizer147;
	bSizer147 = new wxBoxSizer( wxHORIZONTAL );
	
	m_staticText214 = new wxStaticText( this, wxID_ANY, wxT("Estimated Molecular Weight (kDa) :"), wxDefaultPosition, wxDefaultSize, 0 );
	m_staticText214->Wrap( -1 );
	bSizer147->Add( m_staticText214, 0, wxALIGN_CENTER_VERTICAL|wxALL, 5 );
	
	MolecularWeightTextCtrl = new NumericTextCtrl( this, wxID_ANY, wxT("0.0"), wxDefaultPosition, wxDefaultSize, 0 );
	bSizer147->Add( MolecularWeightTextCtrl, 1, wxALL, 5 );
	
	
	bSizer153->Add( bSizer147, 0, wxEXPAND, 5 );
	
	
	bSizer153->Add( 0, 0, 1, wxEXPAND, 5 );
	
	InfoText = new AutoWrapStaticText( this, wxID_ANY, wxT("Please enter the estimated molecular weight for the refinement.  In general, this should be the molecular weight of the coherent parts (e.g. micelle would not be included)."), wxDefaultPosition, wxDefaultSize, 0 );
	InfoText->Wrap( -1 );
	bSizer153->Add( InfoText, 0, wxALL|wxEXPAND, 5 );
	
	
	this->SetSizer( bSizer153 );
	this->Layout();
}

MolecularWeightWizardPanel::~MolecularWeightWizardPanel()
{
}

InitialReferenceSelectWizardPanel::InitialReferenceSelectWizardPanel( wxWindow* parent, wxWindowID id, const wxPoint& pos, const wxSize& size, long style ) : wxPanel( parent, id, pos, size, style )
{
	MainSizer = new wxBoxSizer( wxVERTICAL );
	
	TitleText = new wxStaticText( this, wxID_ANY, wxT("Initial class references :-"), wxDefaultPosition, wxDefaultSize, 0 );
	TitleText->Wrap( -1 );
	MainSizer->Add( TitleText, 0, wxALL, 5 );
	
	m_staticline108 = new wxStaticLine( this, wxID_ANY, wxDefaultPosition, wxDefaultSize, wxLI_HORIZONTAL );
	MainSizer->Add( m_staticline108, 0, wxEXPAND | wxALL, 5 );
	
	ScrollWindow = new wxScrolledWindow( this, wxID_ANY, wxDefaultPosition, wxSize( 100,-1 ), wxHSCROLL|wxVSCROLL );
	ScrollWindow->SetScrollRate( 5, 5 );
	ScrollSizer = new wxBoxSizer( wxVERTICAL );
	
	
	ScrollWindow->SetSizer( ScrollSizer );
	ScrollWindow->Layout();
	MainSizer->Add( ScrollWindow, 1, wxALL|wxEXPAND, 5 );
	
	InfoText = new AutoWrapStaticText( this, wxID_ANY, wxT("Select the volume asset to use for the initial reference for each of the classes.  If \"Generate from params.\" is selected, a new volume asset will be calculated from the starting parameters at the start of the next refinement. In general, you will provide a reference for a new refinement package, and generate one when using a template."), wxDefaultPosition, wxDefaultSize, 0 );
	InfoText->Wrap( -1 );
	MainSizer->Add( InfoText, 0, wxALL|wxEXPAND, 5 );
	
	
	this->SetSizer( MainSizer );
	this->Layout();
}

InitialReferenceSelectWizardPanel::~InitialReferenceSelectWizardPanel()
{
}

LargestDimensionWizardPanel::LargestDimensionWizardPanel( wxWindow* parent, wxWindowID id, const wxPoint& pos, const wxSize& size, long style ) : wxPanel( parent, id, pos, size, style )
{
	wxBoxSizer* bSizer153;
	bSizer153 = new wxBoxSizer( wxVERTICAL );
	
	wxBoxSizer* bSizer147;
	bSizer147 = new wxBoxSizer( wxHORIZONTAL );
	
	m_staticText214 = new wxStaticText( this, wxID_ANY, wxT("Estimated Largest Dimension / Diameter (Å) :"), wxDefaultPosition, wxDefaultSize, 0 );
	m_staticText214->Wrap( -1 );
	bSizer147->Add( m_staticText214, 0, wxALIGN_CENTER_VERTICAL|wxALL, 5 );
	
	LargestDimensionTextCtrl = new NumericTextCtrl( this, wxID_ANY, wxT("0.0"), wxDefaultPosition, wxDefaultSize, 0 );
	bSizer147->Add( LargestDimensionTextCtrl, 1, wxALL, 5 );
	
	
	bSizer153->Add( bSizer147, 0, wxEXPAND, 5 );
	
	
	bSizer153->Add( 0, 0, 1, wxEXPAND, 5 );
	
	InfoText = new AutoWrapStaticText( this, wxID_ANY, wxT("Please enter the estimated largest dimension of the particle. This will be used for autosizing masks and shift limits, it need not be exact, and it is better to err on the large size. Please note, this is the largest dimension / diameter - NOT RADIUS."), wxDefaultPosition, wxDefaultSize, 0 );
	InfoText->Wrap( -1 );
	bSizer153->Add( InfoText, 0, wxALL|wxEXPAND, 5 );
	
	
	this->SetSizer( bSizer153 );
	this->Layout();
}

LargestDimensionWizardPanel::~LargestDimensionWizardPanel()
{
}

OutputPixelSizeWizardPanel::OutputPixelSizeWizardPanel( wxWindow* parent, wxWindowID id, const wxPoint& pos, const wxSize& size, long style ) : wxPanel( parent, id, pos, size, style )
{
	wxBoxSizer* bSizer153;
	bSizer153 = new wxBoxSizer( wxVERTICAL );
	
	wxBoxSizer* bSizer147;
	bSizer147 = new wxBoxSizer( wxHORIZONTAL );
	
	m_staticText214 = new wxStaticText( this, wxID_ANY, wxT("Wanted Output Pixel Size (Å) :"), wxDefaultPosition, wxDefaultSize, 0 );
	m_staticText214->Wrap( -1 );
	bSizer147->Add( m_staticText214, 0, wxALIGN_CENTER_VERTICAL|wxALL, 5 );
	
	OutputPixelSizeTextCtrl = new NumericTextCtrl( this, wxID_ANY, wxT("0.0"), wxDefaultPosition, wxDefaultSize, 0 );
	bSizer147->Add( OutputPixelSizeTextCtrl, 1, wxALL, 5 );
	
	
	bSizer153->Add( bSizer147, 0, wxEXPAND, 5 );
	
	
	bSizer153->Add( 0, 0, 1, wxEXPAND, 5 );
	
	InfoText = new AutoWrapStaticText( this, wxID_ANY, wxT("Please enter the pixel size for outputs.  In most cases this will be the pixel size of your images, however in the case where you have data of mixed pixel sizes you must enter a specific pixel size here.  All images will be scaled to this pixel size during process, and all outputs will be this pixel size."), wxDefaultPosition, wxDefaultSize, 0 );
	InfoText->Wrap( -1 );
	bSizer153->Add( InfoText, 0, wxALL|wxEXPAND, 5 );
	
	
	this->SetSizer( bSizer153 );
	this->Layout();
}

OutputPixelSizeWizardPanel::~OutputPixelSizeWizardPanel()
{
}

ParticleGroupWizardPanel::ParticleGroupWizardPanel( wxWindow* parent, wxWindowID id, const wxPoint& pos, const wxSize& size, long style ) : wxPanel( parent, id, pos, size, style )
{
	wxBoxSizer* bSizer1531;
	bSizer1531 = new wxBoxSizer( wxVERTICAL );
	
	wxBoxSizer* bSizer1471;
	bSizer1471 = new wxBoxSizer( wxHORIZONTAL );
	
	m_staticText2141 = new wxStaticText( this, wxID_ANY, wxT("Particle Position Group :"), wxDefaultPosition, wxDefaultSize, 0 );
	m_staticText2141->Wrap( -1 );
	bSizer1471->Add( m_staticText2141, 0, wxALIGN_CENTER_VERTICAL|wxALL, 5 );
	
	ParticlePositionsGroupComboBox = new wxComboBox( this, wxID_ANY, wxT("Combo!"), wxDefaultPosition, wxDefaultSize, 0, NULL, wxCB_READONLY ); 
	bSizer1471->Add( ParticlePositionsGroupComboBox, 1, wxALL, 5 );
	
	
	bSizer1531->Add( bSizer1471, 0, wxEXPAND, 5 );
	
	
	bSizer1531->Add( 0, 0, 1, wxEXPAND, 5 );
	
	InfoText = new AutoWrapStaticText( this, wxID_ANY, wxT("Please select a group of particle positions to use for the refinement package.  The particles will be \"cut\" from the active image assets using the active CTF estimation parameters.  From this point on the particle images and defocus parameters will be fixed, and changing the active results will not change the refinement package."), wxDefaultPosition, wxDefaultSize, 0 );
	InfoText->Wrap( -1 );
	bSizer1531->Add( InfoText, 0, wxALL|wxEXPAND, 5 );
	
	
	this->SetSizer( bSizer1531 );
	this->Layout();
}

ParticleGroupWizardPanel::~ParticleGroupWizardPanel()
{
}

BoxSizeWizardPanel::BoxSizeWizardPanel( wxWindow* parent, wxWindowID id, const wxPoint& pos, const wxSize& size, long style ) : wxPanel( parent, id, pos, size, style )
{
	wxBoxSizer* bSizer15311;
	bSizer15311 = new wxBoxSizer( wxVERTICAL );
	
	wxBoxSizer* bSizer14711;
	bSizer14711 = new wxBoxSizer( wxHORIZONTAL );
	
	m_staticText21411 = new wxStaticText( this, wxID_ANY, wxT("Box size for particles (pixels) :"), wxDefaultPosition, wxDefaultSize, 0 );
	m_staticText21411->Wrap( -1 );
	bSizer14711->Add( m_staticText21411, 0, wxALIGN_CENTER_VERTICAL|wxALL, 5 );
	
	BoxSizeSpinCtrl = new wxSpinCtrl( this, wxID_ANY, wxT("1"), wxDefaultPosition, wxDefaultSize, wxSP_ARROW_KEYS, 1, 2048, 1 );
	bSizer14711->Add( BoxSizeSpinCtrl, 1, wxALL, 5 );
	
	
	bSizer15311->Add( bSizer14711, 0, wxEXPAND, 5 );
	
	
	bSizer15311->Add( 0, 0, 1, wxEXPAND, 5 );
	
	InfoText = new AutoWrapStaticText( this, wxID_ANY, wxT("Select the box size that should be used to extract the particles.  This box size should be big enough to not only contain the full particle, but also to take into account the delocalisation of information due to the CTF, as well as any aliasing of the CTF."), wxDefaultPosition, wxDefaultSize, 0 );
	InfoText->Wrap( -1 );
	bSizer15311->Add( InfoText, 0, wxALL|wxEXPAND, 5 );
	
	
	this->SetSizer( bSizer15311 );
	this->Layout();
}

BoxSizeWizardPanel::~BoxSizeWizardPanel()
{
}

NumberofClassesWizardPanel::NumberofClassesWizardPanel( wxWindow* parent, wxWindowID id, const wxPoint& pos, const wxSize& size, long style ) : wxPanel( parent, id, pos, size, style )
{
	wxBoxSizer* bSizer153111;
	bSizer153111 = new wxBoxSizer( wxVERTICAL );
	
	wxBoxSizer* bSizer147111;
	bSizer147111 = new wxBoxSizer( wxHORIZONTAL );
	
	m_staticText214111 = new wxStaticText( this, wxID_ANY, wxT("Number of classes for 3D refinement :"), wxDefaultPosition, wxDefaultSize, 0 );
	m_staticText214111->Wrap( -1 );
	bSizer147111->Add( m_staticText214111, 0, wxALIGN_CENTER_VERTICAL|wxALL, 5 );
	
	NumberOfClassesSpinCtrl = new wxSpinCtrl( this, wxID_ANY, wxT("1"), wxDefaultPosition, wxDefaultSize, wxSP_ARROW_KEYS, 1, 100, 0 );
	bSizer147111->Add( NumberOfClassesSpinCtrl, 1, wxALL, 5 );
	
	
	bSizer153111->Add( bSizer147111, 0, wxEXPAND, 5 );
	
	
	bSizer153111->Add( 0, 0, 1, wxEXPAND, 5 );
	
	InfoText = new AutoWrapStaticText( this, wxID_ANY, wxT("Select the number 3D of classes to use during 3D refinement.  This number only affects 3D refinements, and will have no effect during 2D classification.  This number is fixed for a refinement package, but can be changed by creating a new refinement package and using an old refinement package as a template.  In general you will start with 1 class, and split into more classes later in the refinement.If this package is based on a previous package, you will be given options about how to split/merge classes in the following steps.  "), wxDefaultPosition, wxDefaultSize, 0 );
	InfoText->Wrap( -1 );
	bSizer153111->Add( InfoText, 0, wxALL|wxEXPAND, 10 );
	
	
	this->SetSizer( bSizer153111 );
	this->Layout();
}

NumberofClassesWizardPanel::~NumberofClassesWizardPanel()
{
}

ClassesSetupWizardPanelA::ClassesSetupWizardPanelA( wxWindow* parent, wxWindowID id, const wxPoint& pos, const wxSize& size, long style ) : wxPanel( parent, id, pos, size, style )
{
	wxBoxSizer* bSizer15311;
	bSizer15311 = new wxBoxSizer( wxVERTICAL );
	
	wxBoxSizer* bSizer14711;
	bSizer14711 = new wxBoxSizer( wxHORIZONTAL );
	
	m_staticText21411 = new wxStaticText( this, wxID_ANY, wxT("Carry over all Particles? : "), wxDefaultPosition, wxDefaultSize, 0 );
	m_staticText21411->Wrap( -1 );
	bSizer14711->Add( m_staticText21411, 0, wxALIGN_CENTER_VERTICAL|wxALL, 5 );
	
	wxBoxSizer* bSizer367;
	bSizer367 = new wxBoxSizer( wxHORIZONTAL );
	
	CarryOverYesButton = new wxRadioButton( this, wxID_ANY, wxT("Yes"), wxDefaultPosition, wxDefaultSize, 0 );
	bSizer367->Add( CarryOverYesButton, 0, wxALL, 5 );
	
	m_radioBtn40 = new wxRadioButton( this, wxID_ANY, wxT("No"), wxDefaultPosition, wxDefaultSize, 0 );
	bSizer367->Add( m_radioBtn40, 0, wxALL, 5 );
	
	
	bSizer14711->Add( bSizer367, 1, wxEXPAND, 5 );
	
	
	bSizer15311->Add( bSizer14711, 0, wxEXPAND, 5 );
	
	
	bSizer15311->Add( 0, 0, 1, wxEXPAND, 5 );
	
	InfoText = new AutoWrapStaticText( this, wxID_ANY, wxT("Do you want to carry over all particles from the template refinement package to the new refinement package?  If No, you will be able to select the classes from which particles should be carried over (based on which class has the highest occupancy for that particle), and a new particle stack will be created. If yes all the particles will be included in the new refinement package."), wxDefaultPosition, wxDefaultSize, 0 );
	InfoText->Wrap( -1 );
	bSizer15311->Add( InfoText, 0, wxALL|wxEXPAND, 5 );
	
	
	this->SetSizer( bSizer15311 );
	this->Layout();
}

ClassesSetupWizardPanelA::~ClassesSetupWizardPanelA()
{
}

RecentrePicksWizardPanel::RecentrePicksWizardPanel( wxWindow* parent, wxWindowID id, const wxPoint& pos, const wxSize& size, long style ) : wxPanel( parent, id, pos, size, style )
{
	wxBoxSizer* bSizer15311;
	bSizer15311 = new wxBoxSizer( wxVERTICAL );
	
	wxBoxSizer* bSizer14711;
	bSizer14711 = new wxBoxSizer( wxHORIZONTAL );
	
	m_staticText21411 = new wxStaticText( this, wxID_ANY, wxT("Re-Centre Based on Classification? : "), wxDefaultPosition, wxDefaultSize, 0 );
	m_staticText21411->Wrap( -1 );
	bSizer14711->Add( m_staticText21411, 0, wxALIGN_CENTER_VERTICAL|wxALL, 5 );
	
	wxBoxSizer* bSizer367;
	bSizer367 = new wxBoxSizer( wxHORIZONTAL );
	
	ReCentreYesButton = new wxRadioButton( this, wxID_ANY, wxT("Yes"), wxDefaultPosition, wxDefaultSize, 0 );
	bSizer367->Add( ReCentreYesButton, 0, wxALL, 5 );
	
	m_radioBtn40 = new wxRadioButton( this, wxID_ANY, wxT("No"), wxDefaultPosition, wxDefaultSize, 0 );
	bSizer367->Add( m_radioBtn40, 0, wxALL, 5 );
	
	
	bSizer14711->Add( bSizer367, 1, wxEXPAND, 5 );
	
	
	bSizer15311->Add( bSizer14711, 0, wxEXPAND, 5 );
	
	
	bSizer15311->Add( 0, 0, 1, wxEXPAND, 5 );
	
	InfoText = new AutoWrapStaticText( this, wxID_ANY, wxT("Do you want to use the shifts estimated for the class averages to re-centre the picking co-ordinates prior to cutting out the images?  This can be especially helpful when the centreing on the picking was not very good, and a subsequent 2D classification with centre averages set to yes was run.   Note : This will only apply if cisTEM has images and co-ordinates for the particles, it will do nothing if the images come from an imported stack."), wxDefaultPosition, wxDefaultSize, 0 );
	InfoText->Wrap( -1 );
	bSizer15311->Add( InfoText, 0, wxALL|wxEXPAND, 5 );
	
	
	this->SetSizer( bSizer15311 );
	this->Layout();
}

RecentrePicksWizardPanel::~RecentrePicksWizardPanel()
{
}

RemoveDuplicatesWizardPanel::RemoveDuplicatesWizardPanel( wxWindow* parent, wxWindowID id, const wxPoint& pos, const wxSize& size, long style ) : wxPanel( parent, id, pos, size, style )
{
	wxBoxSizer* bSizer15311;
	bSizer15311 = new wxBoxSizer( wxVERTICAL );
	
	wxBoxSizer* bSizer14711;
	bSizer14711 = new wxBoxSizer( wxHORIZONTAL );
	
	m_staticText21411 = new wxStaticText( this, wxID_ANY, wxT("Remove Duplicate Particle Positions? : "), wxDefaultPosition, wxDefaultSize, 0 );
	m_staticText21411->Wrap( -1 );
	bSizer14711->Add( m_staticText21411, 0, wxALIGN_CENTER_VERTICAL|wxALL, 5 );
	
	wxBoxSizer* bSizer367;
	bSizer367 = new wxBoxSizer( wxHORIZONTAL );
	
	RemoveDuplicateYesButton = new wxRadioButton( this, wxID_ANY, wxT("Yes"), wxDefaultPosition, wxDefaultSize, 0 );
	bSizer367->Add( RemoveDuplicateYesButton, 0, wxALL, 5 );
	
	m_radioBtn40 = new wxRadioButton( this, wxID_ANY, wxT("No"), wxDefaultPosition, wxDefaultSize, 0 );
	bSizer367->Add( m_radioBtn40, 0, wxALL, 5 );
	
	
	bSizer14711->Add( bSizer367, 1, wxEXPAND, 5 );
	
	
	bSizer15311->Add( bSizer14711, 0, wxEXPAND, 5 );
	
	
	bSizer15311->Add( 0, 0, 1, wxEXPAND, 5 );
	
	InfoText = new AutoWrapStaticText( this, wxID_ANY, wxT("Do you want to remove duplicate (within a certain distance) picks?  After re-centring based on 2D classification it can happen that multiple different picks are moved onto the same point. In this case, selecting yes here will keep only one of these picks and help reduce having multiple copies of the same particle."), wxDefaultPosition, wxDefaultSize, 0 );
	InfoText->Wrap( -1 );
	bSizer15311->Add( InfoText, 0, wxALL|wxEXPAND, 5 );
	
	
	this->SetSizer( bSizer15311 );
	this->Layout();
}

RemoveDuplicatesWizardPanel::~RemoveDuplicatesWizardPanel()
{
}

RemoveDuplicateThresholdWizardPanel::RemoveDuplicateThresholdWizardPanel( wxWindow* parent, wxWindowID id, const wxPoint& pos, const wxSize& size, long style ) : wxPanel( parent, id, pos, size, style )
{
	wxBoxSizer* bSizer153;
	bSizer153 = new wxBoxSizer( wxVERTICAL );
	
	wxBoxSizer* bSizer147;
	bSizer147 = new wxBoxSizer( wxHORIZONTAL );
	
	m_staticText214 = new wxStaticText( this, wxID_ANY, wxT("Duplicate Pick Threshold Distance (Å) :"), wxDefaultPosition, wxDefaultSize, 0 );
	m_staticText214->Wrap( -1 );
	bSizer147->Add( m_staticText214, 0, wxALIGN_CENTER_VERTICAL|wxALL, 5 );
	
	DuplicatePickThresholdTextCtrl = new NumericTextCtrl( this, wxID_ANY, wxT("0.0"), wxDefaultPosition, wxDefaultSize, 0 );
	bSizer147->Add( DuplicatePickThresholdTextCtrl, 1, wxALL, 5 );
	
	
	bSizer153->Add( bSizer147, 0, wxEXPAND, 5 );
	
	
	bSizer153->Add( 0, 0, 1, wxEXPAND, 5 );
	
	InfoText = new AutoWrapStaticText( this, wxID_ANY, wxT("When removing duplicate picks, picks within this distance will be counted as duplicate and thus removed.  Typically this would be around 1/3rd to 1/2 the size of your particle."), wxDefaultPosition, wxDefaultSize, 0 );
	InfoText->Wrap( -1 );
	bSizer153->Add( InfoText, 0, wxALL|wxEXPAND, 5 );
	
	
	this->SetSizer( bSizer153 );
	this->Layout();
}

RemoveDuplicateThresholdWizardPanel::~RemoveDuplicateThresholdWizardPanel()
{
}

ClassesSetupWizardPanelB::ClassesSetupWizardPanelB( wxWindow* parent, wxWindowID id, const wxPoint& pos, const wxSize& size, long style ) : wxPanel( parent, id, pos, size, style )
{
	wxBoxSizer* bSizer15311;
	bSizer15311 = new wxBoxSizer( wxVERTICAL );
	
	wxBoxSizer* bSizer14711;
	bSizer14711 = new wxBoxSizer( wxVERTICAL );
	
	m_staticText21411 = new wxStaticText( this, wxID_ANY, wxT("Carry over particles from which classes? :-"), wxDefaultPosition, wxDefaultSize, 0 );
	m_staticText21411->Wrap( -1 );
	bSizer14711->Add( m_staticText21411, 0, wxALIGN_CENTER_VERTICAL|wxALL, 5 );
	
	m_staticline103 = new wxStaticLine( this, wxID_ANY, wxDefaultPosition, wxDefaultSize, wxLI_HORIZONTAL );
	bSizer14711->Add( m_staticline103, 0, wxEXPAND | wxALL, 5 );
	
	ClassListCtrl = new wxListCtrl( this, wxID_ANY, wxDefaultPosition, wxDefaultSize, wxLC_REPORT );
	bSizer14711->Add( ClassListCtrl, 10, wxALL|wxEXPAND, 5 );
	
	
	bSizer15311->Add( bSizer14711, 100, wxEXPAND, 5 );
	
	
	bSizer15311->Add( 0, 0, 1, wxEXPAND, 5 );
	
	InfoText = new AutoWrapStaticText( this, wxID_ANY, wxT("Only particles from the selected classes will be carried over to the new refinement package.  At least one class must be selected, multiple selections are allowed (with CTRL/SHIFT)."), wxDefaultPosition, wxDefaultSize, 0 );
	InfoText->Wrap( -1 );
	bSizer15311->Add( InfoText, 0, wxALL|wxEXPAND, 5 );
	
	
	this->SetSizer( bSizer15311 );
	this->Layout();
}

ClassesSetupWizardPanelB::~ClassesSetupWizardPanelB()
{
}

ClassesSetupWizardPanelC::ClassesSetupWizardPanelC( wxWindow* parent, wxWindowID id, const wxPoint& pos, const wxSize& size, long style ) : wxPanel( parent, id, pos, size, style )
{
	wxBoxSizer* bSizer15311;
	bSizer15311 = new wxBoxSizer( wxVERTICAL );
	
	wxBoxSizer* bSizer14711;
	bSizer14711 = new wxBoxSizer( wxVERTICAL );
	
	m_staticText21411 = new wxStaticText( this, wxID_ANY, wxT("How should the new classes be created? :-"), wxDefaultPosition, wxDefaultSize, 0 );
	m_staticText21411->Wrap( -1 );
	bSizer14711->Add( m_staticText21411, 0, wxALIGN_CENTER_VERTICAL|wxALL, 5 );
	
	m_staticline104 = new wxStaticLine( this, wxID_ANY, wxDefaultPosition, wxDefaultSize, wxLI_HORIZONTAL );
	bSizer14711->Add( m_staticline104, 0, wxEXPAND | wxALL, 5 );
	
	wxBoxSizer* bSizer378;
	bSizer378 = new wxBoxSizer( wxHORIZONTAL );
	
	wxBoxSizer* bSizer391;
	bSizer391 = new wxBoxSizer( wxVERTICAL );
	
	NewClassListCtrl = new wxListCtrl( this, wxID_ANY, wxDefaultPosition, wxDefaultSize, wxLC_REPORT|wxLC_SINGLE_SEL );
	NewClassListCtrl->SetMinSize( wxSize( 10,-1 ) );
	
	bSizer391->Add( NewClassListCtrl, 100, wxALL|wxEXPAND, 5 );
	
	
	bSizer378->Add( bSizer391, 1, wxEXPAND, 5 );
	
	wxBoxSizer* bSizer392;
	bSizer392 = new wxBoxSizer( wxVERTICAL );
	
	OldClassListCtrl = new wxListCtrl( this, wxID_ANY, wxDefaultPosition, wxDefaultSize, wxLC_REPORT );
	OldClassListCtrl->SetMinSize( wxSize( 10,-1 ) );
	
	bSizer392->Add( OldClassListCtrl, 1, wxALL|wxEXPAND, 5 );
	
	
	bSizer378->Add( bSizer392, 1, wxEXPAND, 5 );
	
	
	bSizer14711->Add( bSizer378, 1, wxEXPAND, 5 );
	
	
	bSizer15311->Add( bSizer14711, 100, wxEXPAND, 5 );
	
	
	bSizer15311->Add( 0, 0, 1, wxEXPAND, 5 );
	
	InfoText = new AutoWrapStaticText( this, wxID_ANY, wxT("For each new class (Left) select class(es) to copy parameters from.  If two or more classes are selected (e.g. to merge similar classes), parameters can be taken either from the class with the highest occupancy, or from a random class. At least one class must be selected, multiple selections are allowed (with CTRL/SHIFT).  Note : You are selecting which parameters to copy over, all particles are represented in each class. "), wxDefaultPosition, wxDefaultSize, 0 );
	InfoText->Wrap( -1 );
	bSizer15311->Add( InfoText, 0, wxALL|wxEXPAND, 5 );
	
	
	this->SetSizer( bSizer15311 );
	this->Layout();
}

ClassesSetupWizardPanelC::~ClassesSetupWizardPanelC()
{
}

ClassesSetupWizardPanelD::ClassesSetupWizardPanelD( wxWindow* parent, wxWindowID id, const wxPoint& pos, const wxSize& size, long style ) : wxPanel( parent, id, pos, size, style )
{
	wxBoxSizer* bSizer15311;
	bSizer15311 = new wxBoxSizer( wxVERTICAL );
	
	wxBoxSizer* bSizer14711;
	bSizer14711 = new wxBoxSizer( wxVERTICAL );
	
	m_staticText21411 = new wxStaticText( this, wxID_ANY, wxT("Which parameters should be used when combining classes? :- "), wxDefaultPosition, wxDefaultSize, 0 );
	m_staticText21411->Wrap( -1 );
	bSizer14711->Add( m_staticText21411, 0, wxALIGN_CENTER_VERTICAL|wxALL, 5 );
	
	wxBoxSizer* bSizer367;
	bSizer367 = new wxBoxSizer( wxHORIZONTAL );
	
	BestOccupancyRadioButton = new wxRadioButton( this, wxID_ANY, wxT("Best Occupancy"), wxDefaultPosition, wxDefaultSize, 0 );
	bSizer367->Add( BestOccupancyRadioButton, 0, wxALL, 5 );
	
	m_radioBtn40 = new wxRadioButton( this, wxID_ANY, wxT("Random"), wxDefaultPosition, wxDefaultSize, 0 );
	bSizer367->Add( m_radioBtn40, 0, wxALL, 5 );
	
	
	bSizer14711->Add( bSizer367, 1, wxEXPAND, 5 );
	
	
	bSizer15311->Add( bSizer14711, 0, wxEXPAND, 5 );
	
	
	bSizer15311->Add( 0, 0, 1, wxEXPAND, 5 );
	
	InfoText = new AutoWrapStaticText( this, wxID_ANY, wxT("In the case where a new class will be created by merging parameters from one or more classes, do you want to take the parameters from the class with the highest occupancy, or the parameters from a random class."), wxDefaultPosition, wxDefaultSize, 0 );
	InfoText->Wrap( -1 );
	bSizer15311->Add( InfoText, 0, wxALL|wxEXPAND, 5 );
	
	
	this->SetSizer( bSizer15311 );
	this->Layout();
}

ClassesSetupWizardPanelD::~ClassesSetupWizardPanelD()
{
}

ClassesSetupWizardPanelE::ClassesSetupWizardPanelE( wxWindow* parent, wxWindowID id, const wxPoint& pos, const wxSize& size, long style ) : wxPanel( parent, id, pos, size, style )
{
	wxBoxSizer* bSizer15311;
	bSizer15311 = new wxBoxSizer( wxVERTICAL );
	
	wxBoxSizer* bSizer14711;
	bSizer14711 = new wxBoxSizer( wxVERTICAL );
	
	m_staticText21411 = new wxStaticText( this, wxID_ANY, wxT("How should the new occupancies be set? :- "), wxDefaultPosition, wxDefaultSize, 0 );
	m_staticText21411->Wrap( -1 );
	bSizer14711->Add( m_staticText21411, 0, wxALIGN_CENTER_VERTICAL|wxALL, 5 );
	
	wxBoxSizer* bSizer367;
	bSizer367 = new wxBoxSizer( wxHORIZONTAL );
	
	RandomiseOccupanciesRadioButton = new wxRadioButton( this, wxID_ANY, wxT("Random occupancies"), wxDefaultPosition, wxDefaultSize, 0 );
	bSizer367->Add( RandomiseOccupanciesRadioButton, 0, wxALL, 5 );
	
	m_radioBtn40 = new wxRadioButton( this, wxID_ANY, wxT("Keep input occupancies"), wxDefaultPosition, wxDefaultSize, 0 );
	bSizer367->Add( m_radioBtn40, 0, wxALL, 5 );
	
	
	bSizer14711->Add( bSizer367, 1, wxEXPAND, 5 );
	
	
	bSizer15311->Add( bSizer14711, 0, wxEXPAND, 5 );
	
	
	bSizer15311->Add( 0, 0, 1, wxEXPAND, 5 );
	
	InfoText = new AutoWrapStaticText( this, wxID_ANY, wxT("Should the new occupancies be randomised, or should the input occupancies be kept. Randomising the occupanices is a way of creating new \"seed\" references. For example, a refinement done with 1 class, could be classified into multiple classes by creating the new classes with parameters from the single class and randomising the occupancies.  This will lead to small differences which will hopefully converge to an accurate 3D classification after a number of rounds of additional refinement."), wxDefaultPosition, wxDefaultSize, 0 );
	InfoText->Wrap( -1 );
	bSizer15311->Add( InfoText, 0, wxALL|wxEXPAND, 5 );
	
	
	this->SetSizer( bSizer15311 );
	this->Layout();
}

ClassesSetupWizardPanelE::~ClassesSetupWizardPanelE()
{
}

FSCPanel::FSCPanel( wxWindow* parent, wxWindowID id, const wxPoint& pos, const wxSize& size, long style ) : wxPanel( parent, id, pos, size, style )
{
	wxBoxSizer* bSizer200;
	bSizer200 = new wxBoxSizer( wxVERTICAL );
	
	TitleSizer = new wxBoxSizer( wxHORIZONTAL );
	
	TitleStaticText = new wxStaticText( this, wxID_ANY, wxT("Current FSC"), wxDefaultPosition, wxDefaultSize, 0 );
	TitleStaticText->Wrap( -1 );
	TitleStaticText->SetFont( wxFont( 10, 74, 90, 92, false, wxT("Sans") ) );
	
	TitleSizer->Add( TitleStaticText, 0, wxALIGN_BOTTOM|wxALIGN_CENTER_VERTICAL|wxALL, 5 );
	
	
	TitleSizer->Add( 0, 0, 1, wxEXPAND, 5 );
	
	EstimatedResolutionLabel = new wxStaticText( this, wxID_ANY, wxT("Best Est. Resolution : "), wxDefaultPosition, wxDefaultSize, 0 );
	EstimatedResolutionLabel->Wrap( -1 );
	TitleSizer->Add( EstimatedResolutionLabel, 0, wxALIGN_BOTTOM|wxALIGN_CENTER_VERTICAL|wxALL, 5 );
	
	EstimatedResolutionText = new wxStaticText( this, wxID_ANY, wxT("00.00  ‎Å"), wxDefaultPosition, wxDefaultSize, 0 );
	EstimatedResolutionText->Wrap( -1 );
	TitleSizer->Add( EstimatedResolutionText, 0, wxALIGN_BOTTOM|wxALIGN_CENTER_VERTICAL|wxALL, 5 );
	
	m_staticline104 = new wxStaticLine( this, wxID_ANY, wxDefaultPosition, wxDefaultSize, wxLI_HORIZONTAL|wxLI_VERTICAL );
	TitleSizer->Add( m_staticline104, 0, wxEXPAND | wxALL, 5 );
	
	wxBoxSizer* bSizer505;
	bSizer505 = new wxBoxSizer( wxHORIZONTAL );
	
	SaveButton = new NoFocusBitmapButton( this, wxID_ANY, wxNullBitmap, wxDefaultPosition, wxDefaultSize, wxBU_AUTODRAW );
	SaveButton->SetDefault(); 
	bSizer505->Add( SaveButton, 1, wxEXPAND|wxLEFT, 5 );
	
	FSCDetailsButton = new NoFocusBitmapButton( this, wxID_ANY, wxNullBitmap, wxDefaultPosition, wxDefaultSize, wxBU_AUTODRAW );
	FSCDetailsButton->SetDefault(); 
	bSizer505->Add( FSCDetailsButton, 1, wxEXPAND|wxRIGHT, 5 );
	
	
	TitleSizer->Add( bSizer505, 0, wxEXPAND, 5 );
	
	
	bSizer200->Add( TitleSizer, 0, wxEXPAND, 5 );
	
	m_staticline52 = new wxStaticLine( this, wxID_ANY, wxDefaultPosition, wxDefaultSize, wxLI_HORIZONTAL );
	bSizer200->Add( m_staticline52, 0, wxEXPAND | wxALL, 5 );
	
	PlotPanel = new PlotFSCPanel( this, wxID_ANY, wxDefaultPosition, wxDefaultSize, wxTAB_TRAVERSAL );
	bSizer200->Add( PlotPanel, 1, wxEXPAND | wxALL, 5 );
	
	
	this->SetSizer( bSizer200 );
	this->Layout();
	
	// Connect Events
	SaveButton->Connect( wxEVT_COMMAND_BUTTON_CLICKED, wxCommandEventHandler( FSCPanel::SaveImageClick ), NULL, this );
	FSCDetailsButton->Connect( wxEVT_COMMAND_BUTTON_CLICKED, wxCommandEventHandler( FSCPanel::PopupTextClick ), NULL, this );
}

FSCPanel::~FSCPanel()
{
	// Disconnect Events
	SaveButton->Disconnect( wxEVT_COMMAND_BUTTON_CLICKED, wxCommandEventHandler( FSCPanel::SaveImageClick ), NULL, this );
	FSCDetailsButton->Disconnect( wxEVT_COMMAND_BUTTON_CLICKED, wxCommandEventHandler( FSCPanel::PopupTextClick ), NULL, this );
	
}

DisplayPanelParent::DisplayPanelParent( wxWindow* parent, wxWindowID id, const wxPoint& pos, const wxSize& size, long style ) : wxPanel( parent, id, pos, size, style )
{
	MainSizer = new wxBoxSizer( wxVERTICAL );
	
	Toolbar = new wxToolBar( this, wxID_ANY, wxDefaultPosition, wxDefaultSize, wxTB_FLAT|wxTB_HORIZONTAL ); 
	Toolbar->Realize(); 
	
	MainSizer->Add( Toolbar, 0, wxEXPAND, 5 );
	
	
	this->SetSizer( MainSizer );
	this->Layout();
}

DisplayPanelParent::~DisplayPanelParent()
{
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
	
	minimum_text_ctrl = new wxTextCtrl( this, wxID_ANY, wxEmptyString, wxDefaultPosition, wxDefaultSize, wxTE_CENTRE|wxTE_PROCESS_ENTER );
	bSizer262->Add( minimum_text_ctrl, 0, wxALL, 5 );
	
	m_staticText316 = new wxStaticText( this, wxID_ANY, wxT("/"), wxDefaultPosition, wxDefaultSize, 0 );
	m_staticText316->Wrap( -1 );
	bSizer262->Add( m_staticText316, 0, wxALIGN_CENTER_VERTICAL|wxALL, 5 );
	
	maximum_text_ctrl = new wxTextCtrl( this, wxID_ANY, wxEmptyString, wxDefaultPosition, wxDefaultSize, wxTE_CENTRE|wxTE_PROCESS_ENTER );
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

ClassificationPlotPanelParent::ClassificationPlotPanelParent( wxWindow* parent, wxWindowID id, const wxPoint& pos, const wxSize& size, long style ) : wxPanel( parent, id, pos, size, style )
{
	wxBoxSizer* bSizer278;
	bSizer278 = new wxBoxSizer( wxVERTICAL );
	
	my_notebook = new wxAuiNotebook( this, wxID_ANY, wxDefaultPosition, wxDefaultSize, 0 );
	SigmaPanel = new wxPanel( my_notebook, wxID_ANY, wxDefaultPosition, wxDefaultSize, wxTAB_TRAVERSAL );
	my_notebook->AddPage( SigmaPanel, wxT("Sigma"), false, wxNullBitmap );
	LikelihoodPanel = new wxPanel( my_notebook, wxID_ANY, wxDefaultPosition, wxDefaultSize, wxTAB_TRAVERSAL );
	my_notebook->AddPage( LikelihoodPanel, wxT("Likelihood"), false, wxNullBitmap );
	MobilityPanel = new wxPanel( my_notebook, wxID_ANY, wxDefaultPosition, wxDefaultSize, wxTAB_TRAVERSAL );
	my_notebook->AddPage( MobilityPanel, wxT("Image Mobility"), true, wxNullBitmap );
	
	bSizer278->Add( my_notebook, 1, wxEXPAND | wxALL, 5 );
	
	
	this->SetSizer( bSizer278 );
	this->Layout();
}

ClassificationPlotPanelParent::~ClassificationPlotPanelParent()
{
}

AbInitioPlotPanelParent::AbInitioPlotPanelParent( wxWindow* parent, wxWindowID id, const wxPoint& pos, const wxSize& size, long style ) : wxPanel( parent, id, pos, size, style )
{
	wxBoxSizer* bSizer278;
	bSizer278 = new wxBoxSizer( wxVERTICAL );
	
	my_notebook = new wxAuiNotebook( this, wxID_ANY, wxDefaultPosition, wxDefaultSize, 0 );
	SigmaPanel = new wxPanel( my_notebook, wxID_ANY, wxDefaultPosition, wxDefaultSize, wxTAB_TRAVERSAL );
	my_notebook->AddPage( SigmaPanel, wxT("Sigma"), false, wxNullBitmap );
	
	bSizer278->Add( my_notebook, 1, wxEXPAND | wxALL, 5 );
	
	
	this->SetSizer( bSizer278 );
	this->Layout();
}

AbInitioPlotPanelParent::~AbInitioPlotPanelParent()
{
}

ClassumSelectionCopyFromDialogParent::ClassumSelectionCopyFromDialogParent( wxWindow* parent, wxWindowID id, const wxString& title, const wxPoint& pos, const wxSize& size, long style ) : wxDialog( parent, id, title, pos, size, style )
{
	this->SetSizeHints( wxDefaultSize, wxDefaultSize );
	
	wxBoxSizer* bSizer279;
	bSizer279 = new wxBoxSizer( wxVERTICAL );
	
	m_staticText414 = new wxStaticText( this, wxID_ANY, wxT("Please select a valid (same no. classums) selection :-"), wxDefaultPosition, wxDefaultSize, 0 );
	m_staticText414->Wrap( -1 );
	bSizer279->Add( m_staticText414, 0, wxALL, 5 );
	
	m_staticline72 = new wxStaticLine( this, wxID_ANY, wxDefaultPosition, wxDefaultSize, wxLI_HORIZONTAL );
	bSizer279->Add( m_staticline72, 0, wxEXPAND | wxALL, 5 );
	
	SelectionListCtrl = new wxListCtrl( this, wxID_ANY, wxDefaultPosition, wxSize( -1,100 ), wxLC_REPORT|wxLC_SINGLE_SEL );
	bSizer279->Add( SelectionListCtrl, 0, wxALL|wxEXPAND, 5 );
	
	m_staticline71 = new wxStaticLine( this, wxID_ANY, wxDefaultPosition, wxDefaultSize, wxLI_HORIZONTAL );
	bSizer279->Add( m_staticline71, 0, wxEXPAND | wxALL, 5 );
	
	wxBoxSizer* bSizer280;
	bSizer280 = new wxBoxSizer( wxHORIZONTAL );
	
	
	bSizer280->Add( 0, 0, 1, wxEXPAND, 5 );
	
	OkButton = new wxButton( this, wxID_ANY, wxT("OK"), wxDefaultPosition, wxDefaultSize, 0 );
	bSizer280->Add( OkButton, 0, wxALL, 5 );
	
	CancelButton = new wxButton( this, wxID_ANY, wxT("Cancel"), wxDefaultPosition, wxDefaultSize, 0 );
	bSizer280->Add( CancelButton, 0, wxALL, 5 );
	
	
	bSizer280->Add( 0, 0, 1, wxEXPAND, 5 );
	
	
	bSizer279->Add( bSizer280, 1, wxEXPAND, 5 );
	
	
	this->SetSizer( bSizer279 );
	this->Layout();
	bSizer279->Fit( this );
	
	this->Centre( wxBOTH );
	
	// Connect Events
	OkButton->Connect( wxEVT_COMMAND_BUTTON_CLICKED, wxCommandEventHandler( ClassumSelectionCopyFromDialogParent::OnOKButtonClick ), NULL, this );
	CancelButton->Connect( wxEVT_COMMAND_BUTTON_CLICKED, wxCommandEventHandler( ClassumSelectionCopyFromDialogParent::OnCancelButtonClick ), NULL, this );
}

ClassumSelectionCopyFromDialogParent::~ClassumSelectionCopyFromDialogParent()
{
	// Disconnect Events
	OkButton->Disconnect( wxEVT_COMMAND_BUTTON_CLICKED, wxCommandEventHandler( ClassumSelectionCopyFromDialogParent::OnOKButtonClick ), NULL, this );
	CancelButton->Disconnect( wxEVT_COMMAND_BUTTON_CLICKED, wxCommandEventHandler( ClassumSelectionCopyFromDialogParent::OnCancelButtonClick ), NULL, this );
	
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

UnblurResultsPanelParent::UnblurResultsPanelParent( wxWindow* parent, wxWindowID id, const wxPoint& pos, const wxSize& size, long style ) : wxPanel( parent, id, pos, size, style )
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
	m_staticText372->SetFont( wxFont( 10, 74, 90, 92, false, wxT("Sans") ) );
	
	bSizer299->Add( m_staticText372, 0, wxALL, 5 );
	
	SpectraNyquistStaticText = new wxStaticText( m_panel82, wxID_ANY, wxT("2.8 Å)"), wxDefaultPosition, wxDefaultSize, 0 );
	SpectraNyquistStaticText->Wrap( -1 );
	SpectraNyquistStaticText->SetFont( wxFont( 10, 74, 90, 92, false, wxT("Sans") ) );
	
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
	m_staticText373->SetFont( wxFont( 10, 74, 90, 92, false, wxT("Sans") ) );
	
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
	m_staticText371->SetFont( wxFont( 10, 74, 90, 92, false, wxT("Sans") ) );
	
	bSizer300->Add( m_staticText371, 0, wxALL, 5 );
	
	FilenameStaticText = new wxStaticText( m_panel81, wxID_ANY, wxEmptyString, wxDefaultPosition, wxDefaultSize, 0 );
	FilenameStaticText->Wrap( -1 );
	FilenameStaticText->SetFont( wxFont( 10, 74, 90, 92, false, wxT("Sans") ) );
	
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

ListCtrlDialog::ListCtrlDialog( wxWindow* parent, wxWindowID id, const wxString& title, const wxPoint& pos, const wxSize& size, long style ) : wxDialog( parent, id, title, pos, size, style )
{
	this->SetSizeHints( wxDefaultSize, wxDefaultSize );
	
	wxBoxSizer* bSizer321;
	bSizer321 = new wxBoxSizer( wxVERTICAL );
	
	MyListCtrl = new AssetPickerListCtrl( this, wxID_ANY, wxDefaultPosition, wxSize( 500,300 ), wxLC_NO_HEADER|wxLC_REPORT|wxLC_SINGLE_SEL|wxLC_VIRTUAL );
	bSizer321->Add( MyListCtrl, 0, wxALL, 5 );
	
	wxBoxSizer* bSizer322;
	bSizer322 = new wxBoxSizer( wxHORIZONTAL );
	
	
	bSizer322->Add( 0, 0, 1, wxEXPAND, 5 );
	
	m_button108 = new wxButton( this, wxID_OK, wxT("OK"), wxDefaultPosition, wxDefaultSize, 0 );
	bSizer322->Add( m_button108, 0, wxALL, 5 );
	
	m_button109 = new wxButton( this, wxID_CANCEL, wxT("Cancel"), wxDefaultPosition, wxDefaultSize, 0 );
	bSizer322->Add( m_button109, 0, wxALL, 5 );
	
	
	bSizer322->Add( 0, 0, 1, wxEXPAND, 5 );
	
	
	bSizer321->Add( bSizer322, 1, wxEXPAND, 5 );
	
	
	this->SetSizer( bSizer321 );
	this->Layout();
	bSizer321->Fit( this );
	
	this->Centre( wxBOTH );
}

ListCtrlDialog::~ListCtrlDialog()
{
}

DisplayRefinementResultsPanelParent::DisplayRefinementResultsPanelParent( wxWindow* parent, wxWindowID id, const wxPoint& pos, const wxSize& size, long style ) : wxPanel( parent, id, pos, size, style )
{
	wxBoxSizer* bSizer92;
	bSizer92 = new wxBoxSizer( wxVERTICAL );
	
	LeftRightSplitter = new wxSplitterWindow( this, wxID_ANY, wxDefaultPosition, wxDefaultSize, wxSP_3D|wxSP_PERMIT_UNSPLIT );
	LeftRightSplitter->SetSashGravity( 0.5 );
	LeftRightSplitter->Connect( wxEVT_IDLE, wxIdleEventHandler( DisplayRefinementResultsPanelParent::LeftRightSplitterOnIdle ), NULL, this );
	
	LeftPanel = new wxPanel( LeftRightSplitter, wxID_ANY, wxDefaultPosition, wxDefaultSize, wxTAB_TRAVERSAL );
	wxBoxSizer* bSizer301;
	bSizer301 = new wxBoxSizer( wxVERTICAL );
	
	TopBottomSplitter = new wxSplitterWindow( LeftPanel, wxID_ANY, wxDefaultPosition, wxDefaultSize, wxSP_3D|wxSP_PERMIT_UNSPLIT );
	TopBottomSplitter->SetSashGravity( 0.5 );
	TopBottomSplitter->Connect( wxEVT_IDLE, wxIdleEventHandler( DisplayRefinementResultsPanelParent::TopBottomSplitterOnIdle ), NULL, this );
	
	TopPanel = new wxPanel( TopBottomSplitter, wxID_ANY, wxDefaultPosition, wxDefaultSize, wxTAB_TRAVERSAL );
	wxBoxSizer* bSizer302;
	bSizer302 = new wxBoxSizer( wxVERTICAL );
	
	AngularPlotText = new wxStaticText( TopPanel, wxID_ANY, wxT("Angular Plot"), wxDefaultPosition, wxDefaultSize, 0 );
	AngularPlotText->Wrap( -1 );
	AngularPlotText->SetFont( wxFont( 10, 74, 90, 92, false, wxT("Sans") ) );
	
	bSizer302->Add( AngularPlotText, 0, wxALL, 5 );
	
	AngularPlotLine = new wxStaticLine( TopPanel, wxID_ANY, wxDefaultPosition, wxDefaultSize, wxLI_HORIZONTAL );
	bSizer302->Add( AngularPlotLine, 0, wxEXPAND | wxALL, 5 );
	
	AngularPlotPanel = new AngularDistributionPlotPanel( TopPanel, wxID_ANY, wxDefaultPosition, wxDefaultSize, wxTAB_TRAVERSAL );
	bSizer302->Add( AngularPlotPanel, 1, wxEXPAND | wxALL, 5 );
	
	
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
	OrthText->SetFont( wxFont( 10, 74, 90, 92, false, wxT("Sans") ) );
	
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

DisplayRefinementResultsPanelParent::~DisplayRefinementResultsPanelParent()
{
}

PopupTextDialogParent::PopupTextDialogParent( wxWindow* parent, wxWindowID id, const wxString& title, const wxPoint& pos, const wxSize& size, long style ) : wxDialog( parent, id, title, pos, size, style )
{
	this->SetSizeHints( wxSize( 600,600 ), wxDefaultSize );
	
	wxBoxSizer* bSizer363;
	bSizer363 = new wxBoxSizer( wxVERTICAL );
	
	OutputTextCtrl = new wxTextCtrl( this, wxID_ANY, wxEmptyString, wxDefaultPosition, wxDefaultSize, wxHSCROLL|wxTE_MULTILINE|wxTE_READONLY );
	OutputTextCtrl->SetFont( wxFont( wxNORMAL_FONT->GetPointSize(), 70, 90, 90, false, wxT("Fixed") ) );
	
	bSizer363->Add( OutputTextCtrl, 1, wxALL|wxEXPAND, 5 );
	
	wxBoxSizer* bSizer380;
	bSizer380 = new wxBoxSizer( wxHORIZONTAL );
	
	CloseButton = new wxButton( this, wxID_CLOSE, wxT("Close"), wxDefaultPosition, wxDefaultSize, 0 );
	bSizer380->Add( CloseButton, 0, wxALL, 5 );
	
	
	bSizer380->Add( 0, 0, 1, wxEXPAND, 5 );
	
	ClipBoardButton = new wxButton( this, wxID_COPY, wxT("Copy All"), wxDefaultPosition, wxDefaultSize, 0 );
	bSizer380->Add( ClipBoardButton, 0, wxALL, 5 );
	
	m_button146 = new wxButton( this, wxID_SAVE, wxT("Save"), wxDefaultPosition, wxDefaultSize, 0 );
	bSizer380->Add( m_button146, 0, wxALL, 5 );
	
	
	bSizer363->Add( bSizer380, 0, wxEXPAND, 5 );
	
	
	this->SetSizer( bSizer363 );
	this->Layout();
	bSizer363->Fit( this );
	
	this->Centre( wxBOTH );
	
	// Connect Events
	CloseButton->Connect( wxEVT_COMMAND_BUTTON_CLICKED, wxCommandEventHandler( PopupTextDialogParent::OnCloseButtonClick ), NULL, this );
	ClipBoardButton->Connect( wxEVT_COMMAND_BUTTON_CLICKED, wxCommandEventHandler( PopupTextDialogParent::OnCopyToClipboardClick ), NULL, this );
	m_button146->Connect( wxEVT_COMMAND_BUTTON_CLICKED, wxCommandEventHandler( PopupTextDialogParent::OnSaveButtonClick ), NULL, this );
}

PopupTextDialogParent::~PopupTextDialogParent()
{
	// Disconnect Events
	CloseButton->Disconnect( wxEVT_COMMAND_BUTTON_CLICKED, wxCommandEventHandler( PopupTextDialogParent::OnCloseButtonClick ), NULL, this );
	ClipBoardButton->Disconnect( wxEVT_COMMAND_BUTTON_CLICKED, wxCommandEventHandler( PopupTextDialogParent::OnCopyToClipboardClick ), NULL, this );
	m_button146->Disconnect( wxEVT_COMMAND_BUTTON_CLICKED, wxCommandEventHandler( PopupTextDialogParent::OnSaveButtonClick ), NULL, this );
	
}

LargeAngularPlotDialogParent::LargeAngularPlotDialogParent( wxWindow* parent, wxWindowID id, const wxString& title, const wxPoint& pos, const wxSize& size, long style ) : wxDialog( parent, id, title, pos, size, style )
{
	this->SetSizeHints( wxSize( 1000,800 ), wxDefaultSize );
	
	wxBoxSizer* bSizer363;
	bSizer363 = new wxBoxSizer( wxVERTICAL );
	
	AngularPlotPanel = new AngularDistributionPlotPanel( this, wxID_ANY, wxDefaultPosition, wxDefaultSize, wxTAB_TRAVERSAL );
	bSizer363->Add( AngularPlotPanel, 1, wxEXPAND | wxALL, 5 );
	
	wxBoxSizer* bSizer380;
	bSizer380 = new wxBoxSizer( wxHORIZONTAL );
	
	CloseButton = new wxButton( this, wxID_CLOSE, wxT("Close"), wxDefaultPosition, wxDefaultSize, 0 );
	bSizer380->Add( CloseButton, 0, wxALL, 5 );
	
	
	bSizer380->Add( 0, 0, 1, wxEXPAND, 5 );
	
	ClipBoardButton = new wxButton( this, wxID_COPY, wxT("Copy"), wxDefaultPosition, wxDefaultSize, 0 );
	bSizer380->Add( ClipBoardButton, 0, wxALL, 5 );
	
	SaveButton = new wxButton( this, wxID_SAVE, wxT("Save"), wxDefaultPosition, wxDefaultSize, 0 );
	bSizer380->Add( SaveButton, 0, wxALL, 5 );
	
	
	bSizer363->Add( bSizer380, 0, wxEXPAND, 5 );
	
	
	this->SetSizer( bSizer363 );
	this->Layout();
	
	this->Centre( wxBOTH );
	
	// Connect Events
	CloseButton->Connect( wxEVT_COMMAND_BUTTON_CLICKED, wxCommandEventHandler( LargeAngularPlotDialogParent::OnCloseButtonClick ), NULL, this );
	ClipBoardButton->Connect( wxEVT_COMMAND_BUTTON_CLICKED, wxCommandEventHandler( LargeAngularPlotDialogParent::OnCopyToClipboardClick ), NULL, this );
	SaveButton->Connect( wxEVT_COMMAND_BUTTON_CLICKED, wxCommandEventHandler( LargeAngularPlotDialogParent::OnSaveButtonClick ), NULL, this );
}

LargeAngularPlotDialogParent::~LargeAngularPlotDialogParent()
{
	// Disconnect Events
	CloseButton->Disconnect( wxEVT_COMMAND_BUTTON_CLICKED, wxCommandEventHandler( LargeAngularPlotDialogParent::OnCloseButtonClick ), NULL, this );
	ClipBoardButton->Disconnect( wxEVT_COMMAND_BUTTON_CLICKED, wxCommandEventHandler( LargeAngularPlotDialogParent::OnCopyToClipboardClick ), NULL, this );
	SaveButton->Disconnect( wxEVT_COMMAND_BUTTON_CLICKED, wxCommandEventHandler( LargeAngularPlotDialogParent::OnSaveButtonClick ), NULL, this );
	
}

RefinementParametersDialogParent::RefinementParametersDialogParent( wxWindow* parent, wxWindowID id, const wxString& title, const wxPoint& pos, const wxSize& size, long style ) : wxDialog( parent, id, title, pos, size, style )
{
	this->SetSizeHints( wxSize( 1000,800 ), wxDefaultSize );
	
	wxBoxSizer* bSizer363;
	bSizer363 = new wxBoxSizer( wxVERTICAL );
	
	ClassToolBar = new wxToolBar( this, wxID_ANY, wxDefaultPosition, wxDefaultSize, wxTB_HORIZONTAL|wxTB_NOICONS|wxTB_TEXT ); 
	m_staticText831 = new wxStaticText( ClassToolBar, wxID_ANY, wxT("Select Class : "), wxDefaultPosition, wxDefaultSize, 0 );
	m_staticText831->Wrap( -1 );
	ClassToolBar->AddControl( m_staticText831 );
	ClassToolBar->Realize(); 
	
	bSizer363->Add( ClassToolBar, 0, wxEXPAND, 5 );
	
	m_staticline137 = new wxStaticLine( this, wxID_ANY, wxDefaultPosition, wxDefaultSize, wxLI_HORIZONTAL );
	bSizer363->Add( m_staticline137, 0, wxEXPAND | wxALL, 5 );
	
	ParameterListCtrl = new RefinementParametersListCtrl( this, wxID_ANY, wxDefaultPosition, wxDefaultSize, wxLC_REPORT|wxLC_VIRTUAL|wxLC_VRULES );
	bSizer363->Add( ParameterListCtrl, 1, wxALL|wxEXPAND, 5 );
	
	wxBoxSizer* bSizer380;
	bSizer380 = new wxBoxSizer( wxHORIZONTAL );
	
	CloseButton = new wxButton( this, wxID_CLOSE, wxT("Close"), wxDefaultPosition, wxDefaultSize, 0 );
	bSizer380->Add( CloseButton, 0, wxALL, 5 );
	
	
	bSizer380->Add( 0, 0, 1, wxEXPAND, 5 );
	
	SaveButton = new wxButton( this, wxID_SAVE, wxT("Save"), wxDefaultPosition, wxDefaultSize, 0 );
	bSizer380->Add( SaveButton, 0, wxALL, 5 );
	
	
	bSizer363->Add( bSizer380, 0, wxEXPAND, 5 );
	
	
	this->SetSizer( bSizer363 );
	this->Layout();
	
	this->Centre( wxBOTH );
	
	// Connect Events
	CloseButton->Connect( wxEVT_COMMAND_BUTTON_CLICKED, wxCommandEventHandler( RefinementParametersDialogParent::OnCloseButtonClick ), NULL, this );
	SaveButton->Connect( wxEVT_COMMAND_BUTTON_CLICKED, wxCommandEventHandler( RefinementParametersDialogParent::OnSaveButtonClick ), NULL, this );
}

RefinementParametersDialogParent::~RefinementParametersDialogParent()
{
	// Disconnect Events
	CloseButton->Disconnect( wxEVT_COMMAND_BUTTON_CLICKED, wxCommandEventHandler( RefinementParametersDialogParent::OnCloseButtonClick ), NULL, this );
	SaveButton->Disconnect( wxEVT_COMMAND_BUTTON_CLICKED, wxCommandEventHandler( RefinementParametersDialogParent::OnSaveButtonClick ), NULL, this );
	
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

DistributionPlotDialogParent::DistributionPlotDialogParent( wxWindow* parent, wxWindowID id, const wxString& title, const wxPoint& pos, const wxSize& size, long style ) : wxDialog( parent, id, title, pos, size, style )
{
	this->SetSizeHints( wxDefaultSize, wxDefaultSize );
	
	wxBoxSizer* bSizer489;
	bSizer489 = new wxBoxSizer( wxVERTICAL );
	
	wxBoxSizer* bSizer493;
	bSizer493 = new wxBoxSizer( wxHORIZONTAL );
	
	wxBoxSizer* bSizer497;
	bSizer497 = new wxBoxSizer( wxVERTICAL );
	
	
	bSizer497->Add( 0, 0, 1, wxEXPAND, 5 );
	
	m_staticText64711 = new wxStaticText( this, wxID_ANY, wxT("Max"), wxDefaultPosition, wxDefaultSize, 0 );
	m_staticText64711->Wrap( -1 );
	bSizer497->Add( m_staticText64711, 0, wxALL, 5 );
	
	UpperBoundYNumericCtrl = new NumericTextCtrl( this, wxID_ANY, wxT("0.0"), wxDefaultPosition, wxDefaultSize, wxTE_PROCESS_ENTER );
	bSizer497->Add( UpperBoundYNumericCtrl, 0, wxALL, 5 );
	
	m_staticText6472 = new wxStaticText( this, wxID_ANY, wxT("Min"), wxDefaultPosition, wxDefaultSize, 0 );
	m_staticText6472->Wrap( -1 );
	bSizer497->Add( m_staticText6472, 0, wxALL, 5 );
	
	LowerBoundYNumericCtrl = new NumericTextCtrl( this, wxID_ANY, wxT("0.0"), wxDefaultPosition, wxDefaultSize, wxTE_PROCESS_ENTER );
	bSizer497->Add( LowerBoundYNumericCtrl, 0, wxALL, 5 );
	
	
	bSizer493->Add( bSizer497, 0, wxEXPAND, 5 );
	
	PlotCurvePanelInstance = new PlotCurvePanel( this, wxID_ANY, wxDefaultPosition, wxDefaultSize, wxTAB_TRAVERSAL );
	bSizer493->Add( PlotCurvePanelInstance, 1, wxEXPAND | wxALL, 5 );
	
	
	bSizer489->Add( bSizer493, 1, wxEXPAND, 5 );
	
	wxBoxSizer* DistributionPlotButtonsSizer;
	DistributionPlotButtonsSizer = new wxBoxSizer( wxHORIZONTAL );
	
	wxArrayString DataSeriesToPlotChoiceChoices;
	DataSeriesToPlotChoice = new wxChoice( this, wxID_ANY, wxDefaultPosition, wxDefaultSize, DataSeriesToPlotChoiceChoices, 0 );
	DataSeriesToPlotChoice->SetSelection( 0 );
	DistributionPlotButtonsSizer->Add( DataSeriesToPlotChoice, 0, wxALL|wxEXPAND, 5 );
	
	wxBoxSizer* bSizer492;
	bSizer492 = new wxBoxSizer( wxHORIZONTAL );
	
	m_staticText647 = new wxStaticText( this, wxID_ANY, wxT("Min"), wxDefaultPosition, wxDefaultSize, 0 );
	m_staticText647->Wrap( -1 );
	bSizer492->Add( m_staticText647, 0, wxALIGN_CENTER_VERTICAL|wxALL, 5 );
	
	LowerBoundXNumericCtrl = new NumericTextCtrl( this, wxID_ANY, wxT("0.0"), wxDefaultPosition, wxDefaultSize, wxTE_PROCESS_ENTER );
	bSizer492->Add( LowerBoundXNumericCtrl, 0, wxALIGN_CENTER_VERTICAL|wxALL, 5 );
	
	m_staticText6471 = new wxStaticText( this, wxID_ANY, wxT("Max"), wxDefaultPosition, wxDefaultSize, 0 );
	m_staticText6471->Wrap( -1 );
	bSizer492->Add( m_staticText6471, 0, wxALIGN_CENTER_VERTICAL|wxALL, 5 );
	
	UpperBoundXNumericCtrl = new NumericTextCtrl( this, wxID_ANY, wxT("0.0"), wxDefaultPosition, wxDefaultSize, wxTE_PROCESS_ENTER );
	bSizer492->Add( UpperBoundXNumericCtrl, 0, wxALIGN_CENTER_VERTICAL|wxALL, 5 );
	
	
	DistributionPlotButtonsSizer->Add( bSizer492, 1, wxEXPAND, 5 );
	
	
	DistributionPlotButtonsSizer->Add( 0, 0, 1, wxEXPAND, 5 );
	
	
	bSizer489->Add( DistributionPlotButtonsSizer, 0, wxEXPAND, 5 );
	
	wxBoxSizer* bSizer491;
	bSizer491 = new wxBoxSizer( wxHORIZONTAL );
	
	
	bSizer491->Add( 0, 0, 1, wxEXPAND, 5 );
	
	SaveTXTButton = new wxButton( this, wxID_ANY, wxT("Save TXT"), wxDefaultPosition, wxDefaultSize, 0 );
	bSizer491->Add( SaveTXTButton, 0, wxALL, 5 );
	
	SavePNGButton = new wxButton( this, wxID_ANY, wxT("Save PNG"), wxDefaultPosition, wxDefaultSize, 0 );
	bSizer491->Add( SavePNGButton, 0, wxALL, 5 );
	
	
	bSizer489->Add( bSizer491, 0, wxEXPAND, 5 );
	
	
	this->SetSizer( bSizer489 );
	this->Layout();
	
	this->Centre( wxBOTH );
	
	// Connect Events
	UpperBoundYNumericCtrl->Connect( wxEVT_KILL_FOCUS, wxFocusEventHandler( DistributionPlotDialogParent::OnUpperBoundYKillFocus ), NULL, this );
	UpperBoundYNumericCtrl->Connect( wxEVT_SET_FOCUS, wxFocusEventHandler( DistributionPlotDialogParent::OnUpperBoundYSetFocus ), NULL, this );
	UpperBoundYNumericCtrl->Connect( wxEVT_COMMAND_TEXT_ENTER, wxCommandEventHandler( DistributionPlotDialogParent::OnUpperBoundYTextEnter ), NULL, this );
	LowerBoundYNumericCtrl->Connect( wxEVT_KILL_FOCUS, wxFocusEventHandler( DistributionPlotDialogParent::OnLowerBoundYKillFocus ), NULL, this );
	LowerBoundYNumericCtrl->Connect( wxEVT_SET_FOCUS, wxFocusEventHandler( DistributionPlotDialogParent::OnLowerBoundYSetFocus ), NULL, this );
	LowerBoundYNumericCtrl->Connect( wxEVT_COMMAND_TEXT_ENTER, wxCommandEventHandler( DistributionPlotDialogParent::OnLowerBoundYTextEnter ), NULL, this );
	DataSeriesToPlotChoice->Connect( wxEVT_COMMAND_CHOICE_SELECTED, wxCommandEventHandler( DistributionPlotDialogParent::OnDataSeriesToPlotChoice ), NULL, this );
	LowerBoundXNumericCtrl->Connect( wxEVT_KILL_FOCUS, wxFocusEventHandler( DistributionPlotDialogParent::OnLowerBoundXKillFocus ), NULL, this );
	LowerBoundXNumericCtrl->Connect( wxEVT_SET_FOCUS, wxFocusEventHandler( DistributionPlotDialogParent::OnLowerBoundXSetFocus ), NULL, this );
	LowerBoundXNumericCtrl->Connect( wxEVT_COMMAND_TEXT_ENTER, wxCommandEventHandler( DistributionPlotDialogParent::OnLowerBoundXTextEnter ), NULL, this );
	UpperBoundXNumericCtrl->Connect( wxEVT_KILL_FOCUS, wxFocusEventHandler( DistributionPlotDialogParent::OnUpperBoundXKillFocus ), NULL, this );
	UpperBoundXNumericCtrl->Connect( wxEVT_SET_FOCUS, wxFocusEventHandler( DistributionPlotDialogParent::OnUpperBoundXSetFocus ), NULL, this );
	UpperBoundXNumericCtrl->Connect( wxEVT_COMMAND_TEXT_ENTER, wxCommandEventHandler( DistributionPlotDialogParent::OnUpperBoundXTextEnter ), NULL, this );
	SaveTXTButton->Connect( wxEVT_COMMAND_BUTTON_CLICKED, wxCommandEventHandler( DistributionPlotDialogParent::OnSaveTXTButtonClick ), NULL, this );
	SavePNGButton->Connect( wxEVT_COMMAND_BUTTON_CLICKED, wxCommandEventHandler( DistributionPlotDialogParent::OnSavePNGButtonClick ), NULL, this );
}

DistributionPlotDialogParent::~DistributionPlotDialogParent()
{
	// Disconnect Events
	UpperBoundYNumericCtrl->Disconnect( wxEVT_KILL_FOCUS, wxFocusEventHandler( DistributionPlotDialogParent::OnUpperBoundYKillFocus ), NULL, this );
	UpperBoundYNumericCtrl->Disconnect( wxEVT_SET_FOCUS, wxFocusEventHandler( DistributionPlotDialogParent::OnUpperBoundYSetFocus ), NULL, this );
	UpperBoundYNumericCtrl->Disconnect( wxEVT_COMMAND_TEXT_ENTER, wxCommandEventHandler( DistributionPlotDialogParent::OnUpperBoundYTextEnter ), NULL, this );
	LowerBoundYNumericCtrl->Disconnect( wxEVT_KILL_FOCUS, wxFocusEventHandler( DistributionPlotDialogParent::OnLowerBoundYKillFocus ), NULL, this );
	LowerBoundYNumericCtrl->Disconnect( wxEVT_SET_FOCUS, wxFocusEventHandler( DistributionPlotDialogParent::OnLowerBoundYSetFocus ), NULL, this );
	LowerBoundYNumericCtrl->Disconnect( wxEVT_COMMAND_TEXT_ENTER, wxCommandEventHandler( DistributionPlotDialogParent::OnLowerBoundYTextEnter ), NULL, this );
	DataSeriesToPlotChoice->Disconnect( wxEVT_COMMAND_CHOICE_SELECTED, wxCommandEventHandler( DistributionPlotDialogParent::OnDataSeriesToPlotChoice ), NULL, this );
	LowerBoundXNumericCtrl->Disconnect( wxEVT_KILL_FOCUS, wxFocusEventHandler( DistributionPlotDialogParent::OnLowerBoundXKillFocus ), NULL, this );
	LowerBoundXNumericCtrl->Disconnect( wxEVT_SET_FOCUS, wxFocusEventHandler( DistributionPlotDialogParent::OnLowerBoundXSetFocus ), NULL, this );
	LowerBoundXNumericCtrl->Disconnect( wxEVT_COMMAND_TEXT_ENTER, wxCommandEventHandler( DistributionPlotDialogParent::OnLowerBoundXTextEnter ), NULL, this );
	UpperBoundXNumericCtrl->Disconnect( wxEVT_KILL_FOCUS, wxFocusEventHandler( DistributionPlotDialogParent::OnUpperBoundXKillFocus ), NULL, this );
	UpperBoundXNumericCtrl->Disconnect( wxEVT_SET_FOCUS, wxFocusEventHandler( DistributionPlotDialogParent::OnUpperBoundXSetFocus ), NULL, this );
	UpperBoundXNumericCtrl->Disconnect( wxEVT_COMMAND_TEXT_ENTER, wxCommandEventHandler( DistributionPlotDialogParent::OnUpperBoundXTextEnter ), NULL, this );
	SaveTXTButton->Disconnect( wxEVT_COMMAND_BUTTON_CLICKED, wxCommandEventHandler( DistributionPlotDialogParent::OnSaveTXTButtonClick ), NULL, this );
	SavePNGButton->Disconnect( wxEVT_COMMAND_BUTTON_CLICKED, wxCommandEventHandler( DistributionPlotDialogParent::OnSavePNGButtonClick ), NULL, this );
	
}
