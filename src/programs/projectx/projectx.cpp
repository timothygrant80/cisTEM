#include "../../core/core_headers.h"
#include "../../core/gui_core_headers.h"

// icons..

#include "../../gui/icons/overview_icon.cpp"
#include "../../gui/icons/assets_icon.cpp"
#include "../../gui/icons/action_icon.cpp"
#include "../../gui/icons/settings_icon.cpp"

#include "../../gui/icons/movie_icon.cpp"
#include "../../gui/icons/image_icon.cpp"

#include "../../gui/icons/movie_align_icon.cpp"
#include "../../gui/icons/ctf_icon.cpp"

#include "../../gui/icons/run_profiles_icon.cpp"

class
MyGuiApp : public wxApp
{

	public:
		virtual bool OnInit();
		virtual int OnExit();
};


IMPLEMENT_APP(MyGuiApp)

MyMainFrame *main_frame;
MyAlignMoviesPanel *align_movies_panel;

OverviewPanel *overview_panel;
ActionsPanel *actions_panel;
AssetsPanel *assets_panel;
SettingsPanel *settings_panel;

MyMovieAssetPanel *movie_asset_panel;
MyImageAssetPanel *image_asset_panel;

MyRunProfilesPanel *run_profiles_panel;

wxImageList *MenuBookIconImages;
wxImageList *ActionsBookIconImages;
wxImageList *AssetsBookIconImages;
wxImageList *SettingsBookIconImages;


SETUP_SOCKET_CODES


bool MyGuiApp::OnInit()
{
	wxImage::AddHandler(new wxPNGHandler);

	wxImageList *MenuBookIconImages;
	wxImageList *ActionsBookIconImages;
	wxImageList *AssetsBookIconImages;
	wxImageList *SettingsBookIconImages;

	main_frame = new MyMainFrame( (wxWindow*)NULL);

	// Left hand Panels

	overview_panel = new OverviewPanel(main_frame->MenuBook, wxID_ANY);
	actions_panel = new ActionsPanel(main_frame->MenuBook, wxID_ANY);
	assets_panel = new AssetsPanel(main_frame->MenuBook, wxID_ANY);
	settings_panel = new SettingsPanel(main_frame->MenuBook, wxID_ANY);

	// Individual Panels

	movie_asset_panel = new MyMovieAssetPanel(assets_panel->AssetsBook);
	image_asset_panel = new MyImageAssetPanel(assets_panel->AssetsBook);
	align_movies_panel = new MyAlignMoviesPanel(actions_panel->ActionsBook);

	run_profiles_panel = new MyRunProfilesPanel(settings_panel->SettingsBook);


	main_frame->Show();

	// Setup list books

	MenuBookIconImages = new wxImageList();
	ActionsBookIconImages = new wxImageList();
	AssetsBookIconImages = new wxImageList();
	SettingsBookIconImages = new wxImageList();

	wxBitmap overview_icon_bmp = wxBITMAP_PNG_FROM_DATA(overview_icon);
	wxBitmap assets_icon_bmp = wxBITMAP_PNG_FROM_DATA(assets_icon);
	wxBitmap action_icon_bmp = wxBITMAP_PNG_FROM_DATA(action_icon);
	wxBitmap settings_icon_bmp = wxBITMAP_PNG_FROM_DATA(settings_icon);

	wxBitmap movie_icon_bmp = wxBITMAP_PNG_FROM_DATA(movie_icon);
	wxBitmap image_icon_bmp = wxBITMAP_PNG_FROM_DATA(image_icon);

	wxBitmap movie_align_icon_bmp = wxBITMAP_PNG_FROM_DATA(movie_align_icon);
	wxBitmap ctf_icon_bmp = wxBITMAP_PNG_FROM_DATA(ctf_icon);

	wxBitmap run_profiles_icon_bmp = wxBITMAP_PNG_FROM_DATA(run_profiles_icon);



	MenuBookIconImages->Add(overview_icon_bmp);
	MenuBookIconImages->Add(assets_icon_bmp);
	MenuBookIconImages->Add(action_icon_bmp);
	MenuBookIconImages->Add(settings_icon_bmp);

	ActionsBookIconImages->Add(movie_align_icon_bmp);
	ActionsBookIconImages->Add(ctf_icon_bmp);

	AssetsBookIconImages->Add(movie_icon_bmp);
	AssetsBookIconImages->Add(image_icon_bmp);

	SettingsBookIconImages->Add(run_profiles_icon_bmp);


	main_frame->MenuBook->AssignImageList(MenuBookIconImages);
	actions_panel->ActionsBook->AssignImageList(ActionsBookIconImages);
	assets_panel->AssetsBook->AssignImageList(AssetsBookIconImages);
	settings_panel->SettingsBook->AssignImageList(SettingsBookIconImages);

	main_frame->MenuBook->AddPage(overview_panel, "Overview", true, 0);
	main_frame->MenuBook->AddPage(assets_panel, "Assets", false, 1);
	main_frame->MenuBook->AddPage(actions_panel, "Actions", false, 2);
	main_frame->MenuBook->AddPage(settings_panel, "Settings", false, 3);

	assets_panel->AssetsBook->AddPage(movie_asset_panel, "Movies", false, 0);
	assets_panel->AssetsBook->AddPage(image_asset_panel, "Images", false, 1);

	actions_panel->ActionsBook->AddPage(align_movies_panel, "Align Movies", true, 0);
	actions_panel->ActionsBook->AddPage(image_asset_panel, "Find CTF", false, 1);

	settings_panel->SettingsBook->AddPage(run_profiles_panel, "Run Profiles", true, 0);

	// Setup Movie Panel

	//movie_asset_panel->GroupListBox->InsertItem(0, "All Movies", 0);
	//vie_asset_panel->GroupListBox->SetItemState(0, wxLIST_STATE_SELECTED, wxLIST_STATE_SELECTED);

	// Setup the Asset Tree.

	main_frame->RecalculateAssetBrowser();


	//Particles_Branch = main_frame->AssetTree->AppendItem(Tree_Root, "Particles (0)");
	//Classes_Branch = main_frame->AssetTree->AppendItem(Tree_Root, "Classes (0)");
	//Maps_Branch = main_frame->AssetTree->AppendItem(Tree_Root, "3D Maps (0)");

	// welcome text..

	//main_frame->StatusText->AppendText(wxT("Welcome to Project...X!\n"));

	//GuiJob test_job;

	//test_job.parent_panel = align_movies_panel;
	//test_job.parent_panel->UpdateJobDetails("Hello\n");


	//delete MenuBookIconImages;
	//delete ActionsBookIconImages;
	//delete AssetsBookIconImages;
	//delete SettingsBookIconImages;

	return true;
}

int MyGuiApp::OnExit()
{

/*	main_frame->Destroy();
	overview_panel->Destroy();
	actions_panel->Destroy();
	assets_panel->Destroy();
	settings_panel->Destroy();

	// Individual Panels

	movie_asset_panel->Destroy();
	image_asset_panel->Destroy();
	align_movies_panel->Destroy();
	run_profiles_panel->Destroy();*/

}

