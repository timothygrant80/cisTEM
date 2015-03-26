#include <wx/wx.h>
#include "../../gui/ProjectX_gui.h"

#include "../../gui/MainFrame.h"
#include "../../gui/MovieAssetPanel.h"
#include "../../gui/MovieImportDialog.h"
#include "../../gui/AlignMoviesPanel.h"

#include "../../gui/icons/overview_icon.cpp"
#include "../../gui/icons/assets_icon.cpp"
#include "../../gui/icons/action_icon.cpp"

#include "../../gui/icons/movie_icon.cpp"
#include "../../gui/icons/image_icon.cpp"
#include "../../gui/icons/ctf_icon.cpp"

#include "../../gui/icons/movie_align_icon.cpp"


class
MyApp : public wxApp
{

	public:
		virtual bool OnInit();
};


IMPLEMENT_APP(MyApp)

MyMainFrame *main_frame;
MyAlignMoviesPanel *align_movies_panel;

OverviewPanel *overview_panel;
ActionsPanel *actions_panel;
AssetsPanel *assets_panel;
ImageAssetPanel *image_asset_panel;
MyMovieAssetPanel *movie_asset_panel;


bool MyApp::OnInit()
{
	wxImage::AddHandler(new wxPNGHandler);

	wxImageList *MenuBookIconImages;
	wxImageList *ActionsBookIconImages;
	wxImageList *AssetsBookIconImages;

	main_frame = new MyMainFrame( (wxWindow*)NULL);

	// Left hand Panels

	overview_panel = new OverviewPanel(main_frame->MenuBook, wxID_ANY);
	actions_panel = new ActionsPanel(main_frame->MenuBook, wxID_ANY);
	assets_panel = new AssetsPanel(main_frame->MenuBook, wxID_ANY);

	// Individual Panels

	movie_asset_panel = new MyMovieAssetPanel(assets_panel->AssetsBook);
	image_asset_panel = new ImageAssetPanel(assets_panel->AssetsBook, wxID_ANY);

	align_movies_panel = new MyAlignMoviesPanel(actions_panel->ActionsBook);


	main_frame->Show();

	// Setup list books

	MenuBookIconImages = new wxImageList();
	ActionsBookIconImages = new wxImageList();
	AssetsBookIconImages = new wxImageList();

	wxBitmap overview_icon_bmp = wxBITMAP_PNG_FROM_DATA(overview_icon);
	wxBitmap assets_icon_bmp = wxBITMAP_PNG_FROM_DATA(assets_icon);
	wxBitmap action_icon_bmp = wxBITMAP_PNG_FROM_DATA(action_icon);

	wxBitmap movie_icon_bmp = wxBITMAP_PNG_FROM_DATA(movie_icon);
	wxBitmap image_icon_bmp = wxBITMAP_PNG_FROM_DATA(image_icon);
	wxBitmap ctf_icon_bmp = wxBITMAP_PNG_FROM_DATA(ctf_icon);

	wxBitmap movie_align_icon_bmp = wxBITMAP_PNG_FROM_DATA(movie_align_icon);


	MenuBookIconImages->Add(overview_icon_bmp);
	MenuBookIconImages->Add(assets_icon_bmp);
	MenuBookIconImages->Add(action_icon_bmp);

	ActionsBookIconImages->Add(movie_align_icon_bmp);
	ActionsBookIconImages->Add(ctf_icon_bmp);

	AssetsBookIconImages->Add(movie_icon_bmp);
	AssetsBookIconImages->Add(image_icon_bmp);


	main_frame->MenuBook->SetImageList(MenuBookIconImages);
	actions_panel->ActionsBook->SetImageList(ActionsBookIconImages);
	assets_panel->AssetsBook->SetImageList(AssetsBookIconImages);

	main_frame->MenuBook->AddPage(overview_panel, "Overview", true, 0);
	main_frame->MenuBook->AddPage(assets_panel, "Assets", false, 1);
	main_frame->MenuBook->AddPage(actions_panel, "Actions", false, 2);

	assets_panel->AssetsBook->AddPage(movie_asset_panel, "Movies", false, 0);
	assets_panel->AssetsBook->AddPage(image_asset_panel, "Images", false, 1);

	actions_panel->ActionsBook->AddPage(align_movies_panel, "Align Movies", true, 0);
	actions_panel->ActionsBook->AddPage(image_asset_panel, "Find CTF", false, 1);

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
	return true;
}
