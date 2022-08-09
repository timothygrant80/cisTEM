#include "../core/gui_core_headers.h"

extern MyMovieAssetPanel* movie_asset_panel;
extern MyImageAssetPanel* image_asset_panel;

MyOverviewPanel::MyOverviewPanel(wxWindow* parent, wxWindowID id, const wxPoint& pos, const wxSize& size, long style)
    : OverviewPanel(parent, id, pos, size, style) {
    SetWelcomeInfo( );
}

void MyOverviewPanel::SetProjectInfo( ) {
//#include "icons/cisTEM_logo_800.cpp"
#include "icons/cisTEM_beta_logo_800.cpp"

    InfoText->Clear( );
    InfoText->EndAllStyles( );

    wxLogNull* suppress_png_warnings = new wxLogNull;
    wxBitmap   logo_bmp              = wxBITMAP_PNG_FROM_DATA(cisTEM_beta_logo_800);
    delete suppress_png_warnings;

    InfoText->GetCaret( )->Hide( );
    InfoText->BeginSuppressUndo( );

    InfoText->Newline( );
    InfoText->Newline( );

    InfoText->BeginAlignment(wxTEXT_ALIGNMENT_CENTRE);
    InfoText->WriteImage(logo_bmp);
    InfoText->Newline( );
    InfoText->Newline( );
    InfoText->EndAlignment( );

    if ( image_asset_panel->ReturnNumberOfAssets( ) == 0 && movie_asset_panel->ReturnNumberOfAssets( ) == 0 ) {
        InfoText->BeginAlignment(wxTEXT_ALIGNMENT_CENTRE);
        InfoText->Newline( );
        InfoText->Newline( );
        InfoText->WriteText(wxT("To get started, go to the Assets panel and import some movies or images..."));
        InfoText->Newline( );
        InfoText->Newline( );
    }

    InfoText->BeginAlignment(wxTEXT_ALIGNMENT_CENTRE);
    InfoText->Newline( );
    InfoText->Newline( );
    InfoText->BeginBold( );
    InfoText->BeginUnderline( );
    InfoText->WriteText(wxString::Format("Project summary"));
    InfoText->EndUnderline( );
    InfoText->EndBold( );
    InfoText->Newline( );
    InfoText->Newline( );
    InfoText->EndAlignment( );

    InfoText->BeginAlignment(wxTEXT_ALIGNMENT_CENTER);
    InfoText->BeginFontSize(12);
    InfoText->WriteText(wxString::Format("Project name: %s\n", main_frame->current_project.project_name));
    InfoText->WriteText(wxString::Format("Project directory: %s\n", main_frame->current_project.project_directory.GetFullPath( )));
    InfoText->WriteText(wxString::Format("Total job runtime : %0.2lf hours (%0.2lf days)\n", main_frame->current_project.total_cpu_hours, main_frame->current_project.total_cpu_hours / 24.0f));
    InfoText->WriteText(wxString::Format("Total number of jobs run: %i\n", main_frame->current_project.total_jobs_run));
    //InfoText->WriteText(wxString::Format("Est. real processing hours: 0.2lf\n", main_frame->current_project.total_jobs_run));

    InfoText->EndSuppressUndo( );
    InfoText->EndAllStyles( );
}

void MyOverviewPanel::SetWelcomeInfo( ) {
//#include "icons/cisTEM_logo_800.cpp"
#include "icons/cisTEM_beta_logo_800.cpp"

    InfoText->Clear( );
    InfoText->EndAllStyles( );

    wxLogNull* suppress_png_warnings = new wxLogNull;
    wxBitmap   logo_bmp              = wxBITMAP_PNG_FROM_DATA(cisTEM_beta_logo_800);
    delete suppress_png_warnings;

    InfoText->GetCaret( )->Hide( );
    InfoText->BeginSuppressUndo( );

    InfoText->Newline( );
    InfoText->Newline( );

    InfoText->BeginAlignment(wxTEXT_ALIGNMENT_CENTRE);
    InfoText->WriteImage(logo_bmp);
    InfoText->Newline( );
    InfoText->Newline( );
    InfoText->EndAlignment( );

    InfoText->BeginAlignment(wxTEXT_ALIGNMENT_CENTRE);
    InfoText->BeginFontSize(12);
    InfoText->WriteText(wxT("Welcome to "));
    InfoText->BeginItalic( );
    InfoText->WriteText(wxT("cis"));
    InfoText->EndItalic( );
    InfoText->WriteText(wxT("TEM (Computational Imaging System for Transmission Electron Microscopy)"));
    InfoText->Newline( );
    InfoText->EndFontSize( );
    InfoText->EndAlignment( );

    InfoText->BeginAlignment(wxTEXT_ALIGNMENT_CENTRE);
    InfoText->BeginFontSize(12);
    InfoText->WriteText(wxT("For more information, manuals and tutorials please visit "));
    InfoText->BeginURL("http://cistem.org");
    InfoText->BeginUnderline( );
    InfoText->BeginTextColour(*wxBLUE);
    InfoText->WriteText(wxT("cistem.org"));
    InfoText->EndURL( );
    InfoText->EndTextColour( );
    InfoText->EndUnderline( );
    InfoText->Newline( );
    InfoText->EndFontSize( );
    InfoText->EndAlignment( );

    InfoText->BeginAlignment(wxTEXT_ALIGNMENT_CENTRE);
    InfoText->BeginFontSize(12);

// Currently this is only defined in autoconf builds
#ifdef CISTEM_VERSION_TEXT
    InfoText->WriteText(wxString::Format("cisTEM version : %s", CISTEM_VERSION_TEXT));
    InfoText->Newline( );
#endif
// Currently this is only defined in autoconf builds
#ifdef CISTEM_TIME_READABLE
    InfoText->WriteText(wxString::Format("Commit datetime: %s", CISTEM_TIME_READABLE));
    InfoText->Newline( );
#endif
// Currently this is only defined in autoconf builds
#ifdef CISTEM_CURRENT_BRANCH
    InfoText->WriteText(wxString::Format("From branch : %s", CISTEM_CURRENT_BRANCH));
    InfoText->Newline( );
#endif
// Currently this is only defined in autoconf builds, only include human readable time here
#ifdef CISTEM_TIME_YYMMDDHHMMSS
    // InfoText->WriteText(wxString::Format("Compiled from branch : %s", CISTEM_TIME_YYMMDDHHMMSS));
    // InfoText->Newline();
#endif

#ifdef CISTEM_SVN_REV
    InfoText->WriteText(wxString::Format("SVN revision : %s", CISTEM_SVN_REV));
    InfoText->Newline( );
#endif

// If compiled with debug and any other special configure flags are set, display them here
#ifdef DEBUG
    InfoText->WriteText(wxString::Format("Compiled in DEBUG mode with additional configure flags: "));
#ifdef ENABLEGPU
    InfoText->WriteText(wxString::Format("ENABLEGPU "));
#endif
#ifdef MKL
    InfoText->WriteText(wxString::Format("MKL "));
#endif
#ifdef EXPERIMENTAL
    InfoText->WriteText(wxString::Format("EXPERIMENTAL "));
#endif
#ifdef RIGOROUS_SOCKET_CHECK
    InfoText->WriteText(wxString::Format("RIGOROUS_SOCKET_CHECK "));
#endif
#ifdef ENABLESAMPLES
    InfoText->WriteText(wxString::Format("ENABLESAMPLES "));
#endif
#ifdef ROTATEFORSPEED
    InfoText->WriteText(wxString::Format("ROTATEFORSPEED "));
#endif
#ifdef PROFILING
    InfoText->WriteText(wxString::Format("Compiled in PROFILING mode"));
#endif
    InfoText->Newline( );
#endif

    InfoText->Newline( );
    InfoText->Newline( );
    InfoText->EndFontSize( );
    InfoText->EndAlignment( );

    InfoText->BeginAlignment(wxTEXT_ALIGNMENT_CENTRE);
    InfoText->BeginFontSize(12);
    InfoText->BeginUnderline( );
    InfoText->BeginBold( );
    InfoText->WriteText(wxT("Begin"));
    InfoText->EndUnderline( );
    InfoText->EndBold( );
    InfoText->Newline( );
    InfoText->EndFontSize( );
    InfoText->EndAlignment( );

    InfoText->BeginAlignment(wxTEXT_ALIGNMENT_CENTRE);
    InfoText->BeginFontSize(12);
    InfoText->BeginURL("CreateProject");
    InfoText->BeginUnderline( );
    InfoText->BeginTextColour(*wxBLUE);
    InfoText->WriteText(wxT("Create a new project"));
    InfoText->EndURL( );
    InfoText->EndTextColour( );
    InfoText->EndUnderline( );
    InfoText->Newline( );
    InfoText->EndFontSize( );
    InfoText->EndAlignment( );

    InfoText->BeginAlignment(wxTEXT_ALIGNMENT_CENTRE);
    InfoText->BeginFontSize(12);
    InfoText->BeginURL("OpenProject");
    InfoText->BeginUnderline( );
    InfoText->BeginTextColour(*wxBLUE);
    InfoText->WriteText(wxT("Open an exisiting project"));
    InfoText->EndURL( );
    InfoText->EndTextColour( );
    InfoText->EndUnderline( );
    InfoText->Newline( );
    InfoText->Newline( );
    InfoText->EndFontSize( );
    InfoText->EndAlignment( );

    // do we have previous projects..

    wxArrayString recent_projects = GetRecentProjectsFromSettings( );

    if ( recent_projects.GetCount( ) > 0 ) {
        InfoText->BeginAlignment(wxTEXT_ALIGNMENT_CENTRE);
        InfoText->BeginFontSize(12);
        InfoText->BeginUnderline( );
        InfoText->BeginBold( );
        InfoText->WriteText(wxT("Open Recent Project"));
        InfoText->EndUnderline( );
        InfoText->EndBold( );
        InfoText->Newline( );
        InfoText->EndFontSize( );
        InfoText->EndAlignment( );

        for ( int counter = 0; counter < recent_projects.GetCount( ); counter++ ) {
            InfoText->BeginAlignment(wxTEXT_ALIGNMENT_CENTRE);
            InfoText->BeginFontSize(12);
            InfoText->BeginURL(recent_projects.Item(counter));
            InfoText->BeginUnderline( );
            InfoText->BeginTextColour(*wxBLUE);
            InfoText->WriteText(recent_projects.Item(counter));
            InfoText->EndURL( );
            InfoText->EndTextColour( );
            InfoText->EndUnderline( );
            InfoText->Newline( );
            InfoText->EndFontSize( );
            InfoText->EndAlignment( );
        }
    }

    InfoText->EndSuppressUndo( );
    InfoText->EndAllStyles( );
}

void MyOverviewPanel::OnInfoURL(wxTextUrlEvent& event) {
    const wxMouseEvent& ev = event.GetMouseEvent( );

    // filter out mouse moves, too many of them
    if ( ev.Moving( ) )
        return;

    long start = event.GetURLStart( );

    wxTextAttr my_style;
    InfoText->GetStyle(start, my_style);

    // Launch the URL

    InfoText->SetInsertionPoint(0);

    if ( my_style.GetURL( ) == "CreateProject" ) {
        main_frame->StartNewProject( );
    }
    else if ( my_style.GetURL( ) == "OpenProject" ) {
        main_frame->GetFileAndOpenProject( );
    }
    else if ( my_style.GetURL( ) == "http://cistem.org" )
        wxLaunchDefaultBrowser(my_style.GetURL( ));
    else
        main_frame->OpenProject(my_style.GetURL( ));
}
