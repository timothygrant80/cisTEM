#include "../core/gui_core_headers.h"

MyOverviewPanel::MyOverviewPanel( wxWindow* parent, wxWindowID id, const wxPoint& pos, const wxSize& size, long style)
:
OverviewPanel( parent, id, pos, size, style )
{
	SetWelcomeInfo();

}

void MyOverviewPanel::SetWelcomeInfo()
{
	#include "icons/cistem_logo_600.cpp"

	InfoText->Clear();
	InfoText->EndAllStyles();

	wxLogNull *suppress_png_warnings = new wxLogNull;
	wxBitmap logo_bmp = wxBITMAP_PNG_FROM_DATA(cistem_logo_600);
	delete suppress_png_warnings;

	InfoText->GetCaret()->Hide();
	InfoText->BeginSuppressUndo();

	InfoText->Newline();
	InfoText->Newline();

	InfoText->BeginAlignment(wxTEXT_ALIGNMENT_CENTRE);
	InfoText->WriteImage(logo_bmp);
	InfoText->Newline();
	InfoText->Newline();
	InfoText->EndAlignment();

	InfoText->BeginAlignment(wxTEXT_ALIGNMENT_CENTRE);
	InfoText->BeginFontSize(12);
	InfoText->WriteText(wxT("Welcome to cisTEM (Computational Imaging System for Transmission Electron Microscopy)"));
	InfoText->Newline();
	InfoText->EndFontSize();
	InfoText->EndAlignment();


	InfoText->BeginAlignment(wxTEXT_ALIGNMENT_CENTRE);
	InfoText->BeginFontSize(12);
	InfoText->WriteText(wxT("For more information, manuals and tutorials please visit "));
	InfoText->BeginURL("http://www.cistem.org");
	InfoText->BeginUnderline();
	InfoText->BeginTextColour(*wxBLUE);
	InfoText->WriteText(wxT("www.cistem.org"));
	InfoText->EndURL();
	InfoText->EndTextColour();
	InfoText->EndUnderline();
	InfoText->Newline();
	InfoText->EndFontSize();
	InfoText->EndAlignment();

	InfoText->BeginAlignment(wxTEXT_ALIGNMENT_CENTRE);
	InfoText->BeginFontSize(12);
	InfoText->WriteText(wxString::Format("Version : 0.1 (Compiled : %s )",  __DATE__));
	InfoText->Newline();
	InfoText->Newline();
	InfoText->EndFontSize();
	InfoText->EndAlignment();

	InfoText->BeginAlignment(wxTEXT_ALIGNMENT_CENTRE);
	InfoText->BeginFontSize(12);
	InfoText->BeginUnderline();
	InfoText->BeginBold();
	InfoText->WriteText(wxT("Begin"));
	InfoText->EndUnderline();
	InfoText->EndBold();
	InfoText->Newline();
	InfoText->EndFontSize();
	InfoText->EndAlignment();

	InfoText->BeginAlignment(wxTEXT_ALIGNMENT_CENTRE);
	InfoText->BeginFontSize(12);
	InfoText->BeginURL("CreateProject");
	InfoText->BeginUnderline();
	InfoText->BeginTextColour(*wxBLUE);
	InfoText->WriteText(wxT("Create a new project"));
	InfoText->EndURL();
	InfoText->EndTextColour();
	InfoText->EndUnderline();
	InfoText->Newline();
	InfoText->EndFontSize();
	InfoText->EndAlignment();

	InfoText->BeginAlignment(wxTEXT_ALIGNMENT_CENTRE);
	InfoText->BeginFontSize(12);
	InfoText->BeginURL("OpenProject");
	InfoText->BeginUnderline();
	InfoText->BeginTextColour(*wxBLUE);
	InfoText->WriteText(wxT("Open an exisiting project"));
	InfoText->EndURL();
	InfoText->EndTextColour();
	InfoText->EndUnderline();
	InfoText->Newline();
	InfoText->Newline();
	InfoText->EndFontSize();
	InfoText->EndAlignment();

	// do we have previous projects..

	wxArrayString recent_projects = GetRecentProjectsFromSettings();

	if (recent_projects.GetCount() > 0)
	{
		InfoText->BeginAlignment(wxTEXT_ALIGNMENT_CENTRE);
		InfoText->BeginFontSize(12);
		InfoText->BeginUnderline();
		InfoText->BeginBold();
		InfoText->WriteText(wxT("Open Recent Project"));
		InfoText->EndUnderline();
		InfoText->EndBold();
		InfoText->Newline();
		InfoText->EndFontSize();
		InfoText->EndAlignment();

		for (int counter = 0; counter < recent_projects.GetCount(); counter++)
		{
			InfoText->BeginAlignment(wxTEXT_ALIGNMENT_CENTRE);
			InfoText->BeginFontSize(12);
			InfoText->BeginURL(recent_projects.Item(counter));
			InfoText->BeginUnderline();
			InfoText->BeginTextColour(*wxBLUE);
			InfoText->WriteText(recent_projects.Item(counter));
			InfoText->EndURL();
			InfoText->EndTextColour();
			InfoText->EndUnderline();
			InfoText->Newline();
			InfoText->EndFontSize();
			InfoText->EndAlignment();
		}

	}


	InfoText->EndSuppressUndo();
	InfoText->EndAllStyles();
}

void MyOverviewPanel::OnInfoURL(wxTextUrlEvent& event)
{
	 const wxMouseEvent& ev = event.GetMouseEvent();

	 // filter out mouse moves, too many of them
	 if ( ev.Moving() ) return;

	 long start = event.GetURLStart();

	 wxTextAttr my_style;
	 InfoText->GetStyle(start, my_style);

	 // Launch the URL

	 InfoText->SetInsertionPoint(0);

	 if (my_style.GetURL() == "CreateProject")
	 {
		 main_frame->StartNewProject();
	 }
	 else
	 if (my_style.GetURL() == "OpenProject")
	 {
		 main_frame->GetFileAndOpenProject();
	 }
	 else
	 if (my_style.GetURL() == "www.cistem.org") wxLaunchDefaultBrowser(my_style.GetURL());
	 else main_frame->OpenProject(my_style.GetURL());
}
