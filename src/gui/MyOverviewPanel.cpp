#include "../core/gui_core_headers.h"

MyOverviewPanel::MyOverviewPanel( wxWindow* parent, wxWindowID id, const wxPoint& pos, const wxSize& size, long style)
:
OverviewPanel( parent, id, pos, size, style )
{
	SetWelcomeInfo();

}

void MyOverviewPanel::SetWelcomeInfo()
{
	#include "icons/cistem_logo_800.cpp"

	wxLogNull *suppress_png_warnings = new wxLogNull;
	wxBitmap logo_bmp = wxBITMAP_PNG_FROM_DATA(cistem_logo_800);
	delete suppress_png_warnings;

	InfoText->BeginAlignment(wxTEXT_ALIGNMENT_CENTRE);
	InfoText->WriteImage(logo_bmp);
	InfoText->Newline();
	InfoText->Newline();
	InfoText->EndAlignment();

	InfoText->BeginSuppressUndo();
	InfoText->BeginAlignment(wxTEXT_ALIGNMENT_CENTRE);
	InfoText->BeginFontSize(14);
	InfoText->WriteText(wxT("Welcome to cisTEM (Computational Imaging System for Transmission Electron Microscopy)"));
	InfoText->EndFontSize();
	InfoText->Newline();
	InfoText->EndAlignment();


	InfoText->BeginAlignment(wxTEXT_ALIGNMENT_CENTRE);
	InfoText->BeginFontSize(14);
	InfoText->WriteText(wxT("For more information, manuals and tutorials please visit "));
	InfoText->BeginURL("http://www.cistem.org");
	InfoText->BeginUnderline();
	InfoText->BeginTextColour(*wxBLUE);
	InfoText->WriteText(wxT("www.cistem.org"));
	InfoText->EndURL();
	InfoText->EndTextColour();
	InfoText->EndUnderline();
	InfoText->EndFontSize();
	InfoText->Newline();
	InfoText->Newline();
	InfoText->EndAlignment();
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

	 wxLaunchDefaultBrowser(my_style.GetURL());
}
