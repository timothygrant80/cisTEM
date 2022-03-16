//#include <wx/listctrl.h>

//IMPLEMENT_CLASS(wxCheckedListCtrl, wxListCtrl)

#ifndef __CHECKED_LISTCTRL_H__
#define __CHECKED_LISTCTRL_H__

#include <wx/listctrl.h>
#include <wx/imaglist.h>

class wxCheckedListCtrl : public wxListCtrl {
  protected:
    wxImageList m_imagelist;

  public:
    wxCheckedListCtrl(wxWindow* parent, wxWindowID id, const wxPoint& pt, const wxSize& sz, long style);
    bool Create(wxWindow* parent, wxWindowID id, const wxPoint& pt, const wxSize& sz, long style, const wxValidator& validator, const wxString& name);

    bool IsChecked(long item) const;
    void SetChecked(long item, bool checked);
};

#endif
