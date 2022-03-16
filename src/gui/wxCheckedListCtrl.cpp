//#include "../core/core_headers.h"
#include "../core/gui_core_headers.h"

wxCheckedListCtrl::wxCheckedListCtrl(wxWindow* parent, wxWindowID id, const wxPoint& pt, const wxSize& sz, long style) : wxListCtrl(parent, id, pt, sz, style) {

    // Get the native size of the checkbox
    int width  = wxRendererNative::Get( ).GetCheckBoxSize(this).GetWidth( );
    int height = wxRendererNative::Get( ).GetCheckBoxSize(this).GetHeight( );

    m_imagelist.Create(width, height, TRUE);
    SetImageList(&m_imagelist, wxIMAGE_LIST_SMALL);

    wxBitmap unchecked_bmp(width, height),
            checked_bmp(width, height),
            unchecked_disabled_bmp(width, height),
            checked_disabled_bmp(width, height);

    wxMemoryDC renderer_dc;

    // Unchecked
    renderer_dc.SelectObject(unchecked_bmp);
    renderer_dc.SetBackground(*wxTheBrushList->FindOrCreateBrush(GetBackgroundColour( ), wxSOLID));
    renderer_dc.Clear( );
    wxRendererNative::Get( ).DrawCheckBox(this, renderer_dc, wxRect(0, 0, width, height), 0);

    // Checked
    renderer_dc.SelectObject(checked_bmp);
    renderer_dc.SetBackground(*wxTheBrushList->FindOrCreateBrush(GetBackgroundColour( ), wxSOLID));
    renderer_dc.Clear( );
    wxRendererNative::Get( ).DrawCheckBox(this, renderer_dc, wxRect(0, 0, width, height), wxCONTROL_CHECKED);

    // Unchecked and Disabled
    renderer_dc.SelectObject(unchecked_disabled_bmp);
    renderer_dc.SetBackground(*wxTheBrushList->FindOrCreateBrush(GetBackgroundColour( ), wxSOLID));
    renderer_dc.Clear( );
    wxRendererNative::Get( ).DrawCheckBox(this, renderer_dc, wxRect(0, 0, width, height), 0 | wxCONTROL_DISABLED);

    // Checked and Disabled
    renderer_dc.SelectObject(checked_disabled_bmp);
    renderer_dc.SetBackground(*wxTheBrushList->FindOrCreateBrush(GetBackgroundColour( ), wxSOLID));
    renderer_dc.Clear( );
    wxRendererNative::Get( ).DrawCheckBox(this, renderer_dc, wxRect(0, 0, width, height), wxCONTROL_CHECKED | wxCONTROL_DISABLED);

    // Deselect the renderers Object
    renderer_dc.SelectObject(wxNullBitmap);

    // the add order must respect the wxCLC_XXX_IMGIDX defines in the headers !
    m_imagelist.Add(unchecked_bmp);
    m_imagelist.Add(checked_bmp);
    m_imagelist.Add(unchecked_disabled_bmp);
    m_imagelist.Add(checked_disabled_bmp);

    //SetChecked(0, true);
}

/*
void wxCheckedListCtrl::OnMouseEvent(wxMouseEvent& event)
{
      if (event.LeftDown())
      {
         int flags;
         long item = HitTest(event.GetPosition(), flags);
         if (item > -1 && (flags & wxLIST_HITTEST_ONITEMICON))
         {
             SetChecked(item, !IsChecked(item));
         }
         else
            event.Skip();
      }
      else
      {
         event.Skip();
      }
   }
*/

bool wxCheckedListCtrl::IsChecked(long item) const {
    wxListItem info;
    info.m_mask   = wxLIST_MASK_IMAGE;
    info.m_itemId = item;

    if ( GetItem(info) ) {
        return (info.m_image == 1);
    }
    else
        return FALSE;
}

void wxCheckedListCtrl::SetChecked(long item, bool checked) {
    SetItemImage(item, (checked ? 1 : 0), -1);
}

bool wxCheckedListCtrl::Create(wxWindow* parent, wxWindowID id, const wxPoint& pt,
                               const wxSize& sz, long style, const wxValidator& validator, const wxString& name) {
    if ( ! wxListCtrl::Create(parent, id, pt, sz, style, validator, name) )
        return FALSE;

    return TRUE;
}
