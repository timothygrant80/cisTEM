#ifndef __RESULTS_DATAVIEWLISTCTRL_H__
#define __RESULTS_DATAVIEWLISTCTRL_H__

#define BLANK -1
#define UNCHECKED 0
#define CHECKED 1
#define UNCHECKED_WITH_EYE 2
#define CHECKED_WITH_EYE 3

#include <wx/dataview.h>

class ResultsDataViewListCtrl : public wxDataViewListCtrl {
  protected:
    wxWindow* my_parent;
    wxIcon    checked_icon;
    wxIcon    unchecked_icon;
    wxIcon    checked_eye_icon;
    wxIcon    unchecked_eye_icon;

    // TODO: shouldn't these really be called current_eye_row and current_eye_column?
    int currently_selected_row;
    int currently_selected_column;

  public:
    std::string my_parents_name;
    ResultsDataViewListCtrl(wxWindow* parent, wxWindowID id, const wxPoint& pt, const wxSize& sz, long style);
    ~ResultsDataViewListCtrl( );
    void AppendCheckColumn(wxString column_title);

    void Deselect( );
    void UncheckItem(const int row, const int column);
    void CheckItem(const int row, const int column);
    void ChangeDisplayTo(const int row, const int column);

    void OnSelectionChange(wxDataViewEvent& event);
    void OnContextMenu(wxDataViewEvent& event);
    void OnValueChanged(wxDataViewEvent& event);
    void OnHeaderClick(wxDataViewEvent& event);

    void NextEye( );
    void PreviousEye( );

    int ReturnCheckedColumn(int wanted_row);

    inline int ReturnEyeColumn( ) { return currently_selected_column; };

    inline int ReturnEyeRow( ) { return currently_selected_row; };

    void Clear( );

    void SizeColumns( );
};

class CheckboxRenderer : public wxDataViewCustomRenderer {
  public:
    static wxBitmap checked_bmp;
    static wxBitmap unchecked_bmp;
    static wxBitmap checked_eye_bmp;
    static wxBitmap unchecked_eye_bmp;

    long current_mode;

    CheckboxRenderer(const wxString& varianttype, wxDataViewCellMode mode, int align);

    bool ActivateCell(const wxRect& cell, wxDataViewModel* model, const wxDataViewItem& item, unsigned int col, const wxMouseEvent* mouseEvent);
    bool GetValue(wxVariant& value) const;
    bool SetValue(const wxVariant& value);
    bool Render(wxRect cell, wxDC* dc, int state);

    wxSize GetSize( ) const;
};

#endif
