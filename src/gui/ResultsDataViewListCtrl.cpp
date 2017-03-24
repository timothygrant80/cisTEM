//#include "../core/core_headers.h"
#include "../core/gui_core_headers.h"

//#include "icons/open_eye_icon_20.cpp"
#include "icons/checked_checkbox_icon_20.cpp"
#include "icons/unchecked_checkbox_icon_20.cpp"
#include "icons/checked_checkbox_eye_icon_20.cpp"
#include "icons/unchecked_checkbox_eye_icon_20.cpp"


wxBitmap CheckboxRenderer::checked_bmp;
wxBitmap CheckboxRenderer::unchecked_bmp;
wxBitmap CheckboxRenderer::checked_eye_bmp;
wxBitmap CheckboxRenderer::unchecked_eye_bmp;


ResultsDataViewListCtrl::ResultsDataViewListCtrl(wxWindow* parent, wxWindowID id, const wxPoint& pt, const wxSize& sz, long style):
   wxDataViewListCtrl(parent, id, pt, sz, style)
{
	my_parent = parent;
	currently_selected_row = -1;
	currently_selected_column = -1;

	// set the icons..

	wxLogNull *suppress_png_warnings = new wxLogNull;
	CheckboxRenderer::checked_bmp = wxBITMAP_PNG_FROM_DATA(checked_checkbox_icon_20);
	CheckboxRenderer::unchecked_bmp = wxBITMAP_PNG_FROM_DATA(unchecked_checkbox_icon_20);
	CheckboxRenderer::checked_eye_bmp = wxBITMAP_PNG_FROM_DATA(checked_checkbox_eye_icon_20);
	CheckboxRenderer::unchecked_eye_bmp = wxBITMAP_PNG_FROM_DATA(unchecked_checkbox_eye_icon_20);
	delete suppress_png_warnings;


	// selection change event..

	Bind(wxEVT_DATAVIEW_SELECTION_CHANGED, wxDataViewEventHandler( ResultsDataViewListCtrl::OnSelectionChange), this);
//	Bind(wxEVT_DATAVIEW_ITEM_CONTEXT_MENU, wxDataViewEventHandler( ResultsDataViewListCtrl::OnContextMenu), this);
	Bind(wxEVT_DATAVIEW_ITEM_VALUE_CHANGED, wxDataViewEventHandler( ResultsDataViewListCtrl::OnValueChanged), this);
	Bind(wxEVT_DATAVIEW_COLUMN_HEADER_CLICK, wxDataViewEventHandler( ResultsDataViewListCtrl::OnHeaderClick), this);


}

ResultsDataViewListCtrl::~ResultsDataViewListCtrl()
{
	Unbind(wxEVT_DATAVIEW_SELECTION_CHANGED, wxDataViewEventHandler( ResultsDataViewListCtrl::OnSelectionChange), this);
//	Unbind(wxEVT_DATAVIEW_ITEM_CONTEXT_MENU, wxDataViewEventHandler( ResultsDataViewListCtrl::OnContextMenu), this);
	Unbind(wxEVT_DATAVIEW_ITEM_VALUE_CHANGED, wxDataViewEventHandler( ResultsDataViewListCtrl::OnValueChanged), this);
	Unbind(wxEVT_DATAVIEW_COLUMN_HEADER_CLICK, wxDataViewEventHandler( ResultsDataViewListCtrl::OnHeaderClick), this);
}

void ResultsDataViewListCtrl::OnSelectionChange(wxDataViewEvent &event)
{
	// we don't want selections shown - unselect..
	Unselect(event.GetItem());
}

void ResultsDataViewListCtrl::SizeColumns()
{
	Freeze();

	// get the client area..
	int my_width;
	int my_height;
	int number_of_columns;
	int total_size_of_checkbox_columns;

	GetClientSize(&my_width,&my_height);
	my_width -=  wxSystemSettings::GetMetric(wxSYS_VSCROLL_X);
	number_of_columns = GetColumnCount();

	GetColumn(0)->SetWidth(60);

	//GetColumn(0)->SetMinWidth(60);
	//GetColumn(1)->SetWidth(wxCOL_WIDTH_AUTOSIZE);

	if (number_of_columns > 2)
	{

		total_size_of_checkbox_columns = ((number_of_columns - 2) * 53) + 60;

		int new_width = my_width-total_size_of_checkbox_columns - 2;

		if (new_width > 100) GetColumn(1)->SetWidth(new_width);
		else
		GetColumn(1)->SetWidth(100);

		//GetColumn(1)->SetMinWidth(my_width-total_size_of_checkbox_columns-2);


	}

	//wxPrintf("Width = %i\n", my_width);
	//wxPrintf("Setting Width = %i\n", my_width-total_size_of_checkbox_columns-2);

	Thaw();
}

void ResultsDataViewListCtrl::UncheckItem(const int row, const int column)
{
	wxVariant temp_variant;
	GetValue(temp_variant, row, column);
	long value = temp_variant.GetLong();

	if (value == CHECKED) SetValue(wxVariant(UNCHECKED), row, column);
	else
	if (value == CHECKED_WITH_EYE) SetValue(wxVariant(UNCHECKED_WITH_EYE), row, column);

}

void ResultsDataViewListCtrl::CheckItem(const int row, const int column)
{
	wxVariant temp_variant;
	GetValue(temp_variant, row, column);
	long value = temp_variant.GetLong();

	if (value == UNCHECKED) SetValue(wxVariant(CHECKED), row, column);
	else
	if (value == UNCHECKED_WITH_EYE) SetValue(wxVariant(CHECKED_WITH_EYE), row, column);

}

void ResultsDataViewListCtrl::OnHeaderClick(wxDataViewEvent &event)
{
	wxDataViewColumn *current_column =  event.GetDataViewColumn();

	if (current_column->GetModelColumn() > 1)
	{
		wxMessageDialog *check_dialog = new wxMessageDialog(this, wxString::Format("Do you want to set the active job for all possible movies to %s?", current_column->GetTitle()), "Please Confirm", wxYES_NO);

		if (check_dialog->ShowModal() ==  wxID_YES)
		{
			OneSecondProgressDialog *my_dialog = new OneSecondProgressDialog ("Select Column", "Setting Selections", GetItemCount(), this);

			for (long counter = 0; counter < GetItemCount(); counter++)
			{
				// get the item at row counter, column whatevs..

				CheckItem(counter, current_column->GetModelColumn());
				my_dialog->Update(counter);
			}

			my_dialog->Destroy();

		}

	}

}

void ResultsDataViewListCtrl::OnValueChanged(wxDataViewEvent &event)
{

	//wxDataViewColumn *current_column = NULL;
	//wxPoint position = wxGetMousePosition();
	//wxPoint new_position = ScreenToClient(position);

	//HitTest	(ScreenToClient(wxGetMousePosition()), current_item, current_column);

	wxDataViewItem current_item = event.GetItem();
	int row =  ItemToRow(current_item);
	int column = event.GetColumn();

	wxVariant temp_variant;
	GetValue(temp_variant, row, column);
	long value = temp_variant.GetLong();

	if (value == CHECKED_WITH_EYE || value == UNCHECKED_WITH_EYE)
	{
		Deselect();
		currently_selected_row = row;
		currently_selected_column = column;
	}


	if (value == CHECKED || value == CHECKED_WITH_EYE) // make sure there are no other checked items in this row..
	{
		for (int column_counter = 2; column_counter < GetColumnCount(); column_counter++)
		{
			if (column_counter != column)
			{
				GetValue(temp_variant, row, column_counter);
				value = temp_variant.GetLong();

				if (value == CHECKED || value == CHECKED_WITH_EYE) UncheckItem(row, column_counter);
			}
		}
	}


	event.Skip();


}

void ResultsDataViewListCtrl::NextEye()
{
	if (currently_selected_row < GetItemCount() - 1)
	{
		// find out which column is checked in the next row, and select that instead..

		ChangeDisplayTo(currently_selected_row + 1, ReturnCheckedColumn(currently_selected_row + 1));
	}

}

void ResultsDataViewListCtrl::PreviousEye()
{
	if (currently_selected_row > 0)
	{
		// find out which column is checked in the next row, and select that instead..

		ChangeDisplayTo(currently_selected_row - 1, ReturnCheckedColumn(currently_selected_row - 1));
	}

}

int ResultsDataViewListCtrl::ReturnCheckedColumn(int wanted_row)
{
	wxVariant temp_variant;
	long value;

	for (int column_counter = 2; column_counter < GetColumnCount(); column_counter++)
	{
		GetValue(temp_variant, wanted_row, column_counter);
		value = temp_variant.GetLong();

		if (value == CHECKED || value == CHECKED_WITH_EYE) return column_counter;

	}

	return -1;
}

void ResultsDataViewListCtrl::Deselect()
{
	if (currently_selected_column >= 0 && currently_selected_row >= 0)
	{
		wxVariant temp_variant;
		GetValue(temp_variant, currently_selected_row, currently_selected_column);
		long value = temp_variant.GetLong();

		if (value == CHECKED_WITH_EYE) SetValue(wxVariant(CHECKED), currently_selected_row, currently_selected_column);
		else
		if (value == UNCHECKED_WITH_EYE) SetValue(wxVariant(UNCHECKED), currently_selected_row, currently_selected_column);

		currently_selected_column = -1;
		currently_selected_row = -1;
	}

}

void ResultsDataViewListCtrl::ChangeDisplayTo(const int row, const int column)
{
	if (row != currently_selected_row || column != currently_selected_column)
	{
		if (row >= 0 && column >= 0)
		{
			Freeze();

			wxVariant temp_variant;
			GetValue(temp_variant, row, column);
			long value = temp_variant.GetLong();

			if (value != CHECKED_WITH_EYE && value != UNCHECKED_WITH_EYE)
			{
				Deselect();

				if (value == CHECKED) SetValue(wxVariant(CHECKED_WITH_EYE), row, column);
				else
				if (value == UNCHECKED) SetValue(wxVariant(UNCHECKED_WITH_EYE), row, column);

				currently_selected_row = row;
				currently_selected_column = column;


				EnsureVisible(RowToItem(row), GetColumn(column));
//				Update();
	//			Refresh();

			}

			Thaw();
		}
	}
}

void ResultsDataViewListCtrl::OnContextMenu(wxDataViewEvent &event)
{
	// bit of a hack to get the row column of a right click..

	wxDataViewItem current_item;
	wxDataViewColumn *current_column = NULL;
	wxPoint position = wxGetMousePosition();

	int scroll_amount = 0;

	if (HasScrollbar(wxHORIZONTAL) == true)
	{
		int x_pixels_per_unit;
		int y_pixels_per_unit;
		//my_parent->GetScrollPixelsPerUnit(x_pixels_per_unit, y_pixels_per_unit);
		scroll_amount = GetScrollPos(wxHORIZONTAL);
		wxPrintf("scroll amount = %i\n", scroll_amount);

	}
	else
	{
		wxPrintf("no scroll - scroll amount = %i\n", scroll_amount);
	}


	wxPoint new_position = ScreenToClient(position);

	HitTest	(ScreenToClient(wxGetMousePosition()), current_item, current_column);

	int row = ItemToRow(current_item);

	if (row >= 0)
	{
		int column = current_column->GetModelColumn();

			wxVariant temp_variant;

			if (column > 1)
			{
				GetValue(temp_variant, row, column);
				long value = temp_variant.GetLong();

				if (value != CHECKED_WITH_EYE && value != UNCHECKED_WITH_EYE && value != BLANK) // this is not currently selected, and it is not blank select it..
				{
					ChangeDisplayTo(row, column);
				}
			}

	}

}


void ResultsDataViewListCtrl::AppendCheckColumn(wxString column_title)
{
	wxDataViewColumn *column = new wxDataViewColumn(column_title, new CheckboxRenderer("long", wxDATAVIEW_CELL_ACTIVATABLE, wxDVR_DEFAULT_ALIGNMENT), GetColumnCount(), 50, wxALIGN_CENTER, 0);
 	AppendColumn(column);
}


void ResultsDataViewListCtrl::Clear()
{
	Freeze();
	currently_selected_row = -1;
	currently_selected_column = -1;
	ClearColumns();
	DeleteAllItems();
	Thaw();

}

CheckboxRenderer::CheckboxRenderer(const wxString &varianttype, wxDataViewCellMode mode, int align) :
	wxDataViewCustomRenderer (varianttype, mode, align)
{

}

bool CheckboxRenderer::ActivateCell	(const wxRect & cell, wxDataViewModel * 	model,	const wxDataViewItem & 	item,	unsigned int 	col,const wxMouseEvent * 	mouseEvent	)
{
	int mouse_x, mouse_y;

	mouseEvent->GetPosition(&mouse_x, &mouse_y);
	//wxPrintf("Activate, X=%i, Y=%i    Mouse X=%i, Mouse Y=%i\n", cell.x, cell.y, mouse_x, mouse_y);

	if (mouse_x >= 0 && mouse_x <= 18)
	{
		//if (current_mode == CHECKED) model->ChangeValue(wxVariant(UNCHECKED), item, col);
		//else
		if (current_mode == UNCHECKED) model->ChangeValue(wxVariant(CHECKED), item, col);
		else
		//if (current_mode == CHECKED_WITH_EYE) model->ChangeValue(wxVariant(UNCHECKED_WITH_EYE), item, col);
		//else
		if (current_mode == UNCHECKED_WITH_EYE) model->ChangeValue(wxVariant(CHECKED_WITH_EYE), item, col);
	}
	else
	if (mouse_x >= 20 && mouse_x <= 40)
	{
		if (current_mode == CHECKED) model->ChangeValue(wxVariant(CHECKED_WITH_EYE), item,col);
		else
		if (current_mode == UNCHECKED) model->ChangeValue(wxVariant(UNCHECKED_WITH_EYE), item, col);
	}



}




bool CheckboxRenderer::GetValue(wxVariant &value) const
{
	wxVariant temp(current_mode);
	value = temp;
	return true;
}

bool CheckboxRenderer::SetValue(const wxVariant &value)
{
	current_mode = value.GetLong();

	return true;
}

bool CheckboxRenderer:: Render(wxRect 	cell, wxDC *dc,	int state)
{
	if (current_mode == UNCHECKED) dc->DrawBitmap(unchecked_bmp, cell.x, cell.y, true);
	else
	if (current_mode == CHECKED) dc->DrawBitmap(checked_bmp, cell.x, cell.y, true);
	else
	if (current_mode == UNCHECKED_WITH_EYE) dc->DrawBitmap(unchecked_eye_bmp, cell.x, cell.y, true);
	else
	if (current_mode == CHECKED_WITH_EYE) dc->DrawBitmap(checked_eye_bmp, cell.x, cell.y, true);
}


wxSize CheckboxRenderer::GetSize() const
{
	wxSize my_size(40, 20);
	return my_size;
}

