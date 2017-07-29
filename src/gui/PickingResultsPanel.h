class MyPickingResultsPanel : public PickingResultsPanel
{
	public:
		MyPickingResultsPanel( wxWindow* parent );
		~MyPickingResultsPanel();

		void OnProjectOpen();
		void OnProjectClose();

		void OnDefineFilterClick( wxCommandEvent& event );
		void OnUpdateUI( wxUpdateUIEvent& event );
		void OnValueChanged(wxDataViewEvent &event);
		void FillGroupComboBox();
		void FillBasedOnSelectCommand(wxString wanted_command);
		int ReturnRowFromAssetID(int asset_id, int start_location = 0);
		void FillResultsPanelAndDetails(int row, int column);
		void UpdateResultsFromBitmapPanel(const int row, const int column);
		void UpdateResultsFromBitmapPanel();
		void OnNextButtonClick( wxCommandEvent& event );
		void OnPreviousButtonClick( wxCommandEvent& event );
		void OnAddToGroupClick( wxCommandEvent& event );
		void OnRemoveFromGroupClick( wxCommandEvent& event );
		void OnJobDetailsToggle( wxCommandEvent& event );
		void Clear();
		bool CheckBoxIsChecked(const int row, const int column);

		int GetFilter();
		void OnAllMoviesSelect( wxCommandEvent& event );
		void OnByFilterSelect( wxCommandEvent& event );
		void OnCharHook( wxKeyEvent& event );


		int *picking_job_ids;
		int number_of_picking_jobs;
		int *per_row_asset_id;
		int *per_row_array_position;
		int number_of_assets;

		int selected_row;
		int selected_column;

		bool doing_panel_fill;

		bool is_dirty;
		bool group_combo_is_dirty;

		wxString current_fill_command;

};

