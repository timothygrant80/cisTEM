class ClassificationSelection {

  public:
    ClassificationSelection( );
    ~ClassificationSelection( );

    long       selection_id;
    wxString   name;
    wxDateTime creation_date;
    long       refinement_package_asset_id;
    long       classification_id;
    int        number_of_classes;
    int        number_of_selections;

    wxArrayLong selections;
};

WX_DECLARE_OBJARRAY(ClassificationSelection, ArrayofClassificationSelections);
