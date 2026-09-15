class ClassificationSelection {

  public:
    ClassificationSelection( );
    ~ClassificationSelection( );

    long       selection_id;
    std::string   name;
    DateTime creation_date;
    long       refinement_package_asset_id;
    long       classification_id;
    int        number_of_classes;
    int        number_of_selections;

    std::vector<long> selections;
};

typedef std::vector<ClassificationSelection> ArrayofClassificationSelections;
