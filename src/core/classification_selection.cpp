#include "core_headers.h"

ClassificationSelection::ClassificationSelection( ) {
    selection_id                = -1;
    refinement_package_asset_id = -1;
    classification_id           = -1;
    name                        = "New Selection";
    creation_date               = DateTime::Now( );
    number_of_classes           = 0;
    number_of_selections        = 0;
}

ClassificationSelection::~ClassificationSelection( ) {
}
