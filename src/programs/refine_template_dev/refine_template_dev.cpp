#include "../../core/core_headers.h"
#include "./refine_template_dev.h"

class
        RefineTemplateDevApp : public MyApp {
  public:
    bool DoCalculation( );
    void DoInteractiveUserInput( );

  private:
};

IMPLEMENT_APP(RefineTemplateDevApp)

// override the DoInteractiveUserInput

void RefineTemplateDevApp::DoInteractiveUserInput( ) {
    /* RefineTemplateArguments arguments;
    arguments.GetFromUser( );
    arguments.SetForLocalJob( ); */
}

// override the do calculation method which will be what is actually run..

bool RefineTemplateDevApp::DoCalculation( ) {
    wxDateTime start_time = wxDateTime::Now( );
    /*RefineTemplateArguments arguments;
    arguments.Recieve( );

    wxPrintf(arguments.value("input_starfile")); */
    return true;
}
