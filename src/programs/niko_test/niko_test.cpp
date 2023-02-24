#include "../../core/core_headers.h"

class
        NikoTestApp : public MyApp {

  public:
    bool DoCalculation( );
    void DoInteractiveUserInput( );

  private:
};

IMPLEMENT_APP(NikoTestApp)

// override the DoInteractiveUserInput

void NikoTestApp::DoInteractiveUserInput( ) {
}

// override the do calculation method which will be what is actually run..

bool NikoTestApp::DoCalculation( ) {
    EulerSearch  global_euler_search;
    ParameterMap parameter_map;
    parameter_map.SetAllTrue( );
    global_euler_search.InitGrid("C1", 2.5, 0.0f, 50.0f, 360.0, 10.0, 0.0, 1.0 / 3.0, parameter_map, 20);
    global_euler_search.theta_max = 180.0f;
    global_euler_search.CalculateGridSearchPositions(false);
    wxPrintf("Number of search positions: %i\n", global_euler_search.number_of_search_positions);
    //for ( int i = 0; i < global_euler_search.number_of_search_positions; i++ ) {
    //    wxPrintf("Search position %i: %f %f %f\n", i, global_euler_search.list_of_search_parameters[i][0], global_euler_search.list_of_search_parameters[i][1], global_euler_search.list_of_search_parameters[i][2]);
    //}
    global_euler_search.for_mt = true;
    global_euler_search.InitGrid("C1", 2.5, 20.0f, 80.0f, 360.0, 10.0, 0.0, 1.0 / 3.0, parameter_map, 20);
    global_euler_search.theta_max = 100.0f;
    global_euler_search.phi_max   = 15.0f;
    global_euler_search.CalculateGridSearchPositions(false);
    wxPrintf("Number of search positions: %i\n", global_euler_search.number_of_search_positions);
    for ( int i = 0; i < global_euler_search.number_of_search_positions; i++ ) {
        wxPrintf("Search position %i: %f %f %f\n", i, global_euler_search.list_of_search_parameters[i][0], global_euler_search.list_of_search_parameters[i][1], global_euler_search.list_of_search_parameters[i][2]);
    }

    return true;
}
