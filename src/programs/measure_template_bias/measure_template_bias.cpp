#include "../../core/core_headers.h"

class
        MeasureTemplateBiasApp : public MyApp {

  public:
    bool DoCalculation( );
    void DoInteractiveUserInput( );

  private:
};

IMPLEMENT_APP(MeasureTemplateBiasApp)

// override the DoInteractiveUserInput

void MeasureTemplateBiasApp::DoInteractiveUserInput( ) {
    wxString input_reconstruction_full_template;
    wxString input_reconstruction_omit_template;
    wxString input_diff_template;
    wxString input_full_template;
    wxString input_omit_template;
    bool input_diff_map;

    UserInput* my_input = new UserInput("MeasureTemplateBias", 1.0);

    input_diff_map = my_input->GetYesNoFromUser("Input template diff map?", "If No, the difference map will be calculated from the full and omit templates", "No");

    if ( input_diff_map == true ) {
    	input_diff_template = my_input->GetFilenameFromUser("Input diff template", "The difference map, full - omit template", "diff_template.mrc", true);
    } else{
        input_full_template = my_input->GetFilenameFromUser("Input full template", "The 3D map of the full template", "full_template.mrc", true);
        input_omit_template = my_input->GetFilenameFromUser("Input omit template", "The 3D map of the omit template", "omit_template.mrc", true);
    }
    input_reconstruction_full_template = my_input->GetFilenameFromUser("Input reconstruction full template", "The 3D reconstruction calculated from targets found with the full template", "reconstruction_full_template.mrc", true);
    input_reconstruction_omit_template = my_input->GetFilenameFromUser("Input reconstruction omit template", "The 3D reconstruction calculated from targets found with the omit template", "reconstruction_omit_template.mrc", true);
    delete my_input;

    my_current_job.ManualSetArguments("bttttt",
    		input_diff_map,
			input_diff_template.ToUTF8( ).data( ),
			input_full_template.ToUTF8( ).data( ),
			input_omit_template.ToUTF8( ).data( ),
    		input_reconstruction_full_template.ToUTF8( ).data( ),
    		input_reconstruction_omit_template.ToUTF8( ).data( ));
}

// override the do calculation method which will be what is actually run..

bool MeasureTemplateBiasApp::DoCalculation( ) {
    int   i, j;
    long  count1 = 0;
    float correlation_3ds;
    float correlation_templates;
//    float sigma_diff_template;
//    float sigma_full_template;
//    float sigma_omit_template;
    float sum_3d_full = 0.0f;
    float sum_3d_omit = 0.0f;
    float sum_difference = 0.0f;
//    float mask_radius;
    float max_value;
    Image input_3d_full;
	Image input_3d_omit;
	Image input_diff;
	Image input_full;
	Image input_omit;
//	Image mask_3d;

    bool input_diff_map = my_current_job.arguments[0].ReturnBoolArgument( );
    wxString input_diff_template = my_current_job.arguments[1].ReturnStringArgument( );
    wxString input_full_template = my_current_job.arguments[2].ReturnStringArgument( );
    wxString input_omit_template = my_current_job.arguments[3].ReturnStringArgument( );
    wxString input_reconstruction_full_template = my_current_job.arguments[4].ReturnStringArgument( );
    wxString input_reconstruction_omit_template = my_current_job.arguments[5].ReturnStringArgument( );

    MRCFile input_file_3d_full(input_reconstruction_full_template.ToStdString( ), false);
    MRCFile input_file_3d_omit(input_reconstruction_omit_template.ToStdString( ), false);
    if ( input_diff_map == true ) {
        MRCFile input_file_diff(input_diff_template.ToStdString( ), false);
        if ( (input_file_3d_full.ReturnXSize( ) != input_file_diff.ReturnXSize( )) || (input_file_3d_full.ReturnYSize( ) != input_file_diff.ReturnYSize( ))  || (input_file_3d_full.ReturnZSize( ) != input_file_diff.ReturnZSize( )) ) {
            MyPrintWithDetails("Error: Input maps do not have the same dimensions\n");
            DEBUG_ABORT;
        }
        input_diff.Allocate(input_file_diff.ReturnXSize( ), input_file_diff.ReturnYSize( ), input_file_diff.ReturnZSize( ), true);
        // Making sure that the memory reserved for FFTs is also set to zero
        input_diff.SetToConstant(0.0f);
        input_diff.ReadSlices(&input_file_diff, 1, input_file_diff.ReturnZSize( ));
    } else {
        MRCFile input_file_full(input_full_template.ToStdString( ), false);
        MRCFile input_file_omit(input_omit_template.ToStdString( ), false);
        if ( (input_file_3d_full.ReturnXSize( ) != input_file_full.ReturnXSize( )) || (input_file_3d_full.ReturnYSize( ) != input_file_full.ReturnYSize( ))  || (input_file_3d_full.ReturnZSize( ) != input_file_full.ReturnZSize( )) ) {
            MyPrintWithDetails("Error: Input maps do not have the same dimensions\n");
            DEBUG_ABORT;
        }
        if ( (input_file_3d_full.ReturnXSize( ) != input_file_omit.ReturnXSize( )) || (input_file_3d_full.ReturnYSize( ) != input_file_omit.ReturnYSize( ))  || (input_file_3d_full.ReturnZSize( ) != input_file_omit.ReturnZSize( )) ) {
            MyPrintWithDetails("Error: Input maps do not have the same dimensions\n");
            DEBUG_ABORT;
        }
        input_full.Allocate(input_file_full.ReturnXSize( ), input_file_full.ReturnYSize( ), input_file_full.ReturnZSize( ), true);
        input_omit.Allocate(input_file_omit.ReturnXSize( ), input_file_omit.ReturnYSize( ), input_file_omit.ReturnZSize( ), true);
        // Making sure that the memory reserved for FFTs is also set to zero
        input_full.SetToConstant(0.0f);
        input_omit.SetToConstant(0.0f);
        input_full.ReadSlices(&input_file_full, 1, input_file_full.ReturnZSize( ));
        input_omit.ReadSlices(&input_file_omit, 1, input_file_omit.ReturnZSize( ));
    }

    if ( (input_file_3d_full.ReturnXSize( ) != input_file_3d_omit.ReturnXSize( )) || (input_file_3d_full.ReturnYSize( ) != input_file_3d_omit.ReturnYSize( ))  || (input_file_3d_full.ReturnZSize( ) != input_file_3d_omit.ReturnZSize( )) ) {
        MyPrintWithDetails("Error: Input maps do not have the same dimensions\n");
        DEBUG_ABORT;
    }

    input_3d_full.Allocate(input_file_3d_full.ReturnXSize( ), input_file_3d_full.ReturnYSize( ), input_file_3d_full.ReturnZSize( ), true);
    input_3d_omit.Allocate(input_file_3d_omit.ReturnXSize( ), input_file_3d_omit.ReturnYSize( ), input_file_3d_omit.ReturnZSize( ), true);
//    mask_3d.Allocate(input_file_3d_full.ReturnXSize( ), input_file_3d_full.ReturnYSize( ), input_file_3d_full.ReturnZSize( ), true);
    // Making sure that the memory reserved for FFTs is also set to zero
    input_3d_full.SetToConstant(0.0f);
    input_3d_omit.SetToConstant(0.0f);
//    mask_3d.SetToConstant(0.0f);

    input_3d_full.ReadSlices(&input_file_3d_full, 1, input_file_3d_full.ReturnZSize( ));
    input_3d_omit.ReadSlices(&input_file_3d_omit, 1, input_file_3d_omit.ReturnZSize( ));

//    mask_radius = std::min(std::min(float(input_file_3d_full.ReturnXSize( )), float(input_file_3d_full.ReturnYSize( ))), float(input_file_3d_full.ReturnZSize( )));

    wxPrintf("\nStarting calculation...\n");

    if ( input_diff_map == true ) {
//        sigma_diff_template = sqrtf(input_diff.ReturnVarianceOfRealValues(mask_radius/2.0f, 0.0f, 0.0f, 0.0f, true));
        max_value = input_diff.ReturnAverageOfMaxN(100);
        for ( long address = 0; address < input_diff.real_memory_allocated; address++ ) {
            if ( input_diff.real_values[address] > max_value / 10.0f) {
           		sum_3d_full += input_3d_full.real_values[address];
           		sum_3d_omit += input_3d_omit.real_values[address];
           		sum_difference += input_3d_full.real_values[address] - input_3d_omit.real_values[address];
//    			mask_3d.real_values[address] = 1.0f;
           		count1++;
            }
        }
    } else {
//        sigma_full_template = sqrtf(input_full.ReturnVarianceOfRealValues( ));
//        sigma_omit_template = sqrtf(input_omit.ReturnVarianceOfRealValues( ));
//        max_value = 0.0f;
        for ( long address = 0; address < input_full.real_memory_allocated; address++ ) {
        	input_full.real_values[address] -= input_omit.real_values[address];
//        	max_value = std::max(max_value, input_full.real_values[address] - input_omit.real_values[address]);
        }
        max_value = input_full.ReturnAverageOfMaxN(100);
//	    wxPrintf("\nSigma full template = %g\n", sigma_full_template);
//	    wxPrintf("\nSigma omit template = %g\n", sigma_omit_template);
/*        for ( long address = 0; address < input_full.real_memory_allocated; address++ ) {
            if ( input_full.real_values[address] < sigma_full_template / 10.0f ) {
            	input_3d_full.real_values[address] = 0.0f;
            	input_3d_omit.real_values[address] = 0.0f;
            	input_full.real_values[address] = 0.0f;
            	input_omit.real_values[address] = 0.0f;
            }
        }
*/
        for ( long address = 0; address < input_full.real_memory_allocated; address++ ) {
            if ( input_full.real_values[address] > max_value / 10.0f) {
//            if ( input_full.real_values[address] > sigma_full_template) {
//            	if ( input_omit.real_values[address] / input_full.real_values[address] < 0.5f ) {
            		sum_3d_full += input_3d_full.real_values[address];
            		sum_3d_omit += input_3d_omit.real_values[address];
//        			wxPrintf("\nAddress, value1, value 2 = %li, %g, %g\n", address, input_full.real_values[address], input_omit.real_values[address]);
            		sum_difference += input_3d_full.real_values[address] - input_3d_omit.real_values[address];
//        			mask_3d.real_values[address] = 1.0f;
            		count1++;
//            	} else {
//            		input_3d_full.real_values[address] = 0.0f;
//            		input_3d_omit.real_values[address] = 0.0f;
//            	}
            }
        }
    }

/*    wxPrintf("\nWriting out map 1...\n");
    input_3d_full.QuickAndDirtyWriteSlices("diff_full.mrc", 1, input_3d_full.logical_z_dimension);
    wxPrintf("\nWriting out map 2...\n");
    input_3d_omit.QuickAndDirtyWriteSlices("diff_omit.mrc", 1, input_3d_omit.logical_z_dimension);
    wxPrintf("\nWriting out map 3...\n");
    input_full.QuickAndDirtyWriteSlices("full.mrc", 1, input_full.logical_z_dimension);
    wxPrintf("\nWriting out map 4...\n");
    input_omit.QuickAndDirtyWriteSlices("omit.mrc", 1, input_omit.logical_z_dimension);
*/
//    wxPrintf("\nWriting out mask... Number of voxels set: %li\n", count1);
//    mask_3d.QuickAndDirtyWriteSlices("mask.mrc", 1, mask_3d.logical_z_dimension);

    correlation_3ds = input_3d_full.ReturnCorrelationCoefficientUnnormalized(input_3d_omit, 0.0f)/sqrtf(input_3d_full.ReturnVarianceOfRealValues( ))/sqrtf(input_3d_omit.ReturnVarianceOfRealValues( ));
    if ( input_diff_map == false ) {
    	correlation_templates = input_full.ReturnCorrelationCoefficientUnnormalized(input_omit, 0.0f)/sqrtf(input_full.ReturnVarianceOfRealValues( ))/sqrtf(input_omit.ReturnVarianceOfRealValues( ));
    }

    wxPrintf("\nAverage of densities of full reconstruction = %g\n", sum_3d_full / count1);
    wxPrintf("\nAverage of densities of omit reconstruction = %g\n", sum_3d_omit / count1);
    wxPrintf("\nAverage of difference densities = %g\n", sum_difference / count1);
    wxPrintf("\nCorrelation coefficient of reconstructions = %g\n", correlation_3ds);
    if ( input_diff_map == false ) {
    	wxPrintf("\nCorrelation coefficient of templates = %g\n", correlation_templates);
    	wxPrintf("\nRatio of correlation coefficients = %g\n", correlation_3ds/correlation_templates);
    }
    wxPrintf("\n\nDegree of bias = %g\n\n", ((sum_3d_full - sum_3d_omit) / sum_3d_full));

    wxPrintf("\n\nMeasureTemplateBias: Normal termination\n\n");

    return true;
}
