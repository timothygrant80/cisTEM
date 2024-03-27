#include "core_headers.h"

ImageFile::ImageFile( ) {
    filename         = wxFileName("");
    file_type        = UNSUPPORTED_FILE_TYPE;
    file_type_string = "Unsupported file type";
}

ImageFile::ImageFile(std::string wanted_filename, bool overwrite) {
    OpenFile(wanted_filename, overwrite);
}

ImageFile::~ImageFile( ) {
    CloseFile( );
}

void ImageFile::SetFileTypeFromExtension( ) {
    wxString ext = filename.GetExt( );
    if ( ext.IsSameAs("tif") || ext.IsSameAs("tiff") ) {
        file_type        = TIFF_FILE;
        file_type_string = "TIFF";
    }
    else if ( ext.IsSameAs("mrc") || ext.IsSameAs("mrcs") || ext.IsSameAs("ccp4") ) {
        file_type        = MRC_FILE;
        file_type_string = "MRC";
    }
    else if ( ext.IsSameAs("dm3") || ext.IsSameAs("dm4") || ext.IsSameAs("dm") ) {
        file_type        = DM_FILE;
        file_type_string = "DM";
    }
    else if ( ext.IsSameAs("eer") ) {
        file_type        = EER_FILE;
        file_type_string = "EER";
    }
    else {
        // FIXME: Change for display prog:
        // We will perform quick check on MRC format so any properly formatted MRC type with any extension
        // can be opened. Maybe a bit ugly, so alternatives should potentially be considered.
        // GetMRCDetails is also wanted as a member of the MRCFile class, and this method relies on the function
        // being a non-member
        int x_size, y_size, number_of_frames;
        if ( GetMRCDetails(filename.GetFullPath( ), x_size, y_size, number_of_frames) ) {
            file_type        = MRC_FILE;
            file_type_string = "MRC";
        }
        else {
            file_type        = UNSUPPORTED_FILE_TYPE;
            file_type_string = "Unsupported file type";
        }
    }
}

bool ImageFile::OpenFile(std::string wanted_filename, bool overwrite, bool wait_for_file_to_exist, bool check_only_the_first_image, int eer_super_res_factor, int eer_frames_per_image) {
    bool file_seems_ok = false;
    filename           = wanted_filename;
    SetFileTypeFromExtension( );
    // FIXME:
    // The default eer_frames_per_image is being grabbed from the GUI even for tif files.
    // This causes an encoding error and crash. It really should be fixed there, however,
    // By overriding it here, all a user has to do is rename their files with eer extension to get around this hack.
    if ( file_type != EER_FILE ) {
        eer_frames_per_image = 0;
    }
    switch ( file_type ) {
        case TIFF_FILE: file_seems_ok = tiff_file.OpenFile(wanted_filename, overwrite, wait_for_file_to_exist, check_only_the_first_image, eer_super_res_factor, eer_frames_per_image); break;
        case MRC_FILE: file_seems_ok = mrc_file.OpenFile(wanted_filename, overwrite, wait_for_file_to_exist, check_only_the_first_image, eer_super_res_factor, eer_frames_per_image); break;
        case DM_FILE: file_seems_ok = dm_file.OpenFile(wanted_filename, overwrite, wait_for_file_to_exist, check_only_the_first_image, eer_super_res_factor, eer_frames_per_image); break;
        case EER_FILE: file_seems_ok = eer_file.OpenFile(wanted_filename, overwrite, wait_for_file_to_exist, check_only_the_first_image, eer_super_res_factor, eer_frames_per_image); break;
        default:
            MyPrintWithDetails("Unsupported file type\n");
            MyDebugAssertTrue(false, "Unsupported file type: %s\n", filename.GetFullPath( ).ToStdString( ));
            DEBUG_ABORT;
            break;
    }
    return file_seems_ok;
}

void ImageFile::CloseFile( ) {
    switch ( file_type ) {
        case TIFF_FILE: tiff_file.CloseFile( ); break;
        case MRC_FILE: mrc_file.CloseFile( ); break;
        case DM_FILE: dm_file.CloseFile( ); break;
        case EER_FILE: eer_file.CloseFile( ); break;
    }
}

void ImageFile::ReadSliceFromDisk(int slice_number, float* output_array) {
    ReadSlicesFromDisk(slice_number, slice_number, output_array);
}

void ImageFile::ReadSlicesFromDisk(int start_slice, int end_slice, float* output_array) {
    switch ( file_type ) {
        case TIFF_FILE: tiff_file.ReadSlicesFromDisk(start_slice, end_slice, output_array); break;
        case MRC_FILE: mrc_file.ReadSlicesFromDisk(start_slice, end_slice, output_array); break;
        case DM_FILE: dm_file.ReadSlicesFromDisk(start_slice - 1, end_slice - 1, output_array); break;
        case EER_FILE: eer_file.ReadSlicesFromDisk(start_slice, end_slice, output_array); break;
        default:
            MyPrintWithDetails("Unsupported file type\n");
            DEBUG_ABORT;
            break;
    }
}

void ImageFile::WriteSliceToDisk(int slice_number, float* input_array) {
    WriteSlicesToDisk(slice_number, slice_number, input_array);
}

void ImageFile::WriteSlicesToDisk(int start_slice, int end_slice, float* input_array) {
    switch ( file_type ) {
        case TIFF_FILE: tiff_file.WriteSlicesToDisk(start_slice, end_slice, input_array); break;
        case MRC_FILE: mrc_file.WriteSlicesToDisk(start_slice, end_slice, input_array); break;
        case DM_FILE: dm_file.WriteSlicesToDisk(start_slice, end_slice, input_array); break;
        case EER_FILE: eer_file.WriteSlicesToDisk(start_slice, end_slice, input_array); break;
        default:
            MyPrintWithDetails("Unsupported file type\n");
            DEBUG_ABORT;
            break;
    }
}

int ImageFile::ReturnXSize( ) {
    switch ( file_type ) {
        case TIFF_FILE: return tiff_file.ReturnXSize( ); break;
        case MRC_FILE: return mrc_file.ReturnXSize( ); break;
        case DM_FILE: return dm_file.ReturnXSize( ); break;
        case EER_FILE: return eer_file.ReturnXSize( ); break;
        default:
            MyPrintWithDetails("Unsupported file type\n");
            DEBUG_ABORT;
            break;
    }
    return -1;
}

int ImageFile::ReturnYSize( ) {
    switch ( file_type ) {
        case TIFF_FILE: return tiff_file.ReturnYSize( ); break;
        case MRC_FILE: return mrc_file.ReturnYSize( ); break;
        case DM_FILE: return dm_file.ReturnYSize( ); break;
        case EER_FILE: return eer_file.ReturnYSize( ); break;
        default:
            MyPrintWithDetails("Unsupported file type\n");
            DEBUG_ABORT;
            break;
    }
    return -1;
}

int ImageFile::ReturnZSize( ) {
    switch ( file_type ) {
        case TIFF_FILE: return tiff_file.ReturnZSize( ); break;
        case MRC_FILE: return mrc_file.ReturnZSize( ); break;
        case DM_FILE: return dm_file.ReturnZSize( ); break;
        case EER_FILE: return eer_file.ReturnZSize( ); break;
        default:
            MyPrintWithDetails("Unsupported file type\n");
            DEBUG_ABORT;
            break;
    }
    return -1;
}

int ImageFile::ReturnNumberOfSlices( ) {
    switch ( file_type ) {
        case TIFF_FILE: return tiff_file.ReturnNumberOfSlices( ); break;
        case MRC_FILE: return mrc_file.ReturnNumberOfSlices( ); break;
        case DM_FILE: return dm_file.ReturnNumberOfSlices( ); break;
        case EER_FILE: return eer_file.ReturnNumberOfSlices( ); break;
        default:
            MyPrintWithDetails("Unsupported file type\n");
            DEBUG_ABORT;
            break;
    }
    return -1;
}

float ImageFile::ReturnPixelSize( ) {
    switch ( file_type ) {
        case TIFF_FILE: return tiff_file.ReturnPixelSize( ); break;
        case MRC_FILE: return mrc_file.ReturnPixelSize( ); break;
        case DM_FILE: return dm_file.ReturnPixelSize( ); break;
        case EER_FILE: return eer_file.ReturnPixelSize( ); break;
        default:
            MyPrintWithDetails("Unsupported file type\n");
            DEBUG_ABORT;
            break;
    }
    return -1;
}

bool ImageFile::IsOpen( ) {
    switch ( file_type ) {
        case TIFF_FILE: return tiff_file.IsOpen( ); break;
        case MRC_FILE: return mrc_file.IsOpen( ); break;
        case DM_FILE: return dm_file.IsOpen( ); break;
        case EER_FILE: return eer_file.IsOpen( ); break;
        default:
            MyPrintWithDetails("Unsupported file type\n");
            DEBUG_ABORT;
            break;
    }
    return false;
}

void ImageFile::PrintInfo( ) {
    wxPrintf("File name: %s\n", filename.GetFullName( ));
    wxPrintf("File type: %s\n", file_type_string);
    wxPrintf("Dimensions: X = %i Y = %i Z = %i\n", ReturnXSize( ), ReturnYSize( ), ReturnZSize( ));
    wxPrintf("Number of slices: %i\n", ReturnNumberOfSlices( ));
}
