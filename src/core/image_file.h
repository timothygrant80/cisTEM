/*
 * Provide a common interface to interact with all the different
 * files we support (MRC, DM, TIF, etc.)
 */

enum supported_file_types
{
	MRC_FILE,
	TIFF_FILE,
	DM_FILE,
	UNSUPPORTED_FILE_TYPE
};


class ImageFile : public AbstractImageFile {
private:
	// These are the actual file objects doing the work
	MRCFile mrc_file;
	TiffFile tiff_file;
	DMFile dm_file;


	int file_type;
	wxString file_type_string;
	void SetFileTypeFromExtension();

public:
	ImageFile();
	ImageFile(std::string wanted_filename, bool overwrite = false);
	~ImageFile();

	int ReturnXSize();
	int ReturnYSize();
	int ReturnZSize();
	int ReturnNumberOfSlices();

	bool IsOpen();

	void OpenFile(std::string wanted_filename, bool overwrite);
	void CloseFile();

	void ReadSliceFromDisk(int slice_number, float *output_array);
	void ReadSlicesFromDisk(int start_slice, int end_slice, float *output_array);

	void WriteSliceToDisk(int slice_number, float *input_array);
	void WriteSlicesToDisk(int start_slice, int end_slice, float *input_array);

	void PrintInfo();
};


