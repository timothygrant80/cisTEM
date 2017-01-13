class TiffFile : public AbstractImageFile {
private:
	TIFF * tif;
	int	logical_dimension_x;
	int logical_dimension_y;
	int number_of_images;

	void ReadLogicalDimensionsFromDisk();

public:
	TiffFile();
	TiffFile(std::string wanted_filename, bool overwrite = false);
	~TiffFile();

	inline int ReturnXSize() {return logical_dimension_x;};
	inline int ReturnYSize() {return logical_dimension_y;};
	inline int ReturnZSize() {return number_of_images;};
	inline int ReturnNumberOfSlices() {return number_of_images;};

	inline bool IsOpen() {if (tif) {return true;} else { return false;}};

	void OpenFile(std::string filename, bool overwrite = false);
	void CloseFile();

	void ReadSliceFromDisk(int slice_number, float *output_array);
	void ReadSlicesFromDisk(int start_slice, int end_slice, float *output_array);

	void WriteSliceToDisk(int slice_number, float *input_array);
	void WriteSlicesToDisk(int start_slice, int end_slice, float *input_array);

	void PrintInfo();
};
