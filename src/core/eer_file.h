class EerFile {
private:
	TIFF * tif;
	FILE * fh;
	int logical_dimension_x;
	int logical_dimension_y;
	int bits_per_rle;
	int number_of_frames;
	unsigned long long * frame_starts;
	unsigned long long * frame_sizes;
	unsigned char * buf;
	//unsigned int * ion_of_each_frame;
	unsigned long long file_size;
	unsigned long long frame_size_bits;




public:
	EerFile();
	EerFile(std::string wanted_filename, bool overwrite = false);
	~EerFile();
	void ReadLogicalDimensionsFromDisk();
	inline int ReturnXSize() {return logical_dimension_x;};
	inline int ReturnYSize() {return logical_dimension_y;};
	inline int ReturnZSize() {return number_of_frames;};
	inline int ReturnNumberOfSlices() {return number_of_frames;};
	inline bool IsOpen() {if (tif) {return true;} else { return false;}};

	bool OpenFile(std::string filename, bool overwrite = false, bool wait_for_file_to_exist = false, bool check_only_the_first_image = false);
	void CloseFile();
	void ReadSlicesFromDisk();

	void PrintInfo();
	void rleFrames(std::string output_file, int super_res_factor = 1, int temporal_frame_bin_factor = 1, wxString *output_sum_filename = NULL);
};
