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
	void rleFrames();
	void ReadCoordinateFromRle1(unsigned int ion_number, unsigned char * rle_in_each_frame, unsigned char * subpixels_in_each_frame);
	void ReadCoordinateFromRle2(unsigned int ion_number, unsigned char * rle_in_each_frame, unsigned char * subpixels_in_each_frame);
};
