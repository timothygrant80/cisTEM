//  \brief  Object for manipulating MRC Files..
//

class MRCFile {

	public:

	std::fstream my_file;
	MRCHeader my_header;

	bool rewrite_header_on_close;

	MRCFile();
	MRCFile(std::string filename, bool overwrite = false);
	~MRCFile();

	inline int ReturnXSize() {MyDebugAssertTrue(my_file.is_open(), "File not open!");	return my_header.nx[0];};
	inline int ReturnYSize() {MyDebugAssertTrue(my_file.is_open(), "File not open!");	return my_header.ny[0];};
	inline int ReturnZSize() {MyDebugAssertTrue(my_file.is_open(), "File not open!");	return my_header.nz[0];};
	inline int ReturnNumberOfSlices() {MyDebugAssertTrue(my_file.is_open(), "File not open!");	return my_header.nz[0];};


	inline void SetXSize(int wanted_x_size) {MyDebugAssertTrue(my_file.is_open(), "File not open!");	my_header.nx[0] = wanted_x_size;};
	inline void SetYSize(int wanted_y_size) {MyDebugAssertTrue(my_file.is_open(), "File not open!");	my_header.ny[0] = wanted_y_size;};
	inline void SetZSize(int wanted_z_size) {MyDebugAssertTrue(my_file.is_open(), "File not open!");	my_header.nz[0] = wanted_z_size;};
	inline void SetNumberOfSlices(int wanted_z_size) {MyDebugAssertTrue(my_file.is_open(), "File not open!");	my_header.nz[0] = wanted_z_size;};



	void OpenFile(std::string filename, bool overwrite = false);
	void CloseFile();

	inline void ReadSliceFromDisk(int slice_number, float *output_array) {ReadSlicesFromDisk(slice_number, slice_number, output_array);}
	void ReadSlicesFromDisk(int start_slice, int end_slice, float *output_array);

	inline void WriteSliceToDisk(int slice_number, float *input_array) {WriteSlicesToDisk(slice_number, slice_number, input_array);}
	void WriteSlicesToDisk(int start_slice, int end_slice, float *input_array);

	inline void WriteHeader() {my_header.WriteHeader(&my_file);};

};

