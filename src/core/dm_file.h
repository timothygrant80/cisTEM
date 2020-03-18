/* \brief 	Object to handle Gatan's Digital Micrograph files
 *
 * Adapted from Bernard Heynman's Bsoft
 *
 */
void		swapbytes(unsigned char* v, size_t n);
void		swapbytes(size_t size, unsigned char* v, size_t n);

/**
@enum 	CompoundType
@brief 	Compound data type specifier.

	This determines what compound data type is used in an image.
**/
enum CompoundType {
	TSimple = 0,		// Single value data type
	TComplex = 1,		// 2-value complex type
	TVector3 = 3,		// 3-value vector type
	TView = 4,			// 4-value view type
	TRGB = 10,			// Red-green-blue interleaved
	TRGBA = 11,			// Red-green-blue-alpha interleaved
	TCMYK = 12,			// Cyan-magenta-yellow-black interleaved
	TMulti = 99			// Arbitrary number of channels
} ;

enum DMDataType {
	NULL_DATA, SIGNED_INT16_DATA, REAL4_DATA, COMPLEX8_DATA, OBSELETE_DATA,	// 4
	PACKED_DATA, UNSIGNED_INT8_DATA, SIGNED_INT32_DATA, RGB_DATA,			// 8
	SIGNED_INT8_DATA, UNSIGNED_INT16_DATA, UNSIGNED_INT32_DATA, REAL8_DATA,	// 12
	COMPLEX16_DATA, BINARY_DATA, RGBA_FLOAT32_DATA, RGB_UINT16_DATA ,		// 16
	RGB_FLOAT64_DATA, RGBA_FLOAT64_DATA, RGBA_UINT16_DATA,					// 19
	RGB_UINT8_DATA , RGBA_UINT8_DATA, LAST_DATA, OS_RGBA_UINT8_DATA			// 23
} ;


#ifndef _datatype_
/**
@enum	DataType
@brief 	Base data type specifier.

	This determines what simple data type is used in an image.
**/
enum DataType {
	Unknown_Type = 0,	// Undefined data type
	Bit = 1,			// Bit/binary type
	UCharacter = 2,		// Unsigned character or byte type
	SCharacter = 3,		// Signed character
	UShort = 4,			// Unsigned integer (2-byte)
	Short = 5,			// Signed integer (2-byte)
	UInteger = 6,		// Unsigned integer (4-byte)
	Integer = 7,		// Signed integer (4-byte)
	ULong = 8,			// Unsigned integer (4 or 8 byte, depending on system)
	Long = 9,			// Signed integer (4 or 8 byte, depending on system)
	Float = 10,			// Floating point (4-byte)
	Double = 11,		// Double precision floating point (8-byte)
} ;
#define _datatype_
#endif


class DMFile : public AbstractImageFile {

private:
	wxString 		filename;
	int				version;
	int 			show;
	int				level;
	int				sb;
	int				endianness;
	int				keep;
	size_t			offset; 		// Data offset
	size_t			n;				// Number of images
	size_t			c; 				// Number of channels
	CompoundType	compoundtype;	// Compound data type
	size_t			x, y, z; 		// Dimensions, xyz
	DataType 		datatype;		// Base data type
	size_t 			px, py, pz; 	// Page dimensions
	double 			ux, uy, uz;		// Voxel units (angstrom/pixel edge)
	double 			ss;				// Display scale
	double 			min, max;		// Limits
	float			pixel_size;

	int				readFixedDMHeader(std::ifstream* fimg, unsigned char* p, bool readdata = true);
	int				readTagGroupWithVersion(std::ifstream* fimg, unsigned char* p, bool readdata = true, int img_select = -1);
	DataType		datatype_from_dm3_type(DMDataType dm3_type);
	unsigned long	dm_read_integer(std::ifstream* fimg, long len);
	int				readTagGroupData(std::ifstream* fimg, int dim_flag, unsigned char* p, bool readdata);
	int				readTag(std::ifstream* fimg, int dim_flag, unsigned char* p, bool readdata, int& notag);
	double			dm3_value(std::ifstream* fimg, int dm3_type);
	int				dm3_type_length(int dm3_type);
	int				tag_convert(unsigned char* tag);

	size_t			data_offset() { return offset; }
	void			data_offset(size_t doff) { offset = doff; }
	size_t			images() { return n; }
	void			images(size_t nn) { n = nn; }
	size_t			channels() { return c; }
	void			channels(size_t cc) { c = cc; }
	CompoundType	compound_type() { return compoundtype; }
	void			compound_type(CompoundType ct) { compoundtype = ct; }
	void			size(size_t nx, size_t ny, size_t nz) { x=nx; y=ny; z=nz; }
	void sizeX(size_t nx) {
		x = nx;
	}
	void sizeY(size_t ny) {
		y = ny;
	}
	void sizeZ(size_t nz) {
		z = nz;
	}
	void sampling(double s) {
		if (x > 1)
			ux = s;
		if (y > 1)
			uy = s;
		if (z > 1)
			uz = s;
		//check_sampling();
	}
	void sampling(double sx, double sy, double sz) {
		ux = sx;
		uy = sy;
		uz = sz;
		//check_sampling();
	}
	double samplingX() { return ux; };
	double samplingY() { return uy; };
	double samplingZ() { return uz; };
	double show_scale() {
		return ss;
	}
	void show_scale(double scale) {
		ss = scale;
	}
	DataType data_type() {
		return datatype;
	}
	void data_type(DataType dt) {
		datatype = dt;
	}
	// Data allocation and assignment
	size_t alloc_size() {
		MyDebugAssertFalse(datatype == Bit, "Bit datatype not supported yet. px not set - need to understand Bsoft better to implement this properly. Sorry.\n");
		return (datatype == Bit) ?
				(px / 8) * y * z * n : c * x * y * z * n * data_type_size();
	}

	double minimum() {
		return min;
	}
	double maximum() {
		return max;
	}
	void minimum(double d) {
		min = d;
	}
	void maximum(double d) {
		max = d;
	}

public:
	// Constructors, destructors
	DMFile();
	DMFile(wxString wanted_filename);
	DMFile(std::string filename, bool overwrite = false);
	~DMFile();

	//
	int ReturnXSize() { return x; };
	int ReturnYSize() { return y; };
	int ReturnZSize() { return z; };
	int ReturnNumberOfSlices() { return z; };
	inline float ReturnPixelSize() { return pixel_size; };

	bool IsOpen();

	bool OpenFile(std::string filename, bool overwrite = false, bool wait_for_file_to_exist = false, bool check_only_the_first_image = false);
	void CloseFile();

	void ReadSliceFromDisk(int wanted_slice, float *output_array);
	void ReadSlicesFromDisk(int start_slice, int end_slice, float *output_array);

	void WriteSliceToDisk(int slice_number, float *input_array) { wxPrintf("WriteSliceToDisk not yet implemented for DM files\n"); DEBUG_ABORT; };
	void WriteSlicesToDisk(int start_slice, int end_slice, float *input_array) { wxPrintf("WriteSlicesToDisk not yet implemented for DM files\n"); DEBUG_ABORT; };

	void PrintInfo() { wxPrintf("PrintInfo not yet implemented for DM files\n"); DEBUG_ABORT; };

private:

	//
	int readDM(wxString wanted_filename, unsigned char *p, bool readdata = true, int img_select = -1);

	//
	size_t sizeX() const {
		return x;
	}
	size_t sizeY() const {
		return y;
	}
	size_t sizeZ() const {
		return z;
	}
	size_t data_type_size();
};




struct DMhead {             // file header for DM Fixed Format Specification
	unsigned int endian;	// if not 0x0000FFFF then byte swap (including  next 5 parameters).
	int xSize;
	int ySize;
	int zSize;
	int depth; 				// data type size in bytes - may be determined by data atype below
	DMDataType type; 		// An enumerated value
} ;

struct DMMachead {			// file header for old DM Macintosh Format Specification
	short width;
	short height;
	short bytes_per_pixel;
	short type;
} ;

/* The header information is stored in four short integers (16 bits each).
Each integer is stored with the high byte first (Motorola or 'big-endian' format).
The image data is stored row by row as a continuous stream of data (i.e. no line feeds,
carriage returns, or end of file control characters are present in the file).
The amount of storage space that each pixel in the image occupies depends on the
data type. The header and the image data are stored in the binary format
(not in the ASCII format).

The format for the small header image file is described below.

Field			 Position  Length
Width of image   0  	   2 bytes (integer)
Height of image  2  	   2 bytes (integer)
Bytes per pixel  4  	   2 bytes (integer)
Encoded data type6  	   2 bytes (integer)
Image data  	 8  	   Width * height * bytes per pixel

The encoding of the data type (at position 6) is shown below.

Value Data Type
1	  2 byte signed integer
2	  Floating point IEEE standard 754
3	  8 byte complex floating point (real, imaginary)
5	  Packed complex
6	  1 byte unsigned integer
7	  4 byte signed integer
8	  4 byte RGB
9	  1 byte signed integer
10    2 byte unsigned integer
11    4 byte unsigned integer
12    8 byte double
13    16 byte complex double (real, imaginary)
14    1 byte binary data
*/

struct DM3head {		// All 3 fields must be big-endian
	int version;		// 2/3/4
	int file_length;	// Actually file length - 16 = size of root tag directory
	int endianness;		// 0=big, 1=little
	char sorted;		// Flag for sorted tag (1)
	char open;			// Flag for open tag (1)
	int	ntag;			// Number of tags
} ;

struct DM4head {		// All 3 fields must be big-endian
	int version;		// 2/3/4
	unsigned long file_length;	// Actually file length - 16 = size of root tag directory
	int endianness;		// 0=big, 1=little
	char sorted;		// Flag for sorted tag (1)
	char open;			// Flag for open tag (1)
	int	ntag;			// Number of tags
} ;

