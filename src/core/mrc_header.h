/*  \brief  Object for manipulating MRC headers..

	// MRC format from http://www2.mrc-lmb.cam.ac.uk/image2000.html

	 Map/Image Header Format

	Length = 1024 bytes, organized as 56 LONG words followed
	by space for 10 80 byte text labels.

	1 	NX 	number of columns (fastest changing in map)
	2 	NY 	number of rows
	3 	NZ 	number of sections (slowest changing in map)
	4 	MODE
	data type : 	0 	image : signed 8-bit bytes range -128 to 127 (actually UNSIGNED)
	x				1 	image : 16-bit halfwords
					2 	image : 32-bit reals
					3 	transform : complex 16-bit integers
					4 	transform : complex 32-bit reals
	5 	NXSTART 	number of first column in map (Default = 0)
	6 	NYSTART 	number of first row in map
	7 	NZSTART 	number of first section in map
	8 	MX 	number of intervals along X
	9 	MY 	number of intervals along Y
	10 	MZ 	number of intervals along Z
	11-13 	CELLA 	cell dimensions in angstroms
	14-16 	CELLB 	cell angles in degrees
	17 	MAPC 	axis corresp to cols (1,2,3 for X,Y,Z)
	18 	MAPR 	axis corresp to rows (1,2,3 for X,Y,Z)
	19 	MAPS 	axis corresp to sections (1,2,3 for X,Y,Z)
	20 	DMIN 	minimum density value
	21 	DMAX 	maximum density value
	22 	DMEAN 	mean density value
	23 	ISPG 	space group number 0 or 1 (default=0)
	24 	NSYMBT 	number of bytes used for symmetry data (0 or 80)
	25-49 	EXTRA 	extra space used for anything - 0 by default
	50-52 	ORIGIN 	origin in X,Y,Z used for transforms
	53 	MAP 	character string 'MAP ' to identify file type
	54 	MACHST 	machine stamp
	55 	RMS 	rms deviation of map from mean density
	56 	NLABL 	number of labels being used
	57-256 	LABEL(20,10) 	10 80-character text labels

	Symmetry records follow - if any - stored as text as in International Tables, operators separated by and grouped into 'lines' of 80 characters (ie. symmetry operators do not cross the ends of the 80-character 'lines' and the 'lines' do not terminate in a ).

	Data records follow.
*/

//
enum MRCDataTypes { MRCByte,
                    MRCInteger,
                    MRCFloat,
                    MRC4Bit };

class MRCHeader {

    char* buffer; // !< The true byte data

    //  The following are all pointers and just point to the relevant area of the buffer..

    int*   nx; // !< number of columns (fastest changing in map)
    int*   ny; // !< number of rows
    int*   nz; // !< NZ 	number of sections (slowest changing in map)
    int*   mode; // !< MODE (data type) 0 = signed 8-bit bytes range -128 to 127 (actually UNSIGNED), 1 = 16-bit halfwords, 2 = 32-bit reals, 3 = complex 16-bit integers, 4 = complex 32-bit reals
    int*   nxstart; // !< number of first column in map (Default = 0)
    int*   nystart; // !< number of first row in map
    int*   nzstart; // !< number of first section in map
    int*   mx; // !< number of intervals along X
    int*   my; // !< number of intervals along Y
    int*   mz; // !< number of intervals along Z
    float* cell_a_x; // !< cell dimensions in angstroms (X)
    float* cell_a_y; // !< cell dimensions in angstroms (Y)
    float* cell_a_z; // !< cell dimensions in angstroms (Z)
    float* cell_b_x; // !< cell angles in degrees (X)
    float* cell_b_y; // !< cell angles in degrees (Y)
    float* cell_b_z; // !< cell angles in degrees (Z)
    int*   map_c; // !< axis corresp to cols (1,2,3 for X,Y,Z)
    int*   map_r; // !< axis corresp to rows (1,2,3 for X,Y,Z)
    int*   map_s; // !< axis corresp to sections (1,2,3 for X,Y,Z)
    float* dmin; // !< minimum density value
    float* dmax; // !< maximum density value
    float* dmean; // !< mean density value
    int*   space_group_number; // !< space group number 0 or 1 (default=0)
    int*   symmetry_data_bytes; // !< number of bytes used for symmetry data (0 or 80)
    int*   extra; // !< extra space used for anything - 0 by default (100 bytes)
    int*   imodStamp; // 1146047817 indicates that file was created by IMOD or other software that uses bit flags in the following field
    int*   imodFlags; // Bit flags:
            // 1 = bytes are stored as signed
            // 2 = pixel spacing was set from size in extended header
            // 4 = origin is stored with sign inverted from definition
            //     below
            // 8 = RMS value is negative if it was not computed
            // 16 = Bytes have two 4-bit values, the first one in the
            //      low 4 bits and the second one in the high 4 bits
    float* origin_x; // !< origin in X used for transforms
    float* origin_y; // !< origin in Y used for transforms
    float* origin_z; // !< origin in Z used for transforms
    char*  map; // !< character string 'MAP ' to identify file type
    int*   machine_stamp; // !< machine stamp
    float* rms; // !< rms deviation of map from mean density
    int*   number_of_labels_used; // !< number of labels being used
    char*  labels; // !< Labels. 10 80-character text labels

    // some extra info..

    float bytes_per_pixel;
    bool  pixel_data_are_signed;
    int   pixel_data_are_of_type;
    bool  pixel_data_are_complex;

    bool this_is_in_mastronarde_4bit_hack_format;

  public:
    // methods

    MRCHeader( );
    ~MRCHeader( );
    void InitPointers( );

    void ReadHeader(std::fstream* MRCFile);
    void WriteHeader(std::fstream* MRCFile);
    void BlankHeader( );

    void ResetLabels( );
    void ResetOrigin( );

    void SetLocalMachineStamp( );

    void PrintInfo( );

    inline int ReturnDimensionX( ) { return nx[0]; };

    inline int ReturnDimensionY( ) { return ny[0]; };

    inline int ReturnDimensionZ( ) { return nz[0]; };

    inline int ReturnMapC( ) { return map_c[0]; };

    inline int ReturnMapR( ) { return map_r[0]; };

    inline int ReturnMapS( ) { return map_s[0]; };

    inline bool ReturnIfThisIsInMastronarde4BitHackFormat( ) { return this_is_in_mastronarde_4bit_hack_format; }

    inline bool PixelDataAreSigned( ) { return pixel_data_are_signed; }

    float ReturnPixelSize( );
    void  SetPixelSize(float wanted_pixel_size);

    void SetDimensionsImage(int wanted_x_dim, int wanted_y_dim);
    void SetDimensionsVolume(int wanted_x_dim, int wanted_y_dim, int wanted_z_dim);
    void SetNumberOfImages(int wanted_number_of_images);
    void SetNumberOfVolumes(int wanted_number_of_volumes);
    void SetDensityStatistics(float wanted_min, float wanted_max, float wanted_mean, float wanted_rms);
    void SetOrigin(float wanted_x, float wanted_y, float wanted_z);

    inline float BytesPerPixel( ) { return bytes_per_pixel; };

    inline int Mode( ) { return mode[0]; };

    inline int SymmetryDataBytes( ) { return symmetry_data_bytes[0]; };
};
