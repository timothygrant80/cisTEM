/* \brief 	Object to handle Gatan's Digital Micrograph files
 *
 * Adapted from Bernard Heynman's Bsoft
 *
 */
#include "core_headers.h"
#include <iostream>
#include <iomanip>



DMFile::DMFile()
{
	version = 0;
	show = 0;
	level = 0;
	sb = 0;
	endianness = 1;
	keep = 0;
}

DMFile::DMFile(std::string wanted_filename, bool overwrite)
{
	version = 0;
	show = 0;
	level = 0;
	sb = 0;
	endianness = 1;
	keep = 0;
	OpenFile(wanted_filename);
}

DMFile::DMFile(wxString wanted_filename)
{
	version = 0;
	show = 0;
	level = 0;
	sb = 0;
	endianness = 1;
	keep = 0;
	OpenFile(wanted_filename.ToStdString());
}


DMFile::~DMFile()
{
	CloseFile();
	true;
}

bool DMFile::OpenFile(std::string wanted_filename, bool overwrite)
{
	MyDebugAssertFalse(overwrite,"Overwriting is not supported for DM files");
	unsigned char *fake_pointer;
	readDM(wxString(wanted_filename), fake_pointer, false);
	filename = wxString(wanted_filename);

	// TODO: return false if something is fishy about this file
	return true;
}

void DMFile::CloseFile()
{
	filename = "";
}

bool DMFile::IsOpen(){
	return filename != "";
}

void DMFile::ReadSlicesFromDisk(int start_slice, int end_slice, float *output_array)
{
	MyDebugAssertTrue(start_slice==end_slice, "Can only read single slice at a time from DM files. Sorry");
	ReadSliceFromDisk(start_slice,output_array);
}

// Wanted_slice should be numbered from 0. It's assumed output_array is already allocated to correct dimensions
void DMFile::ReadSliceFromDisk(int wanted_slice, float *output_array)
{
	MyDebugAssertTrue(wanted_slice == 0, "Bad start_slice: %i.\n", wanted_slice);
	MyDebugAssertTrue(wanted_slice < n,"Can't read slice %i. There are only %i.\n",wanted_slice,n);


	unsigned char *p;



	// Allocate a temporary array to hold data from disk
	p = new unsigned char[alloc_size()];



	// Read data from disk
	readDM(filename,p,true,wanted_slice);


	// Apply mirror operation to achieve the same data layout as IMOD's dm2mrc
	long counter = 0;
	long mirror_address = 0;
	int i,j;
	for (j=0; j < int(sizeY()); j++)
	{
		mirror_address = (sizeY() - 1 - j) * sizeX();
		for (i=0; i < int(sizeX()); i++)
		{
			output_array[counter] = reinterpret_cast < float * > (p)[mirror_address];
			counter++;
			mirror_address++;
		}
	}

	delete [] p;
}


/**
@brief	Reading a Digital Micrograph image file format.
@param	*p				the image structure.
@param 	readdata		flag to activate reading of image data.
@param 	img_select		image selection in multi-image file (-1 = all images).
@return	int				error code (<0 means failure).
A 2D/3D image format used with CCD cameras in electron microscopy.
	File format extensions:  	.dm, .DM
	Two types: Fixed format (new) and the Macintosh format (old)
	Fixed format:
		Header size:				24 bytes (fixed).
		Byte order determination:	An endian flag: Must be 65535 or swap everything
		Data types: 				many.
	Macintosh format: 			Hermitian
		Header size:				8 bytes (fixed).
		Byte order determination:	Big-endian
		Data types: 				many.
**/
int DMFile::readDM(wxString wanted_filename, unsigned char *p, bool readdata, int img_select)
{
	std::ifstream*		fimg = new std::ifstream(wanted_filename);
	if ( fimg->fail() ) return -1;

	// Read a small block at the beginning to see if it is the fixed or tagged format
	char		buf[1024];
	fimg->read(buf, 4);
	if ( fimg->fail() ) return -2;

	version = *((int *) buf);

	if ( version > 100 ) version = buf[3];

	if ( show == 1 ) MyDebugPrint("Magic number = %i\n",version);

	fimg->seekg(0, std::ios::beg);

	switch ( version ) {
		case 0: readFixedDMHeader(fimg, p, readdata); break;
		case 3:
		case 4: readTagGroupWithVersion(fimg, p, readdata, img_select); break;
		default:
			MyDebugAssertFalse(true,"Digital Micrograph format version %i not supported!\n",version);
			wxPrintf("Digital Micrograph format version %i not supported!\n",version);
			abort;
	}

	fimg->close();
	delete fimg;

	return 0;
}


int			DMFile::readFixedDMHeader(std::ifstream* fimg, unsigned char* p, bool readdata)
{
	DMhead*		header = new DMhead;
	DMMachead*	macheader = (DMMachead *) header;

//	if ( fread( header, sizeof(DMhead), 1, fimg ) < 1 ) return -2;
	fimg->read((char *)header, sizeof(DMhead));
	if ( fimg->fail() ) return -2;

	if ( show ) MyDebugPrint("DEBUG readFixedDMHeader: macheader= %i width = %i height = %i\n", macheader, macheader->width, macheader->height);

	data_offset(8);
	if ( ( macheader->width < 1 ) || ( macheader->height < 1 ) ) {
		macheader = (DMMachead *) ((char*)header + 6);
		data_offset(data_offset() + 6);
	}

	if ( show ) MyDebugPrint("DEBUG readFixedDMHeader: macheader= %i width = %i height = %i\n", macheader, macheader->width, macheader->height);

	// Determine header type
    int     	i, fixed(1), sb(0);
/*	if ( macheader->width == 0 && macheader->height == -1 ) {
		fixed = 1;
		sb = 0;
	} else if ( macheader->width == -1 && macheader->height == 0 ) {
		fixed = 1;
		sb = 1;
	} else if ( systype(0) >= LittleIEEE ) sb = 1;
*/
    // Swap bytes if necessary
    unsigned char*   	b = (unsigned char *) header;
    if ( sb ) {
		if ( fixed ) {
	    	if ( show ) MyDebugPrint("Warning: Swapping header byte order for 4-byte types\n");
    		for ( i=0; i<24; i+=4 ) swapbytes(b+i, 4);
		} else {
			b = (unsigned char *) macheader;
			if ( show ) MyDebugPrint("Warning: Swapping header byte order for 2-byte types\n");
    		for ( i=0; i<8; i+=2 ) swapbytes(b+i, 2);
		}
    }

	// Map the parameters
	images(1);
	channels(1);
	compound_type(TSimple);
	if ( fixed ) {
		size(header->xSize, header->ySize, header->zSize);
		data_offset(24);
		datatype_from_dm3_type(header->type);
	} else {
		size(macheader->width, macheader->height, 1);
    	switch( macheader->type ) {
        	case 6 :  data_type(UCharacter); break;
        	case 14 : data_type(UCharacter); break;
        	case 9 :  data_type(SCharacter); break;
        	case 10 : data_type(UShort); break;
        	case 1 :  data_type(Short); break;
        	case 7 :  data_type(Integer); break;
        	case 2 :  data_type(Float); break;
        	case 3 :  data_type(Float); compound_type(TComplex); channels(2); break;
        	case 13 : data_type(Double); compound_type(TComplex); channels(2); break;
        	default : data_type(UCharacter);
		}
	}

	//p->image = new Bsub_image[p->images()];

	delete header;

	if ( readdata ) {
		MyDebugPrintWithDetails("About to read %li bytes\n",long(alloc_size()));
		fimg->read((char *)p, alloc_size());
		if ( fimg->fail() ) return -3;
		if ( sb ) swapbytes(alloc_size(), p, data_type_size());
	}

	fimg->close();

	return sb;
}

/*
	Main DM3/4 file database:

	Arrangement:
		The file header (TagGroupWithVersion) contains 4 items:
		Version: 1-4 (4 byte)
		Size of contained data (TagGroupData): v1-v3 (4 byte), v4 (8 byte)
		Endianness: 0,1 (4 byte)
		Content (TagGroupData)

	Important image information:
		Main block: "ImageList"
			May contain more than one image, usually a small display version
			followed by the real data
		Important elements:
			Pixel size (nm):		ImageData/Calibrations/Dimension/Scale
			Data element type:		ImageData/DataType
			Image size:				ImageData/Dimensions
			Data element size:		ImageData/PixelDepth
			Camera pixel size (um):	ImageTags/Acquisition/Device/CCD/Pixel Size (um)
			Dose rate:				ImageTags/Calibration/Dose rate/Calibration
			Magnification:			ImageTags/Microscope Info/Actual Magnification
			Acceleration voltage:	ImageTags/Microscope Info/Voltage
*/
int			DMFile::readTagGroupWithVersion(std::ifstream* fimg, unsigned char* p, bool readdata, int img_select)
{
	//if ( verbose & VERB_DEBUG_DM ) show = 2;

	keep = level = 0;

	size_t			file_length(0), val(0);
	char			buf[1024];

	fimg->read(buf, 4);
	if ( fimg->fail() ) return -2;

	version = *((int *) buf);

	if ( version > 100 ) {
		sb = 1;
		version = buf[3];
	}

	file_length = dm_read_integer(fimg, sizeof(size_t));	// File length
	file_length += 16;

	endianness = dm_read_integer(fimg, sizeof(unsigned int));	// Endianness

	if ( show == 1 ) {
		if ( sb ) std::cout << "Warning: Swapping header bytes" << endl;
		std::cout << "Version: " << version << endl;
		std::cout << "File length: " << file_length << endl;
		std::cout << "Endianness: " << endianness << endl;
	}

	/*
	if ( file_length <= 16 ) {
		std::cout << "Error: file length = " << file_length << endl;
		return error_show("File length specifier incorrect!\n", __FILE__, __LINE__);
	}
	*/
	MyDebugAssertTrue(file_length > 16,"Error: file length specifier incorrect. File length = %i\n",file_length);

	if ( show == 2 ) {
		std::cout << "<?xml version=\"1.0\" encoding=\"UTF-8\"?>" << endl;
		std::cout << "<!DOCTYPE plist PUBLIC \"-//Apple Computer//DTD PLIST 1.0//EN\" \"http://www.apple.com/DTDs/PropertyList-1.0.dtd\">" << endl;
		std::cout << "<plist version=\"1.0\">" << endl;
	}

	// Set up default parameters
	size(1, 1, 1);
	images(1);
	channels(1);

	readTagGroupData(fimg, 0, p, readdata);

	val = dm_read_integer(fimg, sizeof(unsigned int));	// 500
	val = dm_read_integer(fimg, sizeof(unsigned int));	// Label size
	if ( val ) {
		fimg->read(buf, val);
		if ( show == 2 ) std::cout << "<" << buf << "/>" << endl;
	}

	if ( show == 1 ) std::cout << endl;
	if ( show == 2 ) std::cout << "</plist>" << endl;

	if ( img_select >= 0 ) {
		data_offset(data_offset()+img_select*sizeX()*sizeY()*sizeZ()*channels()*data_type_size());
		images(1);
	}

	/*
	if ( !p->image ) {
		if ( images() < 1 ) images(1);
		p->image = new Bsub_image[p->images()];
	}
	*/

	if ( show_scale() != 1 ) {
		sampling(samplingX()/show_scale(), samplingY()/show_scale(), 1);
		show_scale(1);
	}

	if ( readdata ) {
		//MyDebugPrintWithDetails("About to read %li bytes\n",long(alloc_size()));
		fimg->seekg(data_offset(), std::ios_base::beg);
		//p->data_alloc();
		//fimg->read((char *)p->data_pointer(), p->alloc_size());
		fimg->read((char *)p, alloc_size());
	}

	return version;
}

// TagGroupData contains 3 items
// Sorted flag: 0,1; v3-v4 (1 byte)
// Open flag: 0,1; v3-v4 (1 byte)
// Number of records (TagEntry/TagSubGroup): v3 (4 byte), v4 (8 byte)
// Content (TagGroupData)
int			DMFile::readTagGroupData(std::ifstream* fimg, int dim_flag, unsigned char* p, bool readdata)
{
unsigned char	sorted, open;
size_t			i, ntag(0);
int				notag(0);

*fimg >> sorted;
*fimg >> open;

ntag = dm_read_integer(fimg, sizeof(size_t));	// Number of tags

if ( show == 1 ) std::cout << "\tntag=" << ntag;
if ( show == 2 ) {
	for ( i=0; i<level; i++ ) std::cout << "\t";
	std::cout << "<dict>" << endl;
}

for ( i=0; i<ntag; i++ ) {
	readTag(fimg, dim_flag, p, readdata, notag);
	if ( dim_flag ) dim_flag++;
}

if ( show == 2 ) {
	for ( i=0; i<level; i++ ) std::cout << "\t";
	std::cout << "</dict>" << endl;
}

return 0;
}



/*
	Image data is assigned to the data pointer based on the keep flag
	Associated properties are also kept
*/
int			DMFile::readTag(std::ifstream* fimg, int dim_flag, unsigned char* p, bool readdata, int& notag)
{
	unsigned char	t, tag[128];
	unsigned short	len;
	size_t			i, j, k, data_size(0);
	size_t			nnum, size(0), nstr, narr, nel, dt, dtarr[128], dtlen;	// Change in v4
	double			val, arr[256];
	char			buf[1024];

	level++;

	// Common part of TagSubGroup and TagEntry records
	// Selector (1 byte): TagSubGroup=20, TagEntry=21
	// Length of label (2 bytes)
	// Label
	*fimg >> t;
	len = dm_read_integer(fimg, sizeof(unsigned short));
//	cout << "\tlen=" << len << endl;

	if ( len > 128 ) {
		std::cout << "\tlen=" << len << endl;
		std::cerr << "Error: tag length too long!" << endl;
		abort();
	}

	if ( len ) {
		fimg->read((char *)tag, len);
		tag[len] = 0;
		tag_convert(tag);
		notag = 0;
	} else {
		snprintf((char *)tag, 128, "%d", notag);
		notag++;
	}

	if ( show == 1 ) {
		std::cout << endl;
		for ( i=0; i<level; i++ ) std::cout << "\t";
		std::cout << "Tag=" << (int)t << "\tlen=" << len << "\t" << tag;
	}
	if ( show == 2 ) {
		for ( i=0; i<level; i++ ) std::cout << "\t";
		std::cout << "<key>" << tag << "</key>" << endl;
	}

	if ( version == 4 )	{	// Changed in v4
		size = dm_read_integer(fimg, sizeof(size_t));	// Size of data type & data
		if ( show == 1 )
			std::cout << "\tsize=" << size;
	}

	if ( t == 20 ) {			// TagSubGroup record
		if ( strcmp((char *)tag, "Dimensions") == 0 ) dim_flag = 1;
		readTagGroupData(fimg, dim_flag, p, readdata);
	} else if ( t == 21 ) {		// TagEntry record
		fimg->read((char *)buf, 4);		// String = "%%%%"
		buf[4] = 0;
		nnum = dm_read_integer(fimg, sizeof(size_t));	// Number of integers in array
		if ( show == 1 ) std::cout << "\t" << std::setw(4) << buf << "\tnnum=" << nnum;

		for ( i=0; i<nnum; i++ ) {
			dtarr[i] = dm_read_integer(fimg, sizeof(size_t));	// Data type array
			if ( show == 1 ) std::cout << "\tdt[" << i << "]=" << dtarr[i];
		}
		if ( show == 1 ) std::cout << endl;

		dt = dtarr[0];
		if ( dt <= 12 ) {
			val = dm3_value(fimg, dt);
		} else if ( dt == 15 ) {										// Struct
			narr = dtarr[2];
			nel = 1;
			if ( show == 1 )
				std::cout << "\tstruct=" << narr << "x" << nel;
			if ( show == 2 ) {
				for ( i=0; i<level; i++ ) std::cout << "\t";
				std::cout << "<array>" << endl;
			}
			level++;
			for ( k=0; k<narr; k++ )
				arr[k] = dm3_value(fimg, dtarr[4+2*k]);
			level--;
			if ( show == 2 ) {
				for ( i=0; i<level; i++ ) std::cout << "\t";
				std::cout << "</array>" << endl;
			}
		} else if ( dt == 18 ) {	// String
			fimg->read((char *)buf, dtarr[1]);
			if ( show == 1 ) std::cout << "\tstrlen=" << dtarr[1];
			if ( show == 2 ) {
				for ( i=0; i<level; i++ ) std::cout << "\t";
				std::cout << "<string>" << buf << "</string>" << endl;
			}
		} else if ( dt == 20 ) {	// Data array
			if ( dtarr[1] == 15 ) narr = dtarr[3];
			else narr = 1;
			nel = dtarr[nnum-1];		// Number of elements
			if ( show == 1 )
				std::cout << "\tarr=" << narr << "x" << nel;
			if ( show == 2 ) {
				for ( i=0; i<level; i++ ) std::cout << "\t";
				std::cout << "<array>" << endl;
			}
			level++;
			if ( dtarr[1] == 15 ) {	// Struct
				if ( show == 1 )
					std::cout << "\tstruct=" << narr << "x" << nel;
				if ( show == 2 ) {
					for ( i=0; i<level; i++ ) std::cout << "\t";
					std::cout << "<array>" << endl;
				}
				level++;
				for ( i=0; i<nel; i++ )
					for ( k=0; k<narr; k++ )
						arr[k] = dm3_value(fimg, dtarr[5+2*k]);
				level--;
				if ( show == 2 ) {
					for ( i=0; i<level; i++ ) std::cout << "\t";
					std::cout << "</array>" << endl;
				}
			} else {
				narr = 1;
				dtlen = dm3_type_length(dtarr[1]);
				data_size = nel*dtlen;
				// The big image data block is not printed for the plist format
				if ( strcmp((char *)tag, "Data") == 0 && nel > 1e6 ) {
					keep = 1;
					if ( show == 1 )
						std::cout << "\tdata_size=" << data_size;
//					if ( readdata ) {
//						p->data_alloc(data_size);
//						fimg->read((char *)p->data_pointer(), data_size);
//					} else {
//						fimg->seekg(data_size, ios_base::cur);
//					}
					data_offset(fimg->tellg());
					fimg->seekg(data_size, std::ios_base::cur);
				} else {
					for ( i=0; i<nel; i++ ) {
						if ( i < 10 ) val = dm3_value(fimg, dtarr[1]);	// Element value
						else fimg->read((char *)buf, dtlen);			// Element value
					}
				}
			}
			level--;
			if ( show == 2 ) {
				for ( i=0; i<level; i++ ) std::cout << "\t";
				std::cout << "</array>" << endl;
			}
		} else {
			std::cerr << "Error: Data type " << dt << " not defined!" << endl;
		}

		if ( keep && strcmp((char *)tag, "ImageData") == 0 ) keep = 0;

		if ( keep ) {
			if ( strcmp((char *)tag, "DataType") == 0 ) {
				datatype_from_dm3_type((DMDataType)val);
				if ( show == 1 )
					std::cout << "\tdatatype=" << data_type() << "\tcompoundtype=" <<
						compound_type() << "\tc=" << channels() << "\tval=" << val;
			}
			if ( strcmp((char *)tag, "Minimum Value (counts)") == 0 ) {
				minimum(val);
			}
			if ( strcmp((char *)tag, "Maximum Value (counts)") == 0 ) {
				maximum(val);
			}
			if ( dim_flag == 1 ) {
				sizeX((size_t) val);
				dim_flag = 2;
			} else if ( dim_flag == 2 ) {
				sizeY((size_t) val);
				dim_flag = 3;
			} else if ( dim_flag == 3 ) {
				images((size_t) val);
				dim_flag = 0;
				if ( show == 1 ) std::cout << "\tx=" << sizeX() << "\ty=" << sizeY() << "\tn=" << images();
			}
			if ( strcmp((char *)tag, "PixelDepth") == 0 ) {
				if ( show == 1 ) std::cout << "\t" << tag << "=" << val << endl;
			}
			if ( strcmp((char *)tag, "Pixel Size (um)") == 0 ) {
				sampling(arr[0], arr[1], 1);
				if ( show == 1 ) std::cout << "\t" << tag << "=" << samplingX() << samplingY() << samplingZ() << endl;
//				cout << tab << tag << "=" << p->sampling() << endl;
			}
			if ( strcmp((char *)tag, "Actual Magnification") == 0 ) {
				show_scale(val/1e4);
				if ( show == 1 ) std::cout << "\t" << tag << "=" << val << endl;
//				cout << tab << tag << "=" << val << endl;
			}
			if ( strcmp((char *)tag, "Pixel Upsampling") == 0 ) {
				if ( show == 1 ) std::cout << "\t" << "\t" << "=" << arr[0] << "x" << arr[1] << endl;
			}
			if ( strcmp((char *)tag, "SourceSize_Pixels") == 0 ) {
				if ( show == 1 ) std::cout << "\t" << "\t" << "=" << arr[0] << "x" << arr[1] << endl;
			}
//			if ( strcmp(tag, "ImageIndex") == 0 ) {
//				if ( show == 1 ) cout << tab << tag << "=" << val << endl;
//			}
			if ( strstr((char *)tag, "Emission") ) {
				if ( show == 1 ) {
					std::cout << "\t" << tag << "=" << val << endl;
					for ( i=0; i<len; i++ ) std::cout << (int)tag[i] << "\t" << tag[i] << endl;
				}
			}
		}
	} else {
		std::cerr << "Error: Undefined tag type! (" << t << ")" << endl;
	}

	level--;

	return 0;
}

double		DMFile::dm3_value(std::ifstream* fimg, int dm3_type)
{
	int			dtlen = dm3_type_length(dm3_type);
	if ( dtlen < 1 ) return 0;

	long		i, ivalue(0);
	double		dvalue(0);
	char		buf[1024];

	fimg->read(buf, dtlen);

	if ( sb - endianness ) swapbytes((unsigned char *)buf, dtlen);

	switch ( dm3_type ) {
		case 2: ivalue = *((short *) buf); break;
		case 3: ivalue = *((int *) buf); break;
		case 4: ivalue = *((unsigned short *) buf); break;
		case 5: ivalue = *((unsigned int *) buf); break;
		case 6: dvalue = *((float *) buf); break;
		case 7: dvalue = *((double *) buf); break;
		case 8: ivalue = *buf; break;
		case 9: ivalue = *buf; break;
		case 10: ivalue = *buf; break;
		case 11: ivalue = *((long *) buf); break;
		case 12: ivalue = *((unsigned long *) buf); break;
		default:
			std::cerr << "Error: Data type " << dm3_type << " not defined!" << endl;
	}

	if ( !dvalue ) dvalue = ivalue;

	if ( show == 1 ) std::cout << "\t" << dvalue;

	if ( show == 2 ) {
		for ( i=0; i<level; i++ ) std::cout << "\t";
		if ( dm3_type == 6 || dm3_type == 7 )
			std::cout << "<real>" << dvalue << "</real>" << endl;
		else
			std::cout << "<integer>" << ivalue << "</integer>" << endl;
	}

	return dvalue;
}

int			DMFile::dm3_type_length(int dm3_type)
{
	int			dtlen = 0;

	switch ( dm3_type ) {
		case 2: dtlen = 2; break;
		case 3: dtlen = 4; break;
		case 4: dtlen = 2; break;
		case 5: dtlen = 4; break;
		case 6: dtlen = 4; break;
		case 7: dtlen = 8; break;
		case 8: dtlen = 1; break;
		case 9: dtlen = 1; break;
		case 10: dtlen = 1; break;
		case 11: dtlen = 8; break;
		case 12: dtlen = 8; break;
		default:
			std::cerr << "Error: Data type " << dm3_type << " length not defined!" << endl;
	}

	return dtlen;
}

unsigned long	DMFile::dm_read_integer(std::ifstream* fimg, long len)
{
	unsigned short		sval(0);
	unsigned int		ival(0);
	unsigned long		lval(0);
	char				buf[1024];

	if ( version < 4 && len > 4 ) len = 4;

	fimg->read(buf, len);

	if ( sb ) swapbytes((unsigned char *)buf, len);

	switch ( len ) {
		case 2: sval = *((unsigned short *) buf); lval = sval; break;
		case 4: ival = *((unsigned int *) buf); lval = ival; break;
		case 8: lval = *((unsigned long *) buf); break;
		default: lval = *buf;
	}

	return lval;
}

int			DMFile::tag_convert(unsigned char* tag)
{
	for ( ; *tag; tag++ ) if ( *tag > 127 ) *tag -= 64;

	return 0;
}


DataType	DMFile::datatype_from_dm3_type(DMDataType dm3_type)
{
	if ( show ) MyDebugPrint("\tDMDataType= %i\n",dm3_type);

	compound_type(TSimple);
	channels(1);

	DataType		datatype = Unknown_Type;

	switch( dm3_type ) {
		case BINARY_DATA :         datatype = UCharacter; break;
		case UNSIGNED_INT8_DATA :  datatype = UCharacter; break;
		case SIGNED_INT8_DATA :    datatype = SCharacter; break;
		case RGB_DATA :
		case RGB_UINT8_DATA :
			datatype = UCharacter;
			channels(3);
			compound_type(TRGB);
			break;
		case OS_RGBA_UINT8_DATA :
			datatype = UCharacter;
			channels(4);
			compound_type(TRGBA);
			break;
		case UNSIGNED_INT16_DATA : datatype = UShort; break;
		case RGB_UINT16_DATA :
			datatype = UShort;
			channels(3);
			compound_type(TRGB);
			break;
		case RGBA_UINT16_DATA :
			datatype = UShort;
			channels(4);
			compound_type(TRGBA);
			break;
		case SIGNED_INT16_DATA :   datatype = Short; break;
		case UNSIGNED_INT32_DATA : datatype = UInteger; break;
		case SIGNED_INT32_DATA :   datatype = Integer; break;
		case REAL4_DATA :          datatype = Float; break;
		case RGBA_FLOAT32_DATA :
			datatype = Float;
			channels(4);
			compound_type(TRGBA);
			break;
		case REAL8_DATA :          datatype = Double; break;
		case RGB_FLOAT64_DATA :
			datatype = Double;
			channels(3);
			compound_type(TRGB);
			break;
		case RGBA_FLOAT64_DATA :
			datatype = Double;
			channels(4);
			compound_type(TRGBA);
			break;
		case COMPLEX8_DATA :       datatype = Float; compound_type(TComplex); break;
		case COMPLEX16_DATA :      datatype = Double; compound_type(TComplex); break;
		default : datatype = UCharacter;
	}

	data_type(datatype);
	if ( compound_type() == TComplex ) channels(2);

	return datatype;
}

/**
@brief 	Returns the size of the datatype in bytes.
@return size_t			data type size.

	The Bit type returns 1.

**/
size_t	DMFile::data_type_size()
{
	size_t	typesize(0);

	switch ( datatype ) {
		case Bit:
		case UCharacter: case SCharacter:	typesize = sizeof(char); break;
		case UShort: case Short:typesize = sizeof(short); break;
		case UInteger: case Integer: 	typesize = sizeof(int); break;
		case ULong: case Long: 	typesize = sizeof(long); break;
		case Float:				typesize = sizeof(float); break;
		case Double:			typesize = sizeof(double); break;
		default: typesize = 0;
	}

	return typesize;
}

/**
@brief 	Swaps bytes.
@param	*v 			a pointer to the bytes.
@param 	n			number of bytes to swap.

	Byte swapping is done in place.

**/
void		swapbytes(unsigned char* v, size_t n)
{
	unsigned char	t;
	size_t	i;

	for ( i=0, n--; i<n; i++, n-- ) {
		t = v[i];
		v[i] = v[n];
		v[n] = t;
	}
}

/**
@brief 	Swaps bytes.
@param 	size		size of the block to be swapped.
@param 	*v 			a pointer to the bytes.
@param 	n			number of bytes to swap.

	Byte swapping is done in place.

**/
void		swapbytes(size_t size, unsigned char* v, size_t n)
{
	if ( n < 2 ) return;

	MyDebugPrintWithDetails("DEBUG swapbytes: size = %i n= %i\n",size,n);

	size_t	i;

	for ( i=0; i<size; i+=n, v+=n ) swapbytes(v, n);
}

