bool GetMRCDetails(const char *filename, int &x_size, int &y_size, int &number_of_images);

inline void ZeroBoolArray(bool *array_to_zero, int size_of_array)
{
	for (int counter = 0; counter < size_of_array; counter++)
	{
		array_to_zero[counter] = false;
	}
};

inline void ZeroIntArray(int *array_to_zero, int size_of_array)
{
	for (int counter = 0; counter < size_of_array; counter++)
	{
		array_to_zero[counter] = 0;
	}
};

inline void ZeroFloatArray(float *array_to_zero, int size_of_array)
{
	for (int counter = 0; counter < size_of_array; counter++)
	{
		array_to_zero[counter] = 0.0;
	}
};

inline void ZeroDoubleArray(double *array_to_zero, int size_of_array)
{
	for (int counter = 0; counter < size_of_array; counter++)
	{
		array_to_zero[counter] = 0.0;
	}
};

inline bool IsEven(int number_to_check)
{
	  if ( number_to_check % 2== 0 ) return true;
	  else return false;
};

inline bool DoesFileExist(wxString filename)
{
    std::ifstream file_to_check (filename.c_str());

    if(file_to_check.is_open()) return true;
    return false;
};

inline float rad_2_deg(float radians)
{
  return radians / (PI / 180.);
}

inline float deg_2_rad(float degrees)
{
  return degrees * PI / 180.;
}

inline float sinc(float radians)
{
	if (radians == 0.0) return 1.0;
	if (radians >= 0.01) return sinf(radians) / radians;
	float temp_float = radians * radians;
	return 1.0 - temp_float / 6.0 + temp_float * temp_float / 120.0;
}

inline double myround(double a)
{
	if (a > 0) return double(long(a + 0.5));
	else return double(long(a - 0.5));
};

inline float myround(float a)
{
	if (a > 0) return float(int(a + 0.5));	else return float(int(a - 0.5));
};

inline bool IsOdd(int number)
{
	if ((number & 1) == 0) return false;
	else return true;
};

wxString ReturnIPAddress();
wxString ReturnIPAddressFromSocket(wxSocketBase *socket);

void SendwxStringToSocket(wxString *string_to_send, wxSocketBase *socket);
wxString ReceivewxStringFromSocket(wxSocketBase *socket);

inline float ReturnPhaseFromShift(float real_space_shift, float distance_from_origin, float dimension_size)
{
	return real_space_shift * distance_from_origin * 2.0 * PI / dimension_size;
};

inline fftw_complex Return3DPhaseFromIndividualDimensions( float phase_x, float phase_y, float phase_z)
{
	float temp_phase = -phase_x-phase_y-phase_z;

	return cos(temp_phase) + sin(temp_phase) * I;
}

inline bool DoublesAreAlmostTheSame(double a, double b)
{
	return (fabs(a-b) < 0.000001);
}

inline bool InputIsATerminal()
{
   return isatty(fileno(stdin));
};

inline bool OutputIsAtTerminal()
{
    return isatty(fileno(stdout));
};


/*
 *
 * String manipulations
 *
 */
std::string FilenameReplaceExtension(std::string filename, std::string new_extension);
std::string FilenameAddSuffix(std::string filename, std::string suffix_to_add);
