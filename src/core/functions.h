bool GetMRCDetails(const char *filename, long &x_size, long &y_size, long &number_of_images);

inline bool IsEven(int number_to_check)
{
	  if ( number_to_check % 2== 0 ) return true;
	  else return false;
};

inline bool DoesFileExist(std::string filename)
{
    std::ifstream file_to_check (filename.c_str());

    if(file_to_check.is_open()) return true;
    return false;
};

