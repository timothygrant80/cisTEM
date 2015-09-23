bool GetMRCDetails(const char *filename, int &x_size, int &y_size, int &number_of_images);

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

void SendwxStringToSocket(wxString *string_to_send, wxSocketBase *socket);
wxString ReceivewxStringFromSocket(wxSocketBase *socket);


