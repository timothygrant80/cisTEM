/*
A class to help with temporary files.
Provides destructors that clear the file from drive.
Inherited by specific sub classes in the same folder
*/

#include "../../../core/core_headers.h"
class TestFile {
  

public:

  ~TestFile(void) {
    if (filePath.IsNull() || filePath.IsEmpty() || !filePath || filePath.Len() == 0) { return; }
  
    if (!filePath.IsNull() && !filePath.IsEmpty()) {
     wxPrintf("\tDeleting %s... ", filePath.mb_str());
      const int result = remove(filePath.mb_str());
  
      if (result == 0) {
        wxPrintf("success.\n");
      } else {
        wxPrintf("error\n%s\n", strerror(errno)); // No such file or directory
      }
    }
  }
  
  wxString filePath;
};