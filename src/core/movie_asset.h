#include <wx/string.h>
#include <wx/filename.h>

class MovieAsset {
    
  public:

	long x_size;
	long y_size;
	long number_of_frames;
	
	wxFileName filename;
	
	double pixel_size;
	double microscope_voltage;
	double spherical_aberration;
	double dose_per_frame;
	double total_dose;
	
	bool movie_is_valid;
	
	MovieAsset();
	MovieAsset(wxString wanted_filename);
	~MovieAsset();
	
	void Update(wxString wanted_filename);
	void Recheck_if_valid();
	void CopyFrom(MovieAsset *other_asset);long FindMember(long member_to_find);

	wxString ReturnFullPathString();
	wxString ReturnShortNameString();



};


class MovieAssetList {
	
	long number_allocated;

public:
	
	MovieAssetList();
	~MovieAssetList();

	MovieAsset *assets;

	long number_of_assets;
	long ReturnNumberOfAssets();
	
	void AddMovie(MovieAsset *asset_to_add);
	void RemoveMovie(long number_to_remove);
	
	void RemoveAll();

	long FindFile(wxFileName file_to_find);

};
