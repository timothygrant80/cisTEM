#ifndef __ASSETS_H__
#define __ASSETS_H__

class Asset {

public :

	int asset_id;
	int parent_id;
	wxFileName filename;
	bool is_valid;
	wxString asset_name;

	Asset();
	~Asset();

	// pure virtual


	wxString ReturnFullPathString();
	wxString ReturnShortNameString();

};

class MovieAsset : public Asset {

  public:

	MovieAsset();
	MovieAsset(wxString wanted_filename);
	~MovieAsset();

	int position_in_stack;

	int x_size;
	int y_size;
	int number_of_frames;
	

	double pixel_size;
	double microscope_voltage;
	double spherical_aberration;
	double dose_per_frame;
	double total_dose;

	wxString gain_filename;

	double output_binning_factor; // If this is a super-resolution movie, but we never intend to use the "super resolution" part of the spectrum, this factor should be > 1

	bool correct_mag_distortion;
	double mag_distortion_angle;
	double mag_distortion_major_scale;
	double mag_distortion_minor_scale;

	void Update(wxString wanted_filename);
	//void Recheck_if_valid();
	void CopyFrom(Asset *other_asset);
	//long FindMember(long member_to_find);

};

class ImageAsset : public Asset {

  public:

	ImageAsset();
	ImageAsset(wxString wanted_filename);
	~ImageAsset();

	int position_in_stack;
	int alignment_id;
	int ctf_estimation_id;

	int x_size;
	int y_size;

	double pixel_size;
	double microscope_voltage;
	double spherical_aberration;

	void Update(wxString wanted_filename);
	void CopyFrom(Asset *other_asset);
};

class ParticlePositionAsset : public Asset {

  public:

	ParticlePositionAsset();
	ParticlePositionAsset(const float &wanted_x_in_angstroms, const float &wanted_y_in_angstroms);
	~ParticlePositionAsset();

	void Reset();

	int picking_id;
	int pick_job_id;
	int parent_template_id;

	double x_position;
	double y_position;
	double peak_height;
	double template_phi;
	double template_theta;
	double template_psi;

	void CopyFrom(Asset *other_asset);
};

WX_DECLARE_OBJARRAY(ParticlePositionAsset, ArrayOfParticlePositionAssets);

class VolumeAsset : public Asset {

  public:

	VolumeAsset();
	VolumeAsset(wxString wanted_filename);
	~VolumeAsset();

	long reconstruction_job_id;

	int x_size;
	int y_size;
	int z_size;

	double pixel_size;

	void Update(wxString wanted_filename);
	void CopyFrom(Asset *other_asset);
};
/*
class ClassesAsset : public Asset {

	public:

	ClassesAsset();
	~ClassesAsset();

	long classification_job_id;
	long refinement_package_id;

	void CopyFrom(Asset *other_asset);
};*/

class AssetList {

protected :

	long number_allocated;

public :

	AssetList();
	~AssetList();

	long number_of_assets;

	Asset *assets;

	virtual void AddAsset(Asset *asset_to_add) = 0;
	virtual void RemoveAsset(long number_to_remove) = 0;
	virtual void RemoveAll() = 0;
//	virtual long FindFile(wxFileName file_to_find) = 0;
	virtual void CheckMemory() = 0;

	virtual Asset * ReturnAssetPointer(long wanted_asset) = 0;
	virtual MovieAsset * ReturnMovieAssetPointer(long wanted_asset);
	virtual ImageAsset * ReturnImageAssetPointer(long wanted_asset);
	virtual ParticlePositionAsset * ReturnParticlePositionAssetPointer(long wanted_asset);
	virtual VolumeAsset* ReturnVolumeAssetPointer(long wanted_asset);

	virtual int ReturnAssetID(long wanted_asset) = 0;
	virtual wxString ReturnAssetName(long wanted_asset) = 0;
	virtual int ReturnArrayPositionFromID(int wanted_id, int last_found_position = 0) = 0;
	virtual int ReturnArrayPositionFromParentID(int wanted_id) = 0;
	virtual wxString ReturnAssetFullFilename(long wanted_asst) = 0;

	long ReturnNumberOfAssets();
};


class MovieAssetList : public AssetList {

public:
	
	MovieAssetList();
	~MovieAssetList();
	

	Asset * ReturnAssetPointer(long wanted_asset);
	MovieAsset * ReturnMovieAssetPointer(long wanted_asset);

	int ReturnAssetID(long wanted_asset);
	wxString ReturnAssetName(long wanted_asset);
	int ReturnArrayPositionFromID(int wanted_id, int last_found_position = 0 );
	int ReturnArrayPositionFromParentID(int wanted_id);
	wxString ReturnAssetFullFilename(long wanted_asst);

	void AddAsset(Asset *asset_to_add);
	void RemoveAsset(long number_to_remove);
	void RemoveAll();
	long FindFile(wxFileName file_to_find, bool also_check_vs_shortname = false, long max_asset_number_to_check = -1);
	void CheckMemory();

};

class ImageAssetList : public AssetList {

public:

	ImageAssetList();
	~ImageAssetList();


	Asset * ReturnAssetPointer(long wanted_asset);
	ImageAsset * ReturnImageAssetPointer(long wanted_asset);

	int ReturnAssetID(long wanted_asset);
	wxString ReturnAssetName(long wanted_asset);
	int ReturnArrayPositionFromID(int wanted_id, int last_found_position = 0 );
	int ReturnArrayPositionFromParentID(int wanted_id);
	wxString ReturnAssetFullFilename(long wanted_asst);

	void AddAsset(Asset *asset_to_add);
	void RemoveAsset(long number_to_remove);
	void RemoveAll();
	long FindFile(wxFileName file_to_find, bool also_check_vs_shortname = false, long max_asset_number_to_check = -1);
	void CheckMemory();

};

class ParticlePositionAssetList : public AssetList {

public:

	ParticlePositionAssetList();
	~ParticlePositionAssetList();


	Asset * ReturnAssetPointer(long wanted_asset);
	ParticlePositionAsset * ReturnParticlePositionAssetPointer(long wanted_asset);

	int ReturnAssetID(long wanted_asset);
	wxString ReturnAssetName(long wanted_asset) {return wxEmptyString;};
	wxString ReturnAssetFullFilename(long wanted_asst)  {return wxEmptyString;};
	int ReturnArrayPositionFromID(int wanted_id, int last_found_position = 0 );
	int ReturnArrayPositionFromParentID(int wanted_id);

	void AddAsset(Asset *asset_to_add);
	void RemoveAsset(long number_to_remove);
	void RemoveAssetsWithGivenParentImageID(long parent_image_id);
	void RemoveAll();
	void CheckMemory();

};


class VolumeAssetList : public AssetList {

public:

	VolumeAssetList();
	~VolumeAssetList();


	Asset * ReturnAssetPointer(long wanted_asset);
	VolumeAsset * ReturnVolumeAssetPointer(long wanted_asset);

	int ReturnAssetID(long wanted_asset);
	wxString ReturnAssetName(long wanted_asset);
	int ReturnArrayPositionFromID(int wanted_id, int last_found_position = 0 );
	int ReturnArrayPositionFromParentID(int wanted_id);
	wxString ReturnAssetFullFilename(long wanted_asst);

	void AddAsset(Asset *asset_to_add);
	void RemoveAsset(long number_to_remove);
	void RemoveAll();
	long FindFile(wxFileName file_to_find, bool also_check_vs_shortname = false, long max_asset_number_to_check = -1);
	void CheckMemory();

};




#endif
