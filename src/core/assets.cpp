#include "core_headers.h"
//#include "gui_core_headers.h"


AssetList::AssetList()
{

}

AssetList::~AssetList()
{

}

Asset::Asset()
{

}

Asset::~Asset()
{

}

wxString Asset::ReturnFullPathString()
{
	return filename.GetFullPath();

}

wxString Asset::ReturnShortNameString()
{
	return filename.GetFullName();
}

MovieAsset::MovieAsset()
{
	asset_id = -1;
	parent_id = -1;
	position_in_stack = 1;
	number_of_frames = 0;
	x_size = 0;
	y_size = 0;
	pixel_size = 0;
	microscope_voltage = 0;
	spherical_aberration = 0;
	dose_per_frame = 0;
	total_dose = 0;
	
	filename = wxEmptyString;
	asset_name = wxEmptyString;
	
	is_valid = false;

}

MovieAsset::~MovieAsset()
{
	//Don't have to do anything for now
}

MovieAsset::MovieAsset(wxString wanted_filename)
{
	filename = wanted_filename;
	asset_name = wanted_filename;
	asset_id = -1;
	position_in_stack = 1;

	number_of_frames = 0;
	x_size = 0;
	y_size = 0;
	pixel_size = 0;
	microscope_voltage = 0;
	dose_per_frame = 0;
	spherical_aberration = 0;
	total_dose = 0;
	
	if (filename.IsOk() == true && filename.FileExists() == true)
	{
		is_valid = GetMRCDetails(filename.GetFullPath().fn_str(), x_size, y_size, number_of_frames);
	}
	
}


/*
void MovieAsset::Recheck_if_valid()
{	
	if (filename.IsOk() == true && filename.FileExists() == true)
	{
		movie_is_valid = GetMRCDetails(filename.GetFullPath().fn_str(), x_size, y_size, number_of_frames);
	}

}*/

void MovieAsset::Update(wxString wanted_filename)
{
	filename = wanted_filename;
	
	if (filename.IsOk() == true && filename.FileExists() == true)
	{
		is_valid = GetMRCDetails(filename.GetFullPath().fn_str(), x_size, y_size, number_of_frames);
	}
	
}



void MovieAsset::CopyFrom(Asset *other_asset)
{
	MovieAsset *casted_asset = reinterpret_cast < MovieAsset *> (other_asset);
	asset_id = casted_asset->asset_id;
	position_in_stack = casted_asset->position_in_stack;
	x_size = casted_asset->x_size;
	y_size = casted_asset->y_size;
	number_of_frames = casted_asset->number_of_frames;
	filename = casted_asset->filename;
	pixel_size = casted_asset->pixel_size;
	microscope_voltage = casted_asset->microscope_voltage;
	spherical_aberration = casted_asset->spherical_aberration;
	dose_per_frame = casted_asset->dose_per_frame;
	is_valid = casted_asset->is_valid;
	total_dose = casted_asset->total_dose;
	asset_name = casted_asset->asset_name;
}

// Image asset///

ImageAsset::ImageAsset()
{
	asset_id = -1;
	parent_id = -1;
	alignment_id = -1;
	position_in_stack = 1;
	x_size = 0;
	y_size = 0;
	pixel_size = 0;
	microscope_voltage = 0;
	spherical_aberration = 0;
	is_valid = false;

	filename = wxEmptyString;
	asset_name = wxEmptyString;

}

ImageAsset::~ImageAsset()
{
	//Don't have to do anything for now
}

ImageAsset::ImageAsset(wxString wanted_filename)
{
	filename = wanted_filename;
	asset_name = wanted_filename;
	asset_id = -1;
	position_in_stack = 1;
	parent_id = -1;
	alignment_id = -1;
	ctf_estimation_id = -1;

	x_size = 0;
	y_size = 0;
	pixel_size = 0;
	microscope_voltage = 0;
	spherical_aberration = 0;
	is_valid = false;

	int number_in_stack;

	if (filename.IsOk() == true && filename.FileExists() == true)
	{
		is_valid = GetMRCDetails(filename.GetFullPath().fn_str(), x_size, y_size, number_in_stack);
	}

}


void ImageAsset::Update(wxString wanted_filename)
{
	filename = wanted_filename;

	int number_in_stack;


	if (filename.IsOk() == true && filename.FileExists() == true)
	{
		is_valid = GetMRCDetails(filename.GetFullPath().fn_str(), x_size, y_size, number_in_stack);
	}

}


void ImageAsset::CopyFrom(Asset *other_asset)
{
	ImageAsset *casted_asset = reinterpret_cast < ImageAsset *> (other_asset);
	asset_id = casted_asset->asset_id;
	position_in_stack = casted_asset->position_in_stack;
	parent_id = casted_asset->parent_id;
	x_size = casted_asset->x_size;
	y_size = casted_asset->y_size;
	alignment_id = casted_asset->alignment_id;
	ctf_estimation_id = casted_asset->ctf_estimation_id;

	filename = casted_asset->filename;
	pixel_size = casted_asset->pixel_size;
	microscope_voltage = casted_asset->microscope_voltage;
	spherical_aberration = casted_asset->spherical_aberration;
	is_valid = casted_asset->is_valid;
	asset_name = casted_asset->asset_name;
}

// Particle Position Asset

ParticlePositionAsset::ParticlePositionAsset()
{
	Reset();
}

ParticlePositionAsset::ParticlePositionAsset(const float &wanted_x_in_angstroms, const float &wanted_y_in_angstroms)
{
	Reset();
	x_position = wanted_x_in_angstroms;
	y_position = wanted_y_in_angstroms;
}

ParticlePositionAsset::~ParticlePositionAsset()
{
	//Don't have to do anything for now
}

void ParticlePositionAsset::Reset()
{
	asset_id = -1;
	parent_id = -1;
	picking_id = -1;
	pick_job_id = -1;
	parent_template_id = -1;
	x_position = 0.0;
	y_position = 0.0;
	peak_height = 0.0;
	template_phi = 0.0;
	template_theta = 0.0;
	template_psi = 0.0;
	asset_name = wxEmptyString;
	filename = wxEmptyString;
}


void ParticlePositionAsset::CopyFrom(Asset *other_asset)
{
	ParticlePositionAsset *casted_asset = reinterpret_cast < ParticlePositionAsset *> (other_asset);
	asset_id = casted_asset->asset_id;
	parent_id = casted_asset->parent_id;
	picking_id = casted_asset->picking_id;
	pick_job_id = casted_asset->pick_job_id;
	parent_template_id = casted_asset->parent_template_id;
	x_position = casted_asset->x_position;
	y_position = casted_asset->y_position;
	peak_height = casted_asset->peak_height;
	template_phi = casted_asset->template_phi;
	template_theta = casted_asset->template_theta;
	template_psi = casted_asset->template_psi;
	asset_name = casted_asset->asset_name;
}

#include <wx/arrimpl.cpp>
WX_DEFINE_OBJARRAY(ArrayOfParticlePositionAssets);

// Volume asset///

VolumeAsset::VolumeAsset()
{
	asset_id = -1;
	parent_id = -1;
	reconstruction_job_id = -1;
	x_size = 0;
	y_size = 0;
	z_size = 0;
	pixel_size = 0;

	is_valid = false;
	filename = wxEmptyString;
	asset_name = wxEmptyString;

}

VolumeAsset::~VolumeAsset()
{
	//Don't have to do anything for now
}

VolumeAsset::VolumeAsset(wxString wanted_filename)
{
	filename = wanted_filename;
	asset_name = wanted_filename;
	asset_id = -1;
	parent_id = -1;
	reconstruction_job_id = -1;

	x_size = 0;
	y_size = 0;
	z_size = 0;
	pixel_size = 0;
	is_valid = false;

	int number_in_stack;

	if (filename.IsOk() == true && filename.FileExists() == true)
	{
		is_valid = GetMRCDetails(filename.GetFullPath().fn_str(), x_size, y_size, z_size);
	}

}


void VolumeAsset::Update(wxString wanted_filename)
{
	filename = wanted_filename;


	if (filename.IsOk() == true && filename.FileExists() == true)
	{
		is_valid = GetMRCDetails(filename.GetFullPath().fn_str(), x_size, y_size, z_size);
	}

}


void VolumeAsset::CopyFrom(Asset *other_asset)
{
	VolumeAsset *casted_asset = reinterpret_cast < VolumeAsset *> (other_asset);
	asset_id = casted_asset->asset_id;
	parent_id = casted_asset->parent_id;
	reconstruction_job_id = casted_asset->reconstruction_job_id;

	x_size = casted_asset->x_size;
	y_size = casted_asset->y_size;
	z_size = casted_asset->z_size;

	filename = casted_asset->filename;
	pixel_size = casted_asset->pixel_size;
	is_valid = casted_asset->is_valid;
	asset_name = casted_asset->asset_name;
}

// Return Pointers


MovieAsset * AssetList::ReturnMovieAssetPointer(long wanted_asset)
{
	MyPrintWithDetails("This should never be called!!");
	abort();
}

ImageAsset * AssetList::ReturnImageAssetPointer(long wanted_asset)
{
	MyPrintWithDetails("This should never be called!!");
	abort();
}

ParticlePositionAsset * AssetList::ReturnParticlePositionAssetPointer(long wanted_asset)
{
	MyPrintWithDetails("This should never be called!!");
	abort();
}

VolumeAsset* AssetList::ReturnVolumeAssetPointer(long wanted_asset)
{
	MyPrintWithDetails("This should never be called!!");
	abort();
}


////////////////////////Movie Asset List//////////////////


MovieAssetList::MovieAssetList()
{
	number_of_assets = 0;	
	number_allocated = 15;
	assets = new MovieAsset[15];
	
}

MovieAssetList::~MovieAssetList()
{	
	delete [] reinterpret_cast < MovieAsset *> (assets);
}

void MovieAssetList::CheckMemory()
{
	MovieAsset *buffer;

	// check we have enough memory

	if (number_of_assets >= number_allocated)
	{
		// reallocate..

		if (number_of_assets < 10000) number_allocated *= 2;
		else number_allocated += 10000;

		buffer = new MovieAsset[number_allocated];

		for (long counter = 0; counter < number_of_assets; counter++)
		{
			buffer[counter].CopyFrom(& reinterpret_cast < MovieAsset *> (assets)[counter]);
		}

		delete [] reinterpret_cast < MovieAsset *> (assets);
		assets = buffer;
	}


}

long MovieAssetList::FindFile(wxFileName file_to_find, bool also_check_vs_shortname)
{
	long found_position = -1;

	for (long counter = 0; counter < number_of_assets; counter++)
	{
		if (reinterpret_cast < MovieAsset *> (assets)[counter].filename == file_to_find)
		{
			found_position = counter;
			break;
		}

		if (also_check_vs_shortname == true)
		{
			if (reinterpret_cast < MovieAsset *> (assets)[counter].filename.GetFullName() == file_to_find.GetFullName())
			{
				found_position = counter;
				break;
			}
		}
	}

	return found_position;

}

Asset * MovieAssetList::ReturnAssetPointer(long wanted_asset)
{
	MyDebugAssertTrue(wanted_asset >= 0 && wanted_asset < number_of_assets, "Requesting an asset (%li) that doesn't exist!", wanted_asset);
	return & reinterpret_cast <MovieAsset *> (assets)[wanted_asset];
}

MovieAsset * MovieAssetList::ReturnMovieAssetPointer(long wanted_asset)
{
	MyDebugAssertTrue(wanted_asset >= 0 && wanted_asset < number_of_assets, "Requesting an asset (%li) that doesn't exist!", wanted_asset);
	return & reinterpret_cast <MovieAsset *> (assets)[wanted_asset];
}

int MovieAssetList::ReturnAssetID(long wanted_asset)
{
	return  reinterpret_cast <MovieAsset *> (assets)[wanted_asset].asset_id;
}

wxString MovieAssetList::ReturnAssetName(long wanted_asset)
{
	return  reinterpret_cast <MovieAsset *> (assets)[wanted_asset].asset_name;
}

int MovieAssetList::ReturnArrayPositionFromID(int wanted_id)
{
	for (int counter = 0; counter < number_of_assets; counter++)
	{
		if (reinterpret_cast <MovieAsset *> (assets)[counter].asset_id == wanted_id) return counter;
	}

	return -1;
}

int MovieAssetList::ReturnArrayPositionFromParentID(int wanted_id)
{
	for (int counter = 0; counter < number_of_assets; counter++)
	{
		if (reinterpret_cast <MovieAsset *> (assets)[counter].parent_id == wanted_id) return counter;
	}

	return -1;
}


void MovieAssetList::AddAsset(Asset *asset_to_add)
{
	MovieAsset *buffer;
	
	CheckMemory();
	
	// Should be fine for memory, so just add one.

	reinterpret_cast < MovieAsset *> (assets)[number_of_assets].CopyFrom(asset_to_add);
	number_of_assets++;
	


}


void MovieAssetList::RemoveAsset(long number_to_remove)
{
	if (number_to_remove < 0 || number_to_remove >= number_of_assets)
	{
		wxPrintf("Error! Trying to remove a movie that does not exist\n\n");
		exit(-1);
	}

	for (long counter = number_to_remove; counter < number_of_assets -1; counter++)
	{
		reinterpret_cast < MovieAsset *> (assets)[counter].CopyFrom(& reinterpret_cast < MovieAsset *> (assets)[counter + 1]);
	}

	number_of_assets--;
}

void MovieAssetList::RemoveAll()
{
	number_of_assets = 0;

	if (number_allocated > 100)
	{
		reinterpret_cast < MovieAsset *> (assets);
		number_allocated = 100;
		assets = new MovieAsset[number_allocated];
	}
}


////////////////////////Image Asset List//////////////////


ImageAssetList::ImageAssetList()
{
	number_of_assets = 0;
	number_allocated = 15;
	assets = new ImageAsset[15];

}

ImageAssetList::~ImageAssetList()
{
	delete [] reinterpret_cast < ImageAsset *>  (assets);
}

void ImageAssetList::CheckMemory()
{
	ImageAsset *buffer;

	// check we have enough memory

	if (number_of_assets >= number_allocated)
	{
		// reallocate..

		if (number_of_assets < 10000) number_allocated *= 2;
		else number_allocated += 10000;

		buffer = new ImageAsset[number_allocated];

		for (long counter = 0; counter < number_of_assets; counter++)
		{
			buffer[counter].CopyFrom(& reinterpret_cast < ImageAsset *> (assets)[counter]);
		}

		delete [] reinterpret_cast < ImageAsset *>  (assets);
		assets = buffer;
	}


}

long ImageAssetList::FindFile(wxFileName file_to_find, bool also_check_vs_shortname)
{
	long found_position = -1;

	for (long counter = 0; counter < number_of_assets; counter++)
	{
		if (reinterpret_cast < ImageAsset *> (assets)[counter].filename == file_to_find)
		{
			found_position = counter;
			break;
		}

		if (also_check_vs_shortname == true)
		{
			if (reinterpret_cast < ImageAsset *> (assets)[counter].filename.GetFullName() == file_to_find.GetFullName())
			{
				found_position = counter;
				break;
			}
		}
	}

	return found_position;

}

Asset * ImageAssetList::ReturnAssetPointer(long wanted_asset)
{
	MyDebugAssertTrue(wanted_asset >= 0 && wanted_asset < number_of_assets, "Requesting an asset (%li) that doesn't exist!", wanted_asset);
	return & reinterpret_cast <ImageAsset *> (assets)[wanted_asset];
}

ImageAsset * ImageAssetList::ReturnImageAssetPointer(long wanted_asset)
{
	MyDebugAssertTrue(wanted_asset >= 0 && wanted_asset < number_of_assets, "Requesting an asset (%li) that doesn't exist!", wanted_asset);
	return & reinterpret_cast <ImageAsset *> (assets)[wanted_asset];
}

int ImageAssetList::ReturnAssetID(long wanted_asset)
{
	return  reinterpret_cast <ImageAsset *> (assets)[wanted_asset].asset_id;
}

wxString ImageAssetList::ReturnAssetName(long wanted_asset)
{
	return  reinterpret_cast <ImageAsset *> (assets)[wanted_asset].asset_name;
}

int ImageAssetList::ReturnArrayPositionFromID(int wanted_id)
{
	for (int counter = 0; counter < number_of_assets; counter++)
	{
		if (reinterpret_cast <ImageAsset *> (assets)[counter].asset_id == wanted_id) return counter;
	}

	return -1;
}

int ImageAssetList::ReturnArrayPositionFromParentID(int wanted_id)
{
	for (int counter = 0; counter < number_of_assets; counter++)
	{
		if (reinterpret_cast <ImageAsset *> (assets)[counter].parent_id == wanted_id) return counter;
	}

	return -1;
}



void ImageAssetList::AddAsset(Asset *asset_to_add)
{
	CheckMemory();

	// Should be fine for memory, so just add one.

	reinterpret_cast < ImageAsset *> (assets)[number_of_assets].CopyFrom(asset_to_add);
	number_of_assets++;



}

void ImageAssetList::RemoveAsset(long number_to_remove)
{
	if (number_to_remove < 0 || number_to_remove >= number_of_assets)
	{
		wxPrintf("Error! Trying to remove a movie that does not exist\n\n");
		exit(-1);
	}

	for (long counter = number_to_remove; counter < number_of_assets -1; counter++)
	{
		reinterpret_cast < ImageAsset *> (assets)[counter].CopyFrom(& reinterpret_cast < ImageAsset *> (assets)[counter + 1]);
	}

	number_of_assets--;
}

void ImageAssetList::RemoveAll()
{
	number_of_assets = 0;

	if (number_allocated > 100)
	{
		delete [] reinterpret_cast < ImageAsset *> (assets);
		number_allocated = 100;
		assets = new ImageAsset[number_allocated];
	}
}

// Particle Asset List

ParticlePositionAssetList::ParticlePositionAssetList()
{
	number_of_assets = 0;
	number_allocated = 15;
	assets = new ParticlePositionAsset[15];

}

ParticlePositionAssetList::~ParticlePositionAssetList()
{
	delete [] reinterpret_cast < ParticlePositionAsset *>  (assets);
}


void ParticlePositionAssetList::CheckMemory()
{
	ParticlePositionAsset *buffer;

	// check we have enough memory

	if (number_of_assets >= number_allocated)
	{
		// reallocate..

		if (number_of_assets < 10000) number_allocated *= 2;
		else number_allocated += 10000;

		buffer = new ParticlePositionAsset[number_allocated];

		for (long counter = 0; counter < number_of_assets; counter++)
		{
			buffer[counter].CopyFrom(& reinterpret_cast < ParticlePositionAsset *> (assets)[counter]);
		}

		delete [] reinterpret_cast < ParticlePositionAsset *>  (assets);
		assets = buffer;
	}


}

Asset * ParticlePositionAssetList::ReturnAssetPointer(long wanted_asset)
{
	MyDebugAssertTrue(wanted_asset >= 0 && wanted_asset < number_of_assets, "Requesting an asset (%li) that doesn't exist!", wanted_asset);
	return & reinterpret_cast <ParticlePositionAsset *> (assets)[wanted_asset];
}

ParticlePositionAsset * ParticlePositionAssetList::ReturnParticlePositionAssetPointer(long wanted_asset)
{
	MyDebugAssertTrue(wanted_asset >= 0 && wanted_asset < number_of_assets, "Requesting an asset (%li) that doesn't exist!", wanted_asset);
	return & reinterpret_cast <ParticlePositionAsset *> (assets)[wanted_asset];
}

int ParticlePositionAssetList::ReturnAssetID(long wanted_asset)
{
	return  reinterpret_cast <ParticlePositionAsset *> (assets)[wanted_asset].asset_id;
}

int ParticlePositionAssetList::ReturnArrayPositionFromID(int wanted_id)
{
	for (int counter = 0; counter < number_of_assets; counter++)
	{
		if (reinterpret_cast <ParticlePositionAsset *> (assets)[counter].asset_id == wanted_id) return counter;
	}

	return -1;
}

int ParticlePositionAssetList::ReturnArrayPositionFromParentID(int wanted_id)
{
	for (int counter = 0; counter < number_of_assets; counter++)
	{
		if (reinterpret_cast <ParticlePositionAsset *> (assets)[counter].parent_id == wanted_id) return counter;
	}

	return -1;
}



void ParticlePositionAssetList::AddAsset(Asset *asset_to_add)
{
	CheckMemory();

	// Should be fine for memory, so just add one.

	reinterpret_cast < ParticlePositionAsset *> (assets)[number_of_assets].CopyFrom(asset_to_add);
	number_of_assets++;



}

void ParticlePositionAssetList::RemoveAsset(long number_to_remove)
{
	if (number_to_remove < 0 || number_to_remove >= number_of_assets)
	{
		wxPrintf("Error! Trying to remove a movie that does not exist\n\n");
		exit(-1);
	}

	for (long counter = number_to_remove; counter < number_of_assets -1; counter++)
	{
		reinterpret_cast < ParticlePositionAsset *> (assets)[counter].CopyFrom(& reinterpret_cast < ParticlePositionAsset *> (assets)[counter + 1]);
	}

	number_of_assets--;
}

void ParticlePositionAssetList::RemoveAll()
{
	number_of_assets = 0;

	if (number_allocated > 100)
	{
		delete [] reinterpret_cast < ParticlePositionAsset *> (assets);
		number_allocated = 100;
		assets = new ParticlePositionAsset[number_allocated];
	}
}

// Volume Asset List

VolumeAssetList::VolumeAssetList()
{
	number_of_assets = 0;
	number_allocated = 15;
	assets = new VolumeAsset[15];

}

VolumeAssetList::~VolumeAssetList()
{
	delete [] reinterpret_cast < VolumeAsset *> (assets);
}

void VolumeAssetList::CheckMemory()
{
	VolumeAsset *buffer;

	// check we have enough memory

	if (number_of_assets >= number_allocated)
	{
		// reallocate..

		if (number_of_assets < 10000) number_allocated *= 2;
		else number_allocated += 10000;

		buffer = new VolumeAsset[number_allocated];

		for (long counter = 0; counter < number_of_assets; counter++)
		{
			buffer[counter].CopyFrom(& reinterpret_cast < VolumeAsset *> (assets)[counter]);
		}

		delete [] reinterpret_cast < VolumeAsset *>  (assets);
		assets = buffer;
	}


}

long VolumeAssetList::FindFile(wxFileName file_to_find, bool also_check_vs_shortname)
{
	long found_position = -1;

	for (long counter = 0; counter < number_of_assets; counter++)
	{
		if (reinterpret_cast < VolumeAsset *> (assets)[counter].filename == file_to_find)
		{
			found_position = counter;
			break;
		}

		if (also_check_vs_shortname == true)
		{
			if (reinterpret_cast < VolumeAsset *> (assets)[counter].filename.GetFullName() == file_to_find.GetFullName())
			{
				found_position = counter;
				break;
			}
		}
	}

	return found_position;

}

Asset * VolumeAssetList::ReturnAssetPointer(long wanted_asset)
{
	MyDebugAssertTrue(wanted_asset >= 0 && wanted_asset < number_of_assets, "Requesting an asset (%li) that doesn't exist!", wanted_asset);
	return & reinterpret_cast <VolumeAsset *> (assets)[wanted_asset];
}

VolumeAsset * VolumeAssetList::ReturnVolumeAssetPointer(long wanted_asset)
{
	MyDebugAssertTrue(wanted_asset >= 0 && wanted_asset < number_of_assets, "Requesting an asset (%li) that doesn't exist!", wanted_asset);
	return & reinterpret_cast <VolumeAsset *> (assets)[wanted_asset];
}

int VolumeAssetList::ReturnAssetID(long wanted_asset)
{
	return  reinterpret_cast <VolumeAsset *> (assets)[wanted_asset].asset_id;
}

wxString VolumeAssetList::ReturnAssetName(long wanted_asset)
{
	return  reinterpret_cast <VolumeAsset *> (assets)[wanted_asset].asset_name;
}

int VolumeAssetList::ReturnArrayPositionFromID(int wanted_id)
{
	for (int counter = 0; counter < number_of_assets; counter++)
	{
		if (reinterpret_cast <VolumeAsset *> (assets)[counter].asset_id == wanted_id) return counter;
	}

	return -1;
}

int VolumeAssetList::ReturnArrayPositionFromParentID(int wanted_id)
{
	for (int counter = 0; counter < number_of_assets; counter++)
	{
		if (reinterpret_cast <VolumeAsset *> (assets)[counter].parent_id == wanted_id) return counter;
	}

	return -1;
}



void VolumeAssetList::AddAsset(Asset *asset_to_add)
{
	CheckMemory();

	// Should be fine for memory, so just add one.

	reinterpret_cast < VolumeAsset *> (assets)[number_of_assets].CopyFrom(asset_to_add);
	number_of_assets++;



}

void VolumeAssetList::RemoveAsset(long number_to_remove)
{
	if (number_to_remove < 0 || number_to_remove >= number_of_assets)
	{
		wxPrintf("Error! Trying to remove a movie that does not exist\n\n");
		exit(-1);
	}

	for (long counter = number_to_remove; counter < number_of_assets -1; counter++)
	{
		reinterpret_cast < VolumeAsset *> (assets)[counter].CopyFrom(& reinterpret_cast < VolumeAsset *> (assets)[counter + 1]);
	}

	number_of_assets--;
}

void VolumeAssetList::RemoveAll()
{
	number_of_assets = 0;

	if (number_allocated > 100)
	{
		delete [] reinterpret_cast < VolumeAsset *> (assets);
		number_allocated = 100;
		assets = new VolumeAsset[number_allocated];
	}
}

