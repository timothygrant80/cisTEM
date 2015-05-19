#include "core_headers.h"
#include "gui_core_headers.h"


MovieAsset::MovieAsset()
{
	number_of_frames = 0;
	x_size = 0;
	y_size = 0;
	pixel_size = 0;
	microscope_voltage = 0;
	spherical_aberration = 0;
	dose_per_frame = 0;
	total_dose = 0;
	
	filename = wxEmptyString;
	
	movie_is_valid = false;	

}

MovieAsset::~MovieAsset()
{
	//Don't have to do anything for now
}

MovieAsset::MovieAsset(wxString wanted_filename)
{
	filename = wanted_filename;

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
		movie_is_valid = GetMRCDetails(filename.GetFullPath().fn_str(), x_size, y_size, number_of_frames);
	}
	
}

void MovieAsset::Recheck_if_valid()
{	
	if (filename.IsOk() == true && filename.FileExists() == true)
	{
		movie_is_valid = GetMRCDetails(filename.GetFullPath().fn_str(), x_size, y_size, number_of_frames);
	}

}

void MovieAsset::Update(wxString wanted_filename)
{
	filename = wanted_filename;
	
	if (filename.IsOk() == true && filename.FileExists() == true)
	{
		movie_is_valid = GetMRCDetails(filename.GetFullPath().fn_str(), x_size, y_size, number_of_frames);
	}
	
}



void MovieAsset::CopyFrom(MovieAsset *other_asset)
{
	x_size = other_asset->x_size;
	y_size = other_asset->y_size;
	number_of_frames = other_asset->number_of_frames;
	filename = other_asset->filename;
	pixel_size = other_asset->pixel_size;
	microscope_voltage = other_asset->microscope_voltage;
	dose_per_frame = other_asset->dose_per_frame;
	movie_is_valid = other_asset->movie_is_valid;	
	total_dose = other_asset->total_dose;
}

wxString MovieAsset::ReturnFullPathString()
{
	return filename.GetFullPath();

}

wxString MovieAsset::ReturnShortNameString()
{
	return filename.GetFullName();
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
	delete [] assets;
}

long MovieAssetList::FindFile(wxFileName file_to_find)
{
	long found_position = -1;

	for (long counter = 0; counter < number_of_assets; counter++)
	{
		if (assets[counter].filename == file_to_find)
		{
			found_position = counter;
			break;
		}
	}

	return found_position;

}


void MovieAssetList::AddMovie(MovieAsset *asset_to_add)
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
			buffer[counter].CopyFrom(&assets[counter]);
		}
		
		delete [] assets;
		assets = buffer;
	}
	
	// Should be fine for memory, so just add one.

	assets[number_of_assets].CopyFrom(asset_to_add);
	number_of_assets++;
	
}


void MovieAssetList::RemoveMovie(long number_to_remove)
{
	if (number_to_remove < 0 || number_to_remove >= number_of_assets)
	{
		wxPrintf("Error! Trying to remove a movie that does not exist\n\n");
		exit(-1);
	}

	for (long counter = number_to_remove; counter < number_of_assets -1; counter++)
	{
		assets[counter] = assets[counter + 1];
	}

	number_of_assets--;
}

void MovieAssetList::RemoveAll()
{
	number_of_assets = 0;

	if (number_allocated > 100)
	{
		delete [] assets;
		number_allocated = 100;
		assets = new MovieAsset[number_allocated];
	}
}

