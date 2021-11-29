#include "../core/gui_core_headers.h"
#include <wx/arrimpl.cpp>
#include <wx/filename.h>

extern MyRefinementPackageAssetPanel *refinement_package_asset_panel;

CombineRefinementPackagesDialog::CombineRefinementPackagesDialog(MyRefinementPackageAssetPanel *parent)
:
	CombineRefinementPackagesDialogParent( parent )
{
	wxArrayString refinement_name;

	for (int counter = 0; counter < refinement_package_asset_panel->all_refinement_packages.GetCount(); counter++)
	{
		refinement_name.Add(refinement_package_asset_panel->all_refinement_packages[counter].name);  // Will use .stack_filename instead of .name when preparing to open files
	}
	RefinementPackagesCheckListBox->InsertItems(refinement_name, 0);
}

void CombineRefinementPackagesDialog::OnCancelClick( wxCommandEvent& event )
{
	EndModal(0);
	Destroy();
}

void CombineRefinementPackagesDialog::OnCombineClick( wxCommandEvent& event )
{
	int counter = 0;
	int package_counter = 1;
	int output_particle_counter = 1;
	int input_particle_counter = 1;
	Refinement combined_refinement;
	wxArrayString packages_to_combine_filenames;
	MRCFile combined_stacks_file("/home/tim/New_Project7/test.mrc", true);  // New combined stack; gets written to--must specify directory explicitly for now
	Image image_for_combined_stack;
	Image image_from_previous_stack;

	for (counter = 0; counter < refinement_package_asset_panel->all_refinement_packages.GetCount(); counter++)
	{
		if (RefinementPackagesCheckListBox->IsChecked(counter) == true)
		{
			packages_to_combine_filenames.Add(refinement_package_asset_panel->all_refinement_packages[counter].stack_filename);  // Retrieve filename of stack
		}
	}

	// Now loop through the existing MRC filenames to get to the files; open, read through, then write each image to new MRC file, close.
	for (package_counter = 0; package_counter < packages_to_combine_filenames.GetCount(); package_counter++)
	{
		MRCFile input_file(packages_to_combine_filenames[package_counter].ToStdString(), false);

		for (input_particle_counter = 1; input_particle_counter <= refinement_package_asset_panel->all_refinement_packages[package_counter].contained_particles.GetCount(); input_particle_counter++)
		{
			image_from_previous_stack.ReadSlice(&input_file, input_particle_counter);
			image_from_previous_stack.WriteSlice(&combined_stacks_file, output_particle_counter); // Write from PREVIOUS stack
			output_particle_counter++;
		}
		input_file.CloseFile();
	}
	/* TODO: Declare file for each stack, retrieve each output file
	 * TODO: loop over number of refinement packages
	 * TODO: declare each mrc file with the name of refinement package stack
	 * TODO: loop over number of particles in the stack
	 * TODO: read in from previous stack (.read(number of the stack, image)).
	 * TODO: need to use wizard panel to select params? Or create new panel for combination?
	 */
	combined_stacks_file.CloseFile();
	Destroy();
}
