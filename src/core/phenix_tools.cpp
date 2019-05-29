#include "core_headers.h"

CommandLineTools::CommandLineTools()
{
	args = wxString("");
}

CommandLineTools::~CommandLineTools() {}

void CommandLineTools::SetupLauncher(wxString wanted_program_dir, wxString wanted_working_dir)
{
	program_dir = wanted_program_dir;
	working_dir = wanted_working_dir;
}

PhenixTools::PhenixTools()
{
	CommandLineTools();
	program_dir = wxString("");
	args = wxString("");
}

PhenixTools::~PhenixTools() {}

FmodelRegrid::FmodelRegrid()
{
	model_path = wxString("");
	mrc_path = wxString("");
	output_path = wxString("");
	resolution = 0;
	model_basename = wxString("");
	model_ext = wxString("");
	output_basename = wxString("");
	output_ext = wxString("");
	fmodel_regrid_args = wxString("");
	fmodel_regrid_main = wxString("");
	fmodel_regrid_echo = wxString("");
	fmodel_regrid_code = 0;
	success_message = wxString("");
	fmodel_regrid_script = wxString("import iotbx.pdb\nimport libtbx.phil\nfrom libtbx.test_utils import approx_equal\nimport iotbx.ccp4_map\nfrom cctbx import maptbx, crystal\nfrom scitbx.array_family import flex\nimport os\n\nmaster_phil_str = \"\"\"\nmodel_file = None\n  .type = path\n  .help = \"Path to the model, in pdb or mmcif format\"\nmap_file = None\n  .type = path\n  .help = \"Path to the .mrc file to supply the target map gridding\"\nd_min = 3\n  .type = float\n  .help = \"Resolution to use for the fmodel calculation\"\n\"\"\"\n\nmaster_phil = libtbx.phil.parse(master_phil_str)\n\ndef fmodel_regrid(emmap, model, d_min):\n  map_inp = iotbx.ccp4_map.map_reader(file_name=emmap)\n  m = map_inp.map_data()\n  symm = crystal.symmetry(\n    space_group_symbol=\"P1\",\n    unit_cell=map_inp.unit_cell_parameters)\n  xrs = iotbx.pdb.input(file_name=model).xray_structure_simple(\n    crystal_symmetry=symm)\n  assert symm.is_similar_symmetry(xrs.crystal_symmetry())\n  f_calc = xrs.structure_factors(d_min=d_min).f_calc()\n  cg = maptbx.crystal_gridding(\n    unit_cell             = map_inp.unit_cell(),\n    space_group_info      = symm.space_group_info(),\n    pre_determined_n_real = m.accessor().all())\n  fft_map = f_calc.fft_map(crystal_gridding = cg)\n  outmap = os.path.splitext(model)[0] + \".mrc\"\n  iotbx.ccp4_map.write_ccp4_map(\n    file_name   = outmap,\n    unit_cell   = cg.unit_cell(),\n    space_group = cg.space_group(),\n    map_data    = fft_map.real_map_unpadded(),\n    labels      = flex.std_string([\"some comments\"]))\n\ndef run(args):\n  import iotbx.phil\n  cmdline = iotbx.phil.process_command_line_with_files(\n    args=args,\n    master_phil=master_phil,\n    )\n  params = cmdline.work.extract()\n  fmodel_regrid(params.map_file, params.model_file, params.d_min)\n\nif (__name__ == \"__main__\"):\n  import sys\n  run(sys.argv[1:])");
	fmodel_regrid_script_path = wxString("");
}

FmodelRegrid::~FmodelRegrid() {}

void FmodelRegrid::SetAllUserParameters(	wxString wanted_model_path,
											wxString wanted_mrc_path,
											wxString wanted_output_path,
											float wanted_resolution)
{
//	MyDebugAssertTrue(wxFileName::IsFileReadable(wanted_model_path), "Can't read the model:%s",wanted_model_path.ToStdString());
	model_path = wanted_model_path;
	MyDebugAssertTrue(wxFileName::IsFileReadable(wanted_mrc_path), "Can't read the map:%s",wanted_mrc_path.ToStdString());
	mrc_path = wanted_mrc_path;
	output_path = wanted_output_path;
	resolution = wanted_resolution;
}

bool FmodelRegrid::RunFmodelRegrid()
{
	wxFileName::SplitPath(model_path, &working_dir, &model_basename, &model_ext, wxPATH_NATIVE); // TEMP see if we can drop the working_dir
	wxFileName::SplitPath(output_path, &working_dir, &output_basename, &output_ext, wxPATH_NATIVE);
	MyDebugAssertTrue(wxFileName::IsDirWritable(working_dir), "Can't write to the target output directory:%s",working_dir.ToStdString());

	if (working_dir != wxString(""))
	{
		wxFileName::SetCwd(working_dir);
	} else {
		working_dir = wxString(".");
	}

	// write the python script we're about to run (if it's not already there)

	fmodel_regrid_script_path = working_dir + wxString("/fmodel_regrid.py");
	if (! wxFileExists(fmodel_regrid_script_path))
	{
		wxTextFile fmodel_regrid_script_file;
		fmodel_regrid_script_file.Create(fmodel_regrid_script_path);
		fmodel_regrid_script_file.Open();
		fmodel_regrid_script_file.AddLine(fmodel_regrid_script);
		fmodel_regrid_script_file.Write();
		fmodel_regrid_script_file.Close();
	}
	MyDebugAssertTrue(wxFileName::IsFileReadable(fmodel_regrid_script_path), "Can't read the fmodel_regrid script at %s",fmodel_regrid_script_path.ToStdString());

	// construct the command to run fmodel_regrid.py and execute with wxwidgets

	fmodel_regrid_args = " map_file=" + mrc_path + " model_file=" + model_path + wxString::Format(" d_min=%f", resolution);
	wxPrintf("\nLaunching regridded fmodel calculation...\n");
	fmodel_regrid_main = program_dir + wxString("/libtbx.python ") + fmodel_regrid_script_path + fmodel_regrid_args;
	fmodel_regrid_echo = "\nExecuting command \"" + fmodel_regrid_main + "\"\n";
	wxPrintf(wxString(fmodel_regrid_echo));
	fmodel_regrid_code = wxExecute(fmodel_regrid_main, wxEXEC_SYNC, NULL);
	if ( fmodel_regrid_code != 0 )
	{
		wxLogError("Execution failed; exiting.\n\n");
		return false;
	}
	wxRenameFile(working_dir + "/" + model_basename + ".mrc", output_basename + "." + output_ext);

	// print a success message
	success_message = "\nWrote " + output_path + "\n";
	wxPrintf(success_message);

	return true;
}
