message("In the src/gui CMakeLists file")

add_library(cisTEM_gui STATIC 
	gui_globals.cpp 
	MyFSCPanel.cpp 
	MyRefinementResultsPanel.cpp 
	PlotFSCPanel.cpp 
	gui_functions.cpp 
	my_controls.cpp 
	MyRenameDialog.cpp 
	BitmapPanel.cpp 
	PickingBitmapPanel.cpp 
	job_panel.cpp 
	ProjectX_gui.cpp 
	ResultsDataViewListCtrl.cpp 
	MyMovieFilterDialog.cpp 
	MainFrame.cpp 
	MyAssetParentPanel.cpp 
	MyMovieAssetPanel.cpp 
	MyImageAssetPanel.cpp 
	MyParticlePositionAssetPanel.cpp 
	MyVolumeChooserDialog.cpp
	MovieImportDialog.cpp 
	MyImageImportDialog.cpp 
	MyVolumeImportDialog.cpp 
	ErrorDialog.cpp 
	AlignMoviesPanel.cpp 
	MyRunProfilesPanel.cpp 
	mathplot.cpp 
	MyMovieAlignResultsPanel.cpp 
	MyFindCTFResultsPanel.cpp 
	PickingResultsPanel.cpp 
	MyAddRunCommandDialog.cpp 
	MyNewProjectWizard.cpp 
	MyNewRefinementPackageWizard.cpp 
	UnblurResultsPanel.cpp 
	ShowCTFResultsPanel.cpp 
	PickingResultsDisplayPanel.cpp 
	CTF1DPanel.cpp 
	FindCTFPanel.cpp 
	FindParticlesPanel.cpp 
	MyVolumeAssetPanel.cpp 
	MyResultsPanel.cpp 
	MyParticlePositionExportDialog.cpp 
	MyFrealignExportDialog.cpp 
	MyRelionExportDialog.cpp 
	MyRefinementPackageAssetPanel.cpp 
	MyRefine3DPanel.cpp 
	MyRefine2DPanel.cpp 
	ClassificationPlotPanel.cpp 
	AbInitioPlotPanel.cpp 
	Refine2DResultsPanel.cpp 
	AngularDistributionPlotPanel.cpp 
	DisplayPanel.cpp 
	ClassumSelectionCopyFromDialog.cpp 
	MyOverviewPanel.cpp 
	AbInitio3DPanel.cpp 
	AssetPickerComboPanel.cpp 
	AutoRefine3dPanel.cpp 
	DisplayRefinementResultsPanel.cpp 
	ImportRefinementPackageWizard.cpp 
	PopupTextDialog.cpp 
	ExportRefinementPackageWizard.cpp 
	Generate3DPanel.cpp 
	LargeAngularPlotDialog.cpp 
	RefinementParametersDialog.cpp 
	Sharpen3DPanel.cpp 
	PlotCurvePanel.cpp 
	DistributionPlotDialog.cpp 
	RefineCTFPanel.cpp
	ShowTemplateMatchResultsPanel.cpp
)

add_dependencies(cisTEM_gui cisTEM_gui_core)
add_dependencies(cisTEM_gui cisTEM_core)
target_link_libraries(cisTEM_gui cisTEM_core)
target_link_libraries(cisTEM_gui cisTEM_gui_core)
target_link_libraries(cisTEM_gui ${wxWidgets_LIBRARIES})
