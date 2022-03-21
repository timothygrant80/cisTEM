#ifndef _gui_gui_core_headers_h_
#define _gui_gui_core_headers_h_

#include <wx/artprov.h>
#include <wx/xrc/xmlres.h>
#include <wx/gdicmn.h>
#include <wx/listbook.h>
#include <wx/listctrl.h>
#include <wx/font.h>
#include <wx/colour.h>
#include <wx/settings.h>
#include <wx/string.h>
#include <wx/sizer.h>
#include <wx/panel.h>
#include <wx/treectrl.h>
#include <wx/button.h>
#include <wx/statbox.h>
#include <wx/splitter.h>
#include <wx/frame.h>
#include <wx/statline.h>
#include <wx/stattext.h>
#include <wx/checkbox.h>
#include <wx/combobox.h>
#include <wx/textctrl.h>
#include <wx/dialog.h>
#include <wx/bitmap.h>
#include <wx/rawbmp.h>
#include <wx/image.h>
#include <wx/icon.h>
#include <wx/statbmp.h>
#include <wx/tglbtn.h>
#include <wx/spinctrl.h>
#include <wx/gauge.h>
#include <wx/string.h>
#include <wx/dnd.h>
#include <wx/progdlg.h>
#include <wx/choicdlg.h>
#include <wx/richtext/richtextctrl.h>
#include <wx/richtext/richtextstyles.h>
#include <wx/datectrl.h>
#include <wx/dateevt.h>
#include <wx/dataview.h>
#include <wx/graphics.h>
#include <wx/aui/auibook.h>
#include <wx/clipbrd.h>
#include <wx/odcombo.h>
#include <wx/wupdlock.h>

#include "../core/core_headers.h"

#include "../gui/gui_globals.h"

#include "../gui/job_panel.h"
#include "../gui/ProjectX_gui.h"
#include "../gui/gui_functions.h"
#include "../gui/ResultsDataViewListCtrl.h"
#include "../gui/BitmapPanel.h"
#include "../gui/PickingBitmapPanel.h"
#include "gui_job_controller.h"
#include "../gui/mathplot.h"
#include "../gui/my_controls.h"
#include "../gui/UnblurResultsPanel.h"
#include "../gui/CTF1DPanel.h"
#include "../gui/MainFrame.h"
#include "../gui/ErrorDialog.h"
#include "../gui/MyAssetPanelParent.h"
#include "../gui/MyMovieAssetPanel.h"
#include "../gui/MyImageAssetPanel.h"
#include "../gui/MyParticlePositionAssetPanel.h"
#include "../gui/MyVolumeAssetPanel.h"
#include "../gui/MovieImportDialog.h"
#include "../gui/MyVolumeImportDialog.h"
#include "../gui/MyMovieFilterDialog.h"
#include "../gui/MyImageImportDialog.h"
#include "../gui/AlignMoviesPanel.h"
#include "../gui/ShowCTFResultsPanel.h"
#include "../gui/ShowTemplateMatchResultsPanel.h"
#include "../gui/FindCTFPanel.h"
#include "../gui/FindParticlesPanel.h"
#include "../gui/MyMovieAlignResultsPanel.h"
#include "../gui/MyFindCTFResultsPanel.h"
#include "../gui/PickingResultsDisplayPanel.h"
#include "../gui/MyRunProfilesPanel.h"
#include "../gui/MyAddRunCommandDialog.h"
#include "../gui/MyNewProjectWizard.h"
#include "../gui/ImportRefinementPackageWizard.h"
#include "../gui/ExportRefinementPackageWizard.h"
#include "../gui/MyNewRefinementPackageWizard.h"
#include "../gui/MyResultsPanel.h"
#include "../gui/ActionsPanelSpa.h"
#include "../gui/ActionsPanelTm.h"
#include "../gui/MyAssetsPanel.h"
#include "../gui/MySettingsPanel.h"
#include "../gui/PickingResultsPanel.h"
#include "../gui/MyParticlePositionExportDialog.h"
#include "../gui/MyFrealignExportDialog.h"
#include "../gui/MyRelionExportDialog.h"
#include "../gui/MyRefinementPackageAssetPanel.h"
#include "../gui/MyRenameDialog.h"
#include "../gui/MyVolumeChooserDialog.h"
#include "../gui/MyRefine2DPanel.h"
#include "../gui/MyRefine3DPanel.h"
#include "../gui/Generate3DPanel.h"
#include "../gui/PlotFSCPanel.h"
#include "../gui/MyFSCPanel.h"
#include "../gui/MyRefinementResultsPanel.h"
#include "../gui/AngularDistributionPlotPanel.h"
#include "../gui/DisplayPanel.h"
#include "../gui/ClassificationPlotPanel.h"
#include "../gui/AbInitioPlotPanel.h"
#include "../gui/Refine2DResultsPanel.h"
#include "../gui/ClassumSelectionCopyFromDialog.h"
#include "../gui/MyOverviewPanel.h"
#include "../gui/AbInitio3DPanel.h"
#include "../gui/AssetPickerComboPanel.h"
#include "../gui/AutoRefine3dPanel.h"
#include "../gui/DisplayRefinementResultsPanel.h"
#include "../gui/PopupTextDialog.h"
#include "../gui/LargeAngularPlotDialog.h"
#include "../gui/RefinementParametersDialog.h"
#include "../gui/Sharpen3DPanel.h"
#include "../gui/PlotCurvePanel.h"
#include "../gui/DistributionPlotDialog.h"
#include "../gui/RefineCTFPanel.h"
#include "../gui/MatchTemplatePanel.h"
#include "../gui/MatchTemplateResultsPanel.h"
#include "../gui/RefineTemplatePanel.h"

#ifdef EXPERIMENTAL
#include "../gui/MyExperimentalPanel.h"
#include "../gui/RefineTemplateDevPanel.h"
#include "../gui/AtomicCoordinatesAssetPanel.h"
#include "../gui/AtomicCoordinatesChooserDialog.h"
#include "../gui/AtomicCoordinatesImportDialog.h"
#endif
// FIXME: These and all the panel integers should be in defines as enums, and should be in their own header
// included at the top of core headers
#define REFINEMENT 0
#define RECONSTRUCTION 1
#define MERGE 2
#define STARTUP 3
#define PREPARE_STACK 4
#define ALIGN_SYMMETRY 5
#define ESTIMATE_BEAMTILT 6
#define NOJOB 7

#endif
