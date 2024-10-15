#ifndef _SRC_GUI_UPDATE_PROGRESS_TRACKER_H
#define _SRC_GUI_UPDATE_PROGRESS_TRACKER_H

// This class exists only as an interface for overriding in MainFrame.cpp.
// Its main purpose is to allow the GUI to track database schema update
// progress without exposing any more GUI code to the database than is
// strictly necessary to avoid any weird dependency and other issues,
// such as substantially increased compile time.
// It is possible to generalize this for database-releated tracking
class UpdateProgressTracker {
  public:
    virtual void OnUpdateProgress(int progress, wxString new_msg, bool& should_update_text) = 0;
    virtual void OnCompletion( )                                                            = 0;
};
#endif