#ifndef _SRC_CORE_UPDATE_PROGRESS_TRACKER_H
#define _SRC_CORE_UPDATE_PROGRESS_TRACKER_H

// An interface a caller of Database::UpdateSchema() can implement to follow
// the progress of a schema update (the desktop GUI's MainFrame used to; a
// console program may pass nullptr). It lived in src/gui and is kept here,
// wx-base only, so libcore does not depend on GUI code.
class UpdateProgressTracker {
  public:
    virtual void OnUpdateProgress(int progress, wxString new_msg, bool& should_update_text) = 0;
    virtual void OnCompletion( )                                                            = 0;
};
#endif
