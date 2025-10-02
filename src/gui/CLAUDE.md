# GUI Development Guidelines for cisTEM

This file provides GUI-specific guidance for working with wxWidgets in cisTEM's graphical interface.

## Critical wxWidgets Safety Rules

### Printf Format Specifier Safety
**CRITICAL: Format specifier mismatches cause immediate segmentation faults in wxWidgets.**

```cpp
// CORRECT: Match format specifiers exactly to types
long id = 42;
wxPrintf("%ld", id);        // %ld for long
wxPrintf("%d", int(id));    // %d for int (explicitly cast)

// FATAL: Mismatched specifiers cause segfaults
wxPrintf("%d", id);         // SEGFAULT: %d with long
wxPrintf("%ld", int(id));   // SEGFAULT: %ld with int
```

### Unicode Character Restrictions
**Never use Unicode characters in wxPrintf format strings - they cause segfaults.**

```cpp
// FATAL: Unicode causes segmentation fault
wxPrintf("Resolution: 3.5Å");      // SEGFAULT: Å is Unicode
wxPrintf("Angle: 45°");            // SEGFAULT: ° is Unicode

// CORRECT: Use ASCII equivalents
wxPrintf("Resolution: 3.5A");      // Use 'A' not 'Å'
wxPrintf("Angle: 45 deg");         // Use 'deg' not '°'
```

## Memory Management Patterns

### wxWidgets Parent-Child Ownership
```cpp
// Parent-child hierarchy ensures automatic cleanup
wxDialog* dialog = new wxDialog(parent, ...);
wxButton* button = new wxButton(dialog, ...);  // Dialog owns button
// No manual deletion needed - parent deletes children

// NEVER use smart pointers with wxWindow objects
std::unique_ptr<wxDialog> dialog;  // WRONG: Causes double-deletion
```

### Static Members for Persistence
For data that must survive workflow switches or dialog recreation:
```cpp
// In header
class QueueManager {
    static std::deque<QueueItem> execution_queue;
    static long currently_running_id;
};

// In cpp - define static members
std::deque<QueueItem> QueueManager::execution_queue;
long QueueManager::currently_running_id = -1;
```

## Database Access Patterns

### Lazy Loading Pattern
**Never access database in constructors - main_frame may be invalid during workflow switches.**

```cpp
class MyWidget {
    bool needs_database_load = true;

    void OnFirstUse() {
        if (needs_database_load && main_frame && main_frame->current_project.is_open) {
            LoadFromDatabase();
            needs_database_load = false;
        }
    }
};
```

### SQL Query Best Practices
```sql
-- Format multi-line queries for readability
SELECT TM.SEARCH_ID,
       TM.PEAK_NUMBER,
       TM.STATUS AS TEMPLATE_STATUS,
       JS.STATUS AS JOB_STATUS
FROM TEMPLATE_MATCH_QUEUE AS TM
LEFT JOIN TEMPLATE_MATCH_JOB_SEARCH AS JS
    ON TM.SEARCH_ID = JS.SEARCH_ID
WHERE TM.STATUS IN ('pending', 'running')
ORDER BY TM.QUEUE_ORDER;
```

## Queue Manager Development Patterns

### Job Tracking Pattern
Track jobs started from queue manager for proper status updates:
```cpp
// In panel header
long running_queue_job_id = -1;

// When starting job
running_queue_job_id = job.template_match_id;

// In ProcessAllJobsFinished
if (running_queue_job_id > 0) {
    UpdateQueueStatus(running_queue_job_id, "complete");
    running_queue_job_id = -1;
}
```

### Bidirectional Friend Pattern
For clean communication between panels and queue managers:
```cpp
// In TemplateMatchPanel.h
friend class TemplateMatchQueueManager;

// In TemplateMatchQueueManager.h
friend class TemplateMatchPanel;

// Allows direct access to private methods for UI synchronization
queue_manager->UpdateUIAfterJobComplete(search_id);
```

## Common Workflow Panel Files

### Core Panel Infrastructure
- `src/gui/MyPanel.cpp/.h` - Base panel class
- `src/gui/ActionPanel.cpp/.h` - Panel with run controls
- `src/gui/ResultsPanel.cpp/.h` - Results display base

### Template Match Workflow
- `src/gui/MatchTemplatePanel.cpp/.h` - Main panel
- `src/gui/MatchTemplateResultsPanel.cpp/.h` - Results display
- `src/gui/TemplateMatchQueueManager.cpp/.h` - Queue management dialog

### Job Management
- `src/gui/MyRunProfilesPanel.cpp/.h` - Run profile management
- `src/gui/ProjectX_gui_job.cpp/.h` - Job execution framework

## Debugging Patterns

### Temporary Debug Code
Mark all temporary debugging with `// revert`:
```cpp
// revert - debug output for queue status tracking
wxPrintf("Queue status: %s\n", status);
```

### Building After Changes
After making GUI changes, always prompt the user to build the project to verify compilation:
- Ask: "Would you like me to build the project to verify these changes?"
- This ensures immediate feedback on any compilation issues

## Event Handling Best Practices

### Toggle Button State Management
**Note: This pattern may not be complete - toggle buttons sometimes require double-click on first use.**
```cpp
void OnToggleChanged(wxCommandEvent& event) {
    bool new_state = toggle_button->GetValue();

    // Update internal state
    is_enabled = new_state;

    // Update related UI elements
    related_checkbox->SetValue(new_state);

    // Skip event to allow further processing
    event.Skip();
}
```

### Workflow Switching Robustness
- Panels are destroyed and recreated during switches
- Don't assume persistence across workflows
- Store persistent state in database or static members

## Common Pitfalls to Avoid

1. **Never access database in constructors** - causes crashes during workflow switches
2. **Never use Unicode in wxPrintf** - causes immediate segfaults
3. **Never mix format specifiers with wrong types** - causes segfaults
4. **Never use smart pointers with wxWindow objects** - causes double-deletion
5. **Never assume panel persistence** - panels are recreated on workflow switches
6. **Never put complex logic in destructors** - wxWidgets manages cleanup

## File Organization

### Panel Structure
```
src/gui/
├── [Feature]Panel.cpp/.h           # Main workflow panel
├── [Feature]ResultsPanel.cpp/.h    # Results display
├── [Feature]QueueManager.cpp/.h    # Queue management (if applicable)
└── ProjectX_gui_[feature].cpp/.h   # GUI job handling
```

### Resource Files
```
src/gui/icons/
├── [feature]_icon.png              # Workflow icons
└── [action]_icon_*.png            # Action button icons
```