# GitHub Workflows for cisTEM

This directory contains GitHub Actions workflows for the cisTEM repository.

## Maintenance Workflows

### Cleanup Old Actions

The `cleanup_actions.yml` workflow can be manually triggered to delete workflow runs older than a specified number of days.

To use:
1. Go to "Actions" tab in GitHub
2. Select "Cleanup Old Actions" workflow
3. Click "Run workflow"
4. Optional: Enter the number of days to keep (default: 30)
5. Click "Run workflow"

### Delete Workflows By Status

The `delete_workflows_by_status.yml` workflow allows you to cancel and delete workflow runs that have a specific status.

To use:
1. Go to "Actions" tab in GitHub
2. Select "Delete Workflows By Status" workflow
3. Click "Run workflow"
4. Optional: Enter the status of workflows to delete (default: action required)
5. Click "Run workflow"

Common workflow statuses include:
- `action required` - Workflows waiting for user input
- `cancelled` - Workflows that were canceled
- `failure` - Failed workflows
- `success` - Successfully completed workflows
- `in progress` - Currently running workflows
- `queued` - Workflows waiting to be run
- `waiting` - Workflows waiting for conditions or approval

When the workflow runs, it will print out all unique API status values found in your workflow history, which can be helpful for troubleshooting. If the workflow doesn't find any matching runs, you can try running it again using the exact API status value shown in the output.

### Status Matching Logic

The workflow uses three different methods to match the requested status with workflow runs:

1. **Exact match** (case-insensitive): The status strings match exactly (ignoring case)
2. **Normalized match**: After removing spaces, underscores, and dashes, the strings match
3. **Partial match**: One normalized string contains the other

This flexible matching helps ensure that variations like `action required`, `action_required`, and `actionrequired` all match the same workflows.

The workflow will first cancel any active runs with the specified status, then delete them. It automatically maps user-friendly status names to their API equivalents (e.g., "action required" to "action_required").
