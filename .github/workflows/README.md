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

This workflow will first cancel any active runs with the specified status, then delete them. The workflow automatically maps user-friendly status names to their API equivalents (e.g., "action required" to "action_required").