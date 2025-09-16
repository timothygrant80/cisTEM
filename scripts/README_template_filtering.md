# Template Matching Result Filtering Scripts

This directory contains bash scripts to filter template matching results by number of detections and add them to existing image asset groups in cisTEM projects.

## Scripts

### 1. `list_template_match_info.sh`

Lists available template matching jobs and image groups from a cisTEM database to help you choose parameters.

**Usage:**

```bash
./list_template_match_info.sh <database_path>
```

**Example:**

```bash
./list_template_match_info.sh /path/to/project.db
```

This script will show:

- Available template matching jobs with their IDs and number of results
- Available image asset groups
- Detection count statistics for each template matching job
- Usage examples for the filtering script

### 2. `filter_template_matches_to_group.sh`

Filters template matching results by minimum number of detections and adds qualifying images to an existing image asset group.

**Usage:**

```bash
./filter_template_matches_to_group.sh <database_path> <group_name> <min_detections> <template_match_job_id>
```

**Parameters:**

- `database_path`: Path to the cisTEM project database file (.db)
- `group_name`: Name of the existing image asset group to add results to
- `min_detections`: Minimum number of template match detections required
- `template_match_job_id`: Template matching job ID to filter results from

**Example:**

```bash
./filter_template_matches_to_group.sh /path/to/project.db "Good_Matches" 5 1
```

This would add all images from template matching job #1 that have 5 or more detections to the group named "Good_Matches".

## Workflow

1. **First, explore your data:**

   ```bash
   ./list_template_match_info.sh /path/to/your/project.db
   ```

   This will show you available template matching jobs and their detection statistics.

2. **Create or identify a target image group** in the cisTEM GUI if needed.

3. **Filter and add results:**

   ```bash
   ./filter_template_matches_to_group.sh /path/to/your/project.db "Your_Group_Name" 10 2
   ```

   This adds images from job #2 with 10+ detections to "Your_Group_Name".

## Notes

- The target image group must already exist in the database
- Images already in the target group will be skipped (no duplicates)
- The script uses SQLite transactions for safe database operations
- Detection counts are based on the `TEMPLATE_MATCH_PEAK_LIST_` tables
- All operations are logged to show which images are being added/skipped

## Requirements

- `sqlite3` command-line tool
- Read/write access to the cisTEM project database
- Bash shell

## Database Schema

The scripts work with these cisTEM database tables:

- `TEMPLATE_MATCH_LIST`: Contains template matching job results
- `TEMPLATE_MATCH_PEAK_LIST_<id>`: Contains detected peaks for each template match
- `IMAGE_GROUP_LIST`: Contains image group definitions
- `IMAGE_GROUP_<id>`: Contains members of each image group
- `IMAGE_ASSETS`: Contains image asset information

## Safety

- Always backup your database before running these scripts
- The scripts use SQLite transactions to ensure database consistency
- Operations are logged so you can see exactly what changes are made
