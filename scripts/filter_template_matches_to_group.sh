#!/bin/bash

# Script to filter template matching results by number of detections and add to image group
# Usage: filter_template_matches_to_group.sh <database_path> <group_name> <min_detections> <template_match_job_id>

set -e

if [ $# -ne 4 ]; then
    echo "Usage: $0 <database_path> <group_name> <min_detections> <template_match_job_id>"
    echo ""
    echo "Arguments:"
    echo "  database_path      : Path to the cisTEM project database file (.db)"
    echo "  group_name         : Name of the existing image asset group to add results to"
    echo "  min_detections     : Minimum number of template match detections required"
    echo "  template_match_job_id : Template matching job ID to filter results from"
    echo ""
    echo "Example:"
    echo "  $0 /path/to/project.db 'Good_Matches' 5 1"
    exit 1
fi

DATABASE_PATH="$1"
GROUP_NAME="$2"
MIN_DETECTIONS="$3"
TEMPLATE_MATCH_JOB_ID="$4"

# Check if database file exists
if [ ! -f "$DATABASE_PATH" ]; then
    echo "Error: Database file '$DATABASE_PATH' does not exist"
    exit 1
fi

# Check if sqlite3 is available
if ! command -v sqlite3 &> /dev/null; then
    echo "Error: sqlite3 command not found. Please install sqlite3."
    exit 1
fi

echo "Filtering template matching results..."
echo "Database: $DATABASE_PATH"
echo "Group name: $GROUP_NAME"
echo "Minimum detections: $MIN_DETECTIONS"
echo "Template match job ID: $TEMPLATE_MATCH_JOB_ID"
echo ""

# First, check if the group exists and get its ID
GROUP_ID=$(sqlite3 "$DATABASE_PATH" "SELECT GROUP_ID FROM IMAGE_GROUP_LIST WHERE GROUP_NAME='$GROUP_NAME';" 2>/dev/null || echo "")

if [ -z "$GROUP_ID" ]; then
    echo "Error: Image group '$GROUP_NAME' does not exist."
    echo "Available groups:"
    sqlite3 "$DATABASE_PATH" "SELECT GROUP_ID, GROUP_NAME FROM IMAGE_GROUP_LIST ORDER BY GROUP_ID;"
    exit 1
fi

echo "Found group '$GROUP_NAME' with ID: $GROUP_ID"

# Get the template match IDs for the specified job
TEMPLATE_MATCH_IDS=$(sqlite3 "$DATABASE_PATH" "SELECT TEMPLATE_MATCH_ID FROM TEMPLATE_MATCH_LIST WHERE TEMPLATE_MATCH_JOB_ID=$TEMPLATE_MATCH_JOB_ID;")

if [ -z "$TEMPLATE_MATCH_IDS" ]; then
    echo "Error: No template matching results found for job ID $TEMPLATE_MATCH_JOB_ID"
    echo "Available template match job IDs:"
    sqlite3 "$DATABASE_PATH" "SELECT DISTINCT TEMPLATE_MATCH_JOB_ID FROM TEMPLATE_MATCH_LIST ORDER BY TEMPLATE_MATCH_JOB_ID;"
    exit 1
fi

echo "Found $(echo "$TEMPLATE_MATCH_IDS" | wc -l) template match results for job ID $TEMPLATE_MATCH_JOB_ID"

# Get the current maximum member number in the group
MAX_MEMBER_NUM=$(sqlite3 "$DATABASE_PATH" "SELECT COALESCE(MAX(MEMBER_NUMBER), 0) FROM IMAGE_GROUP_$GROUP_ID;" 2>/dev/null || echo "0")

ADDED_COUNT=0
NEXT_MEMBER_NUM=$((MAX_MEMBER_NUM + 1))

# Create a temporary file for batch operations
TEMP_SQL=$(mktemp)

echo "BEGIN TRANSACTION;" > "$TEMP_SQL"

# For each template match ID, count the peaks and add to group if meets criteria
for TM_ID in $TEMPLATE_MATCH_IDS; do
    # Get the image asset ID for this template match
    IMAGE_ASSET_ID=$(sqlite3 "$DATABASE_PATH" "SELECT IMAGE_ASSET_ID FROM TEMPLATE_MATCH_LIST WHERE TEMPLATE_MATCH_ID=$TM_ID;")
    
    # Count the number of peaks/detections for this template match
    PEAK_COUNT=$(sqlite3 "$DATABASE_PATH" "SELECT COUNT(*) FROM TEMPLATE_MATCH_PEAK_LIST_$TM_ID;" 2>/dev/null || echo "0")
    
    if [ "$PEAK_COUNT" -ge "$MIN_DETECTIONS" ]; then
        # Check if this image is already in the group
        EXISTING=$(sqlite3 "$DATABASE_PATH" "SELECT COUNT(*) FROM IMAGE_GROUP_$GROUP_ID WHERE IMAGE_ASSET_ID=$IMAGE_ASSET_ID;" 2>/dev/null || echo "0")
        
        if [ "$EXISTING" -eq 0 ]; then
            # Add to the group
            echo "INSERT INTO IMAGE_GROUP_$GROUP_ID (MEMBER_NUMBER, IMAGE_ASSET_ID) VALUES ($NEXT_MEMBER_NUM, $IMAGE_ASSET_ID);" >> "$TEMP_SQL"
            ADDED_COUNT=$((ADDED_COUNT + 1))
            NEXT_MEMBER_NUM=$((NEXT_MEMBER_NUM + 1))
            
            # Get image asset name for logging
            IMAGE_NAME=$(sqlite3 "$DATABASE_PATH" "SELECT NAME FROM IMAGE_ASSETS WHERE IMAGE_ASSET_ID=$IMAGE_ASSET_ID;")
            echo "Adding image $IMAGE_ASSET_ID ($IMAGE_NAME) with $PEAK_COUNT detections"
        else
            IMAGE_NAME=$(sqlite3 "$DATABASE_PATH" "SELECT NAME FROM IMAGE_ASSETS WHERE IMAGE_ASSET_ID=$IMAGE_ASSET_ID;")
            echo "Skipping image $IMAGE_ASSET_ID ($IMAGE_NAME) - already in group (has $PEAK_COUNT detections)"
        fi
    else
        IMAGE_NAME=$(sqlite3 "$DATABASE_PATH" "SELECT NAME FROM IMAGE_ASSETS WHERE IMAGE_ASSET_ID=$IMAGE_ASSET_ID;")
        echo "Skipping image $IMAGE_ASSET_ID ($IMAGE_NAME) - only $PEAK_COUNT detections (< $MIN_DETECTIONS)"
    fi
done

echo "COMMIT;" >> "$TEMP_SQL"

# Execute the batch operations
if [ "$ADDED_COUNT" -gt 0 ]; then
    echo ""
    echo "Adding $ADDED_COUNT images to group '$GROUP_NAME'..."
    sqlite3 "$DATABASE_PATH" < "$TEMP_SQL"
    echo "Successfully added $ADDED_COUNT images to group '$GROUP_NAME'"
else
    echo ""
    echo "No images met the criteria (>= $MIN_DETECTIONS detections) or all qualifying images were already in the group"
fi

# Clean up
rm -f "$TEMP_SQL"

echo ""
echo "Current group '$GROUP_NAME' contains $(sqlite3 "$DATABASE_PATH" "SELECT COUNT(*) FROM IMAGE_GROUP_$GROUP_ID;") images"
echo "Done!"
