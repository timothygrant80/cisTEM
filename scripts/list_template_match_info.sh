#!/bin/bash

# Helper script to list available template matching jobs and image groups from a cisTEM database
# Usage: list_template_match_info.sh <database_path>

set -e

if [ $# -ne 1 ]; then
    echo "Usage: $0 <database_path>"
    echo ""
    echo "This script lists available template matching jobs and image groups"
    echo "to help you choose parameters for filter_template_matches_to_group.sh"
    echo ""
    echo "Arguments:"
    echo "  database_path : Path to the cisTEM project database file (.db)"
    exit 1
fi

DATABASE_PATH="$1"

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

echo "=== cisTEM Database Information ==="
echo "Database: $DATABASE_PATH"
echo ""

# List available template matching jobs
echo "=== Available Template Matching Jobs ==="
TEMPLATE_JOBS=$(sqlite3 "$DATABASE_PATH" "SELECT TEMPLATE_MATCH_JOB_ID, JOB_NAME, DATETIME_OF_RUN, COUNT(*) as NUM_RESULTS FROM TEMPLATE_MATCH_LIST GROUP BY TEMPLATE_MATCH_JOB_ID ORDER BY TEMPLATE_MATCH_JOB_ID;" 2>/dev/null || echo "")

if [ -z "$TEMPLATE_JOBS" ]; then
    echo "No template matching jobs found in database."
else
    echo "Job ID | Job Name | Date/Time | Number of Results"
    echo "-------|----------|-----------|------------------"
    echo "$TEMPLATE_JOBS" | while IFS='|' read -r job_id job_name datetime num_results; do
        printf "%-6s | %-20s | %-10s | %s\n" "$job_id" "$job_name" "$datetime" "$num_results"
    done
fi

echo ""

# List available image groups
echo "=== Available Image Asset Groups ==="
IMAGE_GROUPS=$(sqlite3 "$DATABASE_PATH" "SELECT g.GROUP_ID, g.GROUP_NAME, COUNT(m.IMAGE_ASSET_ID) as NUM_MEMBERS FROM IMAGE_GROUP_LIST g LEFT JOIN IMAGE_GROUP_1 m ON g.GROUP_ID = 1 GROUP BY g.GROUP_ID, g.GROUP_NAME UNION SELECT g.GROUP_ID, g.GROUP_NAME, COUNT(m.IMAGE_ASSET_ID) as NUM_MEMBERS FROM IMAGE_GROUP_LIST g LEFT JOIN IMAGE_GROUP_2 m ON g.GROUP_ID = 2 GROUP BY g.GROUP_ID, g.GROUP_NAME UNION SELECT g.GROUP_ID, g.GROUP_NAME, COUNT(m.IMAGE_ASSET_ID) as NUM_MEMBERS FROM IMAGE_GROUP_LIST g LEFT JOIN IMAGE_GROUP_3 m ON g.GROUP_ID = 3 GROUP BY g.GROUP_ID, g.GROUP_NAME UNION SELECT g.GROUP_ID, g.GROUP_NAME, COUNT(m.IMAGE_ASSET_ID) as NUM_MEMBERS FROM IMAGE_GROUP_LIST g LEFT JOIN IMAGE_GROUP_4 m ON g.GROUP_ID = 4 GROUP BY g.GROUP_ID, g.GROUP_NAME UNION SELECT g.GROUP_ID, g.GROUP_NAME, COUNT(m.IMAGE_ASSET_ID) as NUM_MEMBERS FROM IMAGE_GROUP_LIST g LEFT JOIN IMAGE_GROUP_5 m ON g.GROUP_ID = 5 GROUP BY g.GROUP_ID, g.GROUP_NAME ORDER BY GROUP_ID;" 2>/dev/null || echo "")

if [ -z "$IMAGE_GROUPS" ]; then
    echo "No image groups found in database."
else
    echo "Group ID | Group Name | Number of Members"
    echo "---------|------------|------------------"
    
    # Use a simpler approach to get group info
    sqlite3 "$DATABASE_PATH" "SELECT GROUP_ID, GROUP_NAME FROM IMAGE_GROUP_LIST ORDER BY GROUP_ID;" | while IFS='|' read -r group_id group_name; do
        # Try to count members in the specific group table
        member_count=$(sqlite3 "$DATABASE_PATH" "SELECT COUNT(*) FROM IMAGE_GROUP_$group_id;" 2>/dev/null || echo "0")
        printf "%-8s | %-20s | %s\n" "$group_id" "$group_name" "$member_count"
    done
fi

echo ""

# For each template matching job, show a summary of detection counts
echo "=== Template Match Detection Summary ==="
TEMPLATE_JOBS_SIMPLE=$(sqlite3 "$DATABASE_PATH" "SELECT DISTINCT TEMPLATE_MATCH_JOB_ID FROM TEMPLATE_MATCH_LIST ORDER BY TEMPLATE_MATCH_JOB_ID;" 2>/dev/null || echo "")

if [ -n "$TEMPLATE_JOBS_SIMPLE" ]; then
    for JOB_ID in $TEMPLATE_JOBS_SIMPLE; do
        echo "Job ID $JOB_ID detection counts:"
        
        # Get all template match IDs for this job
        TM_IDS=$(sqlite3 "$DATABASE_PATH" "SELECT TEMPLATE_MATCH_ID FROM TEMPLATE_MATCH_LIST WHERE TEMPLATE_MATCH_JOB_ID=$JOB_ID;")
        
        if [ -n "$TM_IDS" ]; then
            TOTAL_IMAGES=0
            TOTAL_DETECTIONS=0
            MIN_DETECTIONS=999999
            MAX_DETECTIONS=0
            
            for TM_ID in $TM_IDS; do
                PEAK_COUNT=$(sqlite3 "$DATABASE_PATH" "SELECT COUNT(*) FROM TEMPLATE_MATCH_PEAK_LIST_$TM_ID;" 2>/dev/null || echo "0")
                TOTAL_IMAGES=$((TOTAL_IMAGES + 1))
                TOTAL_DETECTIONS=$((TOTAL_DETECTIONS + PEAK_COUNT))
                
                if [ "$PEAK_COUNT" -lt "$MIN_DETECTIONS" ]; then
                    MIN_DETECTIONS=$PEAK_COUNT
                fi
                if [ "$PEAK_COUNT" -gt "$MAX_DETECTIONS" ]; then
                    MAX_DETECTIONS=$PEAK_COUNT
                fi
            done
            
            if [ "$TOTAL_IMAGES" -gt 0 ]; then
                AVG_DETECTIONS=$((TOTAL_DETECTIONS / TOTAL_IMAGES))
                echo "  Images processed: $TOTAL_IMAGES"
                echo "  Total detections: $TOTAL_DETECTIONS"
                echo "  Average detections per image: $AVG_DETECTIONS"
                echo "  Min detections: $MIN_DETECTIONS"
                echo "  Max detections: $MAX_DETECTIONS"
                
                # Show distribution
                echo "  Distribution of detection counts:"
                for threshold in 0 1 5 10 20 50; do
                    count=0
                    for TM_ID in $TM_IDS; do
                        PEAK_COUNT=$(sqlite3 "$DATABASE_PATH" "SELECT COUNT(*) FROM TEMPLATE_MATCH_PEAK_LIST_$TM_ID;" 2>/dev/null || echo "0")
                        if [ "$PEAK_COUNT" -ge "$threshold" ]; then
                            count=$((count + 1))
                        fi
                    done
                    echo "    >= $threshold detections: $count images"
                done
            fi
        fi
        echo ""
    done
fi

echo "=== Usage Example ==="
echo "Based on the information above, you can use the filter script like this:"
echo ""
if [ -n "$TEMPLATE_JOBS_SIMPLE" ] && [ -n "$(sqlite3 "$DATABASE_PATH" "SELECT GROUP_NAME FROM IMAGE_GROUP_LIST LIMIT 1;" 2>/dev/null)" ]; then
    FIRST_JOB=$(echo "$TEMPLATE_JOBS_SIMPLE" | head -n1)
    FIRST_GROUP=$(sqlite3 "$DATABASE_PATH" "SELECT GROUP_NAME FROM IMAGE_GROUP_LIST LIMIT 1;" 2>/dev/null)
    echo "./filter_template_matches_to_group.sh \"$DATABASE_PATH\" \"$FIRST_GROUP\" 5 $FIRST_JOB"
    echo ""
    echo "This would add all images from job $FIRST_JOB that have 5 or more detections"
    echo "to the group named '$FIRST_GROUP'"
else
    echo "./filter_template_matches_to_group.sh \"$DATABASE_PATH\" \"GROUP_NAME\" MIN_DETECTIONS JOB_ID"
fi
