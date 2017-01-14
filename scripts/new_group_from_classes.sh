#!/bin/bash
#
# Alexis Rohou
# January 2017
#

db_fn=$1
par_fn=$2
sel_fn=$3

#
# Work out list of particles we want to keep
#
tmp_fn="/tmp/selected_images.txt"
rm -f ${tmp_fn}
while read current_class_number; do
  #echo "Pulling out particles from class ${current_class_number}"
  awk '$8 == '$current_class_number' {print $1}' $par_fn >> ${tmp_fn}
done < $sel_fn

number_of_selected_particles=$(wc -l $tmp_fn | awk '{print $1}')
echo "${number_of_selected_particles} particles were selected"


#
# How many groups of particle assets do we already have?
#
new_group_id=$( sqlite3 $db_fn "select max(group_id) from particle_position_group_list;" )
(( new_group_id++ ))
echo "Will create new group with id ${new_group_id}"

#
# Create this new group within the database
#
sqlite3 $db_fn "insert into particle_position_group_list (group_id,group_name,list_id) values (${new_group_id},'group from classes',${new_group_id});"
sqlite3 $db_fn "create table particle_position_group_${new_group_id} (member_number INTEGER, particle_position_asset_id INTEGER);"

tmp_sqlite_cmd_fn="/tmp/sqlite_command.txt"
echo "begin transaction;" > $tmp_sqlite_cmd_fn
member_number=0
while read particle_position_asset_id; do
  (( member_number++ ))
  echo "insert into particle_position_group_${new_group_id} (member_number,particle_position_asset_id) values (${member_number},${particle_position_asset_id});" >> $tmp_sqlite_cmd_fn
done < ${tmp_fn}
printf "\n"
echo "commit;" >> $tmp_sqlite_cmd_fn

sqlite3 $db_fn < $tmp_sqlite_cmd_fn

echo "All done"
