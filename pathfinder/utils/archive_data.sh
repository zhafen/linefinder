#!/bin/bash
########################################################################
# Input Arguments
########################################################################

# What simulation to use, and where to put the output
ahf_data_dir=/scratch/03057/zhafen/pathfinder_data/core/m11b_res2100/data
archive_dir=${ARCHIVER}:/home1/03057/zhafen/SCRATCH_stamp/pathfinder_data/core/m11b_res2100

filetypes_to_archive=("ids*CGM*hdf5" "ptracks*CGM*hdf5" "galids*CGM*hdf5" "classifications*CGM*hdf5" "events*CGM*hdf5")
archive_filenames=(ids_CGM.tar ptracks_CGM.tar galids_CGM.tar classifications_CGM.tar events_CGM.tar)

tar_data=true
archive_data=true

########################################################################
# Start Data Processing
########################################################################

# Stop on errors
set -e

echo 
echo '########################################################################'
echo Starting Up
echo '########################################################################'

echo Storing AHF data in $ahf_data_dir
echo Archiving data at $archive_dir
pipeline_location=$( pwd )
echo Starting in $pipeline_location

echo Checking that we have all the filenames we need...
num_filetypes_to_archive=${#filetypes_to_archive[@]}
num_archive_filenames=${#archive_filenames[@]}
if [ $num_filetypes_to_archive != $num_archive_filenames ]
  then
    echo "Number of filenames doesn't match number of filetypes!"
    echo Number of filetypes to archive = $num_filetypes_to_archive
    echo Number of archive filenames = $num_archive_filenames
    echo "Exiting..."
    exit 1
fi

# Move to the data location
cd $ahf_data_dir

########################################################################
# Tar the data
########################################################################

if $tar_data; then

  echo 
  echo '########################################################################'
  echo Tarring Data
  echo '########################################################################'

  for i in $( seq 0 $(($num_filetypes_to_archive-1)) );
  do
    tar -cvf ${archive_filenames[$i]} ${filetypes_to_archive[$i]} 
  done
fi

########################################################################
# Move the data to the archived location
########################################################################

if $archive_data; then

  echo 
  echo '########################################################################'
  echo Archiving Data
  echo '########################################################################'

  for i in $( seq 0 $(($num_filetypes_to_archive-1)) );
  do
    rsync --progress ${archive_filenames[$i]} $archive_dir/
  done
fi

########################################################################
# Wrap up
########################################################################

echo 
echo '########################################################################'
echo Done!
