#!/bin/bash

#SBATCH --job-name=linefinder
#SBATCH --partition=skx-normal
#SBATCH --nodes=1
#SBATCH --ntasks=48
#SBATCH --time=6:00:00
#SBATCH --output=/scratch/03057/zhafen/linefinder_data/job_scripts/jobs/linefinder.out
#SBATCH --error=/scratch/03057/zhafen/linefinder_data/job_scripts/jobs/linefinder.err
##SBATCH --mail-user=zhafen@u.northwestern.edu
##SBATCH --mail-type=begin
##SBATCH --mail-type=fail
##SBATCH --mail-type=end

# Number of cores to use. Remember to account for memory constraints!
NCORES=$2

JUGFILE=$1

# Run JUG
for i in $(seq $NCORES); do jug execute $JUGFILE & done

# Workaround to check if JUG is done
while ! jug check $JUGFILE; do
   jug sleep-until $JUGFILE 
done


echo 
echo Done waiting!!!!
echo 
