#!/bin/bash

#SBATCH --job-name=linefinder
#SBATCH --partition=skx-normal
#SBATCH --nodes=1
#SBATCH --ntasks=48
#SBATCH --time=6:00:00
#SBATCH --output=/scratch/03057/zhafen/linefinder_data/job_scripts/jobs/linefinder_CGM_trove.out
#SBATCH --error=/scratch/03057/zhafen/linefinder_data/job_scripts/jobs/linefinder_CGM_trove.err
#SBATCH --mail-user=zhafen@u.northwestern.edu
#SBATCH --mail-type=begin
#SBATCH --mail-type=fail
#SBATCH --mail-type=end

JUGFILE=/home1/03057/zhafen/repos/linefinder/linefinder/job_scripts/linefinder_CGM_trove_jugfile.py

for i in $(seq 4); do jug execute $JUGFILE & done

while ! jug check $JUGFILE; do
   jug sleep-until $JUGFILE 
done


echo 
echo Done waiting!!!!
echo 
