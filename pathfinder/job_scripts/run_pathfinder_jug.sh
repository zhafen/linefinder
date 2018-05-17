#!/bin/bash

#SBATCH --job-name=pathfinder
#SBATCH --partition=skx-normal
#SBATCH --nodes=1
#SBATCH --ntasks=48
#SBATCH --time=3:00:00
#SBATCH --output=/scratch/03057/zhafen/pathfinder_data/job_scripts/jobs/%j.out
#SBATCH --error=/scratch/03057/zhafen/pathfinder_data/job_scripts/jobs/%j.err
#SBATCH --mail-user=zhafen@u.northwestern.edu
#SBATCH --mail-type=begin
#SBATCH --mail-type=fail
#SBATCH --mail-type=end
#SBATCH --account=TG-AST150059

JUGFILE=/home1/03057/zhafen/repos/pathfinder/pathfinder/job_scripts/pathfinder_CGM_trove_jugfile.py

for i in $(seq 4); do jug execute $JUGFILE & done

while ! jug check $JUGFILE; do
   jug sleep-until $JUGFILE 
done


echo 
echo Done waiting!!!!
echo 
