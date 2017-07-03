#!/bin/sh

#SBATCH --job-name=worldline
#SBATCH --partition=normal
## Stampede node has 16 processors & 32 GB
#SBATCH --nodes=3
##SBATCH --ntasks=16
#SBATCH --ntasks-per-node=3
#SBATCH --time=05:00:00
#SBATCH --output=worldline_jobs/%j.out
#SBATCH --error=worldline_jobs/%j.err
#SBATCH --mail-user=zhafen@u.northwestern.edu
#SBATCH --mail-type=begin
#SBATCH --mail-type=fail
#SBATCH --mail-type=end
#SBATCH --account=TG-AST140023

# Script for running worldline with some variations

#for galaxy_cut in 0.05 0.15 0.2 0.3 0.4 0.5 0.6 0.7 0.8
#  do ./run_worldline.py $galaxy_cut > ./worldline_jobs/w_$galaxy_cut.out &
#done
#wait

#seq $snap_num_start $snap_step $snap_num_end | xargs -n 1 -P $n_procs sh -c 'python convsnaps_list.py $0 $1 $2 $2 1' $snap_dir $out_dir

#galaxy_cuts=(0.05 0.15 0.2 0.3 0.4 0.5 0.6 0.7 0.8)
#seq ${#galaxy_cuts[@]} | xargs -n 1 -P $n_procs sh -c 'python ./run_worldline.py $0 > ./worldline_jobs/w_$0.out' ${galaxy_cuts[i]}

# Note: the -I command allows xargs to take in the output of what was before
n_procs=9
for galaxy_cut in 0.05 0.15 0.2 0.3 0.4 0.5 0.6 0.7 0.8; do echo $galaxy_cut; done | xargs -I {} -n 1 -P $n_procs sh -c 'python ./run_worldline_vary.py $0 > ./worldline_jobs/w_$0.out' {}

