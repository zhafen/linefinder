#!/bin/bash

#SBATCH --job-name=linefinder
#SBATCH --partition=skx-normal
#SBATCH --nodes=1
#SBATCH --ntasks=48
#SBATCH --time=12:00:00
#SBATCH --output=/your/output/dir/trove.out
#SBATCH --error=/your/output/dir/trove.err
#SBATCH --mail-user=user@example.com
#SBATCH --mail-type=begin
#SBATCH --mail-type=fail
#SBATCH --mail-type=end

# Config path
CONFIG=$1


# trove clean $CONFIG
# trove evaluate $CONFIG
# The number following the "n" indicates the number of cores to use.
# This process is typically memory-limited, not CPU limited, so you will rarely use
# all the cores available.
trove execute -n 3 $CONFIG
