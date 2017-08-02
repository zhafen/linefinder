#!/usr/bin/env python

#SBATCH --job-name=worldline
#SBATCH --partition=normal
## Stampede node has 16 processors & 32 GB
#SBATCH --nodes=1
#SBATCH --ntasks=16
#SBATCH --time=0:05:00
#SBATCH --output=worldline_jobs/%j.out
#SBATCH --error=worldline_jobs/%j.err
#SBATCH --mail-user=zhafen@u.northwestern.edu
#SBATCH --mail-type=begin
#SBATCH --mail-type=fail
#SBATCH --mail-type=end
#SBATCH --account=TG-AST140023

'''Script for tracking and classifying particles.
This can be submitted as a batch job using: sbatch run_worldline.py
Or it can simply be run in an interactive session with: ./run_worldline.py

@author: Zach Hafen
@contact: zachary.h.hafen@gmail.com
@status: Development
'''

import numpy as np

import worldline.classify as classify
import worldline.galaxy_find as galaxy_find
import worldline.track as track

########################################################################
# Input Parameters
########################################################################

# Which parts of the pipeline to run?
run_tracking = True
run_galaxy_finding = True
run_classifying = True

# Information about the input data
sdir = '../../tests/data/test_data_with_new_id_scheme'
ahf_sdir = '../../tests/data/ahf_test_data'
types = [ 0, 4, ]
snap_ini = 500
snap_end = 600
snap_step = 50
# By default, we assume that we've run AHF on every snapshot (we better have),
#   and that we're running tracking on all snapshots
mtree_halos_index = snap_end

# Information about what the output data should be called.
outdir =  '../../tests/data/tracking_output'
tag = 'test'

# Tracking Parameters
tracker_kwargs = {
  'sdir' : sdir,
  'types' : types,
  'snum_start' : snap_ini,
  'snum_end' : snap_end,
  'snum_step' : snap_step,

  'outdir' : outdir,
  'tag' : tag,

  'check_same_sdir' : False,
}

# Galaxy Finding Parameters
gal_finder_kwargs = {
  'sdir' : ahf_sdir,
  'mtree_halos_index' : mtree_halos_index,

  'tracking_dir' : outdir,
  'tag' : tag,
}

# Classifying Parameters
classifier_kwargs = {
  'sdir' : ahf_sdir,
  'mtree_halos_index' : mtree_halos_index,

  'tracking_dir' : outdir,
  'tag' : tag,

  'neg' : 1,
  'wind_vel_min_vc' : 2.,
  'wind_vel_min' : 15.,
  'time_min' : 100., 
  'time_interval_fac' : 5.,
  }

########################################################################
# Run the Tracking
########################################################################

if run_tracking:
  particle_tracker = track.ParticleTracker( **tracker_kwargs )
  particle_tracker.save_particle_tracks()

########################################################################
# Run the Galaxy Finding
########################################################################

if run_galaxy_finding:
  particle_track_gal_finder = galaxy_find.ParticleTrackGalaxyFinder( **gal_finder_kwargs )
  particle_track_gal_finder.find_galaxies_for_particle_tracks()

########################################################################
# Run the Classifying
########################################################################

if run_classifying:
  classifier = classify.Classifier( **classifier_kwargs )
  classifier.classify_particles()
