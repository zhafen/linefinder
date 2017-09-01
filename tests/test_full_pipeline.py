#!/usr/bin/env python
'''Testing.

@author: Zach Hafen
@contact: zachary.h.hafen@gmail.com
@status: Development
'''

import h5py
import numpy as np
import os
import pytest
import unittest

import worldline.classify as classify
import worldline.galaxy_find as galaxy_find
import worldline.track as track

########################################################################
# Global variables

# Information about the input data
sdir = './tests/data/test_data_with_new_id_scheme'
ahf_sdir = './tests/data/ahf_test_data'
types = [ 0, 4, ]
snap_ini = 500
snap_end = 600
snap_step = 50
# By default, we assume that we've run AHF on every snapshot (we better have),
#   and that we're running tracking on all snapshots
mtree_halos_index = snap_end

# Information about what the output data should be called.
outdir =  './tests/data/tracking_output_for_analysis'
tag = 'analyze'

# Tracking Parameters
tracker_kwargs = {
  'out_dir' : outdir,
  'tag' : tag,
}

# Galaxy Finding Parameters
gal_finder_kwargs = {
  'ahf_data_dir' : ahf_sdir,
  'main_mt_halo_id' : 0,
  'mtree_halos_index' : mtree_halos_index,

  'out_dir' : outdir,
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

ptracks_filename = os.path.join( outdir, 'ptracks_analyze.hdf5' )
galids_filename = os.path.join( outdir, 'galids_analyze.hdf5' )
classifications_filename = os.path.join( outdir, 'classifications_analyze.hdf5' )
events_filename = os.path.join( outdir, 'events_analyze.hdf5' )

########################################################################

# Decorator for skipping slow tests
slow = pytest.mark.skipif(
    not pytest.config.getoption("--runslow"),
    reason="need --runslow option to run"
)

########################################################################
########################################################################

class TestFullWorldline( unittest.TestCase ):
  '''These are really integration tests.'''

  def setUp( self ):

    for filename in [ ptracks_filename, galids_filename, classifications_filename, events_filename ]:
      if os.path.isfile( filename ):
        os.remove( filename )

  ########################################################################

  @slow
  def test_full_pipeline( self ):
    '''Except the id selecting... This makes sure the full pipeline just runs.'''

    particle_tracker = track.ParticleTracker( **tracker_kwargs )
    particle_tracker.save_particle_tracks()

    particle_track_gal_finder = galaxy_find.ParticleTrackGalaxyFinder( **gal_finder_kwargs )
    particle_track_gal_finder.find_galaxies_for_particle_tracks()

    classifier = classify.Classifier( **classifier_kwargs )
    classifier.classify_particles()

########################################################################
########################################################################

class TestCreateAnalysisData( unittest.TestCase ):
  '''Strictly speaking, these aren't really tests, so much as a kind of hacky way to generate test data for
  testing the analysis tools. Of course, if they break then something's wrong.'''

  @slow
  def test_create_classification_data( self ):

    f = h5py.File( classifications_filename, 'a' )

    for key in [ 'is_mass_transfer', 'is_merger', 'is_preprocessed', 'is_pristine', 'is_wind' ]:
      del f[key]

    f['is_mass_transfer'] = np.array( [ 0, 1, 0, 0, ] ).astype( bool )
    f['is_merger'] = np.array( [ 0, 0, 1, 1, ] ).astype( bool )
    f['is_preprocessed'] = np.array( [ 0, 1, 1, 1, ] ).astype( bool )
    f['is_pristine'] = np.array( [ 1, 0, 0, 0, ] ).astype( bool )
    f['is_wind'] = np.array( [
      [ 0, 0, 0, ],
      [ 1, 1, 0, ],
      [ 1, 1, 1, ],
      [ 0, 0, 0, ],
      ] ).astype( bool )
    f.close()
