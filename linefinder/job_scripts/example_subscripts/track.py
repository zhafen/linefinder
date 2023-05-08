'''Perform particle tracking.'''

import sys

import linefinder.track as track

import trove

########################################################################

pm = trove.link_params_to_config(
    config_fp = sys.argv[1],
)

tracker = track.ParticleTracker(
    out_dir = pm['data_dir'],
    tag = pm['tag'],
    sdir = pm['sim_data_dir'],
    p_types = [ 0, 4 ],
    snum_start = 0,
    snum_end = 600,
    snum_step = 1,
)
tracker.save_particle_tracks_jug()
    

