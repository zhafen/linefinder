import linefinder.track as track

kwargs = {
    'sdir': './linefinder/tests/data/test_data_with_new_id_scheme',
    'p_types': [0, ],
    'snum_start': 500,
    'snum_end': 600,
    'snum_step': 50,

    'out_dir': './linefinder/tests/data/tracking_output',
    'ids_tag': 'test',
    'tag': 'test_jug',
}

particle_tracker = track.ParticleTracker( **kwargs )
particle_tracker.save_particle_tracks_jug()
