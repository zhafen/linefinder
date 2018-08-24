import pathfinder.pathfinder as pathfinder

########################################################################
# Global variables

# Information about the input data
sdir = './tests/data/test_data_with_new_id_scheme'
halo_data_dir = './tests/data/ahf_test_data'
types = [ 0, 4, ]
snap_ini = 500
snap_end = 600
snap_step = 50
# By default, we assume that we've run AHF on every snapshot (we better have),
#   and that we're running tracking on all snapshots
mtree_halos_index = snap_end

# Information about what the output data should be called.
out_dir = './tests/data/full_pathfinder_output'
tag = 'jug'

selector_kwargs = {
    'snum_start': snap_ini,
    'snum_end': snap_end,
    'snum_step': snap_step,

    'p_types': types,

    'snapshot_kwargs': {
        'sdir': sdir,
        'load_additional_ids': True,
        'ahf_index': mtree_halos_index,
        'analysis_dir': halo_data_dir,
    }
}

sampler_kwargs = {
    'n_samples': 2,
}

# Tracking Parameters
tracker_kwargs = {
}

# Galaxy Finding Parameters
gal_finder_kwargs = {
    'halo_data_dir': halo_data_dir,
    'main_mt_halo_id': 0,

    'length_scale': 'Rvir',
    'mt_length_scale': 'Rvir',
}

# Classifying Parameters
classifier_kwargs = {
    'velocity_scale': 'Vc(Rvir)',
}

pathfinder.run_pathfinder_jug(
    out_dir = out_dir,
    tag = tag,
    selector_kwargs = selector_kwargs,
    sampler_kwargs = sampler_kwargs,
    tracker_kwargs = tracker_kwargs,
    gal_finder_kwargs = gal_finder_kwargs,
    classifier_kwargs = classifier_kwargs,
)
