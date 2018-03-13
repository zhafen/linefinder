import pathfinder.select as select

# For IDSelector
kwargs = {
    'snum_start': 500,
    'snum_end': 600,
    'snum_step': 100,
    'p_types': [0, 4],
    'out_dir': './tests/data/tracking_output',
    'tag': 'test_jug',

    'snapshot_kwargs': {
        'sdir': './tests/data/stars_included_test_data',
        'load_additional_ids': True,
        'ahf_index': 600,
        'ahf_data_dir': './tests/data/ahf_test_data',
    },
}

data_filters = {
    'radial_cut': { 'data_key': 'Rf', 'data_min': 0., 'data_max': 1., },
}

id_selector = select.IDSelector( **kwargs )
id_selector.select_ids_jug( data_filters )
