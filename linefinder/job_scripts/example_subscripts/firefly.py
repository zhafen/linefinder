'''Perform visualization.'''

import sys

import linefinder.visualize as visualize
from linefinder.analyze_data import worldlines as analyze_worldlines
from linefinder.analyze_data import plot_worldlines


import trove

########################################################################

pm = trove.link_params_to_config(
    config_fp = sys.argv[1],
)

export_to_firefly_kwargs = {
    'firefly_dir': pm['firefly_dir'],
    'classifications': [ None, ],
    'classification_ui_labels': [ 'Tracks', ],
    'tracked_properties': [
        'logT',
        'logZ',
        'logDen',
        'PType',
        't_tcools',
        't_tacc',
        'tacc_tcools_tiled',
        'tacc_tiled',
        'tcools_tiled',
        'Vr',
    ],
    'use_default_colors': True,
    'include_halo_tracks': True,
    'data_subdir': pm['variation'],
}

# Setup
w = analyze_worldlines.Worldlines(
    data_dir = pm['data_dir'],
    tag = pm['tag'],
    halo_data_dir = pm['halo_data_dir'],
)

# Calculate
tacc_inds = w.calc_tacc_inds(
    lookback_time_max = pm['lookback_time_max'],
    choose_first = True,
)
w.calc_tacc()
tcool_inds = w.calc_tcools_inds(
    lookback_time_max = pm['lookback_time_max'],
    choose_first = pm['choose_first'],
    B = pm['logTcools'],
)
w.calc_tcools()

w.data['tacc_tcools'] = ( w.get_data( 'tacc' ) - w.get_data( 'tcools' ) )

visualize.export_to_firefly(
    w = w,
    install_firefly = False,
    include_instantaneous = False,
    export_to_firefly_kwargs = export_to_firefly_kwargs,
)
