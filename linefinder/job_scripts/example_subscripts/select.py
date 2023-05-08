'''Perform sampling for particle tracking.'''

import sys

import linefinder.select_particles as select

import trove

########################################################################

pm = trove.link_params_to_config(
    config_fp = sys.argv[1],
)

# Example filters
selector_data_filters = {
    'Radial Cut' : { 'data_key': 'Rf', 'data_min': 0.1, 'data_max': 1.0, },
    # 'Density Cut' : { 'data_key': 'HDen', 'data_min': 0.1, 'data_max': np.inf, }
}

id_selector = select.IDSelector(
    out_dir = pm['data_dir'],
    tag = pm['tag'],
    snum_start = 0,
    snum_end = 600,
    snum_step = 1,
    p_types = [ 0, 4 ],
    snapshot_kwargs = {
        'sdir' : pm['sim_data_dir'],
        'halo_data_dir' : pm['halo_data_dir'],
        'main_halo_id': 0,
        'ahf_index': 600,
        'length_scale_used': 'R_vir',
    },
)
id_selector.select_ids_jug( selector_data_filters )
    

