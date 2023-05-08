'''Perform sampling for particle tracking.'''

import sys

import linefinder.select_particles as select

import trove

########################################################################

pm = trove.link_params_to_config(
    config_fp = sys.argv[1],
)

sampler = select.IDSampler(
    out_dir = pm['data_dir'],
    tag = pm['tag'],
    ignore_duplicates = True,
    p_types = [ 0, 4 ],
    snapshot_kwargs = {
        'sdir' : pm['sim_data_dir'],
        'halo_data_dir' : pm['halo_data_dir'],
        'main_halo_id': 0,
        'ahf_index': 600,
        'length_scale_used': 'R_vir',
    },
)
sampler.sample_ids()
    

