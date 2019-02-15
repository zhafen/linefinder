import linefinder.linefinder as linefinder
import linefinder.config as linefinder_config

import linefinder.utils.file_management as file_management

########################################################################

sim_name = 'm12i'
'''The simulation to run tracking on.'''

tag = '{}_sightline'.format( sim_name )
'''Identifying tag used as part of the filenames.
E.g. the IDs file will have the format `ids_{}.hdf5.format( tag )`.
'''

# Tracking Parameters
tracker_kwargs = {
    # What particle types to track. Typically just stars and gas.
    'p_types': [ 0, 4,],

    # What snapshots to compile the particle tracks for.
    'snum_start': 1,
    'snum_end': 600,
    'snum_step': 1,
}

file_manager = file_management.FileManager()

sampler_kwargs = {
    'ignore_duplicates': True,

    'p_types': [ 0, 4 ],

    'snapshot_kwargs': {
        'sdir': file_manager.get_sim_dir( sim_name ),
        'halo_data_dir': file_manager.get_halo_dir( sim_name ),
        'main_halo_id': linefinder_config.MAIN_MT_HALO_ID[sim_name],
        'ahf_index': 600,
        'length_scale_used': 'R_vir',
    }
}

visualization_kwargs = {
    'install_firefly': True,
    'export_to_firefly_kwargs': {
        'firefly_dir': '/work/03057/zhafen/firefly_repos/sightline',
        'classifications': [
            'is_in_CGM',
            'is_CGM_IGM_accretion',
            'is_CGM_wind',
            'is_CGM_satellite_wind',
            'is_CGM_satellite_ISM',
        ],
        'classification_ui_labels': [ 'All', 'IGMAcc', 'Wind', 'SatWind', 'Sat' ],
        'tracked_properties': [
            'logT',
            'logZ',
            'logDen',
            'vr_div_v_cool',
            'logvr_div_v_cool_offset',
        ],
        'tracked_filter_flags': [ True, ] * 5,
        'tracked_colormap_flags': [ True, ] * 5,
        'snum': 465,
    },
}

# This is the actual function that runs linefinder.
# In general you don't need to touch this function but if you want to,
# for example, turn off one of the steps because you're rerunning and you
# already did that step, you can do so below.
linefinder.run_linefinder_jug(
    sim_name = sim_name,
    tag = tag,
    galdef = '_galdefv3',
    # The galdef is a set of parameters used for the galaxy linking and
    # classification steps. Don't touch this unless you know what you're doing.
    tracker_kwargs = tracker_kwargs,
    sampler_kwargs = sampler_kwargs,
    visualization_kwargs = visualization_kwargs,
    run_id_selecting = False,
    run_id_sampling = False,
    run_tracking = False,
    run_galaxy_linking = False,
    run_classifying = False,
)
