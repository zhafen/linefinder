import linefinder.linefinder as linefinder

# Import for auxilliary function
import galaxy_dive.trends.data_products as data_products
import galaxy_dive.analyze_data.halo_data as halo_data
import numpy as np
import pandas as pd

########################################################################

sim_name = 'm12m'

linefinder_args = {
    # Identifying tag used as part of the filenames.
    # E.g. the IDs file will have the format `ids_{}.hdf5.format( tag )`.
    'tag': '{}_hothaloacc'.format( sim_name ),

    'sim_name': sim_name,
    
    # Location to place output in
    # 'out_dir': '$SCRATCH/linefinder_data/multiphysics/m12i_res7100_mhdcv/data',

    # Location of simulation data
    # 'sim_data_dir': '/scratch/projects/xsede/GalaxiesOnFIRE/multiphysics/m12i_res7100_mhdcv/output',

    # Location of halo file data
    # 'halo_data_dir': '/scratch/03057/zhafen/multiphysics/m12i_res7100_mhdcv/halo',

    # Arguments for id sampling
    'sampler_kwargs': {
        'ignore_duplicates': True,
        'p_types': [0, 4],
        'snapshot_kwargs': {
            'ahf_index': 600,
            'length_scale_used': 'R_vir',
        },
    },

    # Arguments used for the particle tracking step
    'tracker_kwargs': {
        # What particle types to track. Typically just stars and gas.
        'p_types': [ 0, 4,],

        # What snapshots to compile the particle tracks for.
        'snum_start': 1,
        'snum_end': 600,
        'snum_step': 1,
    },

    # Arguments used for the visualization step
    'visualization_kwargs': {
        'install_firefly': True,
        # These kwargs are used for tuning the Firefly visualization
        'export_to_firefly_kwargs': {
            'firefly_dir': '/scratch/03057/zhafen/firefly_repos/hot-halo-accretion',
            # 'classifications': [ None ],
            # 'classification_ui_labels': [ 'All' ],
            'use_default_colors': False,
        },
    },

    # The following arguments are for turning on/off different parts
    # of the pipeline
    'run_id_selecting': False,
    # Most users will identify the list of IDs using their own methods, so
    # we turn ID selecting off.
    'run_id_sampling': False,
    'run_tracking': False,
    'run_galaxy_linking': True,
    'run_classifying': True,
    'run_visualization': True,
}

# Actually run Linefinder
linefinder.run_linefinder_jug(
    **linefinder_args
)
