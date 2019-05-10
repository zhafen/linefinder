import linefinder.linefinder as linefinder

# Import for auxilliary function
import galaxy_dive.trends.data_products as data_products
import galaxy_dive.analyze_data.halo_data as halo_data
import numpy as np
import pandas as pd

########################################################################

def auxilliary_fn(
    dfid,
    df,
    id_finder,
    halo_data_dir,
    main_halo_id,
):
    '''Function for retrieving additional quantities to store in ptracks.
    '''

    # Get the tidal tensor data
    tidal_df = data_products.tidal_tensor_data_grudic(
        id_finder.snum,
        ids = dfid.index,
    )

    # Get the enclosed mass data
    h_data = halo_data.HaloData(
        data_dir = halo_data_dir,
        mt_kwargs = { 'index': 'snum', 'tag': 'smooth' },
    )
    positions = np.array([
        dfid['P0'],
        dfid['P1'],
        dfid['P2'],
    ]).transpose()
    dfid['M_enc'] = h_data.get_enclosed_mass( 
        positions = positions,
        snum = id_finder.snum,
        hubble_param = id_finder.attrs['hubble'],
        mt_halo_id = main_halo_id,
    )

    return dfid

def wrapped_auxilliary_fn( dfid, df, id_finder ):
    
    return auxilliary_fn(
        dfid,
        df,
        id_finder,
        halo_data_dir = '/scratch/03057/zhafen/multiphysics/m12i_res7100_mhdcv/halo',
        main_halo_id = 0,
    )

linefinder_args = {
    # Identifying tag used as part of the filenames.
    # E.g. the IDs file will have the format `ids_{}.hdf5.format( tag )`.
    'tag': 'm12imhdcv_clustersofFIRE_pop1',
    
    # Location to place output in
    'out_dir': '$SCRATCH/linefinder_data/multiphysics/m12i_res7100_mhdcv/data',

    # Location of simulation data
    'sim_data_dir': '/scratch/projects/xsede/GalaxiesOnFIRE/multiphysics/m12i_res7100_mhdcv/output',

    # Location of halo file data
    'halo_data_dir': '/scratch/03057/zhafen/multiphysics/m12i_res7100_mhdcv/halo',

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

        'custom_fns': [ wrapped_auxilliary_fn, ],
    },

    # Arguments used for the visualization step
    'visualization_kwargs': {
        'install_firefly': False,
        # These kwargs are used for tuning the Firefly visualization
        'export_to_firefly_kwargs': {
            'firefly_dir': '/work/03057/zhafen/firefly_repos/clustersofFIRE',
            'classifications': [ None ],
            'classification_ui_labels': [ 'All' ],
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
    'run_galaxy_linking': False,
    'run_classifying': False,
    'run_visualization': True,
}

# Actually run Linefinder
linefinder.run_linefinder_jug(
    **linefinder_args
)
