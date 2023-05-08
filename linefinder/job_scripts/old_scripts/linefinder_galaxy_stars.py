import linefinder.linefinder as linefinder

# Import for auxilliary function
import galaxy_dive.trends.data_products as data_products
import galaxy_dive.analyze_data.halo_data as halo_data
import numpy as np
import pandas as pd

########################################################################

halo_data_dir = '/scratch/03057/zhafen/halo_files/multiphysics/m12i_res7100_mhdcv'

def auxilliary_fn(
    dfid,
    df,
    id_finder,
    halo_data_dir,
    main_halo_id,
):
    '''Function for retrieving additional quantities to store in ptracks.
    '''

    # Get the radius from halo center
    h_data = halo_data.HaloData(
        data_dir = halo_data_dir,
        mt_kwargs = { 'index': 'snum', 'tag': 'smooth' },
    )
    r = np.zeros( df['P0'].shape, )
    pos_keys = [ 'P0', 'P1', 'P2' ]
    for i, h_pos_key in enumerate( [ 'Xc', 'Yc', 'Zc' ] ):
        origin = (
            h_data.get_mt_data(
                h_pos_key,
                snums = [ id_finder.snum ],
                a_power = 1.
            )[0] /
            id_finder.attrs['hubble']
        )
        xi = df[pos_keys[i]] - origin
        r += xi**2.
    r = np.sqrt( r )

    # Identify stars in the galaxy
    galaxy_radius = 4. * h_data.get_mt_data(
        'Rstar0.5',
        snums = [ id_finder.snum ],
        mt_halo_id = main_halo_id,
        a_power = 1.,
    )[0] / id_finder.attrs['hubble']
    in_galaxy = r < galaxy_radius
    is_star = df['PType'] == 4

    # Get the galaxy stellar mass
    m_star = df['M'][in_galaxy & is_star].sum()
    # Format (have to repeat for each particle....)
    dfid['M_star'] = np.full( dfid['P0'].shape, m_star )

    return dfid

def wrapped_auxilliary_fn( dfid, df, id_finder ):
    
    return auxilliary_fn(
        dfid,
        df,
        id_finder,
        halo_data_dir = halo_data_dir,
        main_halo_id = 0,
    )

linefinder_args = {
    # Identifying tag used as part of the filenames.
    # E.g. the IDs file will have the format `ids_{}.hdf5.format( tag )`.
    'tag': 'm12imhdcv_galaxy',
    
    # Location to place output in
    'out_dir': '$SCRATCH/linefinder_data/multiphysics/m12i_res7100_mhdcv/data',

    # Location of simulation data
    'sim_data_dir': '/scratch/projects/xsede/GalaxiesOnFIRE/mhdcv/m12i_res7100_mhdcv_old/output',

    # Location of halo file data
    'halo_data_dir': halo_data_dir,

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
        'install_firefly': True,
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
    'run_tracking': True,
    'run_galaxy_linking': True,
    'run_classifying': True,
    'run_visualization': True,
}

# Actually run Linefinder
linefinder.run_linefinder_jug(
    **linefinder_args
)
