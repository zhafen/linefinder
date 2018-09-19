#!/usr/bin/env python
'''Tools for reading worldline data

@author: Zach Hafen
@contact: zachary.h.hafen@gmail.com
@status: Development
'''

import copy
import numpy as np
import numpy.testing as npt

import galaxy_dive.analyze_data.ahf as analyze_ahf_data
import galaxy_dive.analyze_data.generic_data as generic_data
import galaxy_dive.analyze_data.simulation_data as simulation_data
import galaxy_dive.read_data.snapshot as read_snapshot
import galaxy_dive.utils.astro as astro_tools
import galaxy_dive.utils.utilities as utilities

import ids
import ptracks
import galids
import classifications
import events

import linefinder.utils.presentation_constants as p_constants
import linefinder.config as config

########################################################################
########################################################################


class Worldlines( simulation_data.TimeData ):
    '''Wrapper for analysis of all data products. It loads data in
    on-demand.

    Args:
        data_dir (str):
            Data directory for the classified data

        tag (str):
            Identifying tag for the data to load.

        ids_tag (str):
            Identifying tag for ids data. Defaults to tag.

        ptracks_tag (str):
            Identifying tag for ptracks data. Defaults to tag.

        galids_tag (str):
            Identifying tag for galids data. Defaults to tag.

        classifications_tag (str):
            Identifying tag for classifications data. Defaults to tag.

        events_tag (str):
            Identifying tag for events data. Defaults to tag.

        label (str):
            Identifying label for the worldlines, used for plotting.
            Defaults to tag.

        color (str):
            What color to use when plotting.
    '''

    def __init__(
        self,
        data_dir,
        tag,
        ids_tag = None,
        ptracks_tag = None,
        galids_tag = None,
        classifications_tag = None,
        events_tag = None,
        **kwargs
    ):

        if ids_tag is None:
            ids_tag = tag
        if ptracks_tag is None:
            ptracks_tag = tag
        if galids_tag is None:
            galids_tag = tag
        if classifications_tag is None:
            classifications_tag = tag
        if events_tag is None:
            events_tag = tag

        # Store the arguments
        for arg in locals().keys():
            setattr( self, arg, locals()[arg] )

        self.ptracks_kwargs = dict( kwargs )

        data_masker = WorldlineDataMasker( self )
        key_parser = WorldlineDataKeyParser()

        self.data = {}

        super( Worldlines, self ).__init__( data_dir=data_dir, data_masker=data_masker, key_parser=key_parser, **kwargs )

    ########################################################################
    # Properties for loading data on the fly
    ########################################################################

    @property
    def ids( self ):

        if not hasattr( self, '_ids' ):
            self._ids = ids.IDs( self.data_dir, self.ids_tag, )

        return self._ids

    @ids.deleter
    def ids( self ):
        del self._ids

    ########################################################################

    @property
    def ptracks( self ):

        if not hasattr( self, '_ptracks' ):
            self._ptracks = ptracks.PTracks(
                self.data_dir, self.ptracks_tag, store_ahf_reader=True,
                **self.ptracks_kwargs
            )

        return self._ptracks

    @ptracks.deleter
    def ptracks( self ):
        del self._ptracks

    ########################################################################

    @property
    def galids( self ):

        if not hasattr( self, '_galids' ):
            self._galids = galids.GalIDs( self.data_dir, self.galids_tag )

        return self._galids

    @galids.deleter
    def galids( self ):
        del self._galids

    ########################################################################

    @property
    def classifications( self ):

        if not hasattr( self, '_classifications' ):
            self._classifications = classifications.Classifications( self.data_dir, self.classifications_tag )

        return self._classifications

    @classifications.deleter
    def classifications( self ):
        del self._classifications

    ########################################################################

    @property
    def events( self ):

        if not hasattr( self, '_events' ):
            self._events = events.Events( self.data_dir, self.events_tag )

        return self._events

    @events.deleter
    def events( self ):
        del self._events

    ########################################################################

    @property
    def halo_data( self ):
        '''Halo Data.

        TODO:
            Make it easier to get the parameters to use, without loading as
            much superfluous data.
        '''

        if not hasattr( self, '_halo_data' ):
            self._halo_data = analyze_ahf_data.HaloData(
                data_dir = self.halo_data_dir,
                tag = self.galids.parameters['halo_file_tag'],
            )

        return self._halo_data

    ########################################################################

    @property
    def base_data_shape( self ):

        return self.ptracks.base_data_shape

    ########################################################################

    @property
    def length_scale( self ):

        return self.ptracks.length_scale.values

    ########################################################################

    @property
    def n_snaps( self ):
        '''Number of snapshots, i.e. data points on the time axis.'''

        if not hasattr( self, '_n_snaps' ):
            self._n_snaps = self.ptracks.base_data_shape[1]

        return self._n_snaps

    ########################################################################

    @property
    def n_particles( self ):
        '''The number of particles tracked.'''

        if not hasattr( self, '_n_particles' ):
            self._n_particles = self.ptracks.base_data_shape[0]

        return self._n_particles

    ########################################################################

    @property
    def n_particles_presampled( self ):
        '''The number of particles selected, prior to sampling.'''

        if not hasattr( self, '_n_particles_presampled' ):
            self._n_particles_presampled = self.ids.data_attrs['n_particles']

        return self._n_particles_presampled

    ########################################################################

    @property
    def n_particles_snapshot( self ):
        '''The number of star and gas particles in the last snapshot tracked. Should be the same throughout the simulation,
        if there's conservation of star and gas particles.'''

        return self.n_particles_snapshot_gas + self.n_particles_snapshot_star

    ########################################################################

    @property
    def n_particles_snapshot_gas( self ):
        '''The number of gas particles in the last snapshot tracked.'''

        if not hasattr( self, '_n_particles_snapshot_gas' ):

            snapshot_kwargs = {
                'sdir': self.ids.snapshot_parameters['sdir'],
                'snum': self.ids.parameters['snum_end'],
                'ptype': config.PTYPE_GAS,
                'header_only': True,
            }

            snapshot = read_snapshot.readsnap( **snapshot_kwargs )

            self._n_particles_snapshot_gas = snapshot['npart']

        return self._n_particles_snapshot_gas

    @property
    def n_particles_snapshot_star( self ):
        '''The number of star particles in the last snapshot tracked.'''

        if not hasattr( self, '_n_particles_snapshot_star' ):

            snapshot_kwargs = {
                'sdir': self.ids.snapshot_parameters['sdir'],
                'snum': self.ids.parameters['snum_end'],
                'ptype': config.PTYPE_STAR,
                'header_only': True,
            }

            snapshot = read_snapshot.readsnap( **snapshot_kwargs )

            self._n_particles_snapshot_star = snapshot['npart']

        return self._n_particles_snapshot_star

    ########################################################################

    @property
    def redshift( self ):

        if not hasattr( self, '_redshift' ):
            self._redshift = self.ptracks.redshift

        return self._redshift

    @redshift.setter
    def redshift( self, value ):
        '''Setting function for simulation redshift property.'''

        # If we try to set it, make sure that if it already exists we don't change it.
        if hasattr( self, '_redshift' ):

            if isinstance( value, np.ndarray ) or isinstance( self._redshift, np.ndarray ):

                is_nan = np.any( [ np.isnan( value ), np.isnan( self._redshift ) ], axis=1 )
                not_nan_inds = np.where( np.invert( is_nan ) )[0]

                test_value = np.array(value)[not_nan_inds]  # Cast as np.ndarray because Pandas arrays can cause trouble.
                test_existing_value = np.array(self._redshift)[not_nan_inds]
                npt.assert_allclose( test_value, test_existing_value, atol=1e-5 )

                self._redshift = value

            else:
                npt.assert_allclose( value, self._redshift, atol=1e-5 )

        else:
            self._redshift = value

    ########################################################################

    @property
    def r_gal( self ):

        if not hasattr( self, '_r_gal' ):
            length_scale = self.galids.parameters['mt_length_scale']
            galaxy_cut = self.galids.parameters['galaxy_cut']
            self._r_gal = self.halo_data.get_mt_data(
                length_scale,
                a_power = 1.
            ) * galaxy_cut / self.ptracks.data_attrs['hubble']

        return self._r_gal

    ########################################################################

    @property
    def inner_CGM_boundary( self ):

        if not hasattr( self, '_inner_CGM_boundary' ):

            # Get the inner CGM radius as construed by the galaxy radius
            inner_r_gal = np.zeros( self.n_snaps )
            inner_r_gal_part = self.r_gal * ( 1. + config.F_GAP ) 
            inner_r_gal[:inner_r_gal_part.size] = inner_r_gal_part

            # Get the inner CGM radius as construed by the virial radisu
            inner_r_vir = config.INNER_CGM_BOUNDARY * np.array( self.r_vir )

            # Maximize the two
            self._inner_CGM_boundary = np.max(
                [ inner_r_gal, inner_r_vir ],
                axis = 0,
            )

        return self._inner_CGM_boundary

    ########################################################################

    @property
    def outer_CGM_boundary( self ):

        if not hasattr( self, '_outer_CGM_boundary' ):

            self._outer_CGM_boundary = (
                config.OUTER_CGM_BOUNDARY * np.array( self.r_vir )
            )

        return self._outer_CGM_boundary

    ########################################################################

    @property
    def snums( self ):

        return self.ptracks.snums

    ########################################################################

    @property
    def hubble_param( self ):

        return self.ptracks.data_attrs['hubble']

    ########################################################################

    @property
    def m_tot( self ):
        '''Total mass at the last snapshot.'''

        if not hasattr( self, '_m_tot' ):
            masses = self.get_data( 'M', sl=(slice(None), 0), )
            masses_no_invalids = np.ma.fix_invalid( masses ).compressed()
            self._m_tot = masses_no_invalids.sum()

        return self._m_tot

    ########################################################################

    @property
    def conversion_factor( self ):
        '''The ratio necessary to convert to the total mass from the sample mass.
        '''

        if not hasattr( self, '_conversion_factor' ):
            self._conversion_factor = float( self.n_particles_presampled ) / float( self.n_particles )

        return self._conversion_factor

    ########################################################################

    @property
    def mass_totals( self ):
        '''Get the total mass in the sample in the last snapshot in the canonical classifications.'''

        if not hasattr( self, '_mass_totals' ):
            self._mass_totals = {}
            for mass_category in [ 'is_pristine', 'is_merger', 'is_mass_transfer', 'is_wind' ]:
                self._mass_totals[mass_category] = self.get_selected_data(
                    'M',
                    sl=(slice(None), 0),
                    classification=mass_category,
                    fix_invalid=True,
                ).sum()

            self._mass_totals = utilities.SmartDict( self._mass_totals )

        return self._mass_totals

    ########################################################################

    @property
    def mass_fractions( self ):
        '''Get the mass fraction in the last snapshot in the canonical classifications.'''

        return self.mass_totals / self.m_tot

    ########################################################################

    @property
    def real_mass_totals( self ):
        '''Get the total mass (converted from the sample) in the last snapshot
        in the canonical classifications.'''

        return self.mass_totals * self.conversion_factor

    ########################################################################
    # Top Level Functions
    ########################################################################

    def clear_data( self ):
        '''Clear all loaded data.'''

        data_types = [
            'ids',
            'ptracks',
            'galids',
            'classifications',
            'events'
        ]
    
        for data_type in data_types:

            data_attr = '_{}'.format( data_type )

            if hasattr( self, data_attr ):
                delattr( self, data_attr )

    ########################################################################

    def get_parameters( self ):

        parameters = {}
        for data in [ 'ids', 'ptracks', 'galids', 'classifications' ]:

            parameters[data] = getattr( self, data ).parameters

        return parameters

    ########################################################################
    # Get Data
    ########################################################################

    def get_data( self, data_key, *args, **kwargs ):
        '''Get data. Usually just get it from ptracks. args and kwargs are passed to self.ptracks.get_data()

        Args:
            data_key (str) : What data to get?
            *args, **kwargs : Additional arguments to pass to other get_data() methods.

        Returns:
            data (np.ndarray) : Array of data.
        '''

        # First, look to see if this data is calculated in some easy to access location
        if data_key in self.data:
            data = self.data[data_key]

            # Apply the slice if that needs to be done.
            if 'sl' in kwargs:
                if kwargs['sl'] is not None:
                    data = data[kwargs['sl']]

            return data

        try:
            data = super( Worldlines, self ).get_data( data_key, *args, **kwargs )
            return data

        # A lot of the data can be calculated from the particle tracks data, so we can also try to access it from there.
        except KeyError, AttributeError:
            data = self.ptracks.get_data( data_key, *args, **kwargs )
            return data
        # TODO: Fix the structure s.t. it's improved from this.
        except AssertionError:
            data = self.ptracks.get_data( data_key, *args, **kwargs )
            return data

    ########################################################################

    def get_processed_data(
        self,
        data_key,
        tile_data = False,
        *args, **kwargs
    ):
        '''Get data, handling more complex data keys that indicate doing generic
        things to the data.

        Args:
            data_key (str) :
                What data to get?

            sl (object) :
                How to slice the data before returning it.

            tile_data (bool) :
                If True, tile data along a given direction. This is usually for
                data formatting purposes.

        Returns:
            data (np.ndarray) : Array of data.
        '''

        data_key, tiled_flag = self.key_parser.is_tiled_key( data_key )

        if tiled_flag:
            tile_data = True

        data = super( Worldlines, self ).get_processed_data(
            data_key,
            tile_data = tile_data,
            *args, **kwargs
        )

        return data

    ########################################################################

    def get_data_at_ind(
        self,
        data_key,
        ind_key,
        ind_shift = 0,
        units = None,
        units_a_power = 1.,
        units_h_power = -1.,
        return_units_only = False,
        tile_data = False,
        *args, **kwargs
    ):
        '''Get the data at a specified index for each particle.

        Args:
            data_key (str) : What data to get?
            ind_key (str) : What index to use?
            ind_shift (int) : Relative to the index identified by ind_key, how should the index be shifted?
            units (str) : If given, scale the data by this value, taken from the halo data.
            units_a_power (float) : If using units from the halo data, multiply by a to this power to convert.
            units_h_power (float) : If using units from the halo data, multiply by the hubble param to this power to convert.
            return_units_only (bool) : Return just the units argument. Useful for debugging.
            tile_data (bool) : If True, tile data before getting the data at a specific index.
            *args, **kwargs : Arguments to be passed to self.get_data()

        Returns:
            data_at_ind (np.ndarray) : Array of data, at the specified index.
        '''

        data = self.get_data( data_key, *args, **kwargs ).copy()

        if tile_data:

            if data.shape == ( self.n_particles, ):
                data = np.tile( data, ( self.n_snaps, 1) ).transpose()

            elif data.shape == ( self.n_snaps, ):
                data = np.tile( data, ( self.n_particles, 1) )

            else:
                raise Exception( "Unrecognized data shape, {}".format( data.shape ) )

        if issubclass( data.dtype.type, np.integer ):
            fill_value = config.INT_FILL_VALUE
        elif issubclass( data.dtype.type, np.float ) or issubclass( data.dtype.type, np.float32 ):
            fill_value = config.FLOAT_FILL_VALUE
        else:
            raise Exception( "Unrecognized data type, data.dtype = {}".format( data.dtype ) )

        data_at_ind = fill_value * np.ones( self.n_particles, dtype=data.dtype )

        specified_ind = self.get_data( ind_key, *args, **kwargs )

        # Look only at indices we retrieved successfully
        valid_inds = np.where( specified_ind != config.INT_FILL_VALUE )[0]
        valid_specified_ind = specified_ind[valid_inds]

        # Shift the indices by the specified amount, if desired
        valid_specified_ind += ind_shift

        data_at_ind[valid_inds] = data[valid_inds, valid_specified_ind]

        if units is not None:

            # Get the units out
            units_arr = self.halo_data.get_mt_data(
                units,
                mt_halo_id=self.galids.parameters['main_mt_halo_id'],
                a_power = units_a_power,
            ).copy()

            # Get the right indices out
            units_arr_at_ind = units_arr[valid_specified_ind]

            # Include any factors of h
            units_arr_at_ind *= self.ptracks.data_attrs['hubble']**units_h_power

            if return_units_only:
                units_arr_all = fill_value * np.ones( self.n_particles, dtype=data.dtype )
                units_arr_all[valid_inds] = units_arr_at_ind

                return units_arr_all

            data_at_ind[valid_inds] /= units_arr_at_ind

        return data_at_ind

    def get_data_first_acc( self, data_key, ind_after_first_acc=False, ind_relative_to_first_acc=0, *args, **kwargs ):
        '''Get data the snapshot immediately before accretion.

        Args:
            data_key (str) : What data to get?
            ind_after_first_acc (bool) : If True, get data the index immediately *after* first accretion, instead.
            ind_relative_to_first_acc (int) : Move the snapshot index relative to the snapshot before first accretion.
            *args, **kwargs : Arguments to be passed to self.get_data_at_ind()

        Returns:
            data_first_acc (np.ndarray) : Array of data, the index immediately after first accretion.
        '''

        assert not ( ind_after_first_acc and ( ind_relative_to_first_acc != 0 ) ), "Choose one option."

        # ind_first_acc is defined as the index at which a particle is first found in a galaxy,
        # so we need to shift things around accordingly
        if ind_after_first_acc:
            ind_shift = 0
        else:
            ind_shift = 1 + ind_relative_to_first_acc

        return self.get_data_at_ind( data_key, 'ind_first_acc', ind_shift, *args, **kwargs )

    def get_data_ind_star( self, data_key, *args, **kwargs ):
        '''Get data at the snapshot a particle is first identified as a star.

        Args:
            data_key (str) : What data to get?
            *args, **kwargs : Arguments to be passed to self.get_data_at_ind()

        Returns:
            data_ind_star (np.ndarray) : Array of data, at the index a particle is first identified as a star.
        '''

        return self.get_data_at_ind( data_key, 'ind_star', *args, **kwargs )

    ########################################################################

    def get_fraction_outside( self, data_key, data_min, data_max, *args, **kwargs ):
        '''Get the fraction of data outside a certain range. *args, **kwargs are arguments sent to mask the data.

        Args:
            data_key (str) : What data to get.
            data_min (float) : Lower bound of the data range.
            data_max (float) : Upper bound of the data range.

        Returns:
            f_outside (float) : Fraction outside the range.
        '''

        data = self.get_selected_data( data_key, *args, **kwargs )

        data_ma = np.ma.masked_outside( data, data_min, data_max )

        n_outside = float( data_ma.mask.sum() )
        n_all = float( data.size )

        return n_outside / n_all

    ########################################################################

    def get_selected_quantity(
            self,
            selection_routine='galaxy',
            ptype='star',
            quantity='mass',
            low_memory_mode=False,
            selected_quantity_data_key='M',
            *args,
            **kwargs
        ):
        '''Apply a selection routine, and then get out the total mass (or
        some other quantity) of particles that fulfill that criteria.

        Args:

            selection_routine (str) :
                What selection routine to run. E.g. 'galaxy' selects all
                particles in the main galaxy.

            ptype (str):
                What particle type inside the galaxy to consider.

            quantity (str):
                What quantity of the galaxy to retrieve.

            low_memory_mode (bool):
                If True, unload the data after getting the quantity
                (saves memory at the cost of convenience).

            *args, **kwargs :
                Additional arguments to be passed to self.get_selected_data()

        Returns:
            selected_quantity (np.ndarray) :
                Total mass for a particular particle type in the main galaxy
                (satisfying any additional requirements passed via *args and **kwargs)
                at each specified redshift.
        '''

        # Run the selection routine
        self.data_masker.run_selection_routine( selection_routine, ptype )

        data_ma = self.get_selected_data(
            selected_quantity_data_key,
            fix_invalid = True,
            compress = False,
            *args, **kwargs
        )

        if quantity == 'mass':

            try:
                # Test for the case when everything is masked.
                if np.invert( data_ma.mask ).sum() == 0:
                    return 0.
            # Case when nothing is masked.
            except AttributeError:
                pass

            selected_quantity = data_ma.sum( axis=0 )

            # Replace masked values with 0
            try:
                selected_quantity.fill_value = 0.
                selected_quantity = selected_quantity.filled()

            except AttributeError:
                pass

        elif quantity == 'n_particles':
            selected_quantity = np.invert( data_ma.mask ).sum( axis=0 )

        else:
            raise Exception(
                "Unrecognized selected_quantity, selected_quantity = {}"
                .format( selected_quantity )
            )

        if low_memory_mode:
            self.clear_data()

        return selected_quantity

    ########################################################################

    def get_selected_quantity_radial_bins(
        self,
        selection_routine='galaxy',
        ptype='star',
        quantity='mass',
        radial_bins = np.arange( 0., 1.1, 0.1 ),
        radial_bin_data_kwargs = {
            'scale_key': 'Rvir',
            'scale_a_power': 1.,
            'scale_h_power': -1.,
        },
        low_memory_mode=False,
        *args, **kwargs
    ):
        '''Apply a selection routine, and then get out the total mass (or
        some other quantity) of particles that fulfill that criteria,
        in specified radial bins.

        Args:

            selection_routine (str) :
                What selection routine to run. E.g. 'galaxy' selects all
                particles in the main galaxy.

            ptype (str):
                What particle type inside the galaxy to consider.

            quantity (str):
                What quantity to retrieve.

            radial_bins (np.ndarray) :
                Radial bins to use.

            radial_bin_data_kwargs (dict) :
                Arguments to change how the data is masked. For example,
                if you want to scale the data (done by default), use this
                dictionary to do so. These are arguments that would be passed
                to self.data_masker.mask_data and in turn
                self.data_masker.get_processed_data.

            low_memory_mode (bool):
                If True, unload the data after getting the quantity
                (saves memory at the cost of convenience).

            *args, **kwargs :
                Additional arguments to be passed to self.get_selected_data()

        Returns:
            selected_quantity (np.ndarray) :
                Total mass for a particular particle type in the main galaxy
                (satisfying any additional requirements passed via *args and **kwargs)
                at each specified redshift.
        '''

        # Get a fresh start
        self.data_masker.clear_masks( True )

        # Run the selection routine
        self.data_masker.run_selection_routine( selection_routine, ptype )

        # Loop over each radial bin and get the results out
        selected_quantity_radial_bins = []
        for i in range( radial_bins.size - 1 ):

            r_in = radial_bins[i]
            r_out = radial_bins[i + 1]

            radial_bin_mask_name = 'R{}'.format( i )

            self.data_masker.mask_data(
                'R',
                r_in,
                r_out,
                optional_mask = True,
                mask_name = radial_bin_mask_name,
                **radial_bin_data_kwargs
            )

            data_ma = self.get_selected_data(
                'M',
                fix_invalid = True,
                compress = False,
                optional_masks = [ radial_bin_mask_name ],
                *args, **kwargs
            )

            if quantity == 'mass':

                # Test for the case when everything is masked.
                if np.invert( data_ma.mask ).sum() == 0:
                    selected_quantity_radial_bins.append( 0. )
                    continue

                selected_quantity = data_ma.sum( axis=0 )

                # Replace masked values with 0
                try:
                    selected_quantity.fill_value = 0.
                    selected_quantity = selected_quantity.filled()

                except AttributeError:
                    pass

            selected_quantity_radial_bins.append( selected_quantity )

        if low_memory_mode:
            self.clear_data()

        return np.array( selected_quantity_radial_bins )

    ########################################################################

    def get_categories_selected_quantity(
        self,
        classification_list = p_constants.CLASSIFICATIONS_A,
        selected_quantity_method = 'get_selected_quantity',
        *args, **kwargs
    ):
        '''Get the total mass in the main galaxy for a particular particle type in each
        of a number of classification categories. This is only for particles that are tracked! This is not the real mass!

        Args:
            classification_list (list) :
                What classifications to use.

            selected_quantity_method (str) :
                Method to use for getting the selected quantity.
                For example, use 'get_selected_quantity_radial_bins' if you
                want the selected quantity in, well, radial bins.

            *args, **kwargs :
                Additional arguments to be passed to self.get_selected_data()

        Returns:
            categories_selected_quantity (SmartDict of np.ndarrays) :
                selected_quantity that fits each classification.
        '''

        selected_quantity_fn = getattr( self, selected_quantity_method )

        selected_quantity = {}
        for mass_category in classification_list:
            selected_quantity[mass_category] = selected_quantity_fn(
                classification = mass_category, *args, **kwargs )

        return utilities.SmartDict( selected_quantity )

    def get_categories_selected_quantity_fraction(
        self,
        normalization_category,
        classification_list = p_constants.CLASSIFICATIONS_A,
        selected_quantity_method = 'get_selected_quantity',
        *args, **kwargs
    ):
        '''Same as categories_selected_quantity, but as a fraction of the total
        mass in the main galaxy for a particular particle type.
        '''

        categories_selected_quantity = self.get_categories_selected_quantity(
            classification_list = classification_list,
            selected_quantity_method = selected_quantity_method,
            *args, **kwargs
        )

        selected_quantity_fn = getattr( self, selected_quantity_method )

        normalization = selected_quantity_fn(
            classification=normalization_category,
            *args, **kwargs
        )

        return categories_selected_quantity / normalization

    def get_categories_selected_quantity_extrapolated(
        self,
        classification_list = p_constants.CLASSIFICATIONS_A,
        *args, **kwargs
    ):
        '''Get the total mass in the main galaxy for a particular particle type in each
        of a number of classification categories.

        Args:
            classification_list (list) :
                What classifications to use.

            *args, **kwargs :
                Additional arguments to be passed to self.get_selected_data()

        Returns:
            categories_selected_quantity (SmartDict of np.ndarrays) :
                selected_quantity that fits each classification.
        '''

        categories_mass = self.get_categories_selected_quantity( classification_list=classification_list, *args, **kwargs )

        return categories_mass * self.conversion_factor

    ########################################################################

    def get_max_per_event_count(
        self,
        data_key,
        n_event_key,
        flatten = True,
        verbose = False,
        max_after_vmax = False,
        vmax_kwargs = {},
        *args, **kwargs
    ):
        '''Get the maximum value attained by a quantity for each time an event
        occurs.

        Args:
            data_key (str):
                The data to get the maximum for.

            n_event_key (str):
                The count of times an event has happened per particle.

            flatten (boolean):
                If True, return a flattened array. Else return a list of
                arrays, the ith index of which is the maximum for n_event=i.

            max_after_vmax (boolean):
                If True, get the max per event count, only after the max
                velocity for that event was reached. Useful when calculating
                the results of wind kicks.

            vmax_kwargs (dict):
                The max velocity per the event is the radial velocity,
                but by passing additional keyword arguments through this
                argument that velocity can be scaled by, for example,
                the circular velocity.

            *arg, **kwargs:
                Additional args when getting the data out.

        Returns:
            max_per_event_count (array-like):
                Result, sorted according first to n_event, and second by
                particle index.
        '''

        n_event = self.get_data( n_event_key )

        max_per_event_count = []
        for n in range( np.max( n_event )+1 ):

            # Get the data out
            data = self.get_selected_data(
                data_key,
                compress = False,
                *args, **kwargs
            )

            if verbose:
                print( 'n = {}'.format( n ) )

            # Get the mask for the data
            mask_this_event = ( n != n_event )

            # Get a mask for after max velocity
            if max_after_vmax:

                vel_mask_this_event = copy.copy( mask_this_event )

                vel = self.get_selected_data(
                    'Vr',
                    compress = False,
                    **vmax_kwargs
                )

                try:
                    # Modify data mask to account for matching event count
                    vel.mask = np.ma.mask_or( vel.mask, vel_mask_this_event )
                except AttributeError:
                    # Account for when no data is masked
                    vel = np.ma.masked_array( vel, mask=vel_mask_this_event )

                # Find relevant index
                vel_argmax_this_event = np.nanargmax(
                    vel,
                    axis = 1,
                )

                # Make velocity mask
                inds = self.get_processed_data( 'ind', tile_data = True )
                vel_mask = inds > vel_argmax_this_event[:,np.newaxis]

                # Merge masks
                mask_this_event = np.ma.mask_or( mask_this_event, vel_mask )

            # Mask the data
            try:
                # Modify data mask to account for matching event count
                data.mask = np.ma.mask_or( data.mask, mask_this_event )
            except AttributeError:
                # Account for when no data is masked
                data = np.ma.masked_array( data, mask=mask_this_event )

            # Get the max out
            max_this_event = np.nanmax(
                data,
                axis = 1,
            ).compressed()
            max_per_event_count.append( max_this_event )
                    
        # Format
        if flatten:
            max_per_event_count = np.hstack( np.array( max_per_event_count ) )

        return max_per_event_count

    ########################################################################
    # Generate Data on the Go
    ########################################################################

    def handle_data_key_error( self, data_key ):
        '''If we don't have a data_key stored, try and create it.

        Args:
            data_key (str) : The data key in question.

        Returns:
            self.data (dict) : If successful, stores the data here.
        '''

        try:
            super( Worldlines, self ).handle_data_key_error( data_key )

        # We do this second because it involves loading alot of data...
        except KeyError:
            if data_key in self.classifications.data.keys():
                self.data[data_key] = self.classifications.data[data_key]
                return True

            elif data_key in self.events.data.keys():
                self.data[data_key] = self.events.data[data_key]
                return True

            elif data_key in self.galids.data.keys():
                self.data[data_key] = self.galids.data[data_key]

    ########################################################################

    def calc_is_fresh_accretion( self ):
        '''Find material classified as fresh accretion (pristine gas that has not recycled).

        Returns:
            self.data['is_fresh_accretion'] ( np.ndarray ) : Result.
        '''

        pristine_tiled = np.tile( self.get_data( 'is_pristine' ), (self.n_snaps, 1) ).transpose()
        is_not_wind = np.invert( self.get_data( 'is_wind' ) )

        self.data['is_fresh_accretion'] = np.all( [ pristine_tiled, is_not_wind ], axis=0 )

    ########################################################################

    def calc_is_NEP_wind_recycling( self ):
        '''Find material classified as non-externally-processed wind recycling.

        Returns:
            self.data['is_NEP_wind_recycling'] ( np.ndarray ) : Result.
        '''

        pristine_tiled = np.tile( self.get_data( 'is_pristine' ), (self.n_snaps, 1) ).transpose()

        self.data['is_NEP_wind_recycling'] = np.all( [ pristine_tiled, self.get_data( 'is_wind' ) ], axis=0 )

    ########################################################################

    def calc_is_merger_star( self ):
        '''Find material classified as a merger, while being a star particle at time of first accretion.
        Caution: This is calculated at the snapshot first after accretion. The safer option may be to calculate at the
        snapshot immediately before first accretion.

        Returns:
            self.data['is_merger_star'] ( np.ndarray ) : Result.
        '''

        is_star_first_acc = self.get_data_first_acc( 'PType' ) == config.PTYPE_STAR

        self.data['is_merger_star'] = np.all( [ is_star_first_acc, self.get_data( 'is_merger' ) ], axis=0 )

    ########################################################################

    def calc_is_merger_gas( self ):
        '''Find material classified as a merger, while being gas at time of first accretion.
        Caution: This is calculated at the snapshot first after accretion. The safer option may be to calculate at the
        snapshot immediately before first accretion.

        Returns:
            self.data['is_merger_gas'] ( np.ndarray ) : Result.
        '''

        is_star_first_acc = self.get_data_first_acc( 'PType' ) == config.PTYPE_GAS

        self.data['is_merger_gas'] = np.all( [ is_star_first_acc, self.get_data( 'is_merger' ) ], axis=0 )

    ########################################################################

    def calc_is_classification_NYA(
        self,
        classification,
        tile_classification = True
    ):
        '''Find material with the given classification that is not yet accreted (NYA) onto the main galaxy.

        Args:
            classification (str) :
                What classification to get the result for.

            tile_classification (bool) :
                If True, then the input classification should be tiled.

        Returns:
            is_classification_NYA ( [n_particles, n_snaps] np.ndarray ) :
                The (i,j)th entry is True if particle i is not yet
                accreted by the jth index.
        '''

        if tile_classification:
            classification_key = '{}_tiled'.format( classification )
        else:
            classification_key = classification

        # Get the classification out first, tiled
        is_classification_NYA = self.get_processed_data( classification_key )

        # Find the indices after accreting
        ind_first_acc_tiled = self.get_processed_data( 'ind_first_acc_tiled' )
        ind_tiled = np.tile( range( self.n_snaps ), (self.n_particles, 1) )
        has_accreted = ind_tiled <= ind_first_acc_tiled

        # Update the classification to mask first accretion.
        is_classification_NYA[has_accreted] = False

        return is_classification_NYA

    def calc_is_NEP_NYA( self ):
        '''Find material classified as NEP that is not yet accreted (NYA) onto the main galaxy.

        Returns:
            self.data['is_mass_transfer_NYA'] ( np.ndarray ) : Result
        '''

        self.data['is_NEP_NYA'] = self.calc_is_classification_NYA( 'is_pristine' )

    def calc_is_hitherto_EP_NYA( self ):

        self.data['is_hitherto_EP_NYA'] = \
            self.calc_is_classification_NYA(
                'is_hitherto_EP',
                tile_classification = False )

    def calc_is_hitherto_NEP_NYA( self ):

        self.data['is_hitherto_NEP_NYA'] = \
            self.calc_is_classification_NYA(
                'is_hitherto_NEP',
                tile_classification = False
            )

    def calc_is_merger_NYA( self ):
        '''Find material classified as merger that is not yet accreted (NYA) onto the main galaxy.

        Returns:
            self.data['is_merger_NYA'] ( np.ndarray ) : Result
        '''

        self.data['is_merger_NYA'] = self.calc_is_classification_NYA( 'is_merger' )

    def calc_is_mass_transfer_NYA( self ):
        '''Find material classified as mass transfer that is not yet accreted (NYA) onto the main galaxy.

        Returns:
            self.data['is_mass_transfer_NYA'] ( np.ndarray ) : Result
        '''

        self.data['is_mass_transfer_NYA'] = self.calc_is_classification_NYA( 'is_mass_transfer' )

    ########################################################################

    def calc_is_IP( self ):
        '''Calculate internally processed material, defined as all material
        that has been inside the main galaxy.
        '''

        is_in_main_gal = self.get_data( 'is_in_main_gal' )

        time_weighted = (
            is_in_main_gal *
            self.get_processed_data(
                'dt',
                tile_data = True,
            )
        )

        summed = np.nancumsum( time_weighted[:,::-1], axis=1 )[:,::-1]

        self.data['is_IP'] = summed >= self.classifications.parameters['t_pro']

    ########################################################################

    def calc_is_in_CGM( self ):
        '''Material that is in the CGM.

        Returns:
            self.data['is_in_CGM'] (np.ndarray) :
                If True, the particle is currently in the CGM, as defined
                in Hafen+18.
        '''

        r_rvir = self.get_processed_data(
            'R',
            scale_key = 'Rvir',
            scale_a_power = 1.,
            scale_h_power = -1.,
        )
        is_in_CGM_rvir = ( r_rvir <= config.OUTER_CGM_BOUNDARY ) \
            & ( r_rvir >= config.INNER_CGM_BOUNDARY )

        r_gal_length_scale = self.get_processed_data(
            'R',
            scale_key = self.galids.parameters['length_scale'],
            scale_a_power = 1.,
            scale_h_power = -1.,
        )
        is_in_CGM_length_scale = r_gal_length_scale > (1. + config.F_GAP ) * \
            self.galids.parameters['galaxy_cut']

        is_in_CGM = is_in_CGM_rvir & is_in_CGM_length_scale

        self.data['is_in_CGM'] = is_in_CGM

    ########################################################################

    def calc_is_in_galaxy_halo_interface( self ):
        '''Calculate material that is in the CGM.'''

        r_rvir = self.get_processed_data(
            'R',
            scale_key = 'Rvir',
            scale_a_power = 1.,
            scale_h_power = -1.,
        )
        r_gal_length_scale = self.get_processed_data(
            'R',
            scale_key = self.galids.parameters['length_scale'],
            scale_a_power = 1.,
            scale_h_power = -1.,
        )
        is_in_outer_boundary = (
            ( r_rvir < config.INNER_CGM_BOUNDARY ) |
            ( r_gal_length_scale < (1. + config.F_GAP ) * \
            self.galids.parameters['galaxy_cut'] )
        )

        is_in_interface = (
            is_in_outer_boundary &
            np.invert( self.get_data( 'is_in_main_gal' ) )
        )

        self.data['is_in_galaxy_halo_interface'] = is_in_interface

    ########################################################################

    def calc_is_CGM_satellite( self ):

        self.data['is_CGM_satellite'] = (
            self.get_data( 'is_in_CGM' )
            & self.get_data( 'is_in_other_gal' )
            & np.invert( self.get_data( 'is_IP' ) )
        )

    ########################################################################

    def calc_is_CGM_to_IGM( self ):

        r_rvir = self.get_processed_data(
            'R',
            scale_key = 'Rvir',
            scale_a_power = 1.,
            scale_h_power = -1.,
        )

        

    ########################################################################

    def calc_is_CGM_EP( self ):

        self.data['is_CGM_EP'] = (
            self.get_data( 'is_in_CGM' )
            & self.get_data( 'is_hitherto_EP' )
            & np.invert( self.get_data( 'is_IP' ) )
            & np.invert( self.get_data( 'is_in_other_gal' ) )
        )

    ########################################################################

    def calc_is_CGM_NEP( self ):

        self.data['is_CGM_NEP'] = (
            self.get_data( 'is_in_CGM' )
            & self.get_data( 'is_hitherto_NEP' )
            & np.invert( self.get_data( 'is_IP' ) )
            & np.invert( self.get_data( 'is_in_other_gal' ) )
        )

    ########################################################################

    def calc_is_CGM_IP( self ):

        self.data['is_CGM_IP'] = self.get_data( 'is_in_CGM' ) \
            & self.get_data( 'is_IP' )

    ########################################################################

    def calc_is_outside_any_gal_EP( self ):

        is_outside_any_gal = self.get_data( 'gal_id' ) == -2

        self.data['is_outside_any_gal_EP'] = is_outside_any_gal \
            & self.get_data( 'is_hitherto_EP_NYA' )

    ########################################################################

    def calc_is_outside_any_gal_IP( self ):

        is_outside_any_gal = self.get_data( 'gal_id' ) == -2

        self.data['is_outside_any_gal_IP'] = is_outside_any_gal \
            & self.get_data( 'is_IP' )

    ########################################################################

    def calc_is_after_enrichment( self ):
        '''Find the snapshots at which the metallicity is different from the
        prior snapshot.
        '''
        
        # Set up the data
        is_after_enrichment_full = np.zeros( self.base_data_shape )
        is_after_enrichment_full = is_after_enrichment_full.astype( bool )

        # Get values for most of the data
        met_diff = self.get_data( 'Z' )[:,:-1] - self.get_data( 'Z' )[:,1:]
        is_after_enrichment = met_diff > 1e-6

        # Get values for the earliest traced snapshot
        # (We assume enrichement if above the metallicity floor of 1e-3 to 
        # 1e-4, plus a little room. )
        is_after_enrichment_first_snap = self.get_data( 'Z' )[:,-1] > 2e-3

        # Combine the data
        is_after_enrichment_full[:,:-1] = is_after_enrichment
        is_after_enrichment_full[:,-1] = is_after_enrichment_first_snap

        self.data['is_after_enrichment'] = is_after_enrichment_full

    ########################################################################

    def calc_is_before_enrichment( self ):
        '''Find the snapshots at which the metallicity is different from the
        next snapshot.
        '''
        
        # Set up the data
        is_before_enrichment = np.zeros( self.base_data_shape ).astype( bool )

        # Get the values
        after_enrichment_vals = self.get_data( 'is_after_enrichment' )[:,:-1]
        is_before_enrichment[:,1:] = after_enrichment_vals

        self.data['is_before_enrichment'] = is_before_enrichment

    ########################################################################

    def calc_is_enriched( self ):
        '''Find the snapshots at which the metallicity is different from the
        either the next snapshot or the previous snapshot.
        '''

        self.data['is_enriched'] = np.ma.mask_or(
            self.get_data( 'is_after_enrichment' ),
            self.get_data( 'is_before_enrichment' ),
        )

    ########################################################################

    def calc_is_enriched_in_mgal( self ):
        '''Find the snapshots at which the metallicity is different from the
        either the next snapshot or the previous snapshot, and the particle
        is inside the radius of the main galaxy (note that no density threshold
        is applied).
        '''

        # Get when not in the radius of the main galaxy
        mt_gal_id = self.get_data( 'mt_gal_id' )
        main_mt_halo_id = self.galids.parameters['main_mt_halo_id']
        is_in_mgal = mt_gal_id == main_mt_halo_id

        # Now get when enriched and in another galaxy.
        self.data['is_enriched_in_mgal'] = \
            is_in_mgal & self.get_data( 'is_enriched' )

    ########################################################################

    def calc_is_enriched_in_ogal( self ):
        '''Find the snapshots at which the metallicity is different from the
        either the next snapshot or the previous snapshot, and the particle
        is inside the radius of another galaxy (note that no density threshold
        is applied).
        '''

        # Get when not in the radius of the main galaxy
        mt_gal_id = self.get_data( 'mt_gal_id' )
        main_mt_halo_id = self.galids.parameters['main_mt_halo_id']
        is_not_in_main_gal = mt_gal_id != main_mt_halo_id

        # Get when in the radius of any galaxy
        gal_id = self.get_data( 'gal_id' )
        is_in_gal = gal_id != -2

        # Get when in the radius of a galaxy other than the main galaxy
        is_in_ogal = is_in_gal & is_not_in_main_gal

        # Now get when enriched and in another galaxy.
        self.data['is_enriched_in_ogal'] = \
            is_in_ogal & self.get_data( 'is_enriched' )

    ########################################################################

    def calc_time( self ):
        '''Calc current time in the simulation.

        Returns:
            self.data['time'] (np.ndarray) :
                The value at index i is the time in the simulation
                (i.e. the age of the universe) at that index.
        '''

        # Age of the universe in Gyr
        self.data['time'] = astro_tools.age_of_universe(
            self.get_data( 'redshift' ),
            h = self.ptracks.data_attrs['hubble'],
            omega_matter = self.ptracks.data_attrs['omega_matter'],
        )

    ########################################################################

    def calc_dt( self ):
        '''Calc time difference between snapshots.

        Returns:
            self.data['dt'] (np.ndarray) : self.data['dt'][i] = light_travel_time[i+1] - light_travel_time[i]
        '''

        # Age of the universe in Myr
        time = self.get_data( 'time' )
        dt = time[:-1] - time[1:]

        # dt is shorter than the standard array, so we need to pad the array at the final snapshot
        dt = np.append( dt, config.FLOAT_FILL_VALUE )

        self.data['dt'] = dt

    ########################################################################

    def calc_t_EP( self ):
        '''Calculate the time spent in another galaxy prior to accretion onto the main galaxy of the simulation.

        Returns:
            self.data['t_EP'] (np.ndarray) :
                self.data['t_EP'][i] = time particle i spent in another galaxy prior to first accretion.
        '''

        # Make sure we have a fresh slate to work with.
        self.data_masker.clear_masks()

        # Make sure we only include time when the particle is in another galaxy
        self.data_masker.mask_data( 'is_in_other_gal', data_value=True )

        # Get the individual pieces of time, prior to adding them up.
        dt_masked = self.get_selected_data( 'dt_tiled', mask_after_first_acc=True, compress=False )

        # Now do the sum
        t_EP = dt_masked.sum( axis=1 )

        # Save the data, with fully masked data filled in with 0's (because that's how long it's spent)
        t_EP.fill_value = 0.
        self.data['t_EP'] = t_EP.filled() * 1e3 # It's typically easier to look at this in Myr

        # Clear the masks again so we don't affect future things.
        self.data_masker.clear_masks()

    ########################################################################

    def calc_d_sat_scaled_min( self ):
        '''Calculate the minimum distance to a a galaxy other than the main galaxy, prior to accretion onto the main gal.

        Returns:
            self.data['d_sat_scaled_min'] (np.ndarray of shape (n_particles,)) :
                self.data['d_sat_scaled_min'][i] = min( d_sat_scaled, prior to first acc for particle i )
        '''

        d = self.get_data( 'd_sat_scaled' )

        mask2 = np.isclose( d, -2. )

        mask = self.data_masker.get_mask( mask=mask2, mask_after_first_acc=True )

        d_ma = np.ma.masked_array( d, mask=mask )

        self.data['d_sat_scaled_min'] = d_ma.min( axis=1 )

    ########################################################################

    def calc_ind( self ):
        '''Just the redshift index for each array.'''

        self.data['ind'] = np.arange( self.n_snaps )

    ########################################################################

    def calc_ind_particle( self ):
        '''Just the particle index for each array.'''

        self.data['ind_particle'] = np.arange( self.n_particles )

    ########################################################################

    def calc_ind_star( self ):
        '''Calculate the index at which a particle is first recorded as being a star.

        Returns:
            self.data['ind_star'] (np.ndarray of shape (n_particles,)) :
                self.data['ind_star'][i] = Index at which particle is first recorded as being a star.
        '''

        ptype = self.get_data( 'PType' )

        is_star = ptype == config.PTYPE_STAR

        # Find the first index the particle was last a gas particle
        ind_last_gas = np.argmin( is_star, axis=1 )

        # This is correct for most cases.
        self.data['ind_star'] = ind_last_gas - 1

        # We need to correct entries which are always star or always gas
        always_star = np.invert( is_star ).sum( axis=1 ) == 0
        always_gas = is_star.sum( axis=1 ) == 0
        self.data['ind_star'][always_star] = -1
        self.data['ind_star'][always_gas] = config.INT_FILL_VALUE

    ########################################################################

    def get_event_id( self, boolean ):
        '''Get an "Event ID" for a given boolean, where the particle moves
        from being in a True state to a False state, or vice versa.

        Args:
            boolean (array-like):
                If True, the event happens at the given index.

        Returns:
            array-like, same dimensions as boolean minus one column:
                A value of -1 means the particle has switched from True to
                False. A value of 1 means the particle has switched from False
                to True. A value of 0 indicates no change.
        '''

        return boolean[:,:-1].astype( int ) - boolean[:,1:].astype( int )

    def calc_CGM_event_id( self ):
        '''Indication of when a particle moves in or out of the CGM.

        Returns:
            array-like, (n_particles, n_snaps - 1):
                A value of -1 means the particle has left the CGM.
                A value of 1 means the particle has entered the CGM.
                A value of 0 indicates no change.
        '''

        self.data['CGM_event_id'] = self.get_event_id(
            self.get_data( 'is_in_CGM' ),
        )

        return self.data['CGM_event_id']

    ########################################################################

    def count_n_events( self, boolean ):
        '''Counts the number of events that occur up to this point.

        Args:
            boolean (array-like):
                If true, the event happens at the given index.

        Returns:
            n_events (array-like):
                n_events[i,j] number of times that particle i has an event
                prior to index j.                                               
        '''

        n_event = np.zeros( self.base_data_shape ).astype( int )

        n_event[:,:-1] = np.cumsum(
            boolean[:,::-1].astype( int ),
            axis = 1,
        )[:,::-1]

        return n_event

    def calc_n_out( self ):                                                     
        '''The number of times a particle has left the main galaxy.             
                                                                                
        Returns:                                                               
            array-like:                                    
                self.data['n_out'], where the value of [i,j]th index is number
                of times that particle i has left the galaxy prior to index j.                                               
        ''' 

        is_leaving = self.get_data( 'gal_event_id' ) == -1

        n_out = self.count_n_events( is_leaving )

        self.data['n_out'] = n_out

        return self.data['n_out']

    def calc_n_in( self ):                                                     
        '''The number of times a particle has entered the main galaxy.             
                                                                                
        Returns:                                                               
            self.data['n_out'] (array-like):                                    
                result[i,j] number of times that particle i has entered
                the galaxy prior to index j.                                               
        ''' 

        is_entering = self.get_data( 'gal_event_id' ) == 1

        n_in = self.count_n_events( is_entering )

        self.data['n_in'] = n_in

########################################################################
########################################################################


class WorldlineDataMasker( simulation_data.TimeDataMasker ):
    '''Data masker for worldline data.'''

    def __init__( self, worldlines ):

        super( WorldlineDataMasker, self ).__init__( worldlines )

    ########################################################################

    def get_mask(
        self,
        mask = None,
        classification = None,
        mask_after_first_acc = False,
        mask_before_first_acc = False,
        preserve_mask_shape = False,
        optional_masks = None,
        *args, **kwargs
    ):
        '''Get a mask for the data.

        Args:
            mask (np.array) :
                Mask to apply to the data. If None, use the masks stored in self.masks (which Nones to empty).

            classification (str) :
                If provided, only select particles that meet this classification, as given in
                self.data_object.classifications.data

            tile_classification_mask (bool) :
                Whether or not to tile the classification mask. True for most data that's time dependent, but False
                for data that's one value per particle.

            mask_after_first_acc (bool) :
                If True, only select particles above first accretion.

            mask_before_first_acc (bool) :
                If True, only select particles after first accretion.

            preserve_mask_shape (bool) :
                If True, don't tile masks that are single dimensional, and one per particle.

            optional_masks (list-like) :
                If given, the optional masks to include, by name (masks must be available in self.optional_masks).

        Returns:
            mask (bool np.ndarray) :
                Mask from all the combinations.
        '''

        used_masks = []

        if mask is None:
            if len( self.masks ) > 0 or len( self.optional_masks ) > 0:
                total_mask = self.get_total_mask(
                    optional_masks=optional_masks
                )
                if type( total_mask ) == np.ndarray:
                    used_masks.append( total_mask )
                elif total_mask:
                    used_masks.append( total_mask )
        else:

            # Tile mask if it's single-dimensional
            if ( not preserve_mask_shape ) and ( mask.shape == ( self.data_object.n_particles, ) ):
                mask = np.tile( mask, (self.data_object.n_snaps, 1 ) ).transpose()

            used_masks.append( mask )

        if classification is not None:

            cl_mask = np.invert( self.data_object.get_data( classification ) )
            if ( len( cl_mask.shape ) == 1 ) and ( not preserve_mask_shape ):
                cl_mask = np.tile( cl_mask, (self.data_object.n_snaps, 1) ).transpose()
            used_masks.append( cl_mask )

        if mask_after_first_acc or mask_before_first_acc:

            assert not ( mask_after_first_acc and mask_before_first_acc ), \
                "Attempted to mask both before and after first acc."

            ind_first_acc_tiled = self.data_object.get_processed_data( 'ind_first_acc_tiled' )
            ind_tiled = np.tile( range( self.data_object.n_snaps ), (self.data_object.n_particles, 1) )

            if mask_after_first_acc:
                first_acc_mask = ind_tiled <= ind_first_acc_tiled
            elif mask_before_first_acc:
                first_acc_mask = ind_tiled > ind_first_acc_tiled
            used_masks.append( first_acc_mask )

        # Combine the masks
        mask = np.any( used_masks, axis=0, keepdims=True )[0]

        return mask

    ########################################################################

    def get_selected_data(
        self,
        data_key,
        mask = None,
        classification = None,
        mask_after_first_acc = False,
        mask_before_first_acc = False,
        preserve_mask_shape = False,
        optional_masks = None,
        *args, **kwargs
    ):
        '''Get masked worldline data. Extra arguments are passed to the ParentClass' get_selected_data.

        Args:
            data_key (str) :
                Data to get.

            mask (np.array) :
                Mask to apply to the data. If None, use the masks stored in self.masks (which Nones to empty).

            classification (str) :
                If provided, only select particles that meet this classification, as given in
                self.data_object.classifications.data

            tile_classification_mask (bool) :
                Whether or not to tile the classification mask. True for most data that's time dependent, but False
                for data that's one value per particle.

            mask_after_first_acc (bool) :
                If True, only select particles above first accretion.

            mask_before_first_acc (bool) :
                If True, only select particles after first accretion.

            preserve_mask_shape (bool) :
                If True, don't tile masks that are single dimensional, and one per particle.

        Returns:
            masked_data (np.array) :
                Flattened array of masked data.
        '''

        used_mask = self.get_mask(
            mask = mask,
            classification = classification,
            mask_after_first_acc = mask_after_first_acc,
            mask_before_first_acc = mask_before_first_acc,
            preserve_mask_shape = preserve_mask_shape,
            optional_masks = optional_masks,
        )

        masked_data = super( WorldlineDataMasker, self ).get_selected_data( data_key, mask=used_mask, *args, **kwargs )

        return masked_data

    ########################################################################

    def get_selected_data_over_time(
        self,
        data_key,
        mask = None,
        classification = None,
        mask_after_first_acc = False,
        mask_before_first_acc = False,
        preserve_mask_shape = False,
        optional_masks = None,
        *args, **kwargs
    ):
        '''Get masked worldline data. Extra arguments are passed to the ParentClass' get_selected_data.

        Args:
            data_key (str) :
                Data to get.

            mask (np.array) :
                Mask to apply to the data. If None, use the masks stored in self.masks (which defaults to empty).

            classification (str) :
                If provided, only select particles that meet this classification, as given in
                self.data_object.classifications.data

            tile_classification_mask (bool) :
                Whether or not to tile the classification mask. True for most data that's time dependent, but False
                for data that's one value per particle.

            mask_after_first_acc (bool) :
                If True, only select particles above first accretion.

            mask_before_first_acc (bool) :
                If True, only select particles after first accretion.

            preserve_mask_shape (bool) :
                If True, don't tile masks that are single dimensional, and one per particle.

        Returns:
            masked_data (np.array) :
                Flattened array of masked data.
        '''

        used_mask = self.get_mask(
            mask = mask,
            classification = classification,
            mask_after_first_acc = mask_after_first_acc,
            mask_before_first_acc = mask_before_first_acc,
            preserve_mask_shape = preserve_mask_shape,
            optional_masks = optional_masks,
        )

        super_class = super( WorldlineDataMasker, self )
        masked_data = super_class.get_selected_data_over_time(
            data_key,
            mask = used_mask,
            *args, **kwargs
        )

        return masked_data

    ########################################################################
    # Selection routines
    ########################################################################

    def run_selection_routine( self, selection_routine, ptype ):
        '''Selection routines are routines for adding non-trivial combinations of masks to self.masks.
        Masked data then will be retrieved with these masks in mind.

        Args:
            selection_routine (str) :
                What selection routine to run? If None, don't run any.

            ptype (str) :
                What particle type to select?

        Returns:
            self.masks (list) :
                Clears and adds masks to self.masks.
        '''

        if selection_routine is None:
            return

        if ptype == 'star':
            ptype_value = config.PTYPE_STAR
        elif ptype == 'gas':
            ptype_value = config.PTYPE_GAS
        else:
            raise Exception( "Unrecognized Particle Type, ptype = {}".format( ptype ) )

        self.clear_masks()

        getattr( self, 'select_{}'.format( selection_routine ) )( ptype_value )

    ########################################################################

    def select_ptype( self, ptype_value ):
        '''Simple selection routine for only selecting particle type.

        Args:
            selection_routine (str) :
                What selection routine to run? If None, don't run any.

            ptype (str) :
                What particle type to select?

        Returns:
            self.masks (list) :
                Clears and adds masks to self.masks.
        '''

        self.mask_data( 'PType', data_value=ptype_value )

    ########################################################################

    def select_galaxy( self, ptype_value ):
        '''This selection routine selects only particles in a galaxy.

        ptype_value (int) :
            In the data, what ptype do we select?

        Returns:
            self.masks (list) :
                Adds masks needed to select only particles in a galaxy.
        '''

        self.mask_data( 'PType', data_value=ptype_value )
        self.mask_data( 'is_in_main_gal', data_value=True )

    ########################################################################

    def select_accreted( self, ptype_value ):
        '''This selection routine selects only particles that are the snapshot before being accreted.

        ptype_value (int) :
            In the data, what ptype do we select?

        Returns:
            self.masks (list) :
                Adds masks needed to select only particles in a galaxy.
        '''

        self.mask_data( 'PType', data_value=ptype_value )

        # Because `is_accreted` has one less column, we need to adjust the shape before we add the mask.
        adjusted_accreted_mask = np.ones( (self.data_object.n_particles, self.data_object.n_snaps) ).astype( bool )
        adjusted_accreted_mask[:, 1:] = np.invert( self.data_object.get_data( 'is_accreted' ) )

        self.mask_data( 'is_accreted', custom_mask=adjusted_accreted_mask )

    ########################################################################

    def select_outside_all_galaxies( self, ptype_value ):
        '''This seleciton routine selects only particles that are outside all galaxies.

        ptype_value (int) :
            In the data, what ptype do we select?

        Returns:
            self.masks (list) :
                Adds masks needed to select only particles outside all galaxies.
        '''

        self.mask_data( 'PType', data_value=ptype_value )

        self.mask_data( 'is_in_main_gal', data_value=False )
        self.mask_data( 'is_in_other_gal', data_value=False )

    ########################################################################

    def select_in_CGM( self, ptype_value ):

        self.mask_data( 'PType', data_value=ptype_value )

        self.mask_data( 'is_in_CGM', data_value=True )

########################################################################
########################################################################


class WorldlineDataKeyParser( generic_data.DataKeyParser ):

    ########################################################################

    def is_tiled_key( self, data_key ):
        '''Parse the data key for tiled data.'''

        if data_key[-6:] == '_tiled':
            return data_key[:-6], True
        else:
            return data_key, False
