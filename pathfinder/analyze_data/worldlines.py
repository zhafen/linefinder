#!/usr/bin/env python
'''Tools for reading worldline data

@author: Zach Hafen
@contact: zachary.h.hafen@gmail.com
@status: Development
'''

import numpy as np
import os

import galaxy_diver.analyze_data.generic_data as generic_data
import galaxy_diver.read_data.snapshot as read_snapshot
import galaxy_diver.utils.astro as astro_tools
import galaxy_diver.utils.utilities as utilities

import ids
import ptracks
import galids
import classifications
import events

import pathfinder.utils.presentation_constants as p_constants
import pathfinder.utils.data_constants as d_constants

########################################################################

# For catching default values
default = object()

########################################################################
########################################################################

class Worldlines( generic_data.GenericData ):
  '''Wrapper for analysis of all worldline data products. It loads data in on-demand.
  '''

  def __init__( self,
    data_dir,
    tag,
    ids_tag = default,
    ptracks_tag = default,
    galids_tag = default,
    classifications_tag = default,
    events_tag = default,
    **kwargs ):
    '''
    Args:
      data_dir (str) : Data directory for the classified data
      tag (str) : Identifying tag for the data to load.
      ids_tag (str) : Identifying tag for ids data. Defaults to tag.
      ptracks_tag (str) : Identifying tag for ptracks data. Defaults to tag.
      galids_tag (str) : Identifying tag for galids data. Defaults to tag.
      classifications_tag (str) : Identifying tag for classifications data. Defaults to tag.
      events_tag (str) : Identifying tag for events data. Defaults to tag.
      label (str) : Identifying label for the worldlines. Defaults to tag.
      color (str) : What color to use when plotting.
    '''

    if ids_tag is default:
      ids_tag = tag
    if ptracks_tag is default:
      ptracks_tag = tag
    if galids_tag is default:
      galids_tag = tag
    if classifications_tag is default:
      classifications_tag = tag
    if events_tag is default:
      events_tag = tag

    # Store the arguments
    for arg in locals().keys():
      setattr( self, arg, locals()[arg] )

    self.ptracks_kwargs = dict( kwargs )

    data_masker = WorldlineDataMasker( self )
    key_parser = WorldlineDataKeyParser()

    self.data = {}

    super( Worldlines, self ).__init__( data_masker=data_masker, key_parser=key_parser, **kwargs )

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
      self._ptracks = ptracks.PTracks( self.data_dir, self.ptracks_tag, store_ahf_reader=True,
                                               **self.ptracks_kwargs )

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
  def base_data_shape( self ):

    return self.ptracks.base_data_shape

  ########################################################################

  @property
  def n_snaps( self ):

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
        'sdir' : self.ids.snapshot_parameters['sdir'],
        'snum' : self.ids.parameters['snum_end'],
        'ptype' : d_constants.PTYPE_GAS,
        'header_only' : True,
      }

      snapshot = read_snapshot.readsnap( **snapshot_kwargs )
      
      self._n_particles_snapshot_gas = snapshot['npart']

    return self._n_particles_snapshot_gas

  @property
  def n_particles_snapshot_star( self ):
    '''The number of star particles in the last snapshot tracked.'''

    if not hasattr( self, '_n_particles_snapshot_star' ):

      snapshot_kwargs = {
        'sdir' : self.ids.snapshot_parameters['sdir'],
        'snum' : self.ids.parameters['snum_end'],
        'ptype' : d_constants.PTYPE_STAR,
        'header_only' : True,
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

  ########################################################################

  @property
  def m_tot( self ):
    '''Total mass at the last snapshot.'''

    if not hasattr( self, '_m_tot' ):
      masses = self.get_data( 'M', sl=(slice(None),0), )
      masses_no_invalids = np.ma.fix_invalid( masses ).compressed()
      self._m_tot = masses_no_invalids.sum()

    return self._m_tot

  ########################################################################

  @property
  def conversion_factor( self ):
    '''The ratio necessary to convert to the total mass from the sample mass.
    '''

    if not hasattr( self, '_conversion_factor' ):
      self._conversion_factor = float( self.n_particles_presampled )/float( self.n_particles )

    return self._conversion_factor

  ########################################################################

  @property
  def mass_totals( self ):
    '''Get the total mass in the sample in the last snapshot in the canonical classifications.'''

    if not hasattr( self, '_mass_totals' ):
      self._mass_totals = {}
      for mass_category in [ 'is_pristine', 'is_merger', 'is_mass_transfer', 'is_wind' ]:
        self._mass_totals[mass_category] = self.get_masked_data( 'M', sl=(slice(None),0),
                                                                 classification=mass_category,
                                                                 fix_invalid=True, ).sum()

      self._mass_totals = utilities.SmartDict( self._mass_totals )

    return self._mass_totals

  ########################################################################

  @property
  def mass_fractions( self ):
    '''Get the mass fraction in the last snapshot in the canonical classifications.'''

    return self.mass_totals/self.m_tot

  ########################################################################

  @property
  def real_mass_totals( self ):
    '''Get the total mass (converted from the sample) in the last snapshot in the canonical classifications.'''

    return self.mass_totals*self.conversion_factor

  ########################################################################
  # Display Information
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
      data = super( Worldlines, self ).get_data( data_key, *args, **kwargs )
      return data

    # We assume a lot of the data is held in the particle tracks, so look there by default.
    try:
      data = self.ptracks.get_data( data_key, *args, **kwargs )
      return data

    # If the data isn't there, go through the more-complicated default scheme for accessing the data
    except KeyError:
      data = super( Worldlines, self ).get_data( data_key, *args, **kwargs )
      return data

  ########################################################################

  def get_processed_data( self, data_key, *args, **kwargs ):
    '''Get data, handling more complex data keys that indicate doing generic things to the data.

    Args:
      data_key (str) : What data to get?
      sl (object) : How to slice the data before returning it.

    Returns:
      data (np.ndarray) : Array of data.
    '''
    

    data_key, tiled_flag = self.key_parser.is_tiled_key( data_key )

    data = super( Worldlines, self ).get_processed_data( data_key, *args, **kwargs )

    if tiled_flag:

      if data.shape == ( self.n_particles, ):
        data = np.tile( data, ( self.n_snaps, 1) ).transpose()

      elif data.shape == ( self.n_snaps, ):
        data = np.tile( data, ( self.n_particles, 1) )

      else:
        raise Exception( "Unrecognized data shape, {}".format( data.shape ) )

    return data

  ########################################################################

  def get_data_first_acc( self, data_key, ind_after_first_acc=False, *args, **kwargs ):
    '''Get data the snapshot immediately after before accretion.

    Args:
      data_key (str) : What data to get?
      ind_after_first_acc (bool) : If True, get data the index immediately *after* first accretion, instead.

    Returns:
      data_first_acc (np.ndarray) : Array of data, the index immediately after first accretion.
    '''

    data = self.get_data( data_key )

    if issubclass( data.dtype.type, np.integer ):
      fill_value = d_constants.INT_FILL_VALUE
    elif issubclass( data.dtype.type, np.float ) or issubclass( data.dtype.type, np.float32 ):
      fill_value = d_constants.FLOAT_FILL_VALUE
    else:
      raise Exception( "Unrecognized data type, data.dtype = {}".format( data.dtype ) )

    data_first_acc = fill_value*np.ones( self.n_particles, dtype=data.dtype )

    ind_first_acc = self.get_data( 'ind_first_acc' )

    valid_inds = np.where( ind_first_acc != d_constants.INT_FILL_VALUE )[0]
    valid_ind_first_acc = ind_first_acc[valid_inds]

    if not ind_after_first_acc:
      valid_ind_first_acc += 1

    data_first_acc[valid_inds] = data[valid_inds, valid_ind_first_acc]

    return data_first_acc

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

    data = self.get_masked_data( data_key, *args, **kwargs )

    data_ma = np.ma.masked_outside( data, data_min, data_max )

    n_outside = float( data_ma.mask.sum() )
    n_all = float( data.size )

    return n_outside/n_all

  ########################################################################

  def get_galaxy_quantity( self, ptype='star', quantity='mass', *args, **kwargs ):
    '''Get the total mass in the main galaxy for a particular particle type.

    Args:
      ptype (str):
        What particle type inside the galaxy to consider.

      quantity (str):
        What quantity of the galaxy to retrieve.

      *args, **kwargs :
        Additional arguments to be passed to self.get_masked_data()

    Returns:
      galaxy_quantity (np.ndarray) :
        Total mass for a particular particle type in the main galaxy
        (satisfying any additional requirements passed via *args and **kwargs)
        at each specified redshift.
    '''

    self.data_masker.clear_masks()

    if ptype == 'star':
      ptype_value = d_constants.PTYPE_STAR 
    elif ptype == 'gas':
      ptype_value = d_constants.PTYPE_GAS
    else:
      raise Exception( "Unrecognized Particle Type, ptype = {}".format( ptype ) )

    self.data_masker.mask_data( 'PType', data_value=ptype_value )
    self.data_masker.mask_data( 'is_in_main_gal', data_value=True )

    data_ma = self.get_masked_data( 'M', fix_invalid=True, compress=False, *args, **kwargs )

    if quantity == 'mass':
      galaxy_quantity = data_ma.sum( axis=0 )

      # Replace masked values with 0
      try:
        galaxy_quantity.fill_value = 0.
        galaxy_quantity = galaxy_quantity.filled()

      except AttributeError:
        pass

    elif quantity == 'n_particles':
      galaxy_quantity = np.invert( data_ma.mask ).sum( axis=0 )

    else:
      raise Exception( "Unrecognized galaxy_quantity, galaxy_quantity = {}".format( galaxy_quantity ) )

    return galaxy_quantity

  def get_categories_galaxy_quantity( self, classification_list=p_constants.CLASSIFICATION_LIST_A, *args, **kwargs ):
    '''Get the total mass in the main galaxy for a particular particle type in each
    of a number of classification categories. This is only for particles that are tracked! This is not the real mass!

    Args:
      classification_list (list) :
        What classifications to use.

      *args, **kwargs :
        Additional arguments to be passed to self.get_masked_data()

    Returns:
      categories_galaxy_quantity (SmartDict of np.ndarrays) :
        galaxy_quantity that fits each classification.
    '''
  
    galaxy_quantity = {}
    for mass_category in classification_list:
      galaxy_quantity[mass_category] = self.get_galaxy_quantity( classification=mass_category, *args, **kwargs )

    return utilities.SmartDict( galaxy_quantity )

  def get_categories_galaxy_quantity_fraction( self,
    classification_list = p_constants.CLASSIFICATION_LIST_A,
    *args, **kwargs ):
    '''Same as categories_galaxy_quantity, but as a fraction of the total mass in the main galaxy
    for a particular particle type.
    '''

    categories_mass = self.get_categories_galaxy_quantity( classification_list=classification_list, *args, **kwargs )

    return categories_mass/self.get_galaxy_quantity( *args, **kwargs )

  def get_real_categories_galaxy_quantity( self, classification_list=p_constants.CLASSIFICATION_LIST_A, *args, **kwargs ):
    '''Get the total mass in the main galaxy for a particular particle type in each
    of a number of classification categories.

    Args:
      classification_list (list) :
        What classifications to use.

      *args, **kwargs :
        Additional arguments to be passed to self.get_masked_data()

    Returns:
      categories_galaxy_quantity (SmartDict of np.ndarrays) :
        galaxy_quantity that fits each classification.
    '''

    categories_mass = self.get_categories_galaxy_quantity( classification_list=classification_list, *args, **kwargs )

    # TODO
    raise Exception( "This may not be correct yet!" )

    return categories_mass*self.conversion_factor

  ########################################################################
  # Generate Data on the Go
  ########################################################################

  def handle_data_key_error( self, data_key ):
    '''If we don't have a data_key stored, try and create it.

    Args:
      data_key (str) : The data key in question.

    Modifies:
      self.data (dict) : If successful, stores the data here.
    '''

    # Check if there's a method for calculating the data key, with name calc_data_key
    method_str = 'calc_{}'.format( data_key )
    if hasattr( self, method_str ):
      getattr( self, method_str )()

    elif data_key in self.classifications.data.keys():
      self.data[data_key] =  self.classifications.data[data_key]

    elif data_key in self.events.data.keys():
      self.data[data_key] =  self.events.data[data_key]

    elif data_key in self.galids.data.keys():
      self.data[data_key] =  self.galids.data[data_key]

    else:
      raise KeyError( 'NULL data_key, data_key = {}'.format( data_key ) )

  ########################################################################

  def calc_is_fresh_accretion( self ):
    '''Find material classified as fresh accretion (pristine gas that has not recycled).

    Modifies:
      self.data['is_fresh_accretion'] ( np.ndarray ) : Result.
    '''

    pristine_tiled = np.tile( self.get_data( 'is_pristine' ), (self.n_snaps, 1) ).transpose()
    is_not_wind = np.invert( self.get_data( 'is_wind' ) )

    self.data['is_fresh_accretion'] = np.all( [ pristine_tiled, is_not_wind ], axis=0 )

  ########################################################################

  def calc_is_NEP_wind_recycling( self ):
    '''Find material classified as non-externally-processed wind recycling.

    Modifies:
      self.data['is_NEP_wind_recycling'] ( np.ndarray ) : Result.
    '''

    pristine_tiled = np.tile( self.get_data( 'is_pristine' ), (self.n_snaps, 1) ).transpose()

    self.data['is_NEP_wind_recycling'] = np.all( [ pristine_tiled, self.get_data( 'is_wind' ) ], axis=0 )

  ########################################################################

  def calc_is_merger_star( self ):
    '''Find material classified as a merger, while being a star particle at time of first accretion.
    Caution: This is calculated at the snapshot first after accretion. The safer option may be to calculate at the
    snapshot immediately before first accretion.

    Modifies:
      self.data['is_merger_star'] ( np.ndarray ) : Result.
    '''

    is_star_first_acc = self.get_data_first_acc( 'PType' ) == d_constants.PTYPE_STAR

    self.data['is_merger_star'] = np.all( [ is_star_first_acc, self.get_data( 'is_merger' ) ], axis=0 )

  ########################################################################

  def calc_is_merger_gas( self ):
    '''Find material classified as a merger, while being gas at time of first accretion.
    Caution: This is calculated at the snapshot first after accretion. The safer option may be to calculate at the
    snapshot immediately before first accretion.

    Modifies:
      self.data['is_merger_gas'] ( np.ndarray ) : Result.
    '''

    is_star_first_acc = self.get_data_first_acc( 'PType' ) == d_constants.PTYPE_GAS

    self.data['is_merger_gas'] = np.all( [ is_star_first_acc, self.get_data( 'is_merger' ) ], axis=0 )

  ########################################################################

  def calc_dt( self ):
    '''Calc time difference between snapshots.

    Modifies:
      self.data['dt'] (np.ndarray) : self.data['dt'][i] = light_travel_time[i+1] - light_travel_time[i]
    '''

    # Age of the universe in Myr
    time = 1e3 * astro_tools.age_of_universe(
      self.get_data( 'redshift' ),
      h=self.ptracks.data_attrs['hubble'],
      omega_matter=self.ptracks.data_attrs['omega_matter'] )
    dt = time[:-1] - time[1:] 

    # dt is shorter than the standard array, so we need to pad the array at the final snapshot
    dt = np.append( dt, d_constants.FLOAT_FILL_VALUE )

    self.data['dt'] = dt

  ########################################################################

  def calc_d_sat_scaled_min( self ):
    '''Calculate the minimum distance to a a galaxy other than the main galaxy, prior to accretion onto the main gal.

    Modifies:
      self.data['d_sat_scaled_min'] (np.ndarray of shape (n_particles,)) :
        self.data['d_sat_scaled_min'][i] = min( d_sat_scaled, prior to first acc for particle i )
    '''

    d = self.get_data( 'd_sat_scaled' )

    mask2 = np.isclose( d, -2. )

    mask = self.data_masker.get_mask( mask=mask2, mask_after_first_acc=True )

    d_ma = np.ma.masked_array( d, mask=mask )

    self.data['d_sat_scaled_min'] = d_ma.min( axis=1 )

########################################################################
########################################################################

class WorldlineDataMasker( generic_data.DataMasker ):
  '''Data masker for worldline data.'''

  def __init__( self, worldlines ):

    super( WorldlineDataMasker, self ).__init__( worldlines )

  ########################################################################

  def get_mask( self,
    mask = default,
    classification = None,
    mask_after_first_acc = False,
    mask_before_first_acc = False,
    preserve_mask_shape = False,
    *args, **kwargs ):
    '''Get a mask for the data.

    Args:
      mask (np.array) :
        Mask to apply to the data. If default, use the masks stored in self.masks (which defaults to empty).

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
      mask (bool np.ndarray) :
        Mask from all the combinations.
    '''

    used_masks = []
    if mask is default:
      if len( self.masks ) > 0:
        used_masks.append( self.get_total_mask() )
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

      redshift_tiled = np.tile( self.data_object.redshift, (self.data_object.n_particles, 1) )
      redshift_first_acc_tiled = np.tile( self.data_object.events.data['redshift_first_acc'],
                                          (self.data_object.n_snaps, 1) ).transpose()
      if mask_after_first_acc:
        first_acc_mask = redshift_tiled <= redshift_first_acc_tiled
      elif mask_before_first_acc:
        first_acc_mask = redshift_tiled > redshift_first_acc_tiled
      used_masks.append( first_acc_mask )

    # Combine the masks
    mask = np.any( used_masks, axis=0, keepdims=True )[0]

    return mask

  ########################################################################

  def get_masked_data( self,
    data_key,
    mask=default,
    classification=None,
    mask_after_first_acc=False,
    mask_before_first_acc=False,
    preserve_mask_shape=False,
    *args, **kwargs ):
    '''Get masked worldline data. Extra arguments are passed to the ParentClass' get_masked_data.

    Args:
      data_key (str) :
        Data to get.

      mask (np.array) :
        Mask to apply to the data. If default, use the masks stored in self.masks (which defaults to empty).

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
      mask=mask,
      classification=classification,
      mask_after_first_acc=mask_after_first_acc,
      mask_before_first_acc=mask_before_first_acc,
      preserve_mask_shape=preserve_mask_shape,
    )

    masked_data = super( WorldlineDataMasker, self ).get_masked_data( data_key, mask=used_mask, *args, **kwargs )

    return masked_data

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
































