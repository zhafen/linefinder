'''
Name: tracked_particle_data_handling.py
Author: Zach Hafen (zachary.h.hafen@gmail.com)
Purpose: Used to easily deal with data created in Daniel Angles-Alcazar's
         data tracking pipeline.
'''

import h5py
import numpy as np

########################################################################

class TrackedParticleDataHandler(object):

  def __init__(self, data_dir, file_tag, debug=False):
    '''
    data_dir : Directory the data is stored in.
    file_tag : Additional identifying information to be tagged onto the file name.
    '''

    self.data_dir = data_dir
    self.file_tag = file_tag

    # Turn this on to run more internal checks
    self.debug = debug

    self.load_data()

  ########################################################################

  def load_data(self):

    # File name for the file that tracks all of the accretion modes
    self.accmode_file_name = '{}/accmode_idlist{}.hdf5'.format(self.data_dir, self.file_tag)

    # File name for the file that originally tracked the ids
    self.ptrack_file_name = '{}/ptrack_idlist{}.hdf5'.format(self.data_dir, self.file_tag)

    # Load the data
    self.accmode_data = h5py.File(self.accmode_file_name, 'r')
    self.ptrack_data = h5py.File(self.ptrack_file_name, 'r')

  ########################################################################

  def locate_shared_particles(self, p_data_IDs):
    '''Locates all particles in a data set p_data that are in both p_data and the tracked particle data.
    This is done by sorting the p_data and comparing it to the already sorted tracked data.

    p_data_IDs : 1D Array of particle data IDs.

    Returns
    -------
    valid_ids : An array of IDs that are contained in both
    valid_p_data_indices : The indices in the p_data for each ID
    valid_tracked_data_indices : The indices in the tracked p_data for each ID
    '''

    # Get the indices that allow the IDs to be sorted
    sorted_indices = np.argsort(p_data_IDs)

    # Get the sorted IDs
    sorted_p_data_IDs = p_data_IDs[sorted_indices]

    # Get the ids for the tracked particles
    tracked_data_ids = self.ptrack_data['id'][...]

    # Get the indices that would place the tracked particle ids in the right place
    searched_indices = np.searchsorted(sorted_p_data_IDs, tracked_data_ids)

    # However, many of the tracked particles aren't necessarily included in the snapshot.
    # We need to account for those by only selecting the ones that are in both the particle snapshot
    # and the tracked data. These are found by this simple boolean condition
    match_indices = sorted_p_data_IDs[searched_indices] == tracked_data_ids

    # Since we only want to look at the particles that are in both data sets, let's create versions of those
    valid_searched_indices = np.compress(match_indices, searched_indices)
    valid_ids = np.compress(match_indices, tracked_data_ids)

    # We also want the valid indices for the tracked particle data
    tracked_data_indices = np.arange(len(tracked_data_ids))
    valid_tracked_data_indices = np.compress(match_indices, tracked_data_indices)

    # Finally, we'll get out the valid indices for the p_data
    valid_p_data_indices = sorted_indices[valid_searched_indices]

    return [valid_ids, valid_p_data_indices, valid_tracked_data_indices]

  ########################################################################

  def classify_dataset(self, p_data_IDs, p_data_redshift):
    '''Classify an array of particles based on their IDs, primarily for CGM purposes.

    p_data_IDs : 1D Array of particle data IDs.
    p_data_redshift : Redshift of the particle data.

    Returns
    -------
    classifications : An array of strings that indicate the classification. Options are...
      'u' -- Unclassified, because it's not in the particle tracking data or the particle data set
      'm' -- Particles added to the main galaxy as part of a galaxy merger.
      't' -- Particles added to the main galaxy through stripping and winds from satellite galaxies.
      'p' -- Fresh accretion from the IGM.
      'w' -- Gas currently ejected outside the main galaxy.
      'g' -- Gas that was already accreted onto the main galaxy, and is currently part of the main galaxy.
    '''

    # Get the indices out that are necessary for classification
    valid_ids, valid_p_data_indices, valid_tracked_data_indices = self.locate_shared_particles(p_data_IDs)

    # Make the array where we'll put the classifications
    classifications = np.full(p_data_IDs.shape, 'u', 'string')
    valid_subset = classifications[valid_p_data_indices]

    # We need to find the redshift index in the tracked particle data
    tracked_p_redshift_index = np.where(np.abs(self.accmode_data['redshift'][...] - p_data_redshift) < .00001)[0][0]

    # Label the merger data
    valid_subset = np.where(self.accmode_data['IsMerger'][...][valid_tracked_data_indices], 'm', valid_subset)

    # Label the mass transfer data
    valid_subset = np.where(self.accmode_data['IsMassTransfer'][...][valid_tracked_data_indices], 't', valid_subset)

    # Label the pristine data
    valid_subset = np.where(self.accmode_data['IsPristine'][...][valid_tracked_data_indices], 'p', valid_subset)

    # Label the gas that's currently part of the main galaxy. Must be done after all the redshift independent stuff.
    accreted_condition = self.accmode_data['redshift_FirstAcc'][...][valid_tracked_data_indices] >= p_data_redshift
    valid_subset = np.where(accreted_condition, 'g', valid_subset)

    # Label the wind data. Must be done after everything else, since it's redshift dependent, and comes after first accretion.
    valid_subset = np.where(self.accmode_data['IsWind'][...][valid_tracked_data_indices, tracked_p_redshift_index], 'w', valid_subset)

    # Check that everything's categorized
    if self.debug:
      categories = ['m', 't', 'p', 'g', 'w']
      tot_fraction = 0.
      for i, category in enumerate(categories):
        cat_fraction = float(np.where(valid_subset == category)[0].size)/float(valid_subset.size)
        tot_fraction += cat_fraction
      if np.abs(tot_fraction - 1.0) > 0.001:
        raise Exception('Not all of the valid data is classified')

    # Now put back into the overall data
    classifications[valid_p_data_indices] = valid_subset

    return classifications

