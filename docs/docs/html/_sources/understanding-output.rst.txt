Understanding Output
====================

Particle Worldline Data
-----------------------

Some of the most important data are the files containing the position, density, etc. of the tracked particles at each moment in time.
We typically find the worldlines for 10^5 particles.
These data have filenames like `ptracks_*.hdf5`.

Description of keys
~~~~~~~~~~~~~~~~~~~

* ``ID`` is particle ID.
* ``ChildID`` is particle Child ID.
* ``PType`` is particle type (usually an integer).
* ``Den`` is baryonic number density in cgs.
* ``M`` is mass in solar masses.
* ``P`` is position in physical kpc.
* ``SFR`` is star formation rate in solar masses/year.
* ``T`` is temperature in kelvin.
* ``V`` is velocity in peculiar km/s.
* ``Z`` is metal mass fraction in solar units (using Z_sun = 0.02).
* ``redshift`` is, well, redshift.
* ``snum`` is snapshot number.
* ``parameters`` are the parameters the pathfinding was done with.

Galaxy IDs Data
---------------

These data contain information about what galaxies and halos the tracked particles are associated with.
These data have filenames like `galids_*.hdf5`.

Description of keys
~~~~~~~~~~~~~~~~~~~

Some data may not contain all these keys if the creator of the data decided not to generate them.

* ``gal_id`` is the ID of the galaxy this particle is associated with, at each time. This is defined as the ID of the _least_ massive galaxy that contains the particle within its characteristic radius. The default characteristic radius is five times the stellar half-mass radius. By choosing the least massive galaxy, we preferentially associate particles with satellites. As of right now, the galaxy ID itself is the same as the instantaneous ID (i.e. not merger tree ID) of the host halo. A value of `-2` means the particle is not associated with any galaxy.
* ``mt_gal_id`` is the merger tree ID of the galaxy this particle is associated with, at each time. This is defined as the ID of the _most_ massive galaxy that contains the particle within its characteristic radius. Not all galaxies are tracked by the merger tree, so by choosing the most massive galaxy we are more likely to choose the central galaxy in the simulation. As of right now, the galaxy ID itself is the same as the merger tree ID of the host halo. A value of `-2` means the particle is not associated with any galaxy.
* ``host_halo_id`` is the halo ID of the host halo the particle is part of.
* ``d_gal`` is the distance to the center of the closest galaxy, in proper kpc.
* ``d_gal_scaled`` is the distance to the center of the closest galaxy, after scaling by the stellar half-mass radius. Note: if the characteristic radius is not a multiple of the stellar half-mass radius, but some other length scale, then d_gal_scaled will be scaled by that length scale instead.
* ``d_other_gal`` is the same as ``d_gal``, but only for galaxies other than the simulation's central galaxy.
* ``d_other_gal_scaled`` is the equivalent of ``d_gal_scaled`` but for ``d_other_gal``.

Events Data
-----------

These are derived data products, created by post-processing the particle worldlines and galaxy IDs.

Description of Keys
~~~~~~~~~~~~~~~~~~~

* ``is_in_main_gal`` is a boolean array used to identify particles in the simulation's main galaxy.
* ``gal_event_id`` (n_particles, n_snap-1) is an identifier used when particles leave or enter the main galaxy. A value of 1 (-1) means the particle has just entered (left) the main galaxy, as defined by ``is_in_main_gal[:,0:n_snap-1] - is_in_main_gal[:,1:n_snap]``. A value of 0 indicates no change.

Very Basic Analysis Examples
----------------------------

Load the particle tracks data. ::

    import h5py
    f = h5py.File( 'path_to_data/ptracks_example.hdf5', 'r' )

Get the IDs of all the particles that were tracked. ::

    f['ID'][...]

Get the snapshots used when compiling the data. ::

    f['snum'][...]

Get the density of particle with index 10 at every snapshot. ::

    f['Den'][...][10]

Get the density of all particles at the latest snapshot. ::

    f['Den'][...][:,0]

Get the parameters the particle tracking was done with. ::

    for key in f['parameters'].attrs.keys():
        print '{} = {}'.format( key, f['parameters'].attrs[key] )

The versions of the code that the pathfinding was run with, along with relevant cosmological constants, are stored in the `.hdf5` attributes. ::

    f.attrs.keys()
