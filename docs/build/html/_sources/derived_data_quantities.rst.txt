Derived Data Products
==========================

In addition to the data immediately available in the data products, there are a number of additional derived quantities that can be generated on the fly.
These quantities are available as part of an analysis class, and are documented below as methods of that class.

The quantities can be accessed in two ways:

1. Run the relevant method, after which the derived quantity will be available in the class's `.data` attribute. If `d` is the instance of the class, then this to get the time since the start of the simulation you would use `d.calc_time() ; d.data['time']`
2. Use the class's data retrieval methods, `get_data`, e.g. `d.get_data( 'time' )`. This also works with the more advanced `get_processed_data` and `get_selected_data`.

.. autoclass:: linefinder.analyze_data.worldlines.Worldlines
    :show-inheritance:

    .. automethod:: calc_EnrichedMetalMass
    .. automethod:: calc_MetalMass
    .. automethod:: calc_abs_phi
    .. automethod:: calc_ang_momentum
    .. automethod:: calc_d_sat_scaled_min
    .. automethod:: calc_dt
    .. automethod:: calc_ind
    .. automethod:: calc_ind_particle
    .. automethod:: calc_ind_star
    .. automethod:: calc_is_CGM_EP
    .. automethod:: calc_is_CGM_IP
    .. automethod:: calc_is_CGM_NEP
    .. automethod:: calc_is_CGM_satellite
    .. automethod:: calc_is_IP
    .. automethod:: calc_is_NEP_NYA
    .. automethod:: calc_is_NEP_wind_recycling
    .. automethod:: calc_is_after_enrichment
    .. automethod:: calc_is_before_enrichment
    .. automethod:: calc_is_classification_NYA
    .. automethod:: calc_is_enriched
    .. automethod:: calc_is_enriched_in_mgal
    .. automethod:: calc_is_enriched_in_ogal
    .. automethod:: calc_is_fresh_accretion
    .. automethod:: calc_is_hitherto_EP_NYA
    .. automethod:: calc_is_hitherto_NEP_NYA
    .. automethod:: calc_is_in_CGM
    .. automethod:: calc_is_in_galaxy_halo_interface
    .. automethod:: calc_is_mass_transfer_NYA
    .. automethod:: calc_is_merger_NYA
    .. automethod:: calc_is_merger_gas
    .. automethod:: calc_is_merger_star
    .. automethod:: calc_is_outside_any_gal_EP
    .. automethod:: calc_is_outside_any_gal_IP
    .. automethod:: calc_phi
    .. automethod:: calc_radial_distance
    .. automethod:: calc_radial_velocity
    .. automethod:: calc_t_EP
    .. automethod:: calc_time
    .. automethod:: calc_time_as_classification
