#!/usr/bin/env python
'''Constants for use in displaying.
For example, this contains the colors I will want to consistently use for different categories of accretion,
as well as their names.

@author: Zach Hafen, Daniel Angles-Alcazar
@contact: zachary.h.hafen@gmail.com
@status: Development
'''

########################################################################
########################################################################

CLASSIFICATION_COLORS = {
  'is_fresh_accretion' : '#984ea3',
  'is_NEP_wind_recycling' : '#377eb8',
  'is_mass_transfer' : '#4daf4a',
  'is_merger' : '#e41a1c',
  'is_merger_star' : '#e41a1c',
  'is_merger_gas' : '#ff7f00',
  'is_pristine' : '#377eb8',
}

CLASSIFICATION_ALPHA = 0.7

CLASSIFICATION_LABELS = {
  'is_fresh_accretion' : 'Fresh Accretion',
  'is_NEP_wind_recycling' : 'NEP Wind Recycling',
  'is_mass_transfer' : 'Intergalactic Transfer',
  'is_merger' : 'Merger',
  'is_merger_star' : 'Merger-Stellar',
  'is_merger_gas' : 'Merger-ISM',
}

CLASSIFICATION_LIST_A = [
  'is_fresh_accretion',
  'is_NEP_wind_recycling',
  'is_mass_transfer',
  'is_merger_gas',
  'is_merger_star',
]
CLASSIFICATION_LIST_B = [ 'is_fresh_accretion', 'is_NEP_wind_recycling', 'is_mass_transfer', 'is_merger' ]

