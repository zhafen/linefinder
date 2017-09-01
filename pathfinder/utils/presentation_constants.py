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
  'is_fresh_accretion' : 'blueviolet',
  'is_NEP_wind_recycling' : 'dodgerblue',
  'is_mass_transfer' : 'green',
  'is_merger' : '#e41a1c',
}

CLASSIFICATION_LABELS = {
  'is_fresh_accretion' : 'Fresh Accretion',
  'is_NEP_wind_recycling' : 'NEP Wind Recycling',
  'is_mass_transfer' : 'Intergalactic Transfer',
  'is_merger' : 'Merger',
}

CLASSIFICATION_LIST_A = [ 'is_fresh_accretion', 'is_NEP_wind_recycling', 'is_mass_transfer', 'is_merger' ]
