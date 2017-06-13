#!/usr/bin/env python
'''Tools for astronomical data processing.

@author: Daniel Angles-Alcazar, Zach Hafen
@contact: zachary.h.hafen@gmail.com
@status: Development
'''

import numpy as np

import constants

########################################################################

def hubble_z(redshift, h=0.702, omega_0=0.272, omega_lambda=0.728):
  '''Return Hubble factor in 1/sec for a given redshift.

  Args:
    redshift (float): The input redshift.
    h (float): The hubble parameter.
    omega_0 (float): TODO
    omega_lambda (float): TODO

  Returns:
    hubble_a (float): Hubble factor in 1/sec
  '''

  ascale = 1. / ( 1. + redshift )
  hubble_a = constants.HUBBLE * h * np.sqrt( omega_0 / ascale**3 + (1. - omega_0 - omega_lambda) / ascale**2 + omega_lambda )     # in 1/sec !!

  return hubble_a

