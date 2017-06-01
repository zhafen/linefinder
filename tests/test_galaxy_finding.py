#!/usr/bin/env python
'''Testing for tracking.py

@author: Zach Hafen
@contact: zachary.h.hafen@gmail.com
@status: Development
'''

import numpy as np
import numpy.testing as npt
import pdb
import unittest

from particle_tracking import galaxy_finding

default_data_p = {
  'trackdir' : './tests/test_data/tracking_output',
  'tag' : 'test',
}

########################################################################

class TestGalaxyFinder( unittest.TestCase ):

  def setUp( self ):

    self.galaxy_finder = galaxy_finding.GalaxyFinder()

  ########################################################################

  def test_find_host_halos( self ):

    self.galaxy_finder.find_host_halos()

    comoving_halo0_coords = np.array([ 29338.0986366, 30980.1241434, 32479.90455557 ])
