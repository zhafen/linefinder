#!/usr/bin/env python
'''Testing.

@author: Zach Hafen
@contact: zachary.h.hafen@gmail.com
@status: Development
'''

import copy
import h5py
import numpy as np
import numpy.testing as npt
import os
import pdb
import pytest
import unittest

import linefinder.analyze_data.ptracks as analyze_ptracks

########################################################################

kwargs = {
  'data_dir' : './tests/data/tracking_output_for_analysis',
  'ahf_data_dir' : './tests/data/ahf_test_data',
  'tag' : 'analyze',
  'ahf_index' : 600,
}

########################################################################

class TestPTracksStartup( unittest.TestCase ):

  def test_init( self ):

    ptracks = analyze_ptracks.PTracks( **kwargs )

    assert ptracks.parameters['tag'] == kwargs['tag']

    expected = np.array([ 0.       ,  0.0698467,  0.16946  ])
    actual = ptracks.redshift
    npt.assert_allclose( expected, actual, atol=1e-7 )

    expected = np.array([
      [[ 41742.828125  ,  39968.60546875,  34463.765625  ],
        [ 42668.1015625 ,  39049.01953125,  35956.61328125],
        [ 40801.60546875,  38467.77734375,  34972.05078125],
        [ 41183.14453125,  38845.4453125 ,  35722.984375  ]],

       [[ 44588.48828125,  43632.37109375,  37409.45703125],
        [ 41194.57421875,  41100.8046875 ,  38051.79296875],
        [ 43904.26171875,  42300.6875    ,  39093.890625  ],
        [ 43803.72265625,  39763.73046875,  38222.14453125]],

       [[ 46480.3984375 ,  44047.96484375,  38694.92578125],
        [ 46721.30078125,  43169.08203125,  39601.91796875],
        [ 45615.1875    ,  43665.89453125,  39445.94921875],
        [ 45939.51171875,  43247.99609375,  39245.52734375]] 
    ])
    actual = ptracks.data['P']
    npt.assert_allclose( expected, actual )

    expected = np.array([
      [[ -74.53683472,  -15.72714233,  -58.30252838],
       [ -69.17009735,   -1.2919904 , -101.27483368],
       [ -16.02676392,    7.29637671,   19.52023315],
       [  -6.18479395,  -62.51055908,   -4.00132561]],

      [[ -13.80448914,  109.16255188,   33.89780426],
       [  35.18593979,  208.01019287,   10.167099  ],
       [  42.44924164,  105.24703217,  156.09846497],
       [  85.22612762,   84.99162292,   45.11487961]],

      [[  73.73429108,  164.23052979,  155.80351257],
       [  72.66178131,   76.80445099,   89.73206329],
       [ 105.8130188 ,   23.8477459 ,   70.43486023],
       [ 132.84510803,   73.81738281,   64.64906311]]
    ])
    actual = ptracks.data['V']
    npt.assert_allclose( expected, actual )
