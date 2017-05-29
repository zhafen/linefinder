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

from particle_tracking import code_tools

########################################################################

class TestSetDefaultAttribute( unittest.TestCase ):

  ########################################################################

  def test_not_in_dict( self ):

    search_dict = {}
    code_tools.set_default_attribute( self, 'test_attr', True, search_dict )

    assert self.test_attr

  ########################################################################

  def test_not_in_dict2( self ):

    self.data_p = {}

    code_tools.set_default_attribute( self, 'test_attr', True, )

    assert self.test_attr

  ########################################################################

  def test_in_dict( self ):

    self.data_p = { 'test_attr': True}

    code_tools.set_default_attribute( self, 'test_attr', False, )

    assert self.test_attr
