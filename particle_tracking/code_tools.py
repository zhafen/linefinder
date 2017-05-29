#!/usr/bin/env python
'''Various tools for miscellanious use.

@author: Zach Hafen
@contact: zachary.h.hafen@gmail.com
@status: Development
'''

########################################################################

def set_default_attribute( instance, attr_name, default_value, search_dict=None ):
  '''Setup a value for a class, provided the attr_name isn't in data_p.
  If it is, use the value in data_p.

  Args:
    instance (object): What instance to modify the attribute of.
    attr_name (str): The name of the attribute in the dictionary. Also the name of the attribute in general.
    default_value (anything): What the default should be set to, if the attribute isn't in data_p.
    search_dict (dict): Optional, the dictionary to check the attribute for. If not given, assume the instance has an attribute instance.data_p that holds the information.
  '''

  # Use the given dictionary, or try to search for one the instance has.
  if search_dict is None:
    data_p = instance.data_p
  else:
    data_p = search_dict

  if attr_name in data_p:
    setattr( instance, attr_name, data_p[attr_name] )
  else:
    setattr( instance, attr_name, default_value )
